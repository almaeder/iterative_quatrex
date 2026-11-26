"""Single-GPU density-matrix hierarchy solver (qtxmg): a readable, self-contained twin of the distributed qtxmg.hier (one process,
no MPI, no plan of block exchanges). With --obc folded --bottom btrep it gives the numbers of `qtxmg.hier.run` on one rank.

    python -m qtxmg.single --inputs inputs/mos2_hfo2 --dE 0.05,0.15 --cap-split 2,1 --up y:1e-5,x:1e-5 --mu=-0.15,0.6 \\
        --gpu --cache /scratch/.../single_2808a --out T.json

Equations. Per energy z = E + i eta, Sigma_s = Sigma_s(z), every operator at E = Re z:

    A   = E S - H - Sigma_L - Sigma_R                                   fine operator (complex128)
    M^-1 r:  z1 = S_0 r,  z2 = z1 + P_1 C_1 P_1^T (r - A z1),  M^-1 r = z2 + S_0 (r - A z2)
    C_k r (k < K): the same with S_k, A_k, P_{k+1}, C_{k+1};   C_K = A_K^-1 (block-tridiagonal LU)
    A_k = E S_k - H_k - Y_k^T Sigma_L Y_k - Y_k^T Sigma_R Y_k,  S_k = U_k^T S_{k-1} U_k (block pairs), Y_k = rows of P_1..P_k on a face
    S_0, S_k: restricted additive Schwarz, window matrix + i CAP (- Sigma on lead faces), only the core / parent rows of its inverse
    bases: rho = -1/pi Im sum_q w_q [M(z_q)^-1]_cc  (the rest of the window folded in),  U = eigenvectors of rho above the cut

Outer solver: right-preconditioned flexible GMRES from x0 = 0, every column its own Krylov space. T = sum_cols y^H Gamma_R y,
y = the solution on the right-face rows. Below the fine operator (P, A_k, S_k) the cycle runs in complex64, the bottom LU in complex128.

Lead OBC (setup, once for all contour points and energies): Beyn on the periodized lead (default; injection = its incoming modes)
or the folded lead cell with Sancho decimation (injection = eigenchannels of Gamma_L). The device S, H live on the GPU as one CSR
pattern; every dense window and Galerkin block is filled there.

Not here (see qtxmg.hier.run): MPI / NCCL, closed OBC, --dV, slow-mode coarse levels, distributed bottom LU, profiling.
"""
import argparse, hashlib, json, os, time
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from scipy.spatial import cKDTree

try:                                          # release layout: qtxmg/{single, obc_beyn, gpu, prep}
    from . import gpu
    from .obc_beyn import BeynLead
    from .prep.inputs import read_inputs
except ImportError:                           # development layout: qtx_ms/hier/single.py
    from .. import gpu
    from ..obc_beyn import BeynLead
    from ..prep.inputs import read_inputs

C8 = np.complex64
xp = np                                       # array module: numpy, or cupy after use_gpu()


def log(*a):
    print(*a, flush=True)


# =====================================================================================================================
# array backend
# =====================================================================================================================
def use_gpu():
    """cupy for every dense part here; the Beyn lead (gpu module) follows."""
    global xp
    gpu.enable()
    xp = gpu.cp


def on_gpu():
    return xp is not np


def to_host(a):
    return a.get() if on_gpu() and hasattr(a, "get") else np.asarray(a)


def sync():
    if on_gpu():
        xp.cuda.Device().synchronize()


def free():
    if on_gpu():
        xp.get_default_memory_pool().free_all_blocks()


def lu_factor(A):
    if on_gpu():
        import cupyx.scipy.linalg as cla
        return cla.lu_factor(A, overwrite_a=True, check_finite=False)
    return sla.lu_factor(A, overwrite_a=True, check_finite=False)


def lu_solve(lu, B, trans=0):
    if on_gpu():
        import cupyx.scipy.linalg as cla
        return cla.lu_solve(lu, B, trans=trans, check_finite=False)
    return sla.lu_solve(lu, B, trans=trans, check_finite=False)


def inverse_rows(A, rows):
    """(A^-1)[rows, :] of a dense A (overwritten): solve A^H X = e_rows, then X^H."""
    lu = lu_factor(xp.asfortranarray(A))
    E = xp.zeros((A.shape[0], len(rows)), A.dtype, order="F")
    E[xp.asarray(rows), xp.arange(len(rows))] = 1
    return xp.ascontiguousarray(lu_solve(lu, E, trans=2).conj().T)


def offsets(sizes):
    return np.concatenate([[0], np.cumsum(sizes)]).astype(int)


class DeviceMatrices:
    """The device S and H on the array device: one CSR pattern (of S + i H), two real value arrays. Dense blocks and the
    operator E S - H are filled there (no host -> device copies of dense windows)."""

    def __init__(self, S, H):
        M = (S + 1j * H).tocsr()
        M.sort_indices()
        self.n = M.shape[0]
        self.indptr_host = M.indptr.astype(np.int64)
        self.indptr, self.indices = xp.asarray(self.indptr_host), xp.asarray(M.indices.astype(np.int32))
        self.s, self.h = xp.asarray(M.data.real.copy()), xp.asarray(M.data.imag.copy())

    def dense(self, rows, cols=None):
        """Dense S[rows, cols], H[rows, cols] (float64): gather the nonzeros of the rows, keep those in cols."""
        cols = rows if cols is None else cols
        start, count = self.indptr_host[rows], np.diff(self.indptr_host)[rows]
        o = offsets(count)
        og, start = xp.asarray(o), xp.asarray(start)
        q = xp.arange(int(o[-1]))
        row = xp.searchsorted(og, q, side="right") - 1                # row (position in rows) of every gathered nonzero
        q += start[row] - og[row]                                     # its index in the CSR arrays
        col_pos = xp.full(self.n, -1, np.int64)
        col_pos[xp.asarray(cols)] = xp.arange(len(cols))
        c = col_pos[self.indices[q]]
        keep = c >= 0
        flat, q = row[keep] * len(cols) + c[keep], q[keep]
        S, H = xp.zeros(len(rows) * len(cols)), xp.zeros(len(rows) * len(cols))
        S[flat], H[flat] = self.s[q], self.h[q]
        return S.reshape(len(rows), len(cols)), H.reshape(len(rows), len(cols))

    def operator(self, E):
        """E S - H as a complex128 CSR matrix on the array device."""
        data = (E * self.s - self.h).astype(complex)
        if on_gpu():
            import cupyx.scipy.sparse as csp
            return csp.csr_matrix((data, self.indices, self.indptr.astype(np.int32)), shape=(self.n, self.n))
        return sp.csr_matrix((data, self.indices, self.indptr), shape=(self.n, self.n))


# =====================================================================================================================
# device
# =====================================================================================================================
def periodic_axes(periodic):
    if not periodic:
        return []
    return [(int(p["axis"]), float(p["length"])) for p in ([periodic] if isinstance(periodic, dict) else periodic)]


class Geometry:
    """Distances between atoms, minimum image along the periodic transverse axes."""

    def __init__(self, xyz, periodic):
        self.xyz, self.per, self.box = np.asarray(xyz, float), periodic_axes(periodic), None
        if self.per:
            self.lo = self.xyz.min(0) - 1.0
            self.box = np.full(3, 2 * np.ptp(self.xyz, axis=0).max() + 1e3)
            for ax, L in self.per:
                self.box[ax] = L

    def points(self, atoms):
        r = self.xyz[atoms]
        if self.box is None:
            return r
        q = r - self.lo
        for ax, L in self.per:
            q[:, ax] %= L
        return q

    def tree(self, atoms):
        return cKDTree(self.points(atoms), boxsize=self.box)


def lead_face(H01, S01):
    """Orbitals of a lead cell that couple to the next cell (rows or columns of H01, S01)."""
    P = abs(sp.csr_matrix(H01)) + abs(sp.csr_matrix(S01))
    return np.union1d(*(np.unique(q) for q in P.nonzero()))


class Device:
    """Device part of an input set (prep.inputs): the device blocks in transport order, H and S on them, the lead faces
    bL / bR (device orbitals coupled to a lead) and, per side, the lead cell, its orbitals in the input and the couplings
    (H_ld, S_ld, H_dl, S_dl) between its face F and the device face."""

    def __init__(self, inp):
        self.inp, m = inp, inp["meta"]
        self.meta, tr = m, m["transport"]
        ob = np.repeat(inp["blocks"], inp["norb"])
        O = {b: np.where(ob == b)[0] for b in np.unique(inp["blocks"])}
        slabs = [O[b] for b in tr["device_blocks"]]
        self.dev = np.concatenate(slabs)                              # input orbital of every device orbital
        self.n = len(self.dev)
        self.slab_bounds = offsets([len(s) for s in slabs])
        H, S = inp["H"].tocsr(), inp["S"].tocsr()
        self.H, self.S = H[self.dev][:, self.dev].tocsr(), S[self.dev][:, self.dev].tocsr()
        assert not np.iscomplexobj(self.H.data) and not np.iscomplexobj(self.S.data), "real H, S only"
        atom = np.repeat(np.arange(len(inp["norb"])), inp["norb"])
        self.atom = atom[self.dev]                                    # atom of every device orbital
        self.geo = Geometry(inp["xyz"], m.get("periodic"))
        self.x = np.asarray(inp["xyz"])[self.atom, m["axis"]]         # transport coordinate of every device orbital
        right = inp.get("lead_right") or inp["lead"]
        self.lead_cells = [inp["lead"], right]                        # {H00, H01, S00, S01} per side
        self.lead_orbs = [O[tr["lead_left_block"]], O[tr["lead_right_block"]]]
        self.faces, self.lead_F, self.cpl = [], [], []
        for cell, o in zip(self.lead_cells, self.lead_orbs):
            face = np.unique(np.concatenate([M[o][:, self.dev].nonzero()[1] for M in (H, S)]))   # device orbitals
            F = lead_face(cell["H01"], cell["S01"])                    # lead-cell orbitals of the coupling face
            ol = o[F]
            self.faces.append(face)
            self.lead_F.append(F)
            self.cpl.append([M[ol][:, self.dev[face]].tocsr() for M in (H, S)] + [M[self.dev[face]][:, ol].tocsr() for M in (H, S)])
        self.bL, self.bR = self.faces


def to_orbitals(V, norb, perm=None):
    """A potential per atom or per orbital in the source order (QuaTrEx inputs/potential.npy) -> per orbital of the input set,
    whose atoms may be reordered: perm maps orbital -> source orbital (perm.npy of prep.quatrex), whole atoms move together."""
    V, norb = np.asarray(V, float), np.asarray(norb)
    if perm is None:
        perm = np.arange(int(norb.sum()))
    if len(V) == len(norb):                                           # per atom: the source atom order from the first orbitals
        first = perm[np.concatenate([[0], np.cumsum(norb)[:-1]])]
        order = np.argsort(np.argsort(first))                         # input atom i = source atom order[i]
        nq = np.empty_like(norb)
        nq[order] = norb
        V = np.repeat(V, nq)
    assert len(V) == len(perm), f"potential: {len(V)} entries for {len(norb)} atoms / {len(perm)} orbitals"
    return V[np.asarray(perm)]


def read_dV(text, inp):
    """--dV NEW.npy[,REF.npy] -> NEW - REF per orbital of the input set (eV)."""
    fs = text.split(",")
    V = np.load(fs[0]) - (np.load(fs[1]) if len(fs) > 1 else 0.0)
    return to_orbitals(V, inp["norb"], inp.get("perm"))


def add_potential(device, dV):
    """H += (v S + S v) / 2 on the device (v = dV per orbital of the input set, restricted to the device). v must vanish on the
    left lead face (Sigma_L and the injection are those of the inputs) and be constant c on the right one: the right lead moves
    with it (another drain bias), Sigma_R'(z) = Sigma_R(z - c). Returns shift_R = -c for the OBC."""
    v = np.asarray(dV, float)[device.dev]
    assert not v[device.bL].any(), "dV: must be 0 on the left lead face"
    c = v[device.bR]
    assert np.ptp(c) == 0, "dV: must be constant on the right lead face"
    D = sp.diags(v)
    device.H = (device.H + 0.5 * (D @ device.S + device.S @ D)).tocsr()
    return -float(c[0])


# =====================================================================================================================
# lead OBC: Sigma_L, Sigma_R and the injection, for every point at once (setup)
# =====================================================================================================================
def sancho(A00, A01, A10, tol=1e-11, maxit=300):
    """Lopez-Sancho decimation of the chain with cells A00 = z S00 - H00, couplings A01 (n -> n+1), A10 (n+1 -> n).
    Returns g_L (last cell of the left-extending chain) and g_R (first cell of the right-extending chain)."""
    n = len(A00)
    e = A00.astype(complex)
    esL, esR = e.copy(), e.copy()
    a, b = A01.astype(complex), A10.astype(complex)
    for _ in range(maxit):
        X = xp.linalg.solve(e, xp.hstack([a, b]))
        ga, gb = X[:, :n], X[:, n:]
        agb, bga = a @ gb, b @ ga
        esR -= agb
        esL -= bga
        e -= agb + bga
        a, b = -a @ ga, -b @ gb
        if max(float(xp.abs(a).max()), float(xp.abs(b).max())) < tol:
            break
    else:
        log(f"  sancho: no convergence after {maxit} steps")
    return xp.linalg.inv(esL), xp.linalg.inv(esR)


class FoldedLead:
    """The lead cell as given: its interior I folded onto the face F, A_eff = A_FF - A_FI A_II^-1 A_IF, then decimation of
    the chain on F. surface_gfs(z) -> (g_L, g_R) on F."""

    def __init__(self, cell, F):
        H00, H01, S00, S01 = (sp.csr_matrix(cell[k]) for k in ("H00", "H01", "S00", "S01"))
        I = np.setdiff1d(np.arange(H00.shape[0]), F)
        blk = lambda M, r, c: M[r][:, c].toarray()
        self.m = {k: blk(M, r, c) for k, M, r, c in
                  [("HFF", H00, F, F), ("SFF", S00, F, F), ("HFI", H00, F, I), ("SFI", S00, F, I), ("HIF", H00, I, F),
                   ("SIF", S00, I, F), ("HII", H00, I, I), ("SII", S00, I, I), ("H01", H01, F, F), ("S01", S01, F, F)]}

    def surface_gfs(self, z):
        m = {k: xp.asarray(v) for k, v in self.m.items()}
        Aeff = z * m["SFF"] - m["HFF"]
        if m["HII"].shape[0]:
            Aeff = Aeff - (z * m["SFI"] - m["HFI"]) @ xp.linalg.solve(z * m["SII"] - m["HII"], z * m["SIF"] - m["HIF"])
        return sancho(Aeff, z * m["S01"] - m["H01"], z * m["S01"].conj().T - m["H01"].conj().T)


def lead_cell_perm(inp, orbs, sections):
    """Orbital order of a lead cell (input orbitals orbs, in lead-matrix order) with its `sections` primitive cells contiguous
    along the transport axis and every cell's atoms in the order of the first cell (matched by translation)."""
    if sections <= 1:
        return None
    norb, ax = inp["norb"], inp["meta"]["axis"]
    off = np.concatenate([[0], np.cumsum(norb)])
    atom_of = np.repeat(np.arange(len(norb)), norb)
    atoms = np.unique(atom_of[orbs])
    x = np.asarray(inp["xyz"])[atoms]
    a = inp["meta"].get("lead_cell_length", np.ptp(x[:, ax]) * sections / (sections - 1)) / sections
    cell = np.floor((x[:, ax] - x[:, ax].min() + 0.25 * a) / a).astype(int)
    assert cell.min() == 0 and cell.max() == sections - 1, f"lead cell atoms span cells {cell.min()}..{cell.max()}"
    first = np.where(cell == 0)[0]
    tree = cKDTree(x[first])
    order = []
    for c in range(sections):
        shift = np.zeros(3)
        shift[ax] = c * a
        mine = np.where(cell == c)[0]
        d, j = tree.query(x[mine] - shift)
        assert d.max() < 1e-2 and len(np.unique(j)) == len(first) == len(mine), f"lead cell {c} does not match cell 0"
        order.extend(mine[np.argsort(j)])
    pos = {o: i for i, o in enumerate(orbs)}
    return np.array([pos[o] for k in order for o in range(off[atoms[k]], off[atoms[k] + 1])])


def smallest_cell(inp, orbs):
    """(sections, orbital order): the most primitive cells the lead cell of the orbitals orbs splits into (all cells match by
    translation, same labels); (1, None) if it does not split."""
    m, norb = inp["meta"], inp["norb"]
    off = np.concatenate([[0], np.cumsum(norb)])
    atoms = np.unique(np.repeat(np.arange(len(norb)), norb)[orbs])
    L = m.get("block_length") or m.get("lead_cell_length")
    given = int(m.get("cells_per_block") or m.get("source", {}).get("cells_per_block", 1))
    lab = np.asarray(inp["labels"])
    for s in range(len(atoms), 1, -1):
        if len(atoms) % s or len(orbs) % s or s % given:
            continue
        try:
            p = lead_cell_perm(dict(inp, meta=dict(m, lead_cell_length=L) if L else m), orbs, s)
        except (AssertionError, ValueError, IndexError):
            continue
        a = np.searchsorted(off, np.asarray(orbs)[p], side="right") - 1
        if (lab[a].reshape(s, -1) == lab[a].reshape(s, -1)[0]).all():
            return s, p
    if given > 1:
        raise ValueError(f"the lead cell is not {given} repeated cells")
    return 1, None


def injection(SL, tol=1e-4):
    """Columns B with Gamma_L = B B^H over its eigenvalues > tol (ascending: strongest last), host array."""
    S = xp.asarray(SL)
    G = 1j * (S - S.conj().T)
    lam, X = xp.linalg.eigh(0.5 * (G + G.conj().T))
    k = lam > tol
    return to_host(X[:, k] * xp.sqrt(lam[k]))


class OBC:
    """Sigma_L(z), Sigma_R(z) (on bL, bR) and, at the real energies, the injection B (B B^H = Gamma_L), computed in prepare()
    for every point at once and kept on the host (cache: optional folder of npz files, one per point; Beyn coupling bases).
    model "beyn": Beyn on the periodized lead at every z (a point with |Im z| <= real_eta is taken at E = Re z), injection =
    the incoming modes; "folded": the lead cell as given, Sancho decimation, injection = the eigenchannels of Gamma_L."""
    real_eta = 1e-5

    def __init__(self, device, model="beyn", cache=None, inj_tol=1e-4, shift_R=0.0):
        """shift_R: the right lead and the device next to it shifted rigidly by -shift_R S (another drain bias, --dV):
        Sigma_R'(z) = Sigma_R(z + shift_R) of the given lead, a point of its own; Sigma_L and the injection are those of z."""
        assert model in ("beyn", "folded"), model
        self.dev, self.model, self.cache, self.inj_tol = device, model, cache, inj_tol
        self.shift = float(shift_R)
        self.S, self.B, self._leads = {}, {}, None

    def _make_leads(self):
        d, out = self.dev, []
        for side, (cell, o, F, c) in enumerate(zip(d.lead_cells, d.lead_orbs, d.lead_F, d.cpl)):
            if self.model == "folded":
                out.append(FoldedLead(cell, F))
                continue
            sec, perm = smallest_cell(d.inp, o)
            n = cell["H00"].shape[0]
            bl = BeynLead(cell["H00"], cell["H01"], cell["S00"], cell["S01"], sec, perm, sides="LR"[side], layers_on="host",
                          fp_store="host", m0=min(max(64, n // sec // 2), n // sec))
            bl.basis_dir = self.cache
            E = sp.csr_matrix((np.ones(len(F)), (F, np.arange(len(F)))), shape=(n, len(F)))     # face -> whole lead cell
            Hld, Sld, Hdl, Sdl = c
            out.append((bl, [(E @ Hld).tocsr(), (E @ Sld).tocsr(), (Hdl @ E.T).tocsr(), (Sdl @ E.T).tocsr()]))
            log(f"obc: Beyn, lead {'LR'[side]}: " + (f"periodized, {sec} primitive cells of {n // sec} orbitals" if sec > 1 else "cell as given"))
        self._leads = out

    def _file(self, z):
        return self.cache and os.path.join(self.cache, f"obc_{self.model}_{z.real:+.8f}_{z.imag:.8e}.npz")

    def _compute(self, z, real):
        """[Sigma_L, Sigma_R], B (None off the real axis) at one point."""
        if self._leads is None:
            self._make_leads()
        out, B = [], None
        for side, lead in enumerate(self._leads):
            Hld, Sld, Hdl, Sdl = self.dev.cpl[side]
            if self.model == "folded":
                g = lead.surface_gfs(z)[side]                         # g_L for the left lead, g_R for the right one
                Adl, Ald = xp.asarray((z * Sdl - Hdl).toarray()), xp.asarray((z * Sld - Hld).toarray())
                out.append(to_host(Adl @ g @ Ald))
                continue
            bl, (Hld, Sld, Hdl, Sdl) = lead
            zz = complex(z.real, 0.0) if abs(z.imag) <= self.real_eta else z
            S, b = bl.sigma(zz, "LR"[side], zz * Sdl - Hdl, zz * Sld - Hld)
            out.append(S)
            if side == 0 and real:
                B = b
            gpu.free()
        if real and self.model == "folded":
            B = injection(out[0], self.inj_tol)
        return out, B

    def prepare(self, zs, reals):
        """Sigma at the contour points zs and at the real energies reals (E + i eta), the injection at reals."""
        t = time.time()
        pts = [(complex(z), False) for z in zs] + [(complex(z), True) for z in reals]
        if self.shift:                                                # Sigma_R at the shifted points
            pts += [(complex(z) + self.shift, False) for z, _ in pts]
            log(f"obc: right lead shifted by {-self.shift:+.4f} S: Sigma_R(z) of the lead at z {self.shift:+.4f}")
        done = 0
        for z, real in pts:
            if z in self.S and (not real or z in self.B):
                continue
            f = self._file(z)
            if f and os.path.exists(f):
                d = np.load(f)
                S, B = [d["SL"], d["SR"]], (d["B"] if "B" in d.files else None)
            else:
                S, B = self._compute(z, real)
                done += 1
                if f:
                    os.makedirs(self.cache, exist_ok=True)
                    np.savez(f + ".tmp.npz", SL=S[0], SR=S[1], **({"B": B} if B is not None else {}))
                    os.replace(f + ".tmp.npz", f)
            self.S[z] = S
            if real:
                self.B[z] = B
        log(f"obc: {self.model}, {len(zs)} contour and {len(reals)} real points ({done} computed), {time.time() - t:.1f} s")

    def __call__(self, z):
        """[Sigma_L(z), Sigma_R(z + shift_R)]."""
        z = complex(z)
        return [self.S[z][0], self.S[z + self.shift][1]]

    def inject(self, z):
        return self.B[complex(z)]

    def drop(self, zs):
        """Forget the Sigma of the contour points (only the bases need them)."""
        for z in zs:
            for q in {complex(z), complex(z) + self.shift}:
                self.S.pop(q, None)


def contour(mu1, mu2, npts):
    """Upper half circle over [mu1, mu2]: points z, weights dz (Gauss-Legendre in the angle)."""
    c, r = 0.5 * (mu1 + mu2), 0.5 * (mu2 - mu1)
    t, w = np.polynomial.legendre.leggauss(int(npts))
    th, w = 0.5 * np.pi * (t + 1), 0.5 * np.pi * w
    return c + r * np.exp(1j * th), -1j * r * np.exp(1j * th) * w


def contours(text, npts, Ec):
    """"M1,M2[,NPTS[,WEIGHT]]:..." relative to Ec -> points and weights of rho = sum_s weight_s rho_s."""
    zs, ws = [], []
    for c in text.split(":"):
        v = c.split(",")
        z, w = contour(Ec + float(v[0]), Ec + float(v[1]), int(v[2]) if len(v) > 2 else npts)
        zs.append(z), ws.append(w * (float(v[3]) if len(v) > 3 else 1.0))
    return np.concatenate(zs), np.concatenate(ws)


# =====================================================================================================================
# fine windows (CAP-Schwarz geometry: x windows centred on the slab interfaces and between them, optional transverse strips)
# =====================================================================================================================
@dataclass
class FineWindow:
    idx: np.ndarray                         # device orbitals of the window (sorted)
    core: np.ndarray                        # bool over idx: the orbitals this window owns (the cores partition the device)
    ramp: np.ndarray                        # absorbing ramp 0..1 over idx towards the artificial cuts


def fine_windows(device, inp, width, d, spacing, split, tov):
    """Windows of `width` Å at the slab interfaces and at points between them (spacing), core = the orbitals nearest to the
    centre; each periodic transverse axis cut into split[i] strips of period / n + tov Å on each side. Ramp p = clip(...)^2
    over d Å towards every artificial cut (x cuts that are not lead faces, transverse cuts)."""
    x, sb = device.x, device.slab_bounds
    slabs = [np.arange(sb[k], sb[k + 1]) for k in range(len(sb) - 1)]
    ifaces = [0.5 * (x[slabs[k]].max() + x[slabs[k + 1]].min()) for k in range(len(slabs) - 1)]
    anchors, fill = [x.min()] + ifaces + [x.max()], []
    for lo, hi in zip(anchors[:-1], anchors[1:]):
        m = max(1, int(np.ceil((hi - lo) / spacing)) - 1) if spacing else 1
        fill += [lo + (hi - lo) * (q + 1) / (m + 1) for q in range(m)]
    centres = np.sort(np.array(ifaces + fill))
    near = np.argmin(np.abs(x[:, None] - centres[None, :]), axis=1)
    wins = []                                                         # (idx, core, transverse cuts [(|du|, half width)])
    for k, c in enumerate(centres):
        i = np.where((np.abs(x - c) < width / 2) | (near == k))[0]
        wins.append((i, near[i] == k, []))
    for (ax, L), n in zip(periodic_axes(device.meta.get("periodic")), split or []):
        if n <= 1:
            continue
        u = np.asarray(inp["xyz"])[device.atom, ax]
        box = np.minimum((u % L * (n / L)).astype(int), n - 1)        # owner strip of every orbital
        out = []
        for i, c, cuts in wins:
            for j in range(n):
                du = np.abs((u[i] - (j + 0.5) * L / n + L / 2) % L - L / 2)   # minimum-image distance to the strip centre
                keep = du < L / (2 * n) + tov
                out.append((i[keep], c[keep] & (box[i[keep]] == j), [(a[keep], h) for a, h in cuts] + [(du[keep], L / (2 * n) + tov)]))
        wins = out
    xmin, xmax, res = x.min(), x.max(), []
    for i, c, cuts in wins:
        xi, ramp = x[i], np.zeros(len(i))
        if xi.min() > xmin + 1e-6:                                    # cut on the left (not the lead face)
            ramp = np.maximum(ramp, np.clip((xi.min() + d - xi) / d, 0, 1) ** 2)
        if xi.max() < xmax - 1e-6:
            ramp = np.maximum(ramp, np.clip((xi - (xi.max() - d)) / d, 0, 1) ** 2)
        for du, h in cuts:
            ramp = np.maximum(ramp, np.clip((du - (h - d)) / d, 0, 1) ** 2)
        res.append(FineWindow(i, c, ramp))
    cover = np.zeros(device.n, int)
    for w in res:
        cover[w.idx[w.core]] += 1
    assert (cover == 1).all(), "window cores must partition the device"
    log(f"fine windows: {len(res)} ({len(centres)} x positions at {', '.join(f'{c:.1f}' for c in centres)} Å), "
        f"{min(len(w.idx) for w in res)}..{max(len(w.idx) for w in res)} orbitals")
    return res, len(centres)


# =====================================================================================================================
# structure of the hierarchy (energy and basis independent)
# =====================================================================================================================
@dataclass
class Window:
    blocks: np.ndarray                      # level-k blocks of the window
    parents: list                           # level-(k+1) blocks whose kids are kept (rows of the smoother, basis of the parents)


@dataclass
class Level:
    kids: list                              # level-(k-1) blocks of each block (None on level 1)
    grid: tuple                             # (x index, y index) of each block
    G: np.ndarray                           # coupled blocks (bool matrix)
    of_core: np.ndarray                     # level-k block of every fine core
    wins: list = field(default_factory=list)

    @property
    def n(self):
        return len(self.G)

    def nbr(self, i):
        return np.where(self.G[i])[0]

    def pairs(self):
        return [(i, int(j)) for i in range(self.n) for j in self.nbr(i) if i <= j]

    def inside(self, blocks):
        """Coupled pairs (i <= j) with both blocks in the list."""
        s = set(int(b) for b in blocks)
        return [(i, int(j)) for i in sorted(s) for j in self.nbr(i) if i <= j and int(j) in s]


def merge(L, axis, halo):
    """Next level: pairs of blocks of L along axis. Fills L.wins: per parent, the blocks coupled to its kids (within halo x
    positions; None: all); parents with the same blocks share a window."""
    ix, iy = L.grid
    key = (ix // 2) * 100000 + iy if axis == "x" else ix * 100000 + iy // 2
    kids = [np.where(key == v)[0] for v in np.unique(key)]
    first = np.array([ks[0] for ks in kids])
    parent = np.empty(L.n, int)
    for c, ks in enumerate(kids):
        parent[ks] = c
    M = np.zeros((len(kids), L.n), np.int32)
    M[parent, np.arange(L.n)] = 1
    grid = (ix[first] // 2, iy[first]) if axis == "x" else (ix[first], iy[first] // 2)
    N = Level(kids, grid, (M @ L.G.astype(np.int32) @ M.T) > 0, parent[L.of_core])
    groups = {}
    for c, ks in enumerate(kids):
        w = np.unique(np.concatenate([L.nbr(b) for b in ks]))
        if halo is not None:
            w = w[np.abs(ix[w][:, None] - ix[ks][None, :]).min(1) <= halo]
        groups.setdefault(tuple(int(b) for b in w), []).append(c)
    L.wins = [Window(np.array(w), ps) for w, ps in groups.items()]
    return N


# =====================================================================================================================
# the hierarchy
# =====================================================================================================================
class Hierarchy:
    """Bases, Galerkin blocks, CAPs (setup, once); per energy: fine operator, smoothers, A_k, bottom LU; the V-cycle."""

    def __init__(self, device, inp, sigma, zs, dzs, cap=(16.0, 2.0, 2.0), spacing=8.0, split=None, tov=6.0, cut1=1e-5,
                 ups=(), halos=(), cache=None):
        self.dev, self.sigma, self.zs, self.dzs = device, sigma, zs, dzs
        width, self.cap_s, self.cap_d = cap
        self.fw, nx = fine_windows(device, inp, width, self.cap_d, spacing, split, tov)
        self.cores = [w.idx[w.core] for w in self.fw]                # level-1 blocks = cores of the fine windows
        self.core_of = np.empty(device.n, int)                       # core of every device orbital and its position there
        self.pos_in_core = np.empty(device.n, int)
        for b, c in enumerate(self.cores):
            self.core_of[c], self.pos_in_core[c] = b, np.arange(len(c))
        self.mats = DeviceMatrices(device.S, device.H)                # S, H on the array device: windows filled there
        # levels: level 1 = cores (grid of x positions x strips), then one level per merge
        nw = len(self.fw)
        ny = nw // nx
        self.levels = [None, Level(None, (np.arange(nw) // ny, np.arange(nw) % ny), self._core_graph(), np.arange(nw))]
        halos = list(halos) + [None] * (len(ups) - len(halos))
        for (axis, _), halo in zip(ups, halos):
            self.levels.append(merge(self.levels[-1], axis, halo))
        self.K = len(self.levels) - 1
        self.cuts = [None, cut1] + [c for _, c in ups]
        self.lead_cores = [np.unique(self.core_of[f]) for f in device.faces]
        self.cache = None
        if cache:
            tag = json.dumps([device.n, list(cap), spacing, split, tov, cut1, list(ups), halos, to_host(zs).round(10).tolist(),
                              to_host(dzs).round(12).tolist(), getattr(sigma, "model", ""), "single1"], default=str)
            self.cache = os.path.join(cache, "bases_" + hashlib.sha1(tag.encode()).hexdigest()[:12])
            os.makedirs(self.cache, exist_ok=True)
        self.t = {}
        L1 = self.levels[1]
        log(f"hierarchy: {nw} cores ({nx} x {ny}), {int(L1.G.sum())} coupled core pairs; levels "
            + ", ".join(f"{k}: {self.levels[k].n} blocks" for k in range(1, self.K + 1)))

    def tick(self, key, t0):
        sync()
        self.t[key] = self.t.get(key, 0.0) + time.time() - t0

    def _core_graph(self):
        """Cores coupled by H or S; every core on a lead face is coupled to every other one there (Sigma is dense on the face)."""
        P = (self.dev.S + 1j * self.dev.H).tocsr()
        P.data[:] = 1
        C = sp.csr_matrix((np.ones(self.dev.n), (np.arange(self.dev.n), self.core_of)))
        G = (C.T @ P.real @ C).toarray() > 0
        for f in self.dev.faces:
            u = np.unique(self.core_of[f])
            G[np.ix_(u, u)] = True
        return G | G.T

    # ------------------------------------------------------------------------------------------------ setup
    def setup(self):
        self.U, self.r, self.off, self.N = {}, {}, {}, {}
        self.SH, self.Y, self.order, self.capw = {}, {}, {}, {}
        for k in range(1, self.K + 1):
            t = time.time()
            self._bases(k)
            self.tick("bases", t)
            t = time.time()
            self._galerkin(k)
            self._lead_rows(k)
            self.tick("galerkin", t)
            log(f"  level {k}: N {self.N[k]}, {self.r[k].min()}..{int(np.median(self.r[k]))}..{self.r[k].max()} per block; "
                f"{self.t['bases']:.0f} s bases so far")
            if k < self.K:
                t = time.time()
                self._windows(k)
                self.tick("cap", t)
            free()

    def have_bases(self):
        """The bases of every level are in the cache: no density matrix is built, the contour Sigma are not needed."""
        return bool(self.cache) and all(os.path.exists(os.path.join(self.cache, f"U{k}.npz")) for k in range(1, self.K + 1))

    def _bases(self, k):
        f = self.cache and os.path.join(self.cache, f"U{k}.npz")
        if f and os.path.exists(f):
            d = np.load(f)
            U = {int(q[1:]): xp.asarray(d[q]) for q in d.files}
        else:
            U = self._bases_level1() if k == 1 else self._bases_from_windows(k - 1)
            if f:
                np.savez(f + ".tmp.npz", **{f"b{b}": to_host(u) for b, u in U.items()})
                os.replace(f + ".tmp.npz", f)
        self.U[k] = U
        self.r[k] = np.array([U[b].shape[1] for b in range(self.levels[k].n)])
        self.off[k] = offsets(self.r[k])
        self.N[k] = int(self.r[k].sum())

    def entries(self, k, blocks):
        """Positions of the blocks in the level-k vector."""
        return np.concatenate([np.arange(self.off[k][b], self.off[k][b + 1]) for b in blocks] or [np.zeros(0, int)])

    # ---- the density matrix of a window and its eigenvectors
    def _fold(self, S, K, add_sigma, n_rest):
        """sum_q w_q [M(z_q)^-1]_PP of M(z) = z S + K - Sigma(z), P = the rows after the first n_rest (the rest folded in by
        its Schur complement)."""
        acc = xp.zeros((S.shape[0] - n_rest,) * 2, complex)
        for q, (z, w) in enumerate(zip(self.zs, self.dzs)):
            M = z * S + K
            add_sigma(M, z)
            Mpp = M[n_rest:, n_rest:]
            if n_rest:
                X = lu_solve(lu_factor(xp.asfortranarray(M[:n_rest, :n_rest])), xp.asfortranarray(M[:n_rest, n_rest:]))
                Mpp = Mpp - M[n_rest:, :n_rest] @ X
            acc += w * xp.linalg.inv(Mpp)
            del M, Mpp
        return acc

    @staticmethod
    def _eig_above(rho, cut):
        """Eigenvectors of the real symmetric rho with eigenvalue > cut, largest first."""
        ev, V = xp.linalg.eigh(0.5 * (rho + rho.T))
        return xp.ascontiguousarray(V[:, ev > cut][:, ::-1])

    def _fine_window(self, k, perm=None):
        """Dense S_w, H_w, CAP C_w = s sqrt(p) S_w sqrt(p) of fine window k (in the order perm), and its lead faces
        [(side, window positions, face positions)]."""
        w = self.fw[k]
        perm = np.arange(len(w.idx)) if perm is None else perm
        idx, ramp = w.idx[perm], w.ramp[perm]
        S, H = self.mats.dense(idx)
        p = xp.asarray(np.sqrt(ramp))
        lead = []
        for side, face in enumerate(self.dev.faces):
            q = np.where(np.isin(idx, face))[0]
            if len(q):
                lead.append((side, xp.asarray(q), np.searchsorted(face, idx[q])))
        return S, H, self.cap_s * (p[:, None] * S * p[None, :]), lead

    def _bases_level1(self):
        """U_1 per core: eigenvectors of the core block of the local rho of its fine window (CAP + lead Sigma)."""
        U = {}
        for b, w in enumerate(self.fw):
            perm = np.concatenate([np.where(~w.core)[0], np.where(w.core)[0]])   # the core last
            S, H, C, lead = self._fine_window(b, perm)

            def add_sigma(M, z):
                for side, q, j in lead:
                    M[q[:, None], q] -= xp.asarray(self.sigma(z)[side][np.ix_(j, j)])
            acc = self._fold(S, 1j * C - H, add_sigma, int((~w.core).sum()))
            U[b] = self._eig_above(-acc.imag / np.pi, self.cuts[1])
            del S, H, C, acc
        return U

    def _bases_from_windows(self, k):
        """U_{k+1} per parent: eigenvectors of its diagonal block of the rho of its level-k window (projected CAP and Sigma)."""
        L, N, U = self.levels[k], self.levels[k + 1], {}
        for g, w in enumerate(L.wins):
            order, o, n_par = self.order[k][g]
            D = self.dense_level(k, order, order)
            S, K = D[0], (-D[1]).astype(complex)
            for a, b, C in self.capw[k, g]:
                K[a, b] += 1j * C.astype(float)
            Yw = [(side, p, self.Y[k][side][0][:, c]) for side, p, c in self._window_lead(k, order, o)]

            def add_sigma(M, z):
                for side, p, Y in Yw:
                    M[p[:, None], p] -= Y.T @ (xp.asarray(self.sigma(z)[side]) @ Y)
            acc = self._fold(S, K, add_sigma, int(o[-1]) - n_par)
            rho, a = -acc.imag / np.pi, 0
            for I in w.parents:
                m = int(self.r[k][N.kids[I]].sum())
                U[I] = self._eig_above(rho[a:a + m, a:a + m], self.cuts[k + 1])
                a += m
            del D, S, K, acc
            free()
        return U

    # ---- Galerkin blocks
    def dense_level(self, k, rows, cols):
        """Dense (2, rows, cols) of S_k, H_k for two lists of level-k blocks, from the coupled pairs."""
        L, r = self.levels[k], self.r[k]
        a, b = offsets(r[rows]), offsets(r[cols])
        D = xp.zeros((2, a[-1], b[-1]))
        at = {int(j): q for q, j in enumerate(cols)}
        for p, i in enumerate(rows):
            for j in L.nbr(int(i)):
                q = at.get(int(j))
                if q is not None:
                    D[:, a[p]:a[p + 1], b[q]:b[q + 1]] = self.SH[k][i, j] if i <= j else self.SH[k][j, i].transpose(0, 2, 1)
        return D

    def _galerkin(self, k):
        """S_k[I, J], H_k[I, J] = U_I^T (S, H)_{k-1}[I, J] U_J of the coupled pairs I <= J."""
        L, X = self.levels[k], {}
        for I in range(L.n):
            for J in (int(j) for j in L.nbr(I) if j >= I):
                if k == 1:
                    D = xp.stack(self.mats.dense(self.cores[I], self.cores[J]))
                else:
                    D = self.dense_level(k - 1, L.kids[I], L.kids[J])
                X[I, J] = self.U[k][I].T @ D @ self.U[k][J]
        self.SH[k] = X

    def _lead_rows(self, k):
        """Y_k per side: rows of P_1 ... P_k on the lead face (face x columns of the lead blocks), the lead blocks, their offsets."""
        L, out = self.levels[k], []
        for side, face in enumerate(self.dev.faces):
            lb = np.unique(L.of_core[self.lead_cores[side]])
            o = offsets(self.r[k][lb])
            Y = xp.zeros((len(face), o[-1]))
            for q, B in enumerate(lb):
                if k == 1:
                    i = np.where(self.core_of[face] == B)[0]
                    Y[xp.asarray(i), o[q]:o[q + 1]] = self.U[1][B][xp.asarray(self.pos_in_core[face[i]])]
                    continue
                Yp, lp, op = self.Y[k - 1][side]
                ok = offsets(self.r[k - 1][L.kids[B]])
                for a, kid in enumerate(L.kids[B]):
                    j = np.where(lp == kid)[0]
                    if len(j):
                        j = int(j[0])
                        Y[:, o[q]:o[q + 1]] += Yp[:, op[j]:op[j + 1]] @ self.U[k][int(B)][ok[a]:ok[a + 1]]
            out.append((Y, lb, o))
        self.Y[k] = out

    def _window_lead(self, k, order, o):
        """[(side, window positions, columns of Y_k)] of the lead blocks in a window with blocks `order`, offsets o."""
        out = []
        for side, (_, lb, ol) in enumerate(self.Y[k]):
            p, c = [], []
            for q, b in enumerate(order):
                j = np.where(lb == b)[0]
                if len(j):
                    p.append(np.arange(o[q], o[q + 1])), c.append(np.arange(ol[j[0]], ol[j[0] + 1]))
            if p:
                out.append((side, xp.asarray(np.concatenate(p)), xp.asarray(np.concatenate(c))))
        return out

    # ---- CAPs of the level windows
    def prolong_rows(self, k, b, F):
        """Rows of P_1 ... P_k for block b of level k at the device orbitals F (inside b)."""
        if k == 1:
            return self.U[1][b][xp.asarray(self.pos_in_core[F])]
        L = self.levels[k]
        out, o = xp.zeros((len(F), int(self.r[k][b]))), offsets(self.r[k - 1][L.kids[b]])
        kid_of = self.levels[k - 1].of_core[self.core_of[F]]
        for a, c in enumerate(L.kids[b]):
            i = np.where(kid_of == c)[0]
            if len(i):
                out[xp.asarray(i)] = self.prolong_rows(k - 1, int(c), F[i]) @ self.U[k][b][o[a]:o[a + 1]]
        return out

    def _cap_orbitals(self, k, w):
        """(ramp orbitals F, s sqrt(p) S_FF sqrt(p)) of a level-k window: p = clip((d - (dist - min dist)) / d, 0, 1)^2, dist = distance
        of an atom to the nearest device atom outside the window (minimum image). None if the window is the whole device."""
        dev, d = self.dev, self.cap_d
        inside = np.zeros(dev.n, bool)
        inside[np.concatenate([self.cores[c] for c in np.where(np.isin(self.levels[k].of_core, w.blocks))[0]])] = True
        F = np.where(inside)[0]
        a_in, a_out = np.unique(dev.atom[F]), np.unique(dev.atom[~inside])
        if not len(a_out):
            return None
        dist, _ = dev.geo.tree(a_out).query(dev.geo.points(a_in), distance_upper_bound=d + 6.0)
        dist = np.where(np.isfinite(dist), dist, np.inf)
        ramp = np.clip((d - (dist - dist.min())) / d, 0, 1) ** 2
        ramp = ramp[np.searchsorted(a_in, dev.atom[F])]
        F, P = F[ramp > 0], sp.diags(np.sqrt(ramp[ramp > 0]))
        return F, (self.cap_s * (P @ dev.S[F][:, F] @ P)).tocsr()

    def _windows(self, k):
        """Per level-k window: block order (the parents' kids last), offsets, size of the kept part; its projected CAP blocks
        Y_a^T W Y_b on the coupled block pairs (float32, as in hier)."""
        L, N = self.levels[k], self.levels[k + 1]
        self.order[k] = []
        for g, w in enumerate(L.wins):
            kept = np.concatenate([N.kids[I] for I in w.parents])
            order = np.concatenate([np.setdiff1d(w.blocks, kept), kept]).astype(int)
            o = offsets(self.r[k][order])
            self.order[k].append((order, o, int(self.r[k][kept].sum())))
            self.capw[k, g] = []
            c = self._cap_orbitals(k, w)
            if c is None:
                continue
            F, W = c
            blk = L.of_core[self.core_of[F]]
            hit = [(q, np.where(blk == b)[0]) for q, b in enumerate(order) if (blk == b).any()]
            Ys = [self.prolong_rows(k, int(order[q]), F[i]) for q, i in hit]
            for a, (qa, ia) in enumerate(hit):
                for b, (qb, ib) in enumerate(hit):
                    if L.G[order[qa], order[qb]]:
                        Wab = xp.asarray(W[ia][:, ib].toarray())
                        self.capw[k, g].append((slice(o[qa], o[qa + 1]), slice(o[qb], o[qb + 1]), (Ys[a].T @ Wab @ Ys[b]).astype(np.float32)))

    # ------------------------------------------------------------------------------------------------ one energy
    def set_energy(self, E, Sig):
        """Fine operator, S_0, and per level the operator A_k and smoother S_k; the LU of A_K. Sig = [Sigma_L, Sigma_R] at E + i eta."""
        self.A0 = self.S0 = self.A = self.Sk = self.bottom = None
        free()
        t = time.time()
        self.E, self.Sig = E, [xp.asarray(S) for S in Sig]
        self.A0 = self.mats.operator(E)
        self.faces_x = [xp.asarray(f) for f in self.dev.faces]
        self.tick("fine operator", t)
        t = time.time()
        self.S0 = self._fine_smoother()
        self.tick("S0", t)
        self.A, self.Sk = {}, {}
        for k in range(1, self.K + 1):
            t = time.time()
            sg = [Y.T @ (S @ Y) for S, (Y, _, _) in zip(self.Sig, self.Y[k])]       # Y_k^T Sigma Y_k
            if k == self.K:
                self.bottom = self._bottom_lu(sg)
                self.tick("bottom LU", t)
                break
            blocks = {p: (E * X[0] - X[1]).astype(C8) for p, X in self.SH[k].items()}
            sg = [s.astype(C8) for s in sg]
            self.A[k] = (blocks, sg)
            self.tick("A_k", t)
            t = time.time()
            self.Sk[k] = self._level_smoother(k, blocks, sg)
            self.tick("level smoothers", t)
            free()

    def fine_matvec(self, x):
        """A x = (E S - H) x - Sigma_L x_L - Sigma_R x_R (complex128)."""
        y = self.A0 @ x
        for S, f in zip(self.Sig, self.faces_x):
            y[f] -= S @ x[f]
        return y

    def _fine_smoother(self):
        """S_0: per fine window M_w = E S_w - H_w + i C_w - Sigma (lead faces), complex64; keeps the core rows of M_w^-1."""
        ops = []
        for b, w in enumerate(self.fw):
            S, H, C, lead = self._fine_window(b)
            M = xp.asfortranarray((self.E * S - H).astype(C8) + 1j * C.astype(C8))
            del S, H, C
            for side, q, j in lead:
                M[q[:, None], q] -= self.Sig[side][xp.asarray(j)[:, None], xp.asarray(j)].astype(C8)
            R = inverse_rows(M, np.where(w.core)[0])
            ops.append((xp.asarray(w.idx), xp.asarray(self.cores[b]), R))
            del M

        def S0(r):
            z = xp.zeros_like(r)
            x = r.astype(C8)
            for win, core, R in ops:
                z[core] = R @ x[win]
            return z
        return S0

    def level_matvec(self, k, x):
        """A_k x from the block pairs (complex64)."""
        blocks, sg = self.A[k]
        o, y = self.off[k], xp.zeros_like(x)
        for (i, j), B in blocks.items():
            y[o[i]:o[i + 1]] += B @ x[o[j]:o[j + 1]]
            if i != j:
                y[o[j]:o[j + 1]] += B.T @ x[o[i]:o[i + 1]]
        for s, (_, lb, _) in zip(sg, self.Y[k]):
            e = xp.asarray(self.entries(k, lb))
            y[e] -= s @ x[e]
        return y

    def _level_smoother(self, k, blocks, sg):
        """S_k: per level-k window M = A_k[g, g] + i C_k,g (complex64; A_k[g, g] holds -Y^T Sigma Y on its lead blocks); keeps the
        rows of the parents' kids of M^-1."""
        L, ops = self.levels[k], []
        for g, w in enumerate(L.wins):
            order, o, n_par = self.order[k][g]
            n, at = int(o[-1]), {int(b): q for q, b in enumerate(order)}
            M = xp.zeros((n, n), C8, order="F")
            for i, j in L.inside(order):
                a, b = at[i], at[j]
                M[o[a]:o[a + 1], o[b]:o[b + 1]] = blocks[i, j]
                if i != j:
                    M[o[b]:o[b + 1], o[a]:o[a + 1]] = blocks[i, j].T
            for a, b, C in self.capw[k, g]:
                M[a, b] += 1j * C
            for side, p, c in self._window_lead(k, order, o):
                M[p[:, None], p] -= sg[side][c[:, None], c]
            R = inverse_rows(M, np.arange(n - n_par, n))
            kept = order[np.searchsorted(o, n - n_par):]
            ops.append((xp.asarray(self.entries(k, order)), xp.asarray(self.entries(k, kept)), R))
            del M

        def Sk(r):
            z = xp.zeros_like(r)
            for win, out, R in ops:
                z[out] = R @ r[win]
            return z
        return Sk

    def _bottom_lu(self, sg):
        """Exact block-tridiagonal LU of A_K (complex128) over its slabs = x positions (neighbours joined when blocks couple farther):
        T_a = D_a - L_{a-1} T_{a-1}^-1 U_{a-1}. Returns the solve r -> A_K^-1 r (complex64 in and out)."""
        k, L = self.K, self.levels[self.K]
        ix = np.unique(L.grid[0], return_inverse=True)[1]
        reach = max([abs(int(ix[i]) - int(ix[j])) for i, j in L.pairs()] + [int(np.ptp(ix[lb])) for _, lb, _ in self.Y[k]])
        ix = ix // max(reach, 1)
        ns = int(ix.max()) + 1
        slab = [np.where(ix == a)[0] for a in range(ns)]
        o = [offsets(self.r[k][s]) for s in slab]
        at = {int(b): (a, q) for a, s in enumerate(slab) for q, b in enumerate(s)}
        M = {(a, b): xp.zeros((int(o[a][-1]), int(o[b][-1])), complex) for a in range(ns) for b in (a - 1, a, a + 1) if 0 <= b < ns}

        def put(i, j, B):
            (a, p), (b, q) = at[int(i)], at[int(j)]
            M[a, b][o[a][p]:o[a][p + 1], o[b][q]:o[b][q + 1]] += B
        for (i, j), X in self.SH[k].items():
            B = self.E * X[0] - X[1]
            put(i, j, B)
            if i != j:
                put(j, i, B.T)
        for s, (_, lb, ol) in zip(sg, self.Y[k]):
            for p, i in enumerate(lb):
                for q, j in enumerate(lb):
                    put(i, j, -s[ol[p]:ol[p + 1], ol[q]:ol[q + 1]])
        inv, X = [], []                                               # T_a^-1 and T_a^-1 U_a
        for a in range(ns):
            D = M.pop((a, a))
            if a:
                D -= M[a, a - 1] @ X[-1]
            inv.append(xp.linalg.inv(D))
            if a + 1 < ns:
                X.append(inv[a] @ M.pop((a, a + 1)))
        Lo = [None] + [M[a, a - 1] for a in range(1, ns)]
        ent = [xp.asarray(self.entries(k, s)) for s in slab]
        self.bottom_slabs = [int(v[-1]) for v in o]

        def solve(r):
            r = r.astype(complex)
            t, x = [], xp.empty_like(r)
            for a in range(ns):                                       # forward: t_a = T_a^-1 (r_a - L_{a-1} t_{a-1})
                y = r[ent[a]]
                t.append(inv[a] @ (y - Lo[a] @ t[-1] if a else y))
            z = None
            for a in range(ns - 1, -1, -1):                           # backward: x_a = t_a - X_a x_{a+1}
                z = t[a] if a + 1 == ns else t[a] - X[a] @ z
                x[ent[a]] = z
            return x.astype(C8)
        return solve

    # ------------------------------------------------------------------------------------------------ transfers and V-cycle
    def prolong(self, k, c):
        """P_k c: level k -> level k - 1 (k = 1: the fine level)."""
        if k == 1:
            y = xp.zeros((self.dev.n, c.shape[1]), C8)
            for b, core in enumerate(self.cores):
                y[core] = self.U[1][b].astype(np.float32) @ c[self.off[1][b]:self.off[1][b + 1]]
            return y
        L, y = self.levels[k], xp.zeros((self.N[k - 1], c.shape[1]), C8)
        for b in range(L.n):
            y[self.entries(k - 1, L.kids[b])] = self.U[k][b].astype(np.float32) @ c[self.off[k][b]:self.off[k][b + 1]]
        return y

    def restrict(self, k, x):
        """P_k^T x: level k - 1 -> level k."""
        y = xp.zeros((self.N[k], x.shape[1]), C8)
        for b in range(self.levels[k].n):
            rows = self.cores[b] if k == 1 else self.entries(k - 1, self.levels[k].kids[b])
            y[self.off[k][b]:self.off[k][b + 1]] = self.U[k][b].T.astype(np.float32) @ x[rows].astype(C8)
        return y

    def cycle(self, k, r):
        """C_k r: V(1,1)-cycle on A_k; C_K = A_K^-1."""
        if k == self.K:
            return self.bottom(r)
        A = lambda v: self.level_matvec(k, v)
        z = self.Sk[k](r)
        z = z + self.prolong(k + 1, self.cycle(k + 1, self.restrict(k + 1, r - A(z))))
        return z + self.Sk[k](r - A(z))

    def precondition(self, r):
        """M^-1 r: V(1,1)-cycle on the fine level with S_0, coarse level 1 by cycle(1, .)."""
        z = self.S0(r)
        z = z + self.prolong(1, self.cycle(1, self.restrict(1, r - self.fine_matvec(z))))
        return z + self.S0(r - self.fine_matvec(z))


# =====================================================================================================================
# flexible GMRES, every column its own Krylov space; products on the block of active columns
# =====================================================================================================================
def fgmres(A, b, M, tol=1e-10, m=100, callback=None):
    """x = argmin |b - A x| over x0 + span(Z), Z = M^-1 V, x0 = 0. A column stops when its residual estimate < tol |b|.
    Returns x, iterations per column, true relative residuals."""
    n, nc = b.shape
    bn = to_host(xp.linalg.norm(b, axis=0))
    beta = bn.copy()
    V = xp.zeros((nc, m + 1, n), complex)
    Z = xp.zeros((nc, m, n), complex)
    Hh = np.zeros((nc, m + 1, m), complex)
    V[:, 0] = (b / xp.asarray(np.where(beta > 0, beta, 1))).T
    its, est, y = np.zeros(nc, int), beta / bn, [np.zeros(0, complex)] * nc
    act = np.where(beta > tol * bn)[0]
    for i in range(m):
        if not len(act):
            break
        zb = M(xp.ascontiguousarray(V[act, i].T))
        Z[act, i] = zb.T
        w = A(zb)
        keep = []
        for jj, j in enumerate(act):
            wj = w[:, jj].copy()
            Vj = V[j, :i + 1]
            for _ in range(2):                                        # classical Gram-Schmidt, twice
                h = Vj.conj() @ wj
                wj -= h @ Vj
                Hh[j, :i + 1, i] += to_host(h)
            hn = float(xp.linalg.norm(wj))
            Hh[j, i + 1, i] = hn
            if hn > 0:
                V[j, i + 1] = wj / hn
            e1 = np.zeros(i + 2, complex)
            e1[0] = beta[j]
            y[j] = np.linalg.lstsq(Hh[j, :i + 2, :i + 1], e1, rcond=None)[0]
            its[j] = i + 1
            est[j] = np.linalg.norm(Hh[j, :i + 2, :i + 1] @ y[j] - e1) / bn[j]
            if est[j] > tol and hn > 0:
                keep.append(j)
        act = np.array(keep, int)
        if callback:
            callback(i, est)
    x = xp.zeros_like(b)
    for j in range(nc):
        if its[j]:
            x[:, j] = xp.asarray(y[j]) @ Z[j, :its[j]]
    return x, its, to_host(xp.linalg.norm(b - A(x), axis=0)) / bn


# =====================================================================================================================
# driver
# =====================================================================================================================
def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", required=True, help="input folder (qtxmg.prep.inputs format)")
    p.add_argument("--dE", required=True, help="energies relative to Ec (comma list)")
    p.add_argument("--Ec", type=float, default=None, help="default: meta edges Ec_lead")
    p.add_argument("--eta", type=float, default=1e-6, help="Sigma at E + i eta")
    p.add_argument("--cap", default="16,2,2", help="fine windows: width [Å], CAP strength [eV], ramp width [Å]")
    p.add_argument("--cap-spacing", type=float, default=8.0)
    p.add_argument("--cap-split", default="", help="strips per periodic transverse axis, e.g. 2,1")
    p.add_argument("--cap-tov", type=float, default=6.0, help="transverse overlap of the strips [Å]")
    p.add_argument("--cut", type=float, default=1e-5, help="cut of the level-1 rho eigenvalues")
    p.add_argument("--up", required=True, help="merges AXIS:CUT,... e.g. y:1e-5,x:1e-5")
    p.add_argument("--halo", default="1", help="x halo of the level windows per merge (comma list; missing / 'all': all coupled blocks)")
    p.add_argument("--mu", default="-0.15,0.8", help="contour(s) relative to Ec: M1,M2[,NPTS[,WEIGHT]]:...")
    p.add_argument("--npts", type=int, default=24)
    p.add_argument("--tol", type=float, default=1e-10)
    p.add_argument("--maxit", type=int, default=100)
    p.add_argument("--chunk", type=int, default=0, help="columns per FGMRES call (0: all)")
    p.add_argument("--obc", default="beyn", choices=["beyn", "folded"], help="lead Sigma: Beyn on the periodized lead at every point "
                   "(injection = incoming modes), or the folded lead cell with Sancho decimation (injection = Gamma_L eigenchannels)")
    p.add_argument("--inj-tol", type=float, default=1e-4, help="Gamma_L eigenvalues above this are injected [eV]")
    p.add_argument("--cache", default="", help="folder for Sigma and the bases")
    p.add_argument("--dV", default="", help="NEW.npy[,REF.npy]: potential change per atom or orbital [eV] (NEW - REF, source order of "
                   "the inputs) added to H as (dV S + S dV) / 2. Zero on the left lead face (Sigma_L, injection reused); constant c on the "
                   "right face: another drain bias, Sigma_R(z) = Sigma_R of the inputs at z - c. With the default --dV-bases reuse, the "
                   "bases of the inputs (in --cache) are kept and the Galerkin blocks project the change into them")
    p.add_argument("--dV-bases", default="reuse", choices=["reuse", "build"], help="with --dV: reuse the bases of the inputs (they must "
                   "be in --cache), or build new ones of H + dV (reference; --cache must not hold bases yet: they carry the tag of the inputs)")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--out", default="")
    return p.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    t00 = time.time()
    if a.gpu:
        use_gpu()
    inp = read_inputs(a.inputs)
    device = Device(inp)
    shift = 0.0
    if a.dV:
        dV = read_dV(a.dV, inp)
        shift = add_potential(device, dV)
        log(f"dV {a.dV}: {dV.min():+.3f}..{dV.max():+.3f} eV, right lead {-shift:+.3f} eV, "
            + ("bases of the inputs reused" if a.dV_bases == "reuse" else "new bases of H + dV"))
    Ec = inp["meta"]["edges"]["Ec_lead"] if a.Ec is None else a.Ec
    zs, dzs = contours(a.mu, a.npts, Ec)
    log(f"{os.path.basename(os.path.normpath(a.inputs))}: {device.n} orbitals, Ec {Ec:.4f} eV, faces {len(device.bL)} / {len(device.bR)}, "
        f"{'GPU ' + xp.cuda.runtime.getDeviceProperties(0)['name'].decode() if on_gpu() else 'CPU'}; contour {a.mu} ({len(zs)} points)")
    obc = OBC(device, a.obc, os.path.join(a.cache, "obc") if a.cache else None, a.inj_tol, shift_R=shift)
    ups = [(v.split(":")[0], float(v.split(":")[1])) for v in a.up.split(",") if v]
    halos = [None if v == "all" else int(v) for v in a.halo.split(",") if v]
    h = Hierarchy(device, inp, obc, zs, dzs, tuple(float(v) for v in a.cap.split(",")), a.cap_spacing,
                  [int(v) for v in a.cap_split.split(",")] if a.cap_split else None, a.cap_tov, a.cut, ups, halos, a.cache or None)
    reals = [Ec + float(v) + 1j * a.eta for v in a.dE.split(",")]
    if a.dV and h.have_bases() != (a.dV_bases == "reuse"):
        raise SystemExit("--dV: the bases of the inputs are not in --cache (run without --dV first)" if a.dV_bases == "reuse" else
                         "--dV-bases build: --cache already holds bases (of the inputs?): give another folder")
    t = time.time()
    obc.prepare([] if h.have_bases() else zs, reals)                # every Sigma and injection at once
    h.tick("sigma", t)
    h.setup()
    obc.drop(zs)
    log(f"setup {time.time() - t:.1f} s (" + ", ".join(f"{k} {v:.1f}" for k, v in h.t.items()) + f"); total {time.time() - t00:.0f} s")
    res = dict(Ec=Ec, args=vars(a), N=[h.N[k] for k in range(1, h.K + 1)], t_setup=dict(h.t), E={})
    for z in reals:
        dE, t0, h.t = z.real - Ec, time.time(), {}
        Sig, B = obc(z), obc.inject(z)
        if B is None or B.shape[1] == 0:
            log(f"Ec {dE:+.3f}: T 0, no incoming channel")
            res["E"][f"{dE:.4f}"] = dict(E=z.real, T=0.0, its=[], res=[], converged=True, t={}, t_energy=time.time() - t0)
            continue
        h.set_energy(z.real, Sig)
        t = time.time()
        nb, ch = B.shape[1], a.chunk or B.shape[1]
        ys, its, rs = [], [], []
        for c in range(0, nb, ch):
            b = xp.zeros((device.n, min(ch, nb - c)), complex)
            b[xp.asarray(device.bL)] = xp.asarray(B[:, c:c + ch])
            x, i, r = fgmres(h.fine_matvec, b, h.precondition, a.tol, a.maxit)
            ys.append(x[xp.asarray(device.bR)]), its.append(i), rs.append(r)
        h.tick("solve", t)
        y, its, rs = xp.concatenate(ys, axis=1), np.concatenate(its), np.concatenate(rs)
        SR = h.Sig[1]
        T = float(xp.real(xp.sum(y.conj() * (1j * ((SR - SR.conj().T) @ y)))))
        ok = bool(rs.max() < 10 * a.tol)
        log(f"Ec {dE:+.3f}: T {T:.10f}, {nb} columns, its {its.min()}..{its.max()}, res {rs.max():.0e}" + ("" if ok else " ** NOT CONVERGED **")
            + f"; {time.time() - t0:.1f} s: " + ", ".join(f"{k} {v:.1f}" for k, v in h.t.items()))
        res["E"][f"{dE:.4f}"] = dict(E=z.real, T=T, its=its.tolist(), res=rs.tolist(), converged=ok, t=dict(h.t), t_energy=time.time() - t0)
        if a.out:
            json.dump(res, open(a.out, "w"), indent=1)
    if on_gpu():
        log(f"GPU memory pool {xp.get_default_memory_pool().total_bytes() / 2 ** 30:.1f} GB")
    log(f"done, {time.time() - t00:.0f} s")
    return res, h


if __name__ == "__main__":
    main()

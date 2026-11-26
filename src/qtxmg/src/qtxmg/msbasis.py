"""Two-level mode-space basis (the qtmode method) built from the generic input set (qtxmg.prep.inputs).

Level 1 (orbitals -> block modes), per transport slab:
  cores : crystalline atoms, found from device.xyz: label core_label and at least d_core from any atom
          with another label. Basis = bulk Bloch states in an energy window, restricted to the block
          (real Gram matrix, Re B B^H). The bulk H(k) is read off the device H at the deepest core atom
          (diamond lattice from its neighbour vectors), so no TB model is called.
          Only for empirical TB of a crystalline channel; ab initio inputs use cores=False.
  tiles : all other atoms (every atom with cores=False), cut into tiles of ~tile_orbs orbitals by
          recursive bisection in the transverse plane. Basis = dominant eigenvectors of the local
          spectral function A_PP(E) = -Im G_PP / pi, G = (E S - H_emb + iW)^-1 in a d_emb embedding
          with a CAP W on its outer d_cap, at a few energies (randomized range finder); the sets of all
          energies are merged by one SVD, keep s^2 > tile_thr * a_ref.
  U1 = diag(blocks) per slab (orthonormal), H1 = U1^T H U1, S1 = U1^T S U1 in slab blocks.
  enrichment (optional, enrich_round): residuals of window Ritz pairs of local clusters of H1 (slabs s +- w,
          or a tile and the tiles coupled to it) appended to the blocks (qtmode.enrich with generalized S,
          complex H and local instead of global Ritz problems).
Level 2 (block modes -> slab modes): contour-integrated slab density matrix of the open level-1 system
  rho_ss = -(1/pi) Im int_arc G1_ss(z) dz,  G1 = (z S1 - H1 - Sigma1(z))^-1,  arc over [mu1, mu2],
  eigenvectors with lambda > thr * lambda_max.  U = U1 U2 (block diagonal over slabs).
Leads: exact Sigma(z) at complex z, lead interior folded onto its faces, then decimation of the folded chain
(general: complex Hermitian H, any S; fast path for real H, S = I: qtmode face decimation on the smaller face).
On the real axis (|Im z| <= Device.qtt_eta) Sigma and the injected modes come from qttools' spectral OBC (QuaTrEx)
when it imports (on sys.path or in $QTTOOLS_SRC, a QuaTrEx src folder; needs the `qtx` extra: mpi4py, numba):
flux-normalized incoming Bloch modes, one column per propagating channel, like QuaTrEx's QTBM.
"""
import os, sys, warnings
import socket
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as sla
from scipy.spatial import cKDTree
try:
    import pypardiso
except ImportError:          # pragma: no cover
    pypardiso = None

from qtxmg import gpu


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ exact lead surface GFs (from qtmode.lead_ms)
def _sancho_both(h00, h01, z, tol=1e-11, maxit=300):
    """Lopez-Sancho returning both bulk-side surface on-site matrices:
    esR (cells n+1.. folded onto n) and esL (cells ..n-1 folded onto n), chain n -> n+1 via h01.
    Complex symmetric chain: n+1 -> n coupling is h01^T."""
    n = len(h00)
    I = np.eye(n)
    e = h00.astype(complex)
    esR, esL = e.copy(), e.copy()
    a, b = h01.astype(complex), h01.T.astype(complex)
    for it in range(maxit):
        X = sla.solve(z * I - e, np.hstack([a, b]), check_finite=False)
        ga, gb = X[:, :n], X[:, n:]
        agb, bga = a @ gb, b @ ga
        esR += agb
        esL += bga
        e += agb + bga
        a, b = a @ ga, b @ gb
        if max(np.abs(a).max(), np.abs(b).max()) < tol:
            break
    return esR, esL, it + 1


def face_surface_gfs_small(Heff, iL, iR, V, z):
    """Exact surface GFs using only the chain on the smaller face Lf (eliminate every Rf):
      right lead, first cell, on Lf : (z - esR + V^T gRR V)^-1          (no R_0 on its left)
      left lead, last cell, on Rf   : R_0 couples only to L_0; L_0 without R_0 has on-site
                                      esL - HLR gRR HRL, then one |Rf| solve."""
    HLL, HLR = Heff[np.ix_(iL, iL)], Heff[np.ix_(iL, iR)]
    HRR, HRL = Heff[np.ix_(iR, iR)], Heff[np.ix_(iR, iL)]
    IR, IL = np.eye(len(iR)), np.eye(len(iL))
    gRR = sla.solve(z * IR - HRR, np.hstack([HRL, V]), check_finite=False)
    t_in, t_v = HLR @ gRR[:, :len(iL)], V.T @ gRR[:, len(iL):]
    h00 = HLL + t_in + t_v
    h01 = HLR @ gRR[:, len(iL):]                                   # L_n -> L_n+1
    esR, esL, nit = _sancho_both(h00, h01, z)
    gR = sla.inv(z * IL - (esR - t_v), check_finite=False)
    GL0 = sla.inv(z * IL - (esL - t_in), check_finite=False)
    gL = sla.inv(z * IR - HRR - HRL @ GL0 @ HLR, check_finite=False)
    return gL, gR, nit


# ------------------------------------------------------------------ sparse complex solves
class CSolver:
    """Factorization of a complex sparse matrix A. PARDISO on the real-equivalent
    [[Ar, -Ai], [Ai, Ar]] when pypardiso is available, else SuperLU. Each solve does n_ref steps of
    iterative refinement, then GMRES preconditioned by the factorization on the columns whose residual
    is still above tol (PARDISO perturbs small pivots; plain refinement can then stall or diverge)."""

    def __init__(self, A, n_ref=2, tol=1e-12, maxit=60):
        A = sp.csr_matrix(A)
        self.A, self.n, self.n_ref, self.tol, self.maxit = A, A.shape[0], n_ref, tol, maxit
        self.res, self.its = 0.0, 0                       # max relative residual, max GMRES its (last solve)
        if pypardiso is None:
            self.lu = spla.splu(A.astype(complex).tocsc())
            self.npert = 0
            return
        Ar, Ai = sp.csr_matrix(A.real), sp.csr_matrix(A.imag)
        self.M = sp.bmat([[Ar, -Ai], [Ai, Ar]], format="csr")
        self.ps = pypardiso.PyPardisoSolver(mtype=11)
        self.ps.factorize(self.M)
        self.npert = int(self.ps.get_iparm(14))           # number of perturbed pivots

    def factor_solve(self, B):
        if pypardiso is None:
            return self.lu.solve(B.astype(complex))
        Z = self.ps.solve(self.M, np.vstack([B.real, B.imag]))
        Z = Z.reshape(2 * self.n, -1)
        return Z[:self.n] + 1j * Z[self.n:]

    def solve(self, B):
        B = np.asarray(B)
        one = B.ndim == 1
        B = B.reshape(len(B), -1).astype(complex)
        X = self.factor_solve(B)
        for _ in range(self.n_ref):
            X += self.factor_solve(B - self.A @ X)
        bn = np.maximum(np.linalg.norm(B, axis=0), 1e-300)
        res = np.linalg.norm(B - self.A @ X, axis=0) / bn
        self.its = 0
        for c in np.where(res > self.tol)[0]:
            X[:, c], its = gmres_right(lambda v: self.A @ v, lambda v: self.factor_solve(v[:, None])[:, 0],
                                       B[:, c], X[:, c], self.tol, self.maxit)
            res[c] = np.linalg.norm(B[:, c] - self.A @ X[:, c]) / bn[c]
            self.its = max(self.its, its)
        self.res = float(res.max()) if len(res) else 0.0
        return X[:, 0] if one else X

    def free(self):
        if pypardiso is not None:
            self.ps.free_memory(everything=True)


def gmres_right(mv, prec, b, x0, tol, maxit):
    """GMRES on A x = b from x0, right preconditioner prec ~ A^-1; stops at |b - A x| <= tol |b|."""
    r0 = b - mv(x0)
    bn, be = np.linalg.norm(b), np.linalg.norm(r0)
    if be <= tol * bn:
        return x0, 0
    V = np.zeros((len(b), maxit + 1), complex); Z = np.zeros((len(b), maxit), complex)
    H = np.zeros((maxit + 1, maxit), complex)
    V[:, 0] = r0 / be
    for j in range(maxit):
        Z[:, j] = prec(V[:, j])
        w = mv(Z[:, j])
        for _ in range(2):
            h = V[:, :j + 1].conj().T @ w; H[:j + 1, j] += h; w -= V[:, :j + 1] @ h
        H[j + 1, j] = np.linalg.norm(w)
        e1 = np.zeros(j + 2, complex); e1[0] = be
        y = np.linalg.lstsq(H[:j + 2, :j + 1], e1, rcond=None)[0]
        if np.linalg.norm(H[:j + 2, :j + 1] @ y - e1) <= tol * bn or H[j + 1, j] == 0:
            break
        V[:, j + 1] = w / H[j + 1, j]
    return x0 + Z[:, :j + 1] @ y, j + 1


# ------------------------------------------------------------------ device and leads
_QTT = None
QTT_ERROR = ""                      # why qttools_obc() returned None


def qttools_obc():
    """(Spectral, Full) from qttools (QuaTrEx), or None when it does not import or runs on cupy."""
    global _QTT, QTT_ERROR
    if _QTT is None:
        _QTT = False
        # qttools picks its array module from QTX_ARRAY_MODULE and defaults to cupy when cupy is installed (the gpu
        # extra); the OBC wrapper below works on numpy arrays, so ask for numpy unless the user set it explicitly
        os.environ.setdefault("QTX_ARRAY_MODULE", "numpy")
        src = os.environ.get("QTTOOLS_SRC")
        for path in (None, src) if src else (None,):
            if path and path not in sys.path:
                sys.path.append(path)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    from qttools import xp
                    from qttools.boundary_conditions.obc.spectral import Spectral
                    from qttools.nevp import Full
            except ImportError as e:
                QTT_ERROR = f"qttools does not import: {e}"
                continue
            except SyntaxError as e:     # qttools uses PEP 695 `type X[T] = ...`: Python >= 3.12
                QTT_ERROR = (f"qttools needs Python >= 3.12 (running {sys.version.split()[0]}): "
                             f"{e.msg} in {e.filename}:{e.lineno}")
                break
            _QTT = (Spectral, Full) if xp is np else False
            QTT_ERROR = "" if _QTT else QTT_ERROR
            if not _QTT:
                QTT_ERROR = "qttools runs on cupy (xp is not numpy)"
            break
    return _QTT or None


class QttLead:
    """Lead cell (H00, S00, coupling H01, S01 to the next cell along +x) through qttools' spectral OBC at real E.
    sections = primitive cells per lead cell (the NEVP runs on one primitive cell). Per side:
      g  surface Green's function of the semi-infinite lead (cell next to the device included),
      u  lead-cell amplitudes of the flux-normalized incoming modes: the device source is A_dl u
         (QuaTrEx's contact source -A_ji b, with the contact cell eliminated), so B B^H = Gamma on the device face.
    perm: lead-cell orbital order with the primitive cells contiguous along +x and in the same internal order. qttools
    reads the periodic layer from the first (outermost) primitive row, with the cells numbered towards a_ij (inwards):
    +x for the left lead, reversed for the right lead. g and u are returned in the original order."""

    def __init__(self, H00, H01, S00, S01, sections=1, perm=None, **opts):
        Spectral, Full = qttools_obc()
        n = H00.shape[0]
        assert n % sections == 0, f"lead cell {n} not divisible into {sections}"
        m = n // sections
        p = np.arange(n) if perm is None else np.asarray(perm)
        self.p = {"L": p, "R": p.reshape(sections, m)[::-1].ravel()}
        self.H00, self.H01, self.S00, self.S01 = (np.asarray(sp.csr_matrix(M)[p][:, p].toarray(), complex)
                                                  for M in (H00, H01, S00, S01))
        blk = lambda M, i, j: M[i * m:(i + 1) * m, j * m:(j + 1) * m]
        self.toeplitz_dev = []                       # Toeplitz in the primitive cells? (the NEVP uses cell 0's row)
        for nm, M in (("H", self.H00), ("S", self.S00)):
            d = max([abs(blk(M, i, i) - blk(M, 0, 0)).max() for i in range(sections)] +
                    [abs(blk(M, i, i + 1) - blk(M, 0, 1)).max() for i in range(sections - 1)])
            self.toeplitz_dev.append(float(d))
            if d > 1e-2 * max(abs(M).max(), 1):
                raise RuntimeError(f"lead cell is not {sections} identical primitive cells (max |d{nm}| {d:.2e})")
            if d > 1e-8:
                log(f"  qttools lead: primitive cells of the lead cell differ by up to {d:.1e} in {nm}; "
                    f"the OBC uses the first cell's couplings (as QuaTrEx)")
        kw = dict(num_ref_iterations=4)
        kw.update(opts)
        self.sections = sections
        self.solver = Spectral(Full(), block_sections=sections, **kw)

    def modes(self, E, side):
        a00 = E * self.S00 - self.H00                     # in perm order (cells along +x)
        a01 = E * self.S01 - self.H01                     # cell n -> n+1
        a10 = E * self.S01.conj().T - self.H01.conj().T  # cell n+1 -> n
        a_ji, a_ij = (a10, a01) if side == "L" else (a01, a10)   # a_ji: towards the outer cell
        r = np.argsort(self.p["L"])[self.p[side]]         # perm order -> this side's cell order
        a_xx = tuple(M[np.ix_(r, r)][None] for M in (a_ji, a00, a_ij))
        if self.sections > 1:
            # exactly periodic lead from the outermost primitive cell (as QuaTrEx's contact): refining g with the
            # slightly non-periodic cell blocks breaks the recursion (MoS2/HfO2: 8e-3 eV between cells, error 0.72)
            from qttools.toeplitz.toeplitz import periodize_layer
            a_xx = tuple(np.ascontiguousarray(x) for x in periodize_layer(a_xx, self.sections))
        a_ji = a_xx[0][0]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            g, b = self.solver(a_xx, "", return_injected=True)
        bad = [str(x.message) for x in w if "recursion error" in str(x.message)]
        if bad:
            raise RuntimeError(f"qttools OBC {side} at E = {E}: {bad[0]}")
        g = g[0]
        u = g @ (a_ji @ b[0])
        ip = np.argsort(self.p[side])                      # back to the lead-matrix order
        return g[np.ix_(ip, ip)], u[ip]


def lead_cell_perm(inp, orbs, sections):
    """Orbital order of a lead cell (global orbitals orbs, in lead-matrix order) with its `sections` primitive cells
    contiguous along the transport axis and every cell's atoms in the order of the first cell (matched by
    translation by the primitive period)."""
    if sections <= 1:
        return None
    norb, ax = inp["norb"], inp["meta"]["axis"]
    off = np.concatenate([[0], np.cumsum(norb)])
    atom_of = np.repeat(np.arange(len(norb)), norb)
    atoms = np.unique(atom_of[orbs])
    x = inp["xyz"][atoms]
    a = inp["meta"].get("lead_cell_length", np.ptp(x[:, ax]) * sections / (sections - 1)) / sections
    cell = np.floor((x[:, ax] - x[:, ax].min() + 0.25 * a) / a).astype(int)
    assert cell.min() == 0 and cell.max() == sections - 1, f"lead cell atoms span cells {cell.min()}..{cell.max()}"
    first = np.where(cell == 0)[0]
    tree = cKDTree(x[first])
    order = []
    for c in range(sections):
        shift = np.zeros(3); shift[ax] = c * a
        mine = np.where(cell == c)[0]
        d, j = tree.query(x[mine] - shift)
        assert d.max() < 1e-2 and len(np.unique(j)) == len(first) == len(mine), f"lead cell {c} does not match cell 0"
        order.extend(mine[np.argsort(j)])
    pos = {o: i for i, o in enumerate(orbs)}                      # global orbital -> lead-matrix index
    return np.array([pos[o] for k in order for o in range(off[atoms[k]], off[atoms[k] + 1])])


class Device:
    """Device part of a generic input set: slabs = device blocks, lead coupling, exact Sigma(z).
    obc: "auto" (qttools on the real axis, else the folded-lead decimation), "qttools" (fail instead of falling back
    on the real axis) or "folded" (folded lead only; the old name "mgms" is accepted)."""

    qtt_eta = 1e-5                  # |Im z| up to which Sigma(z) is taken from qttools at E = Re z

    def __init__(self, inp, fast_lead=None, obc="auto"):
        self.inp, m = inp, inp["meta"]
        self.meta = m
        norb, blocks = inp["norb"], inp["blocks"]
        self.off = np.concatenate([[0], np.cumsum(norb)])
        ob = np.repeat(blocks, norb)
        O = {b: np.where(ob == b)[0] for b in np.unique(blocks)}
        tr = m["transport"]
        self.dev_blocks = tr["device_blocks"]
        self.slab_orbs = [O[b] for b in self.dev_blocks]              # global orbitals of each slab
        self.ns = [len(o) for o in self.slab_orbs]
        self.sb = np.concatenate([[0], np.cumsum(self.ns)])           # slab bounds in device order
        dev = np.concatenate(self.slab_orbs)
        self.dev = dev
        self.n = len(dev)
        H, S = inp["H"], inp["S"]
        self.H = H[dev][:, dev].tocsr()
        self.S = S[dev][:, dev].tocsr()
        self.loc = -np.ones(H.shape[0], int)                         # global orbital -> slab-local
        for o in self.slab_orbs:
            self.loc[o] = np.arange(len(o))
        for M in (self.H, self.S):                                  # slab by slab: no global |H| + |S| (12 nm: int64)
            for s, (r0, r1) in enumerate(zip(self.sb[:-1], self.sb[1:])):
                q = slice(M.indptr[r0], M.indptr[r1])
                c = M.indices[q]
                far = (c < self.sb[max(s - 1, 0)]) | (c >= self.sb[min(s + 2, len(self.ns))])
                assert not (M.data[q][far] != 0).any(), "device blocks are not block tridiagonal"
        self.lead = FoldedLead(inp["lead"]["H00"], inp["lead"]["H01"], inp["lead"]["S00"], inp["lead"]["S01"],
                               fast=fast_lead)
        lr = inp.get("lead_right")                                  # different right lead (e.g. under bias)
        self.leadR = self.lead if lr is None else FoldedLead(lr["H00"], lr["H01"], lr["S00"], lr["S01"],
                                                             fast=fast_lead)
        oL = O[tr["lead_left_block"]][self.lead.oL]                # lead-cell orbitals carrying g_L, g_R
        oR = O[tr["lead_right_block"]][self.leadR.oR]
        # nonzeros (row, device column) of H and S between a lead block and the device
        nz = {blk: [M[O[tr[blk]]][:, dev].nonzero() for M in (H, S)] for blk in ("lead_left_block", "lead_right_block")}
        self.bL, self.bR = (np.unique(np.concatenate([c for _, c in nz[blk]]))
                            for blk in ("lead_left_block", "lead_right_block"))
        assert self.bL.max() < self.sb[1] and self.bR.min() >= self.sb[-2], "leads couple beyond end slabs"
        for o, blk in ((oL, "lead_left_block"), (oR, "lead_right_block")):
            rows = O[tr[blk]][np.unique(np.concatenate([r for r, _ in nz[blk]]))]
            assert np.isin(rows, o).all(), f"{blk}: device couples to lead orbitals off its face"
        cut = lambda M, r, c: M[r][:, c].toarray()
        # A = z S - H blocks between device boundary and lead face: (lead, dev) and (dev, lead), CSR
        scut = lambda M, r, c: M[r][:, c].tocsr()
        self.cL = [scut(M, oL, dev[self.bL]) for M in (H, S)] + [scut(M, dev[self.bL], oL) for M in (H, S)]
        self.cR = [scut(M, oR, dev[self.bR]) for M in (H, S)] + [scut(M, dev[self.bR], oR) for M in (H, S)]
        # qttools path: whole lead cell (not only its faces) against the device boundary orbitals
        obc = {"mgms": "folded"}.get(obc, obc)                 # old name of the folded-lead OBC
        assert obc in ("auto", "qttools", "folded"), obc
        self.obc, self.qtt, self._inj = obc, None, {}
        if obc != "folded" and qttools_obc() is not None:
            sec = int(m.get("source", {}).get("cells_per_block", 1))
            perm = lead_cell_perm(inp, O[tr["lead_left_block"]], sec)
            lead = inp["lead"]
            try:
                self.qtt = QttLead(lead["H00"], lead["H01"], lead["S00"], lead["S01"], sec, perm)
                self.qttR = self.qtt if lr is None else QttLead(lr["H00"], lr["H01"], lr["S00"], lr["S01"], sec, perm)
            except RuntimeError as e:
                if obc == "qttools":
                    raise
                log(f"  qttools OBC not used: {e}")
                self.qtt = None
            self.qL = [cut(M, O[tr["lead_left_block"]], dev[self.bL]) for M in (H, S)] + \
                      [cut(M, dev[self.bL], O[tr["lead_left_block"]]) for M in (H, S)]
            self.qR = [cut(M, O[tr["lead_right_block"]], dev[self.bR]) for M in (H, S)] + \
                      [cut(M, dev[self.bR], O[tr["lead_right_block"]]) for M in (H, S)]
        elif obc == "qttools":
            raise RuntimeError(f"obc='qttools': {QTT_ERROR or 'qttools does not import'} (install the qtx extra, set QTTOOLS_SRC)")

    sigma_cache = None              # directory: Sigma(z) is read from / written to sig_<Re z>_<Im z>.npz there
                                    # (qsig_... with the injected modes BL, BR for the qttools path)

    def uses_qttools(self, z):
        return self.qtt is not None and abs(complex(z).imag) <= self.qtt_eta

    def sigma(self, z):
        """Sigma_L on bL, Sigma_R on bR (device indices) at complex z: A_dl g_L A_ld, A_dr g_R A_rd.
        qttools path (real axis): at E = Re z; the injected modes are kept for inject()."""
        z = complex(z)
        q = self.uses_qttools(z)
        if q:
            try:
                return self._sigma_cached(z, "qsig", self._sigma_qtt)
            except RuntimeError as e:
                if self.obc == "qttools":
                    raise
                log(f"  {e}; folded-lead Sigma instead")
        return self._sigma_cached(z, "sig", self._sigma)

    def _sigma_cached(self, z, pre, fun):
        f = self.sigma_cache and os.path.join(self.sigma_cache, f"{pre}_{z.real:+.8f}_{z.imag:.8e}.npz")
        if f and os.path.exists(f):
            c = np.load(f)
            out = c["SL"], c["SR"]
            if "BL" in c:
                self._inj = {id(out[0]): c["BL"], id(out[1]): c["BR"]}
            return out
        out, inj = fun(z)
        self._inj = {id(out[0]): inj[0], id(out[1]): inj[1]} if inj else {}
        if f:
            os.makedirs(self.sigma_cache, exist_ok=True)
            tmp = f"{f}.tmp{socket.gethostname()}-{os.getpid()}.npz"   # several ranks (nodes) may compute the same point
            np.savez(tmp, SL=out[0], SR=out[1], **(dict(BL=inj[0], BR=inj[1]) if inj else {}))
            os.replace(tmp, f)
        return out

    def _sigma_qtt(self, z):
        E = z.real
        out, inj = [], []
        for side, lead, (Hld, Sld, Hdl, Sdl) in (("L", self.qtt, self.qL), ("R", self.qttR, self.qR)):
            g, u = lead.modes(E, side)
            Adl = E * Sdl - Hdl
            out.append(Adl @ g @ (E * Sld - Hld))
            inj.append(Adl @ u)
        return tuple(out), tuple(inj)

    def inject(self, Sg, tol=1e-4):
        """Injection columns for the self-energy Sg returned by the last sigma(): the qttools incoming modes when
        they came with it, else the Gamma eigenchannels above tol (injection())."""
        B = self._inj.get(id(Sg))
        if B is None and gpu.on:                                 # dense eigh of Gamma on the device
            G = gpu.asarray(Sg)
            G = 1j * (G - G.conj().T)
            lam, X = gpu.cp.linalg.eigh(0.5 * (G + G.conj().T))
            k = lam > tol
            B = gpu.asnumpy(X[:, k] * gpu.cp.sqrt(lam[k]))
            del G, lam, X
            gpu.free()
            return B
        return injection(Sg, tol)[0] if B is None else B

    def _sigma(self, z):
        gL, gR = self.lead.surface_gfs(z)
        if self.leadR is not self.lead:
            gR = self.leadR.surface_gfs(z)[1]
        return (self._coupled(z, gL, self.cL), self._coupled(z, gR, self.cR)), None

    @staticmethod
    def _coupled(z, g, c):
        """A_dl g A_ld (numpy) of the sparse couplings c = (H_ld, S_ld, H_dl, S_dl), A = z S - H; GPU mode: on the device."""
        Hld, Sld, Hdl, Sdl = (sp.csr_matrix(M) for M in c)
        Adl, AldT = (z * Sdl - Hdl).tocsr(), (z * Sld - Hld).T.tocsr()
        if gpu.on:
            import cupyx.scipy.sparse as csp
            Adl, AldT, g = csp.csr_matrix(Adl), csp.csr_matrix(AldT), gpu.asarray(g)
        return gpu.asnumpy((AldT @ (Adl @ g).T).T)

    def A(self, E, SL, SR):
        """Sparse fine operator E S - H - Sigma_L - Sigma_R."""
        rows, cols, vals = [], [], []
        for Sg, b in ((SL, self.bL), (SR, self.bR)):
            ii, jj = np.meshgrid(b, b, indexing="ij")
            rows.append(ii.ravel()); cols.append(jj.ravel()); vals.append(Sg.ravel())
        Sig = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(self.n, self.n))
        return (E * self.S - self.H - Sig).tocsr()


def sancho_both(A00, A01, A10, tol=1e-11, maxit=300):
    """Lopez-Sancho decimation of the chain with on-site A00 = z S00 - H00 and couplings A01 (cell n ->
    n+1), A10 (n+1 -> n), general (complex, non-Hermitian at complex z). Returns the surface Green's
    functions gR of the right-extending chain (its first cell) and gL of the left-extending chain (its
    last cell), both from one decimation. numpy or cupy arrays (the result lives where the input does)."""
    xp = gpu.xp_of(A00)
    e = A00.astype(complex)
    esR, esL = e.copy(), e.copy()
    a, b = A01.astype(complex), A10.astype(complex)
    n = len(e)
    for it in range(maxit):
        X = gpu.solve(e, xp.hstack([a, b]))
        ga, gb = X[:, :n], X[:, n:]
        agb, bga = a @ gb, b @ ga
        esR -= agb
        esL -= bga
        e -= agb + bga
        a, b = -a @ ga, -b @ gb
        if max(float(xp.abs(a).max()), float(xp.abs(b).max())) < tol:
            break
    else:
        log(f"  sancho_both: no convergence after {maxit} steps")
    return gpu.inv(esL), gpu.inv(esR)


class FoldedLead:
    """Periodic lead cell: H00, S00 and the coupling H01, S01 (cell n -> n+1 along +x).
    Faces F = orbitals of H01 / S01 (rows Rf, columns Lf); the interior I couples only inside its cell
    and is folded onto F at complex z:  A_eff(z) = A_FF - A_FI A_II^-1 A_IF,  A = z S - H.
      general (complex Hermitian H, any S): Sancho decimation of the folded chain on F with
          A01 = z S01 - H01, A10 = z S01^H - H01^H;  g_L, g_R on F (oL = oR = F).
      real symmetric H, S = I (fast): face_surface_gfs_small, chain on the smaller face;
          g_L on Rf (oL), g_R on Lf (oR).
    """

    def __init__(self, H00, H01, S00, S01, fast=None):
        H00, H01, S00, S01 = (sp.csr_matrix(M) for M in (H00, H01, S00, S01))
        self.n = H00.shape[0]
        P01 = abs(H01) + abs(S01)
        self.Rf = np.unique(P01.nonzero()[0])
        self.Lf = np.unique(P01.nonzero()[1])
        F = np.union1d(self.Lf, self.Rf)
        self.F, self.I = F, np.setdiff1d(np.arange(self.n), F)
        orth = abs(S00 - sp.identity(self.n)).max() < 1e-14 and (S01.nnz == 0 or abs(S01).max() < 1e-14)
        real = not (np.iscomplexobj(H00.data) or np.iscomplexobj(H01.data))
        self.fast = (orth and real) if fast is None else fast
        assert not self.fast or (orth and real), "fast lead path needs real H and S = I"
        pos = {v: i for i, v in enumerate(F)}
        self.iL = np.array([pos[c] for c in self.Lf]); self.iR = np.array([pos[r] for r in self.Rf])
        self.oL, self.oR = (self.Rf, self.Lf) if self.fast else (F, F)
        I = self.I
        self.HFF, self.SFF = H00[F][:, F].toarray(), S00[F][:, F].toarray()
        self.HFI, self.SFI = H00[F][:, I].toarray(), S00[F][:, I].toarray()
        self.HIF, self.SIF = H00[I][:, F].toarray(), S00[I][:, F].toarray()
        self.HII, self.SII = H00[I][:, I].tocsr(), S00[I][:, I].tocsr()
        self.H01, self.S01 = H01[F][:, F].toarray(), S01[F][:, F].toarray()

    def surface_gfs(self, z):
        if gpu.on and not self.fast:
            return self._surface_gfs_gpu(z)
        Aeff = z * self.SFF - self.HFF
        if len(self.I):                             # no interior when H01 couples the whole cell
            cs = CSolver(z * self.SII - self.HII)
            X = cs.solve(z * self.SIF - self.HIF)
            cs.free()
            Aeff = Aeff - (z * self.SFI - self.HFI) @ X
        if self.fast:                               # Aeff = z - Heff, Heff complex symmetric
            Heff = z * np.eye(len(Aeff)) - Aeff
            V = self.H01[np.ix_(self.iR, self.iL)]
            gL, gR, _ = face_surface_gfs_small(0.5 * (Heff + Heff.T), self.iL, self.iR, V, z)
            return gL, gR
        A01 = z * self.S01 - self.H01
        A10 = z * self.S01.conj().T - self.H01.conj().T
        return sancho_both(Aeff, A01, A10)

    def _surface_gfs_gpu(self, z):
        """General path on the GPU: dense LU of the interior, decimation on the device; numpy results."""
        d = getattr(self, "_dev", None)
        if d is None:
            d = self._dev = {k: gpu.asarray(getattr(self, k)) for k in
                             ("HFF", "SFF", "HFI", "SFI", "HIF", "SIF", "H01", "S01")}
            d["HII"], d["SII"] = gpu.csr(self.HII, np.complex128), gpu.csr(self.SII, np.complex128)
        Aeff = z * d["SFF"] - d["HFF"]
        if len(self.I):
            lu = gpu.lu_factor((z * d["SII"] - d["HII"]).toarray())
            X = gpu.lu_solve(lu, (z * d["SIF"] - d["HIF"]).astype(complex))
            del lu
            Aeff = Aeff - (z * d["SFI"] - d["HFI"]) @ X
        gL, gR = sancho_both(Aeff, z * d["S01"] - d["H01"], z * d["S01"].conj().T - d["H01"].conj().T)
        return gpu.asnumpy(gL), gpu.asnumpy(gR)


def injection(SL, tol=1e-4):
    """Columns B with Gamma_L ~ B B^H: eigenvectors of Gamma_L above tol [eV] (propagating channels)."""
    G = 1j * (SL - SL.conj().T)
    lam, X = np.linalg.eigh(0.5 * (G + G.conj().T))
    k = lam > tol
    return X[:, k] * np.sqrt(lam[k]), lam, X


# ------------------------------------------------------------------ level 1: partition
def find_cores(labels, geo, core_label="Si", d_core=6.0):
    """Core flag per atom: label core_label and >= d_core from every atom with another label."""
    other = labels != core_label
    if not other.any():
        return labels == core_label, np.full(len(labels), np.inf)
    d, _ = geo.tree(np.where(other)[0]).query(geo.points(np.arange(len(labels))), k=1)
    return (~other) & (d >= d_core), d


def bisect(atoms, pos2d, w, nt):
    """Split atoms into nt groups of ~equal weight w by recursive bisection along the longer axis."""
    if nt <= 1 or len(atoms) <= 1:
        return [atoms]
    ax = int(np.argmax(np.ptp(pos2d[atoms], axis=0)))
    o = atoms[np.argsort(pos2d[atoms, ax], kind="stable")]
    cum = np.cumsum(w[o])
    k1 = nt // 2
    cut = int(np.searchsorted(cum, cum[-1] * k1 / nt)) + 1
    return bisect(o[:cut], pos2d, w, k1) + bisect(o[cut:], pos2d, w, nt - k1)


def parse_cut(spec):
    """Tile cuts, ';'-separated, applied in order: '<axis>:<c1>,<c2>,...@<s1>,<s2>,...' (cut planes) or
    '<axis>:n<k>@<slabs>' (k pieces of equal orbital weight) -> list of dict(axis, n, at, slabs) (see partition);
    None / '' -> None."""
    if not spec:
        return None
    out = []
    for item in spec.split(";"):
        ax, rest = item.split(":")
        at, sl = rest.split("@") if "@" in rest else (rest, "")
        out.append(dict(axis=int(ax), n=int(at[1:]) if at.startswith("n") else None,
                        at=None if at.startswith("n") else sorted(float(x) for x in at.split(",")),
                        slabs=[int(x) for x in sl.split(",")] if sl else None))
    return out


def cut_pieces(atoms, coord, at, period=None):
    """Atoms grouped by the intervals between the cut positions `at` along one coordinate; with a period the
    coordinate is taken modulo it and the pieces below at[0] and above at[-1] are one piece (wrap-around)."""
    c = coord[atoms] % period if period else coord[atoms]
    k = np.searchsorted(np.asarray(at), c)
    if period:
        k[k == len(at)] = 0
    return [np.sort(atoms[k == j]) for j in np.unique(k)]


def partition(dev, cores=True, core_label="Si", d_core=6.0, tile_orbs=2500, axis=0, xsplit=1, cut=None):
    """Blocks per slab: dict(kind, slab, atoms (global atom ids, sorted), orbs (slab-local orbitals)).
    Tiles: recursive transverse bisection to ~tile_orbs orbitals, then each tile cut into xsplit pieces of equal orbital
    weight along the transport axis (xsplit = 1: tiles span the whole slab in x).
    cut = [dict(axis, n, at, slabs), ...] (parse_cut): instead of the bisection, the slabs in `slabs` (all if None) are cut at the
    planes coord[axis] = at (periodic axis: modulo the period, the outermost pieces joined), every other slab is one
    tile; '0:n2@1,2' = slabs 1, 2 cut into two halves of equal orbital weight along x, slabs 0, 3 one tile each;
    '2:8.5,15.5@1,2' = the MoS2 layer and the (wrapped) oxide of slabs 1, 2 as two tiles each; '0:n2@1,2;2:9,15.5@1,2'
    both (4 tiles in slabs 1, 2)."""
    inp = dev.inp
    labels, xyz, norb = inp["labels"], inp["xyz"], inp["norb"]
    core, dmin = (find_cores(labels, Geometry(xyz, dev.meta.get("periodic")), core_label, d_core) if cores
                  else (np.zeros(len(labels), bool), None))
    tr = [i for i in range(3) if i != axis]
    blocks = []
    for s, b in enumerate(dev.dev_blocks):
        a = np.where(inp["blocks"] == b)[0]
        if core[a].any():
            blocks.append(dict(kind="core", slab=s, atoms=a[core[a]]))
        rest = a[~core[a]]
        if cut is not None:
            pieces = [rest]
            for c in cut:
                if c["slabs"] is not None and s not in c["slabs"]:
                    continue
                per = dict(periodic_axes(dev.meta.get("periodic"))).get(c["axis"])
                pieces = [q for t in pieces for q in (
                    [np.sort(u) for u in bisect(t, xyz[:, [c["axis"]]], norb, c["n"])] if c["n"]
                    else cut_pieces(t, xyz[:, c["axis"]], c["at"], per))]
        else:
            pieces = bisect(rest, xyz[:, tr], norb, max(1, int(round(norb[rest].sum() / tile_orbs))))
        for j, t in enumerate(pieces):
            for tx in bisect(t, xyz[:, [axis]], norb, xsplit):
                blocks.append(dict(kind="tile", slab=s, atoms=np.sort(tx), tpar=j))   # tpar: transverse piece
    for blk in blocks:
        g = np.concatenate([np.arange(dev.off[i], dev.off[i + 1]) for i in blk["atoms"]])
        blk["orbs"] = dev.loc[g]
    return blocks, core, dmin


# ------------------------------------------------------------------ level 1: cores


# ------------------------------------------------------------------ level 1: tiles
def periodic_axes(periodic):
    """[(axis, length)] from meta["periodic"]: None, {"axis": k, "length": L} or a list of such dicts."""
    if not periodic:
        return []
    return [(int(p["axis"]), float(p["length"])) for p in ([periodic] if isinstance(periodic, dict) else periodic)]


class Geometry:
    """Distance queries on atom positions, minimum image along the periodic transverse axes
    (meta["periodic"]: {"axis": k, "length": L} or a list of them, e.g. a 2D sheet folded at k_y = 0, or a
    QuaTrEx Gamma-point supercell periodic in y and z)."""

    def __init__(self, xyz, periodic=None):
        self.xyz = np.asarray(xyz, float)
        self.box = None
        self.per = periodic_axes(periodic)
        if self.per:
            self.lo = self.xyz.min(0) - 1.0
            self.box = np.full(3, 2 * np.ptp(self.xyz, axis=0).max() + 1e3)
            for ax, L in self.per:
                self.box[ax] = L

    def _map(self, r):
        if self.box is None:
            return r
        q = r - self.lo
        for ax, L in self.per:
            q[:, ax] %= L
        return q

    def tree(self, idx=None):
        r = self.xyz if idx is None else self.xyz[idx]
        return cKDTree(self._map(r), boxsize=self.box)

    def points(self, idx):
        return self._map(self.xyz[idx])


ENV_MIN_COMP = 2000        # environment components smaller than this are factorized together


# ------------------------------------------------------------------ level 1: tile embedding by message passing


# ------------------------------------------------------------------ level 1: assembly


# ------------------------------------------------------------------ slab bases: transport residual enrichment


# ------------------------------------------------------------------ level 1: residual enrichment


# ------------------------------------------------------------------ level 2: contour slab density matrix


# ------------------------------------------------------------------ exact solve on the GPU

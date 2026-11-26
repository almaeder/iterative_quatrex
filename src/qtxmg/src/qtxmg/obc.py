"""Lead self-energies Sigma_L(z), Sigma_R(z) on the device faces and the injection B (B B^H = Gamma_L) at the real energies.

    model "beyn"    Beyn on the periodized lead (from its outermost primitive cell, as QuaTrEx) at every point; a point with
                    |Im z| <= 1e-5 is taken at E = Re z; injection = the incoming propagating modes
    model "folded"  the lead cell as given, interior folded onto its face, Sancho decimation; injection = eigenchannels of Gamma_L

OBC.prepare() computes every point once (the points are dealt out to the ranks); with a cache folder the results are files
that later runs reuse and that every rank loads when it needs them.
"""
import os
import time

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from .backend import gpu, log, to_host, xp
from .comm import Comm, wait_file

try:                                          # release layout: qtxmg/beyn.py
    from .beyn import BeynLead
except ImportError:                           # development layout: qtx_ms/qtxmg/
    from ..obc_beyn import BeynLead


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
    """Sigma and injection of every point, computed in prepare() and kept in memory, or (cache folder) in files
    obc_<model>_<Re z>_<Im z>.npz that are loaded on first use. The Beyn coupling bases are stored in the cache folder too."""
    real_eta = 1e-5

    def __init__(self, device, model="beyn", cache=None, inj_tol=1e-4, comm=None):
        assert model in ("beyn", "folded"), model
        self.dev, self.model, self.cache, self.inj_tol = device, model, cache, inj_tol
        self.comm = comm or Comm()
        assert cache or self.comm.size == 1, "several ranks: the OBC needs a cache folder (--cache)"
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

    def compute(self, z, real):
        """[Sigma_L, Sigma_R], B (None off the real axis) at one point (host arrays)."""
        if self._leads is None:
            self._make_leads()
        out, B = [], None
        for side, lead in enumerate(self._leads):
            if self.model == "folded":
                Hld, Sld, Hdl, Sdl = self.dev.cpl[side]
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
        """Sigma at the contour points zs and at the real energies reals (E + i eta), the injection at reals. Collective: the
        missing points are dealt out to the ranks."""
        t, comm = time.time(), self.comm
        pts = [(complex(z), False) for z in zs] + [(complex(z), True) for z in reals]
        have = lambda z, real: (self.cache and os.path.exists(self._file(z))) or (z in self.S and (not real or z in self.B))
        todo = comm.bcast([(z, real) for z, real in pts if not have(z, real)])
        mine = todo[comm.rank::comm.size]
        for z, real in mine:
            S, B = self.compute(z, real)
            if self.cache:
                os.makedirs(self.cache, exist_ok=True)
                f = self._file(z)
                np.savez(f"{f}.{comm.rank}.tmp.npz", SL=S[0], SR=S[1], **({"B": B} if B is not None else {}))
                os.replace(f"{f}.{comm.rank}.tmp.npz", f)
            else:
                self.S[z] = S
                if real:
                    self.B[z] = B
        if self.cache:
            comm.barrier()
            for z, _ in todo:
                wait_file(self._file(z))
        log(f"obc: {self.model}, {len(zs)} contour and {len(reals)} real points ({len(todo)} computed on {comm.size} ranks), "
            f"{time.time() - t:.1f} s")

    def _load(self, z):
        d = np.load(self._file(z))
        self.S[z] = [d["SL"], d["SR"]]
        self.B[z] = d["B"] if "B" in d.files else None

    def __call__(self, z):
        """[Sigma_L, Sigma_R] at z (host arrays)."""
        z = complex(z)
        if z not in self.S:
            self._load(z)
        return self.S[z]

    def inject(self, z):
        z = complex(z)
        if z not in self.B:
            self._load(z)
        return self.B[z]

    def drop(self, zs):
        """Forget the Sigma of the contour points (only the bases need them)."""
        for z in zs:
            self.S.pop(complex(z), None)
            self.B.pop(complex(z), None)

"""The device of an input set (prep.inputs): H and S of the device blocks, geometry, lead faces and couplings; the rows of
S and H on the array device (DeviceMatrices), from which every dense window and the fine operator are filled."""
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from .backend import offsets, on_gpu, xp


def periodic_axes(periodic):
    """[(axis, length)] from meta["periodic"]: None, {"axis": k, "length": L} or a list of such dicts."""
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
    """Device part of an input set: the device blocks in transport order, H and S on them (host CSR), the lead faces bL / bR
    (device orbitals coupled to a lead) and, per side, the lead cell, its orbitals in the input and the couplings
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
        self.inp = {k: v for k, v in inp.items() if k not in ("H", "S")}     # the input H, S are not kept (large devices)


class DeviceMatrices:
    """Rows of the device S and H on the array device: one CSR pattern (of S + i H), two real value arrays, global columns.
    rows: the device orbitals held, in this order (default: all). Dense blocks and the operator E S - H are filled there."""

    def __init__(self, S, H, rows=None):
        rows = np.arange(S.shape[0]) if rows is None else np.asarray(rows)
        M = (S[rows] + 1j * H[rows]).tocsr()
        M.sort_indices()
        self.n = S.shape[1]
        self.row_of = np.full(self.n, -1, np.int64)                   # device orbital -> row held here
        self.row_of[rows] = np.arange(len(rows))
        self.indptr_host = M.indptr.astype(np.int64)
        self.indptr, self.indices = xp.asarray(self.indptr_host), xp.asarray(M.indices.astype(np.int32))
        self.s, self.h = xp.asarray(M.data.real.copy()), xp.asarray(M.data.imag.copy())

    @property
    def nbytes(self):
        return sum(a.nbytes for a in (self.indptr, self.indices, self.s, self.h))

    def dense(self, rows, cols=None):
        """Dense S[rows, cols], H[rows, cols] (float64): gather the nonzeros of the rows, keep those in cols."""
        cols = rows if cols is None else cols
        r = self.row_of[rows]
        assert (r >= 0).all(), "rows not held here"
        start, count = self.indptr_host[r], self.indptr_host[r + 1] - self.indptr_host[r]
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

    def operator(self, E, nrows=None):
        """E S - H of the first nrows rows held (default: all) as a complex128 CSR matrix over all device columns."""
        nrows = len(self.indptr_host) - 1 if nrows is None else nrows
        nnz = int(self.indptr_host[nrows])
        data = (E * self.s[:nnz] - self.h[:nnz]).astype(complex)
        if on_gpu():
            import cupyx.scipy.sparse as csp
            return csp.csr_matrix((data, self.indices[:nnz], self.indptr[:nrows + 1].astype(np.int32)), shape=(nrows, self.n))
        return sp.csr_matrix((data, self.indices[:nnz], self.indptr[:nrows + 1]), shape=(nrows, self.n))

"""The density-matrix hierarchy and its V-cycle, on one or several GPUs (one rank each).

Ownership: a level-1 block (fine core) belongs to the rank of its x position (contiguous chunks of x positions of about equal
window cost); a merged block to the owner of its first kid; a window to the owner of its parents. A vector of level k is held
as the rows of the blocks a rank owns (comm.Part); a rank reads the rows of other ranks it needs through a halo (comm.Halo).

Setup (once):
    bases U_k        computed by the owners of the windows, then copied to every rank (host memory; uploaded when used)
    Galerkin blocks  S_k[I, J], H_k[I, J] = U_I^T (S, H)_{k-1}[kids I, kids J] U_J for the coupled pairs I <= J: on the owner of I,
                     then copied to every rank
    lead rows Y_k    rows of P_1 ... P_k on the lead faces: every rank
    level windows    block order, projected CAP blocks: on the window's owner
Per energy:
    A    = E S - H - Sigma_L - Sigma_R on the owned fine rows (complex128)
    S_0  RAS over the owned fine windows: M_w = E S_w - H_w + i C_w - Sigma[face], core rows of M_w^-1 (complex64)
    A_k  owned block rows of E S_k - H_k - Y_k^T Sigma Y_k (float32 / complex64)
    S_k  RAS over the owned level-k windows: A_k[g, g] + i C_k,g, rows of the parents' kids of the inverse
    A_K^-1  block-tridiagonal LU (complex128) over the slabs of level K, distributed: domains of slabs separated by single slabs,
            one domain per rank; the separator system on one rank (Bottom)
V-cycle: z1 = S_0 r;  z2 = z1 + P_1 C_1 P_1^T (r - A z1);  M^-1 r = z2 + S_0 (r - A z2);  C_k the same on A_k, C_K = A_K^-1.
"""
import hashlib
import json
import os
import time

import numpy as np
import scipy.sparse as sp

from .backend import (C8, Blocks, free, inverse_rows, log, lu_factor, lu_solve, offsets, on_gpu, real_matmul, sync, to_host, xp)
from .comm import Comm, Halo, Part, chunks, domains, swap, wait_file
from .device import DeviceMatrices
from .structure import Level, fine_windows, merge

F4 = np.float32


def _cat(parts):
    return np.concatenate(parts) if len(parts) else np.zeros(0, int)


# ===================================================================================================================== block-tridiagonal LU
def tri_factor(D, U, L):
    """Block LU from the left of the block-tridiagonal matrix with diagonal D, upper U, lower L (lists): inverses of the Schur
    complements T_a = D_a - L_a-1 X_a-1 and X_a = T_a^-1 U_a. D and U are consumed."""
    inv, X = [], []
    for a in range(len(D)):
        T = D[a] if not a else D[a] - L[a - 1] @ X[-1]
        D[a] = None
        inv.append(xp.linalg.inv(T))
        del T
        if a + 1 < len(D):
            X.append(inv[a] @ U[a])
            U[a] = None
    return inv, X, L


def tri_solve(f, ys):
    """Solve with the factors of tri_factor; ys: right-hand side per slab (None = zero). Returns the solution per slab."""
    inv, X, L = f
    t = []
    for a, y in enumerate(ys):                                        # forward: t_a = T_a^-1 (y_a - L_a-1 t_a-1)
        v = y if not a or t[-1] is None else (-(L[a - 1] @ t[-1]) if y is None else y - L[a - 1] @ t[-1])
        t.append(None if v is None else inv[a] @ v)
    for a in range(len(ys) - 2, -1, -1):                              # backward: x_a = t_a - X_a x_a+1
        t[a] = -(X[a] @ t[a + 1]) if t[a] is None else t[a] - X[a] @ t[a + 1]
    return t


class Bottom:
    """A_K^-1 by a block-tridiagonal LU over the slabs of level K (its x positions; neighbours joined when blocks couple farther).

    Several ranks: the slabs are split into p domains separated by single slabs (comm.domains). Every domain rank factors its
    interior and forms its Schur contributions on its one or two separators; the holder of the separators (a rank without a
    domain, else the one with the cheapest domain) factors the separator system, which is block tridiagonal too:
        Sep_j = A_ss - sum over the two neighbouring domains d of A_sd A_dd^-1 A_ds.
    Solve: interior solves, separator right-hand sides to the holder, separator solve, separator values back, interior solves
    again. One rank: one domain, the plain block-tridiagonal LU."""

    def __init__(self, h):
        k, L, comm = h.K, h.levels[h.K], h.comm
        self.h, self.comm, self.me = h, comm, comm.rank
        ix = np.unique(L.grid[0], return_inverse=True)[1]
        reach = max([abs(int(ix[i]) - int(ix[j])) for i, j in L.pairs()] + [int(np.ptp(ix[lb])) for _, lb, _ in h.Y[k]])
        ix = ix // max(reach, 1)
        self.slabs = [np.where(ix == a)[0] for a in range(int(ix.max()) + 1)]
        self.o = [offsets(h.r[k][s]) for s in self.slabs]             # block offsets per slab
        self.n = [int(v[-1]) for v in self.o]                         # slab sizes
        self.at = {int(b): (a, q) for a, s in enumerate(self.slabs) for q, b in enumerate(s)}
        self.dom, self.sep, self.ranks = domains(np.asarray(self.n, float) ** 3, comm.size)
        p = len(self.dom)
        idle = [r for r in range(comm.size) if r not in self.ranks]
        cost = [sum(self.n[a] ** 3 for a in range(*d)) for d in self.dom]
        self.holder = idle[0] if idle else self.ranks[int(np.argmin(cost))]
        own = np.zeros(len(self.slabs), int)                          # rank of every slab
        for d, (a, b) in enumerate(self.dom):
            own[a:b] = self.ranks[d]
        own[self.sep] = self.holder
        self.d = self.ranks.index(self.me) if self.me in self.ranks else None        # my domain
        self.mine = [a for a in range(len(self.slabs)) if own[a] == self.me]          # my slabs (domain and / or separators)
        ent = [h.entries(k, s) for s in self.slabs]
        part = h.part[k]
        self.part_slabs = Part(comm, ent, own, h.N[k])               # the vector laid out by slabs
        self.gather = Halo(part, _cat([ent[a] for a in self.mine]))   # level layout -> my slabs
        self.scatter = Halo(self.part_slabs, part.idx)                # slab layout -> level layout
        self.pos = {a: xp.asarray(self.gather.pos[ent[a]]) for a in self.mine}
        self.dpos = {a: xp.asarray(self.part_slabs.loc[ent[a]]) for a in self.mine}
        self.back = xp.asarray(self.scatter.pos[part.idx])
        self.desc = f"{len(self.slabs)} slabs ({min(self.n)}..{max(self.n)}), " + (
            f"{p} domains, separators {self.sep} on rank {self.holder}" if p > 1 else "one domain")

    def neighbours(self, d):
        """Separators left and right of domain d (None at the device ends)."""
        p = len(self.dom)
        return (self.sep[d - 1] if d > 0 else None), (self.sep[d] if d < p - 1 else None)

    def block(self, a, b, E, sg):
        """Dense block (slab a, slab b) of A_K = E S_K - H_K - Y^T Sigma Y (complex128)."""
        h, k = self.h, self.h.K
        L, o = h.levels[k], self.o
        M = xp.zeros((self.n[a], self.n[b]), complex)
        for p, i in enumerate(self.slabs[a]):
            for j in L.nbr(int(i)):
                sb, q = self.at[int(j)]
                if sb == b:
                    X = h.SH[k][int(i), int(j)] if i <= j else h.SH[k][int(j), int(i)].transpose(0, 2, 1)
                    M[o[a][p]:o[a][p + 1], o[b][q]:o[b][q + 1]] = E * X[0] - X[1]
        for side, (_, lb, ol) in enumerate(h.Y[k]):
            for p, i in enumerate(lb):
                for q, j in enumerate(lb):
                    (si, pi), (sj, pj) = self.at[int(i)], self.at[int(j)]
                    if si == a and sj == b:
                        M[o[a][pi]:o[a][pi + 1], o[b][pj]:o[b][pj + 1]] -= sg[side][ol[p]:ol[p + 1], ol[q]:ol[q + 1]]
        return M

    def factor(self, E, sg):
        p, n, d, sep = len(self.dom), self.n, self.d, self.sep
        self.fac, self.cpl, contrib = None, {}, {}
        if d is not None:                                             # my domain: interior LU, coupling to its separators
            a0, a1 = self.dom[d]
            I = list(range(a0, a1))
            self.fac = tri_factor([self.block(a, a, E, sg) for a in I], [self.block(a, a + 1, E, sg) for a in I[:-1]],
                                  [self.block(a + 1, a, E, sg) for a in I[:-1]])
            sL, sR = self.neighbours(d)
            if sL is not None:
                self.cpl["AL"], self.cpl["LA"] = self.block(a0, sL, E, sg), self.block(sL, a0, E, sg)
            if sR is not None:
                self.cpl["AR"], self.cpl["RA"] = self.block(a1 - 1, sR, E, sg), self.block(sR, a1 - 1, E, sg)
            m = len(I)
            if sL is not None:                                        # -A_sd A_dd^-1 A_ds from the left separator's columns
                Z = tri_solve(self.fac, [self.cpl["AL"]] + [None] * (m - 1))
                contrib["C", d, "LL"] = -(self.cpl["LA"] @ Z[0])
                if sR is not None:
                    contrib["C", d, "RL"] = -(self.cpl["RA"] @ Z[-1])
            if sR is not None:                                        # ... and from the right one's
                Z = tri_solve(self.fac, [None] * (m - 1) + [self.cpl["AR"]])
                contrib["C", d, "RR"] = -(self.cpl["RA"] @ Z[-1])
                if sL is not None:
                    contrib["C", d, "LR"] = -(self.cpl["LA"] @ Z[0])
        self.sfac = None
        if p > 1:                                                     # the separator system on the holder
            keys = [("C", e, t) for e in range(p) for t, ok in (("LL", e > 0), ("RL", 0 < e < p - 1), ("RR", e < p - 1),
                                                                 ("LR", 0 < e < p - 1)) if ok]

            def shape(q):
                sL, sR = self.neighbours(q[1])
                return {"LL": (n[sL], n[sL]) if sL is not None else None, "RR": (n[sR], n[sR]) if sR is not None else None,
                        "LR": (n[sL], n[sR]) if sL is not None and sR is not None else None,
                        "RL": (n[sR], n[sL]) if sL is not None and sR is not None else None}[q[2]]
            wants = [keys if r == self.holder else [] for r in range(self.comm.size)]
            C = {**contrib, **swap(self.comm, contrib, wants, lambda q: self.ranks[q[1]], shape, complex)}
            if self.me == self.holder:
                self.sfac = tri_factor([self.block(s, s, E, sg) + C["C", j, "RR"] + C["C", j + 1, "LL"] for j, s in enumerate(sep)],
                                       [C["C", j + 1, "LR"] for j in range(p - 2)], [C["C", j + 1, "RL"] for j in range(p - 2)])
        free()

    def __call__(self, r):
        """Owned rows of r (level K) -> owned rows of A_K^-1 r (complex64)."""
        p, n, d, sep, nc = len(self.dom), self.n, self.d, self.sep, r.shape[1]
        rx = self.gather(r)
        ys = {a: rx[self.pos[a]].astype(complex) for a in self.mine}
        x = xp.zeros((self.part_slabs.nloc, nc), complex)
        g, xs, y = {}, {}, None
        if d is not None:                                             # interior solve, separator right-hand sides
            a0, a1 = self.dom[d]
            y = tri_solve(self.fac, [ys[a] for a in range(a0, a1)])
            if "LA" in self.cpl:
                g["g", d, "L"] = -(self.cpl["LA"] @ y[0])
            if "RA" in self.cpl:
                g["g", d, "R"] = -(self.cpl["RA"] @ y[-1])
        if p > 1:
            keys = [("g", e, t) for e in range(p) for t, ok in (("L", e > 0), ("R", e < p - 1)) if ok]
            wants = [keys if q == self.holder else [] for q in range(self.comm.size)]
            gshape = lambda q: (n[self.neighbours(q[1])[0 if q[2] == "L" else 1]], nc)
            G = {**g, **swap(self.comm, g, wants, lambda q: self.ranks[q[1]], gshape, complex)}
            if self.me == self.holder:                                # separator solve
                z = tri_solve(self.sfac, [ys[s] + G["g", j, "R"] + G["g", j + 1, "L"] for j, s in enumerate(sep)])
                for j, s in enumerate(sep):
                    xs["x", j] = z[j]
                    x[self.dpos[s]] = z[j]
            wants = [[] for _ in range(self.comm.size)]               # every domain rank: the values of its separators
            for e in range(p):
                wants[self.ranks[e]] += [("x", j) for j in (e - 1, e) if 0 <= j < p - 1]
            xs.update(swap(self.comm, xs, wants, lambda q: self.holder, lambda q: (n[sep[q[1]]], nc), complex))
        if d is not None:                                             # interior solve with the separator values
            a0, a1 = self.dom[d]
            if p > 1:
                rhs = [ys[a] for a in range(a0, a1)]
                if "AL" in self.cpl:
                    rhs[0] = rhs[0] - self.cpl["AL"] @ xs["x", d - 1]
                if "AR" in self.cpl:
                    rhs[-1] = rhs[-1] - self.cpl["AR"] @ xs["x", d]
                y = tri_solve(self.fac, rhs)
            for q, a in enumerate(range(a0, a1)):
                x[self.dpos[a]] = y[q]
        return self.scatter(x)[self.back].astype(C8)


# ===================================================================================================================== the hierarchy
class Hierarchy:
    def __init__(self, device, inp, sigma, zs, dzs, comm=None, cap=(16.0, 2.0, 2.0), spacing=8.0, split=None, tov=6.0, cut1=1e-5,
                 ups=(), halos=(), cache=None):
        self.dev, self.sigma, self.zs, self.dzs = device, sigma, zs, dzs
        self.comm = comm = comm or Comm()
        me = comm.rank
        width, self.cap_s, self.cap_d = cap
        self.fw, nx = fine_windows(device, inp, width, self.cap_d, spacing, split, tov)
        nw = len(self.fw)
        ny = nw // nx
        self.cores = [w.idx[w.core] for w in self.fw]                # level-1 blocks = cores of the fine windows
        self.core_of = np.empty(device.n, int)                       # core of every device orbital and its position there
        self.pos_in_core = np.empty(device.n, int)
        for b, c in enumerate(self.cores):
            self.core_of[c], self.pos_in_core[c] = b, np.arange(len(c))
        cost = [float(sum(len(self.fw[ix * ny + j].idx) ** 3 for j in range(ny))) for ix in range(nx)]
        owner1 = np.repeat(chunks(cost, comm.size), ny)               # whole x positions: the strips of one window stay together
        self.own1 = [b for b in range(nw) if owner1[b] == me]
        # the fine vector: rows of the owned cores; S, H rows of the owned cores and windows on the GPU (owned rows first)
        self.fine = Part(comm, self.cores, owner1, device.n)
        extra = np.setdiff1d(_cat([self.fw[b].idx for b in self.own1]), self.fine.idx)
        self.mats = DeviceMatrices(device.S, device.H, np.concatenate([self.fine.idx, extra]))
        # levels: level 1 = cores (grid of x positions x strips), then one level per merge
        self.levels = [None, Level(None, (np.arange(nw) // ny, np.arange(nw) % ny), self._core_graph(), np.arange(nw), owner1)]
        halos = list(halos) + [None] * (len(ups) - len(halos))
        for (axis, _), halo in zip(ups, halos):
            self.levels.append(merge(self.levels[-1], axis, halo))
        self.K = len(self.levels) - 1
        self.cuts = [None, cut1] + [c for _, c in ups]
        self.lead_cores = [np.unique(self.core_of[f]) for f in device.faces]
        self.cache = None
        if cache:                                                     # the same tag as the single-GPU solver: shared bases
            tag = json.dumps([device.n, list(cap), spacing, split, tov, cut1, list(ups), halos, to_host(zs).round(10).tolist(),
                              to_host(dzs).round(12).tolist(), getattr(sigma, "model", ""), "single1"], default=str)
            self.cache = os.path.join(cache, "bases_" + hashlib.sha1(tag.encode()).hexdigest()[:12])
            if me == 0:
                os.makedirs(self.cache, exist_ok=True)
        self.t = {}
        log(f"hierarchy: {nw} cores ({nx} x {ny}), {int(self.levels[1].G.sum())} coupled core pairs; levels "
            + ", ".join(f"{k}: {L.n} blocks ({_per_rank(L.owner, comm.size)})" for k, L in enumerate(self.levels) if L is not None)
            + f"; fine rows on the GPU {len(self.fine.idx)} owned + {len(extra)} of the windows")

    def tick(self, key, t0):
        sync()
        self.t[key] = self.t.get(key, 0.0) + time.time() - t0

    def _core_graph(self):
        """Cores coupled by H or S; every core on a lead face is coupled to every other one there (Sigma is dense on the face)."""
        C = sp.csr_matrix((np.ones(self.dev.n, np.float32), (np.arange(self.dev.n), self.core_of)))
        G = np.zeros((len(self.cores),) * 2, bool)
        for M in (self.dev.S, self.dev.H):                            # nonzero pattern, no copy of the indices
            B = sp.csr_matrix((np.ones(M.nnz, np.float32), M.indices, M.indptr), shape=M.shape)
            G |= (C.T @ B @ C).toarray() > 0
        for f in self.dev.faces:
            u = np.unique(self.core_of[f])
            G[np.ix_(u, u)] = True
        return G | G.T

    def entries(self, k, blocks):
        """Positions of the blocks in the level-k vector."""
        return _cat([np.arange(self.off[k][b], self.off[k][b + 1]) for b in blocks])

    def owned(self, k):
        return [b for b in range(self.levels[k].n) if self.levels[k].owner[b] == self.comm.rank]

    # ------------------------------------------------------------------------------------------------ setup
    def setup(self):
        self.U, self.r, self.off, self.N = {}, {}, {}, {}
        self.SH, self.Y, self.order, self.capw = {}, {}, {}, {}
        self.part, self.transfer = {}, {}
        for k in range(1, self.K + 1):
            t = time.time()
            self._bases(k)
            self.tick("bases", t)
            t = time.time()
            self._galerkin(k)
            self._lead_rows(k)
            self._vectors(k)
            self.tick("galerkin", t)
            log(f"  level {k}: N {self.N[k]}, {self.r[k].min()}..{int(np.median(self.r[k]))}..{self.r[k].max()} per block; "
                f"{self.t['bases']:.0f} s bases so far")
            if k < self.K:
                t = time.time()
                self._windows(k)
                self.tick("cap", t)
            free()
        self.bottom = Bottom(self)
        log(f"  bottom: level {self.K}, {self.bottom.desc}")

    def have_bases(self):
        """The bases of every level are in the cache: no density matrix is built, the contour Sigma are not needed."""
        ok = bool(self.cache) and all(os.path.exists(os.path.join(self.cache, f"U{k}.npz")) for k in range(1, self.K + 1))
        return self.comm.bcast(ok)

    def _bases(self, k):
        """U_k of every block on every rank: from the cache, or computed by the owners of the windows and copied to all."""
        f = self.cache and os.path.join(self.cache, f"U{k}.npz")
        if f and self.comm.bcast(os.path.exists(f)):
            wait_file(f)
            d = np.load(f)
            U = {int(q[1:]): d[q] for q in d.files}
        else:
            mine = self._bases_level1() if k == 1 else self._bases_from_windows(k - 1)
            U = self.comm.merge({b: to_host(u) for b, u in mine.items()})
            if f and self.comm.rank == 0:
                np.savez(f + ".tmp.npz", **{f"b{b}": u for b, u in U.items()})
                os.replace(f + ".tmp.npz", f)
        self.U[k] = Blocks(U)
        self.r[k] = np.array([U[b].shape[1] for b in range(self.levels[k].n)])
        self.off[k] = offsets(self.r[k])
        self.N[k] = int(self.r[k].sum())

    # ---- the density matrix of a window and its eigenvectors
    def _fold(self, S, K, add_sigma, n_rest):
        """sum_q w_q [M(z_q)^-1]_PP of M(z) = z S + K - Sigma(z), P = the rows after the first n_rest (the rest folded in by
        its Schur complement)."""
        acc = xp.zeros((S.shape[0] - n_rest,) * 2, complex)
        for z, w in zip(self.zs, self.dzs):
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

    def _fine_window(self, b, perm=None):
        """Dense S_w, H_w, CAP C_w = s sqrt(p) S_w sqrt(p) of fine window b (in the order perm), and its lead faces
        [(side, window positions, face positions)]."""
        w = self.fw[b]
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
        """U_1 per owned core: eigenvectors of the core block of the local rho of its fine window (CAP + lead Sigma)."""
        U = {}
        for b in self.own1:
            w = self.fw[b]
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
        """U_{k+1} per parent of the owned level-k windows: eigenvectors of its diagonal block of the window's rho."""
        L, N, U = self.levels[k], self.levels[k + 1], {}
        for g, w in enumerate(L.wins):
            if w.owner != self.comm.rank:
                continue
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
                    i_, j_ = int(i), int(j)
                    D[:, a[p]:a[p + 1], b[q]:b[q + 1]] = self.SH[k][i_, j_] if i_ <= j_ else self.SH[k][j_, i_].transpose(0, 2, 1)
        return D

    def _galerkin(self, k):
        """S_k[I, J], H_k[I, J] = U_I^T (S, H)_{k-1}[I, J] U_J of the coupled pairs I <= J: on the owner of I, then on every rank."""
        L, X = self.levels[k], {}
        for I in self.owned(k):
            for J in (int(j) for j in L.nbr(I) if j >= I):
                if k == 1:
                    D = xp.stack(self.mats.dense(self.cores[I], self.cores[J]))
                else:
                    D = self.dense_level(k - 1, L.kids[I], L.kids[J])
                X[I, J] = to_host(self.U[k][I].T @ D @ self.U[k][J])
        self.SH[k] = Blocks(self.comm.merge(X))

    def _lead_rows(self, k):
        """Y_k per side: rows of P_1 ... P_k on the lead face (face x columns of the lead blocks), the lead blocks, their offsets."""
        L, out = self.levels[k], []
        for side, face in enumerate(self.dev.faces):
            lb = np.unique(L.of_core[self.lead_cores[side]])
            o = offsets(self.r[k][lb])
            Y = xp.zeros((len(face), o[-1]))
            for q, B in enumerate(lb):
                B = int(B)
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
                        Y[:, o[q]:o[q + 1]] += Yp[:, op[j]:op[j + 1]] @ self.U[k][B][ok[a]:ok[a + 1]]
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

    # ---- distributed level-k vectors and the transfers to the level below
    def _vectors(self, k):
        """Part of the level-k vector; P_k (level k -> k - 1) and P_k^T on the owned rows (U in float32)."""
        L, me = self.levels[k], self.comm.rank
        self.part[k] = part = Part(self.comm, [np.arange(self.off[k][b], self.off[k][b + 1]) for b in range(L.n)], L.owner, self.N[k])
        own = self.owned(k)
        at = lambda b: int(part.loc[self.off[k][b]])                  # first owned row of block b
        if k == 1:
            fine = self.fine
            ops = [(xp.asarray(fine.loc[self.cores[b]]), self.U[1][b].astype(F4), at(b)) for b in own]

            def P(c):
                y = xp.zeros((fine.nloc, c.shape[1]), C8)
                for rows, U, a in ops:
                    y[rows] = real_matmul(U, c[a:a + U.shape[1]])
                return y

            def PT(x):
                y = xp.zeros((part.nloc, x.shape[1]), C8)
                for rows, U, a in ops:
                    y[a:a + U.shape[1]] = real_matmul(U.T, x[rows])
                return y
            self.transfer[k] = P, PT
            return
        below, kids = self.part[k - 1], lambda b: self.entries(k - 1, L.kids[b])
        # P_k: every rank computes the rows of its own kids, from the coefficients of their parents (a halo of level k)
        parents = [b for b in range(L.n) if (self.levels[k - 1].owner[L.kids[b]] == me).any()]
        hp = Halo(part, self.entries(k, parents))
        up = []
        for b in parents:
            i = kids(b)
            m = below.owner_of[i] == me
            U = self.U[k][b].astype(F4)
            up.append((xp.asarray(below.loc[i[m]]), U if m.all() else U[xp.asarray(np.where(m)[0])], int(hp.pos[self.off[k][b]])))
        # P_k^T: the owner of a block reads the rows of its kids (a halo of level k - 1)
        hr = Halo(below, _cat([kids(b) for b in own]))
        down = [(xp.asarray(hr.pos[kids(b)]), self.U[k][b].astype(F4), at(b)) for b in own]

        def P(c):
            c, y = hp(c), xp.zeros((below.nloc, c.shape[1]), C8)
            for rows, U, a in up:
                y[rows] = real_matmul(U, c[a:a + U.shape[1]])
            return y

        def PT(x):
            x, y = hr(x), xp.zeros((part.nloc, x.shape[1]), C8)
            for rows, U, a in down:
                y[a:a + U.shape[1]] = real_matmul(U.T, x[rows])
            return y
        self.transfer[k] = P, PT

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
        """Per level-k window: block order (the parents' kids last), offsets, size of the kept part (every rank); its projected
        CAP blocks Y_a^T W Y_b on the coupled block pairs (float32; the owner only)."""
        L, N = self.levels[k], self.levels[k + 1]
        self.order[k] = []
        for g, w in enumerate(L.wins):
            kept = np.concatenate([N.kids[I] for I in w.parents])
            order = np.concatenate([np.setdiff1d(w.blocks, kept), kept]).astype(int)
            o = offsets(self.r[k][order])
            self.order[k].append((order, o, int(self.r[k][kept].sum())))
            self.capw[k, g] = []
            if w.owner != self.comm.rank:
                continue
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
                        self.capw[k, g].append((slice(o[qa], o[qa + 1]), slice(o[qb], o[qb + 1]), (Ys[a].T @ Wab @ Ys[b]).astype(F4)))

    # ------------------------------------------------------------------------------------------------ one energy
    def set_energy(self, E, Sig):
        """Fine operator, S_0, and per level the operator A_k and smoother S_k; the LU of A_K. Sig = [Sigma_L, Sigma_R] at E + i eta."""
        self.S0 = self.A = self.Sk = None
        free()
        t = time.time()
        self.E, self.Sig = E, [xp.asarray(S) for S in Sig]
        self._fine_operator(E)
        self.tick("fine operator", t)
        t = time.time()
        self.S0 = self._fine_smoother()
        self.tick("S0", t)
        self.A, self.Sk = {}, {}
        for k in range(1, self.K + 1):
            t = time.time()
            sg = [Y.T @ (S @ Y) for S, (Y, _, _) in zip(self.Sig, self.Y[k])]       # Y_k^T Sigma Y_k
            if k == self.K:
                self.bottom.factor(E, sg)
                self.tick("bottom LU", t)
                break
            blocks = {}                                               # the real blocks E S_k - H_k this rank uses (float32)
            for i, j in self._pairs_used(k):
                X = self.SH[k][i, j]
                blocks[i, j] = (E * X[0] - X[1]).astype(F4)
            sg = [s.astype(C8) for s in sg]
            self.A[k] = self._level_operator(k, blocks, sg)
            self.tick("A_k", t)
            t = time.time()
            self.Sk[k] = self._level_smoother(k, blocks, sg)
            self.tick("level smoothers", t)
            del blocks
            free()

    def _pairs_used(self, k):
        """Block pairs (i <= j) of the owned block rows of A_k and of the owned windows."""
        L, used = self.levels[k], set()
        for b in self.owned(k):
            used.update((min(b, int(j)), max(b, int(j))) for j in L.nbr(b))
        for g, w in enumerate(L.wins):
            if w.owner == self.comm.rank:
                used.update(L.inside(self.order[k][g][0]))
        return sorted(used)

    def _fine_operator(self, E):
        """A on the owned rows, columns in the halo layout; the halo reads every column the owned rows and windows touch."""
        if not hasattr(self, "fine_halo"):
            n_own, nnz = self.fine.nloc, int(self.mats.indptr_host[self.fine.nloc])
            cols = np.unique(to_host(self.mats.indices[:nnz]))
            need = [cols, _cat([self.fw[b].idx for b in self.own1])]
            self.fine_lead = []
            for side, face in enumerate(self.dev.faces):
                mine = np.where(self.fine.owner_of[face] == self.comm.rank)[0]       # owned rows of the face
                if len(mine):
                    need.append(face)
                    self.fine_lead.append((side, xp.asarray(mine), xp.asarray(self.fine.loc[face[mine]]), face))
            self.fine_halo = Halo(self.fine, _cat(need))
            assert (self.fine_halo.pos[cols] >= 0).all()
            self.fine_cols = xp.asarray(self.fine_halo.pos.astype(np.int32))[self.mats.indices[:nnz]]   # columns -> halo rows
            self.fine_next = self.fine.nloc + self.fine_halo.nghost
            self.fine_lead = [(side, mine, rows, xp.asarray(self.fine_halo.pos[face])) for side, mine, rows, face in self.fine_lead]
        nl, nnz = self.fine.nloc, int(self.mats.indptr_host[self.fine.nloc])
        data = (E * self.mats.s[:nnz] - self.mats.h[:nnz]).astype(complex)
        indptr = self.mats.indptr[:nl + 1]
        if on_gpu():
            import cupyx.scipy.sparse as csp
            self.A0 = csp.csr_matrix((data, self.fine_cols, indptr.astype(np.int32)), shape=(nl, self.fine_next))
        else:
            self.A0 = sp.csr_matrix((data, to_host(self.fine_cols), to_host(indptr)), shape=(nl, self.fine_next))
        self.fine_sig = [(rows, cols, self.Sig[side][mine]) for side, mine, rows, cols in self.fine_lead]

    def fine_matvec(self, x):
        """A x on the owned rows = (E S - H) x - Sigma_L x_L - Sigma_R x_R (complex128)."""
        xh = self.fine_halo(x)
        y = self.A0 @ xh
        for rows, cols, S in self.fine_sig:
            y[rows] -= S @ xh[cols]
        return y

    def _fine_smoother(self):
        """S_0: per owned fine window M_w = E S_w - H_w + i C_w - Sigma (lead faces), complex64; keeps the core rows of M_w^-1."""
        ops, pos = [], self.fine_halo.pos
        for b in self.own1:
            w = self.fw[b]
            S, H, C, lead = self._fine_window(b)
            M = xp.asfortranarray((self.E * S - H).astype(C8) + 1j * C.astype(C8))
            del S, H, C
            for side, q, j in lead:
                M[q[:, None], q] -= self.Sig[side][xp.asarray(j)[:, None], xp.asarray(j)].astype(C8)
            R = inverse_rows(M, np.where(w.core)[0])
            ops.append((xp.asarray(pos[w.idx]), xp.asarray(self.fine.loc[self.cores[b]]), R))
            del M

        def S0(r):
            x = self.fine_halo(r).astype(C8)
            z = xp.zeros((self.fine.nloc, r.shape[1]), r.dtype)
            for win, core, R in ops:
                z[core] = R @ x[win]
            return z
        return S0

    def _level_operator(self, k, blocks, sg):
        """A_k on the owned block rows: per block b, [A_k[b, j] for the coupled j] @ x[j] (float32), minus the projected Sigma."""
        L, part = self.levels[k], self.part[k]
        own = self.owned(k)
        lead_own = [(side, lb) for side, (_, lb, _) in enumerate(self.Y[k]) if np.isin(lb, own).any()]
        halo = Halo(part, _cat([self.entries(k, L.nbr(b)) for b in own] + [self.entries(k, lb) for _, lb in lead_own]))
        rows = []
        for b in own:
            nb = [int(j) for j in L.nbr(b)]
            R = xp.concatenate([blocks[b, j] if b <= j else blocks[j, b].T for j in nb], axis=1)
            rows.append((int(part.loc[self.off[k][b]]), R, xp.asarray(halo.pos[self.entries(k, nb)])))
        sig = []
        for side, lb in lead_own:
            _, _, ol = self.Y[k][side]
            q = [a for a, b in enumerate(lb) if L.owner[b] == self.comm.rank]
            r = _cat([np.arange(ol[a], ol[a + 1]) for a in q])
            sig.append((xp.asarray(part.loc[self.entries(k, lb[q])]), xp.asarray(halo.pos[self.entries(k, lb)]), sg[side][xp.asarray(r)]))

        def A(x):
            xh = halo(x)
            y = xp.zeros((part.nloc, x.shape[1]), C8)
            for a, R, c in rows:
                y[a:a + R.shape[0]] = real_matmul(R, xh[c])
            for r, c, S in sig:
                y[r] -= S @ xh[c]
            return y
        return A

    def _level_smoother(self, k, blocks, sg):
        """S_k: per owned level-k window M = A_k[g, g] + i C_k,g (complex64; A_k[g, g] holds -Y^T Sigma Y on its lead blocks);
        keeps the rows of the parents' kids of M^-1. Reads the window rows through one halo, writes the kept rows of other
        ranks back through another."""
        L, part = self.levels[k], self.part[k]
        mine = [g for g, w in enumerate(L.wins) if w.owner == self.comm.rank]
        kept = {}
        for g in mine:
            order, o, n_par = self.order[k][g]
            kept[g] = order[np.searchsorted(o, o[-1] - n_par):]
        read = Halo(part, _cat([self.entries(k, self.order[k][g][0]) for g in mine]))
        write = Halo(part, _cat([self.entries(k, kept[g]) for g in mine]))
        ops = []
        for g in mine:
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
            ops.append((xp.asarray(read.pos[self.entries(k, order)]), xp.asarray(write.pos[self.entries(k, kept[g])]), R))
            del M

        def Sk(r):
            x = read(r)
            z = xp.zeros((part.nloc + write.nghost, r.shape[1]), C8)
            for win, out, R in ops:
                z[out] = R @ x[win]
            return write.rev(z)
        return Sk

    # ------------------------------------------------------------------------------------------------ V-cycle
    def cycle(self, k, r):
        """C_k r: V(1,1)-cycle on A_k; C_K = A_K^-1."""
        if k == self.K:
            return self.bottom(r)
        A, (P, PT) = self.A[k], self.transfer[k + 1]
        z = self.Sk[k](r)
        z = z + P(self.cycle(k + 1, PT(r - A(z))))
        return z + self.Sk[k](r - A(z))

    def precondition(self, r):
        """M^-1 r: V(1,1)-cycle on the fine level with S_0, coarse level 1 by cycle(1, .)."""
        P, PT = self.transfer[1]
        z = self.S0(r)
        z = z + P(self.cycle(1, PT((r - self.fine_matvec(z)).astype(C8))))
        return z + self.S0(r - self.fine_matvec(z))


def _per_rank(owner, size):
    c = np.bincount(np.asarray(owner), minlength=size)
    return f"{c.min()}..{c.max()} per rank" if size > 1 else "1 rank"

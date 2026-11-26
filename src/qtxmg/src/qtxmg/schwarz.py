"""Overlapping Schwarz smoother along the transport axis + two-level cycle with the Galerkin mode-space coarse level.

Subdomains: device orbitals in windows of `width` Å along x, consecutive windows overlapping by `overlap` Å
(restricted additive Schwarz: each orbital takes the correction of the window whose core holds it). The windows cross
the slab interfaces, where the Galerkin residual concentrates.
Local solve on window j (closed, no Sigma):  z_j = V_j f(E - lambda) V_j^H r_j  from the generalized eigenpairs
(H_jj, S_jj) V = S V Lambda, V^H S V = I, computed once (energy independent), with
    f = 1 / (E - lambda) for |E - lambda| > delta, 0 otherwise:
the local states within delta of E (the local resonances that make a closed local solve singular) are left to the
coarse level, the rest (high-energy, valence, oxide components) is inverted exactly. Windows that touch the lead
faces (bL, bR) use a per-energy dense LU of A_jj with Sigma instead (open, no resonance).

Two-level cycle (multiplicative, on a block of right-hand sides):  nu1 x [x += w S r] ; x += P_c r ; nu2 x [x += w S r]
with P_c = U A_c^-1 U^H (Galerkin, the mode-space solve), x0 = P_c b = mode space. Each smoothing step costs one fine
matvec (residual update). Error, residual, T and the charge per atom are compared with the exact psi of a psi file
(qtxmg.prep.compare_qtbm or exact_T --save-psi) after every cycle.

    python -m qtxmg.schwarz --inputs inputs/mos2_hfo2 --psi results_ms/qtbm_cmp/mos2_hfo2_eta0/psi_+0.55434897.npz \\
        --sigma-cache results_ms/sigma_qtx_eta0/mos2_hfo2 --modes results_ms/mos2_hfo2_l2direct/modes/mu-0.15_0.6_n400_lead
"""
import time
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from concurrent.futures import ThreadPoolExecutor

from qtxmg import gpu, msbasis as mb

log = mb.log


class CapSchwarz:
    """Restricted additive Schwarz with windows of `width` Å centred on the slab interfaces and on points between them
    (cores: nearest centre), local solves = LU of A_jj + i s P S_jj P, P a quadratic absorbing ramp over the outer d Å
    towards each artificial cut (smoother_diag cap<s>/<d>; lead faces keep Sigma).
    Cost options: spacing [Å] = largest distance between neighbouring centres (default: one centre between interfaces,
    i.e. the slab middles); solver dense (complex128 LU) | dense64 (complex64 LU, solves in single precision) |
    pardiso (sparse, real-equivalent, msbasis.CSolver without refinement) | rows / rows64 (only the core rows of each
    window inverse, complex128 / complex64: n_core x n_window per window instead of the n_window^2 LU, one GEMM per
    window and pass; the same correction, since RAS keeps nothing else). GPU mode (gpu.on, every solver but pardiso):
    factors and solves on the device, r and the result are cupy arrays (twin of the release CapSchwarz).
    dE0 = factor at E + dE0 instead of E (one
    factorization reused over an energy interval: A_jj(E + dE0) = A_jj(E) + dE0 S_jj, Sigma kept);
    centres "iface": windows on the slab interfaces only, no correction outside them (the slab interiors are left to
    the coarse level); "ends": interfaces + the two windows on the lead faces.
    split = pieces per periodic transverse axis (meta["periodic"] order): every x-window is cut into boxes of
    period / n (core, disjoint) + tov Å on each side (minimum image), with the same absorbing ramp towards the
    transverse cuts.
    centres "tile": one window per tile of mb.partition(tile_orbs, xsplit) instead of x-windows: core = the tile's
    orbitals, window = every device orbital within tov Å (3D, minimum image) of a tile atom, absorbing ramp over the
    outer d Å of that shell (the lead faces are real boundaries and keep Sigma)."""

    def __init__(self, dev, inp, width=16.0, s=2.0, d=3.0, spacing=None, solver="dense", dE0=0.0, centres="all",
                 workers=1, split=None, tov=2.5, tile_orbs=2500, xsplit=1, tile_cut="", stream_gb=None):
        ax = inp["meta"]["axis"]
        self.stream_gb, self.st = stream_gb, None     # rows solvers on the GPU: core rows in pinned host memory, streamed
        atom = np.repeat(np.arange(len(inp["norb"])), inp["norb"])
        self.x = x = inp["xyz"][atom[dev.dev], ax]
        if centres == "tile":
            self._tiles(dev, inp, atom[dev.dev], tov, tile_orbs, xsplit, ax, tile_cut)
            width = f"tile {tile_orbs}/x{xsplit} +{tov}"
            split = None
        else:
            self._xwin(dev, inp, atom, x, width, spacing, centres, split, tov)
        cov = np.zeros(len(x), int)
        for i, c in zip(self.idx, self.core):
            cov[i[c]] += 1
        assert (cov == 1).all(), "window cores must partition the device"
        assert solver in ("dense", "dense64", "pardiso", "rows", "rows64"), solver
        self.s, self.d, self.solver, self.dE0, self.dev = s, d, solver, dE0, dev
        self.rows = solver.startswith("rows")
        self.gpu = gpu.on and solver != "pardiso"          # GPU: window LUs / rows and solves on the device, serially
        self.ops, self.twall, self.rows_gb, self._ramps = None, 0.0, 0.0, {}
        self.own = list(range(len(self.idx)))                  # windows of this rank (all; distribute() changes it)
        self.comm = self.part = self.ex = None
        if self.gpu:
            workers = 1
            self._device_index(lambda i: i)
        self.pool = ThreadPoolExecutor(workers) if workers > 1 else None   # windows are independent (additive)
        log(f"CapSchwarz: {len(self.centres)} {centres} windows of {width} Å at x = {', '.join(f'{c:.1f}' for c in self.centres)}, "
            f"{min(map(len, self.idx))}..{max(map(len, self.idx))} orbitals (sum {sum(map(len, self.idx))}), "
            f"absorbing {s} eV over {d} Å, " + ("core rows of the window inverses" if self.rows else f"{solver} LU")
            + (" on the GPU" if self.gpu else "") + (f" at E {dE0:+g} eV" if dE0 else "")
            + (f", transverse split {split} (+{tov} Å)" if split and max(split) > 1 else ""))

    def _device_index(self, pos):
        """Device index arrays of the own windows: pos = global device orbital -> row of the (owned + halo) vector."""
        n = len(self.idx)
        self.didx, self.dcore, self.dout = [None] * n, [None] * n, [None] * n
        for k in self.own:
            i, c = self.idx[k], self.core[k]
            self.didx[k], self.dcore[k] = gpu.cp.asarray(pos(i)), gpu.cp.asarray(np.where(c)[0])
            self.dout[k] = gpu.cp.asarray(pos(i[c]))


    def _tiles(self, dev, inp, atom_d, tov, tile_orbs, xsplit, ax, tile_cut=""):
        blocks, _, _ = mb.partition(dev, cores=False, tile_orbs=tile_orbs, xsplit=xsplit, axis=ax,
                                    cut=mb.parse_cut(tile_cut))
        geo = mb.Geometry(inp["xyz"], inp["meta"].get("periodic"))
        dat = np.unique(atom_d)
        tree_d = geo.tree(dat)
        self.idx, self.core, self.dist = [], [], []
        for b in blocks:
            near = dat[np.unique(np.concatenate([np.asarray(q, int) for q in
                                                 tree_d.query_ball_point(geo.points(b["atoms"]), tov + 1e-9)]))]
            wa = np.union1d(near, b["atoms"])
            i = np.where(np.isin(atom_d, wa))[0]
            dt = geo.tree(b["atoms"]).query(geo.points(atom_d[i]))[0]
            self.idx.append(i)
            self.core.append(np.isin(atom_d[i], b["atoms"]))
            self.dist.append((dt, tov))
        self.centres = np.array([np.mean(self.x[i[c]]) for i, c in zip(self.idx, self.core)])
        self.tcut = [[] for _ in self.idx]

    def _xwin(self, dev, inp, atom, x, width, spacing, centres, split, tov):
        self.dist = None
        sl = [np.arange(dev.sb[k], dev.sb[k + 1]) for k in range(len(dev.ns))]
        xi = [0.5 * (x[sl[k]].max() + x[sl[k + 1]].min()) for k in range(len(sl) - 1)]
        anchors = [x.min()] + xi + [x.max()]
        fill = []
        for lo, hi in zip(anchors[:-1], anchors[1:]):
            m = max(1, int(np.ceil((hi - lo) / spacing)) - 1) if spacing else 1
            fill += [lo + (hi - lo) * (k + 1) / (m + 1) for k in range(m)]
        keep = {"all": fill, "iface": [], "ends": [fill[0], fill[-1]]}[centres]
        self.centres = np.sort(np.array(xi + keep))
        near = np.argmin(np.abs(x[:, None] - self.centres[None, :]), axis=1)
        self.idx = [np.where((np.abs(x - c) < width / 2) | ((near == k) & (centres == "all")))[0]
                    for k, c in enumerate(self.centres)]
        self.core = [near[i] == k for k, i in enumerate(self.idx)]
        self.tcut = [[] for _ in self.idx]                      # per window: (u, half width) of each transverse cut
        per = mb.periodic_axes(inp["meta"].get("periodic"))
        for (axt, L), n in zip(per, split or []):
            if n <= 1:
                continue
            u = inp["xyz"][atom[dev.dev], axt]
            box = np.minimum((u % L * (n / L)).astype(int), n - 1)   # owner box: one per orbital, also on a box face
            idx, core, tcut = [], [], []
            for i, c, tc in zip(self.idx, self.core, self.tcut):
                for j in range(n):
                    cj = (j + 0.5) * L / n
                    du = np.abs((u[i] - cj + L / 2) % L - L / 2)       # minimum-image distance to the box centre
                    w = du < L / (2 * n) + tov
                    idx.append(i[w])
                    core.append(c[w] & (box[i[w]] == j))
                    tcut.append([(dv[w], h) for dv, h in tc] + [(du[w], L / (2 * n) + tov)])
            self.idx, self.core, self.tcut = idx, core, tcut

    def ramp(self, k):
        """Absorbing ramp (0 .. 1) of window k towards its artificial cuts (the lead faces are real boundaries)."""
        x, idx = self.x, self.idx[k]
        xmin, xmax = x.min(), x.max()
        xi = x[idx]
        lo, hi = xi.min(), xi.max()
        ramp = np.zeros(len(idx))
        if self.dist is not None:                              # tile window: ramp over the outer d of the shell
            dt, h = self.dist[k]
            ramp = np.clip((dt - (h - self.d)) / self.d, 0, 1) ** 2
        elif lo > xmin + 1e-6:                                 # a cut on the left (not the lead face)
            ramp = np.maximum(ramp, np.clip((lo + self.d - xi) / self.d, 0, 1) ** 2)
        if self.dist is None and hi < xmax - 1e-6:
            ramp = np.maximum(ramp, np.clip((xi - (hi - self.d)) / self.d, 0, 1) ** 2)
        for du, h in self.tcut[k]:                             # transverse cuts at |du| = h
            ramp = np.maximum(ramp, np.clip((du - (h - self.d)) / self.d, 0, 1) ** 2)
        return ramp

    def set_energy(self, E, A, delta=None, log=log):
        S = self.dev.S.tocsr()
        window = A.window if hasattr(A, "window") else (lambda A: lambda idx: A[idx][:, idx])(A.tocsr())   # dist.FineA
        # distributed, GPU: the dense windows are filled on the device from the rows of the windows at one x position
        asm = self.gpu and hasattr(A, "segment")
        seg = [None, None]

        def group(k):
            return k // max(1, len(self.idx) // len(self.centres))

        def dense_window(k, dt):
            if seg[0] != group(k):
                seg[1] = None
                ks_ = [q for q in self.own if group(q) == group(k)]
                seg[:] = group(k), A.segment(int(min(self.idx[q].min() for q in ks_)), int(max(self.idx[q].max() for q in ks_)) + 1)
            if k not in self._ramps:
                self._ramps[k] = self.ramp(k)
            return A.fill(seg[1], self.idx[k], dt, self.dE0, self.s, self._ramps[k])

        def factor(k):
            idx = self.idx[k]
            t = time.time()
            if not asm:
                P = sp.diags(np.sqrt(self.ramp(k)))
                Sjj = S[idx][:, idx]
                Ajj = window(idx) + self.dE0 * Sjj + 1j * self.s * (P @ Sjj @ P)
            if self.solver == "pardiso":
                cs = mb.CSolver(Ajj, n_ref=0)
                op = cs.factor_solve
            else:
                dt = np.complex64 if self.solver in ("dense64", "rows64") else complex
                D = dense_window(k, dt) if asm else gpu.csr_dense(Ajj, dt) if self.gpu else Ajj.toarray().astype(dt)
                lu = gpu.lu_factor(D) if self.gpu else sla.lu_factor(D, check_finite=False, overwrite_a=True)
                del D
                if self.rows:                                      # (A_jj^-1)[core, :] = (A_jj^-H e_core)^H, LU dropped
                    xq = gpu.cp if self.gpu else np
                    c = np.where(self.core[k])[0]
                    Ek = xq.zeros((len(idx), len(c)), dt)
                    Ek[xq.asarray(c), xq.arange(len(c))] = 1
                    Ri = xq.ascontiguousarray((gpu.lu_solve(lu, Ek, trans=2) if self.gpu else
                                               sla.lu_solve(lu, Ek, trans=2, check_finite=False)).conj().T)
                    del lu, Ek
                    self.rows_gb += Ri.nbytes / 2 ** 30
                    if stream:                                     # to pinned host memory at once: never all on the GPU
                        return gpu.pinned(Ri), time.time() - t
                    op = (lambda Ri, dt: lambda rj: (Ri @ rj.astype(dt)).astype(complex))(Ri, dt)
                elif self.gpu and stream:                          # LU to pinned host memory at once
                    return gpu.lu_host(lu), time.time() - t
                elif self.gpu:
                    op = (lambda lu, dt: lambda rj: gpu.lu_solve(lu, rj.astype(dt)).astype(complex))(lu, dt)
                else:
                    op = (lambda lu, dt: lambda rj: sla.lu_solve(lu, rj.astype(dt), check_finite=False).astype(complex))(lu, dt)
            return op, time.time() - t
        t = time.time()
        ks = self.own
        rows_est = sum((int(self.core[k].sum()) if self.rows else len(self.idx[k])) * len(self.idx[k]) for k in ks) \
            * (8 if self.solver in ("rows64", "dense64") else 16)
        stream = bool(self.gpu and self.stream_gb is not None and rows_est / 2 ** 30 > self.stream_gb)
        if self.st is not None:
            self.st.free()
        self.ops, self.st, self.rows_gb = None, None, 0.0          # drop the factors of the previous energy first
        if self.gpu:
            gpu.free()
        out = list(self.pool.map(factor, ks) if self.pool else map(factor, ks))
        seg[:] = None, None
        self.ops, self.tfac = {k: o for k, (o, _) in zip(ks, out)}, [t for _, t in out]
        if stream:
            self.st, self.ops = gpu.Streamer([self.ops[k] if isinstance(self.ops[k], tuple) else (self.ops[k],)
                                              for k in ks]), None
        self.twall = time.time() - t
        log(f"  factor {self.solver}: wall {self.twall:.1f} s, per window (" +
            ", ".join(f"{len(self.idx[k])}: {t:.1f}" for k, t in zip(ks, self.tfac)) + ")"
            + (f"; core rows {self.rows_gb:.1f} GB (LUs would be "
               f"{sum(len(self.idx[k]) ** 2 for k in ks) * (8 if self.solver == 'rows64' else 16) / 2**30:.1f} GB)"
               if self.rows else "") + (", streamed from pinned host memory" if stream else ""))
        self.ndefl = [0]

    def __call__(self, r):
        # r, z: the device vector, or with distribute() the owned rows; the windows read the owned + halo rows
        nloc = r.shape[0]
        if self.ex is not None:
            r = self.ex(gpu.cp.asarray(r) if self.gpu else r)
        if self.st is not None:                                   # streamed core rows: one GEMM per window
            r = gpu.cp.asarray(r)
            z = gpu.cp.zeros((nloc,) + r.shape[1:], r.dtype)
            dt = np.complex64 if self.solver in ("rows64", "dense64") else complex

            def fn(q, v):
                k = self.own[q]
                rk = r[self.didx[k]].astype(dt)
                z[self.dout[k]] = (v[0] @ rk if self.rows else gpu.lu_solve(gpu.lu_view(v), rk)[self.dcore[k]]).astype(complex)
            self.st.run(fn)
            return z
        if self.gpu:                                              # r: cupy (host arrays are moved), result cupy
            r = gpu.cp.asarray(r)
            z = gpu.cp.zeros((nloc,) + r.shape[1:], r.dtype)
            for k in self.own:
                zk = self.ops[k](r[self.didx[k]])
                z[self.dout[k]] = zk if self.rows else zk[self.dcore[k]]
            return z
        z = np.zeros((nloc,) + r.shape[1:], r.dtype)
        pos = (lambda i: i) if self.ex is None else (lambda i: self.ex.pos[i])
        solve = lambda k: (self.ops[k](r[pos(self.idx[k])]) if self.rows else
                           self.ops[k](r[pos(self.idx[k])])[self.core[k]])
        ks = self.own
        for k, zk in zip(ks, self.pool.map(solve, ks) if self.pool else map(solve, ks)):
            z[pos(self.idx[k][self.core[k]])] = zk             # cores are disjoint
        return z

    def free(self):
        self.ops = None
        if self.st is not None:
            self.st.free()
            self.st = None
        gpu.free()


class GpuOp:
    """A @ X on the GPU: the sparse device matrix as a cupyx CSR (complex128); X numpy or cupy, result cupy."""

    def __init__(self, A):
        self.A = gpu.csr(A, np.complex128)
        self.shape = self.A.shape

    def __matmul__(self, X):
        return self.A @ gpu.cp.asarray(X, dtype=complex)


class StreamOp:
    """A @ X on the GPU with A in pinned host memory: row batches of `batch_gb` (CSR data / indices / indptr), streamed
    with the copy of batch k + 1 overlapped with the SpMM of batch k (gpu.Streamer). Device memory: 2 batches."""

    def __init__(self, A, batch_gb=1.0):
        A = sp.csr_matrix(A).astype(np.complex128)
        self.shape, n = A.shape, A.shape[0]
        per_row = np.diff(A.indptr) * 20.0                        # bytes per row: complex128 data + int32 index
        cum = np.concatenate([[0], np.cumsum(per_row)])
        cuts = np.unique(np.concatenate([np.searchsorted(cum, np.arange(0, cum[-1], batch_gb * 2 ** 30)), [n]]))
        cuts = cuts[cuts > 0] if cuts[0] == 0 else cuts
        self.rows = list(zip(np.concatenate([[0], cuts[:-1]]), cuts))
        items = []
        for a, b in self.rows:
            B = A[a:b]
            items.append((gpu.pinned(B.data), gpu.pinned(B.indices.astype(np.int32)),
                          gpu.pinned(B.indptr.astype(np.int32))))
        self.st = gpu.Streamer(items)
        self.gb = self.st.nbytes / 2 ** 30

    def __matmul__(self, X):
        import cupyx.scipy.sparse as csp
        cp = gpu.cp
        X = cp.asarray(X, dtype=complex)
        one = X.ndim == 1
        X2 = X[:, None] if one else X
        Y = cp.empty((self.shape[0], X2.shape[1]), complex)

        def fn(k, v):
            a, b = self.rows[k]
            M = csp.csr_matrix((v[0], v[1], v[2]), shape=(b - a, self.shape[1]), copy=False)
            Y[a:b] = M @ X2
        self.st.run(fn)
        return Y[:, 0] if one else Y


def operator(A, threads=4, stream_gb=None):
    """FastOp on the CPU, GpuOp in GPU mode (twin of the release schwarz.operator); StreamOp when the CSR (complex128
    data + int32 indices) is larger than stream_gb GB."""
    if gpu.on and stream_gb is not None and sp.csr_matrix(A).nnz * 20 / 2 ** 30 > stream_gb:
        return StreamOp(A)
    return GpuOp(A) if gpu.on else FastOp(A, threads)


class FastOp:
    """A @ X for a sparse complex A and a block X, faster than scipy's complex CSR product: A = Ar + i Ai with real
    Ar, Ai (Ai holds only the complex entries, e.g. the lead self-energy blocks), real products on [Re X, Im X], rows
    split over `threads` threads (scipy releases the GIL). MoS2/HfO2, 32M nonzeros, 64 columns: 2.8 s -> 0.3 s (8)."""

    def __init__(self, A, threads=4):
        A = sp.csr_matrix(A)
        self.shape, self.n = A.shape, A.shape[0]             # rectangular: local rows x (owned + ghost) columns
        Ar = sp.csr_matrix(A.real, copy=True)            # copies: A.real / A.imag share A's index arrays, which
        Ai = (sp.csr_matrix(A.imag, copy=True) if np.iscomplexobj(A.data)   # eliminate_zeros changes in place
              else sp.csr_matrix(A.shape))
        Ar.eliminate_zeros(); Ai.eliminate_zeros()
        k = max(1, int(threads))
        rows = np.linspace(0, self.n, k + 1).astype(int)
        self.parts = [(rows[i], rows[i + 1], Ar[rows[i]:rows[i + 1]], Ai[rows[i]:rows[i + 1]] if Ai.nnz else None)
                      for i in range(k)]
        self.pool = ThreadPoolExecutor(k) if k > 1 else None

    def __matmul__(self, X):
        X = np.asarray(X)
        one = X.ndim == 1
        X = X.reshape(self.shape[1], -1)
        nc = X.shape[1]
        Xs = np.hstack([X.real, X.imag]) if np.iscomplexobj(X) else np.hstack([X, np.zeros_like(X)])
        Y = np.empty((self.n, nc), complex)

        def part(p):
            a, b, Ar, Ai = p
            P = Ar @ Xs
            yr, yi = P[:, :nc], P[:, nc:]
            if Ai is not None:
                Q = Ai @ Xs
                yr, yi = yr - Q[:, nc:], yi + Q[:, :nc]
            Y[a:b].real, Y[a:b].imag = yr, yi
        list(self.pool.map(part, self.parts)) if self.pool else [part(p) for p in self.parts]
        return Y[:, 0] if one else Y


def fgmres_block(A, b, x0, prec, tol=1e-8, m=60, clock=None, cb=None, comm=None):
    """Flexible GMRES for all columns of b at once without diagnostics: every column has its own Krylov space and
    Hessenberg matrix, but A and the preconditioner act on the block of the active columns' newest basis vectors
    (one multi-RHS local solve per window). A column stops when its GMRES residual estimate is below tol |b_j|.
    Returns x, iterations per column, true relative residuals; clock[key] accumulates seconds (matvec, prec, orth).
    cb(i, est): called after iteration i with the GMRES residual estimates |r_j| / |b_j| of all columns.
    Storage per column (V[j], Z[j]: (m + 1) x n, rows = basis vectors): the Gram-Schmidt step is one GEMV pair per
    column on contiguous memory. b, x0, A @ and prec: numpy or cupy arrays (Hessenberg and least squares on the host);
    x: an array of b's kind (twin of the release fgmres_block).
    comm (dist.Comm): b, x0, x are the owned rows of this rank; the dot products of all active columns are summed over
    the ranks in one message per Gram-Schmidt pass (same bits on every rank, so all ranks take the same branches)."""
    if comm is not None and comm.size > 1:
        return _fgmres_dist(A, b, x0, prec, tol, m, {} if clock is None else clock, cb, comm)
    clock = {} if clock is None else clock
    def tick(key, t0):
        clock[key] = clock.get(key, 0.0) + time.time() - t0
    xp = gpu.xp_of(b)
    n, nc = b.shape
    t0 = time.time(); r0 = b - A @ x0; tick("matvec", t0)
    beta, bn = gpu.asnumpy(xp.linalg.norm(r0, axis=0)), gpu.asnumpy(xp.linalg.norm(b, axis=0))
    V = xp.zeros((nc, m + 1, n), complex)
    Z = xp.zeros((nc, m, n), complex)
    Hh = np.zeros((nc, m + 1, m), complex)
    V[:, 0] = (r0 / xp.asarray(np.where(beta > 0, beta, 1))).T
    its = np.zeros(nc, int)
    est = beta / bn
    act = np.where(beta > tol * bn)[0]
    Y = [np.zeros(0, complex) for _ in range(nc)]
    for i in range(m):
        if len(act) == 0:
            break
        t0 = time.time(); zb = xp.asarray(prec(xp.ascontiguousarray(V[act, i].T))); tick("prec", t0)
        Z[act, i] = zb.T
        t0 = time.time(); w = xp.asarray(A @ zb); tick("matvec", t0)
        t0 = time.time()
        keep = []
        for jj, j in enumerate(act):
            wj = w[:, jj].copy()
            Vj = V[j, :i + 1]
            for _ in range(2):                                  # classical Gram-Schmidt, twice
                h = Vj.conj() @ wj
                wj -= h @ Vj
                Hh[j, :i + 1, i] += gpu.asnumpy(h)
            hn = float(xp.linalg.norm(wj))
            Hh[j, i + 1, i] = hn
            if hn > 0:
                V[j, i + 1] = wj / hn
            e1 = np.zeros(i + 2, complex)
            e1[0] = beta[j]
            Y[j] = np.linalg.lstsq(Hh[j, :i + 2, :i + 1], e1, rcond=None)[0]
            its[j] = i + 1
            est[j] = np.linalg.norm(Hh[j, :i + 2, :i + 1] @ Y[j] - e1) / bn[j]
            if est[j] > tol and hn > 0:
                keep.append(j)
        act = np.array(keep, int)
        tick("orth", t0)
        if cb is not None:
            cb(i, est)
    x = x0.copy()
    for j in range(nc):
        if its[j]:
            x[:, j] += xp.asarray(Y[j]) @ Z[j, :its[j]]
    del V, Z
    t0 = time.time(); res = gpu.asnumpy(xp.linalg.norm(b - A @ x, axis=0)) / bn; tick("matvec", t0)
    return x, its, res


def _fgmres_dist(A, b, x0, prec, tol, m, clock, cb, comm):
    """fgmres_block on distributed vectors."""
    def tick(key, t0):
        clock[key] = clock.get(key, 0.0) + time.time() - t0
    xp = gpu.xp_of(b)
    n, nc = b.shape
    norm = lambda X: np.sqrt(comm.sum(gpu.asnumpy((xp.abs(X) ** 2).sum(axis=0)))) if X.shape[1] else np.zeros(0)
    t0 = time.time(); r0 = b - A @ x0; tick("matvec", t0)
    beta, bn = norm(r0), norm(b)
    V = xp.zeros((nc, m + 1, n), complex)
    Z = xp.zeros((nc, m, n), complex)
    Hh = np.zeros((nc, m + 1, m), complex)
    V[:, 0] = (r0 / xp.asarray(np.where(beta > 0, beta, 1))).T
    its = np.zeros(nc, int)
    est = beta / bn
    act = np.where(beta > tol * bn)[0]
    Y = [np.zeros(0, complex) for _ in range(nc)]
    for i in range(m):
        if len(act) == 0:
            break
        t0 = time.time(); zb = xp.asarray(prec(xp.ascontiguousarray(V[act, i].T))); tick("prec", t0)
        Z[act, i] = zb.T
        t0 = time.time(); w = xp.ascontiguousarray(xp.asarray(A @ zb).T); tick("matvec", t0)    # rows = columns of A zb
        t0 = time.time()
        for _ in range(2):                                      # classical Gram-Schmidt, twice
            h = xp.stack([V[j, :i + 1].conj() @ w[jj] for jj, j in enumerate(act)])
            h = comm.sum(gpu.asnumpy(h))
            for jj, j in enumerate(act):
                w[jj] -= xp.asarray(h[jj]) @ V[j, :i + 1]
                Hh[j, :i + 1, i] += h[jj]
        hn = norm(w.T)
        keep = []
        for jj, j in enumerate(act):
            Hh[j, i + 1, i] = hn[jj]
            if hn[jj] > 0:
                V[j, i + 1] = w[jj] / hn[jj]
            e1 = np.zeros(i + 2, complex)
            e1[0] = beta[j]
            Y[j] = np.linalg.lstsq(Hh[j, :i + 2, :i + 1], e1, rcond=None)[0]
            its[j] = i + 1
            est[j] = np.linalg.norm(Hh[j, :i + 2, :i + 1] @ Y[j] - e1) / bn[j]
            if est[j] > tol and hn[jj] > 0:
                keep.append(j)
        act = np.array(keep, int)
        tick("orth", t0)
        if cb is not None:
            cb(i, est)
    x = x0.copy()
    for j in range(nc):
        if its[j]:
            x[:, j] += xp.asarray(Y[j]) @ Z[j, :its[j]]
    del V, Z
    t0 = time.time(); res = norm(b - A @ x) / bn; tick("matvec", t0)
    return x, its, res

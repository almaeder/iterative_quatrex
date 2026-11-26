"""Communication of the multi-GPU solver: one MPI rank per GPU.

    Comm        one rank (the single-GPU solver): every collective is the identity
    ThreadComm  ranks as threads of one process (tests)
    MPIComm     mpi4py; with NCCL (bound by ctypes, libnccl.so.2 of nvidia-nccl-cu13 or $QTX_NCCL) GPU arrays go GPU to GPU,
                otherwise they are staged through the host. Sums are reduce + broadcast: the same bits on every rank, so every
                rank takes the same branches (FGMRES).
    Part        the rows of a distributed vector: the entries of the blocks a rank owns, in ascending global order
    Halo        x -> the owned rows of x followed by the rows of other ranks this rank reads (its halo); rev() sends rows
                computed here for blocks of other ranks back to their owners
    chunks      contiguous ranges of items of about equal cost, one per rank
    domains     slabs of the bottom level -> interior domains separated by single slabs (distributed block-tridiagonal LU)
"""
import ctypes
import os
import time

import numpy as np

from .backend import gpu, on_gpu, xp


# ===================================================================================================================== NCCL
class Nccl:
    """Point to point and sum on cupy arrays, on the current stream (bytes for p2p; reals of the precision for sums)."""
    INT8, F32, F64, SUM = 0, 7, 8, 0

    def __init__(self, mpi):
        self.lib = lib = self._load()
        uid = (ctypes.c_char * 128)()
        if mpi.Get_rank() == 0:
            self._check(lib.ncclGetUniqueId(ctypes.byref(uid)))
        raw = mpi.bcast(bytes(uid), root=0)

        class UniqueId(ctypes.Structure):                             # passed by value
            _fields_ = [("b", ctypes.c_char * 128)]
        u = UniqueId()
        ctypes.memmove(ctypes.byref(u), raw, 128)
        self.handle = ctypes.c_void_p()
        lib.ncclCommInitRank.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, UniqueId, ctypes.c_int]
        self._check(lib.ncclCommInitRank(ctypes.byref(self.handle), mpi.Get_size(), u, mpi.Get_rank()))
        v, sz, i = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
        lib.ncclSend.argtypes = lib.ncclRecv.argtypes = [v, sz, i, i, v, v]
        lib.ncclAllReduce.argtypes = [v, v, sz, i, i, v, v]

    @staticmethod
    def _load():
        cand = [os.environ.get("QTX_NCCL")]
        try:
            import nvidia
            cand += [os.path.join(p, "nccl", "lib", "libnccl.so.2") for p in nvidia.__path__]
        except ImportError:
            pass
        for c in [x for x in cand if x and os.path.exists(x)] + ["libnccl.so.2"]:
            try:
                return ctypes.CDLL(c)
            except OSError:
                continue
        raise RuntimeError("libnccl.so.2 not found (pip install nvidia-nccl-cu13, or set $QTX_NCCL); --dist mpi needs none")

    def _check(self, rc):
        if rc != 0:
            self.lib.ncclGetErrorString.restype = ctypes.c_char_p
            raise RuntimeError(f"NCCL: {self.lib.ncclGetErrorString(rc).decode()}")

    def exchange(self, sends, recvs):
        stream = xp.cuda.get_current_stream().ptr
        self._check(self.lib.ncclGroupStart())
        try:
            for p, x in recvs.items():
                self._check(self.lib.ncclRecv(x.data.ptr, x.nbytes, self.INT8, p, self.handle, stream))
            for p, x in sends.items():
                self._check(self.lib.ncclSend(x.data.ptr, x.nbytes, self.INT8, p, self.handle, stream))
        finally:
            rc = self.lib.ncclGroupEnd()
        self._check(rc)

    def sum(self, x):
        reals = x.size * (2 if x.dtype.kind == "c" else 1)
        kind = {4: self.F32, 8: self.F64}[x.dtype.itemsize // (2 if x.dtype.kind == "c" else 1)]
        y = xp.empty_like(x)
        self._check(self.lib.ncclAllReduce(x.data.ptr, y.data.ptr, reals, kind, self.SUM, self.handle,
                                           xp.cuda.get_current_stream().ptr))
        return y


# ===================================================================================================================== communicators
class Comm:
    """One rank. Also the base class: allgather of python objects defines the other collectives."""
    rank, size, local_rank, nccl = 0, 1, 0, None

    def allgather(self, x):
        return [x]

    def barrier(self):
        self.allgather(None)

    def bcast(self, x, root=0):
        return self.allgather(x if self.rank == root else None)[root]

    def alltoall(self, xs):
        return [row[self.rank] for row in self.allgather(xs)]

    def sum(self, x):
        """Sum over the ranks (numpy, cupy or scalar), the same bits on every rank."""
        if self.size == 1:
            return x
        if on_gpu() and isinstance(x, xp.ndarray):
            if self.nccl is not None:
                return self.nccl.sum(xp.ascontiguousarray(x))
            return xp.asarray(self._sum(gpu.asnumpy(x)))
        return self._sum(np.asarray(x))

    def _sum(self, x):
        out = None
        for y in self.allgather(x):
            out = y.copy() if out is None else out + y
        return out

    def merge(self, d):
        """Union of the dictionaries of all ranks (host arrays), on every rank."""
        if self.size == 1:
            return dict(d)
        out = {}
        for x in self.allgather(d):
            out.update(x)
        return out

    def exchange(self, sends, recvs):
        """Point to point: sends {rank: array}, recvs {rank: array to fill} (C-contiguous)."""
        on_dev = on_gpu() and any(isinstance(x, xp.ndarray) for x in [*sends.values(), *recvs.values()])
        if on_dev and self.nccl is not None:
            return self.nccl.exchange(sends, recvs)
        if on_dev:
            host = {p: np.empty(x.shape, x.dtype) for p, x in recvs.items()}
            self._exchange({p: gpu.asnumpy(x) for p, x in sends.items()}, host)
            for p, x in recvs.items():
                x[...] = xp.asarray(host[p])
            return
        self._exchange(sends, recvs)

    def _exchange(self, sends, recvs):
        for p, row in enumerate(self.allgather(sends)):
            if self.rank in row and p != self.rank:
                recvs[p][...] = row[self.rank]


class MPIComm(Comm):
    def __init__(self, nccl=True):
        from mpi4py import MPI
        self.MPI, self.c = MPI, MPI.COMM_WORLD
        self.rank, self.size = self.c.Get_rank(), self.c.Get_size()
        local = self.c.Split_type(MPI.COMM_TYPE_SHARED)
        self.local_rank, self.local_size = local.Get_rank(), local.Get_size()
        self.want_nccl = nccl

    def start_nccl(self):
        """After the GPU of this rank is selected."""
        if self.want_nccl and self.size > 1:
            self.nccl = Nccl(self.c)

    def abort(self):
        self.c.Abort(1)

    def allgather(self, x):
        return self.c.allgather(x)

    def barrier(self):
        self.c.Barrier()

    def bcast(self, x, root=0):
        return self.c.bcast(x, root)

    def alltoall(self, xs):
        return self.c.alltoall(xs)

    @staticmethod
    def _chunks(y, f, chunk=2 ** 30):
        """f on byte chunks of y (MPI counts are 32 bit)."""
        v = y.reshape(-1).view(np.uint8) if y.size else np.empty(0, np.uint8)
        for a in range(0, len(v), chunk):
            f(v[a:a + chunk])

    def _sum(self, x):
        nd, x = np.ndim(x), np.ascontiguousarray(x)
        y = np.empty_like(x)
        fx, fy = x.reshape(-1), y.reshape(-1)
        for a in range(0, max(fx.size, 1), 2 ** 26):
            self.c.Reduce(fx[a:a + 2 ** 26], fy[a:a + 2 ** 26] if self.rank == 0 else None, self.MPI.SUM, 0)
            self.c.Bcast(fy[a:a + 2 ** 26], 0)
        return y if nd else y[0]

    def _exchange(self, sends, recvs):
        rq = []
        for p, x in recvs.items():
            self._chunks(x, lambda v: rq.append(self.c.Irecv(v, p)))
        for p, x in sends.items():
            self._chunks(np.ascontiguousarray(x), lambda v: rq.append(self.c.Isend(v, p)))
        self.MPI.Request.Waitall(rq)


class ThreadComm(Comm):
    """Ranks as threads of one process, numpy only (the unit tests). ThreadComm.run(size, f) -> [f(comm) of every rank]."""

    def __init__(self, rank, size, shared):
        self.rank, self.size, self.shared = rank, size, shared

    def allgather(self, x):
        self.shared["slot"][self.rank] = x
        self.shared["barrier"].wait(300)
        out = list(self.shared["slot"])
        self.shared["barrier"].wait(300)
        return out

    @staticmethod
    def run(size, f):
        import threading
        shared = dict(slot=[None] * size, barrier=threading.Barrier(size))
        out, err = [None] * size, []

        def go(r):
            try:
                out[r] = f(ThreadComm(r, size, shared))
            except BaseException as e:                                # noqa: BLE001
                err.append(e)
                shared["barrier"].abort()
        old = threading.stack_size(256 << 20)                         # threaded LAPACK needs more than the default stack
        try:
            th = [threading.Thread(target=go, args=(r,)) for r in range(size)]
            for t in th:
                t.start()
            for t in th:
                t.join()
        finally:
            threading.stack_size(old)
        if err:
            raise err[0]
        return out


def launch(kind):
    """Communicator of a run: "" one rank, "mpi" (host staging) or "nccl"."""
    if not kind:
        return Comm()
    assert kind in ("mpi", "nccl"), kind
    return MPIComm(nccl=kind == "nccl")


def wait_file(f, seconds=60):
    """Wait until a file written by another rank is visible here (shared file systems cache directory lookups)."""
    t = time.time()
    while not os.path.exists(f):
        if time.time() - t > seconds:
            raise FileNotFoundError(f"{f}: written by another rank, not visible after {seconds} s")
        try:
            os.listdir(os.path.dirname(f) or ".")
        except OSError:
            pass
        time.sleep(0.5)


# ===================================================================================================================== ownership
def chunks(cost, size):
    """Owner of every item: contiguous chunks of about equal cost, every rank at least one item."""
    cost = np.asarray(cost, float)
    n = len(cost)
    assert n >= size, f"{size} ranks for {n} items"
    cum = np.concatenate([[0.0], np.cumsum(cost)])
    own, lo = np.zeros(n, int), 0
    for r in range(size):
        hi = n if r == size - 1 else int(np.searchsorted(cum, cum[-1] * (r + 1) / size - 1e-9 * cum[-1]))
        if r < size - 1 and hi > lo + 1 and cum[hi] - cum[-1] * (r + 1) / size > cum[-1] * (r + 1) / size - cum[hi - 1]:
            hi -= 1                                                   # the nearer of the two cuts
        hi = min(max(hi, lo + 1), n - (size - 1 - r))
        own[lo:hi] = r
        lo = hi
    return own


def domains(cost, size):
    """S slabs -> p = min(size, (S + 1) // 2) domains [a, b) separated by single slabs: (domains, separators, rank of each domain)."""
    cost = np.asarray(cost, float)
    S = len(cost)
    p = min(size, (S + 1) // 2)
    cum = np.concatenate([[0.0], np.cumsum(cost)])
    sep, lo = [], 1
    for j in range(p - 1):
        hi = S - 2 - 2 * (p - 2 - j)                                  # room for the domains behind it
        want = int(np.argmin(np.abs(cum[1:S + 1] - cum[-1] * (j + 1) / p - 0.5 * cost)))
        sep.append(min(max(want, lo), hi))
        lo = sep[-1] + 2
    e = [-1] + sep + [S]
    return [(e[d] + 1, e[d + 1]) for d in range(p)], sep, [int(round(v)) for v in np.linspace(0, size - 1, p)]


# ===================================================================================================================== distributed vectors
class Part:
    """Rows of a distributed vector of n entries: the entries blocks[b] belong to rank owner[b]. A rank holds its entries in
    ascending global order: loc[g] = row of entry g on its owner; idx = the entries of this rank."""

    def __init__(self, comm, blocks, owner, n):
        self.comm, self.n = comm, n
        self.owner_of = np.zeros(n, np.int32)                         # rank of every entry
        for b, i in enumerate(blocks):
            self.owner_of[i] = owner[b]
        self.loc = np.empty(n, np.int64)
        for r in range(comm.size):
            w = np.where(self.owner_of == r)[0]
            self.loc[w] = np.arange(len(w))
            if r == comm.rank:
                self.idx = w
        self.nloc = len(self.idx)

    def gather(self, r):
        """The whole vector from the owned rows, on every rank."""
        if self.comm.size == 1:
            return r
        g = xp.zeros((self.n,) + r.shape[1:], r.dtype)
        g[xp.asarray(self.idx)] = r
        return self.comm.sum(g)


class Halo:
    """Owned rows + the rows of the entries `need` held by other ranks. pos[g] = row of entry g in the extended vector
    (-1: not there). Collective at construction (all ranks)."""

    def __init__(self, part, need):
        self.part, comm = part, part.comm
        self.pos = np.full(part.n, -1, np.int64)
        self.pos[part.idx] = np.arange(part.nloc)
        self.nghost, self.sends, self.recvs = 0, {}, {}
        if comm.size == 1:
            return
        need = np.unique(np.asarray(need, np.int64))
        ghost = need[part.owner_of[need] != comm.rank]
        ghost = ghost[np.argsort(part.owner_of[ghost], kind="stable")]
        self.nghost = len(ghost)
        self.pos[ghost] = part.nloc + np.arange(len(ghost))
        want = [None] * comm.size
        for r in range(comm.size):
            w = np.where(part.owner_of[ghost] == r)[0]
            want[r] = part.loc[ghost[w]]
            if len(w):
                self.recvs[r] = (part.nloc + int(w[0]), part.nloc + int(w[-1]) + 1)
        for r, rows in enumerate(comm.alltoall(want)):
            if r != comm.rank and len(rows):
                self.sends[r] = xp.asarray(rows)

    def __call__(self, r):
        if self.part.comm.size == 1:
            return r
        x = xp.empty((self.part.nloc + self.nghost,) + r.shape[1:], r.dtype)
        x[:self.part.nloc] = r
        self.part.comm.exchange({p: xp.ascontiguousarray(r[i]) for p, i in self.sends.items()},
                                {p: x[a:b] for p, (a, b) in self.recvs.items()})
        return x

    def rev(self, zx):
        """Owned rows of zx, with the ghost rows of zx (computed here) written to their owners."""
        z = zx[:self.part.nloc]
        if self.part.comm.size == 1:
            return z
        got = {p: xp.empty((len(i),) + z.shape[1:], z.dtype) for p, i in self.sends.items()}
        self.part.comm.exchange({p: xp.ascontiguousarray(zx[a:b]) for p, (a, b) in self.recvs.items()}, got)
        for p, i in self.sends.items():
            z[i] = got[p]
        return z


def swap(comm, have, wants, source, shape, dtype):
    """Move keyed arrays between ranks. wants[r]: the keys rank r wants; source(key): the rank that has it (in `have`);
    shape(key): its shape. One message per pair of ranks; returns {key: array} received here. Collective."""
    me, out = comm.rank, {}
    if comm.size == 1:
        return out
    size = lambda q: int(np.prod(shape(q)))
    send = {r: sorted(q for q in wants[r] if source(q) == me) for r in range(comm.size) if r != me}
    recv = {r: sorted(q for q in wants[me] if source(q) == r) for r in range(comm.size) if r != me}
    sb = {r: xp.concatenate([xp.ascontiguousarray(have[q]).ravel() for q in qs]) for r, qs in send.items() if sum(map(size, qs))}
    rb = {r: xp.empty(sum(map(size, qs)), dtype) for r, qs in recv.items() if sum(map(size, qs))}
    comm.exchange(sb, rb)
    for r, qs in recv.items():
        o = 0
        for q in qs:
            out[q] = rb[r][o:o + size(q)].reshape(shape(q)).copy()
            o += size(q)
    return out

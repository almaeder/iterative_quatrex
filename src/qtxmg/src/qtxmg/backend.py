"""Array backend of qtxmg: `xp` is numpy until use_gpu() switches it to cupy, so every module writes xp.zeros, xp.linalg.inv, ...
once for both. Also: dense LU helpers, host <-> device transfers and a dictionary of blocks kept on the host and uploaded on use."""
import numpy as np
import scipy.linalg as sla

try:                                          # release layout: qtxmg/gpu.py
    from . import gpu
except ImportError:                           # development layout: qtx_ms/qtxmg/
    from .. import gpu

C8 = np.complex64


class _ArrayModule:
    """numpy or cupy, chosen at run time (attribute access is forwarded to the module in use)."""
    module = np

    def __getattr__(self, name):
        return getattr(self.module, name)


xp = _ArrayModule()


_quiet = [False]


def log(*a):
    """Print (rank 0 only, see quiet)."""
    if not _quiet[0]:
        print(*a, flush=True)


def quiet(flag=True):
    """Silence log() on this rank (every rank but 0)."""
    _quiet[0] = flag


def use_gpu(device=None):
    """cupy for every dense part; the Beyn lead (gpu module) follows. device: GPU index (one per rank)."""
    gpu.enable(device=device)
    xp.module = gpu.cp


def on_gpu():
    return xp.module is not np


def to_host(a):
    return a.get() if on_gpu() and hasattr(a, "get") else np.asarray(a)


def sync():
    if on_gpu():
        xp.cuda.Device().synchronize()


def free():
    if on_gpu():
        xp.get_default_memory_pool().free_all_blocks()


def gpu_name():
    return xp.cuda.runtime.getDeviceProperties(xp.cuda.Device().id)["name"].decode() if on_gpu() else "CPU"


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


def real_view(x):
    """A C-ordered complex block n x c as a real n x 2c one (re / im interleaved): real matrix @ complex block without a complex
    copy of the matrix. complex_view undoes it."""
    x = xp.ascontiguousarray(x)
    return x.view(x.real.dtype)


def complex_view(y):
    y = xp.ascontiguousarray(y)
    return y.view({4: np.complex64, 8: np.complex128}[y.dtype.itemsize])


def real_matmul(R, x):
    """R @ x for a real R and a complex x (in the precision of R)."""
    return complex_view(R @ real_view(x.astype({4: np.complex64, 8: np.complex128}[R.dtype.itemsize])))


def offsets(sizes):
    return np.concatenate([[0], np.cumsum(sizes)]).astype(int)


class Blocks:
    """Blocks by key, kept on the host (numpy) and uploaded to the array device on first use: every rank can hold all
    bases and Galerkin blocks of the hierarchy while its GPU holds only those it works with."""

    def __init__(self, items=None):
        self.host, self.dev = {}, {}
        for k, v in (items or {}).items():
            self[k] = v

    def __setitem__(self, key, value):
        self.host[key] = to_host(value)
        self.dev.pop(key, None)

    def __getitem__(self, key):
        if key not in self.dev:
            self.dev[key] = xp.asarray(self.host[key])
        return self.dev[key]

    def __contains__(self, key):
        return key in self.host

    def __len__(self):
        return len(self.host)

    def keys(self):
        return self.host.keys()

    def update(self, d):
        for k, v in d.items():
            self[k] = v

    def nbytes(self, on_device=False):
        return sum(v.nbytes for v in (self.dev if on_device else self.host).values())

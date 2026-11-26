"""qtxmg on one or several GPUs: transmission T(E) of an open device by FGMRES preconditioned with the V-cycle of a
density-matrix mode-space hierarchy (hierarchy.py). One MPI rank per GPU (--dist nccl or mpi); without --dist, one rank.

    srun -n 4 python -m qtxmg.run --inputs DIR --dE 0.05,0.15 --up y:1e-5,x:1e-5 --cap-split 2,1 --gpu --dist nccl --cache DIR

Every rank reads the input set; it holds the rows of its own blocks of every level, all bases and Galerkin blocks on the host
and the ones it uses on its GPU. Sigma and the injection are computed once for all points (dealt out to the ranks) and kept in
the cache folder, which must be on a file system all ranks see.
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

from .backend import gpu_name, log, on_gpu, quiet, to_host, use_gpu, xp
from .comm import launch
from .device import Device
from .hierarchy import Hierarchy
from .krylov import fgmres
from .obc import OBC
from .structure import contours

try:                                          # release layout: qtxmg/prep
    from .prep.inputs import read_inputs
except ImportError:                           # development layout: qtx_ms/qtxmg/
    from ..prep.inputs import read_inputs


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
    p.add_argument("--cache", default="", help="folder for Sigma and the bases (required with several ranks)")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--dist", default="", choices=["", "mpi", "nccl"], help="several ranks: MPI (host staging) or NCCL between the GPUs")
    p.add_argument("--out", default="")
    return p.parse_args(argv)


def solve_energy(h, device, B, tol, maxit, chunk):
    """Columns of the injection B (left face) -> T = sum_cols y^H Gamma_R y (y = the right-face rows), iterations, residuals."""
    comm, fine = h.comm, h.fine
    bL, bR = device.bL, device.bR
    left = np.where(fine.owner_of[bL] == comm.rank)[0]                # owned rows of the faces
    right = np.where(fine.owner_of[bR] == comm.rank)[0]
    nb, ch = B.shape[1], chunk or B.shape[1]
    ys, its, rs = [], [], []
    for c in range(0, nb, ch):
        m = min(ch, nb - c)
        b = xp.zeros((fine.nloc, m), complex)
        if len(left):
            b[xp.asarray(fine.loc[bL[left]])] = xp.asarray(B[left, c:c + m])
        x, i, r = fgmres(h.fine_matvec, b, h.precondition, tol, maxit, comm)
        y = xp.zeros((len(bR), m), complex)                           # the right face on every rank
        if len(right):
            y[xp.asarray(right)] = x[xp.asarray(fine.loc[bR[right]])]
        ys.append(comm.sum(y)), its.append(i), rs.append(r)
    y, its, rs = xp.concatenate(ys, axis=1), np.concatenate(its), np.concatenate(rs)
    SR = h.Sig[1]
    T = float(to_host(xp.real(xp.sum(y.conj() * (1j * ((SR - SR.conj().T) @ y))))))
    return T, its, rs


def main(argv=None):
    a = parse(argv)
    comm = launch(a.dist)
    t00 = time.time()
    quiet(comm.rank != 0)                                             # only rank 0 writes the log
    try:
        return _main(a, comm, t00)
    except Exception:                                                 # one failing rank stops the job (no hang in a collective)
        if comm.size > 1:
            traceback.print_exc()
            sys.stderr.flush()
            comm.abort()
        raise


def _main(a, comm, t00):
    if a.gpu:
        use_gpu(device=comm.local_rank % max(1, _gpu_count()))
    if hasattr(comm, "start_nccl"):
        comm.start_nccl()
    inp = read_inputs(a.inputs)
    device = Device(inp)
    inp.pop("H"), inp.pop("S")                                        # the device keeps its own rows
    Ec = inp["meta"]["edges"]["Ec_lead"] if a.Ec is None else a.Ec
    zs, dzs = contours(a.mu, a.npts, Ec)
    log(f"{os.path.basename(os.path.normpath(a.inputs))}: {device.n} orbitals, Ec {Ec:.4f} eV, faces {len(device.bL)} / {len(device.bR)}; "
        f"{comm.size} rank(s), {gpu_name()}" + (f", {a.dist}" if a.dist else "") + f"; contour {a.mu} ({len(zs)} points)")
    obc = OBC(device, a.obc, os.path.join(a.cache, "obc") if a.cache else None, a.inj_tol, comm)
    ups = [(v.split(":")[0], float(v.split(":")[1])) for v in a.up.split(",") if v]
    halos = [None if v == "all" else int(v) for v in a.halo.split(",") if v]
    h = Hierarchy(device, inp, obc, zs, dzs, comm, tuple(float(v) for v in a.cap.split(",")), a.cap_spacing,
                  [int(v) for v in a.cap_split.split(",")] if a.cap_split else None, a.cap_tov, a.cut, ups, halos, a.cache or None)
    reals = [Ec + float(v) + 1j * a.eta for v in a.dE.split(",")]
    t = time.time()
    obc.prepare([] if h.have_bases() else zs, reals)                # every Sigma and injection at once
    h.tick("sigma", t)
    h.setup()
    obc.drop(zs)
    log(f"setup {time.time() - t:.1f} s (" + ", ".join(f"{k} {v:.1f}" for k, v in h.t.items()) + f"); total {time.time() - t00:.0f} s")
    res = dict(Ec=Ec, args=vars(a), ranks=comm.size, N=[h.N[k] for k in range(1, h.K + 1)], t_setup=dict(h.t), E={})
    for z in reals:
        dE, t0, h.t = z.real - Ec, time.time(), {}
        Sig, B = obc(z), obc.inject(z)
        if B is None or B.shape[1] == 0:
            log(f"Ec {dE:+.3f}: T 0, no incoming channel")
            res["E"][f"{dE:.4f}"] = dict(E=z.real, T=0.0, its=[], res=[], converged=True, t={}, t_energy=time.time() - t0)
            continue
        h.set_energy(z.real, Sig)
        t = time.time()
        T, its, rs = solve_energy(h, device, B, a.tol, a.maxit, a.chunk)
        h.tick("solve", t)
        ok = bool(rs.max() < 10 * a.tol)
        log(f"Ec {dE:+.3f}: T {T:.10f}, {B.shape[1]} columns, its {its.min()}..{its.max()}, res {rs.max():.0e}"
            + ("" if ok else " ** NOT CONVERGED **") + f"; {time.time() - t0:.1f} s: " + ", ".join(f"{k} {v:.1f}" for k, v in h.t.items()))
        res["E"][f"{dE:.4f}"] = dict(E=z.real, T=T, its=its.tolist(), res=rs.tolist(), converged=ok, t=dict(h.t), t_energy=time.time() - t0)
        if a.out and comm.rank == 0:
            json.dump(res, open(a.out, "w"), indent=1)
    if on_gpu():
        gb = comm.allgather(xp.get_default_memory_pool().total_bytes() / 2 ** 30)
        log(f"GPU memory pool {min(gb):.1f}..{max(gb):.1f} GB per rank")
    log(f"done, {time.time() - t00:.0f} s")
    return res, h


def _gpu_count():
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount()
    except Exception:                                                 # noqa: BLE001
        return 1


if __name__ == "__main__":
    main()

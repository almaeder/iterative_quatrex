"""Density-matrix hierarchy with repeated merging: a multilevel preconditioner for the QTBM system A psi = b
(user, 2026-11-15; the library form of results_ms/gpu/mlevel/mlevel.py, on the qtmg stack: read_inputs,
msbasis.Device, schwarz.CapSchwarz / SlowCoarse / fgmres_block; numpy or cupy via qtmg.gpu).

Level 0 (fine): A = E S - H - Sigma, smoother S0 = CAP-Schwarz (schwarz.CapSchwarz, x windows, optional transverse
  split), RAS over the window cores. The cores are the level-1 blocks.
Level 1 (local rho, per core): modes of the contour density matrix (Ec + mu1 .. Ec + mu2, npts) of each S0 window
  operator A_kk(z) + i s P S P, restricted to its core. Selection "dm": eigenvectors of the window rho above the cut,
  restricted to the core (SVD); "dmc": eigenvectors of the core block of rho above the cut.
Level k + 1 (one `merge` per further level): blocks = pairs of level-k blocks, neighbours along x (transport) or y
  (the transverse strips). Per pair C: window w = C + the level-k blocks coupled to it (halo), local operator in the
  LEVEL-k BASIS z S_k - H_k - Sigma_k(z) + CAP_k (CAP and Sigma projected through all the bases below), contour rho
  over its own energy window (a narrower window = the energy slice of that level), modes above the cut on C (same
  selection as level 1) = U_{k+1}[C] in level-k coordinates. S_{k+1}, H_{k+1}, A_{k+1} = Galerkin, block-sparse over
  the pair graph.
  halo = H: the window keeps only the coupled level-k blocks within H blocks of the pair along x (0: the x positions
  of the pair only; None: every coupled block). The couplings to the dropped blocks are cut like any other artificial
  cut (projected CAP); the operators A_k stay the full Galerkin products. halo_smooth "full" keeps the full windows
  for the smoother of A_k (halo only for rho). Per level the log lists the coupling blocks of A_k, S_k by distance
  (2-norms) and the window size / rho cost of every halo.
Smoothers: the smoother of A_k is formed in the level-k basis with the level-(k+1) blocks: RAS over the windows w
  (A_k[w, w] + CAP_k), writing the rows of C (l1_solver "rows": only those rows of the window inverse are kept).
Cycle (preconditioner of flexible block GMRES on the fine system, x0 = 0): on every level one smoothing step, gamma
  coarse corrections (1: V, 2: W), one smoothing step. Bottom solver of the deepest level m used:
    exact   dense LU of A_m ("exact level m", for every m = 1 .. n: the loss of each level)
    top     A_n: RAS over its own blocks (block + coupled blocks, projected CAP) + slow modes of that cycle
            (schwarz.SlowCoarse) in rounds of --slow m,q, the first N of them (--counts)
  fine  : two-level baseline, slow modes directly on the fine smoother.

Basis reuse over energies: the block bases of every level are energy independent (contour rho over the window from
  S, H, the projected CAP and Sigma(z)); only A_k(E) = E S_k - H_k - Sigma_k(E) and the smoothers depend on E. merge(...,
  bkey=K) keeps U_{k+1} per key in memory (host) and, with h.basis_dir set, in <basis_dir>/l<k>_<hash>.npz; a later
  merge with the same key skips the contour rho (driver: --basis-cache, default <dm-cache>/<tag>_levels).

Library: h = Hierarchy(dev, inp, sw); DM = h.window_dm(...); U = h.basis1(DM, kind, cut); h.level1(U, E, SL, SR);
h.merge(axis, cut, kind, mu1, mu2, npts, halo, bkey) per further level; prec, timer = h.cycle(m, bottom, A0, S0, gamma).

    python -m qtmg.mlevel --inputs inputs/mos2_hfo2_9072a --dE 0.15 --gpu --sigma-cache DIR --psi-dir DIR \\
        --dm-cache DIR [--cap 16,2,2 --cap-split 2,1 --cap-tov 6 --cap-solver rows64] --l1 dm:1e-2 \\
        --up y:1e-2,x:1e-2:0.05:0.25 [--halo 1 --halo-smooth same] [--sweep fine,exact,top --counts 32,128,256]
"""
import os, sys
import numpy as np

from qtmg import dist, gpu, msbasis as mb

log = mb.log
quiet = lambda *x: None


def _xp():
    return gpu.cp if gpu.on else np


EIG_STATS = []          # per rand_top_eig call: (n, l, k, rounds, fallback); summarized and reset by merge


# ---------------------------------------------------------------------- driver


def start(kind, use_gpu, sweep=None):
    """Communicator and GPU backend of a driver (kind: --dist); the log of the ranks > 0 goes to <QTX_RANK_LOG>.<rank>."""
    comm = dist.launch(kind)
    dd = comm.size > 1
    if dd and comm.rank:
        f_ = os.environ.get("QTX_RANK_LOG")
        sys.stdout = open(f"{f_}.{comm.rank}", "w", buffering=1) if f_ else open(os.devnull, "w")
    if use_gpu:
        ndev = 0
        if dd:
            import cupy
            ndev = cupy.cuda.runtime.getDeviceCount()
        gpu.enable(device=comm.local_rank % ndev if dd else None)
    else:
        gpu.from_env()
    if dd:
        assert sweep in (None, "exact"), "--dist: --sweep exact (the slow modes of top / fine are not distributed)"
        if gpu.on:
            comm.start_nccl()
        log(f"domain decomposition: {comm.size} ranks ({comm.local_size} on this node), GPU halos through "
            f"{'NCCL' if comm.nccl is not None else 'MPI (host)'}")
    return comm


def guarded(f):
    """f() of a driver; on an error: the memory of the failed step, then MPI abort (the other ranks wait in a collective)."""
    try:
        f()
    except BaseException as e_:
        m_ = gpu.mem_mark()                                          # the step that failed: its peak before the error
        if m_:
            log("    mem | failed step: " + ", ".join(f"{k} {v / 2 ** 30:.2f}" for k, v in m_.items()) + " GB")
        if "mpi4py.MPI" in sys.modules and sys.modules["mpi4py.MPI"].COMM_WORLD.Get_size() > 1 and not (
                isinstance(e_, SystemExit) and not e_.code):
            import traceback
            traceback.print_exc()
            sys.stdout.flush(), sys.stderr.flush()
            sys.modules["mpi4py.MPI"].COMM_WORLD.Abort(1)
        raise

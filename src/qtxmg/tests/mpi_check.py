"""The multi-GPU solver under real MPI ranks against one rank, on the synthetic strip of the unit tests:
    mpiexec -n 3 python tests/mpi_check.py [cpu|mpi|nccl] [CACHE]
cpu: numpy and host MPI; mpi / nccl: one GPU per rank, vectors through host-staged MPI or NCCL. Every rank also runs the
one-rank solver itself (as a reference). Prints PASS on rank 0; exit code 1 otherwise."""
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                                    # noqa: E402

from conftest import strip                                            # noqa: E402
from qtxmg import backend, run                                        # noqa: E402
from qtxmg.comm import Comm, MPIComm                                  # noqa: E402
from qtxmg.prep.inputs import write_inputs                            # noqa: E402
from test_solver import ARGS                                          # noqa: E402

kind = sys.argv[1] if len(sys.argv) > 1 else "cpu"
comm = MPIComm(nccl=kind == "nccl")
assert comm.size > 1, "one rank only: the MPI library under mpi4py does not belong to this launcher"
if kind != "cpu":
    import cupy
    backend.use_gpu(device=comm.local_rank % cupy.cuda.runtime.getDeviceCount())
    comm.start_nccl()
root = comm.bcast(sys.argv[2] if len(sys.argv) > 2 else tempfile.mkdtemp(prefix="qtxmg_mpi_check_", dir=os.environ.get("SCRATCH")))
if comm.rank == 0:
    inp = strip()
    write_inputs(os.path.join(root, "dev"), inp["labels"], inp["xyz"], inp["norb"], inp["blocks"], inp["H"], inp["S"], inp["lead"], inp["meta"])
comm.barrier()
backend.quiet(True)
args = ["--inputs", os.path.join(root, "dev")] + ARGS + ["--obc", "folded"] + (["--gpu"] if kind != "cpu" else [])
ref, _ = run._main(run.parse(args + ["--cache", os.path.join(root, f"one{comm.rank}")]), Comm(), time.time())
ok, worst = True, 0.0
for up in ("y:1e-3,x:1e-3,x:1e-3", "y:1e-3,x:1e-3"):                  # deepest level: 4 or 8 slabs (separators on 3+ ranks)
    one = ref if up == "y:1e-3,x:1e-3,x:1e-3" else run._main(run.parse(args + ["--up", up, "--cache", os.path.join(root, f"one{comm.rank}_2")]), Comm(), time.time())[0]
    res, h = run._main(run.parse(args + ["--up", up, "--cache", os.path.join(root, f"many_{up}")]), comm, time.time())
    for k, v in one["E"].items():
        w = res["E"][k]
        d = abs(w["T"] - v["T"])
        worst = max(worst, d)
        ok &= w["converged"] and d < 1e-9 and abs(max(w["its"]) - max(v["its"])) <= 1 and res["N"] == one["N"]
    if comm.rank == 0:
        print(f"{kind}, {comm.size} ranks, --up {up}: N {res['N']}, bottom {h.bottom.desc}, "
              + ", ".join(f"T {w['T']:.10f} its {max(w['its'])}" for w in res["E"].values()), flush=True)
ok = all(comm.allgather(bool(ok)))
if comm.rank == 0:
    print(("PASS" if ok else "FAIL") + f": max |dT| against one rank {worst:.1e}", flush=True)
sys.exit(0 if ok else 1)

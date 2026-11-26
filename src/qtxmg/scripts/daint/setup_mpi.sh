#!/bin/bash
# MPI and NCCL for the multi-GPU solver (qtxmg-multi / python -m qtxmg.run --dist nccl), on top of setup.sh: mpi4py with the
# MPICH wheel (one node, mpiexec -launcher fork) and the NCCL library (nvidia-nccl-cu13, loaded by ctypes). Several nodes:
# multinode_env.sh switches mpi4py to cray-mpich of the uenv. Run on a compute node:
#   srun -A <account> -p debug -N1 -t 10 bash scripts/daint/setup_mpi.sh
set -euo pipefail
REL=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); cd "$REL"
export UV_CACHE_DIR=$REL/.tools/uv-cache UV_LINK_MODE=copy
$REL/.tools/uv-venv/bin/uv pip install --python .venv-daint/bin/python -r scripts/daint/requirements-mpi.txt
$REL/.tools/uv-venv/bin/uv cache clean -q
.venv-daint/bin/python -c "import mpi4py, nvidia.nccl; print('mpi4py', mpi4py.__version__, 'nccl', nvidia.nccl.__path__[0])"

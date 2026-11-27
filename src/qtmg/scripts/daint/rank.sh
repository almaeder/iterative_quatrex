#!/bin/bash
# One MPI rank = one GH200 module: its GPU, its 72 cores, host memory preferably on the LPDDR NUMA nodes (0-3; without a
# policy, host pages of a full module land on the HBM of the GPUs). Local rank: from Slurm with srun (several nodes,
# QTX_LAUNCH=srun), else from the MPI launcher on one node (mpiexec -launcher fork).
if [ "${QTX_LAUNCH:-}" = srun ]; then l=$SLURM_LOCALID; else
  r=${PMI_RANK:-${MPI_LOCALRANKID:-${OMPI_COMM_WORLD_LOCAL_RANK:-0}}}; l=$(( r % 4 )); fi
export CUDA_VISIBLE_DEVICES=$l
M=--membind=0-3; numactl --preferred-many=0-3 true 2>/dev/null && M=--preferred-many=0-3
exec numactl --physcpubind=$(( l * 72 ))-$(( l * 72 + 71 )) $M "$@"

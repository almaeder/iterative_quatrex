#!/bin/bash
# qtxmg on several GH200 (one MPI rank per GPU, NCCL between the GPUs). Options as run.sbatch (environment), plus
#   GPUS     ranks per node (default 4)
#   DIST     nccl (default) | mpi (host staging, no NCCL)
# One node: MPICH wheel, mpiexec -launcher fork (after setup_mpi.sh):
#   INPUTS=/path/to/inputs DE=0.05,0.15 sbatch -A <account> scripts/daint/run_multi.sbatch
# Several nodes: cray-mpich of the uenv under mpi4py, NCCL over Slingshot (multinode_env.sh):
#   INPUTS=... sbatch -A <account> --nodes=2 --uenv=prgenv-gnu/24.11:v1 scripts/daint/run_multi.sbatch
# The cache (Sigma, bases) must be on a shared file system ($SCRATCH).
#SBATCH --job-name=qtxmg-multi
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --output=qtxmg-multi-%j.out
set -eu
cd "$SLURM_SUBMIT_DIR"
REL=$PWD
source scripts/daint/activate.sh
: "${INPUTS:?INPUTS=input folder (python -m qtxmg.prep.quatrex QUATREX_FOLDER --out DIR)}"
OUT=${OUT:-$PWD/qtxmg_$SLURM_JOB_ID}
CACHE=${CACHE:-${SCRATCH:-$PWD}/qtxmg_cache/$(basename "$INPUTS")}
GPUS=${GPUS:-4}; NODES=${SLURM_NNODES:-1}; N=$(( GPUS * NODES ))
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
ARGS=(--inputs "$INPUTS" --dE "${DE:-0.05,0.15,0.30}" --up "${UP:-y:1e-5,x:1e-5}" --cap "${CAP:-16,2,2}" ${SPLIT:+--cap-split $SPLIT}
      --mu="${MU:--0.15,0.6}" --cut "${CUT:-1e-5}" --obc "${OBC:-beyn}" --maxit "${MAXIT:-100}" --cache "$CACHE" --gpu
      --dist "${DIST:-nccl}" --out $OUT.json ${EXTRA:-})
echo "$(hostname) job $SLURM_JOB_ID: $NODES node(s) x $GPUS GPUs, inputs $INPUTS, cache $CACHE"
( while sleep 5; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd' '; done ) > $OUT.mon 2>&1 & MON=$!
rc=0
if [ "$NODES" -gt 1 ]; then
  source scripts/daint/multinode_env.sh
  srun -n $N -N $NODES --ntasks-per-node=$GPUS --mpi=cray_shasta --cpu-bind=none --gpu-bind=none \
    scripts/daint/rank.sh python -u -m qtxmg.run "${ARGS[@]}" > $OUT.log 2>&1 || rc=$?
else
  $REL/.venv-daint/bin/mpiexec -launcher fork -n $N scripts/daint/rank.sh python -u -m qtxmg.run "${ARGS[@]}" > $OUT.log 2>&1 || rc=$?
fi
kill $MON
echo "exit $rc after $SECONDS s, GPU MiB peak $(tr ' ' '\n' < $OUT.mon | sort -n | tail -1)"
grep -E "^ *level|^setup|^obc|^hierarchy|^ *bottom|^Ec|Error|done|GPU memory" $OUT.log | cut -c1-300 || true
exit $rc

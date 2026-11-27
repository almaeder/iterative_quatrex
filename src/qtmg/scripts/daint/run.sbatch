#!/bin/bash
# qtmg on one GH200 of daint.alps. Every option of the solver can be set by the environment (defaults: the 2808a MoS2/HfO2
# device of the tests); extra solver options in EXTRA. Output: OUT.json and OUT.log next to the job's .out file.
#   INPUTS=/path/to/inputs DE=0.05,0.15 sbatch -A <account> scripts/daint/run.sbatch            (from the release folder)
#   INPUTS=... UP=y:1e-5,x:1e-5 SPLIT=2,1 MU=-0.15,0.6 OBC=beyn CACHE=$SCRATCH/qtmg/dev sbatch -A <account> scripts/daint/run.sbatch
#SBATCH --job-name=qtmg
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=qtmg-%j.out
set -eu
cd "$SLURM_SUBMIT_DIR"
source scripts/daint/activate.sh
: "${INPUTS:?INPUTS=input folder (python -m qtmg.prep.quatrex QUATREX_FOLDER --out DIR)}"
OUT=${OUT:-$PWD/qtxmg_$SLURM_JOB_ID}
CACHE=${CACHE:-${SCRATCH:-$PWD}/qtxmg_cache/$(basename "$INPUTS")}
echo "$(hostname) job $SLURM_JOB_ID: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), inputs $INPUTS, cache $CACHE"
( while sleep 5; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; done ) > $OUT.mon 2>&1 & MON=$!
python -u -m qtmg --inputs "$INPUTS" --dE "${DE:-0.05,0.15,0.30}" --up "${UP:-y:1e-5,x:1e-5}" --cap "${CAP:-16,2,2}" \
  ${SPLIT:+--cap-split $SPLIT} --mu="${MU:--0.15,0.6}" --cut "${CUT:-1e-5}" --obc "${OBC:-beyn}" --maxit "${MAXIT:-100}" \
  --cache "$CACHE" --gpu --out $OUT.json ${EXTRA:-} 2>&1 | tee $OUT.log
rc=${PIPESTATUS[0]}
kill $MON
echo "exit $rc after $SECONDS s, GPU MiB peak $(sort -n $OUT.mon | tail -1)"
exit $rc

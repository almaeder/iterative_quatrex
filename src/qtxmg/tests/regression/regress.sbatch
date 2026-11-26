#!/bin/bash
# Regression of the multi-GPU solver on daint against the references in tests/regression/ref (MoS2/HfO2 devices):
#   2808a   22 848 orbitals,  4 GPUs (1 node), Beyn,   11 energies   (reference: qtxmg 0.1.0 single GPU = qtx_ms.hier.run)
#   6nm    ~250 000 orbitals,  4 GPUs (1 node), folded, 8 energies    (reference: qtx_ms.hier.run, 4 GPUs)
#   12nm    714 240 orbitals,  8 GPUs (2 nodes), Beyn,  3 energies    (reference: qtx_ms.hier.run, 8 GPUs)
# Options of the reference run (cuts, contour, CAP, Ec); bases and Sigma built from scratch in a fresh cache folder.
# From the release folder, after setup.sh and setup_mpi.sh:
#   DEV=2808a INPUTS=/path/to/mos2_hfo2     sbatch -A <account> tests/regression/regress.sbatch
#   DEV=6nm   INPUTS=/path/to/mos2_hfo2_6nm sbatch -A <account> tests/regression/regress.sbatch
#   DEV=12nm  INPUTS=/path/to/12nm_inputs   sbatch -A <account> --nodes=2 --uenv=prgenv-gnu/24.11:v1 tests/regression/regress.sbatch
# CACHE (default $SCRATCH/qtxmg_regress/<DEV>_<job>), DIST (nccl), GPUS (4 per node), EXTRA: as run_multi.sbatch.
# The comparison table and PASS / FAIL are at the end of the job output (tests/regression/compare.py).
#SBATCH --job-name=qtxmg-regress
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --output=qtxmg-regress-%j.out
set -eu
cd "$SLURM_SUBMIT_DIR"
: "${DEV:?DEV=2808a | 9072a | 21600a | 6nm | 12nm}" "${INPUTS:?INPUTS=input set of the device}"
REF=tests/regression/ref/$DEV.json
source scripts/daint/activate.sh
eval "$(python - $REF <<'PY'
import json, sys
r = json.load(open(sys.argv[1])); a = r["args"]
print(f"DE={','.join(r['E'])} UP={a['up']} SPLIT={a['cap_split']} MU={a['mu']} CUT={a['cut']} CAP={a['cap']} OBC={r['obc']} MAXIT={a['maxit']}")
print(f"REFX='--Ec {r['Ec']!r} --npts {a['npts']} --eta {a['eta']} --cap-spacing {a['cap_spacing']} --cap-tov {a['cap_tov']} --halo {a['halo']} --tol {a['tol']}'")
PY
)"
export DE UP SPLIT MU CUT CAP OBC MAXIT INPUTS
export CACHE=${CACHE:-${SCRATCH:-$PWD}/qtxmg_regress/${DEV}_$SLURM_JOB_ID} OUT=${OUT:-$PWD/qtxmg_regress_${DEV}_$SLURM_JOB_ID}
export EXTRA="$REFX ${EXTRA:-}"
echo "regression $DEV: $INPUTS, reference $REF"
rc=0
bash scripts/daint/run_multi.sbatch || rc=$?
python tests/regression/compare.py $REF $OUT.json || rc=1
exit $rc

#!/bin/bash
# QuaTrEx / deviceBuilder device folder -> qtmg input set, on a compute node (CPU only).
#   SRC=/path/to/device_MoS2HfO2_2808a_29088o DST=/path/to/inputs/mos2_hfo2 sbatch -A <account> scripts/daint/prep.sbatch
#SBATCH --job-name=qtmg-prep
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:30:00
#SBATCH --output=qtmg-prep-%j.out
set -eu
cd "$SLURM_SUBMIT_DIR"
source scripts/daint/activate.sh
python -u -m qtmg.prep.quatrex "${SRC:?SRC=QuaTrEx device folder}" --out "${DST:?DST=output input folder}" ${EXTRA:-}

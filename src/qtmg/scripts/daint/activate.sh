# source scripts/daint/activate.sh      (from the release folder or anywhere)
_rel=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source "$_rel/.venv-daint/bin/activate"
export CUPY_CACHE_DIR=${CUPY_CACHE_DIR:-$_rel/.tools/cupy_cache}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-${OMP_NUM_THREADS:-16}}
unset _rel

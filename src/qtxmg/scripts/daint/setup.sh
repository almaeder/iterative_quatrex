#!/bin/bash
# Python environment for qtxmg on daint.alps (GH200, aarch64): <release>/.venv-daint with cupy (CUDA toolkit from wheels,
# no uenv / module needed). uv, its python and caches stay in <release>/.tools. Run on a compute node:
#   srun -A <account> -p debug -N1 -t 20 bash scripts/daint/setup.sh
set -euo pipefail
REL=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); cd "$REL"
TOOLS=$REL/.tools
export UV_CACHE_DIR=$TOOLS/uv-cache UV_PYTHON_INSTALL_DIR=$TOOLS/python UV_PYTHON_BIN_DIR=$TOOLS/bin UV_LINK_MODE=copy
unset CONDA_PREFIX VIRTUAL_ENV PYTHONPATH
if [ ! -x "$TOOLS/uv-venv/bin/uv" ]; then
  /usr/bin/python3.11 -m venv "$TOOLS/uv-venv" 2>/dev/null || python3 -m venv "$TOOLS/uv-venv"
  "$TOOLS/uv-venv/bin/pip" install -q uv
fi
UV=$TOOLS/uv-venv/bin/uv
[ -d .venv-daint ] || $UV venv --python 3.12 --python-preference only-managed .venv-daint
$UV pip install --python .venv-daint/bin/python -r scripts/daint/requirements.txt
$UV pip install --python .venv-daint/bin/python --no-deps -e .
$UV cache clean -q
.venv-daint/bin/python -c "import cupy, numpy, scipy, qtxmg; cupy.abs(cupy.ones(1)); print('qtxmg', qtxmg.__version__, 'cupy', cupy.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__)"

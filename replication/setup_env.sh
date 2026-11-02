#!/usr/bin/env bash
# The environment of the stored results on a Linux x86_64 machine: Python 3.13 and the pinned
# packages of requirements.txt and requirements-learning.txt in .venv/ of the repository. uv
# downloads the interpreter, so the Python of the machine does not matter. A C compiler (gcc) is
# needed for the compiled estimators and solvers.
#
#   bash replication/setup_env.sh              torch as on the reference machine (PyPI wheel)
#   bash replication/setup_env.sh --torch-cpu  the smaller CPU build of the same torch version
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v gcc >/dev/null; then
    echo "gcc is missing: install it first (Debian/Ubuntu: apt-get install -y gcc libc6-dev)" >&2
    exit 1
fi
if ! command -v uv >/dev/null; then
    python3 -m pip install --quiet --user uv 2>/dev/null \
        || python3 -m pip install --quiet --user --break-system-packages uv \
        || python3 -m pip install --quiet uv
    export PATH="$HOME/.local/bin:$PATH"
fi
UV="$(command -v uv || echo "python3 -m uv")"

$UV python install 3.13
$UV venv --python 3.13 .venv
$UV pip install --python .venv/bin/python -r requirements.txt
if [ "${1:-}" = "--torch-cpu" ]; then
    $UV pip install --python .venv/bin/python torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
    $UV pip install --python .venv/bin/python optuna==5.0.0
else
    $UV pip install --python .venv/bin/python -r requirements-learning.txt
fi
.venv/bin/python reproduce.py check

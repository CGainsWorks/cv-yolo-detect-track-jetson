#!/bin/sh
# One-time setup on a fresh machine.
#
#   sh scripts/setup.sh            # runtime only, enough for `make demo`
#   sh scripts/setup.sh --train    # plus the training and export toolchain
#
# Everything after this is a make target.  `make demo` needs the runtime and
# the published weights; `make experiments` needs --train and a CUDA GPU.

set -eu

PYTHON=${PYTHON:-python3}
VENV=${VENV:-.venv}
WITH_TRAIN=0
[ "${1:-}" = "--train" ] && WITH_TRAIN=1

"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || {
    echo "python 3.10 or newer is required; found $("$PYTHON" --version)" >&2
    exit 1
}

if [ ! -d "$VENV" ]; then
    "$PYTHON" -m venv "$VENV"
fi

if [ -x "$VENV/bin/python" ]; then
    VPY="$VENV/bin/python"
else
    VPY="$VENV/Scripts/python.exe"   # Windows
fi

"$VPY" -m pip install --upgrade pip
"$VPY" -m pip install -r requirements.txt

if [ "$WITH_TRAIN" = "1" ]; then
    "$VPY" -m pip install -r requirements-train.txt
fi

"$VPY" scripts/fetch_model.py || {
    echo
    echo "the published weights could not be fetched." >&2
    echo "build them locally with: make data train reid export quantise" >&2
}

echo
echo "ready.  next:"
echo "  make demo        detect, track and score one sequence"
echo "  make check       lint, tests and the documentation check"
[ "$WITH_TRAIN" = "1" ] && echo "  make experiments regenerate every published number"

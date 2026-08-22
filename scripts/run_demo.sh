#!/bin/sh
# The demo, for people who would rather not read the Makefile.
#
#   sh scripts/run_demo.sh                 # synthetic sequence
#   sh scripts/run_demo.sh clip.mp4        # a video file
#   sh scripts/run_demo.sh 0               # camera 0

set -eu

SOURCE=${1:-synth:7}
VENV=${VENV:-.venv}
if [ -x "$VENV/bin/python" ]; then
    PY="$VENV/bin/python"
elif [ -x "$VENV/Scripts/python.exe" ]; then
    PY="$VENV/Scripts/python.exe"
else
    PY=${PYTHON:-python3}
fi

MODEL=models/yolov8n_edgetrack_fp32.onnx
if [ ! -f "$MODEL" ]; then
    echo "fetching the published weights"
    "$PY" scripts/fetch_model.py
fi

mkdir -p results
PYTHONPATH=src "$PY" scripts/run_demo.py \
    --source "$SOURCE" \
    --model "$MODEL" \
    --video results/demo.mp4 \
    --out results/demo.json

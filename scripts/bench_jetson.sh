#!/bin/sh
# Benchmark on a Jetson.  NEVER EXECUTED: there is no board.  See docs/jetson.md.
#
# Run it from the repository root on the device, inside a virtualenv created
# with --system-site-packages so that the Jetpack TensorRT is visible.
#
#   sh scripts/bench_jetson.sh
#
# It sweeps the power modes because a Jetson benchmark without one is close to
# meaningless: the same board at 7 W and at MAXN differs by more than the
# difference between two precisions, so a single number invites the reader to
# attribute a power-budget effect to a quantisation choice.

set -eu

PYTHON=${PYTHON:-python3}
DATA=${DATA:-data/processed}
OUT=${OUT:-results}
MODES=${MODES:-"0 1"}   # nvpmodel ids; 0 is usually MAXN

if [ ! -f /etc/nv_tegra_release ]; then
    echo "this is not a Jetson: /etc/nv_tegra_release does not exist" >&2
    echo "the benchmark would run and would be labelled device_class=desktop" >&2
    exit 1
fi

cat /etc/nv_tegra_release
"$PYTHON" -c "import tensorrt; print('tensorrt', tensorrt.__version__)"

mkdir -p "$OUT"

# Calibration frames.  The small preset is enough for INT8 and takes about
# three minutes on a board.
if [ ! -d "$DATA/images/train" ]; then
    "$PYTHON" scripts/make_dataset.py --preset small --root "$DATA"
fi

"$PYTHON" scripts/fetch_model.py

for precision in fp32 fp16 int8; do
    echo "building the $precision engine"
    "$PYTHON" scripts/export_tensorrt.py \
        --onnx models/yolov8n_edgetrack_fp32.onnx \
        --precision "$precision" \
        --data-root "$DATA"
done

for mode in $MODES; do
    echo "power mode $mode"
    sudo nvpmodel -m "$mode"
    sudo jetson_clocks
    # Thermals need a moment to settle after a clock change, and the first
    # seconds after jetson_clocks are not representative of anything.
    sleep 20
    "$PYTHON" scripts/benchmark_latency.py \
        --variants configs/benchmark.yaml \
        --data-root "$DATA" \
        --only tensorrt-fp32 tensorrt-fp16 tensorrt-int8 \
        --iterations 500 \
        --skip-missing \
        --out "$OUT/latency_jetson_mode${mode}.json"
done

echo
echo "wrote $OUT/latency_jetson_mode*.json"
echo "to publish: copy them into results/published/ and run 'make docs-fix'"

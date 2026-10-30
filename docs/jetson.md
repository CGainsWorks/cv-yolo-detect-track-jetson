# The Jetson, and the fact that there isn't one

**Nothing in this repository has run on a Jetson.**  No latency number in
`benchmarks/results.csv` or anywhere else came from edge hardware.  Every
published measurement was taken on a desktop with an RTX 4060 and a Ryzen 7
5700X.  `scripts/benchmark_latency.py` stamps `device_class` into every results
file and CI fails if a published file ever claims otherwise.

**The TensorRT path has run, on the desktop.**  `make tensorrt-desktop` builds
FP32, FP16 and INT8 engines with `scripts/export_tensorrt.py` inside NVIDIA's
`nvcr.io/nvidia/tensorrt:25.06-py3` container (TensorRT `10.11`, CUDA `12.9`) on the
RTX 4060 under WSL2, scores them on the test split and times them through
`edgetrack.detect.trt_backend`.  The results, with the driver, CUDA and
TensorRT versions and the GPU and CPU contention state, are in
`results/published/tensorrt.json` and in the generated tables in the README
and `docs/results.md`.  The first real INT8 build found a bug, a calibrator
with no CUDA context, which is fixed.  The build times in that file come from
the first build, when other containers were using the CPU, so treat them as
indicative only.

That is a real gap and this page is about the shape of it: what exists, what
would have to happen, and which of the conclusions in `docs/results.md` would
survive contact with the board.

---

## What is built and untested

| Piece | State |
| --- | --- |
| `src/edgetrack/detect/trt_backend.py` | TensorRT 10 tensor-name path run on the desktop RTX 4060, at three precisions over the full test split.  The TensorRT 8 binding-index path, kept for Jetpack 5, has not run anywhere. |
| `scripts/export_tensorrt.py` | Run on the desktop RTX 4060 at FP32, FP16 and INT8.  The entropy calibrator streams the same preprocessed frames the deployed model sees and writes a calibration cache. |
| `docker/Dockerfile.tensorrt`, `scripts/measure_tensorrt.py` | The desktop TensorRT image and the script that builds, scores, times and records.  Run. |
| `tests/test_tensorrt_api.py` | Run in CI by the `tensorrt-api` job against the pinned `tensorrt-cu12` wheel in `requirements-tensorrt.txt`, on a runner with no GPU. |
| `docker/Dockerfile.jetson` | Written, never built.  The base image is arm64 and needs an NGC login, so it will not build on a CI runner. |
| `scripts/bench_jetson.sh` | Written, never executed.  Pins the clocks, runs the benchmark at each power mode and writes `benchmarks/latency_jetson.md`. |
| `requirements-jetson.txt` | Written, never installed. |

The interface the Jetson pieces agree on is fixed by
`edgetrack.detect.base.Detector`, which the ONNX Runtime backend already
satisfies and which the benchmark already drives.  So the work on the day the
board arrives is running commands, not writing a backend.

---

## What would have to happen

Roughly two hours of work, most of it waiting for an INT8 build.

```bash
# On the board.  --system-site-packages is not optional: TensorRT, CUDA and
# cuDNN come from Jetpack and live in the system Python.
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
pip install -r requirements-jetson.txt

# Clocks pinned, or the numbers are a lottery.  Jetson clocks scale with
# thermals and with the governor, and an unpinned board can report a
# forty percent spread between two runs of the same benchmark.
sudo nvpmodel -m 0        # MAXN
sudo jetson_clocks

python scripts/make_dataset.py --preset small     # calibration frames
python scripts/fetch_model.py                     # the ONNX graphs

python scripts/export_tensorrt.py --precision fp32
python scripts/export_tensorrt.py --precision fp16
python scripts/export_tensorrt.py --precision int8 --calib-images 512

python scripts/benchmark_latency.py --variants configs/benchmark.yaml \
  --only tensorrt-fp32 tensorrt-fp16 tensorrt-int8 \
  --out results/latency_jetson.json
```

`configs/benchmark.yaml` already lists the three TensorRT rows.  Today
`--skip-missing` drops them and records why in the `skipped` block of the
results file, which is how an absent row stays visible instead of quietly
vanishing from the table.

---

## Which conclusions would survive, and which would not

**Would survive.**  The accuracy column.  mAP is a property of the arithmetic,
not of the runtime, so the FP32-to-FP16 gap measured here is what FP16 costs
anywhere.  The finding that quantising YOLOv8's detection head destroys the
class scores is about where the dynamic range is in the graph, and TensorRT
would hit the same wall for the same reason, which is why
`scripts/export_tensorrt.py` keeps FP16 enabled alongside INT8 so that layers
TensorRT declines to quantise fall back to half rather than to full precision.

**Would not survive.**  Everything in the latency column, and not by a small
factor.  An Orin Nano is roughly a twentieth of an RTX 4060 in raw throughput,
its memory is shared with the CPU rather than dedicated, and TensorRT fuses
convolution, batch normalisation and activation into single kernels in a way
ONNX Runtime's CPU provider does not.  The *ordering* of the precisions would
probably hold; the ratios would not, and the ratios are the interesting part.

**Answered on the desktop and expected to carry over.**  The INT8 accuracy on
TensorRT.  It was unknown when this page was first written; the desktop run
measured it and it landed on the wrong side: TensorRT's entropy calibrator
with implicit quantisation quantises the detection head and loses 0.06408
mAP@0.5:0.95, more than ONNX Runtime's head-excluded INT8.  The same TensorRT
version on a board does the same arithmetic, so the accuracy should carry over
closely; the kernels it picks will differ, so not bit for bit.

Also unknown: whether the tracker keeps up.  The published end-to-end numbers
show the detector dominating on a desktop, but the appearance encoder runs once
per detection rather than once per frame, so on a board a third the speed with
eight objects in shot the balance could invert.  The benchmark reports the
stages separately for exactly this reason.

---

## What a Jetson would change about the design

Two things are sized for hardware that has not been measured, and both are
guesses until it has been:

* **YOLOv8n at 640x384.**  Chosen because an Orin Nano has 8 GB shared between
  the GPU, the ROS2 stack and the operating system.  If the board turns out to
  have headroom, `yolov8s` is a config change; if it does not, the input drops
  to 512x320 before the model does, because resolution is cheaper to give up
  than capacity on small objects.
* **The appearance encoder at 0.1 M parameters.**  Deliberately small because
  its cost multiplies by the object count.  On a desktop that decision was
  measurable but not urgent; on a board it is the difference between DeepSORT
  being an option and not.

---

## What CI checks and the manual step

A hosted runner has no GPU.  The pinned TensorRT wheel installs and imports
there but `trt.Builder` needs a CUDA driver and aborts the process without
one, so CI cannot create a network definition, parse the ONNX graph or build
an engine.  The `tensorrt-api` job does what is possible: it imports the
backend, resolves every `trt.*` symbol the backend and the engine builder use
against the pinned wheel and builds the INT8 calibrator on TensorRT's real base
class.  Building and measuring engines stays a manual step on a machine with an
NVIDIA GPU:

```bash
make tensorrt-desktop      # docker build + run --gpus all --network none
```

On a Jetson the commands in "What would have to happen" above still apply.

## Cost

An Orin Nano developer kit is about £200, plus a camera and a power supply.
That is the only thing standing between this page and a filled-in table.

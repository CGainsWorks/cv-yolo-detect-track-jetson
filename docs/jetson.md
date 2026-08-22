# The Jetson, and the fact that there isn't one

**Nothing in this repository has run on a Jetson.**  No TensorRT engine has been
built, no INT8 calibration has gone through TensorRT's calibrator, and no
latency number in `benchmarks/results.csv` came from edge hardware.  Every
published measurement was taken on a desktop with an RTX 4060 and a Ryzen 7
5700X.  `scripts/benchmark_latency.py` stamps `device_class` into every results
file and CI fails if a published file ever claims otherwise.

That is a real gap and this page is about the shape of it: what exists, what
would have to happen, and which of the conclusions in `docs/results.md` would
survive contact with the board.

---

## What is built and untested

| Piece | State |
| --- | --- |
| `src/edgetrack/detect/trt_backend.py` | Written, never executed.  Handles the TensorRT 8 binding-index API and the TensorRT 10 tensor-name API, because Jetpack 5 and Jetpack 6 sit on either side of that change. |
| `scripts/export_tensorrt.py` | Written, never executed.  Builds FP32, FP16 and INT8 engines, with an entropy calibrator that streams the same preprocessed frames the deployed model sees, and a calibration cache so a rebuild after a TensorRT upgrade is cheap. |
| `docker/Dockerfile.jetson` | Written, never built.  The base image is arm64 and needs an NGC login, so it will not build on a CI runner. |
| `scripts/bench_jetson.sh` | Written, never executed.  Pins the clocks, runs the benchmark at each power mode and writes `benchmarks/latency_jetson.md`. |
| `requirements-jetson.txt` | Written, never installed. |

The interface those five agree on is fixed by
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

**Genuinely unknown.**  The INT8 accuracy on TensorRT.  This repository measures
INT8 through ONNX Runtime's static QDQ quantisation with a MinMax calibrator.
TensorRT uses its own entropy calibrator, fuses differently and picks its own
per-layer precision, so its INT8 mAP could land either side of the number here.
The measurement in `docs/results.md` bounds the question rather than answering
it, and it is labelled that way there.

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

## Cost

An Orin Nano developer kit is about £200, plus a camera and a power supply.
That is the only thing standing between this page and a filled-in table.

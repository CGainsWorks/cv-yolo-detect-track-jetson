# Status

Last updated: 2026-10-16

## Complete

- Procedural warehouse dataset with box and identity ground truth, regenerable
  byte for byte from seeds (`scripts/make_dataset.py`).
- YOLOv8n fine-tuned on it; FP32 and FP16 ONNX exports and three INT8 ONNX
  variants, all scored on the test split (`results/published/detection.json`).
- ByteTrack and DeepSORT written from the papers, CLEAR-MOT and ID metrics
  cross-checked against `py-motmetrics`, every ID switch attributed to a cause.
- ROS2 node publishing `vision_msgs` detections and tracks, checked in CI by a
  real subscriber inside `ros:humble-ros-base`.
- TensorRT engines at FP32, FP16 and INT8 built by `scripts/export_tensorrt.py`
  in NVIDIA's `nvcr.io/nvidia/tensorrt:25.06-py3` container on a desktop RTX
  4060 under WSL2, scored and timed through `edgetrack.detect.trt_backend`
  (`make tensorrt-desktop`, `results/published/tensorrt.json`, the generated
  TensorRT tables in the README and `docs/results.md`).  The first real INT8
  build found and fixed a missing CUDA context in the calibrator.
- CI `tensorrt-api` job: imports the backend and checks every TensorRT symbol
  the backend and the engine builder use against the pinned `tensorrt-cu12`
  wheel, on a runner with no GPU.
- `docs/licences.md` states the AGPL-3.0 obligation on the detector weights.
- Tests: 275 in total.  262 pass locally on Windows with 3 skipped (the 11
  TensorRT API tests as one module and two that need torch or ultralytics);
  the 11 TensorRT API tests pass in the CI job that installs the wheel.

## In progress

Nothing.

## Not started

- **Swap the detector for an Apache-2.0 one (YOLOX, RTMDet or D-FINE).**  The
  current weights are fine-tuned from Ultralytics' AGPL-3.0 `yolov8n.pt`, so
  closed commercial use needs an Ultralytics Enterprise Licence.  YOLOX and
  RTMDet emit a different head, so this needs a second decoder in
  `edgetrack.detect.base`, retraining, re-export and new published numbers.
- TensorRT INT8 with the detection head's output convolutions held at FP16
  through per-layer precision constraints, to test whether that recovers the
  accuracy implicit quantisation loses.
- Validation on one real, commercially licensed tracking clip, so absolute
  accuracy numbers transfer.

## Limitations

- **No Jetson (hardware TODO).**  Every latency number, the TensorRT ones
  included, came from a desktop RTX 4060.  `docker/Dockerfile.jetson`,
  `scripts/bench_jetson.sh` and `benchmarks/latency_jetson.md` have never run
  and the Jetson table is empty.  An Orin Nano developer kit closes it.
- **CI cannot build a TensorRT engine.**  Hosted runners have no GPU and
  `trt.Builder` aborts without a CUDA driver, so network definition, ONNX
  parsing and engine builds are a manual step (`make tensorrt-desktop`).
- The TensorRT 8 branch of `trt_backend.py` (Jetpack 5) has not run anywhere.
- The detector weights carry an AGPL-3.0 obligation; see `docs/licences.md`.
- The dataset is generated, so absolute accuracy does not transfer to a real
  camera.  `docs/limitations.md` has the full list.

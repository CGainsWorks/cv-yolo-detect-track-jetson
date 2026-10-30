# Status

Last updated: 2026-10-30

## Complete

- Procedural warehouse dataset with box and identity ground truth, regenerable
  byte for byte from seeds (`scripts/make_dataset.py`).
- **Detector swapped to YOLOX-Tiny, Apache-2.0 code and weights.**  Fine-tuned
  from Megvii's COCO checkpoint with YOLOX's own trainer (`scripts/train.py`),
  exported through `edgetrack.detect.yolox_net` with the input normalisation
  folded into the stem and the box decode in the graph, so the head contract,
  every backend, the INT8 quantiser and the TensorRT builder are unchanged.
  The AGPL-3.0 obligation the YOLOv8n weights carried is gone;
  `docs/licences.md` says why YOLOX rather than RTMDet or D-FINE and what
  Apache-2.0 asks of a redistributor.
- FP32 and FP16 ONNX exports and three INT8 ONNX variants, all scored on the
  test split (`results/published/detection.json`).
- ByteTrack and DeepSORT written from the papers, CLEAR-MOT and ID metrics
  cross-checked against `py-motmetrics`, every ID switch attributed to a cause.
  Both trackers re-tuned on val for the new detector.
- Every published number re-measured with the new detector and the old
  YOLOv8n numbers kept beside them in
  `results/published/detector_comparison.json`, read from the last commit
  before the swap by `scripts/compare_detectors.py`.
- ROS2 node publishing `vision_msgs` detections and tracks, checked in CI by a
  real subscriber inside `ros:humble-ros-base`.
- TensorRT engines at FP32, FP16 and INT8 built by `scripts/export_tensorrt.py`
  in NVIDIA's `nvcr.io/nvidia/tensorrt:25.06-py3` container on a desktop RTX
  4060 under WSL2, scored and timed through `edgetrack.detect.trt_backend`
  (`make tensorrt-desktop`, `results/published/tensorrt.json`).
- CI `tensorrt-api` job: imports the backend and checks every TensorRT symbol
  the backend and the engine builder use against the pinned `tensorrt-cu12`
  wheel, on a runner with no GPU.
- Weights for the new detector in GitHub Release `v0.2.0`; `v0.1.0` still holds
  the YOLOv8n artefacts under their AGPL-3.0 obligation and nothing uses them.
- Tests: 280 in total.  267 pass locally on Windows in the runtime venv with 3
  skipped (the 11 TensorRT API tests as one module and two that need torch or
  pycocotools); with the training toolchain installed those two pass as well,
  269 in all and the 11 TensorRT API tests pass in the CI job that installs
  the wheel.

## Where the new detector is worse

Measured, not assumed; `results/published/detector_comparison.json` has the
full set and the README the headline.

- Test split mAP@0.5:0.95 0.8605 against 0.8649 (mAP@0.5 is better, 0.9776
  against 0.9727).  The loss is almost all `pallet`.
- INT8 through ONNX Runtime costs 0.0262 mAP@0.5:0.95, against about a point for
  YOLOv8n.
- Slower on every eager backend: 18.00 ms against 15.69 ms on the CPU, 10.65
  ms against 8.93 ms in eager PyTorch on the GPU.  YOLOX-Tiny has 5,033,739
  parameters against 3,006,428.  On TensorRT the gap is small: 2.40 ms against
  2.08 ms at FP32, 1.66 ms against 1.52 ms at FP16.
- Tracking with ByteTrack: MOTA 0.89993 against 0.91257 and IDF1 0.90797
  against 0.91194, from more false positives (347 against 206) and fewer
  misses (445 against 482).
- The ONNX and engine files are about two thirds larger.

## In progress

Nothing.

## Not started

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
- **CI cannot train.**  Fine-tuning needs a CUDA GPU; CI re-exports the FP32
  graph from the published checkpoint and checks its shape instead.
- The TensorRT 8 branch of `trt_backend.py` (Jetpack 5) has not run anywhere.
- The YOLOX checkpoints are read as Apache-2.0 because Megvii publishes them in
  an Apache-2.0 repository without separate terms; `docs/licences.md` flags
  that as a point to confirm before commercial use.
- The dataset is generated, so absolute accuracy does not transfer to a real
  camera.  `docs/limitations.md` has the full list.

# How it fits together

```
                    ┌──────────────────────────────────────────┐
   configs/data     │  edgetrack.data.synth                    │
   seeds ──────────▶│  procedural warehouse scene              │
                    │  boxes + identities + visibility         │
                    └───────┬──────────────────────┬───────────┘
                            │                      │
                 YOLO layout│                      │MOTChallenge layout
                  train/val │                      │test sequences
                            ▼                      ▼
                   ┌─────────────────┐    ┌──────────────────────┐
                   │ scripts/train   │    │  ground truth for    │
                   │ YOLOv8n         │    │  MOTA / IDF1 / IDsw  │
                   └────────┬────────┘    └───────────┬──────────┘
                            │ .pt                     │
              ┌─────────────┴──────────────┐          │
              ▼                            ▼          │
      export_onnx --fp32           export_onnx --fp16 │
              │                            │          │
              ▼                            │          │
       quantize_onnx                       │          │
     int8 / allconv / allops               │          │
              │                            │          │
              └──────────┬─────────────────┘          │
                         ▼                            │
              ┌──────────────────────┐                │
              │ edgetrack.detect     │                │
              │ onnx | torch | trt   │  ◀── one Detector interface
              └──────────┬───────────┘                │
                         │ Detection[]                │
                         ▼                            │
              ┌──────────────────────┐                │
              │ track.appearance     │  histogram or ONNX re-ID
              └──────────┬───────────┘                │
                         │ Detection[] + feature      │
                         ▼                            │
              ┌──────────────────────┐                │
              │ ByteTrack │ DeepSort │  shared Kalman filter
              └──────────┬───────────┘                │
                         │ Track[]                    │
        ┌────────────────┼─────────────────┬──────────┘
        ▼                ▼                 ▼
  annotated video   ROS2 topics      metrics.mot + metrics.idswitch
  (run_demo)        (ros2.node)      (eval_tracking)
```

---

## The seams that matter

**One `Detector` interface, three backends.**  ONNX Runtime, PyTorch and
TensorRT differ in exactly one method: turning a float tensor into a raw head
output.  Letterboxing, decoding and NMS live in `edgetrack.detect.base` and are
shared, so a difference in mAP between FP32 and INT8 is a difference in the
arithmetic rather than in how the boxes were rescaled.
`tests/test_backends.py` holds the backends to that.

**One Kalman filter, two trackers.**  ByteTrack and DeepSORT share
`edgetrack.track.kalman` and `edgetrack.track.base`, so the difference between
their results is a difference in association and nothing else.

**One generator, two on-disk layouts.**  Detection training wants a flat
directory of images with one label file each; tracking evaluation wants whole
sequences with MOTChallenge ground truth.  Both are written from the same
generator in the same pass, so they cannot disagree about where an object was.

**The runtime imports neither torch nor ultralytics.**  `edgetrack.reid_net` is
the only module that imports torch at module scope and nothing in the runtime
path reaches it.  See `docs/licences.md`.

**Detections are computed once and shared.**  `scripts/eval_tracking.py`
detects each frame once and feeds the identical list to every tracker in the
run, and `scripts/tune_tracker.py` caches detections across the whole parameter
sweep.  A comparison where each tracker ran its own detector would be comparing
two things at once.

---

## What runs where

| Stage | Machine | In CI |
| --- | --- | --- |
| Dataset generation | anything | yes, tiny preset, checked for byte determinism |
| Detector training | CUDA GPU | no |
| Re-ID training | CUDA GPU | no |
| ONNX export FP32 | CPU | yes |
| ONNX export FP16 | CUDA GPU | no, needs a GPU to trace a half module |
| INT8 quantisation | CPU | yes |
| Detection accuracy | CPU | yes, on a subset |
| Tracking metrics | CPU | yes, through the demo |
| Latency benchmark | the machine being measured | no |
| ROS2 node | ROS2 Humble | yes, in a container with a real subscriber |
| TensorRT engines | Jetson | no, and never run anywhere |

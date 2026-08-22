# Limitations

Ordered by how much they should change your reading of the results.

---

## 1. There is no Jetson

No TensorRT engine has been built and no measurement was taken on edge
hardware.  Every latency number came from a desktop with an RTX 4060 and a
Ryzen 7 5700X.  The TensorRT backend, the engine builder, the Jetson Dockerfile
and the benchmark script are written and are untested code.

`docs/jetson.md` is the full account, including which conclusions would survive
contact with a board and which would not.  The short version: the accuracy
column is hardware-independent and would hold; the latency column would not,
and the latency column is the interesting half of the headline table.

---

## 2. The dataset is generated, not photographed

`src/edgetrack/data/synth.py` renders a warehouse-floor scene with procedurally
drawn people, forklifts, pallets and cones.  It is not photorealistic and a
detector trained on it will not transfer to a real camera.

What that buys, and it is not nothing: exact ground truth for identities
through occlusions, which is the expensive part of a real MOT annotation; a
controllable crossing and occlusion schedule, so the tracker comparison runs on
the cases that actually separate the two; and a dataset anybody can regenerate
from a seed, which is what makes every number here checkable rather than
merely reported.

What it costs: the absolute mAP is not comparable to anything.  A detector
scoring 0.86 mAP50-95 here would score far less on real footage of the same
classes, and the number should be read as "this pipeline trains" rather than
as "this model is good".  The *differences* between precisions are the part
worth reading, and those are measured on identical images.

The importer for real annotations is not written.  `data/DATASET_CARD.md`
describes the layout a CVAT or Label Studio export would need to land in, and
nothing above `scripts/` would change, but the converter itself does not exist
yet.

---

## 3. ONNX Runtime's CUDA provider does not work on the host

`onnxruntime-gpu` 1.29 links against `cublasLt64_13.dll` and therefore needs a
CUDA 13 runtime; the torch build in `requirements-train.txt` is CUDA 12.6 and
ships CUDA 12 libraries.  Installing both leaves ONNX Runtime listing
`CUDAExecutionProvider` and then silently falling back to the CPU provider at
session creation.

Rather than publish a GPU row that is secretly a CPU row, the ONNX rows in the
results table are CPU only and the GPU numbers come from the PyTorch backend.
That is why the table has no `onnx-cuda-*` row, and `configs/benchmark.yaml`
says so where somebody would look for one.

The consequence for the reader: the FP16 speedup is measured on a torch module,
not on the FP16 ONNX graph, and the INT8 speedup is measured on a CPU where
INT8 has a much bigger advantage than it does on a GPU with tensor cores.  Do
not read the CPU INT8 speedup as a prediction of the Jetson one.

---

## 4. DeepSORT loses to ByteTrack here, and the reason is partly the dataset

The tracking table shows ByteTrack ahead on MOTA, IDF1 and switch count.  That
is a real result on this data and it matches what the ByteTrack paper reports
on MOT17, but two properties of this dataset push it further than it would go
on real footage:

* The scene has at most five moving objects.  Appearance matters most when
  many similar objects are in shot, and five is not many.
* The detector is good (0.86 mAP50-95), so ByteTrack's low-score second pass
  has plenty to work with.  DeepSORT has no equivalent stage: its
  `det_thresh` discards weak detections before the cascade ever sees them.

`docs/tracking.md` separates the two effects by rerunning both trackers on
ground-truth boxes, which removes detection quality from the comparison
entirely.

---

## 5. The appearance embedding is trained and evaluated on the same generator

`scripts/train_reid.py` trains on crops from the train sequences and evaluates
on crops from the val sequences, which are different scenes but the same
renderer with the same palette.  Rank-1 accuracy near 0.98 says the embedding
separates *these* identities; it says nothing about real people in similar
clothing, where colour histograms and small networks both do much worse.

The test sequences use a different palette and heavier noise, so the tracking
numbers do include a domain shift.  The re-identification numbers do not.

---

## 6. The INT8 accuracy is an ONNX Runtime measurement, not a TensorRT one

The published INT8 model is statically quantised with ONNX Runtime's MinMax
calibrator, convolutions only, with the detection head's output convolutions
left in float.  TensorRT uses a different calibrator, fuses layers differently
and picks its own per-layer precision.  The measurement bounds what TensorRT
INT8 would cost; it does not predict it.

---

## 7. Timings are wall-clock on a shared desktop

The benchmark pins nothing.  It cycles real frames, warms up, and reports
percentiles rather than a mean alone, but the machine has a desktop
environment, a browser and a background Ollama server on it.  The p99 numbers
in particular should be read as an upper bound on a quiet machine rather than
as a property of the model.

---

## 8. Not done, and deliberately so

* No `ament_python` package or launch file for the ROS2 node.  It runs with
  `python3 -m` and CI proves it publishes; wrapping it properly is an hour of
  uninteresting work.
* No `tf2` frames and no camera calibration, so trajectories are in image
  pixels rather than metres.
* No multi-camera handover, no re-identification across sequences, no
  batching.  The node processes one camera at one frame at a time.
* No INT8 quantisation of the appearance encoder.  It is 0.46 MB in FP32 and
  quantising it would save less than the measurement would cost.

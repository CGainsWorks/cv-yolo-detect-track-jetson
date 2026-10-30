# cv-yolo-detect-track-jetson

Real-time detection and multi-object tracking for an NVIDIA Jetson: a YOLOX-Tiny
detector fine-tuned on a custom object set, ByteTrack and DeepSORT compared on
identical detections, ONNX and TensorRT exports at three precisions and a ROS2
node publishing detections and persistent track identities.  Detector, code and
weights are permissively licensed end to end, so it can be used commercially.

![Detection and tracking on a held-out sequence](docs/figures/demo.gif)

*Four classes, persistent identities through occlusion, trajectory trails and
the per-frame budget in the corner.  Produced by `make demo`, which also scores
itself against ground truth.*

> **There is no Jetson.**  Every latency number here was measured on a desktop
> with an RTX 4060.  The TensorRT path has run for real on that desktop GPU,
> inside NVIDIA's TensorRT container under WSL2: `scripts/export_tensorrt.py`
> built FP32, FP16 and INT8 engines and `edgetrack.detect.trt_backend` scored
> and timed them ([below](#tensorrt-measured-on-a-desktop-rtx-4060-not-a-jetson)).
> No number in this repository came from a Jetson and the Jetson benchmark
> script has never run.  [`docs/jetson.md`](docs/jetson.md) says what that costs
> and what it would take to close.

---

## The headline

Accuracy against latency across precisions, on the test split.  Full table and
the machine it came from in [`docs/results.md`](docs/results.md).

| Variant | mAP@0.5:0.95 | Δ vs FP32 | Size | Inference p50 |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 0.8605 | | 20.30 MB | 18.00 ms |
| FP16 | 0.8590 | -0.00144 | 10.21 MB | 20.55 ms |
| INT8, detection head left in float | 0.8343 | -0.0262 | 5.52 MB | 16.11 ms |
| INT8, head quantised too | 0.8206 | -0.03986 | 5.53 MB | 16.27 ms |
| INT8, ONNX Runtime defaults | 0.0000 | -0.86049 | 5.58 MB | 11.74 ms |

Three things that table says:

**Quantising the whole head to INT8 destroys the model silently.**  With ONNX
Runtime's default operator list the graph loads, runs a third faster and
detects nothing, on YOLOX exactly as it did on the YOLOv8 detector this
repository used before.  Restricting quantisation to convolutions recovers
almost all of it and excluding the head's nine output convolutions recovers a
little more, for 11 KB.  The exclusion list is derived from the graph rather
than hard-coded, which is why it carried over to a different architecture
without an edit.

**Precision barely moves the latency on this hardware and it is not supposed
to.**  `scripts/probe_launch_bound.py` shows the network is kernel-launch
bound at batch 1: eight times the work takes 1.292 times the wall clock and
four times the pixels takes no longer at all.  FP16 is *slower* than FP32 on
both the CPU and the GPU for the same reason.  That is precisely the gap
TensorRT closes with layer fusion, which is the strongest argument in this
repository for why the Jetson measurement is the one that matters.

**ByteTrack beats DeepSORT here and a colour histogram beats a trained
embedding.**  0.90797 IDF1 against 0.89544 and DeepSORT with a parameter-free
HSV histogram makes 19 identity switches where the trained embedding makes 33.
[`docs/tracking.md`](docs/tracking.md) takes that apart.

---

## The detector swap and where the new one is worse

The detector used to be Ultralytics YOLOv8n, whose weights carry an AGPL-3.0
obligation that blocks closed commercial use.  It is now **YOLOX-Tiny,
Apache-2.0 for the code and for Megvii's COCO checkpoint it is fine-tuned
from**.  [`docs/licences.md`](docs/licences.md) has the licence position and
why YOLOX rather than RTMDet or D-FINE: it is anchor-free on the same three
strides, so with the decode built into the exported graph it emits the same
`(1, 8, 5040)` head and nothing downstream changed.

Every number was re-measured with the same scripts on the same machine.  The
YOLOv8n figures below come from
[`results/published/detector_comparison.json`](results/published/detector_comparison.json),
which `scripts/compare_detectors.py` reads from the last commit before the swap.

| | YOLOv8n | YOLOX-Tiny | |
| --- | ---: | ---: | --- |
| Licence of the weights | AGPL-3.0 | Apache-2.0 | **better** |
| mAP@0.5:0.95, val | 0.89251 | 0.90661 | better |
| mAP@0.5:0.95, test (shifted palette) | 0.86495 | 0.86049 | **worse** |
| mAP@0.5, test | 0.97270 | 0.97756 | better |
| INT8 ONNX, head in float, test | 0.8549 | 0.83429 | **worse** |
| TensorRT INT8, test | 0.80087 | 0.81967 | better |
| ONNX Runtime CPU FP32 inference p50 | 15.69 ms | 18.00 ms | **worse** |
| Eager PyTorch GPU FP32 inference p50 | 8.93 ms | 10.65 ms | **worse** |
| TensorRT FP32 / FP16 inference p50 | 2.08 / 1.52 ms | 2.40 / 1.66 ms | **worse** |
| ByteTrack MOTA / IDF1 | 0.91257 / 0.91194 | 0.89993 / 0.90797 | **worse** |
| ByteTrack false positives / misses | 206 / 482 | 347 / 445 | more FP, fewer misses |
| DeepSORT (re-ID) IDF1 | 0.8946 | 0.89544 | level |

The honest summary: the licence is fixed and accuracy is a wash, a little
better at loose IoU and on the training palette, a little worse at strict IoU
under the palette shift.  The cost is speed and size, because YOLOX-Tiny is
the bigger network: 5,033,739 parameters against 3,006,428 and every exported
file is about two thirds larger.  On TensorRT the
gap closes to a fraction of a millisecond and the INT8 engine is now the more
accurate of the two.  The `pallet` class carries most of the strict-IoU loss;
[`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) has the per-class numbers.

---

## TensorRT, measured on a desktop RTX 4060 (not a Jetson)

The repository's own TensorRT path, run for real: `scripts/export_tensorrt.py`
built the engines from the FP32 ONNX graph inside NVIDIA's TensorRT container,
the INT8 one calibrated on the 256 training frames named in
`configs/export.yaml` and `scripts/evaluate.py` and
`scripts/benchmark_latency.py` scored and timed them through
`edgetrack.detect.trt_backend` with the same decoder, NMS, metric and loop as
every other row.  `make tensorrt-desktop` reproduces it; the table below is
generated from [`results/published/tensorrt.json`](results/published/tensorrt.json).

<!-- BEGIN generated:tensorrt -->
<!-- Generated by scripts/make_results_table.py.  Do not edit. -->

Measured on a desktop RTX 4060, not a Jetson.  Driver `591.86`, CUDA `12.9.1.010`, TensorRT `10.11.0.33`, image `nvcr.io/nvidia/tensorrt:25.06-py3` under WSL2 Docker.  GPU contention: no other compute process and at most 5 percent utilisation before and after the run; host nvidia-smi, sampled every 5 s through the run, listed no other compute process.  CPU not busy: load average 0.29 before and 1.34 after, on 16 logical cores shared with other containers.  Accuracy on the test split, 1440 images; latency over 300 iterations after 30 warmup.

| variant | precision | mAP@0.5 | mAP@0.5:0.95 | vs ONNX Runtime FP32 | engine | inference p50 | inference p95 | end to end p50 | end-to-end FPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tensorrt-fp32` | fp32 | 0.9775 | 0.8605 | -0.00004 | 30841 KiB | 2.40 ms | 3.16 ms | 5.25 ms | 170.2 |
| `tensorrt-fp16` | fp16 | 0.9776 | 0.8603 | -0.00018 | 13832 KiB | 1.66 ms | 2.47 ms | 4.56 ms | 196.8 |
| `tensorrt-int8` | int8 | 0.9760 | 0.8197 | -0.04082 | 9738 KiB | 1.72 ms | 2.39 ms | 5.22 ms | 184.4 |
<!-- END generated:tensorrt -->

**TensorRT closes the launch-bound gap, as predicted.**  The FP32 engine runs
inference in 2.40 ms at p50 where eager PyTorch on the same GPU takes 10.65 ms
and ONNX Runtime on the CPU 18.00 ms, with the same accuracy as the ONNX graph
to within 0.00004 mAP@0.5:0.95.  FP16, slower than FP32 on both eager backends,
is faster here at 1.66 ms and costs nothing measurable in accuracy.

**INT8 through TensorRT's calibrator is still the wrong trade on this GPU.**
It costs 0.04082 mAP@0.5:0.95 against the ONNX Runtime FP32 graph and is no
faster than FP16 at batch 1 (1.72 ms against 1.66 ms).  Implicit quantisation
lets TensorRT quantise the detection head as well, which is the failure the
ONNX Runtime ablation measures when the head is quantised on purpose.  Keeping
the head in higher precision on TensorRT is the obvious next experiment; it
has not been run.

**The detector is no longer the budget.**  With inference near 2 ms, letterbox,
decode, NMS and ByteTrack on the CPU make up most of the 4.56 to 5.25 ms end to
end.  None of these figures is a Jetson figure: [`docs/jetson.md`](docs/jetson.md)
says which would carry over.

---

## Run it

```bash
sh scripts/setup.sh     # venv, runtime dependencies, published weights
make demo               # detect, track, score against truth, write a video
```

or, with nothing installed but Docker:

```bash
docker compose -f docker/docker-compose.yml up
```

Both fetch the published weights from the GitHub Release, generate a sequence
the model has never seen, run the whole pipeline over it and print real
metrics:

```
synthetic sequence, seed 7, palette depot
  120 frames in 5.58s = 21.5 FPS (onnxruntime/fp32)
  detections 704, unique track ids 8, per-frame p50 24.55 ms
  MOTA 0.9651  IDF1 0.9332  IDsw 1  FP 0  FN 24
  switch causes: {"detector_gap": 0, "occlusion": 1, "crossing": 0, ...}
```

Nothing there is a fixture.  The scene is generated from a seed, the ground
truth is regenerated from the same seed rather than cached and the metrics are
computed by the same code that produced every number in the docs.

`make help` lists the rest.

---

## What is in here

| | |
| --- | --- |
| **Custom dataset** | A procedural warehouse scene with exact box *and identity* ground truth, plus per-object visibility.  3240 frames, 87 tracked identities, regenerable from a seed in about a minute.  [`data/DATASET_CARD.md`](data/DATASET_CARD.md) |
| **Detector** | YOLOX-Tiny, Apache-2.0, fine-tuned to 0.8605 mAP@0.5:0.95 on a held-out split rendered in a different palette.  [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) |
| **ByteTrack** | Implemented from the paper, not vendored.  Two-pass IoU association keeping the low-confidence detections. |
| **DeepSORT** | Implemented from the paper.  Matching cascade, Mahalanobis gating, appearance gallery. |
| **Appearance** | A 0.46 MB re-identification network trained on scene crops, exported to ONNX, plus a parameter-free HSV histogram baseline that beats it. |
| **Metrics** | CLEAR-MOT and the ID measures implemented here and cross-checked against `py-motmetrics` on randomised sequences; detection mAP cross-checked against pycocotools on the real model. |
| **ID-switch analysis** | Every switch attributed to a cause using ground-truth visibility, neighbour overlap and speed. |
| **Export** | FP32 and FP16 ONNX, three INT8 variants and TensorRT engines at FP32, FP16 and INT8, built and measured on a desktop GPU in NVIDIA's container. |
| **ROS2** | A real `rclpy` node publishing `vision_msgs` detections and tracks, verified in CI by a real subscriber inside a Humble container. |

---

## Design decisions worth arguing with

**The dataset is generated.**  Hand-labelling identities through occlusions is
the expensive part of a real MOT set and it is the part this project needs
most: MOTA, IDF1 and the ID-switch attribution are all measured against exact
truth here rather than against another model's output.  The cost is that no
absolute accuracy number transfers to a real camera, which
[`docs/limitations.md`](docs/limitations.md) states first and at length.

**Both trackers were written rather than imported.**  The reference DeepSORT is
GPL-3.0 and would have made this repository GPL; the ByteTrack in Ultralytics is
AGPL.  Writing both also meant they could share one Kalman filter and one track
lifecycle, so the difference between their results is a difference in
association and nothing else.

**The export is the training network, rearranged rather than approximated.**
The RGB 0..1 input is folded into the stem convolution's weights and the grid
decode is built into the graph, so every backend keeps the same preprocessing
and the same decoder it had before the swap.  On Megvii's COCO weights the
folded module and stock YOLOX agree to within float rounding.

**The runtime imports neither torch nor yolox.**  Checked from the inside by
importing every runtime module in a fresh interpreter and inspecting
`sys.modules` and from the outside by running `import torch` inside the built
Docker image and failing if it succeeds.  With an Apache-2.0 detector that is a
size boundary rather than a licence one: torch is two gigabytes the deployed
image has no use for.

**Tracker parameters were tuned on val, never on test and re-tuned for the
new detector.**  The spread between the best and worst settings in the sweep is
up to 0.04029 IDF1, which is larger than the gap between the two trackers, so
an untuned comparison would have measured the tuning.

**Every number in the documentation is checked.**  `scripts/check_docs.py`
fails the build if a figure appears in a document that is not in
`results/published/` and the generated tables are compared byte for byte
against what the generator produces.  CI runs it on every push.

---

## Layout

```
configs/            data, model, train, export, benchmark, one file per tracker
src/edgetrack/
  data/             the scene generator and the on-disk layouts
  detect/           Detector interface, ONNX, PyTorch and TensorRT backends
                    and the YOLOX export module (training side only)
  track/            Kalman filter, matching, ByteTrack, DeepSORT, appearance
  metrics/          detection AP, CLEAR-MOT, ID measures, switch attribution
  ros2/             payload builders (no ROS needed) and the node
  pipeline.py       detector + encoder + tracker, one code path for everything
scripts/            one entry point per stage, all with --help
tests/              280 tests, no network, no GPU, no weights needed for most
docker/             runtime, training, desktop TensorRT and Jetson images
benchmarks/         results.csv and the still empty Jetson table
docs/               results, tracking, jetson, licences, limitations, cards
LICENSES/           the Apache-2.0 text the detector weights ship under
results/published/  every number quoted anywhere, as JSON
```

This differs from the original plan in one way: the plan put
`train.py`, `evaluate.py` and friends inside `src/`.  They are in `scripts/`
here so that `src/edgetrack` is an importable library with no entry points in
it, which is what lets `tests/test_runtime_boundary.py` assert that importing
the runtime pulls in nothing from the training toolchain.

---

## Reproducing every number

```bash
sh scripts/setup.sh --train    # adds torch and yolox
make data                      # ~1 minute, deterministic from seeds
make train                     # ~30 minutes on an RTX 4060
make reid export quantise      # ~2 minutes
make experiments               # accuracy, tracking, latency
make publish                   # into results/published, regenerate the tables
make tensorrt-desktop          # TensorRT engines on an NVIDIA GPU, in Docker
make compare                   # the YOLOv8n numbers beside the new ones
make check                     # lint, tests and the documentation check
```

---

## Limitations

The full list is [`docs/limitations.md`](docs/limitations.md), ordered by how
much it should change your reading.  The three that matter most:

1. **No Jetson.**  Every latency figure is from a desktop, the TensorRT ones
   included: the engines were built and measured on an RTX 4060.  Jetson
   numbers remain a hardware TODO in [`STATUS.md`](STATUS.md).
2. **Generated data.**  Absolute accuracy transfers nowhere.  The *differences*
   between precisions are measured on identical images and do transfer.
3. **ONNX Runtime's CUDA provider does not work on the host.**
   `onnxruntime-gpu` needs a CUDA 13 runtime and the torch build ships CUDA
   12.6, so ONNX rows are CPU-only and the GPU rows come from PyTorch.
   Publishing a GPU row that was secretly a CPU row would have been easier.

---

## Licence

The code is MIT, in [`LICENSE`](LICENSE).  The detector weights are
**Apache-2.0**: they are fine-tuned from Megvii's Apache-2.0 YOLOX-Tiny COCO
checkpoint with Megvii's Apache-2.0 training code, so closed commercial use is
permitted.  Redistributing them means shipping the licence text,
[`LICENSES/YOLOX-Apache-2.0.txt`](LICENSES/YOLOX-Apache-2.0.txt), with the
attribution and saying the files were modified.
[`docs/licences.md`](docs/licences.md) has the detail, the one point worth
confirming with counsel and the history: the YOLOv8n weights in Release
`v0.1.0` still carry their AGPL-3.0 obligation and nothing here uses them.

# Licences

This repository is MIT.  The detector and the weights it ships are Apache-2.0,
end to end, so nothing in the deployed system carries a copyleft obligation.
That was not always true and the history is kept below, because the reason the
detector changed is the useful part.

---

## The repository

MIT, in `LICENSE`.  Everything under `src/`, `scripts/`, `configs/`, `tests/`
and `docs/` is original work.

Neither tracker is vendored.  ByteTrack is implemented from Zhang et al. (ECCV
2022) and DeepSORT from Wojke et al. (ICIP 2017); the reference implementation
of DeepSORT is GPL-3.0 and copying it would have made this repository GPL.  The
CLEAR-MOT and ID metrics are implemented from Bernardin and Stiefelhagen (2008)
and Ristani et al. (2016) and cross-checked against `py-motmetrics` in the test
suite rather than derived from it.

---

## The detector: YOLOX-Tiny, Apache-2.0

The detector is YOLOX-Tiny (Ge et al., 2021) from Megvii's `yolox` package,
fine-tuned from Megvii's COCO checkpoint `yolox_tiny.pth`.

| Part | Licence | Where it comes from |
| --- | --- | --- |
| `yolox` 0.3.0, the training and export code | Apache-2.0, Copyright 2021-2022 Megvii Inc. | PyPI, `requirements-yolox.txt` |
| `yolox_tiny.pth`, the COCO starting point | Apache-2.0, as a release asset of the YOLOX repository | GitHub release `0.1.1rc0` of `Megvii-BaseDetection/YOLOX`; URL and SHA-256 in `configs/train.yaml` |
| `yolox_tiny_edgetrack.pt` and every ONNX graph and engine built from it | Apache-2.0, as a derivative of the above | `scripts/train.py`, `scripts/export_onnx.py`, `scripts/quantize_onnx.py` |

**Closed commercial use is permitted.**  Apache-2.0 asks three things of anyone
who redistributes the weights or a product containing them: a copy of the
licence, the attribution notices and a statement that the files were modified.
The licence text as Megvii ships it, with its copyright line, is
[`LICENSES/YOLOX-Apache-2.0.txt`](../LICENSES/YOLOX-Apache-2.0.txt).  The YOLOX
repository has no NOTICE file, so there is no NOTICE text to carry forward.
The fine-tuned weights are a modification of Megvii's: fine-tuned on the
dataset in this repository, with the input normalisation folded into the stem
convolution and the box decode built into the exported graph
(`edgetrack.detect.yolox_net`).  Apache-2.0 also carries an express patent
licence from the contributors, which MIT does not.

**One thing to check rather than assume.**  Megvii publishes the checkpoints as
release assets of an Apache-2.0 repository and states no separate terms for
them, so this repository reads them as Apache-2.0.  That is the plain reading
and the one the YOLOX community works on but it is a reading of what Megvii
published rather than a statement Megvii made about the weights specifically.
Anyone shipping a product on them should have that confirmed by their own
counsel, as they would for any third-party model.

**The training data behind the starting point.**  The COCO checkpoint was
trained by Megvii on COCO, whose annotations are CC BY 4.0 and whose images are
Flickr photographs under their own terms.  That is a provenance question every
COCO-pretrained detector shares, YOLOv8 included and it is not a licence on
the weights.  No COCO image or annotation is used, stored or redistributed by
this repository; fine-tuning uses only the generated dataset.

### Why YOLOX, rather than RTMDet or D-FINE

All three are Apache-2.0, code and weights.  YOLOX was chosen because it is the
one that fits this repository without changing anything it measures:

* **Same head contract.**  YOLOX is anchor-free on the stride 8, 16 and 32
  grids YOLOv8 used, so at 640x384 it has the same 5040 anchors and with the
  grid decode and `obj * cls` scoring done in the exported graph its output is
  the `(1, 4 + classes, 5040)` tensor `edgetrack.detect.base.decode_head`
  already consumed.  Every backend, the INT8 quantiser, the TensorRT builder
  and both trackers are unchanged.
* **ByteTrack's own detector.**  The ByteTrack paper used YOLOX.  A dense,
  NMS-based detector with a long tail of low-confidence boxes is what the
  second association pass is designed around; a set-prediction detector such
  as D-FINE emits a fixed number of queries and no such tail.
* **A plain convolutional graph.**  Conv, Sigmoid, Mul and Concat, which ONNX
  Runtime's static quantiser and TensorRT's INT8 calibrator both handle and
  which keeps the INT8 ablation comparable with the previous one.  D-FINE's
  deformable attention samples features with grid sampling, the classic
  failure case for implicit INT8 quantisation.
* **The lightest toolchain.**  `yolox` is pure PyTorch and installs from PyPI.
  RTMDet lives in MMDetection, which needs `mmcv` with compiled operators built
  against a specific torch and CUDA pair, plus `mmdeploy` for the export.

What it costs: YOLOX-Tiny is the nearest size class to YOLOv8n but not the
same one.  It has 5,033,739 parameters against 3,006,428 and the ONNX
graph is larger.  [`MODEL_CARD.md`](MODEL_CARD.md) and the README have the
measured differences, including where the new detector is worse.

---

## History: Ultralytics YOLOv8, AGPL-3.0

Until October 2026 the detector was Ultralytics YOLOv8n, fine-tuned from
Ultralytics' `yolov8n.pt`.  Ultralytics distributes the library and its
pretrained weights under AGPL-3.0 and its stated position is that models
trained or fine-tuned with them fall under AGPL-3.0 unless the user holds an
Ultralytics Enterprise Licence.  The fine-tuned weights therefore carried an
AGPL-3.0 obligation, which blocked closed commercial use; keeping the library
out of the runtime image did not change that, because the obligation travels
with the weights inside the ONNX graph.  Replacing the detector was the fix.

The YOLOv8 artefacts are still attached to GitHub Release `v0.1.0`, where they
were published and remain under that obligation.  Nothing in the current code
or results uses them.  `ultralytics` stays on the forbidden-import list in
`tests/test_runtime_boundary.py` and in the `docker` CI job, so it cannot come
back by accident.

---

## The runtime boundary

The runtime loads an ONNX graph through ONNX Runtime, or a plan file through
TensorRT and imports neither torch nor yolox.  With an Apache-2.0 detector that
is no longer a licence boundary; it is a size boundary, because torch is two
gigabytes of wheel the deployed image has no use for.  It is enforced rather
than asserted:

* `requirements.txt` contains neither torch nor yolox and
  `tests/test_runtime_boundary.py` fails if either appears.
* Every runtime module is imported in a fresh interpreter and `sys.modules` is
  checked afterwards; `edgetrack.detect.yolox_net`, which imports torch, has
  its own test that the pipeline cannot reach it.
* `docker/Dockerfile.runtime` builds without them and the `docker` job in CI
  runs `import torch`, `import yolox` and `import ultralytics` inside the built
  image and fails if any succeeds.

---

## Everything else

| Dependency | Licence | Where |
| --- | --- | --- |
| ONNX Runtime | MIT | runtime |
| ONNX | Apache-2.0 | export |
| onnxslim | MIT | export, graph simplification |
| NumPy | BSD-3-Clause | runtime |
| SciPy | BSD-3-Clause | runtime, for the Hungarian solver |
| OpenCV (opencv-python-headless) | Apache-2.0 | runtime |
| PyYAML | MIT | runtime |
| imageio | BSD-2-Clause | the README GIF |
| tqdm | MPL-2.0 and MIT | runtime |
| PyTorch, torchvision | BSD-3-Clause | training and export only |
| pycocotools | BSD-2-Clause | YOLOX's evaluator during training and a test cross-check |
| loguru, tabulate, thop | MIT | YOLOX's import-time dependencies, training only |
| psutil | BSD-3-Clause | YOLOX's import-time dependency, training only |
| TensorBoard | Apache-2.0 | YOLOX's training log |
| py-motmetrics | MIT | tests only |
| TensorRT | NVIDIA proprietary | Jetpack on a Jetson; NVIDIA's container (`docker/Dockerfile.tensorrt`) and the pinned PyPI wheel (`requirements-tensorrt.txt`) on a desktop and in CI |
| pycuda | MIT | the TensorRT backend and the INT8 calibrator |

TensorRT is the one to watch.  It is proprietary and governed by NVIDIA's
licence, which is why nothing here bundles it: the Jetson image takes it from
the Jetpack base, the desktop image takes it from NVIDIA's own
`nvcr.io/nvidia/tensorrt` container and CI installs NVIDIA's wheel at run time.
Engines built with it are never committed.

---

## The data

The dataset is generated by `src/edgetrack/data/synth.py` from seeds in
`configs/data.yaml`.  It contains no photographs, no third-party imagery and no
person, so there is no dataset licence to inherit and no personal data to
handle.  `data/DATASET_CARD.md` says the same thing at more length and is
honest about what that costs in realism.

# Licences

This repository is MIT.  The interesting part is the boundary that keeps it
that way, because the tool that trains the detector is not.

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

## Ultralytics, which is AGPL-3.0

Ultralytics YOLOv8 is AGPL-3.0.  It is used here to fine-tune the detector and
to export the ONNX graphs, and it appears in `requirements-train.txt` and in
`docker/Dockerfile.train`.

**The published detector weights carry an AGPL-3.0 obligation and that blocks
closed commercial use.**  Fine-tuning starts from Ultralytics' pretrained
`yolov8n.pt` (`configs/model.yaml`), which Ultralytics distributes under
AGPL-3.0.  The fine-tuned checkpoint and every graph exported from it
(`yolov8n_edgetrack.pt`, the FP32, FP16 and INT8 ONNX files in the Release and
any TensorRT engine built from them) are derived from those weights.
Ultralytics' stated position is that models trained or fine-tuned with its
software and weights fall under AGPL-3.0 unless the user holds an Ultralytics
Enterprise Licence.  An earlier version of this page argued the opposite, that
the weights are mere program output and said Ultralytics agreed.  That was
wrong on the second point and is not a safe basis for the first, so it has
been withdrawn.

What that means in practice:

* Anyone deploying these weights, including over a network, must meet
  AGPL-3.0: the complete corresponding source of the deployed system, under
  AGPL-3.0, to every user who interacts with it.
* A closed or commercial deployment needs either an Ultralytics Enterprise
  Licence or a detector with no Ultralytics lineage.
* The MIT licence on this repository covers the code written here.  It does
  not relicense the weights and cannot.

The fix is a detector with a permissive licence end to end, weights included:
YOLOX, RTMDet or D-FINE, all Apache-2.0.  `STATUS.md` carries it as an open
item.  It has not been done in this pass, so the detector and its numbers are
unchanged.

**What the runtime boundary does and does not achieve.**  The runtime loads an
ONNX graph through ONNX Runtime, or a plan file through TensorRT, and imports
neither ultralytics nor torch.  That keeps Ultralytics' *code* out of the
deployed image, which is worth having: it is the part a swap to an Apache-2.0
detector would not need to touch.  It does not remove the obligation attached
to the weights, which travel inside the ONNX graph whichever runtime loads it.

The code boundary is enforced rather than asserted:

* `requirements.txt` contains neither torch nor ultralytics, and
  `tests/test_runtime_boundary.py` fails if either appears.
* Every runtime module is imported in a fresh interpreter and `sys.modules` is
  checked afterwards, so an accidental import anywhere in the runtime path
  fails the suite.
* `docker/Dockerfile.runtime` builds without them and the `docker` job in CI
  runs `import torch` and `import ultralytics` inside the built image and fails
  if either succeeds.

The swap is contained.  The only Ultralytics-specific steps are training and
`model.export(format="onnx")`.  A detector whose graph emits a `(1, 4 +
num_classes, anchors)` head is consumed without modification, because
`edgetrack.detect.base.decode_yolov8` decodes that shape and nothing above it
knows which trainer produced it.  YOLOX and RTMDet emit different heads, so a
second decoder in `edgetrack.detect.base` is the likely size of the change.

---

## Everything else

| Dependency | Licence | Where |
| --- | --- | --- |
| ONNX Runtime | MIT | runtime |
| ONNX | Apache-2.0 | export |
| NumPy | BSD-3-Clause | runtime |
| SciPy | BSD-3-Clause | runtime, for the Hungarian solver |
| OpenCV (opencv-python-headless) | Apache-2.0 | runtime |
| PyYAML | MIT | runtime |
| imageio | BSD-2-Clause | the README GIF |
| tqdm | MPL-2.0 and MIT | runtime |
| PyTorch | BSD-3-Clause | training only |
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

The pretrained COCO checkpoint that fine-tuning starts from is distributed by
Ultralytics under AGPL-3.0; `scripts/train.py` downloads it on first run.  It is
not redistributed here as such but the fine-tuned weights in the Release are
derived from it, which is the obligation described above.  COCO itself is CC BY
4.0 for the annotations, with the images under their original Flickr terms.
No COCO image or annotation appears in this repository.

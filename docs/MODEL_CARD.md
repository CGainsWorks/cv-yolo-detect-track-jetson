# Model card: yolox_tiny_edgetrack v2

A YOLOX-Tiny detector fine-tuned on a procedurally generated warehouse scene,
plus a small appearance embedding used by the DeepSORT branch.  Version 2
replaces the YOLOv8n detector of version 1, which carried an AGPL-3.0
obligation; both the code and the starting weights are now Apache-2.0
(`docs/licences.md`).

**Trained on generated data.**  It detects rendered rectangles, not real
forklifts.  Read `docs/limitations.md` before drawing any conclusion about
real-world performance from a number on this page.

---

## The detector

| | |
| --- | --- |
| Architecture | YOLOX-Tiny (YOLOX's tiny depth and width multipliers), 5,033,739 parameters, 9.14 GFLOPs at 640x384 by YOLOX's own count |
| Starting point | Megvii's COCO checkpoint `yolox_tiny.pth`, Apache-2.0, digest in `configs/train.yaml` |
| Tensors loaded from it | all but the three class-prediction convolutions, which are re-initialised for four classes |
| Classes | person, forklift, pallet, cone |
| Training input | 640x384, the native aspect, with YOLOX's multi-scale jitter of five strides either side |
| Inference input | 640x384 |
| Training data | 1440 frames over 60 sequences, 8184 boxes |
| Epochs | 60, the last 10 without mosaic |
| Batch | 32, mixed precision |
| Optimiser | YOLOX's SGD with Nesterov momentum 0.9, 0.01 per 64 images, warm cosine schedule, 3 warmup epochs, EMA |
| Trainer | `yolox.core.Trainer` 0.3.0, driven by `scripts/train.py` |
| Hardware | RTX 4060 |
| Training time | 1710.0 s, with other jobs sharing the CPU, so not a benchmark |
| Artefacts | `.pt`, FP32 ONNX, FP16 ONNX, three INT8 ONNX variants |

Unlike version 1, training and inference both run at the native aspect: YOLOX's
multi-scale jitter keeps the aspect ratio, so there is no square resize and no
grey bars at either end.

**What the export changes and why it is exact.**  YOLOX takes BGR in 0..255;
the runtime feeds RGB in 0..1, as every backend and the INT8 calibrator
already do.  `edgetrack.detect.yolox_net.fold_input_into_stem` scales the
first convolution's weights by 255 and reverses their colour order, which is
the same arithmetic in a different order.  The grid decode and the `obj * cls`
scoring are built into the graph, so the output is `(1, 8, 5040)`, the layout
`edgetrack.detect.base.decode_head` reads.  On Megvii's COCO weights the folded
module and stock YOLOX agree to within float rounding.

### Augmentation and one deliberate choice

Mosaic, a horizontal flip, ten per cent translation and a scale of 0.6 to 1.4.
No rotation, no shear, no mixup and **no colour jitter**.

Instance colour is the only appearance cue the DeepSORT branch has on this
dataset.  YOLOX's HSV augmentation has fixed gains (five hue steps, thirty of
saturation and value) that would teach the detector to ignore exactly the
signal the tracker depends on and a detector that is invariant to the thing
the tracker needs is a pipeline arguing with itself.  Version 1 turned
Ultralytics' jitter down hard for the same reason; YOLOX's cannot be turned
down without patching it, so it is off.

### Accuracy

| Split | mAP@0.5 | mAP@0.5:0.95 | P | R |
| --- | ---: | ---: | ---: | ---: |
| val (same palette as train) | 0.99391 | 0.90661 | | |
| test (shifted palette) | 0.9776 | 0.8605 | 0.9540 | 0.9655 |

The val row is YOLOX's own evaluator, which is pycocotools; the test row is
this repository's metric implementation and `tests/test_backends.py` feeds the
same predictions to both and requires them to agree within a stated tolerance.

**Against version 1.**  YOLOv8n scored 0.8649 mAP@0.5:0.95 on the same test
split and YOLOX-Tiny scores 0.8605: **slightly worse at the strict IoU
thresholds**, by 0.00446 and slightly better at mAP@0.5, 0.9776 against
0.9727.  On val YOLOX-Tiny is ahead, 0.90661 against 0.89251, so the gap opens
under the palette shift.

Per class on test, mAP@0.5:0.95:

| Class | FP32 | INT8 | Delta |
| --- | ---: | ---: | ---: |
| person | 0.92745 | 0.89878 | -0.02867 |
| forklift | 0.90504 | 0.87294 | -0.0321 |
| pallet | 0.77845 | 0.75657 | -0.02188 |
| cone | 0.83101 | 0.80886 | -0.02215 |

`pallet` is now the weakest class, at 0.78; under YOLOv8n it was `cone`.
Pallets are flat, low and often partly hidden behind forklifts and the
strict-IoU loss against version 1 is almost all here.  `cone`, the smallest
class, improved.  The INT8 loss is spread evenly across classes rather than
concentrated on the smallest one, as it was before.

---

## The precision variants

| Variant | How it is made | Size | mAP@0.5:0.95 |
| --- | --- | ---: | ---: |
| FP32 | ONNX export at opset 13 | 20.30 MB | 0.8605 |
| FP16 | separate export from a half-precision module | 10.21 MB | 0.8590 |
| INT8 | static QDQ from the FP32 graph, convolutions only, head in float | 5.52 MB | 0.8343 |
| INT8 all-conv | as above with the head quantised too | 5.53 MB | 0.8206 |
| INT8 all-ops | ONNX Runtime's defaults | 5.58 MB | 0.0000 |

INT8 calibration uses 256 training frames drawn with a fixed seed, through
exactly the preprocessing the deployed model applies.  The head left in float
is YOLOX's nine output convolutions, class, box and objectness on each of three
strides, found by name in the graph.  `docs/results.md` section 1 explains why
the last row is what it is and why it is shipped rather than deleted.

**INT8 costs more than it did.**  0.0262 mAP@0.5:0.95 with the head in float,
against about a point for YOLOv8n (0.8549 against 0.86495).  The files are larger too: YOLOX-Tiny has more
parameters, so every variant is about two thirds larger than its YOLOv8n
counterpart.

FP16 is a sibling export rather than a conversion of the FP32 graph; the
docstring of `scripts/export_onnx.py` has the detail.

---

## The appearance embedding

Unchanged from version 1: it is trained on ground-truth crops and does not
depend on the detector.

| | |
| --- | --- |
| Architecture | four stride-2 convolutions, 16 to 128 channels, global average pool, 128-dimensional bottleneck with a BNNeck |
| Parameters | 162,192 total, 114,448 in the embedding |
| Input | 64x128 BGR crop, normalised to [-1, 1] |
| Output | 128 dimensions, L2 normalised |
| Training | 7827 crops over 373 identities, 20 epochs, cross-entropy with 0.1 label smoothing |
| Training time | 8.7 s |
| Val rank-1 | 0.9775 |
| Val mAP | 0.8586 |
| Artefact | 0.46 MB ONNX, opset 12, dynamic batch |

An identity is a `(sequence, instance)` pair, so two objects in different
sequences are different identities.  Crops come from ground-truth boxes rather
than detector output, so the evaluation measures the embedding rather than the
detector.

It is small because its cost multiplies by object count whilst the detector's
does not; `docs/tracking.md` has the sizing argument and the numbers for the
larger version that was tried first.

**It does not beat a colour histogram at the job it was built for.**  On the
test split, DeepSORT with this embedding makes 33 identity switches and
DeepSORT with a parameter-free HSV histogram makes 19.  That is a domain-gap
result and it is written up in `docs/tracking.md` rather than buried.

---

## Intended use

Demonstrating and benchmarking an edge detection-and-tracking pipeline on the
dataset it was trained on.  Nothing else.

## Out of scope

Any real camera.  Any safety-relevant decision.  Any claim about people: the
`person` class detects a rendered figure of a person and has never seen a
photograph during fine-tuning.

## Ethical considerations

The fine-tuning data contains no photographs, no personal data and no real
people, so there is no demographic bias in it to report because there is no
demographic in it.  The COCO checkpoint fine-tuning starts from was trained by
Megvii on photographs, including of people and inherits whatever biases COCO
carries; nothing here measures them.  That is not a claim of fairness; a model
trained this way has simply never been tested on the question and deploying it
anywhere near real people would require starting the analysis from scratch on
real data.

## Licence

The repository's code is MIT.  The detector weights are **Apache-2.0**: they
are a modification of Megvii's Apache-2.0 YOLOX-Tiny COCO checkpoint, made with
Megvii's Apache-2.0 training code and closed commercial use is permitted.
Redistributing them means including the licence text,
`LICENSES/YOLOX-Apache-2.0.txt` and the attribution to Megvii and saying that
the files were modified; `docs/licences.md` has the detail and the one point
worth confirming with counsel.

## Provenance

| Artefact | SHA-256 |
| --- | --- |
| `yolox_tiny.pth`, the starting point | in `configs/train.yaml`, checked before training |
| `yolox_tiny_edgetrack_fp32.onnx` | in `models/yolox_tiny_edgetrack_fp32.json` |
| `yolox_tiny_edgetrack_fp16.onnx` | in `models/yolox_tiny_edgetrack_fp16.json` |
| INT8 variants | in `models/quantisation.json` |
| `reid_edgetrack.onnx` | in `models/reid_edgetrack.json` |

`models/MANIFEST.json` carries the same digests and `scripts/fetch_model.py`
refuses a download that does not match, so a truncated file fails at fetch time
rather than as a strange accuracy number three steps later.

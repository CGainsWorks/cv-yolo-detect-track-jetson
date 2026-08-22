# Model card: yolov8n_edgetrack v1

A YOLOv8n detector fine-tuned on a procedurally generated warehouse scene, plus
a small appearance embedding used by the DeepSORT branch.

**Trained on generated data.**  It detects rendered rectangles, not real
forklifts.  Read `docs/limitations.md` before drawing any conclusion about
real-world performance from a number on this page.

---

## The detector

| | |
| --- | --- |
| Architecture | YOLOv8n, 3,006,428 parameters, 4.9 GFLOPs |
| Starting point | COCO-pretrained `yolov8n.pt` |
| Classes | person, forklift, pallet, cone |
| Training input | 640x640 square, mosaic augmentation |
| Inference input | 640x384, the native aspect of the data |
| Training data | 1440 frames over 60 sequences |
| Epochs | 60 |
| Batch | 32 |
| Optimiser | Ultralytics `auto`, lr0 0.01, cosine schedule, 3 warmup epochs |
| Hardware | RTX 4060 |
| Training time | 759.2 s |
| Artefacts | `.pt`, FP32 ONNX, FP16 ONNX, three INT8 ONNX variants |

Training runs square and inference runs at the native aspect because mosaic
augmentation and the pretrained weights expect a square, whilst padding a 5:3
frame into a square at inference wastes a third of the compute on grey bars.
YOLOv8 is fully convolutional so the two are compatible; both sides just have
to stay multiples of 32.

### Augmentation, and one deliberate choice

The schedule is the Ultralytics default with the colour jitter turned down
hard: `hsv_s` and `hsv_v` at 0.3, `hsv_h` at 0.005.

Instance colour is the only appearance cue the DeepSORT branch has on this
dataset.  An aggressive HSV augmentation would teach the detector to ignore
exactly the signal the tracker depends on, and a detector that is invariant to
the thing the tracker needs is a pipeline arguing with itself.
`tests/test_configs.py` fails if somebody raises it without reading this.

### Accuracy

| Split | mAP@0.5 | mAP@0.5:0.95 | P | R |
| --- | ---: | ---: | ---: | ---: |
| val (same palette as train) | 0.99042 | 0.89251 | 0.97948 | 0.9763 |
| test (shifted palette) | 0.9727 | 0.8649 | 0.9597 | 0.9546 |

The val row is Ultralytics' own validator; the test row is this repository's
metric implementation, and `tests/test_backends.py` compares them on the same
weights with a stated tolerance rather than assuming they agree.

Per class on test, mAP@0.5:0.95:

| Class | FP32 | INT8 | Delta |
| --- | ---: | ---: | ---: |
| person | 0.9305 | 0.92754 | -0.00296 |
| forklift | 0.90837 | 0.90071 | -0.00766 |
| pallet | 0.83831 | 0.82784 | -0.01047 |
| cone | 0.78261 | 0.7635 | -0.01911 |

The smallest class takes the largest quantisation hit, which is the usual
pattern: a small object's activations occupy a smaller part of the calibrated
range.

`cone` is the weakest class in absolute terms too, at 0.78 against `person` at
0.93.  Cones are the smallest objects in the scene and the most often fully
occluded.

---

## The precision variants

| Variant | How it is made | Size | mAP@0.5:0.95 |
| --- | --- | ---: | ---: |
| FP32 | ONNX export at opset 13 | 12.20 MB | 0.8649 |
| FP16 | separate export from a half-precision module | 6.14 MB | 0.8615 |
| INT8 | static QDQ from the FP32 graph, convolutions only, head in float | 3.40 MB | 0.8549 |
| INT8 all-conv | as above with the head quantised too | 3.38 MB | 0.8193 |
| INT8 all-ops | ONNX Runtime's defaults | 3.41 MB | 0.0000 |

INT8 calibration uses 256 training frames drawn with a fixed seed, through
exactly the preprocessing the deployed model applies.  `docs/results.md`
section 1 explains why the last row is what it is, and why it is shipped rather
than deleted.

FP16 is a sibling export rather than a conversion of the FP32 graph, because
`onnxconverter_common` produces an invalid model on this network; the docstring
of `scripts/export_onnx.py` has the detail.

---

## The appearance embedding

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
test split, DeepSORT with this embedding makes 32 identity switches and
DeepSORT with a parameter-free HSV histogram makes 19.  That is a domain-gap
result and it is written up in `docs/tracking.md` rather than buried.

---

## Intended use

Demonstrating and benchmarking an edge detection-and-tracking pipeline on the
dataset it was trained on.  Nothing else.

## Out of scope

Any real camera.  Any safety-relevant decision.  Any claim about people: the
`person` class detects a rendered figure of a person and has never seen a
photograph.

## Ethical considerations

The training data contains no photographs, no personal data and no real people,
so there is no demographic bias in the training set to report because there is
no demographic in it.  That is not a claim of fairness; a model trained this way
has simply never encountered the question, and deploying it anywhere near real
people would require starting the analysis from scratch on real data.

## Licence

The weights are MIT, as is the repository.  Fine-tuning started from an
Ultralytics AGPL-3.0 checkpoint and the reasoning about what that implies, along
with the runtime boundary that enforces it, is in `docs/licences.md`.

## Provenance

| Artefact | SHA-256 |
| --- | --- |
| `yolov8n_edgetrack_fp32.onnx` | in `models/yolov8n_edgetrack_fp32.json` |
| `yolov8n_edgetrack_fp16.onnx` | in `models/yolov8n_edgetrack_fp16.json` |
| INT8 variants | in `models/quantisation.json` |
| `reid_edgetrack.onnx` | in `models/reid_edgetrack.json` |

`models/MANIFEST.json` carries the same digests and `scripts/fetch_model.py`
refuses a download that does not match, so a truncated file fails at fetch time
rather than as a strange accuracy number three steps later.

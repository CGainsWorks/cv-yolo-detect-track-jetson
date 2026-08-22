# Dataset card: edgetrack-warehouse-synth v1

A procedurally generated multi-object tracking set of a warehouse floor, with
exact box and identity ground truth.  Four classes, 3240 frames, 87 tracked
identities in the test split.

**It is generated, not photographed.**  A detector trained on it will not
transfer to a real camera and the absolute mAP is not comparable to anything.
Read `docs/limitations.md` before quoting a number from it.

---

## Provenance

Everything comes from `src/edgetrack/data/synth.py`, driven by seeds recorded
in `configs/data.yaml`.  Regenerate it with:

```bash
python scripts/make_dataset.py --preset full
```

That takes about a minute and produces byte-identical files on the same OpenCV
version.  CI generates it twice in fresh processes and compares SHA-256
digests, so determinism is checked rather than assumed.  The counts below come
from `results/published/dataset.json`, which the generator writes.

---

## Composition

| Split | Sequences | Frames | Labelled boxes | Boxes dropped as occluded | Palette |
| --- | ---: | ---: | ---: | ---: | --- |
| train | 60 | 1440 | 8184 | 1896 | warehouse |
| val | 15 | 360 | 2023 | 497 | warehouse |
| test | 12 | 1440 | 8142 | 3378 | depot |

Class counts, labelled boxes only:

| Class | train | val | test |
| --- | ---: | ---: | ---: |
| person | 3333 | 847 | 3057 |
| forklift | 1556 | 367 | 2054 |
| pallet | 1639 | 406 | 1940 |
| cone | 1656 | 403 | 1091 |

Objects are small.  The median box covers 0.68 percent of the frame in train
and 0.67 percent in test, and the 5th percentile is about 0.11 percent, which
is roughly 30 by 25 pixels on a 640 by 384 frame.  That is deliberate: an edge
detector's hard cases are small and distant, not large and central.

Mean visibility of the labelled boxes runs from 0.919 in train down to 0.852 in
test, because the test split is rendered with three occluders rather than two.

---

## Ground truth, and the two thresholds

Every object carries a visibility ratio, computed exactly rather than
estimated: each object is composited into an integer id map in depth order, the
occluders are stamped over the top, and visibility is the fraction of the
object's own pixels that survive.

Two ground-truth files come out of that, and they disagree on purpose:

* **`labels/{split}/*.txt`**, YOLO format, contains only boxes at or above
  visibility 0.35.  Training a detector on a box that is ninety percent behind
  a shelf teaches it to hallucinate.
* **`mot/{split}/*/gt/gt.txt`**, MOTChallenge format, contains every box with
  its visibility in column nine.  The tracking evaluator needs to know an
  object was present even when no detector could have seen it, and it applies
  its own `--min-visibility` filter at scoring time.

`tests/test_dataset_io.py` pins the direction of that disagreement.

---

## The domain gap

Train and val share the `warehouse` palette, a noise sigma of 4.0 and a motion
blur scale of 0.55.  Test uses the `depot` palette, noise sigma 7.0, blur scale
0.85, harsher JPEG quality and one more occluder.

That gap is the reason the detector scores 0.893 mAP50-95 on val and 0.865 on
test.  Without it a fully synthetic set saturates, and a saturated detector
makes the accuracy cost of quantisation unmeasurable: the whole INT8 comparison
would be rounding error.

---

## How a scene is built

* A background: wall, perspective floor markings converging on a vanishing
  point, racking silhouettes and low-frequency multiplicative grime, so the
  floor is not a gradient fill the detector can key on.
* Two or three static occluders drawn in front of everything, which is what
  makes a track disappear entirely for a run of frames rather than merely
  shrink.
* Four or five moving objects (person, forklift) and three static ones (pallet,
  cone).  Each gets a per-instance colour, which is the only appearance cue the
  DeepSORT branch has.
* A crossing schedule: pairs of movers are aimed at a shared point partway
  through the sequence, so same-class ambiguity happens on purpose rather than
  by luck.
* Degradation in camera order: motion blur proportional to speed, sensor noise,
  ISP gain and bias, then JPEG compression at a random quality.

---

## Intended use, and what it cannot support

**Intended.**  Exercising and comparing a detection-and-tracking pipeline:
quantisation effects, tracker association quality, ID-switch attribution,
latency benchmarking.  All of those need identical inputs across conditions and
exact identity truth, and that is what this set provides cheaply.

**Not supported.**  Any claim about real-world accuracy.  Any claim about a
demographic, because there are no people in it, only rendered rectangles.  Any
transfer-learning claim: a model trained here has learnt this renderer.

---

## Using real data instead

The layout below `data/processed/` is the contract, and nothing above
`scripts/` knows where it came from:

```
images/{train,val,test}/SEQ_FRAME.jpg
labels/{train,val,test}/SEQ_FRAME.txt      class cx cy w h, normalised
mot/{val,test}/SEQ/gt/gt.txt               MOT16 columns, visibility in col 9
mot/{val,test}/SEQ/seqinfo.ini
```

A CVAT or Label Studio export in MOT format drops into `mot/`, and the YOLO
labels can be derived from it.  The converter is not written; that is stated in
`docs/limitations.md` rather than implied by a stub.

---

## Licence

The generator is MIT, along with the rest of the repository, and the data it
produces is MIT.  No third-party imagery is used and no photographs of people
exist anywhere in this repository, so there is no upstream dataset licence to
inherit and no personal data to handle.

The pretrained COCO checkpoint that fine-tuning starts from is a separate
matter and is covered in `docs/licences.md`.

# ByteTrack against DeepSORT

The headline is that ByteTrack wins here, on every headline metric, at about
seven tenths of the cost.  That is the opposite of what "the one with the
learned appearance model" suggests, so this page is about why and about what
the result does and does not generalise to.

All numbers come from `results/published/tracking*.json` and were measured
with the YOLOX-Tiny detector.  The YOLOv8n numbers this page used to quote are
in the git history; the comparison between the two detectors is in the README.

---

## The comparison as run

Both trackers share the Kalman filter in `edgetrack.track.kalman` and the track
bookkeeping in `edgetrack.track.base`, so the only difference between them is
association.  `scripts/eval_tracking.py` detects each frame once and feeds the
identical detection list to both, so detection quality is not a variable
either.  Parameters for both came from `scripts/tune_tracker.py` maximising
IDF1 on the **val** split, never on test, re-run after the detector changed:
ByteTrack's `track_thresh` moved from 0.5 to 0.6 and DeepSORT's best setting
did not move.

| | ByteTrack | DeepSORT (trained re-ID) |
| --- | ---: | ---: |
| MOTA | 0.89993 | 0.89538 |
| IDF1 | 0.90797 | 0.89544 |
| ID switches | 23 | 33 |
| Fragmentations | 55 | 99 |
| Mostly tracked | 84/87 | 82/87 |
| False positives | 347 | 304 |
| Misses | 445 | 515 |
| ms per frame, p50 | 22.59 | 31.98 |

---

## Why ByteTrack wins

**It keeps the weak detections.**  This is the whole idea of the paper and it
is worth 70 fewer misses here.  A detection below the confidence threshold is
still evidence that something is there and ByteTrack's second association pass
matches those against tracks that failed the first pass.  DeepSORT has no
equivalent stage: `det_thresh` discards weak detections before the cascade
sees them, which is why its miss count is higher and its fragmentation count is
nearly double.  YOLOX is the detector the ByteTrack paper was written around
and its `obj * cls` score has exactly the long low-confidence tail that pass
feeds on.

**Its recovery path is geometric and geometry is right most of the time.**
DeepSORT recovers a lost identity through the appearance cascade, gated by a
Mahalanobis test on the Kalman state.  That gate is the part that bites: after
enough missed frames the constant-velocity prediction is wrong by far more than
the covariance admits, the correct detection is gated out and appearance never
gets a vote.  `tests/test_tracking.py` contains a scene sitting on the good
side of that boundary, where DeepSORT keeps two identities through an occlusion
that makes ByteTrack swap them and the same test's docstring says plainly
where the boundary is.

**The scene has five moving objects.**  Appearance earns its cost when many
similar objects are in shot at once.  Five is not many.

---

## The result that changes the reading

Swap the trained embedding for a colour histogram and DeepSORT gets better:

| DeepSORT variant | MOTA | IDF1 | ID switches | ms/frame p50 |
| --- | ---: | ---: | ---: | ---: |
| trained re-ID embedding | 0.89538 | 0.89544 | 33 | 31.98 |
| colour histogram, no weights | 0.89808 | 0.90165 | 19 | 28.75 |

Nineteen switches against ByteTrack's twenty three, from a descriptor with no
learned parameters at all.  ByteTrack still has the better IDF1, 0.90797
against 0.90165; with the previous detector the two were essentially level.

The reason the embedding loses is a domain gap that the re-identification
metrics do not see.  The embedding is trained on crops from the **warehouse**
palette and evaluated, in `results/published/reid_train.json`, on held-out
identities from the same palette, where it scores 0.9775 rank-1 and 0.8586 mAP.
The test sequences use the **depot** palette with higher noise and heavier
blur.  A colour histogram degrades gracefully under a global colour shift
because it is a coarse summary of the same signal; a small convolutional
network trained on one palette can degrade sharply, because nothing forced it
to be robust to that shift.

Two lessons and the second is the one worth keeping.  A re-identification
benchmark measured on the training domain does not predict tracking performance
on a shifted one.  And a hand-built descriptor is a real baseline rather than a
placeholder, so it should be measured rather than assumed to be worse.

---

## How good is the association on its own

Rerun both on ground-truth boxes, filtered by the same visibility threshold the
evaluator uses, so detection error is removed entirely:

| Detections | Tracker | MOTA | IDF1 | ID switches | Fragmentations |
| --- | --- | ---: | ---: | ---: | ---: |
| model | bytetrack | 0.89993 | 0.90797 | 23 | 55 |
| model | deepsort | 0.89538 | 0.89544 | 33 | 99 |
| ground truth | bytetrack | 0.98453 | 0.95495 | 19 | 19 |
| ground truth | deepsort | 0.98379 | 0.95024 | 21 | 24 |

This is the number that says where to spend effort.  The ground-truth rows do
not depend on the detector and are unchanged by the swap.

Perfect detections leave 19 switches for ByteTrack and 21 for DeepSORT, so
about four fifths of ByteTrack's switch count and two thirds of DeepSORT's, is
association failing on cases the detector handled correctly.  Fragmentations
tell the opposite story: 55 falls to 19 and 99 falls to 24, so most of the
track interruptions are the detector dropping the object rather than the
tracker losing it.

MOTA moves from 0.900 to 0.985, which is a reminder that MOTA is dominated by
detection quality and is close to useless for comparing association
strategies.  IDF1 moves much less, from 0.908 to 0.955, which is why it is the
number the tuner optimises.

---

## The appearance encoders

| | Histogram | Trained re-ID |
| --- | --- | --- |
| Parameters | none | 162,192 total, 114,448 in the embedding |
| Descriptor | 72 dimensions, HSV in three horizontal stripes | 128 dimensions |
| Weights on disk | none | 0.46 MB ONNX |
| Val rank-1 | not applicable | 0.9775 |
| Val mAP | not applicable | 0.8586 |

The network is deliberately tiny and the sizing was a measurement rather than
a preference.  The first version used two convolutions per block and 32 to 256
channels, came to 1.2 M parameters and roughly 0.5 GFLOPs per crop.  The
detector is about 9 GFLOPs per *frame* and the encoder runs once per
*detection*, so at eight objects in shot that version would have cost nearly
half the detector's arithmetic again.  Shrinking it seven times cost 0.9 points
of rank-1, from 0.9861 to 0.9775, which is nothing next to what it saved.  The
sizing was done against the previous, smaller detector; the conclusion holds
with more margin against this one.

### Setting the gate from the margin

`scripts/train_reid.py` reports the cosine distance distribution and that is
what `max_cosine_distance` should be read against:

| | Value |
| --- | ---: |
| Within an identity, mean | 0.2779 |
| Within an identity, 95th percentile | 0.6843 |
| Between identities, mean | 0.9957 |
| Between identities, 5th percentile | 0.7478 |

The paper's default gate is 0.2.  On this embedding that sits below the
within-identity mean and would reject most of the true matches it exists to
recover.  `scripts/tune_tracker.py` picked 0.5 on the val split, which sits
between the within-identity 95th percentile and the between-identity 5th
percentile: about as far from both as it can get.  That the sweep and the
margin analysis agree is the kind of coincidence worth checking for, because
when they disagree one of them is measuring the wrong thing.

---

## Parameter sensitivity

`results/published/tuning_*.json` holds the full sweeps.

| Tracker | Settings tried | Best IDF1 | Worst IDF1 | Spread |
| --- | ---: | ---: | ---: | ---: |
| ByteTrack | 81 | 0.96709 | 0.94093 | 0.02616 |
| DeepSORT | 72 | 0.94957 | 0.90928 | 0.04029 |

Two and a half to four points of IDF1 between the best and worst settings,
which is larger than the gap between the two trackers.  That is the argument
for tuning on a validation split rather than reporting whatever the defaults
gave: a comparison between a tuned tracker and an untuned one measures the
tuning.  It is also why the sweep was re-run when the detector changed rather
than the old parameters carried over: the score distribution a threshold is
set against is a property of the detector.

Ties are common: eighty-one ByteTrack settings produce about a dozen distinct
scores, so the tie-break decides the published configuration more often than
the objective does.  It breaks towards fewer switches, then a larger
confirmation count, then a shorter buffer and that rule is written down in
`scripts/tune_tracker.py` rather than left to sort order.

---

## What would change the conclusion

* **More objects.**  Twenty people in a corridor rather than five in a
  warehouse bay and appearance starts to matter far more than it does here.
* **A worse detector.**  ByteTrack's advantage comes largely from its low-score
  pass, which needs a detector whose weak detections are still roughly in the
  right place.  A noisier detector would erode that.
* **Real footage.**  The colour histogram beats the trained embedding here
  because instances are rendered in distinct saturated colours.  On real people
  in similar clothing that ordering would very likely reverse.
* **Longer occlusions.**  Both trackers were tuned with a 30-frame buffer on
  sequences where objects reappear quickly.  A minute-long occlusion is a
  re-identification problem rather than a tracking one and neither tracker as
  configured would attempt it.

# ByteTrack against DeepSORT

The headline is that ByteTrack wins here, on every metric, at three quarters of
the cost.  That is the opposite of what "the one with the learned appearance
model" suggests, so this page is about why, and about what the result does and
does not generalise to.

All numbers come from `results/published/tracking*.json`.

---

## The comparison as run

Both trackers share the Kalman filter in `edgetrack.track.kalman` and the track
bookkeeping in `edgetrack.track.base`, so the only difference between them is
association.  `scripts/eval_tracking.py` detects each frame once and feeds the
identical detection list to both, so detection quality is not a variable
either.  Parameters for both came from `scripts/tune_tracker.py` maximising
IDF1 on the **val** split, never on test.

| | ByteTrack | DeepSORT (trained re-ID) |
| --- | ---: | ---: |
| MOTA | 0.91257 | 0.90238 |
| IDF1 | 0.91194 | 0.8946 |
| ID switches | 24 | 32 |
| Fragmentations | 45 | 67 |
| Mostly tracked | 82/87 | 79/87 |
| False positives | 206 | 174 |
| Misses | 482 | 589 |
| ms per frame, p50 | 19.58 | 25.85 |

---

## Why ByteTrack wins

**It keeps the weak detections.**  This is the whole idea of the paper and it
is worth 107 fewer misses here.  A detection below the confidence threshold is
still evidence that something is there, and ByteTrack's second association pass
matches those against tracks that failed the first pass.  DeepSORT has no
equivalent stage: `det_thresh` discards weak detections before the cascade
sees them, which is why its miss count is higher and its fragmentation count is
half as large again.

**Its recovery path is geometric, and geometry is right most of the time.**
DeepSORT recovers a lost identity through the appearance cascade, gated by a
Mahalanobis test on the Kalman state.  That gate is the part that bites: after
enough missed frames the constant-velocity prediction is wrong by far more than
the covariance admits, the correct detection is gated out, and appearance never
gets a vote.  `tests/test_tracking.py` contains a scene sitting on the good
side of that boundary, where DeepSORT keeps two identities through an occlusion
that makes ByteTrack swap them, and the same test's docstring says plainly
where the boundary is.

**The scene has five moving objects.**  Appearance earns its cost when many
similar objects are in shot at once.  Five is not many.

---

## The result that changes the reading

Swap the trained embedding for a colour histogram and DeepSORT gets better:

| DeepSORT variant | MOTA | IDF1 | ID switches | ms/frame p50 |
| --- | ---: | ---: | ---: | ---: |
| trained re-ID embedding | 0.90238 | 0.8946 | 32 | 25.85 |
| colour histogram, no weights | 0.90533 | 0.91113 | 19 | 26.129 |

Nineteen switches against ByteTrack's twenty four, from a descriptor with no
learned parameters at all, and an IDF1 essentially level with ByteTrack.

The reason is a domain gap that the re-identification metrics do not see.  The
embedding is trained on crops from the **warehouse** palette and evaluated, in
`results/published/reid_train.json`, on held-out identities from the same
palette, where it scores 0.9775 rank-1 and 0.8586 mAP.  The test sequences use
the **depot** palette with higher noise and heavier blur.  A colour histogram
degrades gracefully under a global colour shift because it is a coarse summary
of the same signal; a small convolutional network trained on one palette can
degrade sharply, because nothing forced it to be robust to that shift.

Two lessons, and the second is the one worth keeping.  A re-identification
benchmark measured on the training domain does not predict tracking performance
on a shifted one.  And a hand-built descriptor is a real baseline rather than a
placeholder, so it should be measured rather than assumed to be worse.

---

## How good is the association on its own

Rerun both on ground-truth boxes, filtered by the same visibility threshold the
evaluator uses, so detection error is removed entirely:

| Detections | Tracker | MOTA | IDF1 | ID switches | Fragmentations |
| --- | --- | ---: | ---: | ---: | ---: |
| model | bytetrack | 0.91257 | 0.91194 | 24 | 45 |
| model | deepsort | 0.90238 | 0.8946 | 32 | 67 |
| ground truth | bytetrack | 0.98453 | 0.95495 | 19 | 19 |
| ground truth | deepsort | 0.98379 | 0.95024 | 21 | 24 |

This is the number that says where to spend effort.

Perfect detections leave 19 switches for ByteTrack and 21 for DeepSORT, so
roughly four fifths of the switch count is association failing on cases the
detector handled correctly.  Fragmentations tell the opposite story: 45 falls to
19 and 67 falls to 24, so more than half the track interruptions are the
detector dropping the object rather than the tracker losing it.

MOTA moves from 0.912 to 0.985, which is a reminder that MOTA is dominated by
detection quality and is close to useless for comparing association
strategies.  IDF1 moves much less, from 0.912 to 0.955, which is why it is the
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

The network is deliberately tiny, and the sizing was a measurement rather than
a preference.  The first version used two convolutions per block and 32 to 256
channels, came to 1.2 M parameters and roughly 0.5 GFLOPs per crop.  The
detector is 4.9 GFLOPs per *frame* and the encoder runs once per *detection*,
so at eight objects in shot the appearance branch would have cost as much
arithmetic as the detector.  Shrinking it seven times cost 0.9 points of
rank-1, from 0.9861 to 0.9775, which is nothing next to what it saved.

### Setting the gate from the margin

`scripts/train_reid.py` reports the cosine distance distribution, and that is
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
| ByteTrack | 81 | 0.96 | 0.93402 | 0.02598 |
| DeepSORT | 72 | 0.94494 | 0.90733 | 0.03761 |

Two and a half to nearly four points of IDF1 between the best and worst
settings, which is larger than the gap between the two trackers.  That is the
argument for tuning on a validation split rather than reporting whatever the
defaults gave: a comparison between a tuned tracker and an untuned one measures
the tuning.

Ties are common: eighty-one ByteTrack settings produce about a dozen distinct
scores, so the tie-break decides the published configuration more often than
the objective does.  It breaks towards fewer switches, then a larger
confirmation count, then a shorter buffer, and that rule is written down in
`scripts/tune_tracker.py` rather than left to sort order.

---

## What would change the conclusion

* **More objects.**  Twenty people in a corridor rather than five in a
  warehouse bay, and appearance starts to matter far more than it does here.
* **A worse detector.**  ByteTrack's advantage comes largely from its low-score
  pass, which needs a detector whose weak detections are still roughly in the
  right place.  A noisier detector would erode that.
* **Real footage.**  The colour histogram beats the trained embedding here
  because instances are rendered in distinct saturated colours.  On real people
  in similar clothing that ordering would very likely reverse.
* **Longer occlusions.**  Both trackers were tuned with a 30-frame buffer on
  sequences where objects reappear quickly.  A minute-long occlusion is a
  re-identification problem rather than a tracking one, and neither tracker as
  configured would attempt it.

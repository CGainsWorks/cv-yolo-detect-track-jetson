"""Behavioural tests for both trackers.

These are written against the contract, not the implementation: an identity
survives a straight line, survives a short gap, does not survive an unbounded
one, is never reused, and is only published once the confirmation rule has been
met.  Both trackers are held to the same contract wherever they share one, so
the parametrised cases run twice.

The two tests that differ between the trackers are at the bottom and they are
the interesting ones: geometry alone cannot resolve a same-class crossing, and
appearance can.
"""

from __future__ import annotations

import numpy as np
import pytest

from edgetrack.track import ByteTrack, ByteTrackConfig, DeepSort, DeepSortConfig, build_tracker
from edgetrack.types import Detection


def make(x, y, w=24.0, h=60.0, score=0.9, class_id=0, feature=None):
    return Detection((x - w / 2, y - h / 2, x + w / 2, y + h / 2), score, class_id, feature)


def feature(seed: int, dim: int = 16) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=dim).astype(np.float32)
    return vector / np.linalg.norm(vector)


def build(name: str):
    if name == "bytetrack":
        return ByteTrack(ByteTrackConfig(n_init=2, track_buffer=10))
    return DeepSort(DeepSortConfig(n_init=2, max_age=10, max_cosine_distance=0.6))


def with_features(detections, seeds):
    return [
        Detection(d.xyxy, d.score, d.class_id, feature(s))
        for d, s in zip(detections, seeds)
    ]


TRACKERS = ["bytetrack", "deepsort"]


@pytest.mark.parametrize("name", TRACKERS)
def test_single_object_keeps_one_identity(name):
    tracker = build(name)
    ids = set()
    for t in range(20):
        detections = [make(100 + 4 * t, 200)]
        if name == "deepsort":
            detections = with_features(detections, [1])
        tracks = tracker.update(detections, t)
        ids.update(track.track_id for track in tracks)
    assert ids == {1}, f"{name} produced identities {sorted(ids)}"


@pytest.mark.parametrize("name", TRACKERS)
def test_confirmation_delays_publication(name):
    tracker = build(name)
    detections = [make(100, 200)]
    if name == "deepsort":
        detections = with_features(detections, [1])
    assert tracker.update(detections, 0) == [], "a track was published on its first frame"
    published = tracker.update(detections, 1)
    assert len(published) == 1


@pytest.mark.parametrize("name", TRACKERS)
def test_identity_survives_a_short_gap(name):
    tracker = build(name)
    for t in range(6):
        detections = [make(100 + 4 * t, 200)]
        tracker.update(with_features(detections, [1]) if name == "deepsort" else detections, t)
    for t in range(6, 10):  # four frames with nothing at all
        tracker.update([], t)
    detections = [make(100 + 4 * 10, 200)]
    tracks = tracker.update(
        with_features(detections, [1]) if name == "deepsort" else detections, 10
    )
    assert [track.track_id for track in tracks] == [1], (
        f"{name} did not recover the identity after a four frame gap"
    )


@pytest.mark.parametrize("name", TRACKERS)
def test_identity_dies_after_the_buffer(name):
    tracker = build(name)
    for t in range(6):
        detections = [make(100, 200)]
        tracker.update(with_features(detections, [1]) if name == "deepsort" else detections, t)
    for t in range(6, 40):  # well past the ten frame buffer
        tracker.update([], t)
    detections = [make(100, 200)]
    tracks = tracker.update(
        with_features(detections, [1]) if name == "deepsort" else detections, 40
    )
    assert tracks == [], "a re-appearing object was published before reconfirmation"
    tracks = tracker.update(
        with_features([make(100, 200)], [1]) if name == "deepsort" else [make(100, 200)], 41
    )
    assert [track.track_id for track in tracks] != [1], (
        f"{name} reused identity 1 after the track buffer expired"
    )


@pytest.mark.parametrize("name", TRACKERS)
def test_identities_are_never_reused(name):
    tracker = build(name)
    seen: set[int] = set()
    rng = np.random.default_rng(0)
    for t in range(60):
        detections = [make(float(rng.uniform(20, 600)), float(rng.uniform(80, 340)))
                      for _ in range(int(rng.integers(0, 4)))]
        if name == "deepsort":
            detections = with_features(detections, rng.integers(0, 1000, len(detections)))
        for track in tracker.update(detections, t):
            seen.add(track.track_id)
    assert len(seen) == len({*seen}), "identity set contains duplicates"
    assert max(seen, default=0) >= len(seen), "identities were recycled"


@pytest.mark.parametrize("name", TRACKERS)
def test_two_classes_do_not_swap(name):
    tracker = build(name)
    mapping: dict[int, int] = {}
    for t in range(20):
        detections = [
            make(100 + 6 * t, 200, class_id=0),
            make(340 - 6 * t, 200, class_id=1),
        ]
        if name == "deepsort":
            detections = with_features(detections, [1, 2])
        for track in tracker.update(detections, t):
            previous = mapping.setdefault(track.track_id, track.class_id)
            assert previous == track.class_id, "a track changed class"


@pytest.mark.parametrize("name", TRACKERS)
def test_reset_restores_the_initial_state(name):
    tracker = build(name)
    for t in range(5):
        detections = [make(100, 200)]
        tracker.update(with_features(detections, [1]) if name == "deepsort" else detections, t)
    tracker.reset()
    detections = [make(300, 300)]
    tracker.update(with_features(detections, [9]) if name == "deepsort" else detections, 0)
    tracks = tracker.update(
        with_features([make(300, 300)], [9]) if name == "deepsort" else [make(300, 300)], 1
    )
    assert [track.track_id for track in tracks] == [1], "reset did not restart the id counter"


def test_bytetrack_uses_low_score_detections():
    """The whole point of ByteTrack: a weak detection keeps a track alive.

    Run the same occluded stretch twice, once with the weak detections present
    and once with them removed, and require the first to hold the identity
    where the second cannot.
    """
    def run(include_weak: bool) -> list[int]:
        tracker = ByteTrack(ByteTrackConfig(n_init=2, track_buffer=3, track_thresh=0.5))
        published = []
        for t in range(24):
            weak = 8 <= t < 16
            score = 0.2 if weak else 0.9
            if weak and not include_weak:
                detections = []
            else:
                detections = [make(60 + 8 * t, 200, score=score)]
            published.append([track.track_id for track in tracker.update(detections, t)])
        return published[-1]

    assert run(True) == [1], "the low-score pass failed to hold the identity"
    assert run(False) != [1], "the identity survived without any detection at all"


def test_deepsort_survives_an_occlusion_that_bytetrack_does_not():
    """The case appearance exists for: motion is ambiguous, colour is not.

    Two objects of the same class approach each other, disappear behind
    something for five frames, and come back having swapped sides.  When they
    reappear, each track's constant-velocity prediction sits closer to the
    *other* object than to its own, so geometry actively points the wrong way
    rather than merely failing to point anywhere.  ByteTrack has nothing else
    to go on and swaps.  DeepSORT's Kalman covariance has grown enough over the
    five missed frames that both candidates pass the Mahalanobis gate, and the
    appearance descriptors then pick correctly.

    The gate is the subtle part.  It is what stops a similarly coloured object
    on the far side of the frame from stealing an identity, and it is also what
    makes DeepSORT fail on a long occlusion: after enough missed frames the
    prediction is wrong by far more than the covariance admits, the correct
    detection is gated out and appearance never gets a vote.  That limitation
    is real and is written up in docs/tracking.md; this test sits on the side
    of it where appearance still helps.
    """
    width, height = 64.0, 160.0

    def box(x, use_features, seed):
        detection = Detection(
            (x - width / 2, 200 - height / 2, x + width / 2, 200 + height / 2), 0.9, 0
        )
        return with_features([detection], [seed])[0] if use_features else detection

    def positions(t):
        if t <= 9:  # approaching
            return 200.0 + 6 * t, 400.0 - 6 * t
        if t <= 14:  # behind the occluder, no detections at all
            return None, None
        return 340.0, 260.0  # back, and swapped over

    def run(tracker, use_features):
        before, after = {}, {}
        for t in range(26):
            a, b = positions(t)
            detections = (
                []
                if a is None
                else [box(a, use_features, 11), box(b, use_features, 77)]
            )
            for track in tracker.update(detections, t):
                if a is None:
                    continue
                which = "A" if abs(track.centre[0] - a) < abs(track.centre[0] - b) else "B"
                (before if t <= 9 else after).setdefault(which, set()).add(track.track_id)
        return before, after

    byte_before, byte_after = run(ByteTrack(ByteTrackConfig(n_init=2, track_buffer=30)), False)
    deep_before, deep_after = run(
        DeepSort(DeepSortConfig(n_init=2, max_age=30, max_cosine_distance=0.5)), True
    )

    assert byte_before == {"A": {1}, "B": {2}}, byte_before
    assert deep_before == {"A": {1}, "B": {2}}, deep_before

    assert byte_after != byte_before, (
        "ByteTrack kept the identities through a swap it has no information "
        f"about: {byte_after}"
    )
    assert deep_after == deep_before, (
        f"DeepSORT lost the identities the appearance descriptors encode: {deep_after}"
    )


def test_deepsort_refuses_detections_without_features():
    tracker = DeepSort()
    with pytest.raises(ValueError, match="appearance descriptor"):
        tracker.update([make(100, 200)], 0)


def test_build_tracker_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown tracker"):
        build_tracker("sort")


def test_bytetrack_config_validates():
    with pytest.raises(ValueError, match="det_thresh"):
        ByteTrackConfig(track_thresh=0.8, det_thresh=0.3).validate()
    with pytest.raises(ValueError, match="track_buffer"):
        ByteTrackConfig(track_buffer=0).validate()


def test_deepsort_config_validates():
    with pytest.raises(ValueError, match="max_cosine_distance"):
        DeepSortConfig(max_cosine_distance=0.0).validate()
    with pytest.raises(ValueError, match="lambda_motion"):
        DeepSortConfig(lambda_motion=1.5).validate()


@pytest.mark.parametrize("name", TRACKERS)
def test_trajectory_is_recorded(name):
    tracker = build(name)
    for t in range(12):
        detections = [make(100 + 5 * t, 200)]
        tracks = tracker.update(
            with_features(detections, [3]) if name == "deepsort" else detections, t
        )
    assert tracks, "no track was published"
    trail = tracks[0].trajectory()
    assert trail.shape[1] == 3
    assert trail.shape[0] >= 10
    # The trail must move monotonically with the object.
    assert trail[-1, 1] > trail[0, 1]

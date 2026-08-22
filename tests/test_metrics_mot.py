"""MOT metrics: hand-worked cases, then a cross-check against py-motmetrics.

The hand-worked cases pin the definitions.  The cross-check is the real
argument: `py-motmetrics` is an independent implementation of the same
specification, so agreement on randomised sequences means both are right or
both are wrong in the same way, and the second is unlikely for two
implementations written from the papers by different people.
"""

from __future__ import annotations

import numpy as np
import pytest

from edgetrack.metrics.mot import evaluate_mot, id_measures_from_counts

motmetrics = pytest.importorskip("motmetrics", reason="py-motmetrics is a dev dependency")


def naive_iou_distance(gt_boxes: list[np.ndarray], hyp_boxes: list[np.ndarray],
                       max_iou: float = 0.5) -> np.ndarray:
    """IoU distance matrix, written out longhand for the cross-check.

    ``motmetrics.distances.iou_matrix`` would be the obvious thing to call and
    it does not run: py-motmetrics 1.4.0 uses ``np.asfarray``, which NumPy 2.0
    removed, and pinning NumPy back is not an option when torch and ONNX
    Runtime both want 2.x.

    Written as a nested loop rather than by calling
    ``edgetrack.track.matching.iou_matrix`` on purpose.  Feeding the
    implementation under test into the reference implementation would make the
    comparison partly circular; this version is slow, obviously correct and
    independent of both.  The IoU calculation itself is pinned separately in
    ``tests/test_kalman_matching.py`` against boxes worked out by hand.
    """
    out = np.full((len(gt_boxes), len(hyp_boxes)), np.nan, dtype=np.float64)
    for i, a in enumerate(gt_boxes):
        for j, b in enumerate(hyp_boxes):
            left, top = max(a[0], b[0]), max(a[1], b[1])
            right, bottom = min(a[2], b[2]), min(a[3], b[3])
            inter = max(0.0, right - left) * max(0.0, bottom - top)
            union = (
                (a[2] - a[0]) * (a[3] - a[1])
                + (b[2] - b[0]) * (b[3] - b[1])
                - inter
            )
            iou = inter / union if union > 0 else 0.0
            if iou >= max_iou:
                out[i, j] = 1.0 - iou
    return out


def box(x, y, w=10.0, h=20.0):
    return np.array([x, y, x + w, y + h], dtype=np.float64)


def test_perfect_tracking_scores_one():
    gt = {t: {1: box(t * 5, 0)} for t in range(10)}
    metrics = evaluate_mot(gt, {t: dict(frame) for t, frame in gt.items()})
    assert metrics.mota == pytest.approx(1.0)
    assert metrics.idf1 == pytest.approx(1.0)
    assert metrics.num_switches == 0
    assert metrics.num_misses == 0
    assert metrics.num_false_positives == 0
    assert metrics.mostly_tracked == 1


def test_a_missed_frame_is_a_miss_not_a_switch():
    gt = {t: {1: box(t * 5, 0)} for t in range(10)}
    hyp = {t: ({7: box(t * 5, 0)} if t != 4 else {}) for t in range(10)}
    metrics = evaluate_mot(gt, hyp)
    assert metrics.num_misses == 1
    assert metrics.num_switches == 0
    assert metrics.num_fragmentations == 1
    assert metrics.mota == pytest.approx(1.0 - 1 / 10)


def test_renaming_an_object_is_one_switch():
    gt = {t: {1: box(t * 5, 0)} for t in range(10)}
    hyp = {t: {1 if t < 5 else 2: box(t * 5, 0)} for t in range(10)}
    metrics = evaluate_mot(gt, hyp)
    assert metrics.num_switches == 1
    # Half the trajectory carries the wrong identity, so IDF1 is near a half.
    assert 0.4 < metrics.idf1 < 0.6


def test_extra_detections_are_false_positives():
    gt = {t: {1: box(0, 0)} for t in range(5)}
    hyp = {t: {1: box(0, 0), 2: box(200, 200)} for t in range(5)}
    metrics = evaluate_mot(gt, hyp)
    assert metrics.num_false_positives == 5
    assert metrics.mota == pytest.approx(1.0 - 5 / 5)


def test_empty_hypotheses_are_all_misses():
    gt = {t: {1: box(0, 0), 2: box(50, 0)} for t in range(4)}
    metrics = evaluate_mot(gt, {})
    assert metrics.num_misses == 8
    assert metrics.num_false_positives == 0
    # MOTA is 1 - (FN + FP + IDSW) / GT, so missing everything scores exactly
    # zero rather than minus one.  Only false positives can push it negative.
    assert metrics.mota == pytest.approx(0.0)
    assert metrics.idf1 == 0.0
    assert metrics.mostly_lost == 2


def test_mota_can_go_negative():
    """More false positives than objects is worse than predicting nothing."""
    gt = {t: {1: box(0, 0)} for t in range(4)}
    hyp = {t: {i: box(500 + 100 * i, 0) for i in range(3)} for t in range(4)}
    metrics = evaluate_mot(gt, hyp)
    assert metrics.mota == pytest.approx(1.0 - (4 + 12) / 4)


def test_id_measures_from_counts_is_the_f1_of_the_assignment():
    idf1, idp, idr = id_measures_from_counts(80, 20, 40)
    assert idp == pytest.approx(0.8)
    assert idr == pytest.approx(2 / 3)
    assert idf1 == pytest.approx(2 * 80 / (2 * 80 + 20 + 40))


def _random_sequence(rng, n_frames=60, n_objects=5, miss=0.1, spawn=0.08, swap=0.05):
    """Ground truth plus a deliberately imperfect hypothesis set."""
    gt: dict[int, dict[int, np.ndarray]] = {}
    hyp: dict[int, dict[int, np.ndarray]] = {}
    velocities = rng.uniform(-4, 4, size=(n_objects, 2))
    starts = rng.uniform(20, 400, size=(n_objects, 2))
    alias = {i + 1: i + 1 for i in range(n_objects)}
    next_id = n_objects + 1

    for t in range(n_frames):
        gt_frame, hyp_frame = {}, {}
        for i in range(n_objects):
            x, y = starts[i] + velocities[i] * t
            gt_frame[i + 1] = box(x, y, 30, 60)
            if rng.random() < miss:
                continue
            if rng.random() < swap:
                alias[i + 1] = next_id
                next_id += 1
            jitter = rng.normal(0, 2.0, size=2)
            hyp_frame[alias[i + 1]] = box(x + jitter[0], y + jitter[1], 30, 60)
        if rng.random() < spawn:
            hyp_frame[next_id] = box(rng.uniform(0, 600), rng.uniform(0, 300), 30, 60)
            next_id += 1
        gt[t] = gt_frame
        hyp[t] = hyp_frame
    return gt, hyp


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
def test_matches_py_motmetrics(seed):
    rng = np.random.default_rng(seed)
    gt, hyp = _random_sequence(rng)

    accumulator = motmetrics.MOTAccumulator(auto_id=False)
    for frame_id in sorted(set(gt) | set(hyp)):
        gt_frame = gt.get(frame_id, {})
        hyp_frame = hyp.get(frame_id, {})
        gt_ids = list(gt_frame)
        hyp_ids = list(hyp_frame)
        distances = naive_iou_distance(
            list(gt_frame.values()), list(hyp_frame.values()), max_iou=0.5
        )
        accumulator.update(gt_ids, hyp_ids, distances, frameid=frame_id)

    host = motmetrics.metrics.create()
    summary = host.compute(
        accumulator,
        metrics=[
            "mota", "motp", "idf1", "idp", "idr", "num_switches",
            "num_false_positives", "num_misses", "num_matches", "num_objects",
            "mostly_tracked", "partially_tracked", "mostly_lost", "num_fragmentations",
        ],
        name="seq",
    ).iloc[0]

    mine = evaluate_mot(gt, hyp, iou_threshold=0.5)

    # py-motmetrics counts a frame whose identity changed as a SWITCH rather
    # than a MATCH; num_detections is the total either way.
    assert mine.num_matches == int(summary["num_matches"])
    assert mine.num_detections == mine.num_matches + mine.num_switches
    assert mine.num_misses == int(summary["num_misses"])
    assert mine.num_false_positives == int(summary["num_false_positives"])
    assert mine.num_switches == int(summary["num_switches"])
    assert mine.num_objects == int(summary["num_objects"])
    assert mine.num_fragmentations == int(summary["num_fragmentations"])
    assert mine.mostly_tracked == int(summary["mostly_tracked"])
    assert mine.partially_tracked == int(summary["partially_tracked"])
    assert mine.mostly_lost == int(summary["mostly_lost"])
    assert mine.mota == pytest.approx(float(summary["mota"]), abs=1e-9)
    assert mine.motp == pytest.approx(float(summary["motp"]), abs=1e-6)
    assert mine.idf1 == pytest.approx(float(summary["idf1"]), abs=1e-9)
    assert mine.idp == pytest.approx(float(summary["idp"]), abs=1e-9)
    assert mine.idr == pytest.approx(float(summary["idr"]), abs=1e-9)

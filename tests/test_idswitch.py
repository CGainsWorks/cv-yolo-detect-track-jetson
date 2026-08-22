"""ID-switch attribution: each rule fires on the case it was written for.

The attribution is a precedence chain, so the tests are built as scenes where
exactly one cause applies, plus one where two apply and the precedence decides.
"""

from __future__ import annotations

import numpy as np

from edgetrack.metrics.idswitch import CAUSES, attribute_switches
from edgetrack.metrics.mot import evaluate_mot


def box(x, y, w=30.0, h=60.0):
    return np.array([x, y, x + w, y + h], dtype=np.float64)


def analyse(gt, hyp, classes, visibility=None, **kwargs):
    metrics = evaluate_mot(gt, hyp, 0.5)
    return metrics, attribute_switches(metrics.events, gt, classes, visibility, **kwargs)


def test_no_switches_gives_an_empty_analysis():
    gt = {t: {1: box(t * 4, 0)} for t in range(10)}
    metrics, analysis = analyse(gt, {t: {5: box(t * 4, 0)} for t in range(10)}, {1: 0})
    assert metrics.num_switches == 0
    assert analysis.records == []
    assert set(analysis.counts) == set(CAUSES)
    assert sum(analysis.counts.values()) == 0


def test_a_detector_gap_is_attributed_to_the_detector():
    """The object stays fully visible and alone; the hypothesis simply stops.

    Nothing occluded it and nothing was near it, so the only explanation is a
    detection the model failed to make.
    """
    gt = {t: {1: box(t * 4, 0)} for t in range(20)}
    hyp = {}
    for t in range(20):
        if 8 <= t <= 11:
            hyp[t] = {}
        else:
            hyp[t] = {1 if t < 8 else 2: box(t * 4, 0)}
    visibility = {t: {1: 1.0} for t in range(20)}
    metrics, analysis = analyse(gt, hyp, {1: 0}, visibility)
    assert metrics.num_switches == 1
    assert analysis.counts["detector_gap"] == 1
    assert analysis.records[0].gap_frames >= 1


def test_an_occlusion_is_attributed_to_occlusion_not_to_the_detector():
    """Same shape of failure, but the ground truth says it was hidden.

    Precedence matters here: a hidden object is also an object the detector
    missed, and calling that a detector fault would send somebody to tune a
    confidence threshold that was never the problem.
    """
    gt = {t: {1: box(t * 4, 0)} for t in range(20)}
    hyp = {t: ({} if 8 <= t <= 11 else {1 if t < 8 else 2: box(t * 4, 0)}) for t in range(20)}
    visibility = {t: {1: (0.05 if 8 <= t <= 11 else 1.0)} for t in range(20)}
    _, analysis = analyse(gt, hyp, {1: 0}, visibility)
    assert analysis.counts["occlusion"] == 1
    assert analysis.counts["detector_gap"] == 0
    assert analysis.records[0].min_visibility < 0.5


def test_a_same_class_crossing_is_attributed_to_crossing():
    """Two objects of one class meet, and the identities come out swapped."""
    gt, hyp = {}, {}
    for t in range(20):
        offset = abs(9 - t) * 6.0
        gt[t] = {1: box(200 - offset, 0), 2: box(200 + offset, 0)}
        # The hypothesis follows the correct object until the crossing and the
        # wrong one afterwards.
        if t <= 9:
            hyp[t] = {11: box(200 - offset, 0), 12: box(200 + offset, 0)}
        else:
            hyp[t] = {12: box(200 - offset, 0), 11: box(200 + offset, 0)}
    visibility = {t: {1: 1.0, 2: 1.0} for t in range(20)}
    metrics, analysis = analyse(gt, hyp, {1: 0, 2: 0}, visibility)
    assert metrics.num_switches >= 1
    assert analysis.counts["crossing"] >= 1
    assert analysis.counts["detector_gap"] == 0


def test_a_cross_class_pass_is_not_counted_as_a_crossing():
    """A person walking in front of a pallet is not an ambiguous association.

    The class gate in both trackers forbids it, so the attribution must not
    blame it either.  This scene has an unexplained switch and the residual
    bucket is where it belongs.
    """
    gt, hyp = {}, {}
    for t in range(20):
        offset = abs(9 - t) * 6.0
        gt[t] = {1: box(200 - offset, 0), 2: box(200 + offset, 0)}
        hyp[t] = {(11 if t <= 9 else 12): box(200 - offset, 0), 13: box(200 + offset, 0)}
    visibility = {t: {1: 1.0, 2: 1.0} for t in range(20)}
    _, analysis = analyse(gt, hyp, {1: 0, 2: 2}, visibility)
    assert analysis.counts["crossing"] == 0
    assert sum(analysis.counts.values()) >= 1


def test_fast_motion_is_attributed_when_nothing_else_applies():
    """An isolated object moving further than half a box height per frame."""
    gt, hyp = {}, {}
    for t in range(20):
        x = t * 60.0  # 60 px per frame against a 60 px box height
        gt[t] = {1: box(x, 0)}
        hyp[t] = {1 if t < 10 else 2: box(x, 0)}
    visibility = {t: {1: 1.0} for t in range(20)}
    _, analysis = analyse(gt, hyp, {1: 0}, visibility)
    assert analysis.counts["fast_motion"] == 1
    assert analysis.records[0].max_speed_heights > 0.55


def test_an_unattributable_switch_is_reported_as_such():
    """A slow, isolated, fully visible object that is renamed anyway.

    A residual is a signal that the rules are incomplete, so it is reported
    rather than folded into the nearest bucket.
    """
    gt = {t: {1: box(t * 2, 0)} for t in range(20)}
    hyp = {t: {1 if t < 10 else 2: box(t * 2, 0)} for t in range(20)}
    visibility = {t: {1: 1.0} for t in range(20)}
    _, analysis = analyse(gt, hyp, {1: 0}, visibility)
    assert analysis.counts["unattributed"] == 1


def test_every_switch_gets_exactly_one_cause():
    rng = np.random.default_rng(4)
    gt, hyp = {}, {}
    alias = {1: 1, 2: 2}
    for t in range(60):
        gt[t] = {1: box(rng.uniform(0, 300), 0), 2: box(rng.uniform(0, 300), 60)}
        if rng.random() < 0.15:
            alias[1] = alias[1] + 10
        hyp[t] = {alias[1]: gt[t][1], alias[2]: gt[t][2]}
    metrics, analysis = analyse(gt, hyp, {1: 0, 2: 0})
    assert len(analysis.records) == metrics.num_switches
    assert sum(analysis.counts.values()) == metrics.num_switches
    assert all(record.cause in CAUSES for record in analysis.records)


def test_the_analysis_serialises():
    gt = {t: {1: box(t * 2, 0)} for t in range(12)}
    hyp = {t: {1 if t < 6 else 2: box(t * 2, 0)} for t in range(12)}
    _, analysis = analyse(gt, hyp, {1: 0})
    payload = analysis.as_dict()
    assert payload["total"] == 1
    assert payload["window"] == 12
    assert payload["records"][0]["cause"] in CAUSES

"""Detection metrics on cases small enough to work out by hand.

The mAP implementation is load-bearing for the headline table, so it is pinned
against arithmetic that can be checked on paper rather than against another
library's output.  ``tests/test_backends.py`` adds the comparison against
Ultralytics' validator on the real model, which is the harder check but needs
the weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from edgetrack.metrics.detection import (
    DetectionSample,
    average_precision,
    evaluate_detections,
)


def sample(preds, gts):
    """``preds`` is (box, score, class), ``gts`` is (box, class)."""
    return DetectionSample(
        pred_boxes=np.asarray([p[0] for p in preds], dtype=np.float64).reshape(-1, 4),
        pred_scores=np.asarray([p[1] for p in preds], dtype=np.float64),
        pred_classes=np.asarray([p[2] for p in preds], dtype=np.int64),
        gt_boxes=np.asarray([g[0] for g in gts], dtype=np.float64).reshape(-1, 4),
        gt_classes=np.asarray([g[1] for g in gts], dtype=np.int64),
    )


BOX = (10.0, 10.0, 50.0, 90.0)
OTHER = (200.0, 200.0, 240.0, 280.0)


def test_perfect_predictions_score_one():
    samples = [sample([(BOX, 0.9, 0)], [(BOX, 0)])]
    result = evaluate_detections(samples, [0])
    assert result.map50 == pytest.approx(1.0)
    assert result.map50_95 == pytest.approx(1.0)
    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)


def test_a_missed_object_halves_recall():
    samples = [sample([(BOX, 0.9, 0)], [(BOX, 0), (OTHER, 0)])]
    result = evaluate_detections(samples, [0])
    assert result.recall == pytest.approx(0.5)
    assert result.precision == pytest.approx(1.0)
    # One of two objects found at every threshold, so AP is a half.
    assert result.map50 == pytest.approx(0.5, abs=0.02)


def test_a_spurious_box_costs_precision_not_recall():
    samples = [sample([(BOX, 0.9, 0), (OTHER, 0.8, 0)], [(BOX, 0)])]
    result = evaluate_detections(samples, [0])
    assert result.recall == pytest.approx(1.0)
    assert result.precision == pytest.approx(0.5)


def test_wrong_class_is_both_a_miss_and_a_false_positive():
    samples = [sample([(BOX, 0.9, 1)], [(BOX, 0)])]
    result = evaluate_detections(samples, [0, 1])
    assert result.recall == pytest.approx(0.0)
    assert result.precision == pytest.approx(0.0)
    assert result.map50 == pytest.approx(0.0)


def test_only_one_prediction_can_claim_a_ground_truth_box():
    """The second copy of a correct box is a false positive, not a second hit."""
    samples = [sample([(BOX, 0.9, 0), (BOX, 0.8, 0)], [(BOX, 0)])]
    result = evaluate_detections(samples, [0])
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(1.0)


def test_loose_boxes_fail_the_stricter_thresholds():
    """A box at roughly 0.6 IoU passes at 0.5 and fails from 0.65 upward."""
    loose = (10.0, 10.0, 50.0, 60.0)  # same left/top, shorter
    samples = [sample([(loose, 0.9, 0)], [(BOX, 0)])]
    result = evaluate_detections(samples, [0])
    assert result.map50 == pytest.approx(1.0)
    assert result.map50_95 < 0.5, result.map50_95


def test_classes_absent_from_the_ground_truth_are_excluded():
    samples = [sample([(BOX, 0.9, 0)], [(BOX, 0)])]
    with_absent = evaluate_detections(samples, [0, 1, 2, 3])
    only_present = evaluate_detections(samples, [0])
    assert with_absent.map50 == pytest.approx(only_present.map50)
    assert set(with_absent.per_class_ap50) == {0}


def test_confidence_ordering_matters():
    """A confident false positive hurts more than a hesitant one.

    Two images, one correct box and one spurious box.  Ranking the spurious box
    above the correct one drops AP; ranking it below does not.
    """
    good_first = [
        sample([(BOX, 0.9, 0)], [(BOX, 0)]),
        sample([(OTHER, 0.2, 0)], []),
    ]
    bad_first = [
        sample([(BOX, 0.2, 0)], [(BOX, 0)]),
        sample([(OTHER, 0.9, 0)], []),
    ]
    assert evaluate_detections(good_first, [0]).map50 > evaluate_detections(bad_first, [0]).map50


def test_average_precision_on_a_worked_example():
    """Three predictions, two ground truths, hits at ranks 1 and 3.

    Precision is 1.0 at recall 0.5 and 2/3 at recall 1.0.  After the monotone
    envelope the curve is 1.0 up to recall 0.5 and 2/3 to recall 1.0, so with
    101-point sampling the result is (51 * 1 + 50 * 2/3) / 101.
    """
    tp = np.array([1.0, 0.0, 1.0])
    expected = (51 * 1.0 + 50 * (2 / 3)) / 101
    assert average_precision(tp, n_gt=2) == pytest.approx(expected, abs=1e-9)


def test_average_precision_with_no_ground_truth_is_nan():
    assert np.isnan(average_precision(np.array([1.0]), n_gt=0))


def test_average_precision_with_no_predictions_is_zero():
    assert average_precision(np.zeros(0), n_gt=5) == 0.0


def test_empty_evaluation_does_not_divide_by_zero():
    result = evaluate_detections([], [0, 1])
    assert result.map50 == 0.0
    assert result.n_gt == 0

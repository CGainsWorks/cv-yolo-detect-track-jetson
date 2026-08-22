"""The motion model and the association primitives.

Both trackers share these, so a bug here would move every number in the
tracking table at once and look like a tracker difference.
"""

from __future__ import annotations

import numpy as np
import pytest

from edgetrack.track.kalman import CHI2_INV95, KalmanFilter, xyah_to_xyxy
from edgetrack.track.matching import (
    cosine_distance,
    fuse_score,
    iou_distance,
    iou_matrix,
    linear_assignment,
)


# -- Kalman ---------------------------------------------------------------


def measurement(cx, cy, aspect=0.5, height=80.0):
    return np.array([cx, cy, aspect, height], dtype=np.float64)


def test_filter_learns_a_constant_velocity():
    kf = KalmanFilter()
    mean, cov = kf.initiate(measurement(100, 200))
    for step in range(1, 30):
        mean, cov = kf.predict(mean, cov)
        mean, cov = kf.update(mean, cov, measurement(100 + 5 * step, 200))
    assert mean[4] == pytest.approx(5.0, abs=0.4), f"velocity estimate {mean[4]}"
    assert mean[5] == pytest.approx(0.0, abs=0.4)


def test_prediction_extrapolates_when_no_measurement_arrives():
    kf = KalmanFilter()
    mean, cov = kf.initiate(measurement(100, 200))
    for step in range(1, 20):
        mean, cov = kf.predict(mean, cov)
        mean, cov = kf.update(mean, cov, measurement(100 + 4 * step, 200))
    before = mean[0]
    for _ in range(3):
        mean, cov = kf.predict(mean, cov)
    assert mean[0] > before + 8, "the filter did not coast forward"


def test_uncertainty_grows_without_measurements():
    kf = KalmanFilter()
    mean, cov = kf.initiate(measurement(100, 200))
    initial = np.trace(cov)
    for _ in range(5):
        mean, cov = kf.predict(mean, cov)
    assert np.trace(cov) > initial


def test_update_reduces_uncertainty():
    kf = KalmanFilter()
    mean, cov = kf.initiate(measurement(100, 200))
    mean, cov = kf.predict(mean, cov)
    predicted = np.trace(cov)
    mean, cov = kf.update(mean, cov, measurement(101, 200))
    assert np.trace(cov) < predicted


def test_covariance_stays_symmetric_and_positive_definite():
    kf = KalmanFilter()
    mean, cov = kf.initiate(measurement(100, 200))
    for step in range(200):
        mean, cov = kf.predict(mean, cov)
        mean, cov = kf.update(mean, cov, measurement(100 + step, 200 + 0.5 * step))
    assert np.allclose(cov, cov.T, atol=1e-12)
    # Cholesky is the operation gating_distance performs; it must not throw.
    np.linalg.cholesky(cov[:4, :4])


def test_low_confidence_measurements_move_the_state_less():
    kf = KalmanFilter()
    base_mean, base_cov = kf.initiate(measurement(100, 200))
    base_mean, base_cov = kf.predict(base_mean, base_cov)
    strong, _ = kf.update(base_mean.copy(), base_cov.copy(), measurement(140, 200), 0.95)
    weak, _ = kf.update(base_mean.copy(), base_cov.copy(), measurement(140, 200), 0.15)
    assert abs(strong[0] - 140) < abs(weak[0] - 140)


def test_gating_distance_rejects_a_distant_measurement():
    kf = KalmanFilter()
    mean, cov = kf.initiate(measurement(100, 200))
    mean, cov = kf.predict(mean, cov)
    near = kf.gating_distance(mean, cov, measurement(103, 201)[None, :])
    far = kf.gating_distance(mean, cov, measurement(600, 200)[None, :])
    assert near[0] < CHI2_INV95[4] < far[0]


def test_gating_distance_on_empty_input():
    kf = KalmanFilter()
    mean, cov = kf.initiate(measurement(100, 200))
    assert kf.gating_distance(mean, cov, np.zeros((0, 4))).shape == (0,)


def test_xyah_to_xyxy_round_trip():
    box = xyah_to_xyxy(np.array([100.0, 200.0, 0.5, 80.0, 0, 0, 0, 0]))
    assert box == pytest.approx((80.0, 160.0, 120.0, 240.0))


# -- Matching -------------------------------------------------------------


def test_iou_of_identical_boxes_is_one():
    box = np.array([[0.0, 0.0, 10.0, 10.0]])
    assert iou_matrix(box, box)[0, 0] == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[100.0, 100.0, 110.0, 110.0]])
    assert iou_matrix(a, b)[0, 0] == 0.0


def test_iou_on_a_half_overlap():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[5.0, 0.0, 15.0, 10.0]])
    # Intersection 50, union 150.
    assert iou_matrix(a, b)[0, 0] == pytest.approx(50 / 150)


def test_iou_of_a_degenerate_box_is_zero_not_nan():
    a = np.array([[5.0, 5.0, 5.0, 5.0]])
    b = np.array([[0.0, 0.0, 10.0, 10.0]])
    value = iou_matrix(a, b)[0, 0]
    assert value == 0.0 and not np.isnan(value)


def test_iou_matrix_shapes():
    assert iou_matrix(np.zeros((3, 4)), np.zeros((5, 4))).shape == (3, 5)
    assert iou_matrix(np.zeros((0, 4)), np.zeros((5, 4))).shape == (0, 5)
    assert iou_distance(np.zeros((2, 4)), np.zeros((0, 4))).shape == (2, 0)


def test_cosine_distance_of_identical_vectors_is_zero():
    v = np.array([[3.0, 4.0]])
    assert cosine_distance(v, v)[0, 0] == pytest.approx(0.0, abs=1e-12)


def test_cosine_distance_of_opposite_vectors_is_two():
    a = np.array([[1.0, 0.0]])
    b = np.array([[-1.0, 0.0]])
    assert cosine_distance(a, b)[0, 0] == pytest.approx(2.0)


def test_cosine_distance_ignores_magnitude():
    a = np.array([[1.0, 0.0]])
    b = np.array([[7.0, 0.0]])
    assert cosine_distance(a, b)[0, 0] == pytest.approx(0.0, abs=1e-12)


def test_cosine_distance_survives_a_zero_vector():
    a = np.array([[0.0, 0.0]])
    b = np.array([[1.0, 0.0]])
    assert not np.isnan(cosine_distance(a, b)[0, 0])


def test_cosine_distance_rejects_a_dimension_mismatch():
    with pytest.raises(ValueError, match="dimension mismatch"):
        cosine_distance(np.zeros((2, 4)), np.zeros((2, 8)))


def test_linear_assignment_finds_the_optimum():
    cost = np.array([[0.1, 0.9], [0.9, 0.2]])
    matches, unmatched_rows, unmatched_cols = linear_assignment(cost, 0.5)
    assert sorted(matches) == [(0, 0), (1, 1)]
    assert unmatched_rows == [] and unmatched_cols == []


def test_linear_assignment_rejects_pairs_above_the_ceiling():
    cost = np.array([[0.1, 0.9], [0.9, 0.95]])
    matches, unmatched_rows, unmatched_cols = linear_assignment(cost, 0.5)
    assert matches == [(0, 0)]
    assert unmatched_rows == [1]
    assert unmatched_cols == [1]


def test_linear_assignment_handles_a_row_with_no_admissible_column():
    """The reason costs are filtered after solving rather than before.

    Setting forbidden pairs to infinity first makes the problem infeasible when
    a row has nothing admissible, and scipy raises rather than returning a
    partial assignment.
    """
    cost = np.array([[0.99, 0.99], [0.1, 0.99]])
    matches, unmatched_rows, _ = linear_assignment(cost, 0.5)
    assert matches == [(1, 0)]
    assert unmatched_rows == [0]


def test_linear_assignment_with_infinities():
    cost = np.array([[np.inf, 0.2], [0.3, np.inf]])
    matches, _, _ = linear_assignment(cost, 0.5)
    assert sorted(matches) == [(0, 1), (1, 0)]


def test_linear_assignment_on_empty_matrices():
    assert linear_assignment(np.zeros((0, 3)), 0.5) == ([], [], [0, 1, 2])
    assert linear_assignment(np.zeros((2, 0)), 0.5) == ([], [0, 1], [])


def test_fuse_score_penalises_weak_detections():
    cost = np.array([[0.2, 0.2]])
    fused = fuse_score(cost, np.array([1.0, 0.5]))
    assert fused[0, 0] < fused[0, 1]
    assert fused[0, 0] == pytest.approx(0.2)

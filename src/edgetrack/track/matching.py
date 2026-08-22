"""Association primitives: IoU, cosine distance and linear assignment.

The trackers differ in *what* they put in the cost matrix.  They do not differ
in how the matrix is solved, so the solver lives here and both call it.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = [
    "iou_matrix",
    "iou_distance",
    "cosine_distance",
    "linear_assignment",
    "fuse_score",
]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of ``xyxy`` boxes.

    Returns an ``(len(a), len(b))`` matrix.  Degenerate boxes give zero rather
    than a divide-by-zero warning, which matters because a Kalman prediction
    can collapse a box after a long run of misses.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)

    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)

    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]

    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(union > 0, inter / union, 0.0)
    return out


def iou_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``1 - IoU``, the cost form."""
    return 1.0 - iou_matrix(a, b)


def cosine_distance(features: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Cosine distance between L2-normalised descriptors.

    Inputs are renormalised defensively: an encoder that returns an all-zero
    descriptor for a one-pixel crop would otherwise produce NaN and poison the
    assignment.
    """
    features = (
        np.asarray(features, dtype=np.float64).reshape(len(features), -1)
        if len(features)
        else np.zeros((0, 1))
    )
    gallery = (
        np.asarray(gallery, dtype=np.float64).reshape(len(gallery), -1)
        if len(gallery)
        else np.zeros((0, 1))
    )
    if features.size == 0 or gallery.size == 0:
        return np.zeros((features.shape[0], gallery.shape[0]), dtype=np.float64)
    if features.shape[1] != gallery.shape[1]:
        raise ValueError(
            f"descriptor dimension mismatch: {features.shape[1]} against {gallery.shape[1]}"
        )
    fn = np.linalg.norm(features, axis=1, keepdims=True)
    gn = np.linalg.norm(gallery, axis=1, keepdims=True)
    features = features / np.where(fn > 0, fn, 1.0)
    gallery = gallery / np.where(gn > 0, gn, 1.0)
    return np.clip(1.0 - features @ gallery.T, 0.0, 2.0)


def linear_assignment(
    cost: np.ndarray, max_cost: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian assignment with a cost ceiling.

    Pairs whose cost exceeds ``max_cost`` are rejected after solving rather
    than before.  Rejecting before by setting them to infinity makes the
    problem infeasible whenever a row has no admissible column; rejecting
    after costs one extra pass and always terminates.
    """
    cost = np.asarray(cost, dtype=np.float64)
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))

    # A large finite sentinel keeps the solver feasible while making forbidden
    # pairs unattractive.  It must exceed max_cost by more than the spread of
    # real costs or the solver may still pick one when a row has nothing else.
    padded = np.where(np.isfinite(cost), cost, max_cost + 1e6)
    rows, cols = linear_sum_assignment(padded)

    matches: list[tuple[int, int]] = []
    matched_rows: set[int] = set()
    matched_cols: set[int] = set()
    for r, c in zip(rows, cols):
        if padded[r, c] <= max_cost:
            matches.append((int(r), int(c)))
            matched_rows.add(int(r))
            matched_cols.add(int(c))
    unmatched_rows = [i for i in range(cost.shape[0]) if i not in matched_rows]
    unmatched_cols = [j for j in range(cost.shape[1]) if j not in matched_cols]
    return matches, unmatched_rows, unmatched_cols


def fuse_score(cost: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Fold detection confidence into an IoU cost matrix.

    ByteTrack's reference implementation multiplies the IoU similarity by the
    detection score before converting back to a cost, so a high-IoU match to a
    weak detection is treated as slightly worse than a high-IoU match to a
    strong one.  It is a small effect and it is off by default in
    :class:`~edgetrack.track.bytetrack.ByteTrack`; the ablation is in
    ``docs/tracking.md``.
    """
    if cost.size == 0:
        return cost
    similarity = 1.0 - cost
    fused = similarity * np.asarray(scores, dtype=np.float64)[None, :]
    return 1.0 - fused

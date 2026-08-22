"""CLEAR-MOT and ID metrics.

Two families of number, measuring different failures.

CLEAR-MOT (Bernardin and Stiefelhagen, 2008) is frame-local.  MOTA counts
misses, false positives and identity switches against the number of ground
truth objects, so it is dominated by detection quality; a tracker can score
well on MOTA whilst renaming everything constantly, as long as it renames
consistently within each frame.

The ID measures (Ristani et al., 2016) are trajectory-global.  IDF1 solves one
bipartite matching between ground-truth identities and hypothesis identities
over the whole sequence and reports the F1 of that assignment, so a tracker
that swaps two identities halfway through is penalised for the whole second
half rather than for one frame.  IDF1 is therefore the number to read when
comparing ByteTrack against DeepSORT, and MOTA is the number to read when
comparing FP32 against INT8.

``tests/test_metrics_mot.py`` cross-checks every quantity below against
`py-motmetrics` on randomised sequences.  That is the argument for trusting
these numbers: not that the code looks right, but that an independent
implementation of the same specification agrees with it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..track.matching import iou_matrix

__all__ = ["MotMetrics", "evaluate_mot", "FrameEvent", "id_measures_from_counts"]


@dataclass
class FrameEvent:
    """One frame's worth of association outcome, kept for the ID-switch study."""

    frame_id: int
    matches: list[tuple[int, int, float]] = field(default_factory=list)  # gt, hyp, iou
    switches: list[tuple[int, int, int]] = field(default_factory=list)  # gt, old hyp, new hyp
    misses: list[int] = field(default_factory=list)  # gt ids
    false_positives: list[int] = field(default_factory=list)  # hyp ids


@dataclass
class MotMetrics:
    mota: float
    motp: float
    idf1: float
    idp: float
    idr: float
    precision: float
    recall: float
    num_switches: int
    num_fragmentations: int
    num_false_positives: int
    num_misses: int
    num_matches: int
    num_detections: int
    num_objects: int
    num_predictions: int
    mostly_tracked: int
    partially_tracked: int
    mostly_lost: int
    num_gt_ids: int
    num_hyp_ids: int
    idtp: int = 0
    idfp: int = 0
    idfn: int = 0
    events: list[FrameEvent] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "mota": round(self.mota, 5),
            "motp": round(self.motp, 5),
            "motp_iou": round(1.0 - self.motp, 5),
            "idf1": round(self.idf1, 5),
            "idp": round(self.idp, 5),
            "idr": round(self.idr, 5),
            "precision": round(self.precision, 5),
            "recall": round(self.recall, 5),
            "num_switches": self.num_switches,
            "num_fragmentations": self.num_fragmentations,
            "num_false_positives": self.num_false_positives,
            "num_misses": self.num_misses,
            "num_matches": self.num_matches,
            "num_detections": self.num_detections,
            "num_objects": self.num_objects,
            "num_predictions": self.num_predictions,
            "mostly_tracked": self.mostly_tracked,
            "partially_tracked": self.partially_tracked,
            "mostly_lost": self.mostly_lost,
            "num_gt_ids": self.num_gt_ids,
            "idtp": self.idtp,
            "idfp": self.idfp,
            "idfn": self.idfn,
            "num_hyp_ids": self.num_hyp_ids,
        }


def _frame_matching(
    gt_ids: list[int],
    gt_boxes: np.ndarray,
    hyp_ids: list[int],
    hyp_boxes: np.ndarray,
    previous: dict[int, int],
    iou_threshold: float,
) -> list[tuple[int, int, float]]:
    """Match one frame, honouring last frame's assignment first.

    The order matters and is the part of CLEAR-MOT people get wrong.  A pair
    that was matched last frame and is still valid this frame must stay
    matched, even if a different pairing would give a higher total IoU.
    Re-solving from scratch every frame invents identity switches that the
    tracker never made.
    """
    matches: list[tuple[int, int, float]] = []
    if not gt_ids or not hyp_ids:
        return matches

    ious = iou_matrix(gt_boxes, hyp_boxes)
    gt_index = {g: i for i, g in enumerate(gt_ids)}
    hyp_index = {h: j for j, h in enumerate(hyp_ids)}

    used_gt: set[int] = set()
    used_hyp: set[int] = set()
    for gt_id, hyp_id in previous.items():
        i, j = gt_index.get(gt_id), hyp_index.get(hyp_id)
        if i is None or j is None:
            continue
        if ious[i, j] >= iou_threshold:
            matches.append((gt_id, hyp_id, float(ious[i, j])))
            used_gt.add(gt_id)
            used_hyp.add(hyp_id)

    free_gt = [g for g in gt_ids if g not in used_gt]
    free_hyp = [h for h in hyp_ids if h not in used_hyp]
    if not free_gt or not free_hyp:
        return matches

    sub = ious[np.ix_([gt_index[g] for g in free_gt], [hyp_index[h] for h in free_hyp])]
    cost = 1.0 - sub
    rows, cols = linear_sum_assignment(cost)
    for r, c in zip(rows, cols):
        if sub[r, c] >= iou_threshold:
            matches.append((free_gt[r], free_hyp[c], float(sub[r, c])))
    return matches


def _id_measures(
    gt_frames: dict[int, dict[int, np.ndarray]],
    hyp_frames: dict[int, dict[int, np.ndarray]],
    iou_threshold: float,
) -> tuple[int, int, int]:
    """IDTP, IDFP and IDFN by global bipartite matching over identities.

    Builds the square cost matrix from Ristani et al. section 4: real
    identities in the top-left block, and dummy rows and columns carrying the
    cost of leaving an identity unmatched, so that the assignment is always
    feasible and unmatched identities are charged their full length.
    """
    gt_ids = sorted({i for frame in gt_frames.values() for i in frame})
    hyp_ids = sorted({i for frame in hyp_frames.values() for i in frame})
    gt_count = {i: 0 for i in gt_ids}
    hyp_count = {i: 0 for i in hyp_ids}
    for frame in gt_frames.values():
        for i in frame:
            gt_count[i] += 1
    for frame in hyp_frames.values():
        for i in frame:
            hyp_count[i] += 1

    total_gt = sum(gt_count.values())
    total_hyp = sum(hyp_count.values())
    if not gt_ids or not hyp_ids:
        return 0, total_hyp, total_gt

    gt_pos = {g: i for i, g in enumerate(gt_ids)}
    hyp_pos = {h: j for j, h in enumerate(hyp_ids)}
    overlap = np.zeros((len(gt_ids), len(hyp_ids)), dtype=np.int64)
    for frame_id, gt_frame in gt_frames.items():
        hyp_frame = hyp_frames.get(frame_id)
        if not hyp_frame or not gt_frame:
            continue
        g_ids = list(gt_frame)
        h_ids = list(hyp_frame)
        ious = iou_matrix(
            np.stack([gt_frame[i] for i in g_ids]), np.stack([hyp_frame[i] for i in h_ids])
        )
        for a, g in enumerate(g_ids):
            for b, h in enumerate(h_ids):
                if ious[a, b] >= iou_threshold:
                    overlap[gt_pos[g], hyp_pos[h]] += 1

    n, m = len(gt_ids), len(hyp_ids)
    big = float(total_gt + total_hyp + 1)
    cost = np.full((n + m, m + n), big, dtype=np.float64)
    for i, g in enumerate(gt_ids):
        for j, h in enumerate(hyp_ids):
            cost[i, j] = (gt_count[g] - overlap[i, j]) + (hyp_count[h] - overlap[i, j])
    for i, g in enumerate(gt_ids):
        cost[i, m + i] = gt_count[g]
    for j, h in enumerate(hyp_ids):
        cost[n + j, j] = hyp_count[h]
    cost[n:, m:] = 0.0

    rows, cols = linear_sum_assignment(cost)
    idtp = 0
    for r, c in zip(rows, cols):
        if r < n and c < m:
            idtp += int(overlap[r, c])
    return idtp, total_hyp - idtp, total_gt - idtp


def id_measures_from_counts(idtp: int, idfp: int, idfn: int) -> tuple[float, float, float]:
    """IDF1, IDP and IDR from raw counts.

    Split out so that a multi-sequence aggregate can sum the counts and derive
    the ratios once, rather than averaging per-sequence IDF1 values.  Averaging
    the ratios weights a twenty frame sequence the same as a two hundred frame
    one and is the most common way a tracking benchmark gets misreported.
    """
    if idtp + idfp + idfn == 0:
        return float("nan"), float("nan"), float("nan")
    idp = idtp / max(idtp + idfp, 1e-12)
    idr = idtp / max(idtp + idfn, 1e-12)
    idf1 = 2 * idtp / max(2 * idtp + idfp + idfn, 1e-12)
    return float(idf1), float(idp), float(idr)


def evaluate_mot(
    gt_frames: dict[int, dict[int, np.ndarray]],
    hyp_frames: dict[int, dict[int, np.ndarray]],
    iou_threshold: float = 0.5,
) -> MotMetrics:
    """Score a whole sequence.

    Both inputs are ``{frame_id: {track_id: xyxy}}``.  Frames present in one
    and absent from the other are handled: a frame with ground truth and no
    hypotheses is all misses, the reverse is all false positives.
    """
    frames = sorted(set(gt_frames) | set(hyp_frames))
    previous: dict[int, int] = {}
    assignment_history: dict[int, int] = {}
    events: list[FrameEvent] = []

    num_detections = num_misses = num_fp = num_switches = 0
    iou_sum = 0.0
    matched_frames: dict[int, set[int]] = {}
    was_matched_last: dict[int, bool] = {}
    fragmentations = 0

    for frame_id in frames:
        gt_frame = gt_frames.get(frame_id, {})
        hyp_frame = hyp_frames.get(frame_id, {})
        gt_ids = list(gt_frame)
        hyp_ids = list(hyp_frame)
        gt_boxes = np.stack([gt_frame[i] for i in gt_ids]) if gt_ids else np.zeros((0, 4))
        hyp_boxes = np.stack([hyp_frame[i] for i in hyp_ids]) if hyp_ids else np.zeros((0, 4))

        matches = _frame_matching(
            gt_ids, gt_boxes, hyp_ids, hyp_boxes, previous, iou_threshold
        )
        event = FrameEvent(frame_id=frame_id, matches=matches)

        matched_gt = set()
        matched_hyp = set()
        for gt_id, hyp_id, iou in matches:
            matched_gt.add(gt_id)
            matched_hyp.add(hyp_id)
            num_detections += 1
            iou_sum += iou
            matched_frames.setdefault(gt_id, set()).add(frame_id)
            last = assignment_history.get(gt_id)
            if last is not None and last != hyp_id:
                num_switches += 1
                event.switches.append((gt_id, last, hyp_id))
            assignment_history[gt_id] = hyp_id

        for gt_id in gt_ids:
            if gt_id not in matched_gt:
                num_misses += 1
                event.misses.append(gt_id)
        for hyp_id in hyp_ids:
            if hyp_id not in matched_hyp:
                num_fp += 1
                event.false_positives.append(hyp_id)

        # A fragmentation is a ground-truth object going from tracked to not
        # tracked while it is still present.  Counting resumptions rather than
        # interruptions matches the MOTChallenge definition.
        for gt_id in gt_ids:
            currently = gt_id in matched_gt
            if currently and was_matched_last.get(gt_id) is False:
                fragmentations += 1
            if gt_id in was_matched_last or currently:
                was_matched_last[gt_id] = currently

        previous = {gt_id: hyp_id for gt_id, hyp_id, _ in matches}
        events.append(event)

    num_objects = sum(len(frame) for frame in gt_frames.values())
    num_predictions = sum(len(frame) for frame in hyp_frames.values())

    gt_lengths: dict[int, int] = {}
    for frame in gt_frames.values():
        for gt_id in frame:
            gt_lengths[gt_id] = gt_lengths.get(gt_id, 0) + 1
    mostly_tracked = partially_tracked = mostly_lost = 0
    for gt_id, length in gt_lengths.items():
        ratio = len(matched_frames.get(gt_id, ())) / max(length, 1)
        if ratio >= 0.8:
            mostly_tracked += 1
        elif ratio < 0.2:
            mostly_lost += 1
        else:
            partially_tracked += 1

    idtp, idfp, idfn = _id_measures(gt_frames, hyp_frames, iou_threshold)
    idf1, idp, idr = id_measures_from_counts(idtp, idfp, idfn)

    mota = 1.0 - (num_misses + num_fp + num_switches) / max(num_objects, 1)
    motp = (1.0 - iou_sum / num_detections) if num_detections else float("nan")

    return MotMetrics(
        mota=float(mota),
        motp=float(motp),
        idf1=idf1,
        idp=idp,
        idr=idr,
        precision=num_detections / max(num_predictions, 1),
        recall=num_detections / max(num_objects, 1),
        num_switches=num_switches,
        num_fragmentations=fragmentations,
        num_false_positives=num_fp,
        num_misses=num_misses,
        # MOTChallenge, and py-motmetrics with it, treat a frame where the
        # identity changed as a SWITCH event rather than as a MATCH, so
        # num_matches excludes them and num_detections is the total number of
        # ground-truth boxes that were covered by some hypothesis.  Precision,
        # recall and MOTP are all computed over num_detections.  Reporting only
        # one of the two is how a tracking table ends up disagreeing with the
        # reference tool by exactly the switch count.
        num_matches=num_detections - num_switches,
        num_detections=num_detections,
        num_objects=num_objects,
        num_predictions=num_predictions,
        mostly_tracked=mostly_tracked,
        partially_tracked=partially_tracked,
        mostly_lost=mostly_lost,
        idtp=idtp,
        idfp=idfp,
        idfn=idfn,
        num_gt_ids=len(gt_lengths),
        num_hyp_ids=len({i for frame in hyp_frames.values() for i in frame}),
        events=events,
    )

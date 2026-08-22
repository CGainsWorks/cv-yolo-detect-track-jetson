"""Why each identity switch happened.

An ID-switch count on its own is not actionable.  Twelve switches caused by
people walking behind a shelf and twelve caused by the detector dropping boxes
call for opposite fixes: the first wants a longer track buffer or a better
appearance model, the second wants a lower detection threshold.  This module
attributes every switch reported by :func:`edgetrack.metrics.mot.evaluate_mot`
to one cause, using the ground truth that the synthetic dataset makes available.

Causes, in the precedence order they are tested.  Precedence matters because
the causes overlap: an object behind a shelf is also an object the detector
missed.  The order below runs from the most specific evidence to the least.

``detector_gap``
    The ground-truth object was visible but unmatched for one or more frames
    immediately before the switch, and it was visible enough to detect
    throughout.  The detector missed it and the track died.

``occlusion``
    Ground-truth visibility dropped below ``occlusion_visibility`` during the
    window before the switch.  The object was hidden, so no detector could have
    kept the track alive.

``crossing``
    Another ground-truth object of the same class overlapped this one by more
    than ``crossing_iou`` during the window.  Geometry was ambiguous and the
    association picked the wrong one.  This is the case appearance features are
    supposed to fix.

``fast_motion``
    The object moved further than ``fast_motion_heights`` box heights per frame
    during the window, so the constant-velocity prediction fell outside the
    association gate.

``unattributed``
    None of the above.  A non-trivial residual here is a signal that the
    attribution rules are incomplete, so it is reported rather than folded into
    another bucket.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from ..track.matching import iou_matrix
from .mot import FrameEvent

__all__ = ["SwitchRecord", "SwitchAnalysis", "attribute_switches"]

CAUSES = ("detector_gap", "occlusion", "crossing", "fast_motion", "unattributed")


@dataclass
class SwitchRecord:
    frame_id: int
    gt_id: int
    old_hyp: int
    new_hyp: int
    cause: str
    gap_frames: int
    min_visibility: float
    max_neighbour_iou: float
    max_speed_heights: float


@dataclass
class SwitchAnalysis:
    records: list[SwitchRecord]
    counts: dict[str, int]
    window: int

    def as_dict(self) -> dict:
        return {
            "window": self.window,
            "total": len(self.records),
            "counts": dict(self.counts),
            "records": [asdict(r) for r in self.records],
        }


def _window_frames(frames: Sequence[int], frame_id: int, window: int) -> list[int]:
    return [f for f in frames if frame_id - window <= f <= frame_id]


def attribute_switches(
    events: Sequence[FrameEvent],
    gt_frames: dict[int, dict[int, np.ndarray]],
    gt_classes: dict[int, int],
    gt_visibility: dict[int, dict[int, float]] | None = None,
    *,
    window: int = 12,
    occlusion_visibility: float = 0.5,
    crossing_iou: float = 0.3,
    fast_motion_heights: float = 0.55,
) -> SwitchAnalysis:
    """Attribute every switch in ``events`` to exactly one cause.

    ``gt_classes`` maps ground-truth track id to class id and is needed for the
    crossing test, which only counts same-class neighbours: a person walking in
    front of a pallet does not confuse a class-aware tracker.
    """
    all_frames = sorted(gt_frames)
    matched_by_gt: dict[int, set[int]] = {}
    for event in events:
        for gt_id, _, _ in event.matches:
            matched_by_gt.setdefault(gt_id, set()).add(event.frame_id)

    records: list[SwitchRecord] = []
    for event in events:
        for gt_id, old_hyp, new_hyp in event.switches:
            frames = _window_frames(all_frames, event.frame_id, window)
            present = [f for f in frames if gt_id in gt_frames.get(f, {})]

            gap = 0
            for f in reversed(present[:-1]):
                if f in matched_by_gt.get(gt_id, set()):
                    break
                gap += 1

            visibilities = [
                (gt_visibility or {}).get(f, {}).get(gt_id, 1.0) for f in present
            ]
            min_visibility = float(min(visibilities)) if visibilities else 1.0

            max_neighbour = 0.0
            for f in present:
                frame = gt_frames[f]
                box = frame[gt_id][None, :]
                others = [
                    other
                    for other, other_box in frame.items()
                    if other != gt_id and gt_classes.get(other) == gt_classes.get(gt_id)
                ]
                if not others:
                    continue
                ious = iou_matrix(box, np.stack([frame[o] for o in others]))[0]
                max_neighbour = max(max_neighbour, float(ious.max()))

            max_speed = 0.0
            for a, b in zip(present[:-1], present[1:]):
                box_a, box_b = gt_frames[a][gt_id], gt_frames[b][gt_id]
                centre_a = ((box_a[0] + box_a[2]) / 2, (box_a[1] + box_a[3]) / 2)
                centre_b = ((box_b[0] + box_b[2]) / 2, (box_b[1] + box_b[3]) / 2)
                height = max(box_b[3] - box_b[1], 1.0)
                step = float(np.hypot(centre_b[0] - centre_a[0], centre_b[1] - centre_a[1]))
                max_speed = max(max_speed, step / height / max(b - a, 1))

            if gap > 0 and min_visibility >= occlusion_visibility:
                cause = "detector_gap"
            elif min_visibility < occlusion_visibility:
                cause = "occlusion"
            elif max_neighbour > crossing_iou:
                cause = "crossing"
            elif max_speed > fast_motion_heights:
                cause = "fast_motion"
            else:
                cause = "unattributed"

            records.append(
                SwitchRecord(
                    frame_id=event.frame_id,
                    gt_id=gt_id,
                    old_hyp=old_hyp,
                    new_hyp=new_hyp,
                    cause=cause,
                    gap_frames=gap,
                    min_visibility=round(min_visibility, 4),
                    max_neighbour_iou=round(max_neighbour, 4),
                    max_speed_heights=round(max_speed, 4),
                )
            )

    counts = {cause: 0 for cause in CAUSES}
    for record in records:
        counts[record.cause] += 1
    return SwitchAnalysis(records=records, counts=counts, window=window)

"""COCO-style detection metrics, implemented here rather than imported.

The reason is the precision comparison.  ``scripts/evaluate.py`` scores the
same weights exported at FP32, FP16 and INT8, and the whole point of the
headline table is that the three numbers are produced by identical code on
identical predictions.  Calling out to Ultralytics for one of them and to
pycocotools for another would put a metric implementation difference inside the
quantity being measured.

Agreement with other implementations.  ``tests/test_metrics_detection.py``
pins the arithmetic on cases small enough to check by hand, and
``tests/test_backends.py`` compares the FP32 number against Ultralytics' own
validator on the real model.  Those two will not agree exactly and are not
expected to: Ultralytics integrates the precision-recall curve with a trapezoid
over 101 interpolation points where COCO, and this module, take the mean of the
same 101 samples, and the two differ again in how a tie between two predictions
for one ground-truth box is broken.  The comparison uses a stated tolerance
rather than assuming equality, so a real regression cannot hide inside the
difference.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..track.matching import iou_matrix

__all__ = ["DetectionSample", "APResult", "average_precision", "evaluate_detections"]

IOU_THRESHOLDS = np.round(np.arange(0.5, 0.96, 0.05), 2)
RECALL_POINTS = np.linspace(0.0, 1.0, 101)


@dataclass
class DetectionSample:
    """Predictions and ground truth for one image."""

    pred_boxes: np.ndarray  # (n, 4) xyxy
    pred_scores: np.ndarray  # (n,)
    pred_classes: np.ndarray  # (n,)
    gt_boxes: np.ndarray  # (m, 4) xyxy
    gt_classes: np.ndarray  # (m,)


@dataclass
class APResult:
    map50: float
    map50_95: float
    per_class_ap50: dict[int, float] = field(default_factory=dict)
    per_class_ap50_95: dict[int, float] = field(default_factory=dict)
    precision: float = 0.0
    recall: float = 0.0
    n_gt: int = 0
    n_pred: int = 0

    def as_dict(self) -> dict:
        return {
            "map50": round(self.map50, 5),
            "map50_95": round(self.map50_95, 5),
            "precision": round(self.precision, 5),
            "recall": round(self.recall, 5),
            "n_gt": self.n_gt,
            "n_pred": self.n_pred,
            "per_class_ap50": {str(k): round(v, 5) for k, v in sorted(self.per_class_ap50.items())},
            "per_class_ap50_95": {
                str(k): round(v, 5) for k, v in sorted(self.per_class_ap50_95.items())
            },
        }


def average_precision(tp: np.ndarray, n_gt: int) -> float:
    """101-point interpolated AP from a score-sorted true-positive indicator.

    ``tp`` must already be ordered by descending confidence.  The interpolation
    is COCO's: precision is made monotonically non-increasing from the right
    before sampling at 101 evenly spaced recall points.
    """
    if n_gt == 0:
        return float("nan")
    if tp.size == 0:
        return 0.0
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(1 - tp)
    recall = cum_tp / n_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)

    # Monotone envelope, right to left.
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    indices = np.searchsorted(recall, RECALL_POINTS, side="left")
    sampled = np.zeros(RECALL_POINTS.size, dtype=np.float64)
    inside = indices < precision.size
    sampled[inside] = precision[indices[inside]]
    return float(sampled.mean())


def _match_one_class(
    samples: list[DetectionSample], class_id: int, iou_threshold: float
) -> tuple[np.ndarray, int]:
    """Greedy score-ordered matching for one class at one IoU threshold."""
    records: list[tuple[float, int, int]] = []  # score, image index, prediction index
    n_gt = 0
    gt_boxes_per_image: list[np.ndarray] = []
    for image_index, sample in enumerate(samples):
        gt_mask = sample.gt_classes == class_id
        gt_boxes_per_image.append(sample.gt_boxes[gt_mask])
        n_gt += int(gt_mask.sum())
        pred_mask = np.nonzero(sample.pred_classes == class_id)[0]
        for pred_index in pred_mask:
            records.append((float(sample.pred_scores[pred_index]), image_index, int(pred_index)))

    if not records:
        return np.zeros(0, dtype=np.float64), n_gt

    records.sort(key=lambda r: -r[0])
    used: dict[int, np.ndarray] = {
        i: np.zeros(boxes.shape[0], dtype=bool) for i, boxes in enumerate(gt_boxes_per_image)
    }
    tp = np.zeros(len(records), dtype=np.float64)
    for position, (_, image_index, pred_index) in enumerate(records):
        gt_boxes = gt_boxes_per_image[image_index]
        if gt_boxes.shape[0] == 0:
            continue
        pred_box = samples[image_index].pred_boxes[pred_index][None, :]
        ious = iou_matrix(pred_box, gt_boxes)[0]
        ious = np.where(used[image_index], -1.0, ious)
        best = int(np.argmax(ious))
        if ious[best] >= iou_threshold:
            used[image_index][best] = True
            tp[position] = 1.0
    return tp, n_gt


def evaluate_detections(
    samples: list[DetectionSample], class_ids: list[int], conf_for_pr: float = 0.25
) -> APResult:
    """mAP@0.5 and mAP@0.5:0.95, plus precision and recall at a fixed threshold.

    ``conf_for_pr`` only affects the reported precision and recall, which are
    the numbers a practitioner actually deploys at.  It has no effect on either
    mAP, which integrate over the whole confidence range.
    """
    per_class_ap: dict[int, list[float]] = {c: [] for c in class_ids}
    for iou_threshold in IOU_THRESHOLDS:
        for class_id in class_ids:
            tp, n_gt = _match_one_class(samples, class_id, float(iou_threshold))
            per_class_ap[class_id].append(average_precision(tp, n_gt))

    # A class with no ground truth anywhere has AP nan at every threshold, and
    # np.nanmean over an all-nan slice warns before returning nan.  Deciding
    # which classes are present first avoids the warning and makes the
    # exclusion explicit rather than a side effect of nan propagation.
    ap50 = {c: values[0] for c, values in per_class_ap.items()}
    present = [c for c in class_ids if not np.isnan(ap50[c])]
    ap_all = {c: float(np.nanmean(per_class_ap[c])) for c in present}

    tp_total = fp_total = 0
    n_gt_total = int(sum(int((s.gt_classes >= 0).sum()) for s in samples))
    n_pred_total = int(sum(int((s.pred_classes >= 0).sum()) for s in samples))
    for sample in samples:
        keep = sample.pred_scores >= conf_for_pr
        matched = np.zeros(sample.gt_boxes.shape[0], dtype=bool)
        order = np.argsort(-sample.pred_scores[keep])
        boxes = sample.pred_boxes[keep][order]
        classes = sample.pred_classes[keep][order]
        for box, cls in zip(boxes, classes):
            candidates = np.nonzero((sample.gt_classes == cls) & ~matched)[0]
            if candidates.size == 0:
                fp_total += 1
                continue
            ious = iou_matrix(box[None, :], sample.gt_boxes[candidates])[0]
            best = int(np.argmax(ious))
            if ious[best] >= 0.5:
                matched[candidates[best]] = True
                tp_total += 1
            else:
                fp_total += 1

    precision = tp_total / max(tp_total + fp_total, 1)
    recall = tp_total / max(n_gt_total, 1)

    return APResult(
        map50=float(np.mean([ap50[c] for c in present])) if present else 0.0,
        map50_95=float(np.mean([ap_all[c] for c in present])) if present else 0.0,
        per_class_ap50={c: ap50[c] for c in present},
        per_class_ap50_95={c: ap_all[c] for c in present},
        precision=float(precision),
        recall=float(recall),
        n_gt=n_gt_total,
        n_pred=n_pred_total,
    )

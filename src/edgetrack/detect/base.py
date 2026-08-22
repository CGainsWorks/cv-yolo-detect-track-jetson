"""Detector interface plus the pre and post processing every backend shares.

The backends differ in one thing only: how a float tensor becomes a raw output
tensor.  Letterboxing, normalisation, decoding and NMS are identical across
ONNX Runtime, TensorRT and PyTorch, and they live here so that a difference in
mAP between FP32 and INT8 is a difference in the arithmetic and not a
difference in how the boxes were rescaled.  ``tests/test_backends.py`` asserts
that the backends agree box for box on the same input.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..paths import serialise_path
from ..types import Detection

__all__ = [
    "DetectorConfig",
    "Detector",
    "letterbox",
    "undo_letterbox",
    "nms",
    "decode_yolov8",
]


@dataclass
class DetectorConfig:
    imgsz: tuple[int, int] = (640, 384)  # width, height
    conf_threshold: float = 0.25
    iou_threshold: float = 0.6
    max_detections: int = 100
    class_names: tuple[str, ...] = ()
    agnostic_nms: bool = False

    def validate(self) -> None:
        w, h = self.imgsz
        if w % 32 or h % 32:
            raise ValueError(
                f"imgsz must be a multiple of 32 in both axes for a YOLOv8 stride-32 "
                f"head, got {self.imgsz}"
            )
        if not 0.0 < self.conf_threshold < 1.0:
            raise ValueError(f"conf_threshold must be in (0, 1), got {self.conf_threshold}")


def letterbox(
    image: np.ndarray, size: tuple[int, int], colour: int = 114
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize preserving aspect ratio and pad to ``size`` (width, height).

    Returns the padded image, the scale factor and the padding applied to the
    left and top.  Padding is split evenly, which is what Ultralytics does at
    inference time; putting it all on one side shifts every box by half the pad
    and costs about a point of mAP on wide images.
    """
    target_w, target_h = size
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw, dh = (target_w - new_w) / 2.0, (target_h - new_h) / 2.0
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(colour,) * 3
    )
    return padded, scale, (left, top)


def undo_letterbox(
    boxes: np.ndarray, scale: float, pad: tuple[float, float], shape: tuple[int, int]
) -> np.ndarray:
    """Map ``xyxy`` boxes from letterboxed space back to the original image."""
    if boxes.size == 0:
        return boxes.reshape(-1, 4)
    out = boxes.copy().astype(np.float64)
    out[:, [0, 2]] -= pad[0]
    out[:, [1, 3]] -= pad[1]
    out /= max(scale, 1e-9)
    h, w = shape
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, w)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, h)
    return out


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy non-maximum suppression, returning kept indices in score order.

    Written out rather than calling ``cv2.dnn.NMSBoxes`` because that function
    rounds boxes to integers internally, which changes which of two nearly
    identical boxes survives and made the ONNX and TensorRT outputs disagree on
    about one detection in two hundred.
    """
    if boxes.shape[0] == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
        order = rest[iou <= iou_threshold]
    return keep


def decode_yolov8(
    output: np.ndarray,
    config: DetectorConfig,
    scale: float,
    pad: tuple[float, float],
    shape: tuple[int, int],
) -> list[Detection]:
    """Turn a raw YOLOv8 head output into detections in original-image pixels.

    The export produces ``(1, 4 + num_classes, num_anchors)`` with boxes in
    ``cxcywh`` on the letterboxed image and per-class scores already through
    the sigmoid.  There is no objectness channel; that is the v8 change from v5
    and reading the tensor as if there were one silently shifts every class by
    one index.
    """
    array = np.asarray(output)
    if array.ndim == 3:
        array = array[0]
    if array.shape[0] < array.shape[1]:
        array = array.T  # (anchors, 4 + nc)
    if array.shape[1] < 5:
        raise ValueError(
            f"detector output has {array.shape[1]} channels; expected 4 box "
            "coordinates plus at least one class score"
        )

    boxes_cxcywh = array[:, :4]
    class_scores = array[:, 4:]
    class_ids = class_scores.argmax(axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]

    keep = scores >= config.conf_threshold
    if not keep.any():
        return []
    boxes_cxcywh = boxes_cxcywh[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]

    xyxy = np.empty_like(boxes_cxcywh)
    xyxy[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
    xyxy[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
    xyxy[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
    xyxy[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2
    xyxy = undo_letterbox(xyxy, scale, pad, shape)

    detections: list[Detection] = []
    groups = [None] if config.agnostic_nms else sorted(set(class_ids.tolist()))
    for group in groups:
        mask = np.ones(len(scores), dtype=bool) if group is None else class_ids == group
        indices = np.nonzero(mask)[0]
        kept = nms(xyxy[indices], scores[indices], config.iou_threshold)
        for k in kept:
            index = int(indices[k])
            detections.append(
                Detection(
                    tuple(float(v) for v in xyxy[index]),  # type: ignore[arg-type]
                    float(scores[index]),
                    int(class_ids[index]),
                )
            )
    detections.sort(key=lambda d: -d.score)
    return detections[: config.max_detections]


@dataclass
class BackendInfo:
    backend: str
    precision: str
    model_path: str
    providers: tuple[str, ...] = ()
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "backend": self.backend,
            "precision": self.precision,
            # POSIX and repo-relative, so a results file written on
            # Windows joins correctly on a Linux CI runner.  See
            # edgetrack.paths for what goes wrong otherwise.
            "model_path": serialise_path(self.model_path),
            "providers": list(self.providers),
            **self.extra,
        }


class Detector(ABC):
    """Common lifecycle for every backend."""

    def __init__(self, config: DetectorConfig) -> None:
        config.validate()
        self.config = config
        self.last_inference_ms = 0.0
        self.last_preprocess_ms = 0.0
        self.last_postprocess_ms = 0.0

    @abstractmethod
    def _forward(self, tensor: np.ndarray) -> np.ndarray:
        """Run the network on an NCHW float tensor and return the raw head."""

    @property
    @abstractmethod
    def info(self) -> BackendInfo:
        ...

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, tuple[float, float]]:
        padded, scale, pad = letterbox(image, self.config.imgsz)
        tensor = padded[:, :, ::-1].astype(np.float32) / 255.0  # BGR to RGB
        tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)[None])
        return tensor, scale, pad

    def detect(self, image: np.ndarray) -> list[Detection]:
        start = time.perf_counter()
        tensor, scale, pad = self.preprocess(image)
        mid = time.perf_counter()
        output = self._forward(tensor)
        after = time.perf_counter()
        detections = decode_yolov8(output, self.config, scale, pad, image.shape[:2])
        end = time.perf_counter()
        self.last_preprocess_ms = (mid - start) * 1000.0
        self.last_inference_ms = (after - mid) * 1000.0
        self.last_postprocess_ms = (end - after) * 1000.0
        return detections

    def warmup(self, iterations: int = 5) -> None:
        """Run the network on a blank frame.

        Not optional on any accelerated backend.  The first ONNX Runtime CUDA
        call allocates workspace and picks convolution algorithms, and the
        first TensorRT call pages the engine in; without this the first
        measured frame is five to fifty times the steady-state latency and the
        benchmark reports nonsense.
        """
        w, h = self.config.imgsz
        blank = np.zeros((h, w, 3), dtype=np.uint8)
        for _ in range(max(0, iterations)):
            self.detect(blank)

    def close(self) -> None:
        """Release backend resources.  Safe to call more than once."""

    def __enter__(self) -> "Detector":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

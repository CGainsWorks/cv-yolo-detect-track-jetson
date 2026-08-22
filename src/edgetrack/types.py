"""Core data types shared by the detector, the trackers and the metrics.

Everything here is plain numpy and dataclasses.  Nothing in this module imports
torch, ultralytics, onnxruntime or rclpy, so it can be used from the runtime
image, from the tests and from the ROS2 node without dragging a toolchain in.

Box convention.  There is exactly one box convention in this repository and it
is ``xyxy`` in absolute pixels on the original (un-letterboxed) image, with
``x2``/``y2`` exclusive.  Conversions to and from the other conventions live in
:mod:`edgetrack.track.matching` and in the Kalman filter, which works in
``xyah`` because that is what the constant velocity model was derived for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "Detection",
    "Track",
    "FrameResult",
    "boxes_to_array",
    "array_to_boxes",
]


@dataclass(frozen=True)
class Detection:
    """One detected object in one frame.

    Attributes
    ----------
    xyxy:
        Box in absolute pixels, ``(x1, y1, x2, y2)``.
    score:
        Detector confidence in ``[0, 1]``.
    class_id:
        Index into the dataset class list.
    feature:
        Optional appearance descriptor, L2 normalised.  ByteTrack ignores it.
        DeepSORT needs it and will raise if it is missing.
    """

    xyxy: tuple[float, float, float, float]
    score: float
    class_id: int
    feature: np.ndarray | None = None

    @property
    def width(self) -> float:
        return float(self.xyxy[2] - self.xyxy[0])

    @property
    def height(self) -> float:
        return float(self.xyxy[3] - self.xyxy[1])

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def xyah(self) -> np.ndarray:
        """Centre x, centre y, aspect ratio, height.  The Kalman measurement."""
        x1, y1, x2, y2 = self.xyxy
        h = y2 - y1
        w = x2 - x1
        return np.array(
            [x1 + w / 2.0, y1 + h / 2.0, w / max(h, 1e-6), h], dtype=np.float64
        )


@dataclass
class Track:
    """A persistent identity across frames.

    ``track_id`` is stable for the lifetime of the track and is never reused.
    ``state`` is one of ``tentative``, ``confirmed`` or ``deleted``; ByteTrack
    calls the same three things new/tracked/removed and the mapping is exact.
    """

    track_id: int
    class_id: int
    xyxy: tuple[float, float, float, float]
    score: float
    state: str = "tentative"
    age: int = 0
    hits: int = 1
    time_since_update: int = 0
    start_frame: int = 0
    frame_id: int = 0
    history: list[tuple[int, tuple[float, float, float, float]]] = field(
        default_factory=list
    )

    @property
    def is_confirmed(self) -> bool:
        return self.state == "confirmed"

    @property
    def centre(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def trajectory(self) -> np.ndarray:
        """``(n, 3)`` array of ``frame, cx, cy`` for everything recorded so far."""
        if not self.history:
            return np.zeros((0, 3), dtype=np.float64)
        rows = []
        for frame_id, (x1, y1, x2, y2) in self.history:
            rows.append((frame_id, (x1 + x2) / 2.0, (y1 + y2) / 2.0))
        return np.asarray(rows, dtype=np.float64)


@dataclass
class FrameResult:
    """What the pipeline produces for a single frame."""

    frame_id: int
    detections: list[Detection]
    tracks: list[Track]
    timings_ms: dict[str, float] = field(default_factory=dict)

    def as_mot_rows(self) -> list[tuple[int, int, float, float, float, float, float]]:
        """Rows in MOTChallenge order: frame, id, x, y, w, h, score.

        MOTChallenge frames are one-based.  ``frame_id`` is zero-based
        everywhere inside this repository, so the conversion happens here and
        nowhere else.
        """
        rows = []
        for track in self.tracks:
            if not track.is_confirmed:
                continue
            x1, y1, x2, y2 = track.xyxy
            rows.append(
                (
                    self.frame_id + 1,
                    track.track_id,
                    float(x1),
                    float(y1),
                    float(x2 - x1),
                    float(y2 - y1),
                    float(track.score),
                )
            )
        return rows


def boxes_to_array(detections: Sequence[Detection]) -> np.ndarray:
    """``(n, 4)`` float array of xyxy boxes.  Empty input gives ``(0, 4)``."""
    if not detections:
        return np.zeros((0, 4), dtype=np.float64)
    return np.asarray([d.xyxy for d in detections], dtype=np.float64)


def array_to_boxes(
    array: np.ndarray,
    scores: Iterable[float] | None = None,
    class_ids: Iterable[int] | None = None,
) -> list[Detection]:
    """Inverse of :func:`boxes_to_array`, for tests and for the ONNX decoder."""
    array = np.asarray(array, dtype=np.float64).reshape(-1, 4)
    n = array.shape[0]
    score_list = list(scores) if scores is not None else [1.0] * n
    class_list = list(class_ids) if class_ids is not None else [0] * n
    if len(score_list) != n or len(class_list) != n:
        raise ValueError(
            f"expected {n} scores and class ids, got {len(score_list)} and "
            f"{len(class_list)}"
        )
    return [
        Detection(tuple(float(v) for v in row), float(s), int(c))  # type: ignore[arg-type]
        for row, s, c in zip(array, score_list, class_list)
    ]

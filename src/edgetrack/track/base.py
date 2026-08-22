"""Shared track bookkeeping for both trackers.

The two trackers differ only in association.  Everything else, the Kalman
state, the confirmation rule, the ID allocator and the conversion out to
:class:`edgetrack.types.Track`, is here so that a difference in results is
attributable to the association strategy alone.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Iterable, Sequence

import numpy as np

from ..types import Detection, Track
from .kalman import KalmanFilter, xyah_to_xyxy

__all__ = ["KalmanTrack", "BaseTracker"]


class KalmanTrack:
    """One tracked identity, with its filter state and appearance gallery."""

    __slots__ = (
        "track_id",
        "class_id",
        "score",
        "mean",
        "covariance",
        "state",
        "hits",
        "age",
        "time_since_update",
        "start_frame",
        "frame_id",
        "features",
        "history",
        "_kf",
    )

    def __init__(
        self,
        kf: KalmanFilter,
        track_id: int,
        detection: Detection,
        frame_id: int,
        feature_gallery: int = 64,
    ) -> None:
        self._kf = kf
        self.track_id = track_id
        self.class_id = detection.class_id
        self.score = float(detection.score)
        self.mean, self.covariance = kf.initiate(detection.xyah)
        self.state = "tentative"
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.start_frame = frame_id
        self.frame_id = frame_id
        self.features: deque[np.ndarray] = deque(maxlen=feature_gallery)
        if detection.feature is not None:
            self.features.append(np.asarray(detection.feature, dtype=np.float64))
        self.history: list[tuple[int, tuple[float, float, float, float]]] = [
            (frame_id, self.xyxy)
        ]

    # -- geometry ---------------------------------------------------------

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return xyah_to_xyxy(self.mean)

    @property
    def xyah(self) -> np.ndarray:
        return self.mean[:4].copy()

    @property
    def is_confirmed(self) -> bool:
        return self.state == "confirmed"

    @property
    def is_deleted(self) -> bool:
        return self.state == "deleted"

    # -- lifecycle --------------------------------------------------------

    def predict(self) -> None:
        # A track that is not being updated has no reliable velocity in the
        # aspect and height channels; zeroing them stops a lost box from
        # inflating or collapsing while it coasts, which is the single most
        # common cause of a bogus re-association after a long occlusion.
        if self.time_since_update > 0:
            self.mean[6] = 0.0
            self.mean[7] = 0.0
        self.mean, self.covariance = self._kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, detection: Detection, frame_id: int, n_init: int) -> None:
        self.mean, self.covariance = self._kf.update(
            self.mean, self.covariance, detection.xyah, detection.score
        )
        self.score = float(detection.score)
        self.class_id = detection.class_id
        self.hits += 1
        self.time_since_update = 0
        self.frame_id = frame_id
        if detection.feature is not None:
            self.features.append(np.asarray(detection.feature, dtype=np.float64))
        if self.state == "tentative" and self.hits >= n_init:
            self.state = "confirmed"
        self.history.append((frame_id, self.xyxy))

    def mark_missed(self, max_age: int, n_init: int) -> None:
        if self.state == "tentative" and self.hits < n_init:
            self.state = "deleted"
        elif self.time_since_update > max_age:
            self.state = "deleted"

    def to_public(self) -> Track:
        return Track(
            track_id=self.track_id,
            class_id=self.class_id,
            xyxy=self.xyxy,
            score=self.score,
            state=self.state,
            age=self.age,
            hits=self.hits,
            time_since_update=self.time_since_update,
            start_frame=self.start_frame,
            frame_id=self.frame_id,
            history=list(self.history),
        )


class BaseTracker(ABC):
    """Common interface and ID allocator."""

    def __init__(self) -> None:
        self._next_id = 1
        self.frame_id = -1
        self.tracks: list[KalmanTrack] = []

    def reset(self) -> None:
        """Clear all state.  IDs restart at one, which is what makes a rerun
        of the same sequence produce the same ID sequence."""
        self._next_id = 1
        self.frame_id = -1
        self.tracks = []

    def _allocate_id(self) -> int:
        value = self._next_id
        self._next_id += 1
        return value

    @abstractmethod
    def update(self, detections: Sequence[Detection], frame_id: int | None = None) -> list[Track]:
        """Advance one frame and return the currently visible tracks."""

    @staticmethod
    def _boxes(items: Iterable) -> np.ndarray:
        rows = [item.xyxy for item in items]
        return np.asarray(rows, dtype=np.float64).reshape(-1, 4)

    @property
    def name(self) -> str:
        return type(self).__name__

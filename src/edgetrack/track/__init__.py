"""Trackers, their shared Kalman filter and the appearance encoders."""

from .appearance import HistogramEncoder, attach_features, build_encoder
from .bytetrack import ByteTrack, ByteTrackConfig
from .deepsort import DeepSort, DeepSortConfig
from .kalman import KalmanFilter

__all__ = [
    "ByteTrack",
    "ByteTrackConfig",
    "DeepSort",
    "DeepSortConfig",
    "KalmanFilter",
    "HistogramEncoder",
    "build_encoder",
    "attach_features",
    "build_tracker",
]


def build_tracker(name: str, **kwargs):
    """Factory so that ``--tracker`` means the same thing in every script."""
    key = name.lower().replace("-", "").replace("_", "")
    if key == "bytetrack":
        return ByteTrack(ByteTrackConfig(**kwargs))
    if key in ("deepsort", "deep"):
        return DeepSort(DeepSortConfig(**kwargs))
    raise ValueError(f"unknown tracker {name!r}; expected bytetrack or deepsort")

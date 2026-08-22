"""Payload builders for the ROS2 node, with no ROS dependency.

Splitting the conversion from the publishing is what makes the node testable
off a ROS machine.  Everything here is plain Python: it turns a
:class:`~edgetrack.types.FrameResult` into dictionaries whose keys match the
message fields, and :mod:`edgetrack.ros2.node` copies them across.
``tests/test_ros2_messages.py`` runs on any machine; the CI job that actually
brings up rclpy and subscribes is ``ros2`` in ``.github/workflows/ci.yml``.

Field naming note.  ``vision_msgs`` changed shape between the 3.x series and
the 4.x series that ships with Humble: ``ObjectHypothesisWithPose.id`` became
``hypothesis.class_id`` and turned from an integer into a string, and
``BoundingBox2D.center`` gained a nested ``position``.  The node handles both
by duck typing rather than by pinning a version, because a Jetson image with
whatever ``vision_msgs`` its Jetpack shipped is the normal case.
"""

from __future__ import annotations

from typing import Sequence

from ..types import FrameResult, Track

__all__ = [
    "detection_payloads",
    "track_payloads",
    "trajectory_payloads",
    "image_payload",
]


def detection_payloads(result: FrameResult, class_names: Sequence[str] = ()) -> list[dict]:
    """One dictionary per raw detection.

    ``id`` is empty for detections.  Only tracks carry an identity, and
    publishing a detection with a fabricated id would let a subscriber believe
    the detector is doing association.
    """
    out = []
    for detection in result.detections:
        x1, y1, x2, y2 = detection.xyxy
        name = (
            class_names[detection.class_id]
            if detection.class_id < len(class_names)
            else str(detection.class_id)
        )
        out.append(
            {
                "id": "",
                "class_id": name,
                "class_index": int(detection.class_id),
                "score": float(detection.score),
                "center_x": (x1 + x2) / 2.0,
                "center_y": (y1 + y2) / 2.0,
                "size_x": x2 - x1,
                "size_y": y2 - y1,
            }
        )
    return out


def track_payloads(result: FrameResult, class_names: Sequence[str] = ()) -> list[dict]:
    """One dictionary per confirmed track, carrying the persistent identity."""
    out = []
    for track in result.tracks:
        if not track.is_confirmed:
            continue
        x1, y1, x2, y2 = track.xyxy
        name = (
            class_names[track.class_id]
            if track.class_id < len(class_names)
            else str(track.class_id)
        )
        out.append(
            {
                "id": str(track.track_id),
                "class_id": name,
                "class_index": int(track.class_id),
                "score": float(track.score),
                "center_x": (x1 + x2) / 2.0,
                "center_y": (y1 + y2) / 2.0,
                "size_x": x2 - x1,
                "size_y": y2 - y1,
                "age": int(track.age),
                "hits": int(track.hits),
            }
        )
    return out


def trajectory_payloads(tracks: Sequence[Track], trail: int = 30) -> list[dict]:
    """Polyline per track for a ``visualization_msgs/MarkerArray``.

    Points are in image pixels with z fixed at zero.  A real deployment would
    project them to a ground plane with the camera calibration; that is not in
    this repository and the marker frame is named ``camera_optical`` to make
    the pixel convention explicit rather than to imply metres.
    """
    out = []
    for track in tracks:
        points = track.trajectory()
        if points.shape[0] < 2:
            continue
        tail = points[-trail:]
        out.append(
            {
                "id": int(track.track_id),
                "class_index": int(track.class_id),
                "points": [(float(x), float(y), 0.0) for _, x, y in tail],
            }
        )
    return out


def image_payload(image) -> dict:
    """Fields for a ``sensor_msgs/Image`` carrying a BGR frame.

    Built by hand rather than through ``cv_bridge`` so the node has one fewer
    apt dependency, which matters on a Jetson image where ``cv_bridge`` pulls a
    second OpenCV in beside the one already installed.
    """
    height, width = image.shape[:2]
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 BGR image, got shape {image.shape}")
    contiguous = image if image.flags["C_CONTIGUOUS"] else image.copy()
    return {
        "height": int(height),
        "width": int(width),
        "encoding": "bgr8",
        "is_bigendian": 0,
        "step": int(width * 3),
        "data": contiguous.tobytes(),
    }

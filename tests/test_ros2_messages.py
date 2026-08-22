"""The ROS2 payload builders, which need no ROS installation.

The node itself is exercised by the ``ros2`` job in CI, which brings up a real
rclpy inside a ``ros:humble`` container and subscribes to the topics.  These
tests cover the conversion, which is where the field-name and unit mistakes
live.
"""

from __future__ import annotations

import numpy as np
import pytest

from edgetrack.ros2.messages import (
    detection_payloads,
    image_payload,
    track_payloads,
    trajectory_payloads,
)
from edgetrack.types import Detection, FrameResult, Track

CLASSES = ("person", "forklift", "pallet", "cone")


def result_with(detections=(), tracks=()):
    return FrameResult(3, list(detections), list(tracks))


def test_detection_payload_geometry():
    detection = Detection((100.0, 200.0, 140.0, 320.0), 0.75, 1)
    payload = detection_payloads(result_with([detection]), CLASSES)[0]
    assert payload["center_x"] == pytest.approx(120.0)
    assert payload["center_y"] == pytest.approx(260.0)
    assert payload["size_x"] == pytest.approx(40.0)
    assert payload["size_y"] == pytest.approx(120.0)
    assert payload["class_id"] == "forklift"
    assert payload["class_index"] == 1
    assert payload["score"] == pytest.approx(0.75)


def test_detections_carry_no_identity():
    """A detection has no identity and must not pretend to have one.

    A subscriber reading ``id`` off the detections topic and finding a number
    there would reasonably conclude the detector does association.
    """
    detection = Detection((0.0, 0.0, 10.0, 10.0), 0.9, 0)
    assert detection_payloads(result_with([detection]), CLASSES)[0]["id"] == ""


def test_track_payload_carries_the_identity_as_a_string():
    track = Track(7, 0, (10.0, 20.0, 50.0, 100.0), 0.8, state="confirmed", age=12, hits=9)
    payload = track_payloads(result_with(tracks=[track]), CLASSES)[0]
    assert payload["id"] == "7"
    assert isinstance(payload["id"], str), "vision_msgs 4.x declares Detection2D.id a string"
    assert payload["class_id"] == "person"
    assert payload["age"] == 12
    assert payload["hits"] == 9


def test_tentative_tracks_are_not_published():
    tentative = Track(1, 0, (0.0, 0.0, 10.0, 10.0), 0.9, state="tentative")
    confirmed = Track(2, 0, (0.0, 0.0, 10.0, 10.0), 0.9, state="confirmed")
    payloads = track_payloads(result_with(tracks=[tentative, confirmed]), CLASSES)
    assert [p["id"] for p in payloads] == ["2"]


def test_an_out_of_range_class_index_falls_back_to_the_number():
    detection = Detection((0.0, 0.0, 10.0, 10.0), 0.9, 9)
    assert detection_payloads(result_with([detection]), CLASSES)[0]["class_id"] == "9"


def test_trajectory_payload_needs_at_least_two_points():
    short = Track(1, 0, (0.0, 0.0, 10.0, 10.0), 0.9, history=[(0, (0.0, 0.0, 10.0, 10.0))])
    assert trajectory_payloads([short]) == []

    longer = Track(
        1, 0, (0.0, 0.0, 10.0, 10.0), 0.9,
        history=[(t, (float(t), 0.0, float(t) + 10, 10.0)) for t in range(5)],
    )
    payload = trajectory_payloads([longer])[0]
    assert payload["id"] == 1
    assert len(payload["points"]) == 5
    assert all(len(point) == 3 and point[2] == 0.0 for point in payload["points"])


def test_trajectory_payload_respects_the_trail_length():
    track = Track(
        1, 0, (0.0, 0.0, 10.0, 10.0), 0.9,
        history=[(t, (float(t), 0.0, float(t) + 10, 10.0)) for t in range(100)],
    )
    assert len(trajectory_payloads([track], trail=12)[0]["points"]) == 12


def test_image_payload_matches_the_sensor_msgs_contract():
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[0, 0] = (1, 2, 3)
    payload = image_payload(image)
    assert payload["height"] == 48
    assert payload["width"] == 64
    assert payload["encoding"] == "bgr8"
    assert payload["step"] == 64 * 3
    assert len(payload["data"]) == 48 * 64 * 3
    assert payload["data"][:3] == bytes((1, 2, 3))


def test_image_payload_handles_a_non_contiguous_array():
    """A cropped or sliced frame is not contiguous and ``tobytes`` would
    otherwise serialise the wrong memory."""
    big = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
    view = big[8:40, 8:40]
    assert not view.flags["C_CONTIGUOUS"]
    payload = image_payload(view)
    assert len(payload["data"]) == 32 * 32 * 3
    assert payload["data"] == view.copy().tobytes()


def test_image_payload_rejects_a_non_colour_image():
    with pytest.raises(ValueError, match="HxWx3"):
        image_payload(np.zeros((10, 10), dtype=np.uint8))


def test_empty_result_produces_empty_payloads():
    empty = result_with()
    assert detection_payloads(empty, CLASSES) == []
    assert track_payloads(empty, CLASSES) == []
    assert trajectory_payloads([]) == []

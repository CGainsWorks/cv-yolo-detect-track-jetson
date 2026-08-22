#!/usr/bin/env python3
"""Bring up the real node under a real rclpy and count what arrives.

    source /opt/ros/humble/setup.bash
    PYTHONPATH=src python3 tests/ros2_subscriber_check.py

Not a pytest test, because it needs a sourced ROS2 installation and pytest
collection should stay runnable on a machine without one.  The ``ros2`` job in
``.github/workflows/ci.yml`` runs it inside a ``ros:humble-ros-base`` container.

What it proves.  The node and the subscriber run in one process under a
multi-threaded executor, so the messages travel through the real middleware
rather than through a Python call.  It checks the message counts, the geometry,
the identity field and the image encoding, which between them cover every
mistake the payload builders could make that the unit tests cannot see.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import rclpy  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import (  # noqa: E402
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import Image  # noqa: E402
from vision_msgs.msg import Detection2DArray  # noqa: E402

from edgetrack.ros2.node import _build_node_class  # noqa: E402

FRAMES = 30
MODEL = ROOT / "models" / "yolov8n_edgetrack_fp32.onnx"

SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    durability=QoSDurabilityPolicy.VOLATILE,
    depth=10,
)


class Listener(Node):
    def __init__(self) -> None:
        super().__init__("edgetrack_listener")
        self.detections: list[Detection2DArray] = []
        self.tracks: list[Detection2DArray] = []
        self.images: list[Image] = []
        self.create_subscription(
            Detection2DArray, "/edgetrack/detections", self.detections.append, SENSOR_QOS
        )
        self.create_subscription(
            Detection2DArray, "/edgetrack/tracks", self.tracks.append, SENSOR_QOS
        )
        self.create_subscription(
            Image, "/edgetrack/image_annotated", self.images.append, SENSOR_QOS
        )


def centre_of(bbox) -> tuple[float, float]:
    centre = bbox.center
    if hasattr(centre, "position"):  # vision_msgs 4.x
        return float(centre.position.x), float(centre.position.y)
    return float(centre.x), float(centre.y)


def main() -> int:
    if not MODEL.exists():
        print(f"no model at {MODEL}; run scripts/fetch_model.py first", file=sys.stderr)
        return 1

    rclpy.init()
    node_class = _build_node_class()

    # Parameters are read at construction, so the overrides go in through
    # declare_parameter, which is the same path `ros2 run --ros-args -p`
    # takes.
    class ConfiguredNode(node_class):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            self._overrides = {
                "source": "synth:7",
                "model": str(MODEL),
                "tracker": "bytetrack",
                "max_frames": FRAMES,
                "rate_hz": 60.0,
                "publish_image": True,
                "shutdown_when_done": False,
            }
            super().__init__()

        def declare_parameter(self, name, value=None, *args, **kwargs):  # noqa: D102
            if name in self._overrides:
                value = self._overrides[name]
            return super().declare_parameter(name, value, *args, **kwargs)

    node = ConfiguredNode()
    listener = Listener()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.add_node(listener)

    deadline = node.get_clock().now().nanoseconds + int(90e9)
    try:
        while node.get_clock().now().nanoseconds < deadline:
            executor.spin_once(timeout_sec=0.1)
            if node.frame_index >= FRAMES and len(listener.tracks) >= FRAMES:
                break
    finally:
        # Cancel the timer before tearing anything down.  Without this the
        # callback fires once more while destruction is in progress and rclpy
        # reports an unretrieved "cannot use Destroyable" exception after the
        # check has already passed, which reads like a failure and is not one.
        # Cancel the timer and take both nodes off the executor before
        # destroying them.  rclpy may still print "cannot use Destroyable
        # because destruction was requested" after this: it is an unretrieved
        # exception from a callback already in flight inside the executor, it
        # arrives after the check has passed and the process still exits zero.
        # Two attempts at suppressing it changed nothing, so it is documented
        # here rather than chased, and CI is right to keep passing on it.
        node.timer.cancel()
        executor.remove_node(node)
        executor.remove_node(listener)
        executor.shutdown()
        listener.destroy_node()
        node.destroy_node()
        rclpy.shutdown()

    print(f"frames processed:  {node.frame_index}")
    print(f"detection arrays:  {len(listener.detections)}")
    print(f"track arrays:      {len(listener.tracks)}")
    print(f"annotated images:  {len(listener.images)}")

    assert node.frame_index == FRAMES, node.frame_index
    # Best-effort QoS may drop under load, so the bar is most rather than all.
    for name, received in (
        ("detections", listener.detections),
        ("tracks", listener.tracks),
        ("images", listener.images),
    ):
        assert len(received) >= FRAMES * 0.8, f"only {len(received)} {name} arrived"

    total_detections = sum(len(msg.detections) for msg in listener.detections)
    total_tracks = sum(len(msg.detections) for msg in listener.tracks)
    print(f"detections total:  {total_detections}")
    print(f"tracks total:      {total_tracks}")
    assert total_detections > FRAMES, "the detector found almost nothing"
    assert total_tracks > 0, "no track was ever published"

    identities = {
        detection.id
        for msg in listener.tracks
        for detection in msg.detections
    }
    print(f"distinct track ids: {sorted(identities)}")
    assert identities, "tracks were published with no identity"
    assert all(value and value.isdigit() for value in identities), identities

    # Detections carry no identity; only tracks do.
    assert all(
        detection.id == ""
        for msg in listener.detections
        for detection in msg.detections
    ), "a raw detection was published with a track identity"

    sample = next(
        detection for msg in listener.tracks for detection in msg.detections
    )
    x, y = centre_of(sample.bbox)
    print(f"sample box: centre ({x:.1f}, {y:.1f}) size {sample.bbox.size_x:.1f} "
          f"x {sample.bbox.size_y:.1f}")
    assert 0 <= x <= 640 and 0 <= y <= 384, (x, y)
    assert sample.bbox.size_x > 0 and sample.bbox.size_y > 0
    assert sample.results, "a detection was published with no hypothesis"

    image = listener.images[0]
    print(f"image: {image.width}x{image.height} {image.encoding}")
    assert (image.width, image.height) == (640, 384)
    assert image.encoding == "bgr8"
    assert image.step == image.width * 3
    assert len(image.data) == image.height * image.step

    frames = {msg.header.frame_id for msg in listener.tracks}
    assert frames == {"camera_optical"}, frames

    print("ros2 node check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

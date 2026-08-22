"""ROS2 node publishing detections, tracks and an annotated image.

Topics, all relative to the node name so two instances can run side by side:

===========================  ====================================
``~/detections``             ``vision_msgs/Detection2DArray``
``~/tracks``                 ``vision_msgs/Detection2DArray``
``~/trajectories``           ``visualization_msgs/MarkerArray``
``~/image_annotated``        ``sensor_msgs/Image``, ``bgr8``
===========================  ====================================

Detections and tracks are separate topics carrying the same message type.  A
consumer that only wants "what is in front of me right now" subscribes to the
first and never has to reason about identity; a consumer doing anything
temporal subscribes to the second, where the ``id`` field is the persistent
track identity as a string.

Running it::

    ros2 run edgetrack_ros track_node --ros-args \\
        -p source:=synth:7 -p model:=models/yolov8n_edgetrack_fp32.onnx

or, without a ROS package install, straight from the repository::

    python -m edgetrack.ros2.node --ros-args -p source:=synth:7

The node exits by itself when a finite source runs out, which is what makes it
usable from a CI job and from a shell script without a timeout wrapper.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ..detect import DetectorConfig, build_detector
from ..pipeline import Pipeline, annotate
from ..track import build_tracker
from ..track.appearance import build_encoder
from ..video import open_source
from .messages import detection_payloads, image_payload, track_payloads, trajectory_payloads

__all__ = ["EdgeTrackNode", "main"]

_MISSING_ROS = (
    "rclpy is not importable.  This node needs a sourced ROS2 installation; "
    "on Ubuntu that is `source /opt/ros/humble/setup.bash`.  Everything else in "
    "this repository runs without ROS, and docs/ros2.md has a container recipe."
)


def _require_ros():
    try:
        import rclpy  # noqa: F401
        from rclpy.node import Node  # noqa: F401
    except ImportError as error:  # pragma: no cover - exercised only off ROS
        raise ImportError(_MISSING_ROS) from error


def _set_hypothesis(hypothesis, class_id: str, score: float) -> None:
    """Fill an ``ObjectHypothesisWithPose`` across the 3.x and 4.x layouts."""
    if hasattr(hypothesis, "hypothesis"):  # vision_msgs 4.x
        hypothesis.hypothesis.class_id = str(class_id)
        hypothesis.hypothesis.score = float(score)
    else:  # vision_msgs 3.x
        try:
            hypothesis.id = int(class_id)
        except (TypeError, ValueError):
            hypothesis.id = 0
        hypothesis.score = float(score)


def _set_centre(bbox, x: float, y: float) -> None:
    """Fill ``BoundingBox2D.center`` across the 3.x and 4.x layouts."""
    centre = bbox.center
    if hasattr(centre, "position"):  # vision_msgs 4.x
        centre.position.x = float(x)
        centre.position.y = float(y)
    else:  # vision_msgs 3.x, geometry_msgs/Pose2D
        centre.x = float(x)
        centre.y = float(y)


def _build_node_class():
    """Define the node class only once rclpy is known to be importable.

    Subclassing ``rclpy.node.Node`` at module scope would make this module
    unimportable without ROS, and ``tests/test_ros2_messages.py`` imports the
    payload builders from the same package.
    """
    import rclpy
    from geometry_msgs.msg import Point
    from rclpy.node import Node
    from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import Image
    from std_msgs.msg import Header
    from vision_msgs.msg import (
        BoundingBox2D,
        Detection2D,
        Detection2DArray,
        ObjectHypothesisWithPose,
    )
    from visualization_msgs.msg import Marker, MarkerArray

    class EdgeTrackNode(Node):
        def __init__(self) -> None:
            super().__init__("edgetrack")

            self.declare_parameter("source", "synth:7")
            self.declare_parameter("model", "models/yolov8n_edgetrack_fp32.onnx")
            self.declare_parameter("backend", "onnx")
            self.declare_parameter("tracker", "bytetrack")
            self.declare_parameter("appearance", "histogram")
            self.declare_parameter("reid_model", "")
            self.declare_parameter("conf", 0.25)
            self.declare_parameter("imgsz_w", 640)
            self.declare_parameter("imgsz_h", 384)
            self.declare_parameter("frame_id", "camera_optical")
            self.declare_parameter("publish_image", True)
            self.declare_parameter("rate_hz", 15.0)
            self.declare_parameter("max_frames", 0)
            self.declare_parameter("class_names", ["person", "forklift", "pallet", "cone"])
            self.declare_parameter("shutdown_when_done", True)

            get = lambda name: self.get_parameter(name).value  # noqa: E731

            self.frame_frame_id = str(get("frame_id"))
            self.class_names = tuple(str(n) for n in get("class_names"))
            self.publish_image = bool(get("publish_image"))
            self.shutdown_when_done = bool(get("shutdown_when_done"))

            model_path = Path(str(get("model")))
            if not model_path.exists():
                raise FileNotFoundError(
                    f"model {model_path} not found.  Pass -p model:=<path> or run "
                    "`make export` first."
                )

            config = DetectorConfig(
                imgsz=(int(get("imgsz_w")), int(get("imgsz_h"))),
                conf_threshold=float(get("conf")),
                class_names=self.class_names,
            )
            detector = build_detector(model_path, str(get("backend")), config)
            tracker_name = str(get("tracker"))
            encoder = None
            if tracker_name.lower().startswith("deep"):
                reid = str(get("reid_model")) or None
                encoder = build_encoder(str(get("appearance")), reid)
            self.pipeline = Pipeline(detector, build_tracker(tracker_name), encoder)

            max_frames = int(get("max_frames")) or None
            self.source = open_source(str(get("source")), max_frames)
            self.frames = iter(self.source)
            self.frame_index = 0

            # Sensor-style QoS: best effort and shallow, because a subscriber
            # that falls behind on a 15 Hz vision stream wants the newest frame
            # rather than a backlog of stale ones.
            sensor_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                history=QoSHistoryPolicy.KEEP_LAST,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=1,
            )
            self.detection_pub = self.create_publisher(Detection2DArray, "~/detections", sensor_qos)
            self.track_pub = self.create_publisher(Detection2DArray, "~/tracks", sensor_qos)
            self.marker_pub = self.create_publisher(MarkerArray, "~/trajectories", sensor_qos)
            self.image_pub = self.create_publisher(Image, "~/image_annotated", sensor_qos)

            period = 1.0 / max(float(get("rate_hz")), 1e-3)
            self.timer = self.create_timer(period, self._tick)
            self.get_logger().info(
                f"edgetrack up: {self.source.description}, "
                f"{detector.info.backend}/{detector.info.precision}, tracker {tracker_name}"
            )

        # -- helpers ------------------------------------------------------

        def _header(self) -> Header:
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = self.frame_frame_id
            return header

        def _to_array(self, payloads: list[dict], header: Header) -> Detection2DArray:
            message = Detection2DArray()
            message.header = header
            for payload in payloads:
                detection = Detection2D()
                detection.header = header
                detection.id = payload["id"]
                hypothesis = ObjectHypothesisWithPose()
                _set_hypothesis(hypothesis, payload["class_id"], payload["score"])
                detection.results = [hypothesis]
                bbox = BoundingBox2D()
                _set_centre(bbox, payload["center_x"], payload["center_y"])
                bbox.size_x = float(payload["size_x"])
                bbox.size_y = float(payload["size_y"])
                detection.bbox = bbox
                message.detections.append(detection)
            return message

        def _to_markers(self, payloads: list[dict], header: Header) -> MarkerArray:
            array = MarkerArray()
            for payload in payloads:
                marker = Marker()
                marker.header = header
                marker.ns = "edgetrack_trajectory"
                marker.id = payload["id"]
                marker.type = Marker.LINE_STRIP
                marker.action = Marker.ADD
                marker.scale.x = 2.0
                marker.color.a = 1.0
                marker.color.r = 0.2
                marker.color.g = 0.8
                marker.color.b = 0.4
                marker.pose.orientation.w = 1.0
                for x, y, z in payload["points"]:
                    marker.points.append(Point(x=x, y=y, z=z))
                array.markers.append(marker)
            return array

        # -- loop ---------------------------------------------------------

        def _tick(self) -> None:
            try:
                frame = next(self.frames)
            except StopIteration:
                self.get_logger().info(
                    f"source exhausted after {self.frame_index} frames"
                )
                self.timer.cancel()
                if self.shutdown_when_done:
                    rclpy.shutdown()
                return

            result = self.pipeline.process(frame, self.frame_index)
            self.frame_index += 1
            header = self._header()

            self.detection_pub.publish(
                self._to_array(detection_payloads(result, self.class_names), header)
            )
            self.track_pub.publish(
                self._to_array(track_payloads(result, self.class_names), header)
            )
            self.marker_pub.publish(
                self._to_markers(trajectory_payloads(result.tracks), header)
            )

            if self.publish_image:
                annotated = annotate(frame, result, self.class_names)
                fields = image_payload(annotated)
                message = Image()
                message.header = header
                message.height = fields["height"]
                message.width = fields["width"]
                message.encoding = fields["encoding"]
                message.is_bigendian = fields["is_bigendian"]
                message.step = fields["step"]
                message.data = fields["data"]
                self.image_pub.publish(message)

        def destroy_node(self) -> bool:
            self.pipeline.close()
            return super().destroy_node()

    return EdgeTrackNode


class EdgeTrackNode:  # pragma: no cover - thin lazy wrapper
    """Placeholder that builds the real class on first instantiation."""

    def __new__(cls, *args: object, **kwargs: object):
        _require_ros()
        return _build_node_class()(*args, **kwargs)


def main(argv: list[str] | None = None) -> int:
    _require_ros()
    import rclpy

    rclpy.init(args=argv if argv is not None else sys.argv)
    node = _build_node_class()()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

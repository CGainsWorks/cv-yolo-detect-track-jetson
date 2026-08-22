"""ROS2 integration.

``messages`` imports nothing from ROS and is safe to import anywhere.  ``node``
imports rclpy lazily and raises a readable error rather than an ImportError
traceback when ROS is absent.
"""

from .messages import (
    detection_payloads,
    image_payload,
    track_payloads,
    trajectory_payloads,
)

__all__ = [
    "detection_payloads",
    "track_payloads",
    "trajectory_payloads",
    "image_payload",
]

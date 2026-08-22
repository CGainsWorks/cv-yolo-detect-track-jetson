"""Edge detection and tracking pipeline.

Import cost is deliberate: nothing at this level pulls in onnxruntime, torch or
rclpy.  Backends are imported by the factories in the subpackages, so the ROS2
node, the metrics and the tests each pay only for what they use.
"""

from .types import Detection, FrameResult, Track

__all__ = ["Detection", "Track", "FrameResult", "__version__"]

__version__ = "0.1.0"

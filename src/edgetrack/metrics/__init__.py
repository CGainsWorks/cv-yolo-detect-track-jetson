"""Detection and tracking metrics, plus the ID-switch attribution."""

from .detection import APResult, DetectionSample, evaluate_detections
from .idswitch import SwitchAnalysis, attribute_switches
from .mot import MotMetrics, evaluate_mot

__all__ = [
    "APResult",
    "DetectionSample",
    "evaluate_detections",
    "MotMetrics",
    "evaluate_mot",
    "SwitchAnalysis",
    "attribute_switches",
]

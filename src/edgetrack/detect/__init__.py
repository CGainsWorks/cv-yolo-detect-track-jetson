"""Detector backends and the factory that picks between them.

The factory is where ``--backend`` and ``--precision`` are resolved into a
concrete class.  Backend modules are imported lazily so that asking for ONNX
Runtime does not require torch to be installed, and asking for TensorRT does
not fail at import time on a machine that has neither.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .base import (
    BackendInfo,
    Detector,
    DetectorConfig,
    decode_yolov8,
    letterbox,
    nms,
    undo_letterbox,
)

__all__ = [
    "BackendInfo",
    "Detector",
    "DetectorConfig",
    "build_detector",
    "decode_yolov8",
    "letterbox",
    "nms",
    "undo_letterbox",
    "BACKENDS",
]

BACKENDS = ("onnx", "torch", "tensorrt")


def build_detector(
    model_path: str | Path,
    backend: str = "onnx",
    config: DetectorConfig | None = None,
    *,
    providers: Sequence[str] | None = None,
    device: str = "cuda",
    half: bool = False,
) -> Detector:
    """Construct a detector.

    ``backend`` is the runtime, not the precision.  Precision comes from the
    artefact: an FP16 ONNX file is FP16 whichever provider runs it, and the
    only backend where precision is a constructor argument is PyTorch, where
    ``half`` selects it.  Keeping that distinction is why the results table can
    say "onnxruntime, int8" and mean something exact.
    """
    key = backend.lower()
    if key in ("onnx", "onnxruntime", "ort"):
        from .onnx_backend import OnnxDetector

        return OnnxDetector(model_path, config, providers=providers)
    if key in ("torch", "pytorch"):
        from .torch_backend import TorchDetector

        return TorchDetector(model_path, config, device=device, half=half)
    if key in ("tensorrt", "trt"):
        from .trt_backend import TensorRTDetector

        return TensorRTDetector(model_path, config)
    raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")

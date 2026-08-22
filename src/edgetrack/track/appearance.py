"""Appearance descriptors for DeepSORT.

Two encoders, and the choice between them is a real result rather than a
detail.  ``histogram`` needs no weights and no inference and is what CI runs.
``reid`` is a small convolutional network trained on crops from the training
sequences, exported to ONNX so the runtime never imports torch.
``docs/tracking.md`` reports both, because the honest answer to "does DeepSORT
beat ByteTrack" depends entirely on which one is in the loop.

Every encoder returns L2-normalised float32 rows, so cosine distance is a dot
product and the two are interchangeable at the call site.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import cv2
import numpy as np

from ..types import Detection

__all__ = [
    "AppearanceEncoder",
    "HistogramEncoder",
    "OnnxReIDEncoder",
    "build_encoder",
    "attach_features",
    "CROP_SIZE",
]

# Width, height of the crop every encoder works on.  The 1:2 ratio is the
# standard person re-identification input and the re-ID network's ONNX graph
# has it baked in, so changing it here alone would break that encoder.
CROP_SIZE = (64, 128)


def _crop(image: np.ndarray, box: Sequence[float]) -> np.ndarray:
    """Clamp a box to the image and return the resized crop.

    A box can sit partly outside the frame after a Kalman prediction, and a
    detection on the frame edge can be one pixel wide.  Both cases return a
    valid crop rather than raising, because the tracker must not fall over on
    an object walking out of shot.
    """
    h, w = image.shape[:2]
    x1 = int(np.clip(np.floor(box[0]), 0, w - 1))
    y1 = int(np.clip(np.floor(box[1]), 0, h - 1))
    x2 = int(np.clip(np.ceil(box[2]), x1 + 1, w))
    y2 = int(np.clip(np.ceil(box[3]), y1 + 1, h))
    patch = image[y1:y2, x1:x2]
    if patch.size == 0:
        patch = np.zeros((2, 2, 3), dtype=image.dtype)
    return cv2.resize(patch, CROP_SIZE, interpolation=cv2.INTER_LINEAR)


class AppearanceEncoder(Protocol):
    """Anything that turns boxes into descriptors."""

    dim: int
    name: str

    def encode(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        ...


@dataclass
class HistogramEncoder:
    """Striped HSV colour histogram.

    Three horizontal stripes, so the descriptor is weakly sensitive to where on
    the body a colour appears rather than only to how much of it there is.  Hue
    is weighted by saturation when accumulating, which keeps a grey background
    pixel from voting for an arbitrary hue.

    This is a colour descriptor, not a learned re-identification embedding.  It
    separates the synthetic objects well because they are rendered in distinct
    per-instance colours; it would do considerably worse on real people in
    similar clothing.  That limitation is stated in the README rather than
    quietly relied upon.
    """

    h_bins: int = 12
    s_bins: int = 6
    v_bins: int = 6
    stripes: int = 3
    name: str = "histogram"

    @property
    def dim(self) -> int:
        return self.stripes * (self.h_bins + self.s_bins + self.v_bins)

    def encode(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        out = np.zeros((boxes.shape[0], self.dim), dtype=np.float32)
        for i, box in enumerate(boxes):
            crop = _crop(image, box)
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            rows = np.array_split(hsv, self.stripes, axis=0)
            parts = []
            for stripe in rows:
                hue = stripe[:, :, 0].astype(np.float32)
                sat = stripe[:, :, 1].astype(np.float32)
                val = stripe[:, :, 2].astype(np.float32)
                h_hist = np.histogram(
                    hue, bins=self.h_bins, range=(0, 180), weights=sat / 255.0
                )[0]
                s_hist = np.histogram(sat, bins=self.s_bins, range=(0, 256))[0]
                v_hist = np.histogram(val, bins=self.v_bins, range=(0, 256))[0]
                parts.append(
                    np.concatenate(
                        [
                            h_hist / max(h_hist.sum(), 1e-6),
                            s_hist / max(s_hist.sum(), 1e-6),
                            v_hist / max(v_hist.sum(), 1e-6),
                        ]
                    )
                )
            vector = np.concatenate(parts).astype(np.float32)
            norm = float(np.linalg.norm(vector))
            out[i] = vector / norm if norm > 0 else vector
        return out


class OnnxReIDEncoder:
    """The trained re-identification network, run through ONNX Runtime.

    Weights come from ``scripts/train_reid.py`` and are exported by
    ``scripts/export_reid.py``.  The runtime image carries the ONNX file and
    onnxruntime; it never imports torch.
    """

    name = "reid"

    def __init__(self, model_path: str | Path, providers: Sequence[str] | None = None) -> None:
        import onnxruntime as ort  # imported here so the histogram path needs no ORT

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"re-ID weights not found at {self.model_path}.  Run "
                "`make reid` to train and export them, or use the histogram "
                "encoder with --appearance histogram."
            )
        available = ort.get_available_providers()
        chosen = [p for p in (providers or ["CUDAExecutionProvider", "CPUExecutionProvider"])
                  if p in available] or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.model_path), providers=chosen)
        self.providers = self.session.get_providers()
        self._input = self.session.get_inputs()[0].name
        self._output = self.session.get_outputs()[0].name
        self.dim = int(self.session.get_outputs()[0].shape[-1])

    def encode(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        if boxes.shape[0] == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        batch = np.stack([_crop(image, box) for box in boxes])
        batch = batch[:, :, :, ::-1].astype(np.float32) / 255.0  # BGR to RGB
        batch = (batch - 0.5) / 0.5
        batch = np.ascontiguousarray(batch.transpose(0, 3, 1, 2))
        features = self.session.run([self._output], {self._input: batch})[0]
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        return (features / np.where(norms > 0, norms, 1.0)).astype(np.float32)


def build_encoder(
    kind: str = "histogram", model_path: str | Path | None = None
) -> AppearanceEncoder:
    """Factory used by every script, so the ``--appearance`` flag means one thing."""
    if kind == "histogram":
        return HistogramEncoder()
    if kind == "reid":
        if model_path is None:
            raise ValueError("the reid encoder needs --reid-model pointing at an ONNX file")
        return OnnxReIDEncoder(model_path)
    raise ValueError(f"unknown appearance encoder {kind!r}; expected histogram or reid")


def attach_features(
    encoder: AppearanceEncoder, image: np.ndarray, detections: Sequence[Detection]
) -> list[Detection]:
    """Return copies of ``detections`` carrying descriptors.

    :class:`~edgetrack.types.Detection` is frozen, so this rebuilds rather than
    mutates.  That is deliberate: the same detection list is fed to both
    trackers in ``scripts/eval_tracking.py`` and a mutable feature field would
    let one tracker's preprocessing leak into the other's input.
    """
    if not detections:
        return []
    boxes = np.asarray([d.xyxy for d in detections], dtype=np.float64)
    features = encoder.encode(image, boxes)
    return [
        Detection(d.xyxy, d.score, d.class_id, feature=features[i])
        for i, d in enumerate(detections)
    ]

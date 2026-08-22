"""Detector, appearance encoder and tracker wired together.

One class so that the demo script, the tracking evaluation, the benchmark and
the ROS2 node all run the same code path.  If the ROS2 node had its own loop
then a claim about latency measured by the benchmark would say nothing about
what the node does, and the two would drift.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .detect import Detector
from .track.appearance import AppearanceEncoder, attach_features
from .track.base import BaseTracker
from .types import FrameResult, Track

__all__ = ["Pipeline", "annotate", "CLASS_COLOURS"]

# Distinct, colourblind-safe-ish BGR colours for the four classes.
CLASS_COLOURS = (
    (66, 133, 244),  # person, blue
    (52, 168, 83),  # forklift, green
    (26, 158, 245),  # pallet, amber
    (146, 82, 210),  # cone, purple
)


@dataclass
class Pipeline:
    """Run a detector and a tracker over successive frames.

    ``encoder`` is required by DeepSORT and ignored by ByteTrack.  Passing one
    to ByteTrack is not an error but it is wasted work, so
    ``scripts/eval_tracking.py`` only builds an encoder when it needs one and
    the timing breakdown reports the encoder stage separately for that reason.
    """

    detector: Detector
    tracker: BaseTracker
    encoder: AppearanceEncoder | None = None

    def reset(self) -> None:
        self.tracker.reset()

    def process(self, image: np.ndarray, frame_id: int | None = None) -> FrameResult:
        start = time.perf_counter()
        detections = self.detector.detect(image)
        detect_end = time.perf_counter()

        if self.encoder is not None:
            detections = attach_features(self.encoder, image, detections)
        encode_end = time.perf_counter()

        tracks = self.tracker.update(detections, frame_id)
        end = time.perf_counter()

        resolved = self.tracker.frame_id if frame_id is None else int(frame_id)
        return FrameResult(
            frame_id=resolved,
            detections=list(detections),
            tracks=tracks,
            timings_ms={
                "preprocess": self.detector.last_preprocess_ms,
                "inference": self.detector.last_inference_ms,
                "postprocess": self.detector.last_postprocess_ms,
                "detect_total": (detect_end - start) * 1000.0,
                "appearance": (encode_end - detect_end) * 1000.0,
                "track": (end - encode_end) * 1000.0,
                "total": (end - start) * 1000.0,
            },
        )

    def close(self) -> None:
        self.detector.close()


def _colour_for(track: Track) -> tuple[int, int, int]:
    base = CLASS_COLOURS[track.class_id % len(CLASS_COLOURS)]
    # Shift the hue slightly per identity so two adjacent tracks of the same
    # class are still distinguishable in the annotated video.
    jitter = (track.track_id * 47) % 60 - 30
    return tuple(int(np.clip(c + jitter, 0, 255)) for c in base)  # type: ignore[return-value]


def annotate(
    image: np.ndarray,
    result: FrameResult,
    class_names: tuple[str, ...] = (),
    *,
    trail_length: int = 30,
    draw_detections: bool = False,
) -> np.ndarray:
    """Draw tracks, identities and trajectory trails onto a copy of ``image``.

    Detections are off by default.  With them on, every track box has a nearly
    coincident detection box behind it and the result is unreadable; the flag
    exists because it is the fastest way to see whether a missing track is a
    detector problem or an association problem.
    """
    canvas = image.copy()

    if draw_detections:
        for detection in result.detections:
            x1, y1, x2, y2 = (int(v) for v in detection.xyxy)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (160, 160, 160), 1)

    for track in result.tracks:
        colour = _colour_for(track)
        x1, y1, x2, y2 = (int(v) for v in track.xyxy)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)

        name = (
            class_names[track.class_id]
            if track.class_id < len(class_names)
            else str(track.class_id)
        )
        label = f"{name} #{track.track_id} {track.score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(
            canvas, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1,
            cv2.LINE_AA,
        )

        trail = track.trajectory()
        if trail.shape[0] > 1:
            points = trail[-trail_length:, 1:3].astype(np.int32)
            cv2.polylines(canvas, [points], False, colour, 1, cv2.LINE_AA)

    header = f"frame {result.frame_id}  tracks {len(result.tracks)}  det {len(result.detections)}"
    if "total" in result.timings_ms:
        header += f"  {result.timings_ms['total']:.1f} ms"
    cv2.putText(canvas, header, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(
        canvas, header, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245, 245, 245), 1,
        cv2.LINE_AA,
    )
    return canvas

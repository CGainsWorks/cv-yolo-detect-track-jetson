"""The pipeline, the annotator and the appearance encoders.

A stub detector stands in for the real one so these run without weights.  The
stub is deliberately dumb: it returns whatever it was told to, which is enough
to check that the pipeline wires the stages together and reports timings for
each.
"""

from __future__ import annotations

import numpy as np
import pytest

from edgetrack.detect.base import BackendInfo, Detector, DetectorConfig
from edgetrack.pipeline import Pipeline, annotate
from edgetrack.track import ByteTrack, ByteTrackConfig, DeepSort, DeepSortConfig
from edgetrack.track.appearance import HistogramEncoder, attach_features, build_encoder
from edgetrack.types import Detection, FrameResult


class StubDetector(Detector):
    """Replays a scripted list of detections, one entry per call."""

    def __init__(self, script: list[list[Detection]]) -> None:
        super().__init__(DetectorConfig(class_names=("person",)))
        self.script = script
        self.calls = 0

    def _forward(self, tensor: np.ndarray) -> np.ndarray:  # pragma: no cover - unused
        raise AssertionError("StubDetector overrides detect and never forwards")

    def detect(self, image: np.ndarray) -> list[Detection]:
        self.last_preprocess_ms = 0.1
        self.last_inference_ms = 1.0
        self.last_postprocess_ms = 0.2
        result = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return result

    @property
    def info(self) -> BackendInfo:
        return BackendInfo("stub", "fp32", "none")


def box(x, y, w=24.0, h=60.0):
    return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)


def frame(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (384, 640, 3), dtype=np.uint8)


def test_pipeline_reports_every_stage():
    script = [[Detection(box(100 + 5 * t, 200), 0.9, 0)] for t in range(6)]
    pipeline = Pipeline(StubDetector(script), ByteTrack(ByteTrackConfig(n_init=2)))
    result = pipeline.process(frame(), 0)
    assert set(result.timings_ms) >= {
        "preprocess", "inference", "postprocess", "detect_total", "appearance",
        "track", "total",
    }
    assert result.timings_ms["appearance"] == pytest.approx(0.0, abs=1e-2)


def test_pipeline_tracks_across_frames():
    script = [[Detection(box(100 + 5 * t, 200), 0.9, 0)] for t in range(10)]
    pipeline = Pipeline(StubDetector(script), ByteTrack(ByteTrackConfig(n_init=2)))
    ids = set()
    for t in range(10):
        result = pipeline.process(frame(t), t)
        ids.update(track.track_id for track in result.tracks)
    assert ids == {1}


def test_pipeline_attaches_features_for_deepsort():
    script = [[Detection(box(100 + 5 * t, 200), 0.9, 0)] for t in range(6)]
    encoder = HistogramEncoder()
    pipeline = Pipeline(
        StubDetector(script), DeepSort(DeepSortConfig(n_init=2)), encoder
    )
    result = pipeline.process(frame(), 0)
    assert result.detections[0].feature is not None
    assert result.detections[0].feature.shape == (encoder.dim,)
    assert result.timings_ms["appearance"] > 0.0


def test_pipeline_reset_clears_the_tracker():
    script = [[Detection(box(100, 200), 0.9, 0)]]
    pipeline = Pipeline(StubDetector(script), ByteTrack(ByteTrackConfig(n_init=2)))
    for t in range(4):
        pipeline.process(frame(), t)
    pipeline.reset()
    pipeline.process(frame(), 0)
    result = pipeline.process(frame(), 1)
    assert [track.track_id for track in result.tracks] == [1]


# -- Appearance -----------------------------------------------------------


def test_histogram_encoder_shape_and_norm():
    encoder = HistogramEncoder()
    features = encoder.encode(frame(), np.array([box(100, 200), box(300, 200)]))
    assert features.shape == (2, encoder.dim)
    assert np.allclose(np.linalg.norm(features, axis=1), 1.0, atol=1e-5)


def test_histogram_encoder_separates_colours():
    image = np.zeros((200, 400, 3), dtype=np.uint8)
    image[:, :200] = (0, 0, 220)  # red on the left, in BGR
    image[:, 200:] = (220, 0, 0)  # blue on the right
    encoder = HistogramEncoder()
    features = encoder.encode(image, np.array([[10, 10, 190, 190], [210, 10, 390, 190]]))
    same = encoder.encode(image, np.array([[20, 20, 180, 180]]))
    assert float(features[0] @ same[0]) > float(features[1] @ same[0]) + 0.3


def test_histogram_encoder_survives_a_box_off_the_edge():
    encoder = HistogramEncoder()
    features = encoder.encode(frame(), np.array([[-50.0, -50.0, 5.0, 5.0]]))
    assert features.shape == (1, encoder.dim)
    assert np.isfinite(features).all()


def test_histogram_encoder_on_no_boxes():
    encoder = HistogramEncoder()
    assert encoder.encode(frame(), np.zeros((0, 4))).shape == (0, encoder.dim)


def test_attach_features_does_not_mutate_the_input():
    original = [Detection(box(100, 200), 0.9, 0)]
    attached = attach_features(HistogramEncoder(), frame(), original)
    assert original[0].feature is None
    assert attached[0].feature is not None
    assert attached[0].xyxy == original[0].xyxy


def test_attach_features_on_an_empty_list():
    assert attach_features(HistogramEncoder(), frame(), []) == []


def test_build_encoder_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown appearance encoder"):
        build_encoder("magic")
    with pytest.raises(ValueError, match="needs --reid-model"):
        build_encoder("reid")


def test_reid_encoder_reports_a_missing_file_clearly():
    from edgetrack.track.appearance import OnnxReIDEncoder

    with pytest.raises(FileNotFoundError, match="make reid"):
        OnnxReIDEncoder("models/does_not_exist.onnx")


# -- Annotation -----------------------------------------------------------


def test_annotate_returns_a_copy():
    image = frame()
    result = FrameResult(0, [], [], {"total": 5.0})
    output = annotate(image, result, ("person",))
    assert output is not image
    assert output.shape == image.shape


def test_annotate_draws_something_for_a_track():
    from edgetrack.types import Track

    image = np.zeros((384, 640, 3), dtype=np.uint8)
    track = Track(1, 0, (100.0, 100.0, 200.0, 300.0), 0.9, state="confirmed",
                  history=[(0, (100.0, 100.0, 200.0, 300.0)),
                           (1, (105.0, 100.0, 205.0, 300.0))])
    output = annotate(image, FrameResult(1, [], [track]), ("person",))
    assert output.sum() > 0, "nothing was drawn"


def test_annotate_handles_an_unknown_class_index():
    from edgetrack.types import Track

    track = Track(1, 9, (10.0, 10.0, 40.0, 60.0), 0.5, state="confirmed")
    output = annotate(np.zeros((100, 100, 3), np.uint8), FrameResult(0, [], [track]), ("person",))
    assert output.shape == (100, 100, 3)

"""Pre and post processing, which every backend shares.

These are the functions a precision comparison would silently blame the
arithmetic for.  If letterboxing were off by half a pixel, or NMS kept a
different box on one backend, the INT8 row would look worse than it is and the
conclusion drawn from the table would be wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from edgetrack.detect.base import (
    DetectorConfig,
    decode_yolov8,
    letterbox,
    nms,
    undo_letterbox,
)


def test_letterbox_preserves_aspect_and_fills_the_target():
    image = np.zeros((300, 900, 3), dtype=np.uint8)
    padded, scale, pad = letterbox(image, (640, 384))
    assert padded.shape[:2] == (384, 640)
    assert scale == pytest.approx(640 / 900)
    # 300 * (640/900) = 213.3 rows of content, so the pad is split evenly.
    assert pad[0] == 0
    assert pad[1] == pytest.approx((384 - round(300 * scale)) / 2, abs=1)


def test_letterbox_pads_symmetrically():
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    padded, _, pad = letterbox(image, (640, 384), colour=114)
    left = int(pad[0])
    assert (padded[:, : left - 1] == 114).all(), "the left pad is not the fill colour"
    assert (padded[:, left + 385 :] == 114).all(), "the right pad is not the fill colour"


def test_undo_letterbox_round_trips():
    image = np.zeros((360, 540, 3), dtype=np.uint8)
    _, scale, pad = letterbox(image, (640, 384))
    original = np.array([[10.0, 20.0, 110.0, 220.0], [0.0, 0.0, 540.0, 360.0]])
    forward = original * scale
    forward[:, [0, 2]] += pad[0]
    forward[:, [1, 3]] += pad[1]
    back = undo_letterbox(forward, scale, pad, image.shape[:2])
    assert np.allclose(back, original, atol=1e-6)


def test_undo_letterbox_clips_to_the_image():
    boxes = np.array([[-50.0, -50.0, 5000.0, 5000.0]])
    out = undo_letterbox(boxes, 1.0, (0.0, 0.0), (100, 200))
    assert out.tolist() == [[0.0, 0.0, 200.0, 100.0]]


def test_undo_letterbox_handles_no_boxes():
    assert undo_letterbox(np.zeros((0, 4)), 1.0, (0.0, 0.0), (10, 10)).shape == (0, 4)


def test_nms_keeps_the_best_of_an_overlapping_group():
    boxes = np.array(
        [
            [0, 0, 100, 100],
            [5, 5, 105, 105],  # heavy overlap with the first
            [500, 500, 600, 600],  # separate
        ],
        dtype=np.float64,
    )
    scores = np.array([0.7, 0.9, 0.6])
    keep = nms(boxes, scores, 0.5)
    assert keep == [1, 2], f"kept {keep}"


def test_nms_returns_indices_in_score_order():
    boxes = np.array([[0, 0, 10, 10], [100, 100, 110, 110], [200, 200, 210, 210]], float)
    scores = np.array([0.1, 0.9, 0.5])
    assert nms(boxes, scores, 0.5) == [1, 2, 0]


def test_nms_on_empty_input():
    assert nms(np.zeros((0, 4)), np.zeros(0), 0.5) == []


def test_nms_tolerates_degenerate_boxes():
    boxes = np.array([[10, 10, 10, 10], [0, 0, 20, 20]], dtype=np.float64)
    keep = nms(boxes, np.array([0.9, 0.8]), 0.5)
    assert set(keep) == {0, 1}, "a zero-area box swallowed a real one"


def _fake_head(boxes_cxcywh, scores_per_class, n_anchors=32, n_classes=4):
    """Build a (1, 4 + nc, anchors) tensor with a few real detections in it."""
    array = np.zeros((4 + n_classes, n_anchors), dtype=np.float32)
    for index, (box, scores) in enumerate(zip(boxes_cxcywh, scores_per_class)):
        array[:4, index] = box
        array[4:, index] = scores
    return array[None]


def test_decode_finds_the_planted_detections():
    config = DetectorConfig(imgsz=(640, 384), conf_threshold=0.3, iou_threshold=0.5)
    head = _fake_head(
        [(100, 200, 40, 80), (500, 100, 60, 60)],
        [(0.9, 0.0, 0.0, 0.0), (0.0, 0.0, 0.8, 0.0)],
    )
    detections = decode_yolov8(head, config, scale=1.0, pad=(0.0, 0.0), shape=(384, 640))
    assert len(detections) == 2
    assert [d.class_id for d in detections] == [0, 2]
    first = detections[0]
    assert first.xyxy == pytest.approx((80.0, 160.0, 120.0, 240.0))
    assert first.score == pytest.approx(0.9)


def test_decode_respects_the_confidence_threshold():
    config = DetectorConfig(conf_threshold=0.5)
    head = _fake_head([(100, 200, 40, 80)], [(0.4, 0.0, 0.0, 0.0)])
    assert decode_yolov8(head, config, 1.0, (0.0, 0.0), (384, 640)) == []


def test_decode_suppresses_duplicates_within_a_class():
    config = DetectorConfig(conf_threshold=0.3, iou_threshold=0.5)
    head = _fake_head(
        [(100, 200, 40, 80), (102, 202, 40, 80)],
        [(0.9, 0, 0, 0), (0.7, 0, 0, 0)],
    )
    detections = decode_yolov8(head, config, 1.0, (0.0, 0.0), (384, 640))
    assert len(detections) == 1
    assert detections[0].score == pytest.approx(0.9)


def test_decode_does_not_suppress_across_classes_by_default():
    config = DetectorConfig(conf_threshold=0.3, iou_threshold=0.5, agnostic_nms=False)
    head = _fake_head(
        [(100, 200, 40, 80), (100, 200, 40, 80)],
        [(0.9, 0, 0, 0), (0, 0.8, 0, 0)],
    )
    detections = decode_yolov8(head, config, 1.0, (0.0, 0.0), (384, 640))
    assert len(detections) == 2

    agnostic = DetectorConfig(conf_threshold=0.3, iou_threshold=0.5, agnostic_nms=True)
    assert len(decode_yolov8(head, agnostic, 1.0, (0.0, 0.0), (384, 640))) == 1


def test_decode_accepts_either_channel_order():
    config = DetectorConfig(conf_threshold=0.3)
    head = _fake_head([(100, 200, 40, 80)], [(0.9, 0, 0, 0)], n_anchors=32)
    transposed = np.ascontiguousarray(head.transpose(0, 2, 1))
    a = decode_yolov8(head, config, 1.0, (0.0, 0.0), (384, 640))
    b = decode_yolov8(transposed, config, 1.0, (0.0, 0.0), (384, 640))
    assert [d.xyxy for d in a] == [d.xyxy for d in b]


def test_decode_rejects_a_head_with_no_class_channels():
    config = DetectorConfig()
    with pytest.raises(ValueError, match="expected 4 box coordinates"):
        decode_yolov8(np.zeros((1, 4, 10)), config, 1.0, (0.0, 0.0), (384, 640))


def test_decode_caps_the_detection_count():
    config = DetectorConfig(conf_threshold=0.1, iou_threshold=0.9, max_detections=3)
    boxes = [(50 + 200 * i, 100, 20, 20) for i in range(8)]
    scores = [(0.2 + 0.05 * i, 0, 0, 0) for i in range(8)]
    head = _fake_head(boxes, scores, n_anchors=16)
    detections = decode_yolov8(head, config, 1.0, (0.0, 0.0), (384, 1800))
    assert len(detections) == 3
    assert detections == sorted(detections, key=lambda d: -d.score)


def test_config_rejects_a_stride_incompatible_size():
    with pytest.raises(ValueError, match="multiple of 32"):
        DetectorConfig(imgsz=(641, 384)).validate()
    with pytest.raises(ValueError, match="conf_threshold"):
        DetectorConfig(conf_threshold=1.5).validate()

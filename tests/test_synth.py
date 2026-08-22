"""The generator is deterministic and its ground truth is self-consistent.

Determinism is not a nicety here.  Every published number is produced from a
seed, ``scripts/run_demo.py`` regenerates its ground truth from the same seed
rather than caching it, and the dataset card quotes counts that only mean
something if the same seed gives the same data.
"""

from __future__ import annotations

import numpy as np
import pytest

from edgetrack.data.synth import (
    CLASSES,
    SynthConfig,
    config_for_split,
    generate_sequence,
)


def test_same_seed_gives_identical_frames():
    cfg = config_for_split("train", 42, n_frames=6)
    first = list(generate_sequence(cfg))
    second = list(generate_sequence(cfg))
    assert len(first) == len(second) == 6
    for a, b in zip(first, second):
        assert np.array_equal(a.image, b.image)
        assert a.annotations == b.annotations


def test_different_seeds_give_different_frames():
    a = list(generate_sequence(config_for_split("train", 1, n_frames=3)))
    b = list(generate_sequence(config_for_split("train", 2, n_frames=3)))
    assert not np.array_equal(a[0].image, b[0].image)


def test_frames_have_the_configured_shape():
    cfg = SynthConfig(seed=3, width=320, height=192, n_frames=2)
    for frame in generate_sequence(cfg):
        assert frame.image.shape == (192, 320, 3)
        assert frame.image.dtype == np.uint8


def test_annotations_are_inside_the_frame_and_ordered():
    cfg = config_for_split("test", 7, n_frames=10)
    for frame in generate_sequence(cfg):
        for ann in frame.annotations:
            x1, y1, x2, y2 = ann.xyxy
            assert 0 <= x1 <= x2 <= cfg.width
            assert 0 <= y1 <= y2 <= cfg.height
            assert 0.0 <= ann.visibility <= 1.0
            assert 0 <= ann.class_id < len(CLASSES)


def test_instance_ids_are_stable_across_frames():
    frames = list(generate_sequence(config_for_split("test", 11, n_frames=20)))
    seen = {ann.instance_id: ann.class_id for ann in frames[0].annotations}
    for frame in frames[1:]:
        for ann in frame.annotations:
            # An identity never changes class.  The tracker's class-aware
            # association would be unsound against ground truth that did.
            assert seen.setdefault(ann.instance_id, ann.class_id) == ann.class_id


def test_static_objects_do_not_move():
    frames = list(generate_sequence(config_for_split("train", 5, n_frames=12)))
    positions: dict[int, list[tuple[float, float]]] = {}
    classes: dict[int, int] = {}
    for frame in frames:
        for ann in frame.annotations:
            classes[ann.instance_id] = ann.class_id
            x1, y1, x2, y2 = ann.xyxy
            positions.setdefault(ann.instance_id, []).append(((x1 + x2) / 2, (y1 + y2) / 2))
    static = [i for i, c in classes.items() if c in (2, 3)]
    assert static, "the scene should contain pallets or cones"
    for instance in static:
        xs = [p[0] for p in positions[instance]]
        ys = [p[1] for p in positions[instance]]
        assert max(xs) - min(xs) < 1e-6
        assert max(ys) - min(ys) < 1e-6


def test_occlusion_actually_happens():
    """Some object must be substantially hidden at some point.

    If this ever stops being true the dataset has stopped exercising the case
    the trackers are being compared on, and the ID-switch analysis would have
    nothing to attribute.
    """
    worst = 1.0
    for seed in range(20000, 20004):
        for frame in generate_sequence(config_for_split("test", seed, n_frames=40)):
            for ann in frame.annotations:
                worst = min(worst, ann.visibility)
    assert worst < 0.3, f"least visible annotation across four sequences was {worst}"


def test_objects_of_the_same_class_overlap_sometimes():
    """The crossing schedule produces genuine same-class ambiguity."""
    best = 0.0
    for seed in range(20000, 20006):
        for frame in generate_sequence(config_for_split("test", seed, n_frames=60)):
            by_class: dict[int, list[np.ndarray]] = {}
            for ann in frame.annotations:
                by_class.setdefault(ann.class_id, []).append(np.asarray(ann.xyxy))
            for boxes in by_class.values():
                for i in range(len(boxes)):
                    for j in range(i + 1, len(boxes)):
                        a, b = boxes[i], boxes[j]
                        lt = np.maximum(a[:2], b[:2])
                        rb = np.minimum(a[2:], b[2:])
                        wh = np.clip(rb - lt, 0, None)
                        inter = wh[0] * wh[1]
                        area_a = (a[2] - a[0]) * (a[3] - a[1])
                        area_b = (b[2] - b[0]) * (b[3] - b[1])
                        union = area_a + area_b - inter
                        if union > 0:
                            best = max(best, inter / union)
    assert best > 0.25, f"best same-class overlap across six sequences was {best}"


def test_test_split_differs_from_train_split():
    train = config_for_split("train", 0)
    test = config_for_split("test", 0)
    assert train.palette != test.palette
    assert test.noise_sigma > train.noise_sigma
    assert test.blur_scale > train.blur_scale


def test_unknown_split_is_rejected():
    with pytest.raises(ValueError, match="unknown split"):
        config_for_split("holdout", 0)

"""On-disk dataset layout, both the YOLO side and the MOT side.

Everything here writes to a temporary directory with the smallest possible
scene, so it runs in a couple of seconds and needs no generated dataset.
"""

from __future__ import annotations

import numpy as np
import pytest

from edgetrack.data.dataset import (
    load_yolo_labels,
    read_mot_gt,
    write_dataset_json,
    write_split,
)
from edgetrack.data.synth import CLASSES, config_for_split


@pytest.fixture
def tiny_root(tmp_path):
    configs = [config_for_split("test", seed, n_frames=6) for seed in (900, 901)]
    stats = write_split(tmp_path, "test", configs, write_mot=True)
    return tmp_path, stats


def test_write_split_produces_one_label_per_image(tiny_root):
    root, stats = tiny_root
    images = sorted((root / "images" / "test").glob("*.jpg"))
    labels = sorted((root / "labels" / "test").glob("*.txt"))
    assert len(images) == len(labels) == stats.n_images == 12
    assert [p.stem for p in images] == [p.stem for p in labels]


def test_label_files_are_normalised_yolo_rows(tiny_root):
    root, _ = tiny_root
    for path in sorted((root / "labels" / "test").glob("*.txt")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split()
            assert len(parts) == 5
            assert 0 <= int(parts[0]) < len(CLASSES)
            for value in parts[1:]:
                assert 0.0 <= float(value) <= 1.0


def test_load_yolo_labels_round_trips_to_pixels(tiny_root):
    root, _ = tiny_root
    path = sorted((root / "labels" / "test").glob("*.txt"))[0]
    boxes, classes = load_yolo_labels(path, 640, 384)
    assert boxes.shape[1] == 4
    assert boxes.shape[0] == classes.shape[0]
    assert (boxes[:, 2] >= boxes[:, 0]).all()
    assert (boxes[:, 3] >= boxes[:, 1]).all()


def test_load_yolo_labels_on_a_missing_file(tmp_path):
    boxes, classes = load_yolo_labels(tmp_path / "nothing.txt", 640, 384)
    assert boxes.shape == (0, 4)
    assert classes.shape == (0,)


def test_mot_ground_truth_is_written_for_every_sequence(tiny_root):
    root, _ = tiny_root
    sequences = sorted((root / "mot" / "test").iterdir())
    assert [p.name for p in sequences] == ["test_seq00900", "test_seq00901"]
    for sequence in sequences:
        assert (sequence / "gt" / "gt.txt").exists()
        info = (sequence / "seqinfo.ini").read_text(encoding="utf-8")
        assert "seqLength=6" in info
        assert "imWidth=640" in info


def test_mot_frames_are_zero_based_after_reading(tiny_root):
    root, _ = tiny_root
    rows = read_mot_gt(root / "mot" / "test" / "test_seq00900" / "gt" / "gt.txt")
    assert min(rows) == 0
    assert max(rows) == 5


def test_mot_rows_carry_visibility_and_the_filter_uses_it(tiny_root):
    root, _ = tiny_root
    path = root / "mot" / "test" / "test_seq00900" / "gt" / "gt.txt"
    everything = read_mot_gt(path, min_visibility=0.0)
    visible = read_mot_gt(path, min_visibility=0.9)
    total_all = sum(len(v) for v in everything.values())
    total_visible = sum(len(v) for v in visible.values())
    assert total_visible < total_all, "the visibility filter removed nothing"


def test_mot_ground_truth_keeps_boxes_the_yolo_labels_drop(tiny_root):
    """The two layouts disagree on purpose and this pins the direction.

    The detection labels drop heavily occluded boxes because training on them
    teaches hallucination.  The MOT truth keeps them with a visibility column,
    because the tracking evaluator needs to know the object was there.
    """
    root, stats = tiny_root
    yolo_boxes = sum(
        len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])
        for path in (root / "labels" / "test").glob("*.txt")
    )
    mot_boxes = sum(
        len(rows)
        for sequence in (root / "mot" / "test").iterdir()
        for rows in read_mot_gt(sequence / "gt" / "gt.txt", 0.0).values()
    )
    assert mot_boxes > yolo_boxes
    assert stats.n_dropped_occluded > 0


def test_mot_identity_is_stable_within_a_sequence(tiny_root):
    root, _ = tiny_root
    rows = read_mot_gt(root / "mot" / "test" / "test_seq00900" / "gt" / "gt.txt", 0.0)
    classes: dict[int, int] = {}
    for frame in rows.values():
        for track_id, class_id, _ in frame:
            assert classes.setdefault(track_id, class_id) == class_id


def test_split_stats_are_self_consistent(tiny_root):
    _, stats = tiny_root
    summary = stats.as_dict()
    assert summary["images"] == 12
    assert summary["sequences"] == 2
    assert sum(summary["class_counts"].values()) == summary["labelled_boxes"]
    assert 0.0 <= summary["mean_visibility_of_labelled"] <= 1.0
    area = summary["box_area_fraction"]
    assert 0.0 < area["p05"] <= area["p50"] <= area["p95"] < 1.0


def test_dataset_json_is_written(tmp_path):
    write_dataset_json(tmp_path, {"classes": list(CLASSES)})
    assert (tmp_path / "dataset.json").exists()


def test_read_mot_gt_tolerates_short_rows(tmp_path):
    path = tmp_path / "gt.txt"
    path.write_text("1,3,10,20,30,40,1\n", encoding="utf-8")
    rows = read_mot_gt(path)
    assert rows[0][0][0] == 3
    assert np.allclose(rows[0][0][2], [10, 20, 40, 60])

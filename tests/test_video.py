"""Frame sources and writers.

``--source`` accepts four kinds of thing and every script takes the same
string, so a mistake here breaks the demo, the benchmark and the ROS2 node at
once.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from edgetrack.video import IMAGE_SUFFIXES, VideoWriter, open_source, write_gif


def test_synthetic_source_yields_the_expected_shape():
    source = open_source("synth:5", max_frames=4)
    frames = list(source)
    assert len(frames) == 4
    assert source.width == 640 and source.height == 384
    assert all(frame.shape == (384, 640, 3) for frame in frames)
    assert "seed 5" in source.description


def test_synthetic_source_defaults_to_seed_zero():
    a = list(open_source("synth", max_frames=2))
    b = list(open_source("synth:0", max_frames=2))
    assert np.array_equal(a[0], b[0])


def test_synthetic_source_is_capped_by_max_frames():
    assert open_source("synth:1", max_frames=3).length == 3
    # Without a cap the test-split default length comes through.
    assert open_source("synth:1").length == 120


def test_directory_source_reads_in_name_order(tmp_path):
    for index in range(5):
        image = np.full((32, 48, 3), index * 10, dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"frame_{index:03d}.jpg"), image)
    source = open_source(str(tmp_path))
    frames = list(source)
    assert len(frames) == 5
    assert source.width == 48 and source.height == 32
    means = [float(frame.mean()) for frame in frames]
    assert means == sorted(means), "frames came back out of order"


def test_directory_source_respects_max_frames(tmp_path):
    for index in range(6):
        cv2.imwrite(str(tmp_path / f"f{index}.png"), np.zeros((16, 16, 3), np.uint8))
    assert len(list(open_source(str(tmp_path), max_frames=2))) == 2


def test_directory_source_rejects_a_directory_with_no_images(tmp_path):
    (tmp_path / "notes.txt").write_text("nothing here", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="no images with suffix"):
        open_source(str(tmp_path))


def test_unknown_source_is_rejected_with_a_useful_message():
    with pytest.raises(FileNotFoundError, match="cannot interpret --source"):
        open_source("not-a-thing")


def test_video_round_trip(tmp_path):
    path = tmp_path / "clip.mp4"
    frames = [
        np.full((64, 96, 3), value, dtype=np.uint8) for value in (10, 120, 200, 60)
    ]
    with VideoWriter(path, 10.0, (96, 64)) as writer:
        for frame in frames:
            writer.write(frame)
        assert writer.frames == 4
    assert path.exists() and path.stat().st_size > 0

    source = open_source(str(path))
    decoded = list(source)
    assert len(decoded) == 4
    assert source.width == 96 and source.height == 64
    # Lossy codec, so compare means rather than pixels.
    for original, back in zip(frames, decoded):
        assert abs(float(original.mean()) - float(back.mean())) < 12


def test_video_source_respects_max_frames(tmp_path):
    path = tmp_path / "clip.mp4"
    with VideoWriter(path, 10.0, (32, 32)) as writer:
        for _ in range(10):
            writer.write(np.zeros((32, 32, 3), np.uint8))
    assert len(list(open_source(str(path), max_frames=3))) == 3


def test_gif_is_written(tmp_path):
    frames = [np.full((40, 60, 3), value, dtype=np.uint8) for value in (0, 80, 160)]
    path = write_gif(tmp_path / "out.gif", frames, fps=5.0, scale=0.5)
    assert path.exists() and path.stat().st_size > 0


def test_image_suffixes_are_lowercase():
    assert all(suffix == suffix.lower() for suffix in IMAGE_SUFFIXES)

"""On-disk dataset layout: YOLO for detection, MOTChallenge for tracking.

Two layouts because the two jobs want different things.  Ultralytics wants a
flat directory of images with a sibling directory of one text file per image.
Tracking evaluation wants whole sequences with a single ground-truth file each,
in MOTChallenge column order, so that `py-motmetrics` and every other MOT tool
can read it without a shim.  Both are written from the same generator, so they
cannot drift apart.

    data/processed/
      images/{train,val,test}/SEQ_FRAME.jpg
      labels/{train,val,test}/SEQ_FRAME.txt        class cx cy w h, normalised
      mot/test/SEQ/
        seqinfo.ini
        gt/gt.txt                                   MOT16 columns
      dataset.json                                  what was generated and how

``dataset.json`` is the provenance record: seeds, counts, class histogram and
the config every split was rendered with.  ``scripts/make_dataset.py`` writes
it and ``data/DATASET_CARD.md`` quotes from it, so the card cannot claim a
class balance the data does not have.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from ..paths import serialise_path, write_text_lf
from .synth import CLASSES, Annotation, SynthConfig, generate_sequence

__all__ = [
    "SplitStats",
    "write_split",
    "write_mot_sequence",
    "read_mot_gt",
    "load_yolo_labels",
    "dataset_summary",
]


class SplitStats:
    """Counters accumulated while a split is written."""

    def __init__(self, split: str) -> None:
        self.split = split
        self.n_sequences = 0
        self.n_images = 0
        self.n_boxes = 0
        self.n_dropped_occluded = 0
        self.class_counts = {name: 0 for name in CLASSES}
        self.visibility_sum = 0.0
        self.box_area_fractions: list[float] = []

    def as_dict(self) -> dict:
        areas = np.asarray(self.box_area_fractions, dtype=np.float64)
        return {
            "split": self.split,
            "sequences": self.n_sequences,
            "images": self.n_images,
            "labelled_boxes": self.n_boxes,
            "boxes_dropped_below_visibility": self.n_dropped_occluded,
            "class_counts": dict(self.class_counts),
            "mean_visibility_of_labelled": (
                round(self.visibility_sum / self.n_boxes, 4) if self.n_boxes else 0.0
            ),
            "box_area_fraction": {
                "p05": round(float(np.percentile(areas, 5)), 5) if areas.size else 0.0,
                "p50": round(float(np.percentile(areas, 50)), 5) if areas.size else 0.0,
                "p95": round(float(np.percentile(areas, 95)), 5) if areas.size else 0.0,
            },
        }


def _label_line(ann: Annotation, width: int, height: int) -> str:
    x1, y1, x2, y2 = ann.xyxy
    cx = (x1 + x2) / 2.0 / width
    cy = (y1 + y2) / 2.0 / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    return f"{ann.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def write_split(
    root: Path,
    split: str,
    configs: Iterable[SynthConfig],
    *,
    write_mot: bool = False,
    jpeg_quality: int = 92,
) -> SplitStats:
    """Render every config and write images, YOLO labels and optional MOT truth.

    A box is written to the YOLO label file only if its visibility is at or
    above ``cfg.min_label_visibility``.  Training a detector on boxes that are
    ninety percent behind a shelf teaches it to hallucinate, and the MOT ground
    truth keeps the full set with the visibility column intact, so nothing is
    lost for the tracking metrics.
    """
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    stats = SplitStats(split)
    for cfg in configs:
        seq_name = f"{split}_seq{cfg.seed:05d}"
        rows: list[tuple] = []
        for frame in generate_sequence(cfg):
            stem = f"{seq_name}_{frame.frame_id:04d}"
            cv2.imwrite(
                str(image_dir / f"{stem}.jpg"),
                frame.image,
                [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
            )
            lines = []
            for ann in frame.annotations:
                x1, y1, x2, y2 = ann.xyxy
                if x2 - x1 < 2.0 or y2 - y1 < 2.0:
                    continue
                if ann.visibility < cfg.min_label_visibility:
                    stats.n_dropped_occluded += 1
                else:
                    lines.append(_label_line(ann, cfg.width, cfg.height))
                    stats.n_boxes += 1
                    stats.class_counts[CLASSES[ann.class_id]] += 1
                    stats.visibility_sum += ann.visibility
                    stats.box_area_fractions.append(
                        (x2 - x1) * (y2 - y1) / (cfg.width * cfg.height)
                    )
                if write_mot:
                    rows.append(
                        (
                            frame.frame_id + 1,
                            ann.instance_id,
                            x1,
                            y1,
                            x2 - x1,
                            y2 - y1,
                            1,
                            ann.class_id + 1,
                            round(ann.visibility, 4),
                        )
                    )
            write_text_lf(label_dir / f"{stem}.txt", "\n".join(lines) + "\n")
            stats.n_images += 1
        stats.n_sequences += 1
        if write_mot:
            write_mot_sequence(root / "mot" / split / seq_name, cfg, rows)
    return stats


def write_mot_sequence(seq_dir: Path, cfg: SynthConfig, rows: list[tuple]) -> None:
    """Write ``seqinfo.ini`` and ``gt/gt.txt`` for one sequence."""
    (seq_dir / "gt").mkdir(parents=True, exist_ok=True)
    write_text_lf(
        seq_dir / "seqinfo.ini",
        "\n".join(
            [
                "[Sequence]",
                f"name={seq_dir.name}",
                "imDir=img1",
                "frameRate=15",
                f"seqLength={cfg.n_frames}",
                f"imWidth={cfg.width}",
                f"imHeight={cfg.height}",
                "imExt=.jpg",
                "",
            ]
        ),
    )
    # newline is pinned here for the same reason write_text_lf exists: a MOT
    # ground-truth file written with CRLF is still parseable, but the digest
    # comparison in tests/test_determinism.py would then depend on the platform
    # that generated it.
    with (seq_dir / "gt" / "gt.txt").open("w", newline="\n", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda r: (r[0], r[1])):
            handle.write(
                "{},{},{:.2f},{:.2f},{:.2f},{:.2f},{},{},{}\n".format(*row)
            )


def read_mot_gt(path: Path, min_visibility: float = 0.0) -> dict[int, list[tuple[int, int, np.ndarray]]]:
    """Read a MOT ground-truth file into ``{frame: [(id, class_id, xyxy)]}``.

    Frames come back zero-based, matching everything else in this repository.
    Rows below ``min_visibility`` are dropped: a box that is ninety percent
    occluded is not something a detector can reasonably be scored against, and
    the MOTChallenge protocol excludes them too.
    """
    by_frame: dict[int, list[tuple[int, int, np.ndarray]]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(",")
        frame = int(parts[0]) - 1
        track_id = int(parts[1])
        x, y, w, h = (float(v) for v in parts[2:6])
        class_id = int(parts[7]) - 1 if len(parts) > 7 else 0
        visibility = float(parts[8]) if len(parts) > 8 else 1.0
        if visibility < min_visibility:
            continue
        by_frame.setdefault(frame, []).append(
            (track_id, class_id, np.array([x, y, x + w, y + h], dtype=np.float64))
        )
    return by_frame


def load_yolo_labels(path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Read one YOLO label file into absolute xyxy boxes and class ids."""
    text = Path(path).read_text(encoding="utf-8") if Path(path).exists() else ""
    boxes, classes = [], []
    for line in text.splitlines():
        if not line.strip():
            continue
        cls, cx, cy, w, h = (float(v) for v in line.split())
        boxes.append(
            [
                (cx - w / 2) * width,
                (cy - h / 2) * height,
                (cx + w / 2) * width,
                (cy + h / 2) * height,
            ]
        )
        classes.append(int(cls))
    return (
        np.asarray(boxes, dtype=np.float64).reshape(-1, 4),
        np.asarray(classes, dtype=np.int64).reshape(-1),
    )


def dataset_summary(root: Path, splits: dict[str, SplitStats], seeds: dict[str, list[int]],
                    configs: dict[str, SynthConfig]) -> dict:
    """Provenance record written to ``dataset.json``."""
    return {
        "classes": list(CLASSES),
        "generator": "edgetrack.data.synth",
        "splits": {name: stats.as_dict() for name, stats in splits.items()},
        "seeds": {name: list(values) for name, values in seeds.items()},
        "configs": {name: asdict(cfg) for name, cfg in configs.items()},
        "root": serialise_path(root),
    }


def write_dataset_json(root: Path, summary: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    write_text_lf((root / "dataset.json"), json.dumps(summary, indent=2) + "\n")

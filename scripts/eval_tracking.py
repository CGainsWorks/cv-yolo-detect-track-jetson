#!/usr/bin/env python
"""MOTA, IDF1 and ID switches for each tracker on the test sequences.

    python scripts/eval_tracking.py --model models/yolov8n_edgetrack_fp32.onnx \\
                                    --trackers bytetrack deepsort \\
                                    --out results/tracking.json

Detections are computed once per frame and shared by every tracker in the run.
That is the point of the script: a comparison where each tracker ran its own
detector would be comparing two things at once, and the difference between
ByteTrack and DeepSORT is supposed to be a difference in association.

``--detections gt`` swaps the detector for the ground truth boxes, filtered by
the same visibility threshold the evaluator uses.  That removes detection error
entirely and answers a different question: how good is the association on its
own, given perfect input.  Both numbers are in ``docs/tracking.md`` because the
gap between them is where the engineering effort should go.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

import _bootstrap  # noqa: F401
from edgetrack.data.dataset import read_mot_gt
from edgetrack.data.synth import CLASSES
from edgetrack.detect import DetectorConfig, build_detector
from edgetrack.metrics.idswitch import attribute_switches
from edgetrack.metrics.mot import evaluate_mot, id_measures_from_counts
from edgetrack.paths import serialise_path, write_text_lf
from edgetrack.track import build_tracker
from edgetrack.track.appearance import attach_features, build_encoder
from edgetrack.types import Detection

AGGREGATE_COUNTERS = (
    "num_switches",
    "num_fragmentations",
    "num_false_positives",
    "num_misses",
    "num_matches",
    "num_detections",
    "num_objects",
    "num_predictions",
    "mostly_tracked",
    "partially_tracked",
    "mostly_lost",
    "num_gt_ids",
    "num_hyp_ids",
    "idtp",
    "idfp",
    "idfn",
)


def sequence_frames(data_root: Path, split: str, seq_name: str) -> list[Path]:
    image_dir = data_root / "images" / split
    frames = sorted(image_dir.glob(f"{seq_name}_*.jpg"))
    if not frames:
        raise FileNotFoundError(f"no frames for sequence {seq_name} under {image_dir}")
    return frames


def gt_detections(
    gt_frame: list[tuple[int, int, np.ndarray]], jitter: float, rng: np.random.Generator
) -> list[Detection]:
    """Ground-truth boxes dressed up as detections.

    ``jitter`` adds gaussian noise proportional to box height, because a
    tracker fed geometrically perfect boxes solves a problem no detector
    creates and the resulting numbers say nothing.  The default is zero and
    the published run uses it; the flag exists for the sensitivity study in
    docs/tracking.md.
    """
    out = []
    for _, class_id, box in gt_frame:
        box = box.astype(np.float64).copy()
        if jitter > 0:
            height = max(box[3] - box[1], 1.0)
            box += rng.normal(0.0, jitter * height, size=4)
        out.append(Detection(tuple(float(v) for v in box), 0.99, int(class_id)))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default="models/yolov8n_edgetrack_fp32.onnx", type=Path)
    parser.add_argument("--backend", default="onnx")
    parser.add_argument("--data-root", default="data/processed", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--trackers", nargs="+", default=["bytetrack", "deepsort"])
    parser.add_argument("--detections", default="model", choices=["model", "gt"])
    parser.add_argument("--gt-jitter", type=float, default=0.0)
    parser.add_argument("--appearance", default="histogram", choices=["histogram", "reid"])
    parser.add_argument("--reid-model", default="models/reid_edgetrack.onnx")
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--imgsz", nargs=2, type=int, default=[640, 384], metavar=("W", "H"))
    parser.add_argument("--min-visibility", type=float, default=0.35)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--sequences", nargs="*", default=None)
    parser.add_argument("--mot-out", default=None, type=Path, help="directory for MOT result files")
    parser.add_argument("--out", default="results/tracking.json", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    mot_root = Path(args.data_root) / "mot" / args.split
    sequences = sorted(p.name for p in mot_root.iterdir() if p.is_dir())
    if args.sequences:
        sequences = [s for s in sequences if s in set(args.sequences)]
    if not sequences:
        raise FileNotFoundError(
            f"no MOT sequences under {mot_root}; run `python scripts/make_dataset.py` first"
        )

    detector = None
    encoder = None
    if args.detections == "model":
        config = DetectorConfig(
            imgsz=(int(args.imgsz[0]), int(args.imgsz[1])),
            conf_threshold=float(args.conf),
            class_names=CLASSES,
        )
        detector = build_detector(args.model, args.backend, config)
        detector.warmup(5)
    if any(t.lower().startswith("deep") for t in args.trackers):
        encoder = build_encoder(
            args.appearance, args.reid_model if args.appearance == "reid" else None
        )

    tracker_configs = {
        "bytetrack": yaml.safe_load(
            Path("configs/tracker_bytetrack.yaml").read_text(encoding="utf-8")
        ),
        "deepsort": {
            k: v
            for k, v in yaml.safe_load(
                Path("configs/tracker_deepsort.yaml").read_text(encoding="utf-8")
            ).items()
            if k != "appearance"
        },
    }

    report: dict = {
        "detections": args.detections,
        "model": serialise_path(args.model) if args.detections == "model" else None,
        "appearance": args.appearance if encoder is not None else None,
        "split": args.split,
        "sequences": sequences,
        "min_visibility": args.min_visibility,
        "iou_threshold": args.iou_threshold,
        "conf_threshold": args.conf,
        "trackers": {},
    }

    rng = np.random.default_rng(12345)
    for tracker_name in args.trackers:
        tracker = build_tracker(tracker_name, **tracker_configs[tracker_name.lower()])
        needs_features = tracker_name.lower().startswith("deep")
        per_sequence = {}
        totals = {key: 0 for key in AGGREGATE_COUNTERS}
        iou_sum = 0.0
        all_switch_records = []
        frame_times: list[float] = []

        for seq_name in sequences:
            gt_path = mot_root / seq_name / "gt" / "gt.txt"
            gt_rows = read_mot_gt(gt_path, min_visibility=args.min_visibility)
            gt_all = read_mot_gt(gt_path, min_visibility=0.0)
            gt_frames = {
                frame: {track_id: box for track_id, _, box in rows}
                for frame, rows in gt_rows.items()
            }
            gt_classes = {
                track_id: class_id for rows in gt_all.values() for track_id, class_id, _ in rows
            }
            visibility = _visibility_table(gt_path)

            tracker.reset()
            hyp_frames: dict[int, dict[int, np.ndarray]] = {}
            frames = sequence_frames(args.data_root, args.split, seq_name)
            for frame_index, image_path in enumerate(frames):
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError(f"could not decode {image_path}")
                started = time.perf_counter()
                if detector is not None:
                    detections = detector.detect(image)
                else:
                    # The visibility-filtered rows, not every row.  Feeding the
                    # tracker boxes the evaluator excludes turns each one into
                    # a false positive and the "perfect detections" experiment
                    # reports a worse MOTA than the real detector does, which
                    # is an artefact of the harness rather than a result.
                    detections = gt_detections(
                        gt_rows.get(frame_index, []), args.gt_jitter, rng
                    )
                if needs_features:
                    detections = attach_features(encoder, image, detections)
                tracks = tracker.update(detections, frame_index)
                frame_times.append((time.perf_counter() - started) * 1000.0)
                hyp_frames[frame_index] = {
                    t.track_id: np.asarray(t.xyxy, dtype=np.float64) for t in tracks
                }

            metrics = evaluate_mot(gt_frames, hyp_frames, args.iou_threshold)
            analysis = attribute_switches(
                metrics.events, gt_frames, gt_classes, visibility
            )
            per_sequence[seq_name] = {
                **metrics.as_dict(),
                "switch_causes": analysis.counts,
            }
            for key in AGGREGATE_COUNTERS:
                totals[key] += getattr(metrics, key)
            if metrics.num_detections:
                iou_sum += (1.0 - metrics.motp) * metrics.num_detections
            all_switch_records.extend(analysis.records)

            if args.mot_out:
                _write_mot_results(Path(args.mot_out) / tracker_name, seq_name, hyp_frames)

        idf1, idp, idr = id_measures_from_counts(totals["idtp"], totals["idfp"], totals["idfn"])
        mota = 1.0 - (
            totals["num_misses"] + totals["num_false_positives"] + totals["num_switches"]
        ) / max(totals["num_objects"], 1)
        causes: dict[str, int] = {}
        for record in all_switch_records:
            causes[record.cause] = causes.get(record.cause, 0) + 1

        times = np.asarray(frame_times)
        report["trackers"][tracker_name] = {
            "config": tracker_configs[tracker_name.lower()],
            "aggregate": {
                "mota": round(float(mota), 5),
                "motp": round(1.0 - iou_sum / max(totals["num_detections"], 1), 5),
                "motp_iou": round(iou_sum / max(totals["num_detections"], 1), 5),
                "idf1": round(idf1, 5),
                "idp": round(idp, 5),
                "idr": round(idr, 5),
                "precision": round(
                    totals["num_detections"] / max(totals["num_predictions"], 1), 5
                ),
                "recall": round(totals["num_detections"] / max(totals["num_objects"], 1), 5),
                **{key: totals[key] for key in AGGREGATE_COUNTERS},
            },
            "switch_causes": causes,
            "frame_ms": {
                "mean": round(float(times.mean()), 3),
                "p50": round(float(np.percentile(times, 50)), 3),
                "p95": round(float(np.percentile(times, 95)), 3),
            },
            "per_sequence": per_sequence,
        }
        if not args.quiet:
            aggregate = report["trackers"][tracker_name]["aggregate"]
            print(
                f"{tracker_name:<10} MOTA {aggregate['mota']:.4f}  IDF1 {aggregate['idf1']:.4f}  "
                f"IDsw {aggregate['num_switches']}  Frag {aggregate['num_fragmentations']}  "
                f"MT {aggregate['mostly_tracked']}/{aggregate['num_gt_ids']}",
                flush=True,
            )

    if detector is not None:
        detector.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_text_lf(args.out, json.dumps(report, indent=2) + "\n")
    if not args.quiet:
        print(f"wrote {args.out}")
    return 0


def _visibility_table(gt_path: Path) -> dict[int, dict[int, float]]:
    """``{frame: {track_id: visibility}}`` straight from the MOT columns."""
    table: dict[int, dict[int, float]] = {}
    for line in gt_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 9:
            continue
        table.setdefault(int(parts[0]) - 1, {})[int(parts[1])] = float(parts[8])
    return table


def _write_mot_results(
    root: Path, seq_name: str, hyp_frames: dict[int, dict[int, np.ndarray]]
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"{seq_name}.txt").open("w", encoding="utf-8") as handle:
        for frame_id in sorted(hyp_frames):
            for track_id, box in sorted(hyp_frames[frame_id].items()):
                x1, y1, x2, y2 = box
                handle.write(
                    f"{frame_id + 1},{track_id},{x1:.2f},{y1:.2f},"
                    f"{x2 - x1:.2f},{y2 - y1:.2f},1,-1,-1,-1\n"
                )


if __name__ == "__main__":
    sys.exit(main())

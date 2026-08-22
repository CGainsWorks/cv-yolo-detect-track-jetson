#!/usr/bin/env python
"""Choose tracker parameters on the val split.

    python scripts/tune_tracker.py --tracker bytetrack
    python scripts/tune_tracker.py --tracker deepsort --out results/tuning_deepsort.json

Tuned on val, reported on test.  Choosing thresholds on the split the headline
numbers come from would make those numbers a training score, and the effect is
not small: on this data the best and worst settings in the sweep below differ
by more than ten points of IDF1.

The objective is IDF1 rather than MOTA.  MOTA is dominated by the detector and
barely moves as association parameters change, so optimising it would pick the
parameters that happen to suppress a few false positives and ignore identity
entirely.

Detections are computed once for the whole sweep and reused for every
candidate, which is what makes an eighty-point sweep take a minute rather than
an hour.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import _bootstrap  # noqa: F401
from edgetrack.data.dataset import read_mot_gt
from edgetrack.data.synth import CLASSES
from edgetrack.detect import DetectorConfig, build_detector
from edgetrack.metrics.mot import evaluate_mot, id_measures_from_counts
from edgetrack.paths import serialise_path
from edgetrack.track import build_tracker
from edgetrack.track.appearance import attach_features, build_encoder

GRIDS = {
    "bytetrack": {
        "track_thresh": [0.35, 0.5, 0.6],
        "match_thresh": [0.7, 0.8, 0.9],
        "track_buffer": [10, 30, 60],
        "n_init": [1, 2, 3],
    },
    "deepsort": {
        "max_cosine_distance": [0.15, 0.25, 0.35, 0.5],
        "max_iou_distance": [0.6, 0.7, 0.85],
        "max_age": [10, 30, 60],
        "n_init": [2, 3],
    },
}


def cache_detections(
    detector, encoder, data_root: Path, split: str, sequences: list[str], quiet: bool
) -> dict[str, list[list]]:
    cache: dict[str, list[list]] = {}
    for seq_name in sequences:
        frames = sorted((Path(data_root) / "images" / split).glob(f"{seq_name}_*.jpg"))
        per_frame = []
        for path in frames:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"could not decode {path}")
            detections = detector.detect(image)
            if encoder is not None:
                detections = attach_features(encoder, image, detections)
            per_frame.append(detections)
        cache[seq_name] = per_frame
        if not quiet:
            print(f"  cached {seq_name}: {len(per_frame)} frames", flush=True)
    return cache


def score(tracker_name: str, params: dict, cache: dict, truth: dict, iou_threshold: float) -> dict:
    tracker = build_tracker(tracker_name, **params)
    totals = {"idtp": 0, "idfp": 0, "idfn": 0, "num_misses": 0,
              "num_false_positives": 0, "num_switches": 0, "num_objects": 0}
    for seq_name, per_frame in cache.items():
        tracker.reset()
        hyp: dict[int, dict[int, np.ndarray]] = {}
        for frame_index, detections in enumerate(per_frame):
            tracks = tracker.update(detections, frame_index)
            hyp[frame_index] = {
                t.track_id: np.asarray(t.xyxy, dtype=np.float64) for t in tracks
            }
        metrics = evaluate_mot(truth[seq_name], hyp, iou_threshold)
        for key in totals:
            totals[key] += getattr(metrics, key)
    idf1, idp, idr = id_measures_from_counts(totals["idtp"], totals["idfp"], totals["idfn"])
    mota = 1.0 - (
        totals["num_misses"] + totals["num_false_positives"] + totals["num_switches"]
    ) / max(totals["num_objects"], 1)
    return {
        "params": params,
        "idf1": round(idf1, 5),
        "idp": round(idp, 5),
        "idr": round(idr, 5),
        "mota": round(float(mota), 5),
        "num_switches": totals["num_switches"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tracker", default="bytetrack", choices=sorted(GRIDS))
    parser.add_argument("--model", default="models/yolov8n_edgetrack_fp32.onnx", type=Path)
    parser.add_argument("--backend", default="onnx")
    parser.add_argument("--data-root", default="data/processed", type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--appearance", default="histogram", choices=["histogram", "reid"])
    parser.add_argument("--reid-model", default="models/reid_edgetrack.onnx")
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--min-visibility", type=float, default=0.35)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--out", default=None, type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    mot_root = Path(args.data_root) / "mot" / args.split
    if not mot_root.exists():
        raise FileNotFoundError(
            f"no MOT ground truth under {mot_root}.  Regenerate the dataset: "
            "`python scripts/make_dataset.py`"
        )
    sequences = sorted(p.name for p in mot_root.iterdir() if p.is_dir())
    truth = {
        name: {
            frame: {track_id: box for track_id, _, box in rows}
            for frame, rows in read_mot_gt(
                mot_root / name / "gt" / "gt.txt", args.min_visibility
            ).items()
        }
        for name in sequences
    }

    config = DetectorConfig(
        imgsz=(640, 384), conf_threshold=float(args.conf), class_names=CLASSES
    )
    detector = build_detector(args.model, args.backend, config)
    detector.warmup(5)
    encoder = (
        build_encoder(args.appearance, args.reid_model if args.appearance == "reid" else None)
        if args.tracker == "deepsort"
        else None
    )

    if not args.quiet:
        print(f"caching detections over {len(sequences)} {args.split} sequences", flush=True)
    cache = cache_detections(detector, encoder, args.data_root, args.split, sequences, args.quiet)
    detector.close()

    grid = GRIDS[args.tracker]
    keys = sorted(grid)
    combinations = [
        dict(zip(keys, values))
        for values in itertools.product(*(grid[k] for k in keys))
    ]
    if not args.quiet:
        print(f"sweeping {len(combinations)} combinations", flush=True)

    started = time.perf_counter()
    results = []
    for index, params in enumerate(combinations):
        # det_thresh must stay at or above track_thresh; ByteTrackConfig
        # rejects the alternative rather than silently reordering them.
        candidate = dict(params)
        if args.tracker == "bytetrack":
            candidate["det_thresh"] = max(candidate["track_thresh"], 0.6)
        results.append(score(args.tracker, candidate, cache, truth, args.iou_threshold))
        if not args.quiet and index % 10 == 0:
            print(f"  {index}/{len(combinations)}", flush=True)
    elapsed = time.perf_counter() - started

    # Ties are common: eighty-one settings produce about a dozen distinct
    # scores, so the tie-break decides the published configuration more often
    # than the objective does.  Break towards the more conservative tracker:
    # fewer identity switches, then a larger confirmation count so fewer
    # spurious tracks reach a subscriber, then a shorter buffer so less state
    # is carried on a device with 8 GB shared between everything.
    def rank(result: dict) -> tuple:
        params = result["params"]
        return (
            -result["idf1"],
            -result["mota"],
            result["num_switches"],
            -params.get("n_init", 0),
            params.get("track_buffer", params.get("max_age", 0)),
        )

    results.sort(key=rank)
    report = {
        "tracker": args.tracker,
        "split": args.split,
        "model": serialise_path(args.model),
        "appearance": args.appearance if encoder is not None else None,
        "objective": "idf1",
        "combinations": len(combinations),
        "seconds": round(elapsed, 1),
        "best": results[0],
        "worst": results[-1],
        "spread_idf1": round(results[0]["idf1"] - results[-1]["idf1"], 5),
        "all": results,
    }
    out = args.out or Path(f"results/tuning_{args.tracker}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if not args.quiet:
        print(f"best  IDF1 {results[0]['idf1']:.4f}  {json.dumps(results[0]['params'])}")
        print(f"worst IDF1 {results[-1]['idf1']:.4f}  {json.dumps(results[-1]['params'])}")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

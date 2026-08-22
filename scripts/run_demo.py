#!/usr/bin/env python
"""End to end in one command: generate a scene, detect, track, score, render.

    python scripts/run_demo.py
    python scripts/run_demo.py --source clip.mp4 --tracker deepsort --video out.mp4

With the default synthetic source the demo knows the ground truth, so it prints
real MOTA, IDF1 and ID-switch numbers rather than a frame count.  That is the
difference between a demo that proves the pipeline runs and one that proves it
works.  On a video file or a camera there is no ground truth and the metrics
block is omitted rather than faked.

This is what ``make demo`` and ``docker compose up`` run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from edgetrack.data.synth import CLASSES, config_for_split, generate_sequence
from edgetrack.detect import DetectorConfig, build_detector
from edgetrack.metrics.idswitch import attribute_switches
from edgetrack.metrics.mot import evaluate_mot
from edgetrack.paths import serialise_path, write_text_lf
from edgetrack.pipeline import Pipeline, annotate
from edgetrack.track import build_tracker
from edgetrack.track.appearance import build_encoder
from edgetrack.video import VideoWriter, open_source, write_gif


def synthetic_truth(seed: int, n_frames: int, min_visibility: float) -> tuple[dict, dict, dict]:
    """Ground truth for a synth source, regenerated from the same seed.

    Regenerating rather than caching is the point: if the generator ever
    changed, the demo's ground truth would change with the frames it scores,
    and a mismatch would show up as an impossible metric instead of a silent
    lie.
    """
    cfg = config_for_split("test", seed, n_frames=n_frames)
    frames: dict[int, dict[int, np.ndarray]] = {}
    classes: dict[int, int] = {}
    visibility: dict[int, dict[int, float]] = {}
    for frame in generate_sequence(cfg):
        kept = {}
        vis = {}
        for ann in frame.annotations:
            classes[ann.instance_id] = ann.class_id
            vis[ann.instance_id] = ann.visibility
            if ann.visibility >= min_visibility:
                kept[ann.instance_id] = np.asarray(ann.xyxy, dtype=np.float64)
        frames[frame.frame_id] = kept
        visibility[frame.frame_id] = vis
    return frames, classes, visibility


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", default="synth:7")
    parser.add_argument("--model", default="models/yolov8n_edgetrack_fp32.onnx", type=Path)
    parser.add_argument("--backend", default="onnx")
    parser.add_argument("--tracker", default="bytetrack")
    parser.add_argument("--appearance", default="histogram", choices=["histogram", "reid"])
    parser.add_argument("--reid-model", default="models/reid_edgetrack.onnx")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", nargs=2, type=int, default=[640, 384], metavar=("W", "H"))
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--video", default=None, type=Path, help="write an annotated mp4")
    parser.add_argument("--gif", default=None, type=Path, help="write an annotated gif")
    parser.add_argument("--gif-stride", type=int, default=2)
    parser.add_argument("--min-visibility", type=float, default=0.35)
    parser.add_argument("--out", default="results/demo.json", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if not args.model.exists():
        raise FileNotFoundError(
            f"no model at {args.model}.  Run `python scripts/fetch_model.py` to "
            "download the published weights, or `make train export` to build them."
        )

    config = DetectorConfig(
        imgsz=(int(args.imgsz[0]), int(args.imgsz[1])),
        conf_threshold=float(args.conf),
        class_names=CLASSES,
    )
    detector = build_detector(args.model, args.backend, config)
    detector.warmup(5)
    # Captured before the pipeline is closed.  A backend that has released its
    # session cannot describe itself, and reading `info` after `close()` is a
    # crash rather than a stale value.
    detector_info = detector.info.as_dict()

    needs_features = args.tracker.lower().startswith("deep")
    encoder = (
        build_encoder(args.appearance, args.reid_model if args.appearance == "reid" else None)
        if needs_features
        else None
    )
    pipeline = Pipeline(detector, build_tracker(args.tracker), encoder)

    source = open_source(args.source, args.frames)
    writer = (
        VideoWriter(args.video, source.fps, (source.width, source.height))
        if args.video
        else None
    )
    gif_frames: list[np.ndarray] = []

    hyp_frames: dict[int, dict[int, np.ndarray]] = {}
    stage_times: dict[str, list[float]] = {}
    unique_ids: set[int] = set()
    detections_seen = 0

    started = time.perf_counter()
    frame_count = 0
    for index, frame in enumerate(source):
        result = pipeline.process(frame, index)
        frame_count += 1
        detections_seen += len(result.detections)
        hyp_frames[index] = {
            t.track_id: np.asarray(t.xyxy, dtype=np.float64) for t in result.tracks
        }
        unique_ids.update(t.track_id for t in result.tracks)
        for key, value in result.timings_ms.items():
            stage_times.setdefault(key, []).append(value)
        if writer is not None or args.gif is not None:
            annotated = annotate(frame, result, CLASSES)
            if writer is not None:
                writer.write(annotated)
            if args.gif is not None and index % max(1, args.gif_stride) == 0:
                gif_frames.append(annotated)
    wall = time.perf_counter() - started

    if writer is not None:
        writer.close()
    if args.gif is not None and gif_frames:
        write_gif(args.gif, gif_frames, fps=source.fps / max(1, args.gif_stride))
    pipeline.close()

    report: dict = {
        "source": args.source,
        "source_description": source.description,
        "model": serialise_path(args.model),
        "backend": args.backend,
        "detector": detector_info,
        "tracker": args.tracker,
        "appearance": args.appearance if needs_features else None,
        "frames_run": frame_count,
        "detections_total": detections_seen,
        "unique_track_ids": len(unique_ids),
        "wall_seconds": round(wall, 3),
        "fps": round(frame_count / max(wall, 1e-9), 2),
        "stages_ms": {
            key: {
                "mean": round(float(np.mean(values)), 3),
                "p50": round(float(np.percentile(values, 50)), 3),
                "p95": round(float(np.percentile(values, 95)), 3),
            }
            for key, values in stage_times.items()
        },
        "video": serialise_path(args.video),
        "gif": serialise_path(args.gif),
    }

    if args.source.startswith("synth"):
        seed = int(args.source.split(":", 1)[1]) if ":" in args.source else 0
        gt_frames, gt_classes, visibility = synthetic_truth(
            seed, frame_count, args.min_visibility
        )
        metrics = evaluate_mot(gt_frames, hyp_frames, 0.5)
        analysis = attribute_switches(metrics.events, gt_frames, gt_classes, visibility)
        report["ground_truth"] = "synthetic, regenerated from the source seed"
        report["metrics"] = metrics.as_dict()
        report["switch_causes"] = analysis.counts

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_text_lf(args.out, json.dumps(report, indent=2) + "\n")

    if not args.quiet:
        print(f"{source.description}")
        print(
            f"  {frame_count} frames in {wall:.2f}s = {report['fps']:.1f} FPS "
            f"({detector_info['backend']}/{detector_info['precision']})"
        )
        print(
            f"  detections {detections_seen}, unique track ids {len(unique_ids)}, "
            f"per-frame p50 {report['stages_ms']['total']['p50']:.2f} ms"
        )
        if "metrics" in report:
            m = report["metrics"]
            print(
                f"  MOTA {m['mota']:.4f}  IDF1 {m['idf1']:.4f}  IDsw {m['num_switches']}  "
                f"FP {m['num_false_positives']}  FN {m['num_misses']}"
            )
            print(f"  switch causes: {json.dumps(report['switch_causes'])}")
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

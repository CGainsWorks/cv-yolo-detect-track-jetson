#!/usr/bin/env python
"""Detection accuracy for one or more model variants on the same images.

    python scripts/evaluate.py --models models/yolov8n_edgetrack_fp32.onnx \\
                                        models/yolov8n_edgetrack_int8.onnx \\
                               --split test --out results/detection.json

Every variant is scored on the same frames with the same decoder, the same NMS
and the same metric implementation, so the differences between the rows are
differences in the arithmetic.  This is the accuracy half of the headline
table; ``scripts/benchmark_latency.py`` is the latency half and
``scripts/make_results_table.py`` joins them.

The detection threshold is deliberately low (0.001) for the mAP pass.  mAP
integrates over the whole precision-recall curve and a high threshold silently
truncates it, which flatters whichever model is most confident rather than
whichever is most accurate.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import _bootstrap  # noqa: F401
from edgetrack.data.dataset import load_yolo_labels
from edgetrack.data.synth import CLASSES
from edgetrack.detect import DetectorConfig, build_detector
from edgetrack.metrics.detection import DetectionSample, evaluate_detections
from edgetrack.paths import serialise_path


def collect_samples(
    detector, image_paths: list[Path], label_dir: Path, progress: bool = True
) -> tuple[list[DetectionSample], list[float]]:
    samples: list[DetectionSample] = []
    latencies: list[float] = []
    for index, path in enumerate(image_paths):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"could not decode {path}")
        started = time.perf_counter()
        detections = detector.detect(image)
        latencies.append((time.perf_counter() - started) * 1000.0)
        gt_boxes, gt_classes = load_yolo_labels(
            label_dir / f"{path.stem}.txt", image.shape[1], image.shape[0]
        )
        samples.append(
            DetectionSample(
                pred_boxes=np.asarray(
                    [d.xyxy for d in detections], dtype=np.float64
                ).reshape(-1, 4),
                pred_scores=np.asarray([d.score for d in detections], dtype=np.float64),
                pred_classes=np.asarray([d.class_id for d in detections], dtype=np.int64),
                gt_boxes=gt_boxes,
                gt_classes=gt_classes,
            )
        )
        if progress and index % 200 == 0:
            print(f"  {index}/{len(image_paths)}", flush=True)
    return samples, latencies


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--models", nargs="+", required=True, type=Path)
    parser.add_argument("--backend", default="onnx")
    parser.add_argument("--data-root", default="data/processed", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None, help="score only the first N images")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--imgsz", nargs=2, type=int, default=[640, 384], metavar=("W", "H"))
    parser.add_argument("--providers", nargs="*", default=None)
    parser.add_argument("--out", default="results/detection.json", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    image_dir = Path(args.data_root) / "images" / args.split
    label_dir = Path(args.data_root) / "labels" / args.split
    images = sorted(image_dir.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(
            f"no images under {image_dir}; run `python scripts/make_dataset.py` first"
        )
    if args.limit:
        images = images[: args.limit]

    config = DetectorConfig(
        imgsz=(int(args.imgsz[0]), int(args.imgsz[1])),
        conf_threshold=float(args.conf),
        iou_threshold=float(args.iou),
        class_names=CLASSES,
        max_detections=300,
    )

    report: dict = {
        "split": args.split,
        "images": len(images),
        "conf_threshold": float(args.conf),
        "iou_threshold": float(args.iou),
        "imgsz_wh": [int(args.imgsz[0]), int(args.imgsz[1])],
        "classes": list(CLASSES),
        "variants": {},
    }

    for model_path in args.models:
        detector = build_detector(model_path, args.backend, config, providers=args.providers)
        detector.warmup(5)
        info = detector.info
        if not args.quiet:
            print(f"scoring {model_path.name}: {info.backend}/{info.precision}", flush=True)
        samples, latencies = collect_samples(detector, images, label_dir, not args.quiet)
        result = evaluate_detections(samples, list(range(len(CLASSES))))
        detector.close()

        latency = np.asarray(latencies)
        # Nested rather than flattened.  BackendInfo carries a `precision`
        # naming the numeric format and APResult carries a `precision` naming
        # the detection metric; flattening the two silently overwrote the first
        # with the second and the results table joined on a number between zero
        # and one instead of on the string "fp32".
        report["variants"][model_path.stem] = {
            "model": serialise_path(model_path),
            "detector": info.as_dict(),
            "metrics": result.as_dict(),
            "per_class_ap50_95_named": {
                CLASSES[int(k)]: v for k, v in result.as_dict()["per_class_ap50_95"].items()
            },
            "wall_ms": {
                "mean": round(float(latency.mean()), 3),
                "p50": round(float(np.percentile(latency, 50)), 3),
                "p95": round(float(np.percentile(latency, 95)), 3),
            },
        }
        if not args.quiet:
            metrics = report["variants"][model_path.stem]["metrics"]
            print(
                f"  mAP50 {metrics['map50']:.4f}  mAP50-95 {metrics['map50_95']:.4f}  "
                f"P {metrics['precision']:.4f}  R {metrics['recall']:.4f}",
                flush=True,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not args.quiet:
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

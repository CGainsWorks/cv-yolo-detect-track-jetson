#!/usr/bin/env python
"""Latency and throughput for each model variant on this host.

    python scripts/benchmark_latency.py --variants configs/benchmark.yaml \\
                                        --out results/latency.json

What it measures, and why it is measured that way.

*Steady state, not cold start.*  Every variant gets a warmup pass before the
clock starts.  The first inference on any accelerated backend includes
workspace allocation and algorithm selection and is five to fifty times the
steady-state cost, so including it would make the fastest backend look the
slowest.

*Real frames, not a constant.*  The timed loop cycles through frames from the
test split.  A single repeated frame lets the memory system cache everything
and, on a quantised model, lets branch predictors settle into a pattern no real
stream produces.

*Percentiles, not just the mean.*  An edge deployment that drops one frame in
twenty to a 90 ms stall is a different product from one that never exceeds
25 ms, and the means can be identical.

*The whole pipeline as well as the network.*  ``end_to_end`` includes
letterboxing, decode, NMS and the tracker.  The inference-only number is the
one that changes with precision; the end-to-end number is the one that decides
whether the camera keeps up.

Device honesty.  The record carries ``device_class``, which is ``jetson`` only
when ``/etc/nv_tegra_release`` exists.  Everything published today is
``desktop``.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

import _bootstrap  # noqa: F401
from edgetrack.data.synth import CLASSES
from edgetrack.detect import DetectorConfig, build_detector
from edgetrack.pipeline import Pipeline
from edgetrack.track import build_tracker
from edgetrack.track.appearance import build_encoder


def device_class() -> str:
    """``jetson`` on a Tegra board, ``desktop`` otherwise.

    The check is the presence of ``/etc/nv_tegra_release``, which every Jetpack
    image carries and nothing else does.  It is not a guess from the GPU name:
    an Orin's iGPU reports as ``Orin`` but so would a hypothetical discrete
    part, and the point of this field is that a reader can trust it.
    """
    return "jetson" if Path("/etc/nv_tegra_release").exists() else "desktop"


def host_record() -> dict:
    record = {
        "device_class": device_class(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "cpu_count": __import__("os").cpu_count(),
    }
    try:
        import onnxruntime as ort

        record["onnxruntime"] = ort.__version__
        record["onnxruntime_providers"] = ort.get_available_providers()
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        record["torch"] = torch.__version__
        record["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            record["gpu_name"] = torch.cuda.get_device_name(0)
            record["cuda"] = torch.version.cuda
            props = torch.cuda.get_device_properties(0)
            record["gpu_memory_gb"] = round(props.total_memory / 1024**3, 2)
            record["gpu_sm_count"] = props.multi_processor_count
    except ImportError:  # pragma: no cover
        pass
    if device_class() == "jetson":  # pragma: no cover - no Jetson here
        record["tegra_release"] = Path("/etc/nv_tegra_release").read_text().strip()
    return record


def percentiles(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(array.mean()), 3),
        "std": round(float(array.std()), 3),
        "p50": round(float(np.percentile(array, 50)), 3),
        "p90": round(float(np.percentile(array, 90)), 3),
        "p95": round(float(np.percentile(array, 95)), 3),
        "p99": round(float(np.percentile(array, 99)), 3),
        "min": round(float(array.min()), 3),
        "max": round(float(array.max()), 3),
    }


def load_frames(data_root: Path, split: str, count: int) -> list[np.ndarray]:
    paths = sorted((Path(data_root) / "images" / split).glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(
            f"no frames under {Path(data_root)/'images'/split}; run "
            "`python scripts/make_dataset.py` first"
        )
    step = max(1, len(paths) // count)
    chosen = paths[::step][:count]
    frames = []
    for path in chosen:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"could not decode {path}")
        frames.append(image)
    return frames


def bench_variant(
    variant: dict, frames: list[np.ndarray], iterations: int, warmup: int, tracker_name: str
) -> dict:
    config = DetectorConfig(
        imgsz=tuple(variant.get("imgsz", [640, 384])),
        conf_threshold=float(variant.get("conf", 0.25)),
        class_names=CLASSES,
    )
    detector = build_detector(
        variant["model"],
        variant.get("backend", "onnx"),
        config,
        providers=variant.get("providers"),
        device=variant.get("device", "cuda"),
        half=bool(variant.get("half", False)),
    )
    detector.warmup(warmup)

    stages: dict[str, list[float]] = {
        "preprocess": [], "inference": [], "postprocess": [], "detect_total": []
    }
    for i in range(iterations):
        frame = frames[i % len(frames)]
        started = time.perf_counter()
        detector.detect(frame)
        stages["detect_total"].append((time.perf_counter() - started) * 1000.0)
        stages["preprocess"].append(detector.last_preprocess_ms)
        stages["inference"].append(detector.last_inference_ms)
        stages["postprocess"].append(detector.last_postprocess_ms)

    # End to end, with the tracker in the loop.  A fresh tracker per variant so
    # that one variant's track state cannot change another's association cost.
    encoder = build_encoder("histogram") if tracker_name.startswith("deep") else None
    pipeline = Pipeline(detector, build_tracker(tracker_name), encoder)
    end_to_end: list[float] = []
    for i in range(warmup):
        pipeline.process(frames[i % len(frames)], i)
    pipeline.reset()
    for i in range(iterations):
        started = time.perf_counter()
        pipeline.process(frames[i % len(frames)], i)
        end_to_end.append((time.perf_counter() - started) * 1000.0)

    info = detector.info
    detector.close()

    inference = percentiles(stages["inference"])
    e2e = percentiles(end_to_end)
    return {
        "name": variant["name"],
        **info.as_dict(),
        "iterations": iterations,
        "warmup": warmup,
        "tracker": tracker_name,
        "stages_ms": {key: percentiles(values) for key, values in stages.items()},
        "end_to_end_ms": e2e,
        "fps_inference": round(1000.0 / max(inference["mean"], 1e-9), 2),
        "fps_end_to_end": round(1000.0 / max(e2e["mean"], 1e-9), 2),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--variants", default="configs/benchmark.yaml", type=Path)
    parser.add_argument("--data-root", default="data/processed", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--tracker", default="bytetrack")
    parser.add_argument("--only", nargs="*", default=None, help="run only these variant names")
    parser.add_argument("--out", default="results/latency.json", type=Path)
    parser.add_argument("--skip-missing", action="store_true",
                        help="skip variants whose model file or backend is unavailable")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    spec = yaml.safe_load(args.variants.read_text(encoding="utf-8"))
    variants = spec["variants"]
    if args.only:
        wanted = set(args.only)
        variants = [v for v in variants if v["name"] in wanted]
    if not variants:
        raise ValueError("no variants selected")

    frames = load_frames(args.data_root, args.split, args.frames)
    report = {
        "host": host_record(),
        "split": args.split,
        "frames_cycled": len(frames),
        "iterations": args.iterations,
        "warmup": args.warmup,
        "results": [],
        "skipped": [],
    }

    for variant in variants:
        try:
            entry = bench_variant(variant, frames, args.iterations, args.warmup, args.tracker)
        except (FileNotFoundError, RuntimeError, ImportError, ValueError) as error:
            if not args.skip_missing:
                raise
            report["skipped"].append({"name": variant["name"], "reason": str(error)})
            if not args.quiet:
                print(f"skipping {variant['name']}: {error}", flush=True)
            continue
        report["results"].append(entry)
        if not args.quiet:
            print(
                f"{entry['name']:<24} inference {entry['stages_ms']['inference']['p50']:>8.2f} ms "
                f"p50  end-to-end {entry['end_to_end_ms']['p50']:>8.2f} ms p50  "
                f"{entry['fps_end_to_end']:>7.1f} FPS",
                flush=True,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not args.quiet:
        print(f"wrote {args.out} ({report['host']['device_class']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

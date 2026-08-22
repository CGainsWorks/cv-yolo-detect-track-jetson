#!/usr/bin/env python
"""Is the GPU number measuring the model or the kernel launches?

    python scripts/probe_launch_bound.py

The PyTorch CUDA row in the headline table is slower than it has any right to
be: YOLOv8n is 4.9 GFLOPs and the GPU it runs on does over ten TFLOPs, so a
frame ought to take well under a millisecond of arithmetic.  It does not.  This
script establishes why, because "the GPU number looks wrong" is not a thing to
leave in a results table unexplained.

Two sweeps, and between them they answer it:

*Batch scaling.*  Run the same input at batch 1, 2, 4 and 8.  If the network
were compute bound the wall time would scale with the batch and the per-image
time would stay flat.  If it is launch bound the wall time stays flat and the
per-image time falls in half each time.

*Resolution scaling.*  Run one image at four times the pixels.  Compute bound
means four times the time.  Launch bound means no change, because the kernel
count is a property of the graph rather than of the tensor size.

The conclusion this reaches on the published host is that eager PyTorch at
batch 1 spends its time launching kernels, roughly a hundred layers' worth per
frame, and almost none of it computing.  That matters for two reasons.  It
means the PyTorch GPU row is an upper bound on what the model costs rather than
a measurement of it, and it is exactly the gap TensorRT exists to close: an
engine fuses convolution, batch normalisation and activation into single
kernels and can replay the whole graph as one CUDA graph, which removes most of
the launches rather than making each one faster.

So the honest reading of the table is that the Jetson numbers are not merely
missing, they are the only ones that would say what the model costs on an
accelerator.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401


def bench(model, tensor, torch, iterations: int, warmup: int) -> dict:
    with torch.inference_mode():
        for _ in range(warmup):
            model(tensor)
        torch.cuda.synchronize()
        samples = []
        for _ in range(iterations):
            started = time.perf_counter()
            model(tensor)
            # Synchronising is the whole point: without it this measures how
            # fast Python can enqueue work, not how fast the work finishes.
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - started) * 1000.0)
    array = np.asarray(samples)
    batch, _, height, width = tensor.shape
    return {
        "batch": int(batch),
        "height": int(height),
        "width": int(width),
        "megapixels": round(batch * height * width / 1e6, 4),
        "p50_ms": round(float(np.percentile(array, 50)), 3),
        "mean_ms": round(float(array.mean()), 3),
        "per_image_p50_ms": round(float(np.percentile(array, 50)) / batch, 3),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--weights", default="models/yolov8n_edgetrack.pt", type=Path)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--imgsz", nargs=2, type=int, default=[640, 384], metavar=("W", "H"))
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--out", default="results/launch_bound.json", type=Path)
    args = parser.parse_args(argv)

    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("this probe needs a CUDA device; there is nothing to measure on CPU")
    if not args.weights.exists():
        raise FileNotFoundError(
            f"no checkpoint at {args.weights}; run `make model` or `make train`"
        )

    # cuDNN autotuning on, so the comparison is against PyTorch at its best
    # rather than against PyTorch with an obvious flag left off.
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    model = YOLO(str(args.weights)).model.to(device).eval()

    width, height = args.imgsz
    report: dict = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "platform": platform.platform(),
        "sm_clock_mhz": int(torch.cuda.clock_rate()) if hasattr(torch.cuda, "clock_rate") else None,
        "iterations": args.iterations,
        "cudnn_benchmark": True,
        "batch_sweep": [],
        "resolution_sweep": [],
    }

    for batch in args.batches:
        tensor = torch.randn(batch, 3, height, width, device=device)
        entry = bench(model, tensor, torch, args.iterations, args.warmup)
        report["batch_sweep"].append(entry)
        print(
            f"batch {batch:<2} {width}x{height}   p50 {entry['p50_ms']:7.2f} ms   "
            f"per image {entry['per_image_p50_ms']:6.2f} ms",
            flush=True,
        )

    for scale in (1, 2):
        tensor = torch.randn(1, 3, height * scale, width * scale, device=device)
        entry = bench(model, tensor, torch, args.iterations, args.warmup)
        report["resolution_sweep"].append(entry)
        print(
            f"batch 1  {width*scale}x{height*scale}   p50 {entry['p50_ms']:7.2f} ms   "
            f"{entry['megapixels']:.2f} Mpix",
            flush=True,
        )

    base = report["batch_sweep"][0]["p50_ms"]
    largest = report["batch_sweep"][-1]
    biggest_frame = report["resolution_sweep"][-1]
    report["verdict"] = {
        "batch_time_ratio": round(largest["p50_ms"] / base, 3),
        "batch_size_ratio": largest["batch"],
        "per_image_speedup_at_max_batch": round(
            base / max(largest["per_image_p50_ms"], 1e-9), 3
        ),
        "resolution_time_ratio": round(biggest_frame["p50_ms"] / base, 3),
        "resolution_pixel_ratio": round(
            biggest_frame["megapixels"] / report["batch_sweep"][0]["megapixels"], 3
        ),
        # If eight times the work and four times the pixels both take the same
        # wall time, the wall time is not the work.
        "launch_bound": bool(
            largest["p50_ms"] / base < 1.5 and biggest_frame["p50_ms"] / base < 1.5
        ),
    }
    print("\nverdict:", json.dumps(report["verdict"], indent=2))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

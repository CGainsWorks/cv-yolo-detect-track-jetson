#!/usr/bin/env python
"""Build the TensorRT engines on this GPU, score them and time them.

    docker build -f docker/Dockerfile.tensorrt -t edgetrack:tensorrt .
    docker run --rm --gpus all --network none -v "$PWD:/work" edgetrack:tensorrt \\
        python3 scripts/measure_tensorrt.py

or ``make tensorrt-desktop``, which runs exactly that.

Everything this script measures goes through the repository's own TensorRT
path and nothing else: ``scripts/export_tensorrt.py`` builds the engines,
``scripts/evaluate.py --backend tensorrt`` scores them with the same decoder,
NMS and metric as every other row and ``scripts/benchmark_latency.py`` times
them with the same loop.  This file only runs those three in order, watches
the GPU and writes one compact summary to ``results/published/tensorrt.json``.

**What the numbers are.**  Desktop numbers, from whatever NVIDIA GPU the
container was given, which for the published file was an RTX 4060 under WSL2.
They are not Jetson numbers and the summary says so in its ``device`` field.
The accuracy column transfers to a board running the same TensorRT version,
because the arithmetic is the same; the latency column does not.

**Contention.**  The GPU on the machine that produced the published file is
shared with the desktop.  The script samples ``nvidia-smi`` for a few seconds
before the first build and after the last benchmark and records what it saw
and refuses to call the run uncontended if utilisation was above a threshold
or another compute process was listed.  It cannot see processes on a Windows
host from inside a WSL2 container, so the summary also carries what the host
showed, passed in with ``--host-note`` by whoever ran it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
from edgetrack.paths import write_text_lf

PRECISIONS = ("fp32", "fp16", "int8")
IDLE_UTILISATION_PERCENT = 10
# A one-minute load average above this fraction of the logical cores counts as
# a busy CPU.  The end-to-end column runs on the CPU and so does the host side
# of every inference call and a second run under a load average of 8 on 16
# cores came out several times slower than a run under 2.5.
BUSY_CPU_FRACTION = 0.25


def _smi(*query: str) -> list[list[str]]:
    output = subprocess.run(
        ["nvidia-smi", *query, "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout
    return [[cell.strip() for cell in line.split(",")] for line in output.splitlines() if line]


def gpu_sample() -> dict:
    name, driver, total, used, util = _smi(
        "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu"
    )[0]
    apps = _smi("--query-compute-apps=pid,process_name,used_memory")
    return {
        "name": name,
        "driver": driver,
        "memory_total_mib": int(total),
        "memory_used_mib": int(used),
        "utilisation_percent": int(util),
        "compute_processes": len(apps),
    }


def load_average() -> float:
    """One-minute load average of the kernel this container shares.

    Under WSL2 every container runs on one Linux kernel, so this counts other
    containers' CPU work too.  The end-to-end column includes letterboxing,
    decode, NMS and the tracker on the CPU, so a busy CPU shows up there.
    """
    try:
        return float(Path("/proc/loadavg").read_text().split()[0])
    except OSError:  # pragma: no cover - not Linux
        return -1.0


def watch(seconds: int) -> dict:
    samples = []
    loads = []
    for _ in range(seconds):
        samples.append(gpu_sample())
        loads.append(load_average())
        time.sleep(1.0)
    return {
        "samples": len(samples),
        "utilisation_percent_max": max(s["utilisation_percent"] for s in samples),
        "memory_used_mib_max": max(s["memory_used_mib"] for s in samples),
        "compute_processes_max": max(s["compute_processes"] for s in samples),
        "cpu_load_1min_max": max(loads),
    }


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--models", default="models", type=Path)
    parser.add_argument("--name", default="yolov8n_edgetrack")
    parser.add_argument("--data-root", default="data/processed", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--image", default=os.environ.get("TRT_IMAGE", "unknown"),
                        help="the container image this ran in, for the record")
    parser.add_argument("--host-note", default="",
                        help="what nvidia-smi on the host showed during the run")
    parser.add_argument("--reuse-engines", action="store_true",
                        help="skip the builds if the engines already exist")
    parser.add_argument("--out", default="results/published/tensorrt.json", type=Path)
    args = parser.parse_args(argv)

    import tensorrt

    python = sys.executable
    onnx = args.models / f"{args.name}_fp32.onnx"
    engines = {p: args.models / f"{args.name}_{p}.engine" for p in PRECISIONS}
    build_report = Path("results/tensorrt_build.json")
    detection_out = Path("results/tensorrt_detection.json")
    latency_out = Path("results/tensorrt_latency.json")

    before = watch(5)

    if not (args.reuse_engines and all(e.exists() for e in engines.values())):
        build_report.unlink(missing_ok=True)
        # A stale calibration cache would let the INT8 build skip calibration
        # and then the published INT8 row would describe an older calibration.
        Path("models/calib.cache").unlink(missing_ok=True)
        for precision in PRECISIONS:
            run([python, "scripts/export_tensorrt.py", "--onnx", str(onnx),
                 "--precision", precision, "--data-root", str(args.data_root),
                 "--report", str(build_report)])

    run([python, "scripts/evaluate.py", "--backend", "tensorrt", "--split", args.split,
         "--data-root", str(args.data_root), "--out", str(detection_out), "--quiet",
         "--models", *[str(engines[p]) for p in PRECISIONS]])
    run([python, "scripts/benchmark_latency.py", "--variants", "configs/benchmark.yaml",
         "--data-root", str(args.data_root), "--iterations", str(args.iterations),
         "--out", str(latency_out), "--quiet",
         "--only", *[f"tensorrt-{p}" for p in PRECISIONS]])

    # nvidia-smi reports utilisation averaged over its last sample period, so
    # the first reading straight after the benchmark is this script's own
    # tail.  Waiting a few seconds keeps it out of the contention check.
    time.sleep(5.0)
    after = watch(5)

    builds = json.loads(build_report.read_text(encoding="utf-8"))
    detection = json.loads(detection_out.read_text(encoding="utf-8"))
    latency = json.loads(latency_out.read_text(encoding="utf-8"))
    published = json.loads(Path("results/published/detection.json").read_text(encoding="utf-8"))
    baseline = next(
        e for e in published["variants"].values() if e["detector"]["precision"] == "fp32"
    )["metrics"]

    accuracy = {v["detector"]["precision"]: v["metrics"] for v in detection["variants"].values()}
    timing = {r["precision"]: r for r in latency["results"]}

    rows = []
    for precision in PRECISIONS:
        metrics, bench, build = accuracy[precision], timing[precision], builds[precision]
        rows.append({
            "variant": f"tensorrt-{precision}",
            "precision": precision,
            "engine_bytes": build["bytes"],
            "build_seconds": build["build_seconds"],
            "calibration_images": build["calibration_images"],
            "map50": metrics["map50"],
            "map50_95": metrics["map50_95"],
            "precision_metric": metrics["precision"],
            "recall": metrics["recall"],
            "map50_95_delta_vs_onnxruntime_fp32": round(
                metrics["map50_95"] - baseline["map50_95"], 5
            ),
            "inference_p50_ms": bench["stages_ms"]["inference"]["p50"],
            "inference_p95_ms": bench["stages_ms"]["inference"]["p95"],
            "detect_total_p50_ms": bench["stages_ms"]["detect_total"]["p50"],
            "end_to_end_p50_ms": bench["end_to_end_ms"]["p50"],
            "fps_end_to_end": bench["fps_end_to_end"],
        })

    contended = any(
        w["utilisation_percent_max"] > IDLE_UTILISATION_PERCENT or w["compute_processes_max"] > 0
        for w in (before, after)
    )
    cpu_busy = any(
        w["cpu_load_1min_max"] > (os.cpu_count() or 1) * BUSY_CPU_FRACTION for w in (before, after)
    )
    gpu = gpu_sample()
    import pycuda

    summary = {
        "device": "desktop RTX 4060, not a Jetson" if "4060" in gpu["name"]
        else f"desktop {gpu['name']}, not a Jetson",
        "host": {
            "gpu": gpu["name"],
            "gpu_memory_mib": gpu["memory_total_mib"],
            "driver": gpu["driver"],
            "cuda_runtime": os.environ.get("CUDA_VERSION", "unknown"),
            "tensorrt": tensorrt.__version__,
            "pycuda": pycuda.VERSION_TEXT,
            "container_image": args.image,
            "under": "WSL2 Docker" if "microsoft" in os.uname().release.lower() else "Linux",
            "cpu_count": os.cpu_count(),
        },
        "gpu_contention": {
            "idle_threshold_percent": IDLE_UTILISATION_PERCENT,
            "before": before,
            "after": after,
            "contended": contended,
            "busy_cpu_threshold_load": (os.cpu_count() or 1) * BUSY_CPU_FRACTION,
            "cpu_busy": cpu_busy,
            "host_note": args.host_note,
        },
        "split": detection["split"],
        "images": detection["images"],
        "onnxruntime_fp32_map50_95": baseline["map50_95"],
        "iterations": latency["iterations"],
        "warmup": latency["warmup"],
        "tracker": latency["results"][0]["tracker"],
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_text_lf(args.out, json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

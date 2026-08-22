#!/usr/bin/env python
"""Build a TensorRT engine from the FP32 ONNX export, at a chosen precision.

    python scripts/export_tensorrt.py --precision fp16
    python scripts/export_tensorrt.py --precision int8 --calib-images 512

**This script has never been executed.**  There is no Jetson and no TensorRT
installation on the machine that produced the published results, so nothing in
``results/published/`` or ``benchmarks/results.csv`` came from here.  It is
committed because the interface it targets is fixed by
``edgetrack.detect.trt_backend`` and because writing it now is what makes the
hardware step a matter of running a command rather than a matter of design.
``docs/jetson.md`` carries the full procedure and the honest gap list.

Two things this script does that a minimal one would not, both learned from the
shape of the problem rather than from running it:

*The INT8 calibrator streams the same preprocessed images the deployed model
will see.*  TensorRT's entropy calibrator picks per-tensor scales from the
activation histograms it observes, so calibrating on anything other than real
preprocessed frames produces an engine whose dynamic range is wrong.

*The calibration cache is written and reused.*  Calibration is the slow part of
an INT8 build, several minutes on an Orin Nano, and it only depends on the
network and the calibration set.  Caching it means a rebuild after a TensorRT
upgrade costs seconds.

Engines are not portable.  A plan file is tied to the GPU architecture, the
driver and the TensorRT version that built it, which is why nothing here is
ever committed and why ``.gitignore`` excludes ``*.engine`` and ``*.plan``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import yaml

import _bootstrap  # noqa: F401
from edgetrack.detect.base import letterbox
from edgetrack.paths import serialise_path, write_text_lf

_MISSING = (
    "tensorrt is not importable.  On a Jetson it comes with Jetpack and is "
    "already in the system Python: run this with /usr/bin/python3, not from a "
    "virtualenv built with --no-system-site-packages.  On an x86 host, install "
    "the TensorRT wheel matching your CUDA version."
)


def _calibrator_class(trt, images, imgsz, cache_path, batch_size):
    """Build the calibrator class here, where ``trt`` is known to exist.

    Subclassing ``trt.IInt8EntropyCalibrator2`` at module scope would make this
    file unimportable without TensorRT, and ``tests/test_scripts.py`` imports it
    to check the argument parser.
    """
    import pycuda.driver as cuda

    class ImageCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self) -> None:
            super().__init__()
            self.images = list(images)
            self.batch_size = int(batch_size)
            self.imgsz = imgsz
            self.cache_path = Path(cache_path)
            self.index = 0
            width, height = imgsz
            self.device_input = cuda.mem_alloc(
                self.batch_size * 3 * height * width * np.dtype(np.float32).itemsize
            )

        def get_batch_size(self) -> int:
            return self.batch_size

        def get_batch(self, names, p_str=None):  # noqa: ARG002 - TensorRT signature
            import cv2

            if self.index + self.batch_size > len(self.images):
                return None
            batch = []
            for path in self.images[self.index : self.index + self.batch_size]:
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError(f"could not decode calibration image {path}")
                padded, _, _ = letterbox(image, self.imgsz)
                tensor = padded[:, :, ::-1].astype(np.float32) / 255.0
                batch.append(tensor.transpose(2, 0, 1))
            self.index += self.batch_size
            array = np.ascontiguousarray(np.stack(batch), dtype=np.float32)
            cuda.memcpy_htod(self.device_input, array)
            return [int(self.device_input)]

        def read_calibration_cache(self):
            if self.cache_path.exists():
                return self.cache_path.read_bytes()
            return None

        def write_calibration_cache(self, cache) -> None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_bytes(cache)

    return ImageCalibrator


def build(
    onnx_path: Path,
    engine_path: Path,
    precision: str,
    workspace_gb: float,
    calibration_images: list[Path],
    imgsz: tuple[int, int],
    cache_path: Path,
    calib_batch: int,
) -> dict:
    try:
        import tensorrt as trt
    except ImportError as error:  # pragma: no cover - no TensorRT anywhere here
        raise ImportError(_MISSING) from error

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)

    if not parser.parse(onnx_path.read_bytes()):
        errors = [parser.get_error(i) for i in range(parser.num_errors)]
        raise RuntimeError(
            f"TensorRT could not parse {onnx_path}:\n"
            + "\n".join(str(e) for e in errors)
        )

    config = builder.create_builder_config()
    workspace_bytes = int(workspace_gb * 1024**3)
    # TensorRT 8.4 replaced max_workspace_size with the memory pool API and
    # 10.x removed the old attribute; support both because Jetpack 5 and 6 sit
    # on either side of the change.
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:  # pragma: no cover - TensorRT 8.2 and older
        config.max_workspace_size = workspace_bytes

    calibrator = None
    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("this device has no fast FP16 path; build FP32 instead")
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        if not builder.platform_has_fast_int8:
            raise RuntimeError("this device has no fast INT8 path; build FP16 instead")
        config.set_flag(trt.BuilderFlag.INT8)
        # FP16 stays on alongside INT8 so that layers TensorRT refuses to
        # quantise fall back to half rather than to full precision.
        config.set_flag(trt.BuilderFlag.FP16)
        calibrator_class = _calibrator_class(
            trt, calibration_images, imgsz, cache_path, calib_batch
        )
        calibrator = calibrator_class()
        config.int8_calibrator = calibrator

    started = time.perf_counter()
    if hasattr(builder, "build_serialized_network"):
        serialised = builder.build_serialized_network(network, config)
        if serialised is None:
            raise RuntimeError("TensorRT returned no engine; check the log above")
        payload = bytes(serialised)
    else:  # pragma: no cover - TensorRT 8.0 and older
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError("TensorRT returned no engine; check the log above")
        payload = engine.serialize()
    elapsed = time.perf_counter() - started

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(payload)
    return {
        "engine": serialise_path(engine_path),
        "bytes": len(payload),
        "precision": precision,
        "build_seconds": round(elapsed, 1),
        "tensorrt": trt.__version__,
        "calibration_images": len(calibration_images) if precision == "int8" else 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--onnx", default="models/yolov8n_edgetrack_fp32.onnx", type=Path)
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "int8"])
    parser.add_argument("--config", default="configs/export.yaml", type=Path)
    parser.add_argument("--data-root", default="data/processed", type=Path)
    parser.add_argument("--calib-images", type=int, default=None)
    parser.add_argument("--calib-batch", type=int, default=8)
    parser.add_argument("--out", default=None, type=Path)
    parser.add_argument("--report", default="results/tensorrt_build.json", type=Path)
    args = parser.parse_args(argv)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    width, height = config["onnx"]["imgsz"]
    trt_config = config["tensorrt"]

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        raise FileNotFoundError(f"no ONNX at {onnx_path}; run `make export` first")

    engine_path = Path(args.out) if args.out else onnx_path.with_name(
        onnx_path.name.replace("_fp32.onnx", f"_{args.precision}.engine")
    )

    images: list[Path] = []
    if args.precision == "int8":
        split = config["quantisation"]["int8"].get("calibration_split", "train")
        pool = sorted((Path(args.data_root) / "images" / split).glob("*.jpg"))
        if not pool:
            raise FileNotFoundError(
                f"INT8 needs calibration images and none were found under "
                f"{Path(args.data_root)/'images'/split}"
            )
        wanted = args.calib_images or int(config["quantisation"]["int8"]["calibration_images"])
        rng = random.Random(int(config["quantisation"]["int8"].get("calibration_seed", 7)))
        images = rng.sample(pool, min(wanted, len(pool)))

    record = build(
        onnx_path=onnx_path,
        engine_path=engine_path,
        precision=args.precision,
        workspace_gb=float(trt_config.get("workspace_gb", 2)),
        calibration_images=images,
        imgsz=(int(width), int(height)),
        cache_path=Path(trt_config.get("int8_calibration_cache", "models/calib.cache")),
        calib_batch=int(args.calib_batch),
    )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if args.report.exists():
        existing = json.loads(args.report.read_text(encoding="utf-8"))
    existing[args.precision] = record
    write_text_lf(args.report, json.dumps(existing, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

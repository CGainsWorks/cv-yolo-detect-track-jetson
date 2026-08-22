#!/usr/bin/env python
"""Export the trained checkpoint to ONNX at the inference resolution.

    python scripts/export_onnx.py                      # FP32
    python scripts/export_onnx.py --precision fp16     # FP16, needs a GPU

Produces ``models/<name>_fp32.onnx`` and ``models/<name>_fp16.onnx``.  INT8
comes from ``scripts/quantize_onnx.py``, which derives it from the FP32 graph.

Why FP16 is exported rather than converted.  The obvious route is
``onnxconverter_common.float16.convert_float_to_float16`` on the FP32 file, and
it is what this script did first.  On this graph it emits an invalid model: the
converter casts the tensors around ``/model.10/Resize`` but leaves the declared
type of that node's output at float, and ONNX Runtime refuses to load the
result with a type error.  Blocking ``Resize`` from conversion and keeping the
IO types in float both fail the same way.  Exporting from a half-precision
torch module instead produces a graph that loads and runs, so that is what
happens here, and the cost is that FP16 is a sibling export rather than a
derivative of the FP32 file.  INT8 is still derived, so the INT8 comparison
retains the shared-topology guarantee that FP16 loses.

Opset 13, not 12.  Per-channel quantisation needs the ``axis`` attribute on
``QuantizeLinear`` and ``DequantizeLinear``, which arrived in opset 13; at
opset 12 ``quantize_static(per_channel=True)`` produces a model ONNX Runtime
rejects as invalid.  TensorRT 8.5, which is what Jetpack 5 ships, parses opset
13 without complaint.

Every export is loaded back and run before this script exits.  An export that
produces an unloadable file is the failure mode that costs the most time to
find later, because nothing else in the pipeline touches the graph until the
benchmark does.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

import _bootstrap  # noqa: F401
from edgetrack.paths import serialise_path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, imgsz: tuple[int, int]) -> dict:
    """Load the export and run one blank frame through it."""
    import onnx
    import onnxruntime as ort

    graph = onnx.load(str(path))
    opsets = {(entry.domain or "ai.onnx"): int(entry.version) for entry in graph.opset_import}

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    spec = session.get_inputs()[0]
    dtype = np.float16 if spec.type == "tensor(float16)" else np.float32
    width, height = imgsz
    output = session.run(None, {spec.name: np.zeros((1, 3, height, width), dtype=dtype)})[0]
    if output.ndim != 3:
        raise RuntimeError(f"{path.name}: expected a 3D head output, got shape {output.shape}")
    channels = min(output.shape[1], output.shape[2])
    return {
        "opset_domains": opsets,
        "input_dtype": spec.type,
        "output_shape": list(output.shape),
        "channels": int(channels),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--weights", default="models/yolov8n_edgetrack.pt", type=Path)
    parser.add_argument("--config", default="configs/export.yaml", type=Path)
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    parser.add_argument("--device", default=0)
    parser.add_argument("--out", default=None, type=Path)
    args = parser.parse_args(argv)

    import torch
    from ultralytics import YOLO

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))["onnx"]
    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"no checkpoint at {weights}; run `make train` first")
    if args.precision == "fp16" and not torch.cuda.is_available():
        raise RuntimeError(
            "half precision export needs a CUDA device: Ultralytics converts the "
            "module with .half() and the tracing pass runs on the GPU"
        )

    width, height = config["imgsz"]
    model = YOLO(str(weights))
    exported = model.export(
        format="onnx",
        opset=int(config["opset"]),
        simplify=bool(config.get("simplify", True)),
        dynamic=bool(config.get("dynamic", False)),
        half=(args.precision == "fp16"),
        device=args.device if args.precision == "fp16" else "cpu",
        imgsz=[int(height), int(width)],  # Ultralytics takes height, width
    )

    destination = (
        Path(args.out) if args.out
        else weights.with_name(f"{weights.stem}_{args.precision}.onnx")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    exported_path = Path(exported)
    if exported_path.resolve() != destination.resolve():
        shutil.move(str(exported_path), destination)

    checks = verify(destination, (int(width), int(height)))
    expected_dtype = "tensor(float16)" if args.precision == "fp16" else "tensor(float)"
    if checks["input_dtype"] != expected_dtype:
        raise RuntimeError(
            f"asked for {args.precision} and the exported graph takes "
            f"{checks['input_dtype']}"
        )

    record = {
        "source_weights": serialise_path(weights),
        "onnx": serialise_path(destination),
        "sha256": sha256(destination),
        "bytes": destination.stat().st_size,
        "opset": int(config["opset"]),
        "imgsz_wh": [int(width), int(height)],
        "precision": args.precision,
        **checks,
    }
    destination.with_suffix(".json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

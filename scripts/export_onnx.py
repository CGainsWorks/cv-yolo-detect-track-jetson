#!/usr/bin/env python
"""Export the trained checkpoint to ONNX at the inference resolution.

    python scripts/export_onnx.py                      # FP32
    python scripts/export_onnx.py --precision fp16     # FP16, needs a GPU

Produces ``models/<name>_fp32.onnx`` and ``models/<name>_fp16.onnx``.  INT8
comes from ``scripts/quantize_onnx.py``, which derives it from the FP32 graph.

Why FP16 is exported rather than converted.  The obvious route is
``onnxconverter_common.float16.convert_float_to_float16`` on the FP32 file.  On
the previous detector's graph it emitted a model ONNX Runtime refused to load,
a type error around a Resize node, so FP16 has been a separate export from a
half-precision module ever since and that is still what happens here.  The
cost is that FP16 is a sibling export rather than a derivative of the FP32
file.  INT8 is still derived, so the INT8 comparison retains the
shared-topology guarantee that FP16 loses.

What is exported.  ``edgetrack.detect.yolox_net.DeployYOLOX``: YOLOX-Tiny with
the RGB 0..1 input folded into the stem convolution and the grid decode and
``obj * cls`` scoring done in the graph, so the output is ``(1, 4 +
num_classes, 5040)`` at 640x384, the layout
``edgetrack.detect.base.decode_head`` reads.  The TorchScript exporter is used
(``dynamo=False``) because it names nodes after the module path,
``/model/head/cls_preds.0/Conv`` and ``scripts/quantize_onnx.py`` finds the
detection head by those names.  The graph is then simplified with onnxslim.

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
import sys
from pathlib import Path

import numpy as np
import yaml

import _bootstrap  # noqa: F401
from edgetrack.paths import serialise_path, write_text_lf


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
    parser.add_argument("--weights", default="models/yolox_tiny_edgetrack.pt", type=Path)
    parser.add_argument("--config", default="configs/export.yaml", type=Path)
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    parser.add_argument("--device", default=0)
    parser.add_argument("--out", default=None, type=Path)
    args = parser.parse_args(argv)

    import onnxslim
    import torch

    from edgetrack.detect.yolox_net import deploy_module

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))["onnx"]
    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"no checkpoint at {weights}; run `make train` first")
    half = args.precision == "fp16"
    if half and not torch.cuda.is_available():
        raise RuntimeError(
            "half precision export needs a CUDA device: the module is converted "
            "with .half() and traced on the GPU, because several CPU kernels have "
            "no half implementation"
        )

    width, height = config["imgsz"]
    module, meta = deploy_module(weights, (int(width), int(height)))
    device = torch.device(f"cuda:{args.device}" if half else "cpu")
    module = module.to(device)
    dummy = torch.zeros((1, 3, int(height), int(width)), device=device)
    if half:
        module, dummy = module.half(), dummy.half()

    destination = (
        Path(args.out) if args.out
        else weights.with_name(f"{weights.stem}_{args.precision}.onnx")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            module,
            (dummy,),
            str(destination),
            dynamo=False,
            opset_version=int(config["opset"]),
            input_names=["images"],
            output_names=["output0"],
            do_constant_folding=True,
            dynamic_axes=None,
        )
    if bool(config.get("simplify", True)):
        import onnx

        slim = onnxslim.slim(onnx.load(str(destination)))
        onnx.save(slim, str(destination))

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
        "architecture": meta["architecture"],
        "class_names": meta["class_names"],
        **checks,
    }
    write_text_lf(destination.with_suffix(".json"), json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

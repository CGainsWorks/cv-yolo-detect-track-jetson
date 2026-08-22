#!/usr/bin/env python
"""Produce the INT8 variants from the FP32 ONNX export.

    python scripts/quantize_onnx.py

INT8 is derived from the FP32 graph rather than exported separately, so the
only difference between the two files is numeric precision.  That is what makes
the INT8 rows of the headline table a measurement of quantisation cost rather
than of export variation.  FP16 could not be produced this way and is a sibling
export instead; ``scripts/export_onnx.py`` explains why.

Static QDQ quantisation with a real calibration pass over training images.
Dynamic quantisation would need no calibration set and would also be useless
here, because it leaves activations in float and a convolution-dominated
network gets almost nothing from weight-only quantisation.

Three variants, because the difference between them is the interesting result
----------------------------------------------------------------------------

``int8``
    Convolutions only, with the detection head's output convolutions left in
    float.  This is the one to deploy.

``int8_allconv``
    Every convolution including the head's, which is the obvious thing to write
    and costs real accuracy.

``int8_allops``
    Everything ONNX Runtime is willing to quantise, which is the default when
    ``op_types_to_quantize`` is left unset.  It produces a model that loads,
    runs, and detects nothing at all.

The third is kept rather than deleted because it was the first attempt, and a
repository that ships only the version that worked teaches nobody anything.
All three are scored by ``scripts/evaluate.py`` and the numbers are in
``docs/results.md``.

Why the default fails.  YOLOv8's head ends in elementwise arithmetic: a Sigmoid
over the class logits, then a distribution-focal-loss softmax, a convolution
over the bins and a chain of Slice, Sub, Add, Div and Mul that turns the
distribution into box coordinates.  Quantising that chain to uint8 puts a
coarse step on tensors whose useful range spans several orders of magnitude and
the class scores come out identically zero.  The boxes survive, which is what
makes the failure hard to spot from tensor statistics alone: the geometry looks
right and every confidence is zero.

What this is not.  TensorRT's INT8 calibration is a different implementation
with a different calibrator and different layer fusion, so the INT8 mAP
measured here bounds but does not predict what the same model would score on a
Jetson.  ``docs/jetson.md`` states that plainly and
``scripts/export_tensorrt.py`` is the path that would answer it.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import yaml

import _bootstrap  # noqa: F401
from edgetrack.detect.base import letterbox
from edgetrack.paths import serialise_path, write_text_lf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_onnx import sha256  # noqa: E402


class ImageCalibrationReader:
    """Feeds real, letterboxed training frames to the quantiser.

    Calibrating on random noise is the classic way to get an INT8 model whose
    activation ranges are wrong by an order of magnitude.  The images here go
    through exactly the preprocessing :class:`edgetrack.detect.base.Detector`
    applies at inference, because the calibrator is measuring the distribution
    of the tensors the deployed model will actually see.
    """

    def __init__(self, images: list[Path], input_name: str, imgsz: tuple[int, int]) -> None:
        import cv2

        self._cv2 = cv2
        self.images = list(images)
        self.input_name = input_name
        self.imgsz = imgsz
        self._index = 0

    def get_next(self) -> dict | None:
        if self._index >= len(self.images):
            return None
        path = self.images[self._index]
        self._index += 1
        image = self._cv2.imread(str(path))
        if image is None:
            raise ValueError(f"could not decode calibration image {path}")
        padded, _, _ = letterbox(image, self.imgsz)
        tensor = padded[:, :, ::-1].astype(np.float32) / 255.0
        tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)[None])
        return {self.input_name: tensor}

    def rewind(self) -> None:
        self._index = 0


def head_output_nodes(model_path: Path) -> list[str]:
    """The detection head's final convolutions, by name.

    Found by structure rather than hard-coded: on each of the three strides the
    box branch ``cv2.S`` and the class branch ``cv3.S`` end in a convolution,
    and the distribution-focal-loss convolution sits after them.  A hard-coded
    list would silently stop matching the day the architecture changes, and
    silently quantising the head is the failure this function exists to
    prevent.
    """
    graph = onnx.load(str(model_path)).graph
    names = [node.name for node in graph.node if node.op_type == "Conv"]
    groups: dict[tuple[str, str], list[str]] = {}
    for name in names:
        parts = [p for p in name.split("/") if p.startswith(("cv2.", "cv3."))]
        if not parts:
            continue
        branch = parts[0]
        groups.setdefault((branch[:3], branch.split(".")[1]), []).append(name)
    outputs = [sorted(group)[-1] for group in groups.values()]
    outputs.extend(name for name in names if "/dfl/" in name)
    return sorted(set(outputs))


def to_int8(
    prepared: Path,
    destination: Path,
    images: list[Path],
    imgsz: tuple[int, int],
    settings: dict,
    input_name: str,
    op_types: list[str] | None,
    exclude: list[str],
) -> dict:
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

    weight_type = (
        QuantType.QInt8 if settings.get("weight_type", "int8") == "int8" else QuantType.QUInt8
    )
    activation_type = (
        QuantType.QUInt8 if settings.get("activation_type", "uint8") == "uint8" else QuantType.QInt8
    )
    extra = {"op_types_to_quantize": op_types} if op_types else {}
    quantize_static(
        model_input=str(prepared),
        model_output=str(destination),
        calibration_data_reader=ImageCalibrationReader(images, input_name, imgsz),
        quant_format=QuantFormat.QDQ,
        per_channel=bool(settings.get("per_channel", True)),
        activation_type=activation_type,
        weight_type=weight_type,
        calibrate_method=CalibrationMethod.MinMax,
        nodes_to_exclude=list(exclude),
        # Weights are stored as INT8 initialisers with a DequantizeLinear after
        # them, rather than kept in float with a Quantize/Dequantize pair around
        # them.  Both forms are legal and TensorRT accepts either; this one is a
        # quarter of the file size, which is the point of shipping INT8 to an
        # edge device with 8 GB shared between the GPU and everything else.
        extra_options={"AddQDQPairToWeight": False},
        **extra,
    )
    return {
        "op_types_to_quantize": op_types or "all",
        "nodes_excluded": list(exclude),
        "per_channel": bool(settings.get("per_channel", True)),
        "quant_format": "QDQ",
        "calibrate_method": "MinMax",
    }


def verify(path: Path, imgsz: tuple[int, int]) -> dict:
    """Load it and run it.

    A quantised graph ONNX Runtime rejects is the expected failure when the
    opset is too low for per-channel QDQ, and it does not surface until
    something tries to use the file.
    """
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    spec = session.get_inputs()[0]
    width, height = imgsz
    output = session.run(None, {spec.name: np.zeros((1, 3, height, width), np.float32)})[0]
    nodes = onnx.load(str(path)).graph.node
    qdq = sum(1 for node in nodes if node.op_type in ("QuantizeLinear", "DequantizeLinear"))
    if qdq == 0:
        raise RuntimeError(f"{path.name} has no QDQ nodes; it is not quantised")
    return {"qdq_nodes": qdq, "output_shape": list(output.shape)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default="models/yolov8n_edgetrack_fp32.onnx", type=Path)
    parser.add_argument("--config", default="configs/export.yaml", type=Path)
    parser.add_argument("--data-root", default="data/processed", type=Path)
    parser.add_argument("--variants", nargs="*", default=None,
                        help="subset of int8, int8_allconv, int8_allops")
    args = parser.parse_args(argv)

    from onnxruntime.quantization.shape_inference import quant_pre_process

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    quant = config["quantisation"]["int8"]
    width, height = config["onnx"]["imgsz"]

    source = Path(args.model)
    if not source.exists():
        raise FileNotFoundError(f"no FP32 ONNX at {source}; run `make export` first")

    split = quant.get("calibration_split", "train")
    pool = sorted((Path(args.data_root) / "images" / split).glob("*.jpg"))
    if not pool:
        raise FileNotFoundError(
            f"no calibration images under {Path(args.data_root)/'images'/split}; "
            "run `python scripts/make_dataset.py` first"
        )
    rng = random.Random(int(quant.get("calibration_seed", 7)))
    images = rng.sample(pool, min(int(quant.get("calibration_images", 256)), len(pool)))

    # Pre-processing is not optional.  Without symbolic shape inference and
    # constant folding first, quantize_static leaves a scattering of nodes in
    # float and the resulting model is both slower and less accurate.
    prepared = source.with_name(source.stem + "_prepared.onnx")
    quant_pre_process(str(source), str(prepared), skip_symbolic_shape=False)
    input_name = onnx.load(str(source)).graph.input[0].name
    head = head_output_nodes(source)
    if not head:
        raise RuntimeError(
            "could not find the detection head's output convolutions by name, so "
            "the exclusion list would be empty and the head would be quantised"
        )

    plans = {
        "int8": {"op_types": ["Conv"], "exclude": head},
        "int8_allconv": {"op_types": ["Conv"], "exclude": []},
        "int8_allops": {"op_types": None, "exclude": []},
    }
    wanted = args.variants or list(plans)

    records: dict[str, dict] = {}
    for name in wanted:
        plan = plans[name]
        destination = source.with_name(source.name.replace("_fp32", f"_{name}"))
        details = to_int8(
            prepared, destination, images, (width, height), quant, input_name,
            plan["op_types"], plan["exclude"],
        )
        records[name] = {
            "path": serialise_path(destination),
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
            "fp32_bytes": source.stat().st_size,
            "size_ratio": round(destination.stat().st_size / source.stat().st_size, 4),
            "calibration_split": split,
            "calibration_pool": len(pool),
            **details,
            **verify(destination, (width, height)),
        }
        print(
            f"built {destination.name}: {records[name]['bytes']} bytes, "
            f"{records[name]['qdq_nodes']} QDQ nodes",
            flush=True,
        )

    prepared.unlink(missing_ok=True)
    out = source.with_name("quantisation.json")
    existing = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    existing.update(records)
    existing["head_output_nodes"] = head
    write_text_lf(out, json.dumps(existing, indent=2) + "\n")
    print(json.dumps(records, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""ONNX Runtime backend.  The portable path and the one CI exercises.

Handles three model variants from the same class, because the graph tells us
which it is:

* FP32, the reference export,
* FP16, converted with ``onnxconverter_common``, whose input tensor is
  ``float16`` so the NCHW tensor has to be cast before the call,
* INT8, static QDQ quantisation, whose input stays ``float32`` because the
  quantise node is inside the graph.

Reading the declared input dtype rather than being told which variant this is
means ``--precision`` is a label on the results and never a switch that can be
set wrongly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .base import BackendInfo, Detector, DetectorConfig

__all__ = ["OnnxDetector"]

_DTYPES = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(uint8)": np.uint8,
}

# Operator names that only appear in a quantised graph.  A protobuf stores
# op_type as a plain string, so scanning the bytes finds them without parsing.
_QUANT_MARKERS = (b"QuantizeLinear", b"DequantizeLinear")


def _has_quantisation_nodes(path: Path, chunk: int = 1 << 20) -> bool:
    """Whether the graph contains QDQ operators, without importing ``onnx``.

    The obvious implementation is ``onnx.load(path).graph.node``, and that is
    what this did first.  It was wrong in a way that only showed up in CI: the
    ``onnx`` package is in ``requirements-train.txt`` and deliberately not in
    the runtime, so on any machine that had only installed the runtime the
    import failed, the check fell back to the input dtype and an INT8 model
    reported itself as FP32.  A results table labelled by a value that depends
    on which optional packages happen to be installed is worse than no label.

    Scanning the serialised protobuf for the operator names is crude and it is
    dependency-free, deterministic and decisive: a float graph contains neither
    string anywhere.  The overlap between chunks is the length of the longest
    marker minus one, so a name split across a boundary is still found.
    """
    overlap = max(len(marker) for marker in _QUANT_MARKERS) - 1
    tail = b""
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                return False
            window = tail + block
            if any(marker in window for marker in _QUANT_MARKERS):
                return True
            tail = window[-overlap:]


class OnnxDetector(Detector):
    def __init__(
        self,
        model_path: str | Path,
        config: DetectorConfig | None = None,
        providers: Sequence[str] | None = None,
        intra_op_threads: int | None = None,
    ) -> None:
        super().__init__(config or DetectorConfig())
        import onnxruntime as ort

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"no ONNX model at {self.model_path}.  Run `make export` to produce "
                "one, or `make model` to fetch the pretrained checkpoint."
            )

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_op_threads:
            options.intra_op_num_threads = int(intra_op_threads)

        available = ort.get_available_providers()
        wanted = list(providers) if providers else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        chosen = [p for p in wanted if p in available]
        if not chosen:
            raise RuntimeError(
                f"none of the requested execution providers {wanted} are available; "
                f"onnxruntime offers {available}"
            )
        self.session = ort.InferenceSession(str(self.model_path), options, providers=chosen)

        spec = self.session.get_inputs()[0]
        self._input_name = spec.name
        self._input_dtype = _DTYPES.get(spec.type, np.float32)
        self._output_name = self.session.get_outputs()[0].name
        self._declared_shape = tuple(spec.shape)

        # A fixed-shape export must match the configured imgsz or every box
        # comes out scaled wrongly, which is a silent failure rather than a
        # crash.  Check it once here.
        height, width = self._declared_shape[2], self._declared_shape[3]
        if isinstance(height, int) and isinstance(width, int):
            want_w, want_h = self.config.imgsz
            if (width, height) != (want_w, want_h):
                raise ValueError(
                    f"{self.model_path.name} expects {width}x{height} input but the "
                    f"detector is configured for {want_w}x{want_h}"
                )

        if self._input_dtype == np.float16:
            self.precision = "fp16"
        elif _has_quantisation_nodes(self.model_path):
            self.precision = "int8"
        else:
            self.precision = "fp32"

    def _forward(self, tensor: np.ndarray) -> np.ndarray:
        if tensor.dtype != self._input_dtype:
            tensor = tensor.astype(self._input_dtype)
        return self.session.run([self._output_name], {self._input_name: tensor})[0]

    @property
    def info(self) -> BackendInfo:
        return BackendInfo(
            backend="onnxruntime",
            precision=self.precision,
            model_path=str(self.model_path),
            providers=tuple(self.session.get_providers()),
            extra={"input_dtype": str(np.dtype(self._input_dtype))},
        )

    def close(self) -> None:
        self.session = None  # type: ignore[assignment]

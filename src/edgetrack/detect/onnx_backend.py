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

        quantised = any(
            node.op_type in ("QuantizeLinear", "DequantizeLinear")
            for node in self._graph_ops()
        )
        if self._input_dtype == np.float16:
            self.precision = "fp16"
        elif quantised:
            self.precision = "int8"
        else:
            self.precision = "fp32"

    def _graph_ops(self) -> list:
        """Node list, if onnx is importable.  It is not in the runtime image,
        so absence downgrades the precision label to a guess from the dtype
        rather than failing."""
        try:
            import onnx
        except ImportError:  # pragma: no cover - runtime image path
            return []
        try:
            return list(onnx.load(str(self.model_path)).graph.node)
        except Exception:  # pragma: no cover - corrupt file surfaces at session build
            return []

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

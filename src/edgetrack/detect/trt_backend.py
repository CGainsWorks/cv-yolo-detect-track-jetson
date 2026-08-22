"""TensorRT backend for the Jetson.

Status, stated plainly because the rest of this repository asks you to trust
its numbers: **no code in this file has ever been executed.**  There is no
Jetson and no TensorRT installation on the machine that produced the published
results, so this backend is unverified.  Nothing in ``results/published/`` came
from it, ``benchmarks/results.csv`` carries no TensorRT row and CI does not
import it.  ``docs/jetson.md`` says the same thing and lists the exact commands
to run once hardware exists.

Written now rather than later for two reasons.  It fixes the interface the rest
of the pipeline expects from an accelerated backend, so
``scripts/benchmark_latency.py`` needs no changes on the day the board arrives,
and it makes the shape of the work visible: the awkward part of a TensorRT
deployment is the memory management and the version drift below, not the
conversion.

Version handling.  TensorRT 8 binds by index through ``execute_async_v2`` and
TensorRT 10 binds by tensor name through ``execute_async_v3``.  Both are
supported here because Jetpack 5 ships TRT 8.5 and Jetpack 6 ships TRT 10, and
a Jetson-targeting repository that only handles one of them is a repository
that works on somebody else's board.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import BackendInfo, Detector, DetectorConfig

__all__ = ["TensorRTDetector"]


class TensorRTDetector(Detector):
    """Runs a serialised TensorRT engine.

    Parameters
    ----------
    engine_path:
        A ``.engine`` or ``.plan`` file built by ``scripts/export_tensorrt.py``
        **on the same device, driver and TensorRT version** it will run on.  A
        plan is not portable and loading one built elsewhere fails at
        deserialisation, which is the intended behaviour rather than a bug.
    """

    def __init__(self, engine_path: str | Path, config: DetectorConfig | None = None) -> None:
        super().__init__(config or DetectorConfig())
        import tensorrt as trt  # noqa: F401  (import order matters before pycuda)
        import pycuda.autoinit  # noqa: F401  (creates the CUDA context)
        import pycuda.driver as cuda

        self.engine_path = Path(engine_path)
        if not self.engine_path.exists():
            raise FileNotFoundError(
                f"no engine at {self.engine_path}.  Build one on this device with "
                "`python scripts/export_tensorrt.py --precision fp16`; engines are "
                "not portable between devices or TensorRT versions."
            )

        self._trt = trt
        self._cuda = cuda
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(
                f"TensorRT refused to deserialise {self.engine_path}.  The usual cause "
                "is a version or device mismatch; rebuild the engine here."
            )
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self._v3 = hasattr(self.context, "execute_async_v3")

        self._tensor_names = self._collect_tensor_names()
        self._input_name = self._tensor_names["input"]
        self._output_name = self._tensor_names["output"]
        self._input_shape = tuple(self.engine.get_tensor_shape(self._input_name)) if self._v3 \
            else tuple(self.engine.get_binding_shape(0))
        self._output_shape = tuple(self.engine.get_tensor_shape(self._output_name)) if self._v3 \
            else tuple(self.engine.get_binding_shape(1))

        self._input_dtype = self._numpy_dtype(self._input_name)
        self._output_dtype = self._numpy_dtype(self._output_name)

        self._host_output = cuda.pagelocked_empty(
            int(np.prod(self._output_shape)), self._output_dtype
        )
        self._device_input = cuda.mem_alloc(
            int(np.prod(self._input_shape)) * np.dtype(self._input_dtype).itemsize
        )
        self._device_output = cuda.mem_alloc(self._host_output.nbytes)
        self.precision = self._infer_precision()

    # -- version shims ----------------------------------------------------

    def _collect_tensor_names(self) -> dict[str, str]:
        trt = self._trt
        if self._v3:
            names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
            inputs = [n for n in names
                      if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
            outputs = [n for n in names
                       if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        else:
            names = [self.engine.get_binding_name(i) for i in range(self.engine.num_bindings)]
            inputs = [n for n in names if self.engine.binding_is_input(n)]
            outputs = [n for n in names if not self.engine.binding_is_input(n)]
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(
                f"expected one input and one output tensor, found {inputs} and {outputs}.  "
                "The exporter builds a single-input single-output graph; an engine with "
                "a fused NMS plugin needs a different decoder."
            )
        return {"input": inputs[0], "output": outputs[0]}

    def _numpy_dtype(self, name: str) -> np.dtype:
        trt = self._trt
        dtype = (
            self.engine.get_tensor_dtype(name)
            if self._v3
            else self.engine.get_binding_dtype(name)
        )
        return np.dtype(trt.nptype(dtype))

    def _infer_precision(self) -> str:
        """Best effort label from the engine's own layer precisions.

        TensorRT does not record "this engine was built with --int8" anywhere
        readable, so this inspects the IO dtypes and falls back to the file
        name, which ``scripts/export_tensorrt.py`` always stamps.
        """
        stem = self.engine_path.stem.lower()
        for candidate in ("int8", "fp16", "fp32"):
            if candidate in stem:
                return candidate
        if self._input_dtype == np.float16:
            return "fp16"
        return "fp32"

    # -- inference --------------------------------------------------------

    def _forward(self, tensor: np.ndarray) -> np.ndarray:
        cuda = self._cuda
        if tensor.dtype != self._input_dtype:
            tensor = tensor.astype(self._input_dtype)
        tensor = np.ascontiguousarray(tensor)

        cuda.memcpy_htod_async(self._device_input, tensor, self.stream)
        if self._v3:
            self.context.set_tensor_address(self._input_name, int(self._device_input))
            self.context.set_tensor_address(self._output_name, int(self._device_output))
            self.context.execute_async_v3(stream_handle=self.stream.handle)
        else:
            self.context.execute_async_v2(
                bindings=[int(self._device_input), int(self._device_output)],
                stream_handle=self.stream.handle,
            )
        cuda.memcpy_dtoh_async(self._host_output, self._device_output, self.stream)
        # Synchronising here rather than leaving the copy in flight is what
        # makes the timing in Detector.detect mean anything.
        self.stream.synchronize()
        return self._host_output.reshape(self._output_shape).astype(np.float32)

    @property
    def info(self) -> BackendInfo:
        return BackendInfo(
            backend="tensorrt",
            precision=self.precision,
            model_path=str(self.engine_path),
            providers=("cuda",),
            extra={
                "tensorrt_version": getattr(self._trt, "__version__", "unknown"),
                "api": "v3" if self._v3 else "v2",
                "input_shape": list(self._input_shape),
                "output_shape": list(self._output_shape),
            },
        )

    def close(self) -> None:
        for attribute in ("_device_input", "_device_output"):
            buffer = getattr(self, attribute, None)
            if buffer is not None:
                buffer.free()
                setattr(self, attribute, None)
        self.context = None
        self.engine = None

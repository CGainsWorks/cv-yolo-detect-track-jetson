"""PyTorch backend.  Present so the exported graphs have something to be
checked against and so the FP32/FP16 latency story has a pre-export baseline.

It is not part of the runtime.  ``requirements.txt`` does not contain torch and
``docker/Dockerfile.runtime`` asserts that it is absent, because torch is two
gigabytes the deployed image has no use for once the model is an ONNX graph.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import BackendInfo, Detector, DetectorConfig

__all__ = ["TorchDetector"]


class TorchDetector(Detector):
    """Runs the ``.pt`` checkpoint through the same module the export traces.

    :func:`edgetrack.detect.yolox_net.deploy_module` builds the network with
    the input fold and the in-graph decode the ONNX file has, so this backend
    and the exported ones produce the same head and are compared on the
    arithmetic.  Decoding and NMS come from :mod:`edgetrack.detect.base` as for
    every other backend.
    """

    def __init__(
        self,
        weights: str | Path,
        config: DetectorConfig | None = None,
        device: str = "cuda",
        half: bool = False,
    ) -> None:
        super().__init__(config or DetectorConfig())
        import torch

        from .yolox_net import deploy_module

        self.weights = Path(weights)
        if not self.weights.exists():
            raise FileNotFoundError(f"no checkpoint at {self.weights}; run `make train`")

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "cuda was requested but torch reports no CUDA device.  Either the "
                "CPU wheel is installed or the GPU is not visible; "
                "`python -c \"import torch; print(torch.__version__)\"` will say which."
            )
        self._torch = torch
        # cuDNN autotuning, so the PyTorch row is PyTorch at its best rather
        # than PyTorch with an obvious flag left off.  It changes very little
        # here, which is itself informative: at batch 1 this network is bound
        # by kernel launch overhead rather than by convolution throughput and
        # scripts/probe_launch_bound.py measures that directly.
        torch.backends.cudnn.benchmark = True
        self.device = torch.device(device)
        self.half = bool(half)
        if self.half and self.device.type != "cuda":
            raise ValueError("half precision needs a CUDA device")

        model, meta = deploy_module(self.weights, self.config.imgsz)
        model = model.to(self.device).eval()
        if self.half:
            model = model.half()
        self.model = model
        self._names = dict(enumerate(meta["class_names"]))

    def _forward(self, tensor: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.inference_mode():
            batch = torch.from_numpy(tensor).to(self.device)
            if self.half:
                batch = batch.half()
            output = self.model(batch)
            if isinstance(output, (list, tuple)):
                output = output[0]
            if self.device.type == "cuda":
                # Without this the timing in Detector.detect measures the launch
                # of the kernels rather than their completion.
                torch.cuda.synchronize()
            return output.float().cpu().numpy()

    @property
    def info(self) -> BackendInfo:
        return BackendInfo(
            backend="pytorch",
            precision="fp16" if self.half else "fp32",
            model_path=str(self.weights),
            providers=(str(self.device),),
            extra={"class_names": dict(self._names) if self._names else {}},
        )

    def close(self) -> None:
        self.model = None
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

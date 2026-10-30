"""YOLOX-Tiny in the form this repository trains, exports and deploys it.

Imports torch and yolox, so like :mod:`edgetrack.reid_net` it belongs to the
training and export side only.  The runtime never reaches it:
``tests/test_runtime_boundary.py`` imports every runtime module in a fresh
interpreter and fails if torch shows up and has a dedicated test that this
module is not reachable from the pipeline.

YOLOX is Megvii's detector, Apache-2.0 for the code and for the released COCO
checkpoints alike, which is the whole reason it is here; ``docs/licences.md``
has the reasoning and the alternatives that were weighed.

Two things change between the network YOLOX trains and the graph this
repository exports and both are exact rather than approximate.

The input.  YOLOX takes BGR in 0..255 with no normalisation.  The runtime feeds
RGB in 0..1, because that is what :meth:`edgetrack.detect.base.Detector.preprocess`
produces and what every backend and the INT8 calibrator already share.  Rather
than add a second preprocessing path, :func:`fold_input_into_stem` scales the
first convolution's weights by 255 and reverses their colour order.  The Focus
stem in front of that convolution only slices the image into four phases, so
the fold is the same arithmetic in a different order: there is no extra node in
the graph and nothing for TensorRT to fuse.

The output.  YOLOX's head emits per-anchor box offsets, an objectness score and
per-class scores.  :class:`DeployYOLOX` decodes the offsets against the stride
8, 16 and 32 grids and multiplies objectness into the class scores, then
transposes to ``(1, 4 + num_classes, anchors)``.  That is the contract
:func:`edgetrack.detect.base.decode_head` has always consumed: centre, size and
one score per class, already through the sigmoid, in letterboxed pixels.  At
640x384 the three grids hold 5040 anchors, the same count as before, so every
consumer of the head, the TensorRT builder included, is unchanged.
"""

from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import nn

__all__ = [
    "ARCHITECTURES",
    "STRIDES",
    "build_yolox",
    "load_matching",
    "load_checkpoint",
    "fold_input_into_stem",
    "DeployYOLOX",
    "deploy_module",
]

#: depth and width multipliers, from Megvii's exps/default/*.py
ARCHITECTURES = {
    "yolox_nano": (0.33, 0.25),
    "yolox_tiny": (0.33, 0.375),
    "yolox_s": (0.33, 0.50),
}

STRIDES = (8, 16, 32)


def build_yolox(architecture: str, num_classes: int) -> nn.Module:
    """The YOLOX network, as ``yolox.exp.Exp.get_model`` builds it."""
    from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead

    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"unknown architecture {architecture!r}; pick from {sorted(ARCHITECTURES)}"
        )
    depth, width = ARCHITECTURES[architecture]
    in_channels = [256, 512, 1024]
    depthwise = architecture == "yolox_nano"
    backbone = YOLOPAFPN(depth, width, in_channels=in_channels, depthwise=depthwise, act="silu")
    head = YOLOXHead(num_classes, width, in_channels=in_channels, depthwise=depthwise, act="silu")
    model = YOLOX(backbone, head)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eps = 1e-3
            module.momentum = 0.03
    return model


def load_matching(model: nn.Module, state: dict) -> dict:
    """Load every tensor whose name and shape match; report the rest.

    Fine-tuning from the 80-class COCO checkpoint to four classes leaves the
    three class-prediction convolutions with the wrong shape.  They are the
    only tensors expected to be skipped and the report says so, so a renamed
    module that silently drops half the backbone shows up in the training log
    rather than as a mysteriously slow convergence.
    """
    own = model.state_dict()
    loaded, skipped = {}, []
    for name, tensor in state.items():
        if name in own and own[name].shape == tensor.shape:
            loaded[name] = tensor
        else:
            skipped.append(name)
    missing = sorted(set(own) - set(loaded))
    model.load_state_dict(loaded, strict=False)
    return {"loaded": len(loaded), "skipped": sorted(skipped), "missing": missing}


def load_checkpoint(path: str | Path) -> dict:
    """Read a checkpoint written by ``scripts/train.py`` or Megvii's release.

    ``weights_only=True`` throughout, so loading a checkpoint never executes
    code from it.  Megvii's files carry the optimiser state as well, which is
    plain tensors and loads the same way.
    """
    return torch.load(str(path), map_location="cpu", weights_only=True)


def fold_input_into_stem(model: nn.Module) -> None:
    """Make the network take RGB in 0..1 instead of BGR in 0..255, in place.

    The stem's convolution sees twelve channels: four spatial phases of three
    colours each.  Reversing each group of three and scaling by 255 gives a
    convolution that computes on RGB/255 exactly what the original computed on
    BGR.  Zero padding is zero in both encodings, so the borders agree as well.
    """
    conv = model.backbone.backbone.stem.conv.conv
    weight = conv.weight.data
    out_channels, in_channels, kh, kw = weight.shape
    if in_channels != 12:
        raise ValueError(f"expected a Focus stem with 12 input channels, got {in_channels}")
    folded = weight.view(out_channels, 4, 3, kh, kw).flip(2) * 255.0
    conv.weight.data = folded.reshape(out_channels, in_channels, kh, kw).contiguous()


def _grids(imgsz: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    width, height = imgsz
    grids, strides = [], []
    for stride in STRIDES:
        h, w = height // stride, width // stride
        yv, xv = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        grids.append(torch.stack((xv, yv), 2).reshape(-1, 2))
        strides.append(torch.full((h * w, 1), float(stride)))
    return torch.cat(grids).float()[None], torch.cat(strides)[None]


class DeployYOLOX(nn.Module):
    """RGB 0..1 in, ``(N, 4 + num_classes, anchors)`` out.

    Written without in-place slice assignment, which is how YOLOX's own
    ``decode_outputs`` works and which traces to ScatterND.  Slice, Add, Mul,
    Exp, Concat and one Transpose is a graph every runtime here handles and one
    the INT8 quantiser can be told precisely which parts to leave alone.
    """

    def __init__(self, model: nn.Module, imgsz: tuple[int, int]) -> None:
        super().__init__()
        width, height = imgsz
        if width % 32 or height % 32:
            raise ValueError(f"imgsz must be a multiple of 32 in both axes, got {imgsz}")
        self.model = model
        self.model.head.decode_in_inference = False
        grid, stride = _grids(imgsz)
        self.register_buffer("grid", grid, persistent=False)
        self.register_buffer("stride", stride, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.model(x)
        xy = (raw[..., 0:2] + self.grid) * self.stride
        wh = torch.exp(raw[..., 2:4]) * self.stride
        scores = raw[..., 4:5] * raw[..., 5:]
        return torch.cat((xy, wh, scores), dim=-1).permute(0, 2, 1)


def deploy_module(
    checkpoint: str | Path,
    imgsz: tuple[int, int],
    architecture: str | None = None,
    num_classes: int | None = None,
) -> tuple[DeployYOLOX, dict]:
    """Build the deployable module from a checkpoint, ready for export or timing.

    A checkpoint from ``scripts/train.py`` names its own architecture and class
    count; a raw Megvii checkpoint does not, so those two are arguments too.
    """
    ckpt = load_checkpoint(checkpoint)
    architecture = architecture or ckpt.get("architecture")
    num_classes = num_classes or ckpt.get("num_classes")
    if architecture is None or num_classes is None:
        raise ValueError(f"{checkpoint} does not record its architecture; pass it explicitly")
    model = build_yolox(architecture, int(num_classes))
    report = load_matching(model, ckpt["model"])
    if report["skipped"] or report["missing"]:
        raise RuntimeError(
            f"{checkpoint} does not match {architecture} with {num_classes} classes: "
            f"skipped {report['skipped'][:5]}, missing {report['missing'][:5]}"
        )
    model = copy.deepcopy(model).eval()
    fold_input_into_stem(model)
    meta = {
        "architecture": architecture,
        "num_classes": int(num_classes),
        "class_names": list(ckpt.get("class_names", [])),
    }
    return DeployYOLOX(model, imgsz).eval(), meta

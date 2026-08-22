"""The appearance embedding network.  Train-time only.

This module imports torch at import time, which is exactly why nothing in the
runtime path imports this module.  ``scripts/train_reid.py`` and
``scripts/export_reid.py`` use it; the deployed pipeline loads the exported
ONNX through :class:`edgetrack.track.appearance.OnnxReIDEncoder` and never sees
torch.  ``tests/test_no_torch_in_runtime.py`` enforces the separation.

Architecture.  Four stride-2 convolutions taking a 64x128 crop down to 4x8,
global average pooling, then a 128-dimensional bottleneck.  Roughly 0.1 M
parameters, which is deliberate and was arrived at by measurement rather than
taste: the first version used two convolutions per block and 32 to 256
channels, came to 1.2 M parameters, and cost as much arithmetic per frame as
the detector did once eight objects were in shot.  The descriptor runs once per
detection per frame, so its cost multiplies by the number of objects whilst the
detector's does not, and a re-identification network sized like a detector
destroys the latency budget it was added to improve.  Shrinking it by seven
times cost under a point of rank-1 on this data; both sets of numbers are in
``docs/tracking.md``.

Training follows the standard re-identification recipe of a classification head
over identities with a BNNeck between the embedding and the classifier.  The
embedding is taken before the classifier and after batch normalisation, which
is what makes cosine distance the right metric at inference: the BNNeck
normalises the feature distribution so that the angular separation the softmax
learned survives L2 normalisation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

__all__ = ["ReIDNet", "EMBEDDING_DIM", "CROP_HW"]

EMBEDDING_DIM = 128
CROP_HW = (128, 64)  # height, width


def _block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class ReIDNet(nn.Module):
    """Embedding network with an optional identity classifier.

    ``forward`` returns the L2-normalised embedding, so the exported graph is
    exactly what the tracker consumes and the normalisation cannot be forgotten
    at the call site.  ``forward_train`` returns logits and the pre-normalised
    embedding for the loss.
    """

    def __init__(self, num_identities: int = 0, embedding_dim: int = EMBEDDING_DIM) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            _block(3, 16),
            _block(16, 32),
            _block(32, 64),
            _block(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.bottleneck = nn.Linear(128, embedding_dim)
        # BNNeck.  Affine bias is off because the classifier that follows has
        # no bias either; leaving both on lets the network trade a shift in the
        # embedding against a shift in the logits and the embedding drifts off
        # the unit sphere it is about to be projected onto.
        self.neck = nn.BatchNorm1d(embedding_dim)
        self.neck.bias.requires_grad_(False)
        self.classifier = (
            nn.Linear(embedding_dim, num_identities, bias=False) if num_identities else None
        )
        self.embedding_dim = embedding_dim

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        return self.bottleneck(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return functional.normalize(self.neck(self.features(x)), p=2, dim=1)

    def forward_train(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.features(x)
        normed = self.neck(embedding)
        if self.classifier is None:
            raise RuntimeError("this network was built without a classifier head")
        return self.classifier(normed), embedding

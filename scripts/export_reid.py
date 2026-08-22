#!/usr/bin/env python
"""Export the appearance embedding to ONNX with a dynamic batch dimension.

    python scripts/export_reid.py

Dynamic batch here, fixed batch for the detector, and the asymmetry is
deliberate.  The detector always sees exactly one frame, so a fixed shape lets
TensorRT pick its best kernels.  The embedding sees one crop per detection, and
that count changes every frame; a fixed batch would mean either padding to a
worst case or one session call per object, and the second costs more in call
overhead than the network costs in arithmetic.

The check at the end is not ceremony.  A silently wrong export here would
degrade tracking quality without any error, and the difference between a good
embedding and a scrambled one does not show up until the ID-switch count is
read a week later.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from edgetrack.paths import serialise_path, write_text_lf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_onnx import sha256  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--weights", default="models/reid_edgetrack.pt", type=Path)
    parser.add_argument("--out", default="models/reid_edgetrack.onnx", type=Path)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--tolerance", type=float, default=2e-3)
    args = parser.parse_args(argv)

    import torch
    from edgetrack.reid_net import CROP_HW, ReIDNet

    if not args.weights.exists():
        raise FileNotFoundError(
            f"no re-ID checkpoint at {args.weights}; run `python scripts/train_reid.py`"
        )
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
    model = ReIDNet(
        num_identities=int(checkpoint["num_identities"]),
        embedding_dim=int(checkpoint["embedding_dim"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    height, width = CROP_HW
    dummy = torch.randn(4, 3, height, width)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # dynamo=False selects the TorchScript exporter.  The dynamo path in
    # torch 2.x ignores a downgrade request when its version converter cannot
    # express a node, and silently emits opset 18: fine for ONNX Runtime,
    # not fine for the TensorRT 8.5 parser on Jetpack 5.  The opset is checked
    # below rather than assumed.
    torch.onnx.export(
        model,
        dummy,
        str(args.out),
        input_names=["crops"],
        output_names=["embeddings"],
        opset_version=args.opset,
        dynamic_axes={"crops": {0: "batch"}, "embeddings": {0: "batch"}},
        dynamo=False,
    )

    import onnx

    graph = onnx.load(str(args.out))
    opsets = {(entry.domain or "ai.onnx"): entry.version for entry in graph.opset_import}
    actual_opset = opsets.get("ai.onnx")
    if actual_opset != args.opset:
        raise RuntimeError(
            f"asked for opset {args.opset} and got {actual_opset}.  The exporter "
            "downgraded silently; check the torch version before trusting this file "
            "on a TensorRT 8.x parser."
        )

    # Verify against the torch reference at a batch size the export never saw,
    # which is what catches a dynamic axis that was declared but not honoured.
    import onnxruntime as ort

    session = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    probe = torch.randn(7, 3, height, width)
    with torch.inference_mode():
        expected = model(probe).numpy()
    actual = session.run(["embeddings"], {"crops": probe.numpy()})[0]
    error = float(np.abs(expected - actual).max())
    if error > args.tolerance:
        raise RuntimeError(
            f"ONNX output differs from torch by {error:.2e}, above the tolerance "
            f"{args.tolerance:.0e}.  The export is not usable."
        )
    norms = np.linalg.norm(actual, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise RuntimeError(
            f"exported embeddings are not unit norm (min {norms.min():.4f}, "
            f"max {norms.max():.4f}); cosine distance would be meaningless"
        )

    record = {
        "onnx": serialise_path(args.out),
        "sha256": sha256(args.out),
        "bytes": args.out.stat().st_size,
        "opset": int(actual_opset),
        "opset_domains": {k: int(v) for k, v in opsets.items()},
        "embedding_dim": int(checkpoint["embedding_dim"]),
        "crop_hw": [height, width],
        "max_abs_error_vs_torch": error,
        "dynamic_batch_checked_at": 7,
    }
    write_text_lf(args.out.with_suffix(".json"), json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

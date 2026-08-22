#!/usr/bin/env python
"""Fine-tune YOLOv8 on the custom dataset.

    python scripts/train.py --config configs/train.yaml

Needs ultralytics and a CUDA GPU for anything but a toy run; install with
``pip install -r requirements-train.txt``.  Nothing in the runtime imports
ultralytics, so this is the only place the AGPL toolchain appears.

Two things worth knowing about the shapes.  Training runs at a square 640
because that is what the mosaic augmentation and the pretrained anchors expect,
whilst export and inference run at 640x384 because that is the native aspect of
the data and padding a 16:9.6 frame into a square wastes a third of the compute
on grey bars.  YOLOv8 is fully convolutional so the two are compatible; the
only requirement is that both sides stay a multiple of 32.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import yaml

import _bootstrap  # noqa: F401
from edgetrack.data.synth import CLASSES
from edgetrack.paths import serialise_path


def load_config(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="configs/train.yaml", type=Path)
    parser.add_argument("--data", default="data/processed/data.yaml", type=Path)
    parser.add_argument("--epochs", type=int, default=None, help="override the config")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="models", type=Path)
    parser.add_argument("--name", default="yolov8n_edgetrack")
    args = parser.parse_args(argv)

    from ultralytics import YOLO
    import torch

    config = load_config(args.config)
    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"{data_yaml} not found.  Run `python scripts/make_dataset.py` first."
        )
    names = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))["names"]
    if [names[i] for i in sorted(names)] != list(CLASSES):
        raise ValueError(
            f"class list in {data_yaml} does not match edgetrack.data.synth.CLASSES; "
            "regenerate the dataset rather than editing the yaml"
        )

    device = args.device if args.device is not None else config.get("device", 0)
    if str(device) not in ("cpu",) and not torch.cuda.is_available():
        print(
            "warning: no CUDA device visible, falling back to cpu.  Expect this to "
            "take roughly thirty times as long.",
            file=sys.stderr,
        )
        device = "cpu"

    overrides = dict(config.get("train", {}))
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch is not None:
        overrides["batch"] = args.batch

    # Absolute, deliberately.  A relative `project` is resolved against
    # Ultralytics' own `runs_dir` setting, which is a per-user JSON file left
    # over from whatever project was trained on this machine last, so a
    # relative path here silently writes the run into somebody else's
    # repository.  This was not hypothetical during development.
    project = (Path(args.out) / "runs").resolve()
    model = YOLO(config.get("model", "yolov8n.pt"))

    started = time.perf_counter()
    results = model.train(
        data=str(data_yaml.resolve()),
        project=str(project),
        name=args.name,
        exist_ok=True,
        device=device,
        **overrides,
    )
    elapsed = time.perf_counter() - started

    run_dir = Path(results.save_dir)
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError(f"training finished but {best} does not exist")

    destination = Path(args.out) / f"{args.name}.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, destination)

    # Ultralytics' own validation numbers, kept for the cross-check against the
    # metric implementation in this repository.  scripts/evaluate.py recomputes
    # mAP from raw predictions and tests/test_metrics_detection.py asserts the
    # two agree, which is the whole argument for trusting either.
    metrics = model.val(data=str(data_yaml.resolve()), split="val", device=device)
    summary = {
        "weights": serialise_path(destination),
        # Ultralytics resolves a relative project against its own
        # per-user settings file, so the run directory can legitimately
        # sit outside the repository.  Recording only its name keeps a
        # committed results file from naming somebody's home directory.
        "run_dir": run_dir.name,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "train_seconds": round(elapsed, 1),
        "overrides": overrides,
        "ultralytics_val": {
            "map50": round(float(metrics.box.map50), 5),
            "map50_95": round(float(metrics.box.map), 5),
            "precision": round(float(metrics.box.mp), 5),
            "recall": round(float(metrics.box.mr), 5),
            "per_class_map50_95": {
                CLASSES[i]: round(float(v), 5) for i, v in enumerate(metrics.box.maps)
            },
        },
    }
    out_json = Path(args.out) / f"{args.name}_train.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

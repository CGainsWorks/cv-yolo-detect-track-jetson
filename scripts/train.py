#!/usr/bin/env python
"""Fine-tune YOLOX-Tiny on the custom dataset.

    python scripts/train.py --config configs/train.yaml

Needs torch, yolox and a CUDA GPU for anything but a toy run; install with
``pip install -r requirements-train.txt``.  YOLOX is Megvii's detector and is
Apache-2.0, code and COCO checkpoint alike, so nothing here carries a copyleft
obligation into the weights; ``docs/licences.md`` has the detail.

The trainer is YOLOX's own, ``yolox.core.Trainer``, driven by an ``Exp``
subclass defined here.  Three things are this repository's rather than YOLOX's:

* The labels.  ``scripts/make_dataset.py`` writes YOLO text labels and YOLOX
  reads COCO JSON, so :func:`write_coco_json` converts each split once into
  ``<data-root>/annotations/``.  The conversion is checked by counting: every
  label line becomes exactly one annotation or the script stops.
* The starting weights.  Megvii's checkpoint is loaded with ``weights_only``
  and its 80-class prediction convolutions are the only tensors allowed not to
  match; anything else failing to load is an error, not a warning.
* The checkpoint that comes out.  Only the EMA weights are kept, with the
  architecture and class names beside them, so ``scripts/export_onnx.py`` and
  the PyTorch backend can rebuild the network without being told what it is.

Training and export both run at the native 640x384 aspect.  YOLOX keeps the
aspect through its multi-scale jitter, so no square resize is involved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import time
import urllib.request
from argparse import Namespace
from pathlib import Path

import yaml

import _bootstrap  # noqa: F401
from edgetrack.data.synth import CLASSES
from edgetrack.paths import serialise_path, write_text_lf


def load_config(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_pretrained(spec: dict, out_dir: Path) -> Path:
    """Megvii's checkpoint, downloaded once and checked against its digest."""
    destination = out_dir / spec["file"]
    if not destination.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"downloading {spec['url']}", flush=True)
        temporary = destination.with_suffix(".part")
        with urllib.request.urlopen(spec["url"], timeout=120) as response, temporary.open(
            "wb"
        ) as handle:
            shutil.copyfileobj(response, handle, length=1 << 20)
        temporary.replace(destination)
    actual = sha256(destination)
    if actual != spec["sha256"]:
        raise RuntimeError(
            f"{destination}: sha256 {actual} does not match {spec['sha256']} in the config"
        )
    return destination


def write_coco_json(data_root: Path, split: str) -> tuple[Path, int]:
    """Convert one split's YOLO labels to the COCO JSON YOLOX reads."""
    import cv2

    images_dir = data_root / "images" / split
    labels_dir = data_root / "labels" / split
    images, annotations = [], []
    label_lines = 0
    for image_id, image_path in enumerate(sorted(images_dir.glob("*.jpg")), start=1):
        height, width = cv2.imread(str(image_path)).shape[:2]
        images.append(
            {"id": image_id, "file_name": image_path.name, "width": width, "height": height}
        )
        label_path = labels_dir / (image_path.stem + ".txt")
        if not label_path.exists():
            continue
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            label_lines += 1
            cls, cx, cy, w, h = line.split()
            bw, bh = float(w) * width, float(h) * height
            x0, y0 = float(cx) * width - bw / 2, float(cy) * height - bh / 2
            annotations.append({
                "id": len(annotations) + 1,
                "image_id": image_id,
                "category_id": int(cls) + 1,
                "bbox": [round(x0, 3), round(y0, 3), round(bw, 3), round(bh, 3)],
                "area": round(bw * bh, 3),
                "iscrowd": 0,
            })
    if not images:
        raise FileNotFoundError(f"no images under {images_dir}; run `make data` first")
    if len(annotations) != label_lines:
        raise RuntimeError(f"{split}: {label_lines} label lines became {len(annotations)} boxes")
    categories = [{"id": i + 1, "name": name} for i, name in enumerate(CLASSES)]
    out = data_root / "annotations" / f"{split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_text_lf(out, json.dumps({
        "images": images, "annotations": annotations, "categories": categories,
    }))
    return out, len(annotations)


def use_standard_cocoeval_if_needed() -> str:
    """Fall back to pycocotools when YOLOX's C++ COCO evaluator cannot build.

    YOLOX's evaluator imports ``yolox.layers.COCOeval_opt``, a C++ port of
    ``COCOeval.evaluate`` that is compiled on first use and falls back to
    pycocotools only on ImportError.  On a machine with no C++ toolchain the
    compile raises RuntimeError instead, which escapes the fallback and ends
    training at the first evaluation.  Trying the build here and hiding the
    optimised class if it fails turns that into the ImportError YOLOX expects.
    The two evaluators compute the same numbers; the C++ one is faster.
    """
    import yolox.layers as layers

    if not hasattr(layers, "COCOeval_opt"):
        return "pycocotools"
    try:
        layers.FastCOCOEvalOp().load(verbose=False)
    except Exception:  # noqa: BLE001 - any build failure means the fallback
        delattr(layers, "COCOeval_opt")
        return "pycocotools"
    return "yolox fast_cocoeval"


def make_exp(config: dict, data_root: Path, out_dir: Path, pretrained: Path):
    """A ``yolox.exp.Exp`` for this dataset, with the checkpoint loaded."""
    from yolox.exp import Exp

    from edgetrack.detect.yolox_net import ARCHITECTURES, build_yolox, load_checkpoint
    from edgetrack.detect.yolox_net import load_matching

    class EdgeTrackExp(Exp):
        def __init__(self) -> None:
            super().__init__()
            self.depth, self.width = ARCHITECTURES[config["architecture"]]
            self.num_classes = len(CLASSES)
            for key, value in config.get("exp", {}).items():
                if not hasattr(self, key):
                    raise KeyError(f"exp.{key} is not a yolox.exp.Exp attribute")
                setattr(self, key, tuple(value) if isinstance(value, list) else value)
            self.data_dir = str(data_root)
            self.train_ann = "train.json"
            self.val_ann = "val.json"
            self.output_dir = str(out_dir / "runs")
            self.seed = int(config.get("seed", 0))
            self.exp_name = config["architecture"]
            self.load_report: dict = {}

        def get_model(self):
            if getattr(self, "model", None) is None:
                self.model = build_yolox(config["architecture"], self.num_classes)
                self.model.head.initialize_biases(1e-2)
                state = load_checkpoint(pretrained)["model"]
                self.load_report = load_matching(self.model, state)
                unexpected = [
                    name for name in self.load_report["skipped"]
                    if not name.startswith("head.cls_preds.")
                ]
                if unexpected or len(self.load_report["missing"]) != 6:
                    raise RuntimeError(f"pretrained weights did not load: {self.load_report}")
            self.model.train()
            return self.model

        def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
            from yolox.data import (
                COCODataset,
                DataLoader,
                InfiniteSampler,
                MosaicDetection,
                TrainTransform,
                YoloBatchSampler,
                worker_init_reset_seed,
            )

            dataset = COCODataset(
                data_dir=self.data_dir,
                json_file=self.train_ann,
                name="images/train",
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=50, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob
                ),
                cache=False,
            )
            self.dataset = MosaicDetection(
                dataset,
                mosaic=not no_aug,
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=120, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob
                ),
                degrees=self.degrees,
                translate=self.translate,
                mosaic_scale=self.mosaic_scale,
                mixup_scale=self.mixup_scale,
                shear=self.shear,
                enable_mixup=self.enable_mixup,
                mosaic_prob=self.mosaic_prob,
                mixup_prob=self.mixup_prob,
            )
            sampler = InfiniteSampler(len(self.dataset), seed=self.seed)
            batch_sampler = YoloBatchSampler(
                sampler=sampler, batch_size=batch_size, drop_last=False, mosaic=not no_aug
            )
            return DataLoader(
                self.dataset,
                num_workers=self.data_num_workers,
                pin_memory=True,
                batch_sampler=batch_sampler,
                worker_init_fn=worker_init_reset_seed,
            )

        def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
            import torch
            from yolox.data import COCODataset, ValTransform

            dataset = COCODataset(
                data_dir=self.data_dir,
                json_file=self.val_ann,
                name="images/val",
                img_size=self.test_size,
                preproc=ValTransform(legacy=legacy),
            )
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=torch.utils.data.SequentialSampler(dataset),
                num_workers=self.data_num_workers,
                pin_memory=True,
            )

    return EdgeTrackExp()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="configs/train.yaml", type=Path)
    parser.add_argument("--data-root", default="data/processed", type=Path)
    parser.add_argument("--epochs", type=int, default=None, help="override the config")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--out", default="models", type=Path)
    parser.add_argument("--name", default="yolox_tiny_edgetrack")
    args = parser.parse_args(argv)

    import numpy as np
    import torch
    import yolox
    from yolox.core import Trainer

    config = load_config(args.config)
    if args.epochs is not None:
        config["exp"]["max_epoch"] = args.epochs
        config["exp"]["no_aug_epochs"] = min(config["exp"]["no_aug_epochs"], args.epochs // 4)
        config["exp"]["eval_interval"] = min(config["exp"]["eval_interval"], args.epochs)
    batch = args.batch or int(config["batch"])
    if not torch.cuda.is_available():
        raise RuntimeError("YOLOX's trainer needs a CUDA device; torch reports none")

    data_root = Path(args.data_root)
    counts = {}
    for split in ("train", "val"):
        _, counts[split] = write_coco_json(data_root, split)
    print(f"COCO labels written: {counts}", flush=True)

    out_dir = Path(args.out)
    pretrained = fetch_pretrained(config["pretrained"], out_dir)

    seed = int(config.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    cocoeval = use_standard_cocoeval_if_needed()
    print(f"COCO evaluator: {cocoeval}", flush=True)
    exp = make_exp(config, data_root.resolve(), out_dir.resolve(), pretrained)
    trainer_args = Namespace(
        experiment_name=args.name,
        batch_size=batch,
        fp16=bool(config.get("fp16", True)),
        ckpt=None,
        resume=False,
        start_epoch=None,
        occupy=False,
        cache=False,
        logger="tensorboard",
        devices=1,
    )
    trainer = Trainer(exp, trainer_args)
    started = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - started

    best = Path(trainer.file_name) / "best_ckpt.pth"
    if not best.exists():
        raise RuntimeError(f"training finished but {best} does not exist")
    # The trainer saves the EMA weights under "model" alongside the optimiser
    # state and a numpy scalar, which weights_only refuses.  The file is our
    # own, written seconds ago, so it is read in full once and only the
    # tensors are written out.
    state = torch.load(str(best), map_location="cpu", weights_only=False)["model"]
    destination = out_dir / f"{args.name}.pt"
    torch.save({
        "model": state,
        "architecture": config["architecture"],
        "num_classes": len(CLASSES),
        "class_names": list(CLASSES),
        "imgsz_wh": [int(config["exp"]["input_size"][1]), int(config["exp"]["input_size"][0])],
    }, destination)

    # YOLOX's own validation numbers on the val split, from pycocotools,
    # kept for the cross-check against this repository's metric
    # implementation in tests/test_backends.py.
    evaluator = exp.get_evaluator(batch_size=batch, is_distributed=False)
    model = exp.get_model()
    model.load_state_dict(state)
    model = model.cuda().eval()
    ap50_95, ap50, _ = exp.eval(model, evaluator, False)

    summary = {
        "weights": serialise_path(destination),
        "run_dir": Path(trainer.file_name).name,
        "architecture": config["architecture"],
        "pretrained": {k: config["pretrained"][k] for k in ("url", "sha256", "licence")},
        "pretrained_load": {
            "loaded": exp.load_report["loaded"],
            "skipped": exp.load_report["skipped"],
        },
        "device": "cuda:0",
        "gpu_name": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "yolox": yolox.__version__,
        "train_seconds": round(elapsed, 1),
        "batch": batch,
        "fp16_training": bool(config.get("fp16", True)),
        "exp": config["exp"],
        "label_boxes": counts,
        "yolox_val": {
            "map50": round(float(ap50), 5),
            "map50_95": round(float(ap50_95), 5),
            "evaluator": f"yolox.evaluators.COCOEvaluator ({cocoeval})",
        },
    }
    out_json = out_dir / f"{args.name}_train.json"
    write_text_lf(out_json, json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

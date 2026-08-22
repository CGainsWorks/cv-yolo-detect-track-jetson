#!/usr/bin/env python
"""Train the appearance embedding on crops from the training sequences.

    python scripts/train_reid.py --epochs 20

Identities are ``(sequence, instance)`` pairs: two objects in different
sequences are different identities even if they happen to be the same colour,
which is correct, because the tracker never has to associate across sequences.

Crops come from ground truth boxes rather than from detector output.  Using
detections would put detector error into the appearance model's training set,
and the val split here measures something narrow on purpose: given a clean
crop, does the embedding separate identities.  Whether that survives the real
detector's boxes is what ``scripts/eval_tracking.py`` measures, and the gap
between the two is reported in ``docs/tracking.md``.

Evaluation is rank-1 accuracy and mean average precision on held-out crops of
held-out identities, which is the standard re-identification protocol.  The
number that matters downstream is neither: it is the cosine margin between
same-identity and different-identity pairs, because that is what sets
``max_cosine_distance`` in the DeepSORT config.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import _bootstrap  # noqa: F401
from edgetrack.data.synth import config_for_split, generate_sequence
from edgetrack.paths import serialise_path
from edgetrack.track.appearance import CROP_SIZE


def build_crops(
    seeds: list[int], split: str, frames: int, min_visibility: float, min_pixels: int
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Render the sequences and cut every sufficiently visible object out."""
    crops: list[np.ndarray] = []
    labels: list[int] = []
    names: list[str] = []
    lookup: dict[tuple[int, int], int] = {}

    for seed in seeds:
        cfg = config_for_split(split, seed, n_frames=frames)
        for frame in generate_sequence(cfg):
            for ann in frame.annotations:
                x1, y1, x2, y2 = ann.xyxy
                if ann.visibility < min_visibility:
                    continue
                if (x2 - x1) < min_pixels or (y2 - y1) < min_pixels:
                    continue
                patch = frame.image[int(y1):int(np.ceil(y2)), int(x1):int(np.ceil(x2))]
                if patch.size == 0:
                    continue
                key = (seed, ann.instance_id)
                if key not in lookup:
                    lookup[key] = len(names)
                    names.append(f"seq{seed}_obj{ann.instance_id}")
                crops.append(cv2.resize(patch, CROP_SIZE, interpolation=cv2.INTER_LINEAR))
                labels.append(lookup[key])

    if not crops:
        raise RuntimeError("no crops survived the visibility and size filters")
    return np.stack(crops), np.asarray(labels, dtype=np.int64), names


def to_tensor(batch: np.ndarray) -> np.ndarray:
    array = batch[:, :, :, ::-1].astype(np.float32) / 255.0  # BGR to RGB
    array = (array - 0.5) / 0.5
    return np.ascontiguousarray(array.transpose(0, 3, 1, 2))


def cosine_margin(embeddings: np.ndarray, labels: np.ndarray, rng: random.Random) -> dict:
    """Mean cosine distance within an identity against between identities.

    This is the number that sets ``max_cosine_distance``.  A gate above the
    between-identity mean lets strangers match; a gate below the within-identity
    p95 loses the object it is meant to recover.
    """
    by_label: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        by_label.setdefault(int(label), []).append(index)

    same, different = [], []
    labels_with_pairs = [k for k, v in by_label.items() if len(v) >= 2]
    for _ in range(4000):
        if not labels_with_pairs:
            break
        label = rng.choice(labels_with_pairs)
        a, b = rng.sample(by_label[label], 2)
        same.append(1.0 - float(embeddings[a] @ embeddings[b]))
        other = rng.choice([k for k in by_label if k != label])
        c = rng.choice(by_label[other])
        different.append(1.0 - float(embeddings[a] @ embeddings[c]))

    same_array = np.asarray(same) if same else np.zeros(1)
    different_array = np.asarray(different) if different else np.zeros(1)
    return {
        "within_identity_mean": round(float(same_array.mean()), 4),
        "within_identity_p95": round(float(np.percentile(same_array, 95)), 4),
        "between_identity_mean": round(float(different_array.mean()), 4),
        "between_identity_p05": round(float(np.percentile(different_array, 5)), 4),
        "pairs_sampled": len(same),
    }


def rank1_and_map(embeddings: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Leave-one-out rank-1 accuracy and mean average precision."""
    similarity = embeddings @ embeddings.T
    np.fill_diagonal(similarity, -np.inf)
    order = np.argsort(-similarity, axis=1)
    correct = labels[order] == labels[:, None]

    rank1 = float(correct[:, 0].mean())
    average_precisions = []
    for row in correct:
        positives = np.nonzero(row)[0]
        if positives.size == 0:
            continue
        precision_at = (np.arange(positives.size) + 1) / (positives + 1)
        average_precisions.append(float(precision_at.mean()))
    return rank1, float(np.mean(average_precisions)) if average_precisions else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--train-seeds", nargs=2, type=int, default=[1000, 1060], metavar=("FROM", "TO")
    )
    parser.add_argument(
        "--val-seeds", nargs=2, type=int, default=[11000, 11015], metavar=("FROM", "TO")
    )
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-pixels", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="models/reid_edgetrack.pt", type=Path)
    parser.add_argument("--report", default="results/reid_train.json", type=Path)
    args = parser.parse_args(argv)

    import torch
    from torch import optim
    from edgetrack.reid_net import EMBEDDING_DIM, ReIDNet

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)

    wanted = args.device
    device = torch.device(
        wanted if (torch.cuda.is_available() or wanted == "cpu") else "cpu"
    )
    if str(device) != args.device:
        print(f"warning: falling back to {device}", file=sys.stderr)

    print("building crops", flush=True)
    train_crops, train_labels, train_names = build_crops(
        list(range(args.train_seeds[0], args.train_seeds[1])),
        "train", args.frames, args.min_visibility, args.min_pixels,
    )
    val_crops, val_labels, _ = build_crops(
        list(range(args.val_seeds[0], args.val_seeds[1])),
        "val", args.frames, args.min_visibility, args.min_pixels,
    )
    print(
        f"train {train_crops.shape[0]} crops over {len(train_names)} identities, "
        f"val {val_crops.shape[0]} crops over {len(set(val_labels.tolist()))} identities",
        flush=True,
    )

    model = ReIDNet(num_identities=len(train_names), embedding_dim=EMBEDDING_DIM).to(device)
    optimiser = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimiser,
        max_lr=args.lr,
        total_steps=max(1, args.epochs * (len(train_crops) // args.batch + 1)),
    )
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    x_train = torch.from_numpy(to_tensor(train_crops))
    y_train = torch.from_numpy(train_labels)
    started = time.perf_counter()
    history = []
    for epoch in range(args.epochs):
        model.train()
        permutation = torch.randperm(x_train.shape[0])
        total_loss, correct, seen = 0.0, 0, 0
        for start in range(0, x_train.shape[0], args.batch):
            index = permutation[start : start + args.batch]
            if index.numel() < 2:
                continue  # BatchNorm1d needs more than one sample
            batch = x_train[index].to(device, non_blocking=True)
            target = y_train[index].to(device, non_blocking=True)
            logits, _ = model.forward_train(batch)
            loss = criterion(logits, target)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            scheduler.step()
            total_loss += float(loss) * index.numel()
            correct += int((logits.argmax(1) == target).sum())
            seen += index.numel()
        history.append(
            {"epoch": epoch + 1, "loss": round(total_loss / max(seen, 1), 4),
             "train_acc": round(correct / max(seen, 1), 4)}
        )
        print(f"  epoch {epoch+1}/{args.epochs} loss {history[-1]['loss']:.4f} "
              f"acc {history[-1]['train_acc']:.4f}", flush=True)
    elapsed = time.perf_counter() - started

    model.eval()
    with torch.inference_mode():
        embeddings = []
        x_val = torch.from_numpy(to_tensor(val_crops))
        for start in range(0, x_val.shape[0], 256):
            embeddings.append(model(x_val[start : start + 256].to(device)).cpu().numpy())
    val_embeddings = np.concatenate(embeddings)

    rank1, mean_ap = rank1_and_map(val_embeddings, val_labels)
    margins = cosine_margin(val_embeddings, val_labels, rng)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "num_identities": len(train_names),
         "embedding_dim": EMBEDDING_DIM},
        args.out,
    )

    embedding_parameters = sum(
        param.numel()
        for name, param in model.named_parameters()
        if not name.startswith("classifier")
    )
    report = {
        "weights": serialise_path(args.out),
        "parameters_total": int(sum(p.numel() for p in model.parameters())),
        "parameters_embedding": int(embedding_parameters),
        "device": str(device),
        "epochs": args.epochs,
        "train_crops": int(train_crops.shape[0]),
        "train_identities": len(train_names),
        "val_crops": int(val_crops.shape[0]),
        "val_identities": int(len(set(val_labels.tolist()))),
        "train_seconds": round(elapsed, 1),
        "val_rank1": round(rank1, 4),
        "val_map": round(mean_ap, 4),
        "cosine": margins,
        "history": history,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

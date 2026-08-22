#!/usr/bin/env python
"""Join accuracy and latency into the headline table and benchmarks/results.csv.

    python scripts/make_results_table.py --published results/published

The join key is the model file, and the reason is worth stating.  Accuracy is a
property of the artefact: a given ONNX file produces the same boxes whichever
runtime executes it, to within floating point reassociation.  Latency is a
property of the artefact *and* the runtime *and* the device.  So one accuracy
measurement is shared by every row that runs the same file, and the latency
column differs per row.

Rows whose artefact was never scored for accuracy get ``n/a`` rather than a
borrowed number.  That is the case for the PyTorch rows, which exist to measure
GPU latency: the FP16 they run is a half-precision torch module rather than the
FP16 ONNX graph, and quoting the graph's mAP against it would be a join on a
label rather than on a thing.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from edgetrack.paths import path_name

COLUMNS = [
    "variant",
    "backend",
    "precision",
    "provider",
    "device_class",
    "map50",
    "map50_95",
    "inference_p50_ms",
    "inference_p95_ms",
    "end_to_end_p50_ms",
    "fps_end_to_end",
    "model_bytes",
]


def accuracy_by_model(detection: dict) -> dict[str, dict]:
    """Accuracy keyed by model file name.

    If the same file was scored twice, under two providers say, the two runs
    must agree.  A silent disagreement would mean the table's single accuracy
    column is a fiction, so it raises instead.
    """
    out: dict[str, dict] = {}
    for entry in detection.get("variants", {}).values():
        key = path_name(entry["model"])
        metrics = entry["metrics"]
        if key in out:
            previous = out[key]
            if abs(previous["map50_95"] - metrics["map50_95"]) > 1e-4:
                raise ValueError(
                    f"two runs of {key} disagree on mAP50-95: "
                    f"{previous['map50_95']} against {metrics['map50_95']}"
                )
            continue
        out[key] = metrics
    return out


def artefact_sizes(manifest_path: Path) -> dict[str, int]:
    """Byte counts from the committed manifest, not from the filesystem.

    The table used to call ``Path(...).stat()`` on each model file, which works
    on the machine that produced the results and returns nothing on a CI runner
    where ``models/`` is empty.  The regenerated CSV then differed from the
    committed one in a column nobody was looking at, and the diff step failed
    with no hint as to why.  The manifest is committed, digest-verified and
    present everywhere, so it is the right source.
    """
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {entry["name"]: int(entry["bytes"]) for entry in manifest["artefacts"]}


def build_rows(detection: dict, latency: dict, sizes: dict[str, int] | None = None) -> list[dict]:
    accuracy = accuracy_by_model(detection)
    sizes = sizes or {}
    device = latency["host"]["device_class"]
    rows = []
    for entry in latency["results"]:
        precision = entry["precision"]
        acc = accuracy.get(path_name(entry["model_path"]), {})
        providers = entry.get("providers", [])
        rows.append(
            {
                "variant": entry["name"],
                "backend": entry["backend"],
                "precision": precision,
                "provider": providers[0] if providers else "",
                "device_class": device,
                "map50": acc.get("map50", ""),
                "map50_95": acc.get("map50_95", ""),
                "inference_p50_ms": entry["stages_ms"]["inference"]["p50"],
                "inference_p95_ms": entry["stages_ms"]["inference"]["p95"],
                "end_to_end_p50_ms": entry["end_to_end_ms"]["p50"],
                "fps_end_to_end": entry["fps_end_to_end"],
                "model_bytes": sizes.get(path_name(entry["model_path"]), ""),
            }
        )
    return rows


def markdown(rows: list[dict], skipped: list[dict]) -> str:
    header = (
        "| variant | backend | precision | mAP@0.5 | mAP@0.5:0.95 | "
        "inference p50 | inference p95 | end to end p50 | end-to-end FPS |"
    )
    divider = "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    lines = [header, divider]
    for row in rows:
        map50 = f"{row['map50']:.4f}" if row["map50"] != "" else "n/a"
        map95 = f"{row['map50_95']:.4f}" if row["map50_95"] != "" else "n/a"
        lines.append(
            f"| `{row['variant']}` | {row['backend']} | {row['precision']} | "
            f"{map50} | {map95} | {row['inference_p50_ms']:.2f} ms | "
            f"{row['inference_p95_ms']:.2f} ms | {row['end_to_end_p50_ms']:.2f} ms | "
            f"{row['fps_end_to_end']:.1f} |"
        )
    for entry in skipped:
        lines.append(
            f"| `{entry['name']}` | tensorrt | | not measured | not measured | "
            "not measured | not measured | not measured | not measured |"
        )
    return "\n".join(lines)


def relative_table(rows: list[dict], baseline: str = "fp32") -> str:
    """Cost and benefit of each precision against the FP32 baseline.

    Two baselines, not one.  Accuracy is compared against FP32 accuracy on the
    same artefact family, latency against the FP32 row *on the same backend and
    provider*, because comparing an INT8 CPU time against an FP32 GPU time
    would be arithmetic rather than information.
    """
    by_backend: dict[tuple[str, str], dict[str, dict]] = {}
    for row in rows:
        by_backend.setdefault((row["backend"], row["provider"]), {})[row["variant"]] = row

    lines = [
        "| backend | provider | variant | precision | mAP@0.5:0.95 delta | "
        "inference p50 speedup |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for (backend, provider), group in sorted(by_backend.items()):
        base = next((r for r in group.values() if r["precision"] == baseline), None)
        if base is None:
            continue
        for variant, row in sorted(group.items()):
            precision = row["precision"]
            if row is base:
                continue
            # Five places, not four.  The published deltas in
            # results/published/precision_deltas.json are rounded to six, and a
            # table rounded to four produces a figure that appears in no
            # results file, which scripts/check_docs.py rejects and is right to.
            delta = (
                f"{row['map50_95'] - base['map50_95']:+.5f}"
                if row["map50_95"] != "" and base["map50_95"] != ""
                else "n/a"
            )
            speedup = base["inference_p50_ms"] / max(row["inference_p50_ms"], 1e-9)
            lines.append(
                f"| {backend} | {provider or 'n/a'} | `{variant}` | {precision} | "
                f"{delta} | {speedup:.2f}x |"
            )
    return "\n".join(lines)


def precision_deltas(detection: dict) -> dict:
    """Every accuracy difference the documentation quotes, as a published value.

    A delta is a derived result and belongs in ``results/published`` like any
    other, because ``scripts/check_docs.py`` traces every figure in the prose
    back to a published file.  Computing them here rather than by hand in a
    document is the difference between a number that is checked and a number
    that is typed.
    """
    variants = detection["variants"]
    baseline = next(
        entry for entry in variants.values()
        if entry["detector"]["precision"] == "fp32"
    )
    base = baseline["metrics"]
    out: dict = {
        "baseline": path_name(baseline["model"]),
        "split": detection["split"],
        "variants": {},
    }
    for name, entry in sorted(variants.items()):
        metrics = entry["metrics"]
        if entry is baseline:
            continue
        # per_class_ap50_95 is keyed by the class index as a string, because
        # JSON object keys are strings; per_class_ap50_95_named is the same
        # numbers keyed by name and is what a reader wants.
        per_class = {}
        for index, klass in enumerate(detection["classes"]):
            reference = baseline["per_class_ap50_95_named"].get(klass)
            variant = metrics["per_class_ap50_95"].get(str(index))
            if reference is None or variant is None:
                continue
            per_class[klass] = round(reference - variant, 6)
        out["variants"][name] = {
            "precision": entry["detector"]["precision"],
            "map50_drop": round(base["map50"] - metrics["map50"], 6),
            "map50_95_drop": round(base["map50_95"] - metrics["map50_95"], 6),
            "precision_drop": round(base["precision"] - metrics["precision"], 6),
            "recall_drop": round(base["recall"] - metrics["recall"], 6),
            "per_class_map50_95_drop": per_class,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--published", default="results/published", type=Path)
    parser.add_argument("--csv-out", default="benchmarks/results.csv", type=Path)
    parser.add_argument("--table-dir", default="docs/tables", type=Path)
    parser.add_argument("--deltas-out", default="results/published/precision_deltas.json",
                        type=Path)
    parser.add_argument("--manifest", default="models/MANIFEST.json", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    detection = json.loads((args.published / "detection.json").read_text(encoding="utf-8"))
    latency = json.loads((args.published / "latency.json").read_text(encoding="utf-8"))

    deltas = precision_deltas(detection)
    args.deltas_out.parent.mkdir(parents=True, exist_ok=True)
    args.deltas_out.write_text(json.dumps(deltas, indent=2) + "\n", encoding="utf-8")

    rows = build_rows(detection, latency, artefact_sizes(args.manifest))
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    args.table_dir.mkdir(parents=True, exist_ok=True)
    banner = "<!-- Generated by scripts/make_results_table.py.  Do not edit. -->\n\n"
    (args.table_dir / "headline.md").write_text(
        banner + markdown(rows, latency.get("skipped", [])) + "\n", encoding="utf-8"
    )
    (args.table_dir / "precision_delta.md").write_text(
        banner + relative_table(rows) + "\n", encoding="utf-8"
    )

    tracking_path = args.published / "tracking.json"
    if tracking_path.exists():
        tracking = json.loads(tracking_path.read_text(encoding="utf-8"))
        lines = [
            "| tracker | MOTA | MOTP (IoU) | IDF1 | IDP | IDR | ID switches | "
            "fragmentations | MT | ML | ms/frame p50 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name, entry in tracking["trackers"].items():
            a = entry["aggregate"]
            lines.append(
                f"| {name} | {a['mota']:.4f} | {a['motp_iou']:.4f} | {a['idf1']:.4f} | "
                f"{a['idp']:.4f} | {a['idr']:.4f} | {a['num_switches']} | "
                f"{a['num_fragmentations']} | {a['mostly_tracked']}/{a['num_gt_ids']} | "
                f"{a['mostly_lost']}/{a['num_gt_ids']} | {entry['frame_ms']['p50']:.2f} |"
            )
        (args.table_dir / "tracking.md").write_text(
            banner + "\n".join(lines) + "\n", encoding="utf-8"
        )

    if not args.quiet:
        print(markdown(rows, latency.get("skipped", [])))
        print()
        print(relative_table(rows))
        print(f"\nwrote {args.csv_out} and tables under {args.table_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

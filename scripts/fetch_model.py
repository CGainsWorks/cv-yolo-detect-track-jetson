#!/usr/bin/env python
"""Download the published weights from the GitHub Release named in the manifest.

    python scripts/fetch_model.py            # everything in the manifest
    python scripts/fetch_model.py --only yolov8n_edgetrack_fp32.onnx

Weights live in a Release rather than in Git.  A 12 MB checkpoint committed
once is 12 MB in the pack file forever, and the ONNX and engine artefacts would
be re-added on every retrain.  ``models/MANIFEST.json`` records the URL, size
and SHA-256 of each artefact, and this script refuses anything whose digest
does not match, so a truncated download fails here rather than as a strange
accuracy number three steps later.

The manifest points at the API asset endpoint rather than the browser download
URL, because that is the one that works on a private repository with a token in
``GITHUB_TOKEN``.  On a public repository neither needs authentication.

If the release is unreachable, ``--fallback pretrained`` exports COCO YOLOv8n
instead.  That model detects none of this dataset's classes and is only useful
for exercising the plumbing, so it is opt-in and every artefact it produces is
named ``_coco`` to keep it out of a results run by accident.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import _bootstrap  # noqa: F401


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _headers() -> dict[str, str]:
    """Authorisation, if the environment offers any.

    Release assets on a public repository need none.  While the repository is
    private they do, and a CI job already has ``GITHUB_TOKEN`` in scope, so
    honouring it costs three lines and means the fetch works either side of the
    day the repository is made public.
    """
    headers = {"User-Agent": "edgetrack-fetch", "Accept": "application/octet-stream"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def download(url: str, destination: Path, expected_sha: str, expected_bytes: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)

    actual_bytes = temporary.stat().st_size
    if expected_bytes and actual_bytes != expected_bytes:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"{destination.name}: expected {expected_bytes} bytes, got {actual_bytes}"
        )
    actual_sha = sha256(temporary)
    if expected_sha and actual_sha != expected_sha:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"{destination.name}: sha256 mismatch\n  expected {expected_sha}\n"
            f"  actual   {actual_sha}"
        )
    temporary.replace(destination)


def export_pretrained(out_dir: Path, imgsz: tuple[int, int]) -> Path:
    """Export stock COCO YOLOv8n so the plumbing can be exercised offline."""
    from ultralytics import YOLO

    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO("yolov8n.pt")
    exported = Path(
        model.export(format="onnx", opset=12, simplify=True, imgsz=[imgsz[1], imgsz[0]])
    )
    destination = out_dir / "yolov8n_coco_fp32.onnx"
    exported.replace(destination)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manifest", default="models/MANIFEST.json", type=Path)
    parser.add_argument("--out", default="models", type=Path)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--fallback", choices=["none", "pretrained"], default="none")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    artefacts = manifest["artefacts"]
    if args.only:
        wanted = set(args.only)
        artefacts = [a for a in artefacts if a["name"] in wanted]
        if not artefacts:
            raise ValueError(f"no artefact named in {sorted(wanted)}")

    failures = []
    for artefact in artefacts:
        destination = Path(args.out) / artefact["name"]
        if destination.exists() and not args.force:
            if sha256(destination) == artefact["sha256"]:
                print(f"have  {artefact['name']}")
                continue
            print(f"stale {artefact['name']}, re-downloading")
        try:
            download(
                artefact["url"], destination, artefact["sha256"], int(artefact.get("bytes", 0))
            )
            print(f"got   {artefact['name']} ({destination.stat().st_size} bytes)")
        except (urllib.error.URLError, urllib.error.HTTPError, RuntimeError, TimeoutError) as error:
            failures.append((artefact["name"], str(error)))
            print(f"failed {artefact['name']}: {error}", file=sys.stderr)

    if failures and args.fallback == "pretrained":
        print("falling back to the COCO checkpoint; it detects none of this "
              "dataset's classes", file=sys.stderr)
        path = export_pretrained(Path(args.out), (640, 384))
        print(f"exported {path}")
        return 0
    if failures:
        print(
            f"\n{len(failures)} artefacts could not be fetched.  Rebuild them locally "
            "with `make train export`, or pass --fallback pretrained to exercise the "
            "plumbing with COCO weights.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

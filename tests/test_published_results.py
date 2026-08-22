"""The committed results are portable, self-consistent and honestly labelled.

`results/published/` is the only place a number in the documentation is allowed
to come from, and those files are read on a Linux CI runner after being written
on a Windows desktop.  These tests are the guard on that crossing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from edgetrack.paths import path_name, serialise_path

ROOT = Path(__file__).resolve().parent.parent
PUBLISHED = ROOT / "results" / "published"

PATH_KEYS = ("model", "model_path", "weights", "onnx", "path", "engine", "root",
             "video", "gif", "baseline", "run_dir")
DRIVE_LETTER = re.compile(r"^[A-Za-z]:[\\/]")


def published_files() -> list[Path]:
    return sorted(PUBLISHED.glob("*.json"))


def walk(node, path=""):
    """Yield every (json pointer, value) leaf."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from walk(value, f"{path}/{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from walk(value, f"{path}/{index}")
    else:
        yield path, node


def test_there_are_published_results():
    assert published_files(), "results/published/ is empty; run `make experiments publish`"


@pytest.mark.parametrize("path", published_files(), ids=lambda p: p.name)
def test_no_windows_paths(path):
    """A backslash in a committed path breaks the join on a POSIX runner.

    ``Path("models\\\\x.onnx").name`` is the whole string on Linux, because a
    backslash is a legal filename character there rather than a separator.  The
    results table joins accuracy to latency on the file name, so this failure
    mode is a silent empty join followed by an unexplained CI diff, which is why
    it is checked rather than trusted.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    offenders = [
        (pointer, value)
        for pointer, value in walk(payload)
        if isinstance(value, str) and "\\" in value
    ]
    assert not offenders, f"{path.name} contains Windows separators at {offenders[:5]}"


@pytest.mark.parametrize("path", published_files(), ids=lambda p: p.name)
def test_no_absolute_paths(path):
    """An absolute path records the machine, not the result."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    offenders = [
        (pointer, value)
        for pointer, value in walk(payload)
        if isinstance(value, str)
        and (DRIVE_LETTER.match(value) or value.startswith("/home/") or value.startswith("/Users/"))
    ]
    assert not offenders, f"{path.name} contains absolute paths at {offenders[:5]}"


@pytest.mark.parametrize("path", published_files(), ids=lambda p: p.name)
def test_path_keys_resolve_to_a_file_name(path):
    """Whatever the platform, the last component parses the same way."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    for pointer, value in walk(payload):
        key = pointer.rsplit("/", 1)[-1]
        if key in PATH_KEYS and isinstance(value, str) and value:
            name = path_name(value)
            assert name and "\\" not in name, f"{path.name}{pointer} = {value!r}"


def test_nothing_claims_to_have_come_from_a_jetson():
    """The central honesty check, enforced rather than promised."""
    latency = json.loads((PUBLISHED / "latency.json").read_text(encoding="utf-8"))
    assert latency["host"]["device_class"] == "desktop", latency["host"]
    backends = {entry["backend"] for entry in latency["results"]}
    assert "tensorrt" not in backends, "a TensorRT row is in the published results"
    skipped = {entry["name"] for entry in latency["skipped"]}
    assert {"tensorrt-fp32", "tensorrt-fp16", "tensorrt-int8"} <= skipped, skipped


def test_detection_and_latency_agree_on_the_artefacts():
    """Every accuracy row joins to at least one latency row, and vice versa."""
    detection = json.loads((PUBLISHED / "detection.json").read_text(encoding="utf-8"))
    latency = json.loads((PUBLISHED / "latency.json").read_text(encoding="utf-8"))
    scored = {path_name(entry["model"]) for entry in detection["variants"].values()}
    benched = {path_name(entry["model_path"]) for entry in latency["results"]}
    onnx_benched = {name for name in benched if name.endswith(".onnx")}
    assert onnx_benched <= scored, f"benchmarked but never scored: {onnx_benched - scored}"


def test_the_precision_deltas_match_the_detection_results():
    """The published deltas are derived, so they must still derive."""
    detection = json.loads((PUBLISHED / "detection.json").read_text(encoding="utf-8"))
    deltas = json.loads((PUBLISHED / "precision_deltas.json").read_text(encoding="utf-8"))
    by_name = {
        path_name(entry["model"]): entry["metrics"]
        for entry in detection["variants"].values()
    }
    baseline = by_name[deltas["baseline"]]
    for name, entry in deltas["variants"].items():
        metrics = by_name[f"{name}.onnx"]
        expected = round(baseline["map50_95"] - metrics["map50_95"], 6)
        assert entry["map50_95_drop"] == pytest.approx(expected, abs=1e-9), name


def test_the_tracking_counters_are_internally_consistent():
    tracking = json.loads((PUBLISHED / "tracking.json").read_text(encoding="utf-8"))
    for name, entry in tracking["trackers"].items():
        aggregate = entry["aggregate"]
        assert (
            aggregate["num_detections"]
            == aggregate["num_matches"] + aggregate["num_switches"]
        ), name
        assert (
            aggregate["num_detections"] + aggregate["num_misses"]
            == aggregate["num_objects"]
        ), name
        assert (
            aggregate["mostly_tracked"]
            + aggregate["partially_tracked"]
            + aggregate["mostly_lost"]
            == aggregate["num_gt_ids"]
        ), name
        assert sum(entry["switch_causes"].values()) == aggregate["num_switches"], name


def test_serialise_path_is_repo_relative_and_posix():
    assert serialise_path(ROOT / "models" / "a.onnx") == "models/a.onnx"
    assert serialise_path(None) is None
    # A path outside the repository keeps its form but not its separators.
    assert "\\" not in (serialise_path(Path("/somewhere/else/a.onnx")) or "")

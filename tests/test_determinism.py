"""The dataset regenerates byte for byte from its seeds.

This is the claim the whole repository leans on.  `data/DATASET_CARD.md` says
the set can be rebuilt from a seed, `scripts/run_demo.py` regenerates its
ground truth from the source seed rather than caching it, and every published
number is quoted against data nobody can otherwise obtain.  If generation were
not reproducible, none of that would mean anything.

Two files are excluded from the comparison and the reason is not a fudge.
``data.yaml`` carries an absolute ``path:`` because Ultralytics resolves a
relative one against its own per-user settings directory, and ``dataset.json``
records the output root as provenance.  Both therefore differ between two runs
written to two different directories, by design.  Everything that is actually
data, every image and every label, must match exactly.

This started as a shell step in CI, where it failed on the first run for
precisely that reason.  It lives here now so it fails on a laptop in five
seconds rather than on a runner in four minutes.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Written per run and expected to differ between output directories.
GENERATED_METADATA = {"data.yaml", "dataset.json"}


def generate(root: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_dataset.py"),
         "--preset", "tiny", "--root", str(root), "--quiet"],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]


def digest(root: Path) -> tuple[str, int]:
    """SHA-256 over every data file, in a stable order, with its path."""
    hasher = hashlib.sha256()
    counted = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in GENERATED_METADATA:
            continue
        hasher.update(path.relative_to(root).as_posix().encode())
        hasher.update(path.read_bytes())
        counted += 1
    return hasher.hexdigest(), counted


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory):
    """Generate the same dataset twice, in fresh interpreters.

    Fresh processes rather than two calls in one, because a generator that
    carried state between calls would pass an in-process comparison and still
    be irreproducible from a cold start, which is the case that matters.
    """
    first = tmp_path_factory.mktemp("first")
    second = tmp_path_factory.mktemp("second")
    generate(first)
    generate(second)
    return first, second


def test_the_data_is_identical(two_runs):
    first, second = two_runs
    a, count_a = digest(first)
    b, count_b = digest(second)
    assert count_a == count_b > 0, (count_a, count_b)
    assert a == b, "the same seeds produced different data"


def test_the_two_runs_hold_the_same_files(two_runs):
    first, second = two_runs

    def listing(root: Path) -> set[str]:
        return {
            p.relative_to(root).as_posix()
            for p in root.rglob("*")
            if p.is_file()
        }

    assert listing(first) == listing(second)


def test_only_the_metadata_records_the_output_directory(two_runs):
    """The exclusions are exactly the two files, and nothing else leaks a path.

    Without this the exclusion list could quietly grow to cover a real
    reproducibility failure.
    """
    first, second = two_runs
    differing = set()
    for path in sorted(first.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(first)
        other = second / relative
        if path.read_bytes() != other.read_bytes():
            differing.add(relative.name)
    assert differing <= GENERATED_METADATA, f"unexpected differences in {differing}"


def test_the_metadata_differs_only_in_the_root(two_runs):
    first, second = two_runs
    a = json.loads((first / "dataset.json").read_text(encoding="utf-8"))
    b = json.loads((second / "dataset.json").read_text(encoding="utf-8"))
    # `seconds` is a wall-clock timing and is expected to wobble.
    for payload in (a, b):
        payload.pop("root", None)
        payload.pop("seconds", None)
    assert a == b

    lines_a = (first / "data.yaml").read_text(encoding="utf-8").splitlines()
    lines_b = (second / "data.yaml").read_text(encoding="utf-8").splitlines()
    assert len(lines_a) == len(lines_b)
    differing = [
        (x, y) for x, y in zip(lines_a, lines_b) if x != y
    ]
    assert len(differing) == 1 and differing[0][0].startswith("path:"), differing


def test_the_generated_dataset_is_non_trivial(two_runs):
    first, _ = two_runs
    summary = json.loads((first / "dataset.json").read_text(encoding="utf-8"))
    for split, stats in summary["splits"].items():
        assert stats["images"] > 0, split
        assert stats["labelled_boxes"] > 0, split
    assert (first / "mot" / "test").is_dir()

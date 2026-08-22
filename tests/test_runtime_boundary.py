"""The runtime does not import torch, ultralytics or rclpy.

This is a licence boundary and a size boundary at once.  Ultralytics is
AGPL-3.0; this repository is MIT, and an MIT deployment that imports an AGPL
library at runtime is not something to leave to good intentions.  Torch is
roughly two gigabytes of wheel that the deployed image has no use for once the
model is an ONNX file.  ``docker/Dockerfile.runtime`` builds without either and
the docker CI job asserts the same thing from the outside; this test asserts it
from the inside, where it fails in seconds rather than after an image build.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

RUNTIME_MODULES = [
    "edgetrack",
    "edgetrack.types",
    "edgetrack.pipeline",
    "edgetrack.video",
    "edgetrack.detect",
    "edgetrack.detect.base",
    "edgetrack.detect.onnx_backend",
    "edgetrack.track",
    "edgetrack.track.bytetrack",
    "edgetrack.track.deepsort",
    "edgetrack.track.appearance",
    "edgetrack.metrics",
    "edgetrack.ros2.messages",
]

FORBIDDEN = ("torch", "ultralytics", "rclpy", "matplotlib")


@pytest.mark.parametrize("module", RUNTIME_MODULES)
def test_module_imports_nothing_forbidden(module):
    """Import in a fresh interpreter and check ``sys.modules`` afterwards.

    A fresh process is the only honest way to test this.  Inside the pytest
    process torch is already imported by some other test, and a check against
    the current ``sys.modules`` would pass regardless.
    """
    script = (
        "import sys; sys.path.insert(0, r'{src}');"
        "import {module};"
        "leaked = [n for n in {forbidden} if n in sys.modules];"
        "print(','.join(leaked))"
    ).format(src=ROOT / "src", module=module, forbidden=list(FORBIDDEN))
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0, result.stderr
    leaked = [name for name in result.stdout.strip().split(",") if name]
    assert not leaked, f"importing {module} pulled in {leaked}"


def test_the_reid_network_module_is_not_reachable_from_the_runtime():
    """``edgetrack.reid_net`` imports torch and must stay out of the runtime.

    It is imported only by ``scripts/train_reid.py`` and
    ``scripts/export_reid.py``; the deployed path goes through
    ``OnnxReIDEncoder``.
    """
    script = (
        "import sys; sys.path.insert(0, r'{src}');"
        "import edgetrack.pipeline, edgetrack.track;"
        "print('edgetrack.reid_net' in sys.modules)"
    ).format(src=ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_the_ros2_node_module_says_something_useful_without_ros():
    """Importing the node off ROS must not be a bare ImportError traceback."""
    script = (
        "import sys; sys.path.insert(0, r'{src}');"
        "import edgetrack.ros2.node as node;"
        "print('rclpy' not in sys.modules)"
    ).format(src=ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True"


def test_requirements_do_not_contain_the_training_toolchain():
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    lines = [
        line.split("#", 1)[0].strip().lower()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    for banned in ("torch", "torchvision", "ultralytics"):
        assert not any(line.startswith(banned) for line in lines), (
            f"{banned} is in requirements.txt; it belongs in requirements-train.txt"
        )

"""Shared fixtures and the path bootstrap.

``src`` goes on ``sys.path`` here rather than through an editable install, so
``pytest`` works in a fresh checkout with nothing installed but the
requirements.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT / "src", ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from edgetrack.types import Detection  # noqa: E402


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20240601)


@pytest.fixture
def repo_root() -> Path:
    return ROOT


def make_detection(x: float, y: float, w: float = 20.0, h: float = 50.0,
                   score: float = 0.9, class_id: int = 0,
                   feature: np.ndarray | None = None) -> Detection:
    """Detection from a centre and a size, which is how the tests think."""
    return Detection(
        (x - w / 2, y - h / 2, x + w / 2, y + h / 2), score, class_id, feature
    )


@pytest.fixture
def detection_factory():
    return make_detection


def constant_velocity_track(
    n_frames: int, x0: float, y0: float, vx: float, vy: float, **kwargs
) -> list[Detection]:
    return [
        make_detection(x0 + vx * t, y0 + vy * t, **kwargs) for t in range(n_frames)
    ]


@pytest.fixture
def cv_track():
    return constant_velocity_track


def unit(vector: list[float]) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    return array / np.linalg.norm(array)


@pytest.fixture
def unit_vector():
    return unit

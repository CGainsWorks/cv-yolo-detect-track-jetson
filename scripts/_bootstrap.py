"""Put ``src`` on the path so scripts run without an install step.

Every script starts with ``import _bootstrap  # noqa: F401``.  The alternative
is requiring ``pip install -e .`` before anything works, which is one more
thing to get wrong on a Jetson where the system Python is the only Python.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

"""Path serialisation, so a results file written on Windows reads on Linux.

Every script in this repository writes JSON containing paths, and every one of
those files is committed under ``results/published/`` and read back by
``scripts/make_results_table.py`` and by CI.  A path serialised as
``models\\yolov8n_edgetrack_fp32.onnx`` is not merely ugly on a Linux runner:
``Path(that).name`` returns the whole string, because a backslash is a legal
filename character on POSIX rather than a separator.  The table joins on the
file name, so the join silently finds nothing and the regenerated CSV differs
from the committed one.  That failure surfaces as a diff in CI with no
explanation attached to it.

So every path that reaches a JSON file goes through :func:`serialise_path`
first, and ``tests/test_published_results.py`` fails if a committed results
file contains a backslash or a drive letter.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

__all__ = ["serialise_path", "path_name", "REPO_ROOT"]

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def serialise_path(path: str | os.PathLike[str] | None) -> str | None:
    """Repo-relative POSIX form, falling back to POSIX absolute.

    Relative where possible, because an absolute path in a committed file
    records the machine it was produced on and nothing useful.  Absolute paths
    outside the repository keep their form, with separators normalised, so a
    reader can still tell where an artefact came from.
    """
    if path is None:
        return None
    resolved = Path(path)
    try:
        return resolved.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def path_name(value: str) -> str:
    """The final component of a serialised path, POSIX rules on any platform.

    ``PurePosixPath`` rather than ``Path``: the input is always POSIX by the
    time it is in a results file, and using the platform flavour would make
    this function's answer depend on where it runs, which is the bug it exists
    to prevent.
    """
    return PurePosixPath(value).name

"""Path safety: every file Part 1 reads or writes goes through `safe_join`."""
from __future__ import annotations

import re
from pathlib import Path

from simulator.config import ROOT

DATA_DIR = ROOT / "data"

_DRIVE_RE = re.compile(r"^[A-Za-z]:")


class UnsafePathError(ValueError):
    """Raised when a path would escape its base directory or is malformed."""


def _check_part(part: object) -> str:
    if not isinstance(part, str):
        raise UnsafePathError(f"path part must be a string, got {type(part).__name__}")
    if not part.strip():
        raise UnsafePathError("path part must not be empty")
    if any(ord(c) < 32 or ord(c) == 127 for c in part):
        raise UnsafePathError("path part contains a control character (incl. NUL)")
    if "\\" in part:
        raise UnsafePathError(f"backslash not allowed in path part: {part!r}")
    if part.startswith("/") or _DRIVE_RE.match(part):
        raise UnsafePathError(f"absolute path not allowed: {part!r}")
    if ".." in part.split("/"):
        raise UnsafePathError(f"parent-directory traversal not allowed: {part!r}")
    return part


def safe_join(base: str | Path, *parts: str) -> Path:
    """Join `parts` under `base` and return the resolved path.

    Raises UnsafePathError if any part is absolute, contains `..`, a control
    character or a backslash, or if the resolved result (after following
    symlinks) lies outside `base`. `base` must be an existing directory.
    """
    base_path = Path(base)
    if not base_path.is_dir():
        raise UnsafePathError(f"base is not an existing directory: {base_path}")
    root = base_path.resolve()
    candidate = root.joinpath(*(_check_part(p) for p in parts)).resolve()
    if candidate != root and root not in candidate.parents:
        raise UnsafePathError(f"path escapes base directory: {candidate}")
    return candidate

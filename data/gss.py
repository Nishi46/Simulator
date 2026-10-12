"""Reader for the GSS cumulative file (.dta or .csv).

The caller must supply the file's pinned sha256; the file is hashed and
compared *before* any parser touches it. Only the requested columns are
loaded, and values are returned as-is (special codes are handled in 1.6).
"""
from __future__ import annotations

import csv
import re
import struct
from pathlib import Path
from typing import Iterable

import pandas as pd

from data.download import sha256_file
from data.paths import DATA_DIR, safe_join

SUFFIXES = (".dta", ".csv")
DEFAULT_MAX_BYTES = 1 << 30  # 1 GiB
MAX_COLUMNS = 500
_COL_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class GssReadError(ValueError):
    """Raised when the GSS file is missing, unverified, malformed, or lacks required data."""


def _check_columns(columns: object, numeric: Iterable[str]) -> tuple[list[str], list[str]]:
    if isinstance(columns, str) or not isinstance(columns, (list, tuple)) or not columns:
        raise GssReadError("columns must be a non-empty list of names")
    if len(columns) > MAX_COLUMNS:
        raise GssReadError(f"too many columns requested (max {MAX_COLUMNS})")
    cols: list[str] = []
    for c in columns:
        if not isinstance(c, str) or not _COL_RE.match(c):
            raise GssReadError(f"invalid column name: {c!r}")
        cols.append(c)
    if len(set(cols)) != len(cols):
        raise GssReadError("duplicate names in requested columns")
    num = list(numeric)
    unknown = [c for c in num if c not in cols]
    if unknown:
        raise GssReadError(f"numeric columns not in requested columns: {unknown}")
    return cols, num


def _file_columns_csv(path: Path) -> list[str]:
    with open(path, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f), None)
    if not header:
        raise GssReadError("file is empty")
    return header


def _resolve(requested: list[str], available: list[str]) -> dict[str, str]:
    """Map requested lowercase name -> actual file column name (case-insensitive)."""
    lowered: dict[str, list[str]] = {}
    for a in available:
        lowered.setdefault(a.strip().lower(), []).append(a)
    dupes = sorted(k for k, v in lowered.items() if len(v) > 1 and k in requested)
    if dupes:
        raise GssReadError(f"ambiguous or duplicate column(s) in file: {dupes}")
    missing = [c for c in requested if c not in lowered]
    if missing:
        raise GssReadError(f"required column(s) missing from file: {missing}")
    return {c: lowered[c][0] for c in requested}


def read_gss(
    rel_path: str,
    columns: list[str],
    *,
    sha256: str,
    numeric: Iterable[str] = (),
    max_bytes: int = DEFAULT_MAX_BYTES,
    base_dir: str | Path = DATA_DIR / "raw",
) -> pd.DataFrame:
    """Return a DataFrame with exactly `columns` (lowercase), in the requested order.

    `numeric` lists columns that must hold numbers; stray text in them is an error.
    """
    cols, num = _check_columns(columns, numeric)
    if not isinstance(sha256, str) or not _SHA_RE.match(sha256):
        raise GssReadError("sha256 must be 64 lowercase hex characters")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise GssReadError("max_bytes must be a positive integer")

    path = safe_join(base_dir, rel_path)
    if path.suffix.lower() not in SUFFIXES:
        raise GssReadError(f"unsupported file type {path.suffix!r}; expected one of {SUFFIXES}")
    if not path.is_file():
        raise GssReadError(f"file not found: {rel_path}")
    size = path.stat().st_size
    if size == 0:
        raise GssReadError("file is empty")
    if size > max_bytes:
        raise GssReadError(f"file is {size} bytes, over the {max_bytes} limit")
    if sha256_file(path) != sha256:  # verify BEFORE parsing
        raise GssReadError("sha256 mismatch: refusing to parse unverified file")

    try:
        if path.suffix.lower() == ".dta":
            df = _read_dta(path, cols)
        else:
            df = _read_csv(path, cols)
    except GssReadError:
        raise
    except (ValueError, OSError, EOFError, struct.error, UnicodeDecodeError) as e:  # pandas parser errors subclass ValueError
        raise GssReadError(f"could not parse file ({type(e).__name__})") from None

    if len(df) == 0:
        raise GssReadError("file has no data rows")
    for c in num:
        if not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            raise GssReadError(f"column {c!r} is not numeric")
    return df


def _read_dta(path: Path, cols: list[str]) -> pd.DataFrame:
    with pd.read_stata(path, iterator=True, convert_categoricals=False) as reader:
        mapping = _resolve(cols, list(reader.variable_labels()))
        df = reader.read(columns=[mapping[c] for c in cols])
    df.columns = cols
    return df.reset_index(drop=True)


def _read_csv(path: Path, cols: list[str]) -> pd.DataFrame:
    mapping = _resolve(cols, _file_columns_csv(path))
    df = pd.read_csv(path, usecols=[mapping[c] for c in cols], encoding="utf-8")
    df = df[[mapping[c] for c in cols]]
    df.columns = cols
    return df.reset_index(drop=True)

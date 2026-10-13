"""Gated access to the splits, with an audited, touch-once test set.

train and calibration load freely. The test set needs `allow_test=True`, a
caller label and the run's config hash; the first access is recorded in an
append-only, hash-chained log, and any further access is refused unless
`allow_repeat=True` (which is recorded too).

The chain detects edited, reordered or mid-file-deleted entries. Truncating
the tail of the log, or deleting it, is only detectable against an external
record, so commit the log (or `log_head()`) to git alongside the manifest.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from data.paths import safe_join
from eval.manifest import SPLITS_DIR, DEFAULT_NAME, load_manifest
from eval.splits import SPLIT_NAMES

LOG_NAME = "test_access.log"
GENESIS = "0" * 64
MAX_LOG_BYTES = 10 * 1024 * 1024

_CALLER_RE = re.compile(r"^[A-Za-z0-9_.:/ -]{1,80}$")
_CFG_RE = re.compile(r"^[0-9a-f]{12}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_LINE_KEYS = {"seq", "ts", "kind", "manifest_digest", "config_hash", "caller", "prev"}


class SplitAccessError(ValueError):
    """Base class for access-control failures."""


class SplitLockedError(SplitAccessError):
    """The test set was requested without explicit permission."""


class SplitAlreadyUsedError(SplitAccessError):
    """The test set was already touched for this manifest."""


class AccessLogError(SplitAccessError):
    """The audit log is missing structure, was altered, or cannot be used."""


def _line_hash(line: str) -> str:
    return hashlib.sha256(line.encode()).hexdigest()


def _parse_log(text: str) -> list[dict]:
    entries, prev = [], GENESIS
    if text and not text.endswith("\n"):
        raise AccessLogError("log ends mid-line (truncated or corrupted)")
    for n, line in enumerate(text.splitlines(), 1):
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            raise AccessLogError(f"log line {n} is not valid JSON") from None
        if not isinstance(e, dict) or set(e) != _LINE_KEYS:
            raise AccessLogError(f"log line {n} has unexpected fields")
        if e["seq"] != n or isinstance(e["seq"], bool):
            raise AccessLogError(f"log line {n}: sequence broken")
        if e["prev"] != prev:
            raise AccessLogError(f"log line {n}: hash chain broken (log was altered)")
        if e["kind"] not in ("first", "repeat"):
            raise AccessLogError(f"log line {n}: bad kind")
        if not (isinstance(e["manifest_digest"], str) and _SHA_RE.match(e["manifest_digest"])
                and isinstance(e["config_hash"], str) and _CFG_RE.match(e["config_hash"])
                and isinstance(e["caller"], str) and _CALLER_RE.match(e["caller"])):
            raise AccessLogError(f"log line {n}: malformed field")
        try:
            datetime.strptime(e["ts"], "%Y-%m-%dT%H:%M:%SZ")
        except (TypeError, ValueError):
            raise AccessLogError(f"log line {n}: bad timestamp") from None
        entries.append(e)
        prev = _line_hash(line)
    return entries


def _log_path(base_dir: str | Path) -> Path:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    return safe_join(base, LOG_NAME)


def _read_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    if path.stat().st_size > MAX_LOG_BYTES:
        raise AccessLogError("log is too large")
    try:
        return _parse_log(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        raise AccessLogError("cannot read log") from None


def read_access_log(base_dir: str | Path = SPLITS_DIR) -> list[dict]:
    """Return the verified log entries (raises AccessLogError if altered)."""
    return _read_log(_log_path(base_dir))


def log_head(base_dir: str | Path = SPLITS_DIR) -> str:
    """Hash of the last log line (GENESIS if empty); record it externally to detect truncation."""
    path = _log_path(base_dir)
    if not path.exists():
        return GENESIS
    lines = path.read_text(encoding="utf-8").splitlines()
    _read_log(path)
    return _line_hash(lines[-1]) if lines else GENESIS


def load_split(
    name: str,
    *,
    allow_test: bool = False,
    allow_repeat: bool = False,
    caller: str | None = None,
    run_config_hash: str | None = None,
    base_dir: str | Path = SPLITS_DIR,
    manifest_name: str = DEFAULT_NAME,
    expected_digest: str | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> tuple[str, ...]:
    """Return the sorted respondent ids of a split, enforcing the test-set lock."""
    if not isinstance(name, str) or name not in SPLIT_NAMES:
        raise SplitAccessError(f"split must be one of {SPLIT_NAMES}")
    if name != "test":
        return load_manifest(manifest_name, base_dir=base_dir, expected_digest=expected_digest).ids[name]

    if allow_test is not True:
        raise SplitLockedError("the test set is locked; pass allow_test=True for the final evaluation run")
    if not isinstance(caller, str) or not _CALLER_RE.match(caller):
        raise SplitAccessError("caller must be 1-80 chars of letters, digits and _ . : / space -")
    if not isinstance(run_config_hash, str) or not _CFG_RE.match(run_config_hash):
        raise SplitAccessError("run_config_hash must be 12 lowercase hex characters")
    if allow_repeat is not True and allow_repeat is not False:
        raise SplitAccessError("allow_repeat must be a bool")

    manifest = load_manifest(manifest_name, base_dir=base_dir, expected_digest=expected_digest)
    path = _log_path(base_dir)
    fd = os.open(path, os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)  # serialise "first access" across threads/processes
        size = os.fstat(fd).st_size
        if size > MAX_LOG_BYTES:
            raise AccessLogError("log is too large")
        text = os.pread(fd, size, 0).decode("utf-8", errors="strict") if size else ""
        entries = _parse_log(text)
        prior = [e for e in entries if e["manifest_digest"] == manifest.digest]
        if prior and not allow_repeat:
            raise SplitAlreadyUsedError(
                f"the test set was already accessed {len(prior)} time(s) for this manifest "
                f"(first at {prior[0]['ts']} by {prior[0]['caller']!r})")
        prev = _line_hash(text.splitlines()[-1]) if entries else GENESIS
        entry = {
            "seq": len(entries) + 1, "ts": now().astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "kind": "repeat" if prior else "first", "manifest_digest": manifest.digest,
            "config_hash": run_config_hash, "caller": caller, "prev": prev,
        }
        os.write(fd, (json.dumps(entry, sort_keys=True) + "\n").encode())
        os.fsync(fd)
    except UnicodeDecodeError:
        raise AccessLogError("cannot read log") from None
    finally:
        os.close(fd)  # also releases the lock
    return manifest.ids["test"]


def filter_to_split(table: pd.DataFrame, ids) -> pd.DataFrame:
    """Rows of a long respondent x question table belonging to `ids`."""
    if "respondent_id" not in table.columns:
        raise SplitAccessError("table has no respondent_id column")
    return table[table["respondent_id"].isin(set(ids))]

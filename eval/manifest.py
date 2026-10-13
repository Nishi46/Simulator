"""Split manifest: freezes who is in train / calibration / test.

The manifest holds respondent ids and run metadata only (no demographics, no
answers), so it is safe to commit. Each split carries a sha256 of its ids and
the whole body carries a digest; `load_manifest` re-checks all of them, so a
hand-edited or corrupted manifest fails loudly. To guard against someone
editing the ids *and* the hashes, commit or tag the printed `digest`.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from data.paths import DATA_DIR, safe_join
from eval.splits import SPLIT_NAMES, Splits, check_fractions

SPLITS_DIR = DATA_DIR / "splits"
DEFAULT_NAME = "manifest.json"
VERSION = 1
MAX_BYTES = 100 * 1024 * 1024

_RID_RE = re.compile(r"^\d{4}-\d{1,7}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_CFG_RE = re.compile(r"^[0-9a-f]{12}$")
_COL_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_BODY_KEYS = {"version", "seed", "fractions", "strata_cols", "source_sha256", "config_hash",
              "created_utc", "splits"}
_SPLIT_KEYS = {"n", "sha256", "ids"}


class ManifestError(ValueError):
    """Raised when a manifest is invalid, tampered with, or cannot be written safely."""


@dataclass(frozen=True)
class Manifest:
    seed: int
    fractions: dict[str, float]
    strata_cols: tuple[str, ...]
    source_sha256: str
    config_hash: str
    created_utc: str
    ids: dict[str, tuple[str, ...]]
    digest: str


def ids_sha256(ids) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def _digest(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _check_ids(ids: dict) -> None:
    if set(ids) != set(SPLIT_NAMES):
        raise ManifestError(f"splits must be exactly {SPLIT_NAMES}")
    seen: set[str] = set()
    for name in SPLIT_NAMES:
        lst = ids[name]
        if not lst:
            raise ManifestError(f"split {name!r} is empty")
        if not all(isinstance(i, str) and _RID_RE.match(i) for i in lst):
            raise ManifestError(f"split {name!r} has malformed respondent ids")
        if list(lst) != sorted(lst) or len(set(lst)) != len(lst):
            raise ManifestError(f"split {name!r} ids must be unique and sorted")
        if seen & set(lst):
            raise ManifestError("splits overlap")
        seen |= set(lst)


def _check_meta(seed, fractions, strata_cols, source_sha256, config_hash, created_utc) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ManifestError("seed must be an integer in [0, 2**32)")
    try:
        check_fractions(fractions)
    except ValueError as e:
        raise ManifestError(str(e)) from None
    if (not isinstance(strata_cols, (list, tuple)) or not strata_cols
            or not all(isinstance(c, str) and _COL_RE.match(c) for c in strata_cols)):
        raise ManifestError("strata_cols must be a non-empty list of column names")
    if not isinstance(source_sha256, str) or not _SHA_RE.match(source_sha256):
        raise ManifestError("source_sha256 must be 64 lowercase hex characters")
    if not isinstance(config_hash, str) or not _CFG_RE.match(config_hash):
        raise ManifestError("config_hash must be 12 lowercase hex characters")
    try:
        datetime.strptime(created_utc, "%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError):
        raise ManifestError("created_utc must look like 2026-01-31T12:00:00Z") from None


def _body(seed, fractions, strata_cols, source_sha256, config_hash, created_utc, ids) -> dict:
    return {
        "version": VERSION, "seed": seed,
        "fractions": {k: float(fractions[k]) for k in SPLIT_NAMES},
        "strata_cols": list(strata_cols), "source_sha256": source_sha256,
        "config_hash": config_hash, "created_utc": created_utc,
        "splits": {n: {"n": len(ids[n]), "sha256": ids_sha256(ids[n]), "ids": list(ids[n])}
                   for n in SPLIT_NAMES},
    }


def _path(rel: str, base_dir: str | Path) -> Path:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    p = safe_join(base, rel)
    if p.suffix != ".json":
        raise ManifestError("manifest must be a .json file")
    return p


def write_manifest(
    splits: Splits, *, seed: int, fractions: dict, strata_cols, source_sha256: str, config_hash: str,
    rel_path: str = DEFAULT_NAME, base_dir: str | Path = SPLITS_DIR, force: bool = False,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> tuple[Path, str]:
    """Write the manifest; returns (path, digest). Refuses to overwrite unless force=True."""
    created = now().astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ids = {n: tuple(splits.ids[n]) for n in SPLIT_NAMES if n in splits.ids}
    _check_ids(ids)
    _check_meta(seed, fractions, strata_cols, source_sha256, config_hash, created)
    body = _body(seed, fractions, strata_cols, source_sha256, config_hash, created, ids)
    digest = _digest(body)
    text = json.dumps({**body, "digest": digest}, sort_keys=True, indent=1) + "\n"

    path = _path(rel_path, base_dir)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".part")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.chmod(tmp, 0o644)
        if force:
            os.replace(tmp, path)
        else:
            try:
                os.link(tmp, path)  # atomic, fails if the file already exists
            except FileExistsError:
                raise ManifestError(f"{path.name} already exists; pass force=True to overwrite") from None
    finally:
        tmp.unlink(missing_ok=True)
    return path, digest


def _no_dupes(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise ManifestError(f"duplicate key in manifest: {k!r}")
        d[k] = v
    return d


def _reject_constant(name):
    raise ManifestError(f"invalid JSON constant {name}")


def load_manifest(rel_path: str = DEFAULT_NAME, *, base_dir: str | Path = SPLITS_DIR,
                  expected_digest: str | None = None) -> Manifest:
    """Load and fully verify a manifest. Optionally pin it to an expected digest."""
    path = _path(rel_path, base_dir)
    try:
        if path.stat().st_size > MAX_BYTES:
            raise ManifestError("manifest too large")
        raw = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_dupes,
                         parse_constant=_reject_constant)
    except (OSError, UnicodeDecodeError) as e:
        raise ManifestError(f"cannot read manifest: {type(e).__name__}") from None
    except json.JSONDecodeError:
        raise ManifestError("manifest is not valid JSON") from None

    if not isinstance(raw, dict) or set(raw) != _BODY_KEYS | {"digest"}:
        raise ManifestError("manifest has unexpected or missing fields")
    if raw["version"] != VERSION or isinstance(raw["version"], bool):
        raise ManifestError(f"unsupported manifest version {raw['version']!r}")
    digest = raw.pop("digest")
    if not isinstance(digest, str) or not _SHA_RE.match(digest):
        raise ManifestError("digest is malformed")

    sp = raw["splits"]
    if not isinstance(sp, dict) or set(sp) != set(SPLIT_NAMES):
        raise ManifestError(f"splits must be exactly {SPLIT_NAMES}")
    ids = {}
    for n in SPLIT_NAMES:
        entry = sp[n]
        if not isinstance(entry, dict) or set(entry) != _SPLIT_KEYS or not isinstance(entry["ids"], list):
            raise ManifestError(f"split {n!r} is malformed")
        ids[n] = tuple(entry["ids"])
    _check_ids(ids)
    _check_meta(raw["seed"], raw["fractions"], raw["strata_cols"], raw["source_sha256"],
                raw["config_hash"], raw["created_utc"])
    for n in SPLIT_NAMES:
        if sp[n]["n"] != len(ids[n]) or isinstance(sp[n]["n"], bool):
            raise ManifestError(f"split {n!r}: recorded count does not match its ids")
        if sp[n]["sha256"] != ids_sha256(ids[n]):
            raise ManifestError(f"split {n!r}: id checksum mismatch (manifest was altered)")
    if _digest(raw) != digest:
        raise ManifestError("manifest digest mismatch (manifest was altered)")
    if expected_digest is not None and expected_digest != digest:
        raise ManifestError("manifest digest does not match the pinned value")
    return Manifest(raw["seed"], {k: float(v) for k, v in raw["fractions"].items()},
                    tuple(raw["strata_cols"]), raw["source_sha256"], raw["config_hash"],
                    raw["created_utc"], ids, digest)


def verify_against(manifest: Manifest, *, source_sha256: str | None = None, respondent_ids=None) -> None:
    """Check the manifest still matches the current data (detects data drift)."""
    if source_sha256 is not None and manifest.source_sha256 != source_sha256:
        raise ManifestError("source data checksum differs from the one the splits were made on")
    if respondent_ids is not None:
        current = set(respondent_ids)
        frozen = {i for n in SPLIT_NAMES for i in manifest.ids[n]}
        if current != frozen:
            raise ManifestError(f"respondents differ from the frozen splits "
                                f"({len(current - frozen)} new, {len(frozen - current)} missing)")

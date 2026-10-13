"""`make data`: download -> verify -> extract -> read -> clean -> build -> split -> freeze.

Everything is computed and validated in memory first; files are written only
afterwards, and if a write fails the files this run created are removed, so a
failed run never leaves a half-built processed/ or splits/ directory. Re-running
with unchanged inputs changes nothing. The split manifest is frozen: a re-run
that would produce different splits refuses to overwrite it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from data.build import DEMOGRAPHIC_RAW, PROCESSED_DIR, atomic_write_parquet, build_table, write_table
from data.clean import parse_rules
from data.contamination import STATUSES, flag_questions, load_models
from data.download import DEFAULT_POLICY, Policy, fetch, sha256_file
from data.extract import safe_extract
from data.gss import read_gss
from data.paths import DATA_DIR, safe_join
from data.questions import load_questions
from data.yamlio import parse_strict_yaml
from eval.manifest import DEFAULT_NAME, SPLITS_DIR, load_manifest, write_manifest
from eval.splits import SPLIT_NAMES, check_fractions, make_splits, respondent_frame
from simulator.config import ROOT, config_hash, run_dir

RAW_DIR = DATA_DIR / "raw"
TABLE_NAME = "respondents.parquet"
FLAGS_NAME = "question_flags.parquet"
MAX_CONFIG_BYTES = 100_000
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

_SECTIONS = {
    "name": None,
    "source": {"url", "sha256", "rel_path", "max_bytes", "archive", "member", "max_total", "max_ratio"},
    "data": {"sha256"},
    "inputs": {"questions", "missing_codes", "models"},
    "split": {"seed", "fractions", "strata_cols", "min_stratum_size"},
}


class PipelineError(ValueError):
    """Raised for invalid config or any condition that must stop the pipeline."""


@dataclass(frozen=True)
class PipelineResult:
    table_path: Path
    flags_path: Path
    manifest_path: Path
    manifest_digest: str
    config_hash: str
    summary_path: Path
    manifest_created: bool


# ---- config ----------------------------------------------------------------

def _section(cfg: dict, name: str) -> dict:
    sec = cfg.get(name)
    if not isinstance(sec, dict):
        raise PipelineError(f"config section {name!r} must be a mapping")
    extra = set(sec) - _SECTIONS[name]
    if extra:
        raise PipelineError(f"unknown key(s) in {name!r}: {sorted(map(str, extra))}")
    return sec


def load_pipeline_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    try:
        if p.stat().st_size > MAX_CONFIG_BYTES:
            raise PipelineError("config file too large")
        cfg = parse_strict_yaml(p.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as e:
        raise PipelineError(f"cannot read config: {type(e).__name__}") from None
    except ValueError as e:
        if isinstance(e, PipelineError):
            raise
        raise PipelineError(str(e)) from None
    return validate_config(cfg)


def validate_config(cfg: object) -> dict[str, Any]:
    if not isinstance(cfg, dict) or set(cfg) != set(_SECTIONS):
        raise PipelineError(f"config must have exactly the sections {sorted(_SECTIONS)}")
    if not isinstance(cfg["name"], str) or not _NAME_RE.match(cfg["name"]):
        raise PipelineError("name must be lowercase letters, digits and underscores")
    src, data, inp, spl = (_section(cfg, k) for k in ("source", "data", "inputs", "split"))
    for k in ("url", "rel_path", "max_bytes", "archive"):
        if k not in src:
            raise PipelineError(f"source.{k} is required")
    if not isinstance(src["url"], str) or not isinstance(src["rel_path"], str):
        raise PipelineError("source.url and source.rel_path must be strings")
    for k in ("max_bytes", "max_total", "max_ratio"):
        if k in src and (isinstance(src[k], bool) or not isinstance(src[k], int) or src[k] <= 0):
            raise PipelineError(f"source.{k} must be a positive integer")
    if not isinstance(src["archive"], bool):
        raise PipelineError("source.archive must be true or false")
    if src["archive"]:
        if not isinstance(src.get("member"), str) or "max_total" not in src:
            raise PipelineError("archive sources need source.member and source.max_total")
    elif "member" in src:
        raise PipelineError("source.member only applies to archives")
    for sec, key in ((src, "sha256"), (data, "sha256")):
        v = sec.get(key)
        if v is not None and not (isinstance(v, str) and _SHA_RE.match(v)):
            raise PipelineError(f"{key} must be null or 64 lowercase hex characters")
    for k in ("questions", "missing_codes", "models"):
        if not isinstance(inp.get(k), str):
            raise PipelineError(f"inputs.{k} is required")
    for k in ("seed", "fractions", "strata_cols"):
        if k not in spl:
            raise PipelineError(f"split.{k} is required")
    if isinstance(spl["seed"], bool) or not isinstance(spl["seed"], int):
        raise PipelineError("split.seed must be an integer")
    if not isinstance(spl["strata_cols"], list) or not spl["strata_cols"]:
        raise PipelineError("split.strata_cols must be a non-empty list")
    try:
        check_fractions(spl["fractions"])
    except ValueError as e:
        raise PipelineError(str(e)) from None
    return cfg


# ---- helpers ----------------------------------------------------------------

def _load_rules(path: Path):
    try:
        if path.stat().st_size > MAX_CONFIG_BYTES:
            raise PipelineError("missing-codes file too large")
        return parse_rules(parse_strict_yaml(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError) as e:
        raise PipelineError(f"cannot read {path.name}: {type(e).__name__}") from None


def _acquire_data_file(cfg: dict, raw_dir: Path, policy: Policy) -> tuple[str, str]:
    """Download (and extract) as needed; return (rel path of data file, its sha256)."""
    src, data = cfg["source"], cfg["data"]
    if src.get("sha256") is None:
        raise PipelineError("source.sha256 is not pinned; review the source and pin its checksum first")
    if src["archive"] and data.get("sha256") is None:
        raise PipelineError("data.sha256 is not pinned; pin the extracted data file's checksum first")
    fetch(src["url"], src["rel_path"], src["sha256"], src["max_bytes"], base_dir=raw_dir, policy=policy)
    if not src["archive"]:
        return src["rel_path"], data.get("sha256") or src["sha256"]
    out_rel = Path(src["rel_path"]).with_suffix("").as_posix() + "_extracted"
    data_rel = f"{out_rel}/{src['member']}"
    out_dir = safe_join(raw_dir, out_rel)
    if out_dir.exists():
        member = safe_join(raw_dir, data_rel)
        if not member.is_file() or sha256_file(member) != data["sha256"]:
            raise PipelineError(f"{out_rel} exists but does not hold the pinned data file; remove it and re-run")
    else:
        safe_extract(src["rel_path"], out_rel, max_total=src["max_total"],
                     max_ratio=src.get("max_ratio", 100), base_dir=raw_dir)
    return data_rel, data["sha256"]


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".part")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


# ---- the pipeline -----------------------------------------------------------

def run_pipeline(
    config_path: str | Path | dict,
    *,
    raw_dir: str | Path = RAW_DIR,
    processed_dir: str | Path = PROCESSED_DIR,
    splits_dir: str | Path = SPLITS_DIR,
    run_base: str | Path = ROOT / "runs",
    policy: Policy = DEFAULT_POLICY,
) -> PipelineResult:
    cfg = validate_config(config_path) if isinstance(config_path, dict) else load_pipeline_config(config_path)
    chash = config_hash(cfg)
    raw_dir, processed_dir, splits_dir = Path(raw_dir), Path(processed_dir), Path(splits_dir)

    questions = load_questions(safe_join(ROOT, cfg["inputs"]["questions"]))
    rules = _load_rules(safe_join(ROOT, cfg["inputs"]["missing_codes"]))
    models, lag = load_models(safe_join(ROOT, cfg["inputs"]["models"]))

    # -- acquire, verify, read ------------------------------------------------
    data_rel, data_sha = _acquire_data_file(cfg, raw_dir, policy)
    columns = list(dict.fromkeys(["id", "year", "wtssall", *DEMOGRAPHIC_RAW, *(q.gss_variable for q in questions)]))
    raw = read_gss(data_rel, columns, sha256=data_sha, numeric=columns, base_dir=raw_dir)

    # -- compute everything in memory ------------------------------------------
    table, summaries = build_table(raw, questions, rules)
    flags = flag_questions(questions, models, lag)
    spl = cfg["split"]
    splits = make_splits(respondent_frame(table, spl["strata_cols"]), spl["seed"], spl["fractions"],
                         spl["strata_cols"], spl.get("min_stratum_size", 10))
    split_hash = config_hash({"name": cfg["name"], "split": spl, "data_sha256": data_sha})

    manifest_path = splits_dir.resolve() / DEFAULT_NAME
    reuse = manifest_path.exists()
    if reuse:
        m = load_manifest(base_dir=splits_dir)
        same = (m.ids == splits.ids and m.seed == spl["seed"] and m.strata_cols == tuple(spl["strata_cols"])
                and m.source_sha256 == data_sha and m.config_hash == split_hash
                and m.fractions == {k: float(v) for k, v in spl["fractions"].items()})
        if not same:
            raise PipelineError("an existing split manifest differs from what these inputs produce; "
                                "splits are frozen, so refusing to overwrite it")

    # -- write (rolling back anything this run created if a write fails) --------
    table_path = Path(processed_dir) / TABLE_NAME
    flags_path = Path(processed_dir) / FLAGS_NAME
    created = [p for p in (table_path, flags_path, manifest_path) if not p.exists()]
    try:
        table_path = write_table(table, questions, TABLE_NAME, base_dir=processed_dir)
        atomic_write_parquet(_check_flags(flags), safe_join(_mk(processed_dir), FLAGS_NAME))
        if reuse:
            digest = m.digest
        else:
            manifest_path, digest = write_manifest(
                splits, seed=spl["seed"], fractions=spl["fractions"], strata_cols=spl["strata_cols"],
                source_sha256=data_sha, config_hash=split_hash, base_dir=splits_dir)
    except BaseException:
        for p in created:
            p.unlink(missing_ok=True)
        raise

    summary = {
        "config_hash": chash, "manifest_digest": digest, "data_sha256": data_sha,
        "n_rows": len(table), "n_respondents": int(table["respondent_id"].nunique()),
        "split_sizes": {n: len(splits.ids[n]) for n in SPLIT_NAMES},
        "questions": {qid: asdict(s) for qid, s in summaries.items()},
    }
    summary_path = run_dir(cfg, base=run_base) / "pipeline.json"
    _atomic_text(summary_path, json.dumps(summary, sort_keys=True, indent=1) + "\n")
    return PipelineResult(table_path, flags_path, manifest_path, digest, chash, summary_path, not reuse)


def _mk(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


def _check_flags(flags: pd.DataFrame) -> pd.DataFrame:
    if flags.empty or not set(flags["status"].astype(str)) <= set(STATUSES) or flags.isna().any().any():
        raise PipelineError("contamination flags are incomplete")
    return flags


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the frozen dataset and splits.")
    ap.add_argument("config", nargs="?", default=str(ROOT / "config" / "data.yaml"))
    args = ap.parse_args(argv)
    try:
        r = run_pipeline(args.config)
    except ValueError as e:  # all our errors derive from ValueError
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"table:    {r.table_path}\nflags:    {r.flags_path}\nmanifest: {r.manifest_path}"
          f"{'' if r.manifest_created else ' (unchanged)'}\ndigest:   {r.manifest_digest}\n"
          f"config:   {r.config_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

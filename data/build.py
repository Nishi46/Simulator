"""Assemble the tidy respondent x question table and persist it as parquet.

One row per (respondent, question) with the answer index, weight and
harmonised demographics. `validate_schema` runs before every write and after
every read, so a malformed or tampered table can't reach the modelling steps.
"""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from data import demographics as demo
from data.clean import Rule, normalize_missing
from data.outcomes import OutcomeSummary, extract
from data.paths import DATA_DIR, safe_join
from data.questions import Question

PROCESSED_DIR = DATA_DIR / "processed"
REQUIRED_COLUMNS = ("respondent_id", "year", "question_id", "answer", "weight")
TABLE_COLUMNS = REQUIRED_COLUMNS + demo.DEMOGRAPHIC_COLUMNS
MAX_ID = 10**7
_RID_RE = re.compile(r"^\d{4}-\d{1,7}$")

# raw GSS columns the demographic harmoniser reads
DEMOGRAPHIC_RAW = ("age", "sex", "race", "degree", "realinc", "region", "partyid", "srcbelt")


class SchemaError(ValueError):
    """Raised when a table does not match the expected schema."""


class BuildError(ValueError):
    """Raised when raw inputs cannot be assembled into a table."""


def _expected_categories() -> dict[str, tuple[list[str], bool]]:
    exp = {"age_band": (list(demo.AGE_LABELS), True), "income_band": (list(demo.INCOME_LABELS), True)}
    for name, (_, _, cats, ordered) in demo._CODED.items():
        exp[name] = (list(cats), ordered)
    return exp


def validate_schema(df: pd.DataFrame, questions: Iterable[Question]) -> None:
    """Raise SchemaError unless `df` is a well-formed table for these questions."""
    qmap = {q.id: q for q in questions}
    if not isinstance(df, pd.DataFrame):
        raise SchemaError("table must be a DataFrame")
    if list(df.columns) != list(TABLE_COLUMNS):
        raise SchemaError(f"columns must be exactly {list(TABLE_COLUMNS)}, got {list(df.columns)}")
    if len(df) == 0:
        raise SchemaError("table is empty")

    rid = df["respondent_id"]
    if not (pd.api.types.is_string_dtype(rid) or rid.dtype == object):
        raise SchemaError("respondent_id must be a string column")
    if rid.isna().any() or not rid.astype(str).map(lambda s: bool(_RID_RE.match(s))).all():
        raise SchemaError("respondent_id must look like '<year>-<id>'")
    qid = df["question_id"]
    if not (pd.api.types.is_string_dtype(qid) or qid.dtype == object) or qid.isna().any():
        raise SchemaError("question_id must be a non-null string column")
    unknown = sorted(set(qid) - set(qmap))
    if unknown:
        raise SchemaError(f"question_id not in registry: {unknown}")

    for col in ("year", "answer"):
        if df[col].dtype != "int64":
            raise SchemaError(f"{col} must be int64, got {df[col].dtype}")
    if df["weight"].dtype != "float64":
        raise SchemaError(f"weight must be float64, got {df['weight'].dtype}")
    if df[list(REQUIRED_COLUMNS)].isna().any().any():
        raise SchemaError("required columns must not contain missing values")
    w = df["weight"].to_numpy()
    if not (np.isfinite(w).all() and (w > 0).all()):
        raise SchemaError("weight must be finite and > 0")

    if df.duplicated(["respondent_id", "question_id"]).any():
        raise SchemaError("duplicate (respondent_id, question_id) rows")

    n_opts = qid.map({k: len(q.options) for k, q in qmap.items()}).to_numpy()
    ans = df["answer"].to_numpy()
    if ((ans < 0) | (ans >= n_opts)).any():
        raise SchemaError("answer index outside the question's options")
    lo = qid.map({k: q.years[0] for k, q in qmap.items()}).to_numpy()
    hi = qid.map({k: q.years[1] for k, q in qmap.items()}).to_numpy()
    yr = df["year"].to_numpy()
    if ((yr < lo) | (yr > hi)).any():
        raise SchemaError("year outside the question's year range")
    if (rid.astype(str).str[:4].astype(int).to_numpy() != yr).any():
        raise SchemaError("respondent_id year does not match the year column")

    for col, (cats, ordered) in _expected_categories().items():
        dt = df[col].dtype
        if not isinstance(dt, pd.CategoricalDtype) or list(dt.categories) != cats or bool(dt.ordered) != ordered:
            raise SchemaError(f"{col} must be a categorical with exactly {cats} (ordered={ordered})")
    for col in demo.DEMOGRAPHIC_COLUMNS:
        if (df.groupby("respondent_id", observed=True)[col].nunique() > 1).any():
            raise SchemaError(f"{col} differs between rows of the same respondent")


def build_table(
    raw: pd.DataFrame,
    questions: Iterable[Question],
    rules: Iterable[Rule],
    *,
    id_col: str = "id",
    year_col: str = "year",
    weight_col: str = "wtssall",
    deflators=None,
) -> tuple[pd.DataFrame, dict[str, OutcomeSummary]]:
    """Clean, harmonise and extract every question; return (table, summaries)."""
    questions, rules = list(questions), list(rules)
    if not questions:
        raise BuildError("no questions given")
    ruled = {r.column for r in rules}
    needed = set(DEMOGRAPHIC_RAW) | {q.gss_variable for q in questions}
    uncovered = sorted(needed - ruled)
    if uncovered:
        raise BuildError(f"no cleaning rule for column(s): {uncovered}")
    for col in (id_col, year_col, weight_col):
        if col not in raw.columns:
            raise BuildError(f"required column missing: {col!r}")
    ids = pd.to_numeric(raw[id_col], errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
    if np.isnan(ids).any() or (ids != np.floor(ids)).any() or (ids < 0).any() or (ids >= MAX_ID).any():
        raise BuildError(f"{id_col!r} must be whole numbers in [0, {MAX_ID})")
    if raw.duplicated([year_col, id_col]).any():
        raise BuildError(f"duplicate ({year_col}, {id_col}) respondents in the input")

    cleaned, _ = normalize_missing(raw, rules)
    cleaned = cleaned.reset_index(drop=True)
    demographics = demo.harmonize(cleaned, deflators=deflators)
    years = cleaned[year_col].to_numpy()
    rid = pd.Series([f"{int(y)}-{int(i)}" for y, i in zip(years, ids)], index=cleaned.index, dtype="object")

    parts, summaries = [], {}
    for q in questions:
        out, summary = extract(cleaned, q, year_col=year_col, weight_col=weight_col)
        part = pd.DataFrame({
            "respondent_id": rid.loc[out.index].to_numpy(),
            "year": out["year"].to_numpy(),
            "question_id": q.id,
            "answer": out["answer"].to_numpy(),
            "weight": out["weight"].to_numpy(),
        })
        part = pd.concat([part, demographics.loc[out.index].reset_index(drop=True)], axis=1)
        parts.append(part)
        summaries[q.id] = summary

    table = pd.concat(parts, ignore_index=True)
    for col, (cats, ordered) in _expected_categories().items():  # concat can drop categorical dtype
        table[col] = pd.Categorical(table[col].astype(object), categories=cats, ordered=ordered)
    table["respondent_id"] = table["respondent_id"].astype(str)
    table["question_id"] = table["question_id"].astype(str)
    table = table.sort_values(["question_id", "respondent_id"], kind="stable").reset_index(drop=True)
    validate_schema(table, questions)
    return table, summaries


def _target(rel_path: str, base_dir: str | Path) -> Path:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    path = safe_join(base, rel_path)
    if path.suffix != ".parquet":
        raise BuildError("output must be a .parquet file")
    return path


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write via a temp file and rename, so `path` is never partially written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".part")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        df.to_parquet(tmp, index=False, engine="pyarrow")
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_table(df: pd.DataFrame, questions: Iterable[Question], rel_path: str,
                *, base_dir: str | Path = PROCESSED_DIR) -> Path:
    """Validate, then write atomically. Nothing is written if validation fails."""
    questions = list(questions)
    validate_schema(df, questions)
    path = _target(rel_path, base_dir)
    atomic_write_parquet(df, path)
    return path


def read_table(questions: Iterable[Question], rel_path: str, *, base_dir: str | Path = PROCESSED_DIR) -> pd.DataFrame:
    """Read a table back and re-validate it (catches corruption or tampering)."""
    path = _target(rel_path, base_dir)
    if not path.is_file():
        raise BuildError(f"table not found: {rel_path}")
    try:
        df = pd.read_parquet(path, engine="pyarrow")
    except Exception as e:  # pyarrow raises several unrelated error types
        raise SchemaError(f"could not read parquet ({type(e).__name__})") from None
    validate_schema(df, list(questions))
    return df

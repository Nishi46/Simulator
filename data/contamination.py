"""Contamination flags: could a model have seen a question's data?

A question's data for survey year Y is assumed public `release_lag_months`
after Y ends. Per (question, model), by the model's training cutoff:

  likely_seen    all of the question's data was public before the cutoff
  possibly_seen  the year range straddles the cutoff
  post_cutoff    none of it was public before the cutoff

This is a coarse heuristic: it says nothing about whether the specific
numbers were widely published, so the human-written `contamination_note` is
carried through in the table for that judgement.
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from data.questions import Question
from data.yamlio import YamlError, parse_strict_yaml
from simulator.config import ROOT

DEFAULT_PATH = ROOT / "config" / "model_cutoffs.yaml"
STATUSES = ("likely_seen", "possibly_seen", "post_cutoff")
DEFAULT_LAG_MONTHS = 12
MAX_LAG_MONTHS = 60
MAX_MODELS = 50
MAX_FILE_BYTES = 100_000
CUTOFF_MIN, CUTOFF_MAX = date(1990, 1, 1), date(2100, 12, 31)

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,79}$")
_MONTH_RE = re.compile(r"^(\d{4})-(\d{2})$")
_DAY_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")


class ContaminationError(ValueError):
    """Raised for invalid cutoff configuration or inputs."""


@dataclass(frozen=True)
class ModelCutoff:
    name: str
    cutoff: date


def parse_cutoff(value: Any) -> date:
    """'YYYY-MM-DD', a date, or 'YYYY-MM' (meaning the end of that month)."""
    if isinstance(value, datetime) or isinstance(value, bool):
        raise ContaminationError(f"cutoff must be a date, got {value!r}")
    try:
        if isinstance(value, date):
            d = value
        elif isinstance(value, str) and (m := _DAY_RE.match(value)):
            d = date(int(m[1]), int(m[2]), int(m[3]))
        elif isinstance(value, str) and (m := _MONTH_RE.match(value)):
            y, mo = int(m[1]), int(m[2])
            d = date(y, mo, calendar.monthrange(y, mo)[1])
        else:
            raise ContaminationError(f"cutoff must look like YYYY-MM or YYYY-MM-DD, got {value!r}")
    except ValueError:
        raise ContaminationError(f"cutoff is not a real date: {value!r}") from None
    if not CUTOFF_MIN <= d <= CUTOFF_MAX:
        raise ContaminationError(f"cutoff {d} outside {CUTOFF_MIN.year}-{CUTOFF_MAX.year}")
    return d


def release_date(year: int, lag_months: int) -> date:
    """Last day of the month `lag_months` after December of `year`."""
    total = year * 12 + 11 + lag_months  # month index of December + lag
    y, m = divmod(total, 12)
    return date(y, m + 1, calendar.monthrange(y, m + 1)[1])


def _check_lag(lag: Any) -> int:
    if isinstance(lag, bool) or not isinstance(lag, int) or not 0 <= lag <= MAX_LAG_MONTHS:
        raise ContaminationError(f"release_lag_months must be an integer 0-{MAX_LAG_MONTHS}")
    return lag


def classify(question: Question, cutoff: date, lag_months: int = DEFAULT_LAG_MONTHS) -> str:
    lag = _check_lag(lag_months)
    if not isinstance(cutoff, date) or isinstance(cutoff, datetime):
        raise ContaminationError("cutoff must be a date")
    start_rel = release_date(question.years[0], lag)
    end_rel = release_date(question.years[1], lag)
    if end_rel <= cutoff:
        return "likely_seen"
    if start_rel > cutoff:
        return "post_cutoff"
    return "possibly_seen"


def load_models(path: str | Path = DEFAULT_PATH) -> tuple[list[ModelCutoff], int]:
    """Return (models, release_lag_months) from the config file."""
    path = Path(path)
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ContaminationError(f"{path} exceeds {MAX_FILE_BYTES} bytes")
        data = parse_strict_yaml(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as e:
        raise ContaminationError(f"cannot read {path}: {e}") from e
    except YamlError as e:
        raise ContaminationError(str(e)) from e
    if not isinstance(data, dict) or not set(data) <= {"models", "release_lag_months"} or "models" not in data:
        raise ContaminationError("top level must be a mapping with 'models' and optional 'release_lag_months'")
    lag = _check_lag(data.get("release_lag_months", DEFAULT_LAG_MONTHS))
    raw = data["models"]
    if not isinstance(raw, list) or not raw:
        raise ContaminationError("'models' must be a non-empty list")
    if len(raw) > MAX_MODELS:
        raise ContaminationError(f"too many models (max {MAX_MODELS})")
    models, names = [], set()
    for i, m in enumerate(raw):
        if not isinstance(m, dict) or set(m) != {"name", "cutoff"}:
            raise ContaminationError(f"model[{i}] must have exactly 'name' and 'cutoff'")
        name = m["name"]
        if not isinstance(name, str) or not _NAME_RE.match(name):
            raise ContaminationError(f"model[{i}]: invalid name {name!r}")
        if name in names:
            raise ContaminationError(f"duplicate model name: {name}")
        names.add(name)
        models.append(ModelCutoff(name, parse_cutoff(m["cutoff"])))
    return models, lag


def flag_questions(questions: Iterable[Question], models: Iterable[ModelCutoff],
                   lag_months: int = DEFAULT_LAG_MONTHS) -> pd.DataFrame:
    """One row per (question, model): never missing, status always in STATUSES."""
    qs, ms = list(questions), list(models)
    if not qs or not ms:
        raise ContaminationError("need at least one question and one model")
    rows = [
        {"question_id": q.id, "model": m.name, "cutoff": m.cutoff,
         "years": f"{q.years[0]}-{q.years[1]}", "status": classify(q, m.cutoff, lag_months),
         "note": q.contamination_note}
        for q in qs for m in ms
    ]
    df = pd.DataFrame(rows)
    df["status"] = pd.Categorical(df["status"], categories=list(STATUSES))
    return df

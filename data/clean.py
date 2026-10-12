"""Special-code and missing-value handling.

Survey files encode "inapplicable", "don't know", "no answer" etc. as numbers.
Each column gets an explicit `Rule`: which values are real answers, which are
missing codes (and what they mean). Anything else is an error, never silently
passed through as if it were data.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

_COL_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 _/-]{0,39}$")
MAX_SHOWN = 10


class CleanError(ValueError):
    """Raised for invalid rules or data that violates them."""


def _num(x: Any, what: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float, np.integer, np.floating)) or not math.isfinite(x):
        raise CleanError(f"{what} must be a finite number, got {x!r}")
    return float(x)


@dataclass(frozen=True)
class Rule:
    column: str
    missing: Mapping[float, str] = field(default_factory=dict)  # code -> meaning (e.g. 99: "NA")
    valid_values: tuple[float, ...] = ()
    valid_range: tuple[float, float] | None = None  # inclusive

    def __post_init__(self):
        if not isinstance(self.column, str) or not _COL_RE.match(self.column):
            raise CleanError(f"invalid column name: {self.column!r}")
        missing = {}
        for code, label in dict(self.missing).items():
            c = _num(code, f"{self.column}: missing code")
            if c in missing:
                raise CleanError(f"{self.column}: duplicate missing code {c}")
            if not isinstance(label, str) or not _LABEL_RE.match(label):
                raise CleanError(f"{self.column}: invalid label for code {c}: {label!r}")
            missing[c] = label
        values = tuple(sorted({_num(v, f"{self.column}: valid value") for v in self.valid_values}))
        rng = None
        if self.valid_range is not None:
            if not isinstance(self.valid_range, (tuple, list)) or len(self.valid_range) != 2:
                raise CleanError(f"{self.column}: valid_range must be (low, high)")
            lo, hi = (_num(v, f"{self.column}: valid_range") for v in self.valid_range)
            if lo > hi:
                raise CleanError(f"{self.column}: valid_range low {lo} is above high {hi}")
            rng = (lo, hi)
        if not values and rng is None:
            raise CleanError(f"{self.column}: rule needs valid_values or valid_range")
        for c in missing:
            if c in values or (rng and rng[0] <= c <= rng[1]):
                raise CleanError(f"{self.column}: code {c} is both valid and missing")
        object.__setattr__(self, "missing", missing)
        object.__setattr__(self, "valid_values", values)
        object.__setattr__(self, "valid_range", rng)


@dataclass(frozen=True)
class ColumnReport:
    column: str
    n_rows: int
    n_valid: int
    n_already_missing: int
    n_coded_missing: dict[str, int]  # meaning -> count of cells converted to NaN


def parse_rules(spec: object) -> list[Rule]:
    """Build rules from a plain mapping, e.g. loaded from YAML:
    {col: {missing: {99: "NA"}, valid_values: [1, 2], valid_range: [18, 89]}}"""
    if not isinstance(spec, dict) or not spec:
        raise CleanError("rules spec must be a non-empty mapping of column -> rule")
    rules = []
    for col, body in spec.items():
        if not isinstance(body, dict):
            raise CleanError(f"{col}: rule must be a mapping")
        extra = set(body) - {"missing", "valid_values", "valid_range"}
        if extra:
            raise CleanError(f"{col}: unknown rule key(s): {sorted(map(str, extra))}")
        rules.append(Rule(
            column=col,
            missing=body.get("missing") or {},
            valid_values=tuple(body.get("valid_values") or ()),
            valid_range=tuple(body["valid_range"]) if body.get("valid_range") is not None else None,
        ))
    return rules


def _in_valid(values: np.ndarray, rule: Rule) -> np.ndarray:
    ok = np.zeros(values.shape, dtype=bool)
    if rule.valid_values:
        ok |= np.isin(values, np.array(rule.valid_values))
    if rule.valid_range is not None:
        lo, hi = rule.valid_range
        ok |= (values >= lo) & (values <= hi)
    return ok


def normalize_missing(df: pd.DataFrame, rules: Iterable[Rule]) -> tuple[pd.DataFrame, dict[str, ColumnReport]]:
    """Return (cleaned copy, per-column report). Columns without a rule pass through unchanged.

    Raises CleanError for a rule whose column is absent or non-numeric, and for
    any value that is neither valid nor a declared missing code.
    """
    rules = list(rules)
    names = [r.column for r in rules]
    if len(set(names)) != len(names):
        raise CleanError("more than one rule for the same column")
    out = df.copy()
    reports: dict[str, ColumnReport] = {}
    for rule in rules:
        if rule.column not in out.columns:
            raise CleanError(f"rule for column {rule.column!r} but it is not in the data")
        s = out[rule.column]
        if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            raise CleanError(f"column {rule.column!r} is not numeric (dtype {s.dtype})")
        values = np.array(s.to_numpy(dtype="float64", na_value=np.nan), copy=True)  # writable copy
        already = np.isnan(values)
        present = ~already

        coded = np.zeros(values.shape, dtype=bool)
        counts: dict[str, int] = {}
        for code, label in rule.missing.items():
            hit = present & (values == code)
            if hit.any():
                counts[label] = counts.get(label, 0) + int(hit.sum())
            coded |= hit

        valid = present & ~coded & _in_valid(values, rule)
        unknown = present & ~coded & ~valid
        if unknown.any():
            vals, n = np.unique(values[unknown], return_counts=True)
            shown = ", ".join(f"{v:g} (x{c})" for v, c in list(zip(vals, n))[:MAX_SHOWN])
            more = f" and {len(vals) - MAX_SHOWN} more" if len(vals) > MAX_SHOWN else ""
            raise CleanError(f"{rule.column}: unrecognised value(s): {shown}{more}")

        values[coded] = np.nan
        out[rule.column] = values
        reports[rule.column] = ColumnReport(
            rule.column, len(values), int(valid.sum()), int(already.sum()), counts
        )
    return out, reports


def drop_incomplete(df: pd.DataFrame, columns: Iterable[str]) -> tuple[pd.DataFrame, int]:
    """Drop rows with NaN in any of `columns`; return (frame, number of rows dropped)."""
    cols = list(columns)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise CleanError(f"columns not in data: {missing}")
    kept = df.dropna(subset=cols)
    return kept, len(df) - len(kept)

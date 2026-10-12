"""Per-question outcome extraction.

Takes the cleaned frame (special codes already NaN) and one registry Question
and returns a small frame of (year, answer index, weight) for usable
respondents plus a summary of everything that was dropped or flagged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.questions import Question

DEFAULT_MIN_OPTION_COUNT = 30
DEFAULT_MIN_ROWS = 200


class OutcomeError(ValueError):
    """Raised when the data cannot yield a valid outcome for the question."""


@dataclass(frozen=True)
class OutcomeSummary:
    question_id: str
    n_input: int
    n_outside_years: int
    n_missing_answer: int
    n_bad_weight: int  # NaN or zero weight, dropped
    n_kept: int
    counts: tuple[int, ...]          # unweighted, per option in registry order
    weighted_share: tuple[float, ...]
    unobserved_options: tuple[str, ...]
    rare_options: tuple[str, ...]    # observed but fewer than min_option_count
    flags: tuple[str, ...]           # human-readable warnings


def _numeric(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        raise OutcomeError(f"required column missing: {col!r}")
    s = df[col]
    if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        raise OutcomeError(f"column {col!r} is not numeric (dtype {s.dtype})")
    return np.array(s.to_numpy(dtype="float64", na_value=np.nan), copy=True)


def extract(
    df: pd.DataFrame,
    question: Question,
    *,
    year_col: str = "year",
    weight_col: str = "wtssall",
    min_option_count: int = DEFAULT_MIN_OPTION_COUNT,
    min_rows: int = DEFAULT_MIN_ROWS,
) -> tuple[pd.DataFrame, OutcomeSummary]:
    """Return (frame with columns year, answer, weight; summary).

    `answer` is the index into question.options (registry order, so ordinal
    order is preserved whatever the raw codes are). The input index is kept.
    """
    if not df.index.is_unique:
        raise OutcomeError("input index must be unique")
    q = question
    answers = _numeric(df, q.gss_variable)
    years = _numeric(df, year_col)
    weights = _numeric(df, weight_col)

    if np.isnan(years).any() or (years != np.floor(years)).any():
        raise OutcomeError(f"{year_col!r} must be whole numbers with no missing values")
    if (np.isinf(weights)).any() or (weights < 0).any():
        raise OutcomeError(f"{weight_col!r} contains negative or infinite values")

    n_input = len(df)
    in_years = (years >= q.years[0]) & (years <= q.years[1])
    if not in_years.any():
        raise OutcomeError(f"no rows in {q.id}'s year range {q.years[0]}-{q.years[1]}")

    has_answer = ~np.isnan(answers)
    unknown = in_years & has_answer & ~np.isin(answers, np.array(q.option_codes, dtype="float64"))
    if unknown.any():
        vals, n = np.unique(answers[unknown], return_counts=True)
        shown = ", ".join(f"{v:g} (x{c})" for v, c in list(zip(vals, n))[:10])
        raise OutcomeError(f"{q.id}: answer code(s) not in the registry: {shown}")

    good_weight = ~np.isnan(weights) & (weights > 0)
    step1 = in_years
    step2 = step1 & has_answer
    keep = step2 & good_weight
    n_outside = int((~in_years).sum())
    n_missing_answer = int((step1 & ~has_answer).sum())
    n_bad_weight = int((step2 & ~good_weight).sum())
    if not keep.any():
        raise OutcomeError(f"{q.id}: no usable rows (missing answers or weights)")

    code_to_idx = {c: i for i, c in enumerate(q.option_codes)}
    idx = np.array([code_to_idx[int(a)] for a in answers[keep]], dtype="int64")
    out = pd.DataFrame(
        {"year": years[keep].astype("int64"), "answer": idx, "weight": weights[keep]},
        index=df.index[keep],
    )

    k = len(q.options)
    counts = np.bincount(idx, minlength=k)
    wsum = np.bincount(idx, weights=weights[keep], minlength=k)
    share = wsum / wsum.sum()
    unobserved = tuple(o for o, c in zip(q.options, counts) if c == 0)
    rare = tuple(o for o, c in zip(q.options, counts) if 0 < c < min_option_count)
    flags = []
    if unobserved:
        flags.append(f"options never observed: {list(unobserved)}")
    if rare:
        flags.append(f"options with fewer than {min_option_count} respondents: {list(rare)}")
    if (counts > 0).sum() == 1:
        flags.append("only one option observed (single-class outcome)")
    if len(out) < min_rows:
        flags.append(f"only {len(out)} usable rows (< {min_rows})")

    summary = OutcomeSummary(
        q.id, n_input, n_outside, n_missing_answer, n_bad_weight, len(out),
        tuple(int(c) for c in counts), tuple(float(x) for x in share),
        unobserved, rare, tuple(flags),
    )
    return out, summary

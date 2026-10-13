"""Train / calibration / test split of *respondents*.

Splitting respondents (not rows) means a person never lands in different
sets for different questions. Assignment is stratified on demographic
columns (default year x party), never on outcomes or weights, and is a pure
function of (respondent set, seed, fractions): independent of row order, and
a stratum's assignment does not change when other strata change.
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

SPLIT_NAMES = ("train", "calibration", "test")
DEFAULT_FRACTIONS = {"train": 0.6, "calibration": 0.2, "test": 0.2}
DEFAULT_STRATA = ("year", "party")
DEFAULT_MIN_STRATUM = 10
POOLED = "__pooled__"
MISSING = "missing"
_RID_RE = re.compile(r"^\d{4}-\d{1,7}$")


class SplitError(ValueError):
    """Raised for invalid split inputs or parameters."""


@dataclass(frozen=True)
class Splits:
    ids: dict[str, tuple[str, ...]]  # split name -> sorted respondent ids
    n_strata: int
    n_pooled_strata: int             # small strata merged into one pooled group


def check_fractions(fractions: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(fractions, Mapping) or set(fractions) != set(SPLIT_NAMES):
        raise SplitError(f"fractions must be a mapping with exactly the keys {SPLIT_NAMES}")
    out = {}
    for k in SPLIT_NAMES:
        v = fractions[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise SplitError(f"fraction for {k!r} must be a finite number > 0, got {v!r}")
        out[k] = float(v)
    if abs(sum(out.values()) - 1.0) > 1e-9:
        raise SplitError(f"fractions must sum to 1, got {sum(out.values())}")
    return out


def respondent_frame(table: pd.DataFrame, strata_cols: Sequence[str] = DEFAULT_STRATA) -> pd.DataFrame:
    """Collapse the long respondent x question table to one row per respondent."""
    cols = ["respondent_id", *strata_cols]
    missing = [c for c in cols if c not in table.columns]
    if missing:
        raise SplitError(f"table lacks column(s): {missing}")
    first = table[cols].drop_duplicates()
    if first["respondent_id"].duplicated().any():
        raise SplitError("a respondent has different stratification values in different rows")
    return first.reset_index(drop=True)


def _allocate(n: int, fracs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Largest-remainder allocation of n items to len(fracs) groups."""
    exact = n * fracs
    base = np.floor(exact).astype(int)
    rem = exact - base + rng.random(len(fracs)) * 1e-9  # random tie-break
    extra = n - base.sum()
    base[np.argsort(-rem, kind="stable")[:extra]] += 1
    return base


def _stratum_rng(seed: int, key: str) -> np.random.Generator:
    h = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    return np.random.default_rng([seed, h])


def make_splits(
    respondents: pd.DataFrame,
    seed: int,
    fractions: Mapping[str, float] = DEFAULT_FRACTIONS,
    strata_cols: Sequence[str] = DEFAULT_STRATA,
    min_stratum_size: int = DEFAULT_MIN_STRATUM,
) -> Splits:
    """Assign every respondent to exactly one of train / calibration / test."""
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise SplitError("seed must be an integer in [0, 2**32)")
    fr = check_fractions(fractions)
    if isinstance(min_stratum_size, bool) or not isinstance(min_stratum_size, int) or min_stratum_size < 1:
        raise SplitError("min_stratum_size must be a positive integer")
    strata_cols = list(strata_cols)
    need = ["respondent_id", *strata_cols]
    missing = [c for c in need if c not in respondents.columns]
    if missing:
        raise SplitError(f"respondents lacks column(s): {missing}")
    if len(respondents) == 0:
        raise SplitError("no respondents to split")
    ids = respondents["respondent_id"]
    if ids.isna().any() or not ids.astype(str).map(lambda s: bool(_RID_RE.match(s))).all():
        raise SplitError("respondent_id must be non-null and look like '<year>-<id>'")
    if ids.duplicated().any():
        raise SplitError("respondent_id must be unique (one row per respondent)")
    if len(respondents) < len(SPLIT_NAMES):
        raise SplitError(f"need at least {len(SPLIT_NAMES)} respondents, got {len(respondents)}")

    frame = pd.DataFrame({"rid": ids.astype(str)})
    keys = pd.Series([""] * len(frame), index=frame.index)
    for c in strata_cols:
        col = respondents[c]
        col = col.astype(object).where(col.notna(), MISSING)
        keys = keys + "|" + col.astype(str)
    frame["key"] = keys.to_numpy()

    sizes = frame["key"].value_counts()
    small = set(sizes[sizes < min_stratum_size].index)
    if small:
        log.warning("%d stratum(s) smaller than %d pooled together (%d respondents)",
                    len(small), min_stratum_size, int(sizes[list(small)].sum()))
        frame.loc[frame["key"].isin(small), "key"] = POOLED

    fracs = np.array([fr[n] for n in SPLIT_NAMES])
    assigned: dict[str, list[str]] = {n: [] for n in SPLIT_NAMES}
    n_strata = frame["key"].nunique()
    for key in sorted(frame["key"].unique()):
        members = sorted(frame.loc[frame["key"] == key, "rid"])
        rng = _stratum_rng(seed, key)
        counts = _allocate(len(members), fracs, rng)
        order = rng.permutation(len(members))
        pos = 0
        for name, c in zip(SPLIT_NAMES, counts):
            assigned[name].extend(members[i] for i in order[pos:pos + c])
            pos += c

    _ensure_non_empty(assigned)
    return Splits({n: tuple(sorted(v)) for n, v in assigned.items()}, n_strata,
                  1 if small else 0)


def _ensure_non_empty(assigned: dict[str, list[str]]) -> None:
    """Rare with tiny data: move respondents so no split is empty."""
    for name in SPLIT_NAMES:
        if not assigned[name]:
            donor = max(SPLIT_NAMES, key=lambda n: (len(assigned[n]), n))
            assigned[name].append(sorted(assigned[donor])[-1])
            assigned[donor].remove(assigned[name][-1])

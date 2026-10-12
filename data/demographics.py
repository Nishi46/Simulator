"""Demographic harmonisation: raw GSS codes -> small closed categorical sets.

Input is the *cleaned* frame (special codes already NaN, see clean.py). Output
holds only the allowlisted demographic columns, each a pandas Categorical with
a fixed category list, so no free text or identifiers can leak downstream.
NaN in the input stays NaN in the output; an unrecognised code raises.

The code tables below follow the GSS codebook as I know it and must be
checked against the real codebook in step 1.15.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

AGE_MIN, AGE_MAX = 18, 89  # GSS tops out at 89 ("89 or older")
AGE_EDGES = (18, 30, 45, 60, 75, 90)
AGE_LABELS = ("18-29", "30-44", "45-59", "60-74", "75+")

# Annual family income in constant dollars (GSS `realinc` is in 1986 dollars).
INCOME_EDGES = (0, 15_000, 30_000, 60_000, float("inf"))
INCOME_LABELS = ("lt_15k", "15k_30k", "30k_60k", "60k_plus")

SEX = {1: "male", 2: "female"}
RACE = {1: "white", 2: "black", 3: "other"}
EDUCATION = {0: "lt_high_school", 1: "high_school", 2: "junior_college", 3: "bachelor", 4: "graduate"}
REGION = {1: "northeast", 2: "northeast", 3: "midwest", 4: "midwest", 5: "south", 6: "south",
          7: "south", 8: "west", 9: "west"}
PARTY = {0: "democrat", 1: "democrat", 2: "lean_democrat", 3: "independent", 4: "lean_republican",
         5: "republican", 6: "republican", 7: "other"}
URBANICITY = {1: "urban", 2: "urban", 3: "suburban", 4: "suburban", 5: "small_town_rural", 6: "small_town_rural"}

# output column -> (raw column, mapping, ordered category list, ordered?)
_CODED = {
    "sex": ("sex", SEX, ("male", "female"), False),
    "race": ("race", RACE, ("white", "black", "other"), False),
    "education": ("degree", EDUCATION, tuple(EDUCATION.values()), True),
    "region": ("region", REGION, ("northeast", "midwest", "south", "west"), False),
    "party": ("partyid", PARTY, ("democrat", "lean_democrat", "independent", "lean_republican",
                                 "republican", "other"), False),
    "urbanicity": ("srcbelt", URBANICITY, ("urban", "suburban", "small_town_rural"), True),
}

DEMOGRAPHIC_COLUMNS = ("age_band", "sex", "race", "education", "income_band", "region", "party", "urbanicity")


class DemographicsError(ValueError):
    """Raised for missing columns, non-numeric data, or unrecognised codes."""


def _numeric(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        raise DemographicsError(f"required column missing: {col!r}")
    s = df[col]
    if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        raise DemographicsError(f"column {col!r} is not numeric (dtype {s.dtype})")
    return np.array(s.to_numpy(dtype="float64", na_value=np.nan), copy=True)


def _unrecognised(col: str, values: np.ndarray) -> DemographicsError:
    shown = ", ".join(f"{v:g}" for v in np.unique(values)[:10])
    return DemographicsError(f"{col}: unrecognised value(s): {shown}")


def _from_codes(df: pd.DataFrame, raw: str, mapping: Mapping[int, str], cats: tuple[str, ...], ordered: bool) -> pd.Categorical:
    v = _numeric(df, raw)
    present = ~np.isnan(v)
    bad = present & ~np.isin(v, np.array(list(mapping), dtype="float64"))
    if bad.any():
        raise _unrecognised(raw, v[bad])
    labels = np.full(len(v), None, dtype=object)
    for code, label in mapping.items():
        labels[present & (v == code)] = label
    return pd.Categorical(labels, categories=list(cats), ordered=ordered)


def age_band(df: pd.DataFrame, col: str = "age") -> pd.Categorical:
    v = _numeric(df, col)
    present = ~np.isnan(v)
    bad = present & ((v < AGE_MIN) | (v > AGE_MAX) | (v != np.floor(v)))
    if bad.any():
        raise _unrecognised(col, v[bad])
    return pd.Categorical(
        pd.cut(pd.Series(v), bins=AGE_EDGES, labels=AGE_LABELS, right=False),
        categories=list(AGE_LABELS), ordered=True,
    )


def income_band(df: pd.DataFrame, col: str = "realinc", *, deflators: Mapping[int, float] | None = None,
                year_col: str = "year") -> pd.Categorical:
    """Band family income into fixed constant-dollar bands.

    By default `col` is already in constant dollars (`realinc`). To use a
    nominal income column instead, pass `deflators` (year -> factor converting
    that year's dollars to the base year); every year present must be listed.
    """
    v = _numeric(df, col)
    present = ~np.isnan(v)
    if (present & (~np.isfinite(v) | (v < 0))).any():
        raise _unrecognised(col, v[present & (~np.isfinite(v) | (v < 0))])
    if deflators is not None:
        years = _numeric(df, year_col)
        for y, f in deflators.items():
            if not isinstance(f, (int, float)) or isinstance(f, bool) or not np.isfinite(f) or f <= 0:
                raise DemographicsError(f"deflator for {y} must be a positive finite number")
        factor = np.full(len(v), np.nan)
        for y, f in deflators.items():
            factor[years == y] = f
        need = present & np.isnan(factor)
        if need.any():
            raise DemographicsError(f"no deflator for year(s): {sorted({int(y) for y in years[need] if not np.isnan(y)})}"
                                    + (" (and some rows have no year)" if np.isnan(years[need]).any() else ""))
        v = v * factor
    return pd.Categorical(
        pd.cut(pd.Series(v), bins=INCOME_EDGES, labels=INCOME_LABELS, right=False),
        categories=list(INCOME_LABELS), ordered=True,
    )


def harmonize(df: pd.DataFrame, *, deflators: Mapping[int, float] | None = None,
              income_col: str = "realinc") -> pd.DataFrame:
    """Return only DEMOGRAPHIC_COLUMNS, as categoricals, with df's index."""
    out = {"age_band": age_band(df), "income_band": income_band(df, income_col, deflators=deflators)}
    for name, (raw, mapping, cats, ordered) in _CODED.items():
        out[name] = _from_codes(df, raw, mapping, cats, ordered)
    return pd.DataFrame(out, index=df.index)[list(DEMOGRAPHIC_COLUMNS)]

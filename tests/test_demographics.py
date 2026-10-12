import numpy as np
import pandas as pd
import pytest

from data import demographics as d
from data.demographics import DEMOGRAPHIC_COLUMNS, DemographicsError, age_band, harmonize, income_band


def frame(**over):
    base = {"age": [25], "sex": [1], "race": [1], "degree": [3], "realinc": [40_000.0],
            "region": [3], "partyid": [0], "srcbelt": [1], "year": [2000]}
    base.update(over)
    return pd.DataFrame(base)


def labels(cat):
    return [None if pd.isna(x) else str(x) for x in cat]


@pytest.mark.parametrize("age,band", [
    (18, "18-29"), (29, "18-29"), (30, "30-44"), (44, "30-44"), (45, "45-59"), (59, "45-59"),
    (60, "60-74"), (74, "60-74"), (75, "75+"), (88, "75+"), (89, "75+"),
])
def test_age_band_boundaries(age, band):
    assert labels(age_band(pd.DataFrame({"age": [age]}))) == [band]


@pytest.mark.parametrize("bad", [17, 0, 90, 99, -1, 30.5, 17.999, np.inf])
def test_age_out_of_range_or_fractional_raises(bad):
    with pytest.raises(DemographicsError, match="unrecognised"):
        age_band(pd.DataFrame({"age": [bad]}))


def test_missing_age_stays_missing():
    out = age_band(pd.DataFrame({"age": [np.nan, 40]}))
    assert labels(out) == [None, "30-44"]


def test_float_whole_ages_ok():
    assert labels(age_band(pd.DataFrame({"age": [30.0, 89.0]}))) == ["30-44", "75+"]


@pytest.mark.parametrize("val,band", [
    (0, "lt_15k"), (14_999.99, "lt_15k"), (15_000, "15k_30k"), (29_999, "15k_30k"),
    (30_000, "30k_60k"), (59_999.5, "30k_60k"), (60_000, "60k_plus"), (10**9, "60k_plus"),
])
def test_income_band_boundaries(val, band):
    assert labels(income_band(pd.DataFrame({"realinc": [val]}))) == [band]


@pytest.mark.parametrize("bad", [-1, -0.01, np.inf, -np.inf])
def test_bad_income_raises(bad):
    with pytest.raises(DemographicsError):
        income_band(pd.DataFrame({"realinc": [bad]}))


def test_missing_income_stays_missing():
    assert labels(income_band(pd.DataFrame({"realinc": [np.nan, 20_000]}))) == [None, "15k_30k"]


def test_income_is_year_consistent_via_deflators():
    # same real purchasing power in two years lands in the same band
    df = pd.DataFrame({"realinc": [20_000.0, 40_000.0], "year": [1990, 2010]})
    out = income_band(df, deflators={1990: 1.0, 2010: 0.5})
    assert labels(out) == ["15k_30k", "15k_30k"]
    # without deflation they would differ
    assert labels(income_band(df)) == ["15k_30k", "30k_60k"]


def test_deflators_missing_year_or_bad_factor():
    df = pd.DataFrame({"realinc": [20_000.0], "year": [1995]})
    with pytest.raises(DemographicsError, match="1995"):
        income_band(df, deflators={1990: 1.0})
    for f in (0, -1, np.nan, np.inf, True, "1"):
        with pytest.raises(DemographicsError, match="deflator"):
            income_band(df, deflators={1995: f})
    with pytest.raises(DemographicsError, match="no year"):
        income_band(pd.DataFrame({"realinc": [1.0], "year": [np.nan]}), deflators={1990: 1.0})
    with pytest.raises(DemographicsError, match="required column"):
        income_band(pd.DataFrame({"realinc": [1.0]}), deflators={1990: 1.0})


def test_harmonize_happy_path():
    out = harmonize(frame())
    assert list(out.columns) == list(DEMOGRAPHIC_COLUMNS)
    assert out.iloc[0].astype(str).tolist() == [
        "18-29", "male", "white", "bachelor", "30k_60k", "midwest", "democrat", "urban"]


def test_every_code_maps_to_a_declared_category():
    for out_col, (raw, mapping, cats, _) in d._CODED.items():
        assert set(mapping.values()) <= set(cats), out_col
        keys = list(mapping)
        df = pd.concat([frame()] * len(keys), ignore_index=True)
        df[raw] = keys
        assert harmonize(df)[out_col].astype(str).tolist() == [mapping[k] for k in keys]


@pytest.mark.parametrize("raw,bad", [
    ("sex", 3), ("sex", 0), ("race", 4), ("degree", 5), ("degree", -1), ("region", 0), ("region", 10),
    ("partyid", 8), ("partyid", 1.5), ("srcbelt", 7), ("srcbelt", 0), ("sex", np.inf),
])
def test_unseen_code_raises_naming_column(raw, bad):
    with pytest.raises(DemographicsError, match=f"{raw}: unrecognised"):
        harmonize(frame(**{raw: [bad]}))


def test_missing_values_stay_missing_per_column():
    out = harmonize(frame(sex=[np.nan], partyid=[np.nan], age=[np.nan], realinc=[np.nan]))
    assert out["sex"].isna().iloc[0] and out["party"].isna().iloc[0]
    assert out["age_band"].isna().iloc[0] and out["income_band"].isna().iloc[0]
    assert out["race"].iloc[0] == "white"


def test_outputs_are_closed_categoricals_with_fixed_categories():
    df = pd.concat([frame()] * 3, ignore_index=True)
    out = harmonize(df)
    for col in DEMOGRAPHIC_COLUMNS:
        assert isinstance(out[col].dtype, pd.CategoricalDtype), col
    assert list(out["age_band"].cat.categories) == list(d.AGE_LABELS)
    assert list(out["party"].cat.categories)[0] == "democrat"
    # categories present even when unobserved, so downstream encoding is stable
    assert len(out["sex"].cat.categories) == 2 and set(out["sex"].astype(str)) == {"male"}
    assert out["age_band"].cat.ordered and out["education"].cat.ordered and not out["race"].cat.ordered


def test_only_allowlisted_columns_returned_no_identifiers():
    df = frame()
    df["id"] = 123
    df["name"] = "Jane Doe"
    df["wtssall"] = 1.0
    out = harmonize(df)
    assert set(out.columns) == set(DEMOGRAPHIC_COLUMNS)
    assert "Jane Doe" not in out.astype(str).to_numpy()
    assert 123 not in out.to_numpy()


def test_index_and_input_preserved():
    df = frame()
    df.index = [42]
    before = df.copy()
    out = harmonize(df)
    assert out.index.tolist() == [42]
    pd.testing.assert_frame_equal(df, before)


def test_missing_required_column_named():
    with pytest.raises(DemographicsError, match="'degree'"):
        harmonize(frame().drop(columns="degree"))


@pytest.mark.parametrize("col,val", [("sex", "male"), ("age", "thirty"), ("race", True)])
def test_non_numeric_input_rejected(col, val):
    with pytest.raises(DemographicsError, match="not numeric"):
        harmonize(frame(**{col: [val]}))


def test_empty_frame():
    out = harmonize(frame().iloc[0:0])
    assert len(out) == 0 and list(out.columns) == list(DEMOGRAPHIC_COLUMNS)


def test_many_rows_mixed():
    n = 1000
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "age": rng.integers(18, 90, n), "sex": rng.integers(1, 3, n), "race": rng.integers(1, 4, n),
        "degree": rng.integers(0, 5, n), "realinc": rng.uniform(0, 150_000, n),
        "region": rng.integers(1, 10, n), "partyid": rng.integers(0, 8, n),
        "srcbelt": rng.integers(1, 7, n), "year": 2000,
    })
    out = harmonize(df)
    assert len(out) == n and not out.isna().any().any()

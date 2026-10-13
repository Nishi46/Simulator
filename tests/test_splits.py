import logging

import numpy as np
import pandas as pd
import pytest

from data.build import build_table
from eval.splits import (
    DEFAULT_FRACTIONS, SPLIT_NAMES, SplitError, make_splits, respondent_frame, check_fractions,
)


def people(n=2000, seed=0, years=(2000, 2004, 2008), parties=("dem", "rep", "ind")):
    r = np.random.default_rng(seed)
    yr = r.choice(years, n)
    return pd.DataFrame({
        "respondent_id": [f"{y}-{i}" for i, y in enumerate(yr)],
        "year": yr, "party": r.choice(parties, n), "weight": r.uniform(0.5, 2, n),
    })


def all_ids(s):
    return [i for n in SPLIT_NAMES for i in s.ids[n]]


def test_partition_properties():
    df = people()
    s = make_splits(df, seed=1)
    ids = all_ids(s)
    assert len(ids) == len(set(ids)) == len(df)                      # disjoint, complete
    assert set(ids) == set(df["respondent_id"])
    for n in SPLIT_NAMES:
        assert list(s.ids[n]) == sorted(s.ids[n])
        assert abs(len(s.ids[n]) / len(df) - DEFAULT_FRACTIONS[n]) < 0.01


def test_strata_are_balanced_within_each_split():
    df = people(6000)
    s = make_splits(df, seed=2)
    lookup = df.set_index("respondent_id")["party"]
    overall = lookup.value_counts(normalize=True)
    for n in SPLIT_NAMES:
        share = lookup.loc[list(s.ids[n])].value_counts(normalize=True)
        assert (share - overall).abs().max() < 0.01


def test_same_seed_identical_different_seed_differs():
    df = people()
    a, b, c = make_splits(df, 7), make_splits(df, 7), make_splits(df, 8)
    assert a == b
    assert a.ids["test"] != c.ids["test"]


def test_independent_of_row_order():
    df = people()
    shuffled = df.sample(frac=1, random_state=3).reset_index(drop=True)
    assert make_splits(df, 5) == make_splits(shuffled, 5)


def test_weights_and_extra_columns_do_not_affect_assignment():
    df = people()
    other = df.copy()
    other["weight"] = 1.0
    other["noise"] = "x"
    assert make_splits(df, 5) == make_splits(other, 5)


def test_a_stratum_is_unaffected_by_other_strata():
    df = people(parties=("dem", "rep"))
    base = make_splits(df, 9)
    extra = pd.concat([df, people(300, seed=5, parties=("green",)).assign(
        respondent_id=lambda d: "2012-" + d.index.astype(str))], ignore_index=True)
    more = make_splits(extra, 9)
    for n in SPLIT_NAMES:
        old = set(base.ids[n])
        assert old <= set(more.ids[n])


@pytest.mark.parametrize("seed", [-1, 2**32, 1.5, "1", None, True])
def test_bad_seed(seed):
    with pytest.raises(SplitError, match="seed"):
        make_splits(people(50), seed)


@pytest.mark.parametrize("fr", [
    {"train": 0.5, "calibration": 0.2, "test": 0.2},
    {"train": 0.8, "calibration": 0.3, "test": 0.2},
    {"train": 1.0, "calibration": 0.0, "test": 0.0},
    {"train": 0.7, "calibration": 0.4, "test": -0.1},
    {"train": float("nan"), "calibration": 0.5, "test": 0.5},
    {"train": float("inf"), "calibration": 0.5, "test": 0.5},
    {"train": True, "calibration": 0.0, "test": 0.0},
    {"train": "0.6", "calibration": 0.2, "test": 0.2},
    {"train": 0.6, "calibration": 0.4},
    {"train": 0.6, "calibration": 0.2, "test": 0.1, "extra": 0.1},
    [0.6, 0.2, 0.2], None,
])
def test_bad_fractions(fr):
    with pytest.raises(SplitError):
        make_splits(people(50), 1, fractions=fr)


def test_custom_fractions_honoured():
    s = make_splits(people(3000), 1, fractions={"train": 0.5, "calibration": 0.25, "test": 0.25})
    assert abs(len(s.ids["train"]) / 3000 - 0.5) < 0.01


def test_tiny_strata_pooled_with_warning(caplog):
    df = people(300)
    df.loc[:1, "party"] = "rare"          # a stratum with 1-2 members
    with caplog.at_level(logging.WARNING):
        s = make_splits(df, 1)
    assert s.n_pooled_strata == 1 and "pooled" in caplog.text
    assert len(all_ids(s)) == 300 and all(s.ids[n] for n in SPLIT_NAMES)


def test_every_stratum_of_one():
    df = people(30, parties=tuple(f"p{i}" for i in range(30)), years=(2000,))
    s = make_splits(df, 1)
    assert len(all_ids(s)) == 30 and all(s.ids[n] for n in SPLIT_NAMES)


@pytest.mark.parametrize("n", [3, 4, 5, 7])
def test_tiny_total_never_leaves_an_empty_split(n):
    for seed in range(20):
        s = make_splits(people(n), seed)
        assert all(s.ids[k] for k in SPLIT_NAMES) and len(all_ids(s)) == n


def test_too_few_respondents():
    for n in (1, 2):
        with pytest.raises(SplitError, match="at least"):
            make_splits(people(n), 1)
    with pytest.raises(SplitError, match="no respondents"):
        make_splits(people(5).iloc[0:0], 1)


def test_missing_values_in_strata_columns_form_own_stratum():
    df = people(400)
    df.loc[:49, "party"] = np.nan
    s = make_splits(df, 1)
    assert len(all_ids(s)) == 400


def test_categorical_strata_column():
    df = people(300)
    df["party"] = pd.Categorical(df["party"], categories=["dem", "rep", "ind", "unused"])
    assert len(all_ids(make_splits(df, 1))) == 300


def test_custom_strata_columns_and_missing_column():
    df = people(300)
    assert len(all_ids(make_splits(df, 1, strata_cols=["year"]))) == 300
    with pytest.raises(SplitError, match="lacks"):
        make_splits(df, 1, strata_cols=["nope"])
    with pytest.raises(SplitError, match="lacks"):
        make_splits(df.drop(columns="respondent_id"), 1)


def test_bad_respondent_ids():
    df = people(50)
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(SplitError, match="unique"):
        make_splits(dup, 1)
    bad = df.copy()
    bad.loc[0, "respondent_id"] = "Jane Doe"
    with pytest.raises(SplitError, match="look like"):
        make_splits(bad, 1)
    bad.loc[0, "respondent_id"] = None
    with pytest.raises(SplitError):
        make_splits(bad, 1)


@pytest.mark.parametrize("m", [0, -1, True, 1.5])
def test_bad_min_stratum_size(m):
    with pytest.raises(SplitError, match="min_stratum_size"):
        make_splits(people(50), 1, min_stratum_size=m)


def test_input_not_mutated():
    df = people(200)
    before = df.copy()
    make_splits(df, 1)
    pd.testing.assert_frame_equal(df, before)


def test_check_fractions_returns_floats():
    out = check_fractions({"train": 3 / 5, "calibration": 1 / 5, "test": 1 / 5})
    assert all(isinstance(v, float) for v in out.values())


# ---- respondent stays in one set across questions -------------------------

def test_respondent_in_one_set_across_questions():
    from tests.test_build import QS, RULES, raw
    table, _ = build_table(raw(200), QS, RULES)
    rf = respondent_frame(table, strata_cols=["year"])
    s = make_splits(rf, 4, strata_cols=["year"])
    where = {i: n for n in SPLIT_NAMES for i in s.ids[n]}
    per_row = table["respondent_id"].map(where)
    assert per_row.notna().all()
    assert (per_row.groupby(table["respondent_id"]).nunique() == 1).all()
    assert table["respondent_id"].nunique() == len(rf) == len(where)


def test_respondent_frame_errors():
    from tests.test_build import QS, RULES, raw
    table, _ = build_table(raw(60), QS, RULES)
    with pytest.raises(SplitError, match="lacks"):
        respondent_frame(table, strata_cols=["nope"])
    bad = table.copy()
    i = bad.index[bad["respondent_id"].duplicated(keep=False)][0]
    bad.loc[i, "year"] = bad.loc[i, "year"] + 1
    with pytest.raises(SplitError, match="different stratification"):
        respondent_frame(bad, strata_cols=["year"])

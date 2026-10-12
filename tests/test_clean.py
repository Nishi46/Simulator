import numpy as np
import pandas as pd
import pytest

from data.clean import CleanError, Rule, drop_incomplete, normalize_missing, parse_rules

GRASS = Rule("grass", missing={0: "IAP", 8: "DK", 9: "NA"}, valid_values=(1, 2))
AGE = Rule("age", missing={99: "NA"}, valid_range=(18, 89))


def clean(data, *rules):
    return normalize_missing(pd.DataFrame(data), rules)


def test_each_missing_code_becomes_nan_with_counts():
    df, rep = clean({"grass": [1, 2, 0, 8, 9, 1, 8]}, GRASS)
    assert df["grass"].isna().tolist() == [False, False, True, True, True, False, True]
    r = rep["grass"]
    assert r.n_rows == 7 and r.n_valid == 3 and r.n_already_missing == 0
    assert r.n_coded_missing == {"IAP": 1, "DK": 2, "NA": 1}


def test_legitimate_values_untouched_including_look_alikes():
    # 0 is a code for grass but a perfectly valid value for a column that says so
    zero_ok = Rule("kids", missing={9: "NA"}, valid_range=(0, 8))
    df, rep = clean({"kids": [0, 3, 8, 9]}, zero_ok)
    assert df["kids"].tolist()[:3] == [0, 3, 8] and np.isnan(df["kids"][3])
    assert rep["kids"].n_valid == 3


def test_top_code_89_kept_99_missing():
    df, _ = clean({"age": [18, 89, 99, 45]}, AGE)
    assert df["age"].dropna().tolist() == [18, 89, 45]


def test_unknown_code_raises_naming_value_and_count():
    with pytest.raises(CleanError, match=r"7 \(x2\)"):
        clean({"grass": [1, 7, 7, 2]}, GRASS)


def test_unknown_values_shown_are_capped():
    with pytest.raises(CleanError, match="and 5 more"):
        clean({"grass": list(range(100, 115))}, GRASS)


@pytest.mark.parametrize("bad", [1.5, -1.0, np.inf, -np.inf, 3])
def test_non_member_values_raise(bad):
    with pytest.raises(CleanError, match="unrecognised"):
        clean({"grass": [1, bad]}, GRASS)


def test_range_boundaries():
    clean({"age": [18, 89]}, AGE)
    for bad in (17, 17.999, 89.5, 90, 0):
        with pytest.raises(CleanError):
            clean({"age": [bad]}, AGE)


def test_existing_nan_preserved_and_counted():
    df, rep = clean({"grass": [1, np.nan, 9]}, GRASS)
    assert df["grass"].isna().tolist() == [False, True, True]
    assert rep["grass"].n_already_missing == 1 and rep["grass"].n_coded_missing == {"NA": 1}


def test_all_nan_column_ok():
    df, rep = clean({"grass": [np.nan, np.nan]}, GRASS)
    assert df["grass"].isna().all() and rep["grass"].n_valid == 0


def test_all_missing_codes_column_ok():
    df, rep = clean({"grass": [8, 9, 0]}, GRASS)
    assert df["grass"].isna().all() and rep["grass"].n_valid == 0


def test_empty_frame_ok():
    df, rep = clean({"grass": pd.Series([], dtype="float64")}, GRASS)
    assert len(df) == 0 and rep["grass"].n_rows == 0


def test_labels_aggregate_across_codes():
    rule = Rule("x", missing={8: "DK", 98: "DK"}, valid_values=(1,))
    _, rep = clean({"x": [1, 8, 98, 98]}, rule)
    assert rep["x"].n_coded_missing == {"DK": 3}


def test_input_not_mutated_and_other_columns_pass_through():
    src = pd.DataFrame({"grass": [1, 9], "id": [10, 11], "note": ["a", "b"]})
    before = src.copy()
    out, rep = normalize_missing(src, [GRASS])
    pd.testing.assert_frame_equal(src, before)
    assert out["id"].tolist() == [10, 11] and out["note"].tolist() == ["a", "b"]
    assert list(rep) == ["grass"]


def test_index_preserved():
    df = pd.DataFrame({"grass": [1, 9]}, index=[5, 7])
    out, _ = normalize_missing(df, [GRASS])
    assert out.index.tolist() == [5, 7]


def test_nullable_integer_dtype():
    df = pd.DataFrame({"grass": pd.array([1, 9, None, 2], dtype="Int64")})
    out, rep = normalize_missing(df, [GRASS])
    assert out["grass"].isna().tolist() == [False, True, True, False]
    assert rep["grass"].n_already_missing == 1


def test_float32_codes_match():
    df = pd.DataFrame({"grass": np.array([1, 9, 2], dtype="float32")})
    out, _ = normalize_missing(df, [GRASS])
    assert out["grass"].isna().tolist() == [False, True, False]


def test_missing_rule_column_and_non_numeric_column():
    with pytest.raises(CleanError, match="not in the data"):
        clean({"other": [1]}, GRASS)
    with pytest.raises(CleanError, match="not numeric"):
        clean({"grass": ["1", "2"]}, GRASS)
    with pytest.raises(CleanError, match="not numeric"):
        clean({"grass": [True, False]}, GRASS)


def test_duplicate_rules_for_one_column():
    with pytest.raises(CleanError, match="more than one"):
        clean({"grass": [1]}, GRASS, GRASS)


# ---- rule validation -------------------------------------------------------

@pytest.mark.parametrize("kw", [
    dict(column="Grass", valid_values=(1,)),
    dict(column="", valid_values=(1,)),
    dict(column="a b", valid_values=(1,)),
    dict(column="g"),                                           # no valid spec
    dict(column="g", valid_range=(5, 1)),
    dict(column="g", valid_range=(1,)),
    dict(column="g", valid_values=(float("nan"),)),
    dict(column="g", valid_values=(float("inf"),)),
    dict(column="g", valid_values=("1",)),
    dict(column="g", valid_values=(True,)),
    dict(column="g", valid_values=(1, 2), missing={2: "NA"}),   # code both valid and missing
    dict(column="g", valid_range=(0, 10), missing={5: "NA"}),
    dict(column="g", valid_values=(1,), missing={9: ""}),
    dict(column="g", valid_values=(1,), missing={9: "bad\nlabel"}),
    dict(column="g", valid_values=(1,), missing={9: 5}),
    dict(column="g", valid_values=(1,), missing={"x": "NA"}),
])
def test_invalid_rules_rejected(kw):
    with pytest.raises(CleanError):
        Rule(**kw)


def test_parse_rules():
    rules = parse_rules({"grass": {"missing": {0: "IAP", 9: "NA"}, "valid_values": [1, 2]},
                         "age": {"missing": {99: "NA"}, "valid_range": [18, 89]}})
    assert [r.column for r in rules] == ["grass", "age"]
    df, _ = normalize_missing(pd.DataFrame({"grass": [1, 0], "age": [30, 99]}), rules)
    assert df.isna().sum().tolist() == [1, 1]


@pytest.mark.parametrize("spec", [None, {}, [], {"g": 5}, {"g": {"valid_values": [1], "oops": 1}},
                                  {"g": {}}, "text"])
def test_parse_rules_rejects_bad_specs(spec):
    with pytest.raises(CleanError):
        parse_rules(spec)


# ---- drop_incomplete -------------------------------------------------------

def test_drop_incomplete_counts_rows():
    df = pd.DataFrame({"a": [1, np.nan, 3, 4], "b": [1, 2, np.nan, 4]})
    kept, n = drop_incomplete(df, ["a", "b"])
    assert n == 2 and kept.index.tolist() == [0, 3]
    kept, n = drop_incomplete(df, ["a"])
    assert n == 1
    _, n = drop_incomplete(df.dropna(), ["a", "b"])
    assert n == 0


def test_drop_incomplete_unknown_column_and_everything_dropped():
    df = pd.DataFrame({"a": [np.nan]})
    with pytest.raises(CleanError):
        drop_incomplete(df, ["zzz"])
    kept, n = drop_incomplete(df, ["a"])
    assert len(kept) == 0 and n == 1

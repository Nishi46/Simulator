import numpy as np
import pandas as pd
import pytest

from data.outcomes import OutcomeError, extract
from data.questions import Question


def q(**kw):
    base = dict(id="grass_legal", gss_variable="grass", text="t", options=("Legal", "Not legal"),
                option_codes=(1, 2), ordinal=False, category="contested_social",
                years=(2000, 2010), contamination_note="n")
    base.update(kw)
    return Question(**base)


def frame(n=300, **over):
    rng = np.random.default_rng(0)
    d = {"grass": rng.choice([1, 2], n).astype(float), "year": rng.choice([2000, 2004, 2008], n),
         "wtssall": rng.uniform(0.5, 2, n)}
    d.update(over)
    return pd.DataFrame(d)


def test_basic_extraction_and_shapes():
    df = frame()
    out, s = extract(df, q())
    assert list(out.columns) == ["year", "answer", "weight"]
    assert set(out["answer"]) == {0, 1} and out["answer"].dtype == "int64"
    assert s.n_kept == len(out) == 300 and sum(s.counts) == 300
    assert abs(sum(s.weighted_share) - 1) < 1e-12
    assert out.index.equals(df.index)
    assert s.flags == ()


def test_answer_follows_registry_order_not_code_order():
    df = frame(n=4, grass=[3.0, 1.0, 2.0, 3.0], year=[2000] * 4, wtssall=[1.0] * 4)
    out, _ = extract(df, q(options=("Agree", "Neutral", "Disagree"), option_codes=(3, 2, 1),
                           ordinal=True), min_rows=1, min_option_count=1)
    assert out["answer"].tolist() == [0, 2, 1, 0]


def test_year_range_inclusive_and_counts():
    df = frame(n=6, grass=[1.0] * 6, year=[1999, 2000, 2005, 2010, 2011, 2020], wtssall=[1.0] * 6)
    out, s = extract(df, q(), min_rows=1, min_option_count=1)
    assert out["year"].tolist() == [2000, 2005, 2010] and s.n_outside_years == 3


def test_year_range_excludes_everything_errors():
    with pytest.raises(OutcomeError, match="year range"):
        extract(frame(year=[1990] * 300), q())


def test_missing_answers_dropped_and_counted():
    df = frame(n=10, grass=[1.0, np.nan, 2.0] + [1.0] * 7, year=[2000] * 10, wtssall=[1.0] * 10)
    out, s = extract(df, q(), min_rows=1, min_option_count=1)
    assert len(out) == 9 and s.n_missing_answer == 1 and 1 not in out.index


def test_answer_code_not_in_registry_raises_even_if_rare():
    df = frame(grass=[1.0, 2.0, 3.0] + [1.0] * 297)
    with pytest.raises(OutcomeError, match=r"not in the registry: 3 \(x1\)"):
        extract(df, q())


def test_out_of_range_year_rows_with_bad_codes_do_not_raise():
    df = frame(n=4, grass=[1.0, 2.0, 7.0, 1.0], year=[2000, 2000, 1980, 2000], wtssall=[1.0] * 4)
    out, s = extract(df, q(), min_rows=1, min_option_count=1)
    assert len(out) == 3 and s.n_outside_years == 1


def test_nan_and_zero_weights_dropped_counted():
    df = frame(n=6, grass=[1.0] * 6, year=[2000] * 6, wtssall=[1.0, np.nan, 0.0, 2.0, 0.0, 1.0])
    out, s = extract(df, q(), min_rows=1, min_option_count=1)
    assert out["weight"].tolist() == [1.0, 2.0, 1.0] and s.n_bad_weight == 3


@pytest.mark.parametrize("bad", [-0.5, np.inf, -np.inf])
def test_negative_or_infinite_weight_raises(bad):
    with pytest.raises(OutcomeError, match="negative or infinite"):
        extract(frame(wtssall=[bad] + [1.0] * 299), q())


def test_weights_all_invalid_errors():
    with pytest.raises(OutcomeError, match="no usable rows"):
        extract(frame(wtssall=[0.0] * 300), q())
    with pytest.raises(OutcomeError, match="no usable rows"):
        extract(frame(grass=[np.nan] * 300), q())


def test_unobserved_option_kept_with_zero_count_and_flagged():
    df = frame(grass=[1.0] * 300)
    _, s = extract(df, q())
    assert s.counts == (300, 0) and s.unobserved_options == ("Not legal",)
    assert s.weighted_share[1] == 0.0
    assert any("never observed" in f for f in s.flags)
    assert any("single-class" in f for f in s.flags)


def test_rare_option_flagged():
    df = frame(grass=[2.0] * 5 + [1.0] * 295)
    _, s = extract(df, q())
    assert s.rare_options == ("Not legal",) and any("fewer than 30" in f for f in s.flags)
    _, s2 = extract(df, q(), min_option_count=5)
    assert s2.rare_options == ()


def test_small_sample_flagged():
    _, s = extract(frame(n=50), q(), min_option_count=1)
    assert any("usable rows" in f for f in s.flags)


def test_weighted_share_uses_weights():
    df = frame(n=4, grass=[1.0, 1.0, 2.0, 2.0], year=[2000] * 4, wtssall=[1.0, 1.0, 3.0, 3.0])
    _, s = extract(df, q(), min_rows=1, min_option_count=1)
    assert s.weighted_share == (0.25, 0.75) and s.counts == (2, 2)


@pytest.mark.parametrize("col", ["grass", "year", "wtssall"])
def test_missing_required_column(col):
    with pytest.raises(OutcomeError, match=col):
        extract(frame().drop(columns=col), q())


@pytest.mark.parametrize("col,val", [("grass", "x"), ("wtssall", "x"), ("year", "x")])
def test_non_numeric_columns_rejected(col, val):
    with pytest.raises(OutcomeError, match="not numeric"):
        extract(frame(**{col: [val] * 300}), q())


def test_year_must_be_whole_and_present():
    with pytest.raises(OutcomeError, match="whole numbers"):
        extract(frame(year=[2000.5] * 300), q())
    with pytest.raises(OutcomeError, match="whole numbers"):
        extract(frame(year=[np.nan] + [2000] * 299), q())


def test_duplicate_index_rejected():
    df = frame(n=4)
    df.index = [0, 0, 1, 2]
    with pytest.raises(OutcomeError, match="unique"):
        extract(df, q(), min_rows=1)


def test_input_not_mutated():
    df = frame()
    before = df.copy()
    extract(df, q())
    pd.testing.assert_frame_equal(df, before)


def test_custom_column_names():
    df = frame().rename(columns={"year": "yr", "wtssall": "w"})
    out, _ = extract(df, q(), year_col="yr", weight_col="w")
    assert len(out) == 300


def test_works_with_shipped_registry_shape():
    from data.questions import load_questions
    first = load_questions()[0]
    df = frame(grass=np.random.default_rng(1).choice(first.option_codes, 300).astype(float),
               year=[2000] * 300)
    df = df.rename(columns={"grass": first.gss_variable})
    out, s = extract(df, first, min_option_count=1)
    assert s.n_kept == 300 and len(s.counts) == len(first.options)

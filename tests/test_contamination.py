from datetime import date, datetime

import pandas as pd
import pytest

from data.contamination import (
    STATUSES, ContaminationError, ModelCutoff, classify, flag_questions, load_models,
    parse_cutoff, release_date,
)
from data.questions import Question, load_questions


def q(start, end, qid="q"):
    return Question(id=qid, gss_variable="v", text="t", options=("a", "b"), option_codes=(1, 2),
                    ordinal=False, category="factual", years=(start, end), contamination_note="note")


# ---- parse_cutoff ---------------------------------------------------------

def test_parse_cutoff_forms():
    assert parse_cutoff("2025-03-15") == date(2025, 3, 15)
    assert parse_cutoff("2025-03") == date(2025, 3, 31)
    assert parse_cutoff("2024-02") == date(2024, 2, 29)   # leap year
    assert parse_cutoff("2023-02") == date(2023, 2, 28)
    assert parse_cutoff(date(2025, 1, 1)) == date(2025, 1, 1)


@pytest.mark.parametrize("bad", [
    "2025", 2025, "2025-13", "2025-00", "2025-02-30", "2025-1-5", "25-01", "January 2025", "", " 2025-01 ",
    None, True, 20250101, 2025.5, datetime(2025, 1, 1), "2025-01-01T00:00", "1989-12", "2101-01", ["2025-01"],
])
def test_parse_cutoff_rejects(bad):
    with pytest.raises(ContaminationError):
        parse_cutoff(bad)


# ---- release_date ---------------------------------------------------------

@pytest.mark.parametrize("year,lag,expected", [
    (2022, 12, date(2023, 12, 31)), (2022, 0, date(2022, 12, 31)), (2022, 1, date(2023, 1, 31)),
    (2023, 2, date(2024, 2, 29)), (2022, 14, date(2024, 2, 29)), (1972, 12, date(1973, 12, 31)),
])
def test_release_date(year, lag, expected):
    assert release_date(year, lag) == expected


# ---- classify -------------------------------------------------------------

CUT = date(2024, 6, 30)


def test_all_data_public_before_cutoff_is_likely_seen():
    assert classify(q(1972, 2010), CUT) == "likely_seen"


def test_straddling_is_possibly_seen():
    assert classify(q(2010, 2023), CUT) == "possibly_seen"   # 2023 data public Dec 2024


def test_only_post_cutoff_years():
    assert classify(q(2024, 2024), CUT) == "post_cutoff"     # public Dec 2025
    assert classify(q(2023, 2024), CUT) == "post_cutoff"     # 2023 public Dec 2024 > cutoff


def test_boundary_end_release_equals_cutoff_is_likely_seen():
    c = release_date(2022, 12)
    assert classify(q(2000, 2022), c) == "likely_seen"
    assert classify(q(2000, 2022), date(c.year, c.month, c.day - 1)) == "possibly_seen"


def test_boundary_start_release_equals_cutoff_is_possibly_seen():
    c = release_date(2022, 12)
    assert classify(q(2022, 2024), c) == "possibly_seen"
    assert classify(q(2022, 2024), date(c.year, c.month, c.day - 1)) == "post_cutoff"


def test_single_year_question_never_possibly_seen_unless_it_is_the_boundary():
    for cut in (date(2020, 1, 1), date(2030, 1, 1)):
        assert classify(q(2018, 2018), cut) in ("likely_seen", "post_cutoff")


def test_lag_changes_status():
    c = date(2023, 6, 30)
    assert classify(q(2022, 2022), c, lag_months=0) == "likely_seen"
    assert classify(q(2022, 2022), c, lag_months=12) == "post_cutoff"


@pytest.mark.parametrize("lag", [-1, 61, 1.5, True, "12", None])
def test_bad_lag(lag):
    with pytest.raises(ContaminationError):
        classify(q(2000, 2001), CUT, lag)


@pytest.mark.parametrize("cut", ["2024-06-30", None, 2024, datetime(2024, 1, 1)])
def test_classify_requires_date(cut):
    with pytest.raises(ContaminationError):
        classify(q(2000, 2001), cut)


# ---- table ----------------------------------------------------------------

def test_flag_table_complete_and_closed():
    qs = [q(1972, 2000, "a"), q(2010, 2023, "b"), q(2024, 2024, "c")]
    ms = [ModelCutoff("m1", date(2024, 6, 30)), ModelCutoff("m2", date(1999, 1, 1))]
    t = flag_questions(qs, ms)
    assert len(t) == 6 and not t.isna().any().any()
    assert set(t["status"].astype(str)) <= set(STATUSES)
    assert isinstance(t["status"].dtype, pd.CategoricalDtype)
    got = {(r.question_id, r.model): str(r.status) for r in t.itertuples()}
    assert got[("a", "m1")] == "likely_seen" and got[("c", "m1")] == "post_cutoff"
    assert got[("b", "m1")] == "possibly_seen" and got[("a", "m2")] == "possibly_seen" and got[("c", "m2")] == "post_cutoff"
    assert (t["note"] == "note").all()


def test_flag_table_requires_inputs():
    with pytest.raises(ContaminationError):
        flag_questions([], [ModelCutoff("m", CUT)])
    with pytest.raises(ContaminationError):
        flag_questions([q(2000, 2001)], [])


def test_shipped_configs_produce_flags_for_every_question():
    models, lag = load_models()
    t = flag_questions(load_questions(), models, lag)
    assert len(t) == len(load_questions()) * len(models) and not t["status"].isna().any()


# ---- config loading --------------------------------------------------------

def write(tmp_path, text):
    p = tmp_path / "m.yaml"
    p.write_text(text)
    return p


def test_load_models_ok(tmp_path):
    models, lag = load_models(write(tmp_path, 'release_lag_months: 6\nmodels:\n  - {name: a-1, cutoff: "2024-01"}\n  - {name: "b:2", cutoff: 2024-02-01}\n'))
    assert [m.name for m in models] == ["a-1", "b:2"] and lag == 6
    assert models[1].cutoff == date(2024, 2, 1)


def test_default_lag(tmp_path):
    assert load_models(write(tmp_path, 'models:\n  - {name: a, cutoff: "2024-01"}\n'))[1] == 12


@pytest.mark.parametrize("text", [
    "", "models: []", "models: 5", "other: 1", "models:\n  - {name: a}",
    'models:\n  - {name: a, cutoff: "2024-01", extra: 1}',
    'models:\n  - {name: "a b", cutoff: "2024-01"}',
    'models:\n  - {name: "", cutoff: "2024-01"}',
    'models:\n  - {name: a, cutoff: "2024-13"}',
    'models:\n  - {name: a, cutoff: 2024}',
    'models:\n  - {name: a, cutoff: "2024-01"}\n  - {name: a, cutoff: "2024-02"}',
    'release_lag_months: -1\nmodels:\n  - {name: a, cutoff: "2024-01"}',
    'models:\n  - {name: a, cutoff: "2024-01"}\nmodels: []',
    'models: [unclosed',
    "models: !!python/object/apply:os.system ['true']",
    "models:\n  - &x {name: a, cutoff: '2024-01'}\n  - *x",
    "- just a list",
])
def test_load_models_rejects(tmp_path, text):
    with pytest.raises(ContaminationError):
        load_models(write(tmp_path, text))


def test_load_models_file_problems(tmp_path):
    with pytest.raises(ContaminationError, match="cannot read"):
        load_models(tmp_path / "nope.yaml")
    big = tmp_path / "big.yaml"
    big.write_text("#" * 100_001)
    with pytest.raises(ContaminationError, match="exceeds"):
        load_models(big)
    bad = tmp_path / "bin.yaml"
    bad.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(ContaminationError, match="cannot read"):
        load_models(bad)


def test_too_many_models(tmp_path):
    body = "".join(f'  - {{name: m{i}, cutoff: "2024-01"}}\n' for i in range(51))
    with pytest.raises(ContaminationError, match="too many"):
        load_models(write(tmp_path, "models:\n" + body))

import hashlib
import os

import numpy as np
import pandas as pd
import pytest

from data import build
from data.build import BuildError, SchemaError, TABLE_COLUMNS, build_table, read_table, validate_schema, write_table
from data.clean import Rule
from data.questions import Question


def mkq(qid, var, years=(2000, 2010)):
    return Question(id=qid, gss_variable=var, text="t", options=("a", "b"), option_codes=(1, 2),
                    ordinal=False, category="factual", years=years, contamination_note="n")


QS = [mkq("grass_legal", "grass"), mkq("cappun_favor", "cappun", (2004, 2010))]

RULES = [
    Rule("grass", {0: "IAP", 8: "DK", 9: "NA"}, valid_values=(1, 2)),
    Rule("cappun", {0: "IAP", 8: "DK", 9: "NA"}, valid_values=(1, 2)),
    Rule("age", {99: "NA"}, valid_range=(18, 89)),
    Rule("sex", {}, valid_values=(1, 2)),
    Rule("race", {}, valid_values=(1, 2, 3)),
    Rule("degree", {8: "DK", 9: "NA"}, valid_range=(0, 4)),
    Rule("realinc", {}, valid_range=(0, 10**7)),
    Rule("region", {}, valid_range=(1, 9)),
    Rule("partyid", {8: "DK", 9: "NA"}, valid_range=(0, 7)),
    Rule("srcbelt", {}, valid_range=(1, 6)),
]


def raw(n=60, seed=0):
    r = np.random.default_rng(seed)
    return pd.DataFrame({
        "id": np.arange(1, n + 1), "year": r.choice([2000, 2004, 2008], n),
        "wtssall": r.uniform(0.5, 2, n), "grass": r.choice([1, 2, 0, 8], n).astype(float),
        "cappun": r.choice([1, 2, 9], n).astype(float), "age": r.integers(18, 90, n).astype(float),
        "sex": r.integers(1, 3, n), "race": r.integers(1, 4, n), "degree": r.integers(0, 5, n),
        "realinc": r.uniform(1000, 90000, n), "region": r.integers(1, 10, n),
        "partyid": r.integers(0, 8, n), "srcbelt": r.integers(1, 7, n),
    })


def built(**kw):
    return build_table(kw.pop("raw_df", raw()), QS, RULES, **kw)


# ---- build_table ---------------------------------------------------------

def test_table_shape_and_schema():
    t, summ = built()
    assert list(t.columns) == list(TABLE_COLUMNS)
    assert set(t["question_id"]) == {"grass_legal", "cappun_favor"}
    assert not t.duplicated(["respondent_id", "question_id"]).any()
    assert set(summ) == {"grass_legal", "cappun_favor"}
    assert summ["grass_legal"].n_kept == (t["question_id"] == "grass_legal").sum()
    validate_schema(t, QS)


def test_values_match_source_rows():
    df = raw()
    t, _ = built(raw_df=df)
    row = t[t["question_id"] == "grass_legal"].iloc[0]
    year, i = row["respondent_id"].split("-")
    src = df[(df["year"] == int(year)) & (df["id"] == int(i))].iloc[0]
    assert row["answer"] == int(src["grass"]) - 1
    assert row["weight"] == pytest.approx(src["wtssall"])
    assert str(row["sex"]) == {1: "male", 2: "female"}[int(src["sex"])]


def test_respondent_only_in_questions_they_answered():
    df = raw()
    df.loc[0, "cappun"] = 9.0  # DK/NA -> dropped for cappun only
    df.loc[0, "year"], df.loc[0, "grass"] = 2008, 1.0
    t, _ = built(raw_df=df)
    rid = f"2008-{int(df.loc[0, 'id'])}"
    assert set(t[t["respondent_id"] == rid]["question_id"]) == {"grass_legal"}


def test_question_year_range_applied():
    t, _ = built()
    assert t.loc[t["question_id"] == "cappun_favor", "year"].min() >= 2004


def test_deterministic_and_input_unchanged():
    df = raw()
    before = df.copy()
    a, _ = built(raw_df=df)
    b, _ = built(raw_df=df)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(df, before)


def test_same_respondent_same_demographics_across_questions():
    t, _ = built()
    for col in ("age_band", "party"):
        assert (t.groupby("respondent_id", observed=True)[col].nunique() <= 1).all()


def test_missing_demographic_allowed_but_required_fields_clean():
    df = raw()
    df.loc[:5, "age"] = 99.0
    t, _ = built(raw_df=df)
    assert t["age_band"].isna().sum() > 0
    assert not t[list(build.REQUIRED_COLUMNS)].isna().any().any()


def test_unknown_special_code_stops_build():
    df = raw()
    df.loc[3, "grass"] = 5.0
    with pytest.raises(ValueError, match="unrecognised"):
        built(raw_df=df)


def test_rule_coverage_required():
    with pytest.raises(BuildError, match="no cleaning rule.*cappun"):
        build_table(raw(), QS, [r for r in RULES if r.column != "cappun"])


@pytest.mark.parametrize("col", ["id", "year", "wtssall"])
def test_missing_core_column(col):
    with pytest.raises(Exception):
        built(raw_df=raw().drop(columns=col))


def test_bad_ids():
    for bad in (np.nan, -1, 1.5, 10**7):
        df = raw()
        df["id"] = df["id"].astype(float)
        df.loc[0, "id"] = bad
        with pytest.raises(BuildError, match="whole numbers"):
            built(raw_df=df)


def test_duplicate_respondents_rejected():
    df = raw()
    df.loc[1, ["id", "year"]] = df.loc[0, ["id", "year"]].to_numpy()
    with pytest.raises(BuildError, match="duplicate"):
        built(raw_df=df)


def test_same_id_in_different_years_is_fine():
    df = raw(3)
    df["id"], df["year"], df["grass"], df["cappun"] = 1, [2000, 2004, 2008], 1.0, 1.0
    t, _ = built(raw_df=df)
    assert t["respondent_id"].nunique() == 3


def test_no_questions():
    with pytest.raises(BuildError, match="no questions"):
        build_table(raw(), [], RULES)


# ---- validate_schema (corrupting a good table) ----------------------------

@pytest.fixture
def good():
    return built()[0]


def broken(good, fn, match):
    df = good.copy()
    fn(df)
    with pytest.raises(SchemaError, match=match):
        validate_schema(df, QS)


def test_schema_rejections(good):
    broken(good, lambda d: d.drop(columns="sex", inplace=True), "columns must be")
    broken(good, lambda d: d.__setitem__("extra", 1), "columns must be")
    broken(good, lambda d: d.__setitem__("weight", d["weight"].astype("float32")), "weight must be float64")
    broken(good, lambda d: d.__setitem__("answer", d["answer"].astype("int32")), "answer must be int64")
    broken(good, lambda d: d.__setitem__("weight", np.where(d.index == 0, 0.0, d["weight"])), "weight must be")
    broken(good, lambda d: d.__setitem__("weight", np.where(d.index == 0, np.inf, d["weight"])), "weight must be")
    broken(good, lambda d: d.__setitem__("weight", np.where(d.index == 0, np.nan, d["weight"])), "missing values")
    broken(good, lambda d: d.__setitem__("answer", np.where(d.index == 0, 2, d["answer"])), "outside the question")
    broken(good, lambda d: d.__setitem__("answer", np.where(d.index == 0, -1, d["answer"])), "outside the question")
    broken(good, lambda d: d.__setitem__("question_id", "mystery"), "not in registry")
    broken(good, lambda d: d.__setitem__("respondent_id", "Jane Doe"), "respondent_id must look")
    broken(good, lambda d: d.__setitem__("year", d["year"] + 1), "year")
    broken(good, lambda d: d.__setitem__("year", np.where(d.index == 0, 1990, d["year"]).astype("int64")), "year")
    broken(good, lambda d: d.__setitem__("sex", d["sex"].astype(object)), "sex must be a categorical")
    broken(good, lambda d: d.__setitem__("sex", d["sex"].cat.rename_categories(["x", "y"])), "sex must be a categorical")
    broken(good, lambda d: d.__setitem__("age_band", d["age_band"].cat.as_unordered()), "age_band must be")
    broken(good, lambda d: d.drop(d.index, inplace=True), "empty")


def test_duplicate_rows_rejected(good):
    df = pd.concat([good, good.iloc[[0]]], ignore_index=True)
    with pytest.raises(SchemaError, match="duplicate"):
        validate_schema(df, QS)


def test_inconsistent_demographics_for_one_respondent(good):
    two = good.groupby("respondent_id").filter(lambda g: len(g) == 2)
    rid = two["respondent_id"].iloc[0]
    df = good.copy()
    i = df.index[df["respondent_id"] == rid][0]
    cur = df.loc[i, "sex"]
    df.loc[i, "sex"] = "male" if cur == "female" else "female"
    with pytest.raises(SchemaError, match="differs between rows"):
        validate_schema(df, QS)


def test_not_a_dataframe():
    with pytest.raises(SchemaError):
        validate_schema([1, 2], QS)


# ---- write / read ---------------------------------------------------------

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_round_trip_preserves_everything(tmp_path, good):
    p = write_table(good, QS, "respondents.parquet", base_dir=tmp_path)
    back = read_table(QS, "respondents.parquet", base_dir=tmp_path)
    pd.testing.assert_frame_equal(back, good)
    assert p.parent == tmp_path.resolve() and oct(p.stat().st_mode & 0o777) == "0o644"


def test_write_is_deterministic(tmp_path, good):
    a = write_table(good, QS, "a.parquet", base_dir=tmp_path)
    b = write_table(good, QS, "b.parquet", base_dir=tmp_path)
    assert sha(a) == sha(b)
    c = write_table(built()[0], QS, "c.parquet", base_dir=tmp_path)
    assert sha(a) == sha(c)


def test_invalid_table_blocks_write_and_keeps_existing(tmp_path, good):
    write_table(good, QS, "t.parquet", base_dir=tmp_path)
    before = sha(tmp_path / "t.parquet")
    bad = good.copy()
    bad["weight"] = -1.0
    with pytest.raises(SchemaError):
        write_table(bad, QS, "t.parquet", base_dir=tmp_path)
    with pytest.raises(SchemaError):
        write_table(bad, QS, "new.parquet", base_dir=tmp_path)
    assert sha(tmp_path / "t.parquet") == before and not (tmp_path / "new.parquet").exists()
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".part")]


def test_failed_replace_cleans_temp(tmp_path, good, monkeypatch):
    def boom(*a):
        raise OSError("disk full")
    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        write_table(good, QS, "t.parquet", base_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("rel", ["../x.parquet", "/abs.parquet", "a\\b.parquet"])
def test_output_path_traversal_rejected(tmp_path, good, rel):
    with pytest.raises(ValueError):
        write_table(good, QS, rel, base_dir=tmp_path / "base")


def test_output_must_be_parquet(tmp_path, good):
    for name in ("t.csv", "t.pkl", "t"):
        with pytest.raises(BuildError, match="parquet"):
            write_table(good, QS, name, base_dir=tmp_path)


def test_symlink_escape_rejected(tmp_path, good):
    base, outside = tmp_path / "base", tmp_path / "outside"
    base.mkdir()
    outside.mkdir()
    os.symlink(outside, base / "link")
    with pytest.raises(ValueError):
        write_table(good, QS, "link/t.parquet", base_dir=base)
    assert list(outside.iterdir()) == []


def test_read_detects_tampering_and_corruption(tmp_path, good):
    write_table(good, QS, "t.parquet", base_dir=tmp_path)
    tampered = good.copy()
    tampered["weight"] = tampered["weight"] * -1
    tampered.to_parquet(tmp_path / "t.parquet", index=False)
    with pytest.raises(SchemaError, match="weight"):
        read_table(QS, "t.parquet", base_dir=tmp_path)
    (tmp_path / "junk.parquet").write_bytes(os.urandom(200))
    with pytest.raises(SchemaError, match="could not read"):
        read_table(QS, "junk.parquet", base_dir=tmp_path)
    with pytest.raises(BuildError, match="not found"):
        read_table(QS, "nope.parquet", base_dir=tmp_path)


def test_read_with_wrong_registry_rejected(tmp_path, good):
    write_table(good, QS, "t.parquet", base_dir=tmp_path)
    with pytest.raises(SchemaError, match="not in registry"):
        read_table(QS[:1], "t.parquet", base_dir=tmp_path)

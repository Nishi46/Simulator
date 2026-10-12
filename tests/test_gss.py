import hashlib
import os

import numpy as np
import pandas as pd
import pytest

from data.gss import GssReadError, read_gss

FRAME = pd.DataFrame({
    "year": [2000, 2002, 2004, 2006],
    "id": [1, 2, 3, 4],
    "age": [30, 45, 89, 22],
    "grass": [1.0, 2.0, np.nan, 1.0],
    "wtssall": [0.9, 1.1, 1.0, 1.3],
    "unused": [9, 9, 9, 9],
})


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture
def dta(tmp_path):
    p = tmp_path / "gss.dta"
    FRAME.to_stata(p, write_index=False)
    return p


@pytest.fixture
def csv(tmp_path):
    p = tmp_path / "gss.csv"
    FRAME.to_csv(p, index=False)
    return p


def rd(p, cols, **kw):
    return read_gss(p.name, cols, sha256=kw.pop("sha256", sha(p)), base_dir=p.parent, **kw)


@pytest.mark.parametrize("fixture", ["dta", "csv"])
def test_reads_only_requested_columns_in_order(fixture, request):
    p = request.getfixturevalue(fixture)
    df = rd(p, ["grass", "year", "age"], numeric=["grass", "year"])
    assert list(df.columns) == ["grass", "year", "age"]
    assert len(df) == 4 and df["year"].tolist() == [2000, 2002, 2004, 2006]
    assert df["grass"].isna().tolist() == [False, False, True, False]
    assert "unused" not in df.columns


def test_mixed_case_file_columns_are_lowercased(tmp_path):
    p = tmp_path / "gss.dta"
    FRAME.rename(columns={"age": "AGE", "grass": "Grass"}).to_stata(p, write_index=False)
    assert list(rd(p, ["age", "grass"]).columns) == ["age", "grass"]
    c = tmp_path / "gss.csv"
    FRAME.rename(columns={"age": "AGE"}).to_csv(c, index=False)
    assert list(rd(c, ["age"]).columns) == ["age"]


@pytest.mark.parametrize("fixture", ["dta", "csv"])
def test_missing_column_named_in_error(fixture, request):
    p = request.getfixturevalue(fixture)
    with pytest.raises(GssReadError, match="nope") as e:
        rd(p, ["year", "nope"])
    assert "year" not in str(e.value).split("missing")[1]


def test_duplicate_columns_in_csv_rejected(tmp_path):
    p = tmp_path / "gss.csv"
    p.write_text("year,age,age\n2000,30,31\n")
    with pytest.raises(GssReadError, match="duplicate|ambiguous"):
        rd(p, ["age"])


def test_case_collision_in_csv_rejected(tmp_path):
    p = tmp_path / "gss.csv"
    p.write_text("age,AGE\n30,31\n")
    with pytest.raises(GssReadError, match="ambiguous"):
        rd(p, ["age"])


def test_duplicate_columns_in_unrequested_part_do_not_matter(tmp_path):
    p = tmp_path / "gss.csv"
    p.write_text("year,x,x\n2000,1,2\n")
    assert rd(p, ["year"])["year"].tolist() == [2000]


@pytest.mark.parametrize("cols", [[], None, "year", ["year", "year"], ["Year"], [""], ["a b"], [5],
                                  ["year;drop"], ["1year"], ["x" * 33], [f"c{i}" for i in range(501)]])
def test_bad_column_requests(dta, cols):
    with pytest.raises(GssReadError):
        rd(dta, cols)


def test_numeric_must_be_subset(dta):
    with pytest.raises(GssReadError, match="not in requested"):
        rd(dta, ["year"], numeric=["age"])


def test_text_in_numeric_column_rejected(tmp_path):
    p = tmp_path / "gss.csv"
    p.write_text("year,age\n2000,30\n2002,thirty\n")
    with pytest.raises(GssReadError, match="not numeric"):
        rd(p, ["year", "age"], numeric=["age"])
    assert rd(p, ["year", "age"], numeric=["year"])["year"].tolist() == [2000, 2002]


def test_bool_column_not_numeric(tmp_path):
    p = tmp_path / "gss.csv"
    p.write_text("flag\nTrue\nFalse\n")
    with pytest.raises(GssReadError, match="not numeric"):
        rd(p, ["flag"], numeric=["flag"])


# ---- verification and file checks ---------------------------------------

def test_checksum_mismatch_blocks_parsing(dta, monkeypatch):
    called = []
    monkeypatch.setattr(pd, "read_stata", lambda *a, **k: called.append(1))
    with pytest.raises(GssReadError, match="sha256 mismatch"):
        rd(dta, ["year"], sha256="0" * 64)
    assert called == []


def test_tampered_file_after_pinning_rejected(csv):
    pinned = sha(csv)
    csv.write_text(csv.read_text() + "2010,9,50,1,1.0,9\n")
    with pytest.raises(GssReadError, match="sha256 mismatch"):
        rd(csv, ["year"], sha256=pinned)


@pytest.mark.parametrize("s", ["", "abc", "A" * 64, None, 5])
def test_bad_sha_param(dta, s):
    with pytest.raises(GssReadError, match="sha256"):
        rd(dta, ["year"], sha256=s)


def test_wrong_extension(tmp_path):
    p = tmp_path / "gss.xlsx"
    p.write_bytes(b"x")
    with pytest.raises(GssReadError, match="unsupported"):
        rd(p, ["year"])
    q = tmp_path / "gss.csv.exe"
    q.write_bytes(b"x")
    with pytest.raises(GssReadError, match="unsupported"):
        rd(q, ["year"])


def test_missing_file_and_directory(tmp_path):
    with pytest.raises(GssReadError, match="not found"):
        read_gss("nope.csv", ["year"], sha256="0" * 64, base_dir=tmp_path)
    (tmp_path / "d.csv").mkdir()
    with pytest.raises(GssReadError, match="not found"):
        read_gss("d.csv", ["year"], sha256="0" * 64, base_dir=tmp_path)


def test_empty_and_header_only_files(tmp_path):
    e = tmp_path / "e.csv"
    e.write_bytes(b"")
    with pytest.raises(GssReadError, match="empty"):
        rd(e, ["year"])
    h = tmp_path / "h.csv"
    h.write_text("year,age\n")
    with pytest.raises(GssReadError, match="no data rows"):
        rd(h, ["year"])
    d = tmp_path / "h.dta"
    FRAME.iloc[0:0].to_stata(d, write_index=False)
    with pytest.raises(GssReadError, match="no data rows"):
        rd(d, ["year"])


def test_size_cap(csv):
    with pytest.raises(GssReadError, match="over the"):
        rd(csv, ["year"], max_bytes=10)


@pytest.mark.parametrize("mb", [0, -1, True, 1.5, None])
def test_bad_max_bytes(csv, mb):
    with pytest.raises(GssReadError, match="max_bytes"):
        rd(csv, ["year"], max_bytes=mb)


def test_corrupt_files_wrapped(tmp_path):
    d = tmp_path / "bad.dta"
    d.write_bytes(os.urandom(300))
    with pytest.raises(GssReadError, match="could not parse"):
        rd(d, ["year"])
    c = tmp_path / "bad.csv"
    c.write_bytes(b"\xff\xfe\x00\x00garbage\n1,2\n")
    with pytest.raises(GssReadError):
        rd(c, ["year"])
    t = tmp_path / "trunc.dta"
    FRAME.to_stata(t, write_index=False)
    t.write_bytes(t.read_bytes()[:120])
    with pytest.raises(GssReadError, match="could not parse"):
        rd(t, ["year"])


def test_path_traversal_and_symlink_escape(tmp_path):
    base, outside = tmp_path / "base", tmp_path / "outside"
    base.mkdir()
    outside.mkdir()
    (outside / "gss.csv").write_text("year\n2000\n")
    for bad in ["../outside/gss.csv", "/etc/passwd.csv"]:
        with pytest.raises(ValueError):
            read_gss(bad, ["year"], sha256="0" * 64, base_dir=base)
    os.symlink(outside / "gss.csv", base / "link.csv")
    with pytest.raises(ValueError):
        read_gss("link.csv", ["year"], sha256="0" * 64, base_dir=base)


def test_no_pickle_or_code_execution_formats(tmp_path):
    for name in ["gss.pkl", "gss.pickle", "gss.py", "gss.parquet"]:
        p = tmp_path / name
        p.write_bytes(b"x")
        with pytest.raises(GssReadError, match="unsupported"):
            rd(p, ["year"])


def test_special_code_values_returned_untouched(tmp_path):
    p = tmp_path / "gss.csv"
    p.write_text("age,grass\n89,1\n99,8\n0,9\n")
    df = rd(p, ["age", "grass"])
    assert df["age"].tolist() == [89, 99, 0] and df["grass"].tolist() == [1, 8, 9]

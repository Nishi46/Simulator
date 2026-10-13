import json
import os
from datetime import datetime, timezone

import pytest

from eval.manifest import ManifestError, ids_sha256, load_manifest, verify_against, write_manifest
from eval.splits import SPLIT_NAMES, Splits, make_splits
from tests.test_splits import people

SRC = "a" * 64
CFG = "0123456789ab"
FR = {"train": 0.6, "calibration": 0.2, "test": 0.2}
NOW = lambda: datetime(2026, 10, 13, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def splits():
    return make_splits(people(300), seed=3)


def write(tmp_path, splits, **kw):
    args = dict(seed=3, fractions=FR, strata_cols=["year", "party"], source_sha256=SRC,
                config_hash=CFG, base_dir=tmp_path, now=NOW)
    args.update(kw)
    return write_manifest(splits, **args)


def edit(path, fn):
    d = json.loads(path.read_text())
    fn(d)
    path.write_text(json.dumps(d, sort_keys=True, indent=1))


def test_round_trip(tmp_path, splits):
    path, digest = write(tmp_path, splits)
    m = load_manifest(base_dir=tmp_path, expected_digest=digest)
    assert m.ids == splits.ids and m.digest == digest
    assert (m.seed, m.config_hash, m.source_sha256, m.created_utc) == (3, CFG, SRC, "2026-10-13T12:00:00Z")
    assert m.fractions == FR and m.strata_cols == ("year", "party")
    assert oct(path.stat().st_mode & 0o777) == "0o644"
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".part")]


def test_file_contains_only_ids_and_run_metadata(tmp_path, splits):
    path, _ = write(tmp_path, splits)
    d = json.loads(path.read_text())
    assert set(d) == {"version", "seed", "fractions", "strata_cols", "source_sha256", "config_hash",
                      "created_utc", "splits", "digest"}
    assert all(set(v) == {"n", "sha256", "ids"} for v in d["splits"].values())
    text = path.read_text()
    for forbidden in ("party", "dem", "weight", "answer", "age"):
        assert forbidden not in json.dumps(d["splits"]), forbidden
    assert all("-" in i and i.replace("-", "").isdigit() for v in d["splits"].values() for i in v["ids"])
    assert text.endswith("\n")


def test_same_inputs_same_bytes(tmp_path, splits):
    a, da = write(tmp_path, splits, rel_path="a.json")
    b, db = write(tmp_path, splits, rel_path="b.json")
    assert a.read_bytes() == b.read_bytes() and da == db


def test_refuses_overwrite_without_force(tmp_path, splits):
    path, _ = write(tmp_path, splits)
    original = path.read_bytes()
    other = make_splits(people(300), seed=99)
    with pytest.raises(ManifestError, match="already exists"):
        write(tmp_path, other)
    assert path.read_bytes() == original and not [p for p in tmp_path.iterdir() if p.name.endswith(".part")]
    write(tmp_path, other, force=True)
    assert path.read_bytes() != original
    load_manifest(base_dir=tmp_path)


# ---- tamper detection ------------------------------------------------------

def test_editing_an_id_is_detected(tmp_path, splits):
    path, _ = write(tmp_path, splits)
    edit(path, lambda d: d["splits"]["test"]["ids"].__setitem__(0, "1999-1"))
    with pytest.raises(ManifestError, match="checksum mismatch"):
        load_manifest(base_dir=tmp_path)


def test_moving_an_id_between_splits_is_detected(tmp_path, splits):
    path, _ = write(tmp_path, splits)

    def move(d):
        x = d["splits"]["test"]["ids"].pop()
        d["splits"]["train"]["ids"].append(x)
        d["splits"]["train"]["ids"].sort()
    edit(path, move)
    with pytest.raises(ManifestError):
        load_manifest(base_dir=tmp_path)


def test_editing_ids_and_split_hash_still_caught_by_digest(tmp_path, splits):
    path, _ = write(tmp_path, splits)

    def forge(d):
        ids = d["splits"]["test"]["ids"]
        d["splits"]["test"]["ids"] = ids[1:]
        d["splits"]["test"]["n"] = len(ids) - 1
        d["splits"]["test"]["sha256"] = ids_sha256(ids[1:])
    edit(path, forge)
    with pytest.raises(ManifestError, match="digest mismatch"):
        load_manifest(base_dir=tmp_path)


@pytest.mark.parametrize("field,value", [
    ("seed", 4), ("config_hash", "ffffffffffff"), ("source_sha256", "b" * 64),
    ("created_utc", "2020-01-01T00:00:00Z"), ("strata_cols", ["year"]),
    ("fractions", {"train": 0.5, "calibration": 0.25, "test": 0.25}),
])
def test_editing_metadata_is_detected(tmp_path, splits, field, value):
    path, _ = write(tmp_path, splits)
    edit(path, lambda d: d.__setitem__(field, value))
    with pytest.raises(ManifestError, match="digest mismatch"):
        load_manifest(base_dir=tmp_path)


def test_pinned_digest_catches_a_fully_forged_manifest(tmp_path, splits):
    _, pinned = write(tmp_path, splits)
    other = make_splits(people(300), seed=99)
    write(tmp_path, other, force=True)
    load_manifest(base_dir=tmp_path)                       # internally consistent...
    with pytest.raises(ManifestError, match="pinned"):     # ...but not the one we froze
        load_manifest(base_dir=tmp_path, expected_digest=pinned)


@pytest.mark.parametrize("mutate", [
    lambda d: d.pop("digest"),
    lambda d: d.__setitem__("extra", 1),
    lambda d: d.__setitem__("version", 2),
    lambda d: d.__setitem__("version", True),
    lambda d: d.__setitem__("digest", "xyz"),
    lambda d: d["splits"].pop("test"),
    lambda d: d["splits"]["train"].__setitem__("extra", 1),
    lambda d: d["splits"]["train"].__setitem__("ids", "notalist"),
    lambda d: d["splits"]["train"].__setitem__("n", 1),
    lambda d: d["splits"]["train"].__setitem__("ids", []),
    lambda d: d["splits"]["train"]["ids"].__setitem__(0, "Jane Doe"),
    lambda d: d["splits"]["train"]["ids"].__setitem__(0, 5),
    lambda d: d.__setitem__("seed", -1),
    lambda d: d.__setitem__("seed", True),
    lambda d: d.__setitem__("created_utc", "yesterday"),
    lambda d: d.__setitem__("strata_cols", []),
    lambda d: d.__setitem__("strata_cols", ["../etc"]),
])
def test_structural_corruption_rejected(tmp_path, splits, mutate):
    path, _ = write(tmp_path, splits)
    edit(path, mutate)
    with pytest.raises(ManifestError):
        load_manifest(base_dir=tmp_path)


def test_overlapping_splits_rejected(tmp_path, splits):
    path, _ = write(tmp_path, splits)

    def overlap(d):
        d["splits"]["test"]["ids"].append(d["splits"]["train"]["ids"][0])
        d["splits"]["test"]["ids"].sort()
    edit(path, overlap)
    with pytest.raises(ManifestError):
        load_manifest(base_dir=tmp_path)


def test_unparseable_files(tmp_path, splits):
    path, _ = write(tmp_path, splits)
    for content in (b"", b"not json", b"[1,2]", b"\xff\xfe", b'{"a": NaN}', b'{"a":1,"a":2}'):
        path.write_bytes(content)
        with pytest.raises(ManifestError):
            load_manifest(base_dir=tmp_path)
    path.write_bytes(b" " * (100 * 1024 * 1024 + 1))
    with pytest.raises(ManifestError, match="too large"):
        load_manifest(base_dir=tmp_path)
    path.unlink()
    with pytest.raises(ManifestError, match="cannot read"):
        load_manifest(base_dir=tmp_path)


# ---- write-side validation -------------------------------------------------

def test_write_rejects_invalid_inputs(tmp_path, splits):
    bad_ids = {n: list(v) for n, v in splits.ids.items()}
    for mutate, kw in [
        (lambda i: i["test"].append(i["train"][0]), {}),
        (lambda i: i.__setitem__("test", []), {}),
        (lambda i: i["train"].reverse(), {}),
        (lambda i: i["train"].append("junk"), {}),
    ]:
        ids = {n: list(v) for n, v in bad_ids.items()}
        mutate(ids)
        with pytest.raises(ManifestError):
            write(tmp_path, Splits(ids, 1, 0))
    for kw in [dict(seed=-1), dict(seed=True), dict(source_sha256="abc"), dict(config_hash="XYZ"),
               dict(strata_cols=[]), dict(strata_cols=["Bad Col"]),
               dict(fractions={"train": 1, "calibration": 0, "test": 0})]:
        with pytest.raises(ManifestError):
            write(tmp_path, splits, **kw)
    assert list(tmp_path.iterdir()) == []


# ---- path safety -----------------------------------------------------------

@pytest.mark.parametrize("rel", ["../m.json", "/abs.json", "a\\b.json"])
def test_path_traversal_rejected(tmp_path, splits, rel):
    with pytest.raises(ValueError):
        write(tmp_path / "base", splits, rel_path=rel)
    with pytest.raises(ValueError):
        load_manifest(rel, base_dir=tmp_path / "base")


def test_must_be_json_and_symlink_escape(tmp_path, splits):
    with pytest.raises(ManifestError, match="json"):
        write(tmp_path, splits, rel_path="m.txt")
    base, outside = tmp_path / "base", tmp_path / "outside"
    base.mkdir()
    outside.mkdir()
    os.symlink(outside, base / "link")
    with pytest.raises(ValueError):
        write(base, splits, rel_path="link/m.json")
    assert list(outside.iterdir()) == []


def test_default_location_is_inside_data_dir():
    from data.paths import DATA_DIR
    from eval.manifest import SPLITS_DIR
    assert SPLITS_DIR == DATA_DIR / "splits"


# ---- drift detection -------------------------------------------------------

def test_verify_against_data(tmp_path, splits):
    write(tmp_path, splits)
    m = load_manifest(base_dir=tmp_path)
    everyone = [i for n in SPLIT_NAMES for i in m.ids[n]]
    verify_against(m, source_sha256=SRC, respondent_ids=everyone)
    with pytest.raises(ManifestError, match="source data"):
        verify_against(m, source_sha256="c" * 64)
    with pytest.raises(ManifestError, match="1 new"):
        verify_against(m, respondent_ids=everyone + ["2000-999999"])
    with pytest.raises(ManifestError, match="1 missing"):
        verify_against(m, respondent_ids=everyone[1:])
    verify_against(m)  # nothing to compare is fine

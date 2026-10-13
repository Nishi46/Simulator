import json
import os
import threading
from datetime import datetime, timezone

import pytest

from eval import access
from eval.access import (
    GENESIS, AccessLogError, SplitAccessError, SplitAlreadyUsedError, SplitLockedError,
    filter_to_split, load_split, log_head, read_access_log,
)
from eval.manifest import write_manifest
from eval.splits import SPLIT_NAMES, make_splits
from tests.test_manifest import CFG, FR, NOW, SRC
from tests.test_splits import people

RUN = "abcdef012345"


def freeze(tmp_path, seed=3, n=300):
    s = make_splits(people(n), seed=seed)
    _, digest = write_manifest(s, seed=seed, fractions=FR, strata_cols=["year", "party"],
                               source_sha256=SRC, config_hash=CFG, base_dir=tmp_path, now=NOW,
                               force=True)
    return s, digest


def final(tmp_path, **kw):
    args = dict(allow_test=True, caller="final_run", run_config_hash=RUN, base_dir=tmp_path,
                now=lambda: datetime(2026, 10, 20, 9, 30, 0, tzinfo=timezone.utc))
    args.update(kw)
    return load_split("test", **args)


def log_file(tmp_path):
    return tmp_path / "test_access.log"


# ---- free splits -----------------------------------------------------------

@pytest.mark.parametrize("name", ["train", "calibration"])
def test_train_and_calibration_load_and_never_touch_the_log(tmp_path, name):
    s, _ = freeze(tmp_path)
    assert load_split(name, base_dir=tmp_path) == s.ids[name]
    assert load_split(name, base_dir=tmp_path, allow_test=True) == s.ids[name]
    assert not log_file(tmp_path).exists()


@pytest.mark.parametrize("name", ["TEST", "../test", "val", "", None, 5, ["test"], "test "])
def test_unknown_split_names(tmp_path, name):
    freeze(tmp_path)
    with pytest.raises(SplitAccessError, match="split must be"):
        load_split(name, base_dir=tmp_path)
    assert not log_file(tmp_path).exists()


# ---- the lock --------------------------------------------------------------

def test_default_access_to_test_is_refused(tmp_path):
    freeze(tmp_path)
    with pytest.raises(SplitLockedError):
        load_split("test", base_dir=tmp_path)
    assert not log_file(tmp_path).exists()


@pytest.mark.parametrize("flag", [False, None, 1, "yes", "True", [True]])
def test_only_literal_true_unlocks(tmp_path, flag):
    freeze(tmp_path)
    with pytest.raises(SplitLockedError):
        load_split("test", allow_test=flag, caller="x", run_config_hash=RUN, base_dir=tmp_path)
    assert not log_file(tmp_path).exists()


def test_unlocked_access_returns_ids_and_logs_once(tmp_path):
    s, digest = freeze(tmp_path)
    assert final(tmp_path) == s.ids["test"]
    (e,) = read_access_log(tmp_path)
    assert e == {"seq": 1, "ts": "2026-10-20T09:30:00Z", "kind": "first", "manifest_digest": digest,
                 "config_hash": RUN, "caller": "final_run", "prev": GENESIS}
    assert oct(log_file(tmp_path).stat().st_mode & 0o777) == "0o644"


@pytest.mark.parametrize("kw", [
    dict(caller=None), dict(caller=""), dict(caller="a\nb"), dict(caller='x"y'), dict(caller="x" * 81),
    dict(caller=5), dict(caller="é"), dict(run_config_hash=None), dict(run_config_hash="XYZ"),
    dict(run_config_hash="abcdef01234"), dict(allow_repeat="yes"),
])
def test_bad_caller_or_hash_refused_without_logging(tmp_path, kw):
    freeze(tmp_path)
    with pytest.raises(SplitAccessError):
        final(tmp_path, **kw)
    assert not log_file(tmp_path).exists()


# ---- touch once ------------------------------------------------------------

def test_second_access_refused_and_log_unchanged(tmp_path):
    freeze(tmp_path)
    final(tmp_path)
    before = log_file(tmp_path).read_bytes()
    with pytest.raises(SplitAlreadyUsedError, match="already accessed 1"):
        final(tmp_path, caller="someone_else")
    assert log_file(tmp_path).read_bytes() == before


def test_explicit_repeat_is_allowed_and_recorded(tmp_path):
    s, _ = freeze(tmp_path)
    final(tmp_path)
    first_bytes = log_file(tmp_path).read_bytes()
    assert final(tmp_path, allow_repeat=True, caller="rerun") == s.ids["test"]
    entries = read_access_log(tmp_path)
    assert [e["kind"] for e in entries] == ["first", "repeat"] and entries[1]["seq"] == 2
    assert log_file(tmp_path).read_bytes().startswith(first_bytes)   # append-only
    with pytest.raises(SplitAlreadyUsedError, match="2 time"):
        final(tmp_path)


def test_new_manifest_means_new_test_set(tmp_path):
    freeze(tmp_path, seed=3)
    final(tmp_path)
    freeze(tmp_path, seed=4)
    final(tmp_path, caller="second_manifest")
    assert [e["kind"] for e in read_access_log(tmp_path)] == ["first", "first"]


def test_concurrent_first_access_has_one_winner(tmp_path):
    freeze(tmp_path)
    results = []

    def go():
        try:
            final(tmp_path)
            results.append("ok")
        except SplitAlreadyUsedError:
            results.append("refused")
    threads = [threading.Thread(target=go) for _ in range(10)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(results) == ["ok"] + ["refused"] * 9
    assert len(read_access_log(tmp_path)) == 1


# ---- manifest problems must not burn the test set --------------------------

def test_bad_manifest_logs_nothing(tmp_path):
    freeze(tmp_path)
    p = tmp_path / "manifest.json"
    d = json.loads(p.read_text())
    d["seed"] = 4
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        final(tmp_path)
    assert not log_file(tmp_path).exists()


def test_missing_manifest_and_pinned_digest(tmp_path):
    with pytest.raises(ValueError):
        final(tmp_path)
    assert not log_file(tmp_path).exists()
    freeze(tmp_path)
    with pytest.raises(ValueError, match="pinned"):
        final(tmp_path, expected_digest="f" * 64)
    assert not log_file(tmp_path).exists()
    with pytest.raises(ValueError, match="pinned"):
        load_split("train", base_dir=tmp_path, expected_digest="f" * 64)


# ---- log integrity ---------------------------------------------------------

def three_entries(tmp_path):
    freeze(tmp_path)
    final(tmp_path)
    final(tmp_path, allow_repeat=True, caller="r1")
    final(tmp_path, allow_repeat=True, caller="r2")
    return log_file(tmp_path)


def test_chain_links_entries_and_head_changes(tmp_path):
    p = three_entries(tmp_path)
    e = read_access_log(tmp_path)
    assert e[0]["prev"] == GENESIS and e[1]["prev"] != GENESIS and e[2]["prev"] != e[1]["prev"]
    head = log_head(tmp_path)
    assert head != GENESIS
    final(tmp_path, allow_repeat=True, caller="r3")
    assert log_head(tmp_path) != head


def test_head_of_missing_log_is_genesis(tmp_path):
    assert log_head(tmp_path) == GENESIS and read_access_log(tmp_path) == []


@pytest.mark.parametrize("tamper", [
    lambda L: [L[0].replace("final_run", "someone"), *L[1:]],           # edit an entry
    lambda L: [L[0], L[2], L[1]],                                        # reorder
    lambda L: [L[0], L[2]],                                              # delete from the middle
    lambda L: [L[0], "garbage", L[2]],
    lambda L: [L[0], L[1][:-10], L[2]],                                  # damaged line
    lambda L: [L[0], json.dumps({"seq": 2}), L[2]],
    lambda L: [L[0], L[1].replace('"kind": "repeat"', '"kind": "first"'), L[2]],
])
def test_tampered_log_is_detected_and_blocks_test_access(tmp_path, tamper):
    p = three_entries(tmp_path)
    lines = p.read_text().splitlines()
    p.write_text("\n".join(tamper(lines)) + "\n")
    with pytest.raises(AccessLogError):
        read_access_log(tmp_path)
    with pytest.raises(AccessLogError):
        final(tmp_path, allow_repeat=True, caller="after_tamper")
    with pytest.raises(AccessLogError):
        log_head(tmp_path)
    # free splits are still readable
    assert load_split("train", base_dir=tmp_path)


def test_truncated_final_line_and_binary_log(tmp_path):
    p = three_entries(tmp_path)
    p.write_bytes(p.read_bytes()[:-5])
    with pytest.raises(AccessLogError):
        read_access_log(tmp_path)
    p.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(AccessLogError):
        final(tmp_path, allow_repeat=True)
    p.write_bytes(b"x" * (10 * 1024 * 1024 + 1))
    with pytest.raises(AccessLogError, match="too large"):
        final(tmp_path, allow_repeat=True)


def test_tail_truncation_is_the_documented_gap_but_head_pin_catches_it(tmp_path):
    p = three_entries(tmp_path)
    pinned = log_head(tmp_path)
    lines = p.read_text().splitlines()
    p.write_text("\n".join(lines[:2]) + "\n")       # drop the last entry
    read_access_log(tmp_path)                       # chain still valid on its own...
    assert log_head(tmp_path) != pinned             # ...but the externally recorded head differs


def test_log_symlink_cannot_redirect_writes(tmp_path):
    base, outside = tmp_path / "base", tmp_path / "outside"
    base.mkdir()
    outside.mkdir()
    freeze(base)
    target = outside / "stolen.log"
    os.symlink(target, base / "test_access.log")
    with pytest.raises(ValueError):
        final(base)
    assert not target.exists()


# ---- helper ----------------------------------------------------------------

def test_filter_to_split(tmp_path):
    from data.build import build_table
    from tests.test_build import QS, RULES, raw
    table, _ = build_table(raw(120), QS, RULES)
    ids = sorted(table["respondent_id"].unique())[:10]
    sub = filter_to_split(table, ids)
    assert set(sub["respondent_id"]) == set(ids) and len(sub) < len(table)
    with pytest.raises(SplitAccessError):
        filter_to_split(table.drop(columns="respondent_id"), ids)

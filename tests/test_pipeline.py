import copy
import hashlib
import io
import json
import shutil
import socket
import subprocess
import zipfile

import numpy as np
import pandas as pd
import pytest

from data import pipeline
from data.build import read_table
from data.contamination import load_models
from data.pipeline import PipelineError, load_pipeline_config, main, run_pipeline, validate_config
from data.questions import load_questions
from eval.manifest import load_manifest
from simulator.config import ROOT
from tests.test_download import Server, ok


def sha(b):
    return hashlib.sha256(b).hexdigest()


def gss_csv(n=600, seed=0) -> bytes:
    r = np.random.default_rng(seed)
    df = pd.DataFrame({
        "id": np.arange(1, n + 1), "year": r.choice([1990, 2000, 2010], n),
        "wtssall": r.uniform(0.5, 2, n), "grass": r.choice([1, 2, 0, 8], n),
        "cappun": r.choice([1, 2, 9], n), "age": r.integers(18, 90, n),
        "sex": r.integers(1, 3, n), "race": r.integers(1, 4, n), "degree": r.integers(0, 5, n),
        "realinc": r.uniform(1000, 90000, n).round(2), "region": r.integers(1, 10, n),
        "partyid": r.integers(0, 8, n), "srcbelt": r.integers(1, 7, n),
    })
    return df.to_csv(index=False).encode()


def zip_bytes(member, data) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(member, data)
    return buf.getvalue()


class Env:
    """A fixture dataset served from a local server, plus isolated output dirs."""

    def __init__(self, tmp_path, serve, archive=True, csv=None):
        self.csv = csv if csv is not None else gss_csv()
        self.body = zip_bytes("gss.csv", self.csv) if archive else self.csv
        self.server = serve({"/f": ok(self.body)})
        self.dirs = dict(raw_dir=tmp_path / "raw", processed_dir=tmp_path / "processed",
                         splits_dir=tmp_path / "splits", run_base=tmp_path / "runs")
        self.cfg = {
            "name": "fixture",
            "source": {"url": self.server.url("/f"), "sha256": sha(self.body),
                       "rel_path": "gss.zip" if archive else "gss.csv", "max_bytes": 10_000_000,
                       "archive": archive, **({"member": "gss.csv", "max_total": 10_000_000} if archive else {})},
            "data": {"sha256": sha(self.csv)},
            "inputs": {"questions": "config/questions.yaml", "missing_codes": "config/missing_codes.yaml",
                       "models": "config/model_cutoffs.yaml"},
            "split": {"seed": 5, "fractions": {"train": 0.6, "calibration": 0.2, "test": 0.2},
                      "strata_cols": ["year", "party"]},
        }

    def run(self, cfg=None):
        return run_pipeline(cfg or self.cfg, policy=self.server.policy, **self.dirs)

    def files(self):
        return {p.relative_to(self.dirs["processed_dir"].parent).as_posix(): sha(p.read_bytes())
                for d in ("processed", "splits") if (self.dirs["processed_dir"].parent / d).exists()
                for p in (self.dirs["processed_dir"].parent / d).rglob("*") if p.is_file()}


@pytest.fixture
def env(tmp_path, serve_):
    return Env(tmp_path, serve_)


@pytest.fixture
def serve_():
    servers = []

    def make(routes):
        s = Server(routes)
        servers.append(s)
        return s
    yield make
    for s in servers:
        s.close()


# ---- end to end ------------------------------------------------------------

def test_end_to_end_produces_table_flags_manifest_and_summary(env):
    r = env.run()
    qs = load_questions()
    table = read_table(qs, "respondents.parquet", base_dir=env.dirs["processed_dir"])
    assert len(table) > 0 and set(table["question_id"]) == {q.id for q in qs}
    m = load_manifest(base_dir=env.dirs["splits_dir"], expected_digest=r.manifest_digest)
    assert {i for n in m.ids for i in m.ids[n]} == set(table["respondent_id"])
    flags = pd.read_parquet(r.flags_path)
    assert len(flags) == len(qs) * len(load_models()[0])
    s = json.loads(r.summary_path.read_text())
    assert s["manifest_digest"] == r.manifest_digest and s["n_rows"] == len(table)
    assert sum(s["split_sizes"].values()) == s["n_respondents"]
    assert r.manifest_created


def test_direct_csv_source_without_archive(tmp_path, serve_):
    e = Env(tmp_path, serve_, archive=False)
    assert e.run().table_path.exists()


def test_each_respondent_in_exactly_one_split_in_the_frozen_manifest(env):
    env.run()
    m = load_manifest(base_dir=env.dirs["splits_dir"])
    all_ids = [i for n in m.ids for i in m.ids[n]]
    assert len(all_ids) == len(set(all_ids))


# ---- idempotence ------------------------------------------------------------

def test_rerun_changes_nothing_and_skips_download(env):
    r1 = env.run()
    files_before, hits = env.files(), len(env.server.hits)
    mtime = r1.manifest_path.stat().st_mtime_ns
    summary_before = r1.summary_path.read_bytes()
    r2 = env.run()
    assert env.files() == files_before
    assert len(env.server.hits) == hits                 # verified file reused, no new request
    assert r2.manifest_path.stat().st_mtime_ns == mtime and not r2.manifest_created
    assert r2.manifest_digest == r1.manifest_digest and r2.summary_path.read_bytes() == summary_before


def test_irrelevant_config_change_does_not_break_frozen_splits(env):
    env.run()
    cfg = copy.deepcopy(env.cfg)
    cfg["source"]["max_bytes"] = 20_000_000
    assert not env.run(cfg).manifest_created


# ---- fail closed -------------------------------------------------------------

def nothing_written(env):
    return env.files() == {}


@pytest.mark.parametrize("which", ["source", "data"])
def test_unpinned_checksums_refuse_to_run(env, which):
    cfg = copy.deepcopy(env.cfg)
    cfg[which]["sha256"] = None
    hits = len(env.server.hits)
    with pytest.raises(PipelineError, match="not pinned"):
        env.run(cfg)
    assert nothing_written(env) and len(env.server.hits) == hits


def test_wrong_source_checksum(env):
    cfg = copy.deepcopy(env.cfg)
    cfg["source"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sha256 mismatch"):
        env.run(cfg)
    assert nothing_written(env) and not (env.dirs["raw_dir"] / "gss.zip").exists()


def test_wrong_data_checksum(env):
    cfg = copy.deepcopy(env.cfg)
    cfg["data"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        env.run(cfg)
    assert nothing_written(env)


def test_unknown_special_code_in_data_writes_nothing(tmp_path, serve_):
    df = pd.read_csv(io.BytesIO(gss_csv()))
    df.loc[0, "grass"] = 5
    e = Env(tmp_path, serve_, csv=df.to_csv(index=False).encode())
    with pytest.raises(ValueError, match="grass"):
        e.run()
    assert nothing_written(e)


def test_failure_while_writing_rolls_back_created_files(env, monkeypatch):
    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(pipeline, "write_manifest", boom)
    with pytest.raises(OSError):
        env.run()
    assert nothing_written(env)
    monkeypatch.undo()
    assert env.run().manifest_created           # and a clean retry works


def test_rollback_keeps_files_that_already_existed(env, monkeypatch):
    env.run()
    before = env.files()
    monkeypatch.setattr(pipeline, "atomic_write_parquet", lambda *a, **k: (_ for _ in ()).throw(OSError("x")))
    with pytest.raises(OSError):
        env.run()
    assert env.files() == before


def test_frozen_splits_are_never_overwritten(env):
    env.run()
    before = env.files()
    cfg = copy.deepcopy(env.cfg)
    cfg["split"]["seed"] = 6
    with pytest.raises(PipelineError, match="frozen"):
        env.run(cfg)
    assert env.files() == before


def test_changed_data_with_existing_manifest_is_refused(tmp_path, serve_):
    e = Env(tmp_path, serve_)
    e.run()
    before = e.files()
    e2 = Env(tmp_path, serve_, csv=gss_csv(seed=1))      # same dirs, different data
    e2.dirs = e.dirs
    with pytest.raises(ValueError):
        e2.run()
    assert e.files() == before


def test_stale_extraction_directory_is_not_trusted(env):
    env.run()
    member = env.dirs["raw_dir"] / "gss_extracted" / "gss.csv"
    member.write_bytes(member.read_bytes() + b"1,1,1\n")
    with pytest.raises(PipelineError, match="does not hold the pinned"):
        env.run()


# ---- config validation --------------------------------------------------------

def mutated(env, fn):
    cfg = copy.deepcopy(env.cfg)
    fn(cfg)
    return cfg


@pytest.mark.parametrize("fn", [
    lambda c: c.pop("split"),
    lambda c: c.__setitem__("extra", 1),
    lambda c: c.__setitem__("name", "Bad Name"),
    lambda c: c["source"].__setitem__("surprise", 1),
    lambda c: c["source"].pop("url"),
    lambda c: c["source"].__setitem__("sha256", "ABC"),
    lambda c: c["source"].__setitem__("max_bytes", 0),
    lambda c: c["source"].__setitem__("max_bytes", True),
    lambda c: c["source"].__setitem__("archive", "yes"),
    lambda c: c["source"].pop("member"),
    lambda c: c["source"].pop("max_total"),
    lambda c: c["inputs"].pop("models"),
    lambda c: c["split"].__setitem__("seed", True),
    lambda c: c["split"].__setitem__("strata_cols", []),
    lambda c: c["split"].__setitem__("fractions", {"train": 1, "calibration": 0, "test": 0}),
    lambda c: c.__setitem__("data", None),
    lambda c: c["data"].__setitem__("sha256", "short"),
])
def test_invalid_configs_rejected(env, fn):
    with pytest.raises(PipelineError):
        validate_config(mutated(env, fn))


def test_member_only_valid_for_archives(tmp_path, serve_):
    e = Env(tmp_path, serve_, archive=False)
    e.cfg["source"]["member"] = "x.csv"
    with pytest.raises(PipelineError, match="only applies"):
        validate_config(e.cfg)


def test_input_paths_cannot_escape_the_project(env):
    cfg = mutated(env, lambda c: c["inputs"].__setitem__("questions", "../../etc/passwd"))
    with pytest.raises(ValueError):
        env.run(cfg)
    assert nothing_written(env)


def test_config_file_loading(tmp_path):
    for text in ("", "[]", "name: [", "a: 1\na: 2", "name: !!python/object/apply:os.system ['true']"):
        p = tmp_path / "c.yaml"
        p.write_text(text)
        with pytest.raises(PipelineError):
            load_pipeline_config(p)
    with pytest.raises(PipelineError, match="cannot read"):
        load_pipeline_config(tmp_path / "missing.yaml")
    big = tmp_path / "big.yaml"
    big.write_text("#" * 100_001)
    with pytest.raises(PipelineError, match="too large"):
        load_pipeline_config(big)


# ---- shipped config / CLI / repo hygiene -------------------------------------------

def test_shipped_config_is_valid_but_unpinned():
    cfg = load_pipeline_config(ROOT / "config" / "data.yaml")
    assert cfg["source"]["sha256"] is None and cfg["data"]["sha256"] is None


def test_cli_reports_unpinned_config_and_exits_nonzero(capsys):
    assert main([str(ROOT / "config" / "data.yaml")]) == 1
    assert "not pinned" in capsys.readouterr().err
    assert main(["/nonexistent.yaml"]) == 1


def test_makefile_data_target_runs_the_pipeline():
    text = (ROOT / "Makefile").read_text()
    assert "data:\n\t$(PY) -m data.pipeline config/data.yaml" in text


def git(*args):
    if shutil.which("git") is None:
        pytest.skip("git not available")
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def test_raw_and_processed_data_are_gitignored_but_splits_are_not():
    for path in ("data/raw/gss.dta", "data/raw/x/y.csv", "data/processed/respondents.parquet"):
        assert git("check-ignore", "-q", path).returncode == 0, path
    for path in ("data/splits/manifest.json", "data/splits/test_access.log"):
        assert git("check-ignore", "-q", path).returncode == 1, path


def test_no_data_files_are_tracked_by_git():
    out = git("ls-files", "data", "eval", "config")
    if out.returncode != 0:
        pytest.skip("not a git checkout")
    bad = [f for f in out.stdout.split() if f.endswith((".dta", ".csv", ".parquet", ".zip", ".sav", ".pkl"))]
    assert bad == []


# ---- the test suite itself must be offline -----------------------------------------

def test_external_network_is_blocked_in_tests():
    with pytest.raises(RuntimeError, match="blocked"):
        socket.create_connection(("93.184.216.34", 80), timeout=1)
    with pytest.raises(RuntimeError, match="blocked"):
        socket.getaddrinfo("example.com", 80)

from simulator.config import config_hash, load_config, run_dir, ROOT


def test_hash_stable_and_order_independent():
    a = {"x": 1, "y": {"b": 2, "a": 3}}
    b = {"y": {"a": 3, "b": 2}, "x": 1}
    assert config_hash(a) == config_hash(b)
    assert len(config_hash(a)) == 12


def test_hash_changes_with_value():
    assert config_hash({"x": 1}) != config_hash({"x": 2})


def test_load_example_and_run_dir(tmp_path):
    cfg = load_config(ROOT / "config" / "example.yaml")
    assert config_hash(cfg) == config_hash(load_config(ROOT / "config" / "example.yaml"))
    d = run_dir(cfg, base=tmp_path)
    assert d.name == f"example-{config_hash(cfg)}" and d.is_dir()

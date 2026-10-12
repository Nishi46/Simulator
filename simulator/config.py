"""Experiment config loading and hashing.

Every run is identified by `config_hash(cfg)`, a stable digest of the resolved
config. Outputs are written under `run_dir(cfg)` so results trace to a config.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
HASH_LEN = 12


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: config must be a YAML mapping")
    return cfg


def config_hash(cfg: dict[str, Any]) -> str:
    """Stable hash: independent of key order and YAML formatting."""
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:HASH_LEN]


def run_dir(cfg: dict[str, Any], base: str | Path = ROOT / "runs") -> Path:
    name = cfg.get("name", "run")
    path = Path(base) / f"{name}-{config_hash(cfg)}"
    path.mkdir(parents=True, exist_ok=True)
    return path

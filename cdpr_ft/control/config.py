"""Controller configuration: one YAML file holds every setting of the estimators, the MPC, the
planner and the decision logic, so that a frozen configuration is a single versioned file whose
hash enters the provenance of every result produced with it."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, fields
from pathlib import Path

import yaml

from ..estimation.imm import IMMConfig
from .ftc import FTCConfig
from .mpc import MPCConfig
from .planner import PlannerConfig
from .recovery import RecoveryConfig

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
SECTIONS = {"imm": IMMConfig, "mpc": MPCConfig, "planner": PlannerConfig, "recovery": RecoveryConfig,
            "ftc": FTCConfig}


def _build(cls, values: dict):
    names = {f.name for f in fields(cls)}
    unknown = set(values) - names
    if unknown:
        raise KeyError(f"{cls.__name__}: unknown settings {sorted(unknown)}")
    kw = {}
    for f in fields(cls):
        if f.name in values:
            v = values[f.name]
            if isinstance(v, list):
                v = tuple(tuple(x) if isinstance(x, list) else x for x in v)
            kw[f.name] = v
    return cls(**kw)


def load_controller(name: str = "controller_phase3.yaml"):
    """Return (configs: dict of dataclasses by section, sha256 of the file)."""
    path = CONFIGS / name
    text = path.read_text()
    data = yaml.safe_load(text)
    out = {key: _build(cls, data.get(key, {}) or {}) for key, cls in SECTIONS.items()}
    return out, hashlib.sha256(text.encode()).hexdigest()[:16]


def dump_controller(configs: dict, path: Path, header: str = ""):
    """Write the settings of configs (dict of dataclasses by section) as YAML."""
    body = {key: {k: (list(map(list, v)) if isinstance(v, tuple) and v and isinstance(v[0], tuple) else
                      list(v) if isinstance(v, tuple) else v)
                  for k, v in asdict(cfg).items()} for key, cfg in configs.items()}
    text = header + yaml.safe_dump(body, sort_keys=False, default_flow_style=None)
    Path(path).write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()[:16]

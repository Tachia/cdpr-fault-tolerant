"""Provenance stamp for result files.

The specification asks for every result to carry the version of the code that produced
it. The package is not kept in a git repository yet, so the stamp holds a SHA-256 digest
of the package sources and the configuration files instead, plus the library versions.
Two results with the same digest come from identical code and robot parameters; the hash of
the controller configuration is stamped next to it (from Phase 3 on).
"""
from __future__ import annotations

import hashlib
import platform
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def code_digest() -> str:
    """Digest of the package sources and the robot configuration. Controller configurations
    (configs/controller_*.yaml) are left out: they are identified by their own hash, which a
    frozen configuration can then record together with the code digest it was frozen with."""
    h = hashlib.sha256()
    files = sorted(list((ROOT / "cdpr_ft").rglob("*.py"))
                   + [p for p in (ROOT / "configs").glob("*.yaml") if not p.name.startswith("controller_")])
    for f in files:
        h.update(str(f.relative_to(ROOT)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def stamp() -> dict:
    import casadi
    import numpy
    import scipy
    ctrl = ROOT / "configs" / "controller_phase3.yaml"
    extra = {"controller_config_sha256_16": hashlib.sha256(ctrl.read_bytes()).hexdigest()[:16]} if ctrl.exists() else {}
    return {"code_digest": code_digest(), **extra,
            "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "python": platform.python_version(), "numpy": numpy.__version__,
            "scipy": scipy.__version__, "casadi": casadi.__version__}

"""The Phase 4 protocol (configs/learning_phase4.yaml): run list, splits and provenance."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from ..sim.episode import EpisodeSpec, mode_index

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs" / "learning_phase4.yaml"
N_CABLES = 8
N_MODES = 2 * N_CABLES + 1


def load_protocol(path: Path = PROTOCOL):
    text = Path(path).read_bytes()
    return yaml.safe_load(text), hashlib.sha256(text).hexdigest()[:16]


def controller_digest() -> str:
    """Digest of the files that define the model, the simulator and the frozen controller: the
    code digest of cdpr_ft/provenance.py restricted to the files that existed when the controller
    was frozen (the package without cdpr_ft/learning, and the robot configuration). Equal to the
    code digest of the Phase 3 freeze, fe7722d2d710c537, while those files are unchanged."""
    h = hashlib.sha256()
    files = sorted([p for p in (ROOT / "cdpr_ft").rglob("*.py") if "learning" not in p.relative_to(ROOT).parts]
                   + [p for p in (ROOT / "configs").glob("*.yaml")
                      if not p.name.startswith(("controller_", "learning_", "classifier_"))])
    for f in files:
        h.update(str(f.relative_to(ROOT)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def learning_digest() -> str:
    """Digest of the learning code (cdpr_ft/learning/*.py)."""
    h = hashlib.sha256()
    for f in sorted((ROOT / "cdpr_ft" / "learning").glob("*.py")):
        h.update(str(f.relative_to(ROOT)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


@dataclass(frozen=True)
class Run:
    index: int
    split: str           # "train", "val", "test" (T1) or "test_T2"
    spec: EpisodeSpec

    @property
    def label(self) -> int:
        """Mode index of the fault (0 for a fault-free run)."""
        f = self.spec.fault
        return 0 if f is None else mode_index(N_CABLES, "break" if f[0] == "break" else "actuator", f[1])


def run_list(protocol: dict) -> list[Run]:
    d = protocol["data"]
    base = int(d["seed_base"])
    faults = [tuple(f) for f in d["faults"]]
    noise = [float(x) for x in d["noise"]]
    runs: list[Run] = []

    def onset_of(seed):
        lo, hi = d["onset"]
        return float(np.round(np.random.default_rng(seed + 500000).uniform(lo, hi), 3))

    for traj, seed0, ff0 in (("T1", base, base + 10000), ("T2", base + 20000, base + 30000)):
        cfg = d[traj]
        reps = int(cfg["repetitions"])
        if traj == "T1":
            by_rep = {r: s for s, rs in cfg["split"].items() for r in rs}
            assert sorted(by_rep) == list(range(reps)), "every repetition needs a split"
        j = 0
        for cable in range(N_CABLES):
            for kind, value in faults:
                for rep in range(reps):
                    seed = seed0 + j
                    onset = onset_of(seed)
                    spec = EpisodeSpec(trajectory=traj, seed=seed, noise_factor=noise[rep % len(noise)],
                                       fault=(kind, cable, onset, float(value)),
                                       t_end=onset + float(d["after_onset"]), variant=d["variant"])
                    runs.append(Run(len(runs), by_rep[rep] if traj == "T1" else "test_T2", spec))
                    j += 1
        n_ff = int(cfg["fault_free"])
        if traj == "T1":
            sizes = cfg["fault_free_split"]
            names = ["train"] * sizes["train"] + ["val"] * sizes["val"] + ["test"] * sizes["test"]
            assert len(names) == n_ff
        else:
            names = ["test_T2"] * n_ff
        for r in range(n_ff):
            spec = EpisodeSpec(trajectory=traj, seed=ff0 + r, noise_factor=noise[r % len(noise)], variant=d["variant"])
            runs.append(Run(len(runs), names[r], spec))
    return runs

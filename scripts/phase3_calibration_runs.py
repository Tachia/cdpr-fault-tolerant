"""Phase 3: closed-loop calibration runs with the readings and commands recorded.

The calibration set (specification, Sec. 12) is drawn from its own seeds, 100000 and up; the
test set of Phase 5 uses seeds from 200000 and the learned classifier of Phase 4 seeds from
300000, so the sets never share a parameter draw, a noise sequence or an onset time. Every run
uses the randomised plant.

  fault-free: T1 and T2, noise x1 and x2, N_FREE seeds each, full duration (16 s)
  faults:     every cable x {break, loe 0.3, loe 0.6, bias +10 %, bias -10 %}, one run each;
              trajectory and noise alternate with the index; onset uniform in [3, 7] s; the run
              ends T_AFTER seconds after the onset

Declarations are disabled (mu_iso above 1), so the mode probabilities of a run are those of an
undisturbed diagnosis, and the readings can be replayed offline through other estimator
settings (cdpr_ft/estimation/diagnosis.py). Output: results/phase3_calib/<tag>/ with one .npz
of records per run and runs.json.gz with the specifications, metrics and mode probabilities.
The records (about 100 MB per set) are not kept in the repository; runs.json.gz is.
Usage: python scripts/phase3_calibration_runs.py <tag> [processes] [imm overrides as json] [mpc overrides as json]
The three sets of the study: c1 with {"sigma_acc_ang": 0.5} and the first MPC weights
{"q_vel": 400.0, "q_angvel": 400.0, "r_rate": 0.01}; c2 with {"sigma_acc_ang": 0.5}; c3 with the
defaults (the frozen values). Settings of the FTC that act only after a declaration do not matter
here, because declarations are disabled.
"""
from __future__ import annotations

import gzip
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdpr_ft.control.ftc import FTCConfig  # noqa: E402
from cdpr_ft.control.mpc import MPCConfig  # noqa: E402
from cdpr_ft.control.planner import PlannerConfig  # noqa: E402
from cdpr_ft.estimation.imm import IMMConfig  # noqa: E402
from cdpr_ft.provenance import stamp  # noqa: E402
from cdpr_ft.sim.batch import run_batch  # noqa: E402
from cdpr_ft.sim.episode import EpisodeSpec  # noqa: E402

SEED0 = 100000
N_FREE = 4
T_AFTER = 2.0
FAULTS = [("break", 0.0), ("loe", 0.3), ("loe", 0.6), ("bias", 0.1), ("bias", -0.1)]
IMM_BASE = {}            # the defaults of IMMConfig


def calibration_specs():
    specs = []
    k = 0
    for traj in ("T1", "T2"):
        for noise in (1.0, 2.0):
            for _ in range(N_FREE):
                specs.append(EpisodeSpec(trajectory=traj, seed=SEED0 + k, noise_factor=noise))
                k += 1
    j = 0
    for cable in range(8):
        for kind, value in FAULTS:
            seed = SEED0 + 1000 + j
            onset = float(np.round(np.random.default_rng(seed + 500000).uniform(3.0, 7.0), 3))
            specs.append(EpisodeSpec(trajectory=("T1", "T2")[j % 2], seed=seed, noise_factor=(1.0, 2.0)[(j // 2) % 2],
                                     fault=(kind, cable, onset, value), t_end=onset + T_AFTER))
            j += 1
    return specs


def load_runs(folder: Path) -> dict:
    """Summary of a calibration set: runs.json.gz (as written and kept), or runs.json (older sets)."""
    folder = Path(folder)
    if (folder / "runs.json.gz").exists():
        with gzip.open(folder / "runs.json.gz", "rt") as f:
            return json.load(f)
    return json.loads((folder / "runs.json").read_text())


def main(tag, processes=2, imm_over=None, mpc_over=None):
    out = ROOT / "results" / "phase3_calib" / tag
    out.mkdir(parents=True, exist_ok=True)
    specs = calibration_specs()
    imm_cfg = IMMConfig(**{**IMM_BASE, **(imm_over or {})})
    ftc_cfg = FTCConfig(mu_declare=2.0, record_inputs=True)
    paths = [str(out / f"run{i:03d}.npz") for i in range(len(specs))]
    t0 = time.time()
    mpc_cfg = MPCConfig(**(mpc_over or {}))
    res = run_batch(specs, imm_cfg, mpc_cfg, PlannerConfig(), ftc_cfg, processes=processes, record_paths=paths)
    for r, p in zip(res, paths):
        r["record"] = Path(p).name
        r["t_imm"] = r["t_imm"].tolist()
        r["mu"] = r["mu"].tolist()
    meta = {"imm": asdict(imm_cfg), "ftc": asdict(ftc_cfg), "mpc": asdict(mpc_cfg),
            "seconds": time.time() - t0, "provenance": stamp()}
    with gzip.open(out / "runs.json.gz", "wt") as f:
        f.write(json.dumps({"meta": meta, "runs": res}, default=str))
    print("%d runs in %.0f s" % (len(res), meta["seconds"]))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2,
         json.loads(sys.argv[3]) if len(sys.argv) > 3 else None, json.loads(sys.argv[4]) if len(sys.argv) > 4 else None)

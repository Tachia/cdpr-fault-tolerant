"""Phase 3: MPC weights on the fault-free calibration runs.

With the first weights (velocity weights 400, tension-change weight 0.01), the closed loop
oscillated and lost the platform before any fault in 7 of the 56 calibration runs, all of them
with cables 15 to 20 % softer than nominal. The weights
enter the compiled programme as parameters, so a few scalings of them are compared here on the
16 fault-free calibration specifications (first 8 s of each run, the part that contains the
start transient and the first fault onsets):
  A: velocity weights x0.1, tension-change weight x10
  B: velocity weights x0.3, tension-change weight x3
  C: as A with the pose weights x0.5
Selection rule: no lost platform; among those, the smallest median tracking error.
The scalings apply to the weights in force when the study was run (BASE_MPC: the first weights)
with the estimator setting of that time (BASE_IMM). The defaults of the configuration classes
now hold the outcome (setting B, and sigma_acc_ang = 1 from the later replay), so they are not
the base of this study.
Output: results/phase3_mpc_weights.json
Usage: python scripts/phase3_mpc_weights.py [processes]
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cdpr_ft.control.ftc import FTCConfig  # noqa: E402
from cdpr_ft.control.mpc import MPCConfig  # noqa: E402
from cdpr_ft.control.planner import PlannerConfig  # noqa: E402
from cdpr_ft.estimation.imm import IMMConfig  # noqa: E402
from cdpr_ft.provenance import stamp  # noqa: E402
from cdpr_ft.sim.batch import run_batch  # noqa: E402
from phase3_calibration_runs import calibration_specs  # noqa: E402

BASE_MPC = {"q_vel": 400.0, "q_angvel": 400.0, "r_rate": 0.01}
BASE_IMM = {"sigma_acc_ang": 0.5}
SETTINGS = {
    "first": (),
    "A": (("q_vel", 0.1), ("q_angvel", 0.1), ("r_rate", 10.0)),
    "B": (("q_vel", 0.3), ("q_angvel", 0.3), ("r_rate", 3.0)),
    "C": (("q_pos", 0.5), ("q_rot", 0.5), ("q_vel", 0.1), ("q_angvel", 0.1), ("r_rate", 10.0)),
}


def main(processes=2, names=None):
    specs = [replace(s, t_end=8.0) for s in calibration_specs() if not s.fault]
    out = {}
    for name in (names or SETTINGS):
        t0 = time.time()
        ftc = FTCConfig(mu_declare=2.0, mpc_weight_scale=SETTINGS[name])
        res = run_batch(specs, IMMConfig(**BASE_IMM), MPCConfig(**BASE_MPC), PlannerConfig(), ftc, processes=processes)
        rms = [1e3 * r["rms_error_pre_m"] for r in res]
        out[name] = {"scale": [list(x) for x in SETTINGS[name]], "lost": int(sum(r["lost"] for r in res)),
                     "rms_mm_median": float(np.median(rms)), "rms_mm_max": float(np.max(rms)),
                     "max_fault_mu": float(max(r["mu"][:, 1:].max() for r in res)),
                     "runs": [{"seed": r["seed"], "trajectory": r["trajectory"], "noise": r["spec"]["noise_factor"],
                               "lost": r["stopped"], "rms_mm": 1e3 * r["rms_error_pre_m"]} for r in res],
                     "seconds": time.time() - t0}
        print(name, {k: v for k, v in out[name].items() if k != "runs"}, flush=True)
    ok = [n for n in out if out[n]["lost"] == 0]
    chosen = min(ok, key=lambda n: out[n]["rms_mm_median"]) if ok else None
    (ROOT / "results" / "phase3_mpc_weights.json").write_text(
        json.dumps({"settings": out, "chosen": chosen, "provenance": stamp()}, indent=1, default=str))
    print("chosen:", chosen)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2, sys.argv[2].split(",") if len(sys.argv) > 2 else None)

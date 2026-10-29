"""Phase 5: gains of the computed-torque law of baseline B2, chosen on calibration runs.

B2 replaces the MPC by a computed-torque law with tension distribution (methods, Sec. 7, last
paragraph), whose gains were set in Phase 3 without a study (kp = 100, kd = 20 per axis). So
that B2 is not handicapped by them, settings with critical damping are compared on fault-free
calibration runs (seeds 140000 and up, in the calibration range, never used for a result):
natural frequency w, kp = w^2, kd = 2 w. Declarations are disabled.

Rule (second version, 27 October 2026): among the settings that lose no run and keep the cables
within their tension bounds (share of the time with a cable outside [t_min, t_max], from 2 s, at
most 1 % in every run), the one with the smallest median RMS position error from 2 s to the end;
if no setting qualifies, the one with the smallest largest share outside.

The first version compared 10, 20 and 30 rad/s and took the smallest median error among the
settings that lose no run, without weighing the tension bounds. It chose 20 rad/s, whose runs
left the bounds for up to 31.5 % of the time (T2), against none at 10 rad/s with a median error
larger by 0.5 mm; it was stored as results/phase5_b2_v1/ and its loop is near the edge of
stability (30 rad/s lost runs). The first eight B2 runs of the campaign (breaks on T1, repetition
0) showed the consequence: tensions outside their bounds for 31 to 45 % of the pre-fault window
and the platform lost in all eight. The rule was therefore replaced by the one above, the grid
refined below and between the first settings, and the campaign started again from its first run
(protocol amendment in configs/evaluation_phase5.yaml). Settings 10, 20 and 30 rad/s are run
again and reproduce the first version run by run.
Output: results/phase5_b2_gains.json
Usage: python scripts/phase5_b2_gains.py [processes]
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdpr_ft.provenance import stamp  # noqa: E402

OMEGA = [5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 30.0]
MAX_OUTSIDE_PCT = 1.0
RUNS = [("T1", 140000 + r, (1.0, 2.0)[r % 2]) for r in range(4)] + [("T2", 141000 + r, (1.0, 2.0)[r % 2]) for r in range(4)]
_S = {}


def init():
    from cdpr_ft.control.config import load_controller
    from cdpr_ft.sim import batch
    cfgs, _ = load_controller("controller_phase3.yaml")
    batch._init(cfgs["imm"], cfgs["mpc"], cfgs["planner"], cfgs["ftc"], True, cfgs["recovery"])
    _S.update(batch._STATE)


def job(args):
    from cdpr_ft.sim.episode import EpisodeSpec, run_episode
    w, (traj, seed, noise) = args
    spec = EpisodeSpec(trajectory=traj, seed=seed, noise_factor=noise, variant="tda")
    cfg = replace(_S["ftc"], variant="tda", mu_declare=2.0)
    gains = ((w * w,) * 6, (2 * w,) * 6)
    res, ctrl = run_episode(spec, _S["imm"], _S["mpc"], _S["planner"], _S["safe"], cfg, keep_log=True,
                            return_controller=True, ctrl_kwargs={"ct_gains": gains})
    log = res["log"]
    t = log["t"]
    q_ref = ctrl.reference(t)[0]
    e = np.linalg.norm(log["q"][:, :3] - q_ref[:, :3], axis=1)
    m = t >= 2.0
    r = ctrl.robot
    margin = np.min(np.minimum(log["tension"] - r.t_min, r.t_max - log["tension"]), axis=1)
    return {"omega": w, "trajectory": traj, "seed": seed, "noise_factor": noise, "lost": res["lost"],
            "rmse_mm": float(1e3 * np.sqrt(np.mean(e[m] ** 2))), "max_mm": float(1e3 * e[m].max()),
            "outside_pct": res["time_outside_bounds_pct"],
            "outside_from2_pct": float(100 * np.mean(margin[m] < -1e-9)), "min_margin_N": float(margin[m].min())}


def main(processes=2):
    jobs = [(w, r) for w in OMEGA for r in RUNS]
    with mp.get_context("fork").Pool(processes, initializer=init) as pool:
        rows = pool.map(job, jobs, chunksize=1)
    table = []
    for w in OMEGA:
        g = [r for r in rows if r["omega"] == w]
        table.append({"omega": w, "kp": w * w, "kd": 2 * w, "lost": sum(r["lost"] for r in g),
                      "median_rmse_mm": float(np.median([r["rmse_mm"] for r in g])),
                      "max_rmse_mm": float(np.max([r["rmse_mm"] for r in g])),
                      "median_outside_pct": float(np.median([r["outside_pct"] for r in g])),
                      "max_outside_from2_pct": float(np.max([r["outside_from2_pct"] for r in g])),
                      "min_margin_N": float(np.min([r["min_margin_N"] for r in g]))})
        print(table[-1], flush=True)
    ok = [t for t in table if t["lost"] == 0 and t["max_outside_from2_pct"] <= MAX_OUTSIDE_PCT]
    if ok:
        chosen = min(ok, key=lambda t: t["median_rmse_mm"])
    else:
        chosen = min(table, key=lambda t: t["max_outside_from2_pct"])
    print("chosen", chosen)
    (ROOT / "results" / "phase5_b2_gains.json").write_text(json.dumps(
        {"rule": "lose no run, at most %.1f %% of the time outside the tension bounds from 2 s in every run, "
                 "then smallest median RMS error" % MAX_OUTSIDE_PCT,
         "runs": rows, "table": table, "chosen": chosen, "provenance": stamp()}, indent=1))


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:2]))

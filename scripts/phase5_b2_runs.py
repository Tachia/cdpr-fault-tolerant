"""Phase 5: calibration runs of baseline B2, on which its detection thresholds are chosen.

B2 runs the computed-torque law with tension distribution, not the MPC, so its winch quantities
are not distributed as in the learning runs of Phases 4 and 4b (which ran the MPC). Its
thresholds are therefore chosen on runs of its own controller: the "tda" variant with the gains of
scripts/phase5_b2_gains.py, declarations disabled, the five faults of the learning runs on every
cable and both paths, two repetitions (noise nominal and doubled), and eight fault-free runs per
path; seeds 150000 and up (calibration range, never used for a result). A fault run ends 1 s after
the onset. The inputs of the threshold rule are computed from the recorded readings and commands
exactly as online (cdpr_ft.learning.features).
Output: results/phase5_b2_data.npz (per estimator cycle: run, time, features, true mode) and
results/phase5_b2_runs.json
Usage: python scripts/phase5_b2_runs.py [processes]
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

FAULTS = [("break", 0.0), ("loe", 0.3), ("loe", 0.6), ("bias", 0.1), ("bias", -0.1)]
_S = {}


def run_list():
    rng_onset = lambda seed: float(np.round(np.random.default_rng(seed + 500000).uniform(3.0, 7.0), 3))  # noqa: E731
    runs = []
    for ti, traj in enumerate(("T1", "T2")):
        j = 0
        for cable in range(8):
            for kind, value in FAULTS:
                for rep in range(2):
                    seed = 150000 + 1000 * ti + j
                    onset = rng_onset(seed)
                    runs.append({"trajectory": traj, "seed": seed, "noise_factor": (1.0, 2.0)[rep],
                                 "fault": [kind, cable, onset, value], "t_end": onset + 1.0})
                    j += 1
        for r in range(8):
            runs.append({"trajectory": traj, "seed": 152000 + 1000 * ti + r, "noise_factor": (1.0, 2.0)[r % 2],
                         "fault": None, "t_end": None})
    for i, r in enumerate(runs):
        r["index"] = i
    return runs


def init(gains):
    from cdpr_ft.control.config import load_controller
    from cdpr_ft.sim import batch
    cfgs, _ = load_controller("controller_phase3.yaml")
    batch._init(cfgs["imm"], cfgs["mpc"], cfgs["planner"], cfgs["ftc"], True, cfgs["recovery"])
    _S.update(batch._STATE)
    _S["gains"] = gains


def job(r):
    from cdpr_ft.learning.features import features_from_record
    from cdpr_ft.learning.protocol import N_CABLES
    from cdpr_ft.params import inner_rate, load_cable_winch, load_robot
    from cdpr_ft.sim.episode import EpisodeSpec, controller_start, mode_index, run_episode
    spec = EpisodeSpec(trajectory=r["trajectory"], seed=r["seed"], noise_factor=r["noise_factor"],
                       fault=tuple(r["fault"]) if r["fault"] else None, t_end=r["t_end"], variant="tda")
    cfg = replace(_S["ftc"], variant="tda", mu_declare=2.0, record_inputs=True)
    res, ctrl = run_episode(spec, _S["imm"], _S["mpc"], _S["planner"], _S["safe"], cfg, keep_log=True,
                            return_controller=True, ctrl_kwargs={"ct_gains": _S["gains"]})
    cl = res["ctrl_log"]
    robot, cw = load_robot(), load_cable_winch()
    _, ref, _, t_nom, l_u0 = controller_start(spec, robot, cw)
    rec = {"theta": np.asarray(cl["theta"]), "tau": np.asarray(cl["tau"]), "td_in": np.asarray(cl["td_in"])}
    t, f = features_from_record(rec, robot, cw, l_u0, ref, cfg.eso_bandwidth_hz, cfg.imm_every, 1.0 / inner_rate(), t_nom)
    label = 0 if spec.fault is None else mode_index(N_CABLES, "break" if spec.fault[0] == "break" else "actuator",
                                                     spec.fault[1])
    onset = spec.fault[2] if spec.fault else np.inf
    y = np.where(t >= onset - 1e-9, label, 0)
    return r["index"], t.astype(np.float32), f.reshape(len(t), -1).astype(np.float16), y.astype(np.int8), res["lost"]


def main(processes=2):
    gains = json.loads((ROOT / "results" / "phase5_b2_gains.json").read_text())["chosen"]
    g = ((float(gains["kp"]),) * 6, (float(gains["kd"]),) * 6)
    runs = run_list()
    with mp.get_context("fork").Pool(processes, initializer=init, initargs=(g,)) as pool:
        out = sorted(pool.map(job, runs, chunksize=1), key=lambda x: x[0])
    xs, ys, ts, rs = [], [], [], []
    for (i, t, x, y, lost), r in zip(out, runs):
        r["offset"], r["length"], r["lost"] = int(sum(len(a) for a in ts)), int(len(t)), bool(lost)
        ts.append(t)
        xs.append(x)
        ys.append(y)
        rs.append(np.full(len(t), i, np.int16))
    np.savez_compressed(ROOT / "results" / "phase5_b2_data.npz", x=np.concatenate(xs), y=np.concatenate(ys),
                        t=np.concatenate(ts), run=np.concatenate(rs))
    (ROOT / "results" / "phase5_b2_runs.json").write_text(json.dumps(
        {"gains": gains, "runs": runs, "provenance": stamp()}, indent=1))
    print(f"{len(runs)} runs, {sum(len(a) for a in ts)} cycles, lost: {sum(r['lost'] for r in runs)}")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:2]))

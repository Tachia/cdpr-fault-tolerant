"""Phase 3: the recovery library, and the study of the post-break planner it provides.

For each cable i, each reference path (T1, T2) and break instants from 2.5 to 8.0 s every
0.05 s (the onset range of the specification, 3 to 7 s, with room for the detection delay),
the planner of methods Sec. 8 is called with the reference pose and twist at that instant.
The plans are stored as the library the controller uses (results/phase3_recovery_library.npz):
the reference paths are known before the task runs, so the planning is done before the robot
moves and a declared break is answered at once. (Solved online, the recovery problem takes
0.15 to 0.7 s on this machine, and the closed-loop runs of Phase 3 showed that a latency of
0.3 s already raises the share of post-fault time with a tension out of bounds from 0-5 % to
9-14 % in the cases tried.) Recorded: the branch, the time scaling kappa
(branch 1), and for branches 2 and 3 the solver status, iterations and time, the largest slack,
the end pose, the largest Euler angle along the path, and the drop in height. The paths of
branches 2 and 3 are also checked against the rigid model in two ways:
  local error: on each interval, the model is integrated (RK4, 1 ms) from the node state with
    the tension of that interval, and the largest position difference to the next node is
    recorded (accuracy of the collocation);
  open-loop drift: the whole path is integrated from the initial state with the tensions of the
    solution, and the position difference to the planned path after 0.25 s and at the end is
    recorded. This drift grows with the instability of the platform hanging from seven cables,
    not with the discretisation, and is what the MPC has to correct.
Output: results/phase3_recovery_library.npz and results/phase3_recovery_library.json (study)
Usage: python scripts/phase3_recovery_library.py [processes]
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdpr_ft.analysis.trajectories import reference  # noqa: E402
from cdpr_ft.control.planner import Planner, PlannerConfig  # noqa: E402
from cdpr_ft.params import load_cable_winch, load_robot  # noqa: E402
from cdpr_ft.provenance import stamp  # noqa: E402
from cdpr_ft.sim.episode import reference_fn  # noqa: E402

GRID_STEP = 0.05
ONSETS = np.round(np.arange(2.5, 8.0 + 1e-9, GRID_STEP), 3)
_P = {}


def _init():
    robot, cw = load_robot(), load_cable_winch()
    d = np.load(ROOT / "results" / "phase3_safe_poses.npz")
    safe = {i: (d[f"pos{i}"], d[f"ori{i}"]) for i in range(robot.n)}
    _P.update(robot=robot, planner=Planner(robot, safe, PlannerConfig(), EA=cw.EA),
              refs={k: reference_fn(reference(k, home=tuple(robot.home_pose[:3]))) for k in ("T1", "T2")})


def _case(args):
    traj, cable, onset = args
    robot, planner, ref = _P["robot"], _P["planner"], _P["refs"][traj]
    q, nu, _ = ref(np.array([onset]))
    t0 = time.time()
    plan = planner.after_break(cable, onset, q[0], nu[0], ref)
    out = {"trajectory": traj, "cable": cable, "onset": float(onset), "branch": plan.branch,
           "kappa": float(plan.kappa), "planner_seconds": time.time() - t0}
    n_int = planner.recovery.cfg.steps
    sol = {"X": np.zeros((12, n_int + 1)), "U": np.zeros((robot.n, n_int)), "t_hold": np.zeros(robot.n), "T": 0.0}
    if plan.branch >= 2:
        ts = onset + np.arange(0.0, plan.duration + 1e-9, 0.001)
        qp, nup, _ = plan.sample(ts)
        model = planner.recovery.model
        x = np.r_[q[0], nu[0]]
        drift = np.zeros(len(ts))
        for k in range(1, len(ts)):
            u = plan.tensions(np.array([ts[k - 1] + 1e-9]))[:, 0]
            x = np.asarray(model.step_ext(x, u, np.zeros(6), ts[k] - ts[k - 1])).ravel()
            drift[k] = np.linalg.norm(x[:3] - qp[k, :3])
        h = plan.duration / n_int
        nodes = onset + h * np.arange(n_int + 1)
        qn, nun, _ = plan.sample(nodes)
        local = 0.0
        sub = max(1, int(round(h / 0.001)))
        for k in range(n_int):
            x = np.r_[qn[k], nun[k]]
            u = plan.tensions(np.array([nodes[k] + 1e-9]))[:, 0]
            for _ in range(sub):
                x = np.asarray(model.step_ext(x, u, np.zeros(6), h / sub)).ravel()
            local = max(local, float(np.linalg.norm(x[:3] - qn[k + 1, :3])))
        out.update({k: (v if not isinstance(v, np.ndarray) else v.tolist()) for k, v in plan.info.items()
                    if k != "solution"})
        out.update(end_pose=plan.target.tolist(), duration=plan.duration,
                   max_angle_deg=float(np.rad2deg(np.abs(qp[:, 3:]).max())),
                   drop_m=float(q[0, 2] - qp[:, 2].min()), local_error_m=local,
                   drift_025_m=float(drift[min(250, len(drift) - 1)]), drift_end_m=float(drift[-1]))
        sol = plan.info["solution"]
    return out, sol


def main(processes=2):
    jobs = [(traj, c, float(t)) for traj in ("T1", "T2") for c in range(8) for t in ONSETS]
    t0 = time.time()
    with mp.get_context("fork").Pool(processes, initializer=_init) as pool:
        both = pool.map(_case, jobs, chunksize=4)
    rows = [b[0] for b in both]
    sols = [b[1] for b in both]
    np.savez_compressed(ROOT / "results" / "phase3_recovery_library.npz",
                        task=np.array([r["trajectory"] for r in rows]), cable=np.array([r["cable"] for r in rows]),
                        onset=np.array([r["onset"] for r in rows]), branch=np.array([r["branch"] for r in rows]),
                        kappa=np.array([r["kappa"] for r in rows]), T=np.array([s["T"] for s in sols]),
                        X=np.array([s["X"] for s in sols]), U=np.array([s["U"] for s in sols]),
                        t_hold=np.array([s["t_hold"] for s in sols]), grid_step=GRID_STEP)
    summary = {}
    for c in range(8):
        sel = [r for r in rows if r["cable"] == c]
        rec = [r for r in sel if r["branch"] >= 2]
        summary[str(c + 1)] = {
            "branch_counts": {str(b): sum(r["branch"] == b for r in sel) for b in (1, 2, 3)},
            "kappa_min": min([r["kappa"] for r in sel if r["branch"] == 1], default=None),
            "solver_seconds_median": float(np.median([r["solver_seconds"] for r in rec])) if rec else None,
            "solver_seconds_max": float(np.max([r["solver_seconds"] for r in rec])) if rec else None,
            "iterations_max": int(np.max([r["iterations"] for r in rec])) if rec else None,
            "max_angle_deg": float(np.max([r["max_angle_deg"] for r in rec])) if rec else None,
            "local_error_max_m": float(np.max([r["local_error_m"] for r in rec])) if rec else None,
            "drift_025_max_m": float(np.max([r["drift_025_m"] for r in rec])) if rec else None,
            "drift_end_max_m": float(np.max([r["drift_end_m"] for r in rec])) if rec else None,
            "drop_max_m": float(np.max([r["drop_m"] for r in rec])) if rec else None,
        }
    out = {"cases": rows, "per_cable": summary, "onsets": ONSETS.tolist(), "seconds": time.time() - t0,
           "provenance": stamp()}
    (ROOT / "results" / "phase3_recovery_library.json").write_text(json.dumps(out, indent=1))
    for c, s in summary.items():
        print(c, s)
    print("total %.0f s" % out["seconds"])


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)

"""Phase 1 workspace study: nominal and post-fault wrench-feasible workspace.

A position p is counted feasible for an orientation set Phi if, for every
orientation in Phi, all free cable lengths are within their drum windows
(robot.length_window_lo, robot.cable_stroke) and the static gravity wrench can be
balanced with t_min <= t_i <= t_max. Post-fault sets remove one cable at a time.
The count without the length condition is reported for reference only.
Output: results/phase1_workspace.json
"""
from __future__ import annotations

import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdpr_ft.analysis.feasibility import (GRID_LIMITS, GRID_STEP, static_margin,  # noqa: E402
                                          within_drum_window, workspace_grid)
from cdpr_ft.params import load_robot, with_overrides  # noqa: E402


def orientation_sets(deg: float):
    a = np.deg2rad(deg)
    single = [np.zeros(3)] + [s * a * np.eye(3)[k] for k in range(3) for s in (-1, 1)]
    return {"zero": [np.zeros(3)], f"pm{deg:g}deg_single_axis": single}


def feasible(q, robot, active=None):
    s, _ = static_margin(q, robot, active)
    return s is not None and s >= 0, s


def scan(robot, grid, orientations):
    n = robot.n
    nominal, margins = [], []
    post = np.zeros(n, int)
    n_wrench_only = 0
    for x, y, z in product(*grid):
        qs = [np.r_[x, y, z, phi] for phi in orientations]
        res = [feasible(q, robot) for q in qs]
        if not all(ok for ok, _ in res):
            continue
        n_wrench_only += 1
        if not all(within_drum_window(q, robot) for q in qs):
            continue
        nominal.append((x, y, z))
        margins.append(min(s for _, s in res))
        for k in range(n):
            active = np.ones(n, bool)
            active[k] = False
            if all(feasible(q, robot, active)[0] and within_drum_window(q, robot, active) for q in qs):
                post[k] += 1
    return np.array(nominal), np.array(margins), post, n_wrench_only


def home_loss(robot):
    """Margin after the loss of each cable at the home pose, with t_min and with t_min = 0."""
    out = {}
    for t_min in (robot.t_min, 0.0):
        rob = with_overrides(robot, t_min=t_min)
        margins = []
        for k in range(robot.n):
            active = np.ones(robot.n, bool)
            active[k] = False
            s, _ = static_margin(robot.home_pose, rob, active)
            margins.append(None if s is None else round(float(s), 3))
        out[f"tmin{t_min:g}"] = margins
    return out


def main():
    robot = load_robot()
    grid = workspace_grid()
    out = {"grid": {"limits": GRID_LIMITS, "step": GRID_STEP, "n_points": int(np.prod([len(g) for g in grid]))},
           "stroke": robot.stroke, "window_lo": robot.window_lo.tolist(), "cases": {}, "home_single_loss_margin": home_loss(robot)}
    for t_min in (10.0, 30.0, 40.0):
        rob = with_overrides(robot, t_min=t_min)
        for name, oris in orientation_sets(10.0).items():
            t0 = time.time()
            nominal, margins, post, n_wrench_only = scan(rob, grid, oris)
            key = f"tmin{t_min:g}_{name}"
            out["cases"][key] = {
                "t_min": t_min,
                "orientations": name,
                "n_nominal": int(len(nominal)),
                "n_nominal_without_drum_window": int(n_wrench_only),
                "extent_m": ({ax: [float(nominal[:, i].min()), float(nominal[:, i].max())]
                              for i, ax in enumerate("xyz")} if len(nominal) else None),
                "post_fault_fraction": (post / max(len(nominal), 1)).round(4).tolist(),
                "median_margin": float(np.median(margins)) if len(margins) else None,
                "seconds": round(time.time() - t0, 1),
            }
            print(key, out["cases"][key], flush=True)
    print("home", out["home_single_loss_margin"])
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "phase1_workspace.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

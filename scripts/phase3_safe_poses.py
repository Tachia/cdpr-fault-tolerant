"""Phase 3: poses that the remaining cables can hold after a cable loss (methods, Sec. 8).

For every cable i and every position of a grid around the operating region of T1 and T2,
the orientations of a 10 deg grid with |angle| <= 20 deg about each axis are tried in order
of increasing tilt (largest absolute angle, then the sum of absolute angles). The first one at
which the seven remaining cables balance gravity with a margin of at least s_min and every
remaining cable length lies in its drum window is stored. The planner chooses its target from
these poses, and the MPC uses them in the terminal cost of the fault scenarios.
Grid: x, y in [-1.5, 1.5] m, z in [1.5, 4.0] m, step 0.25 m.
Output: results/phase3_safe_poses.npz (positions, orientations, margins per cable) and
results/phase3_safe_poses.json (summary).
Usage: python scripts/phase3_safe_poses.py [first_cable last_cable]   (default: all cables)
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

from cdpr_ft.analysis.feasibility import static_margin, within_drum_window  # noqa: E402
from cdpr_ft.params import load_robot  # noqa: E402
from cdpr_ft.provenance import stamp  # noqa: E402

S_MIN = 10.0                       # [N] margin to both tension bounds
MAX_DEG, STEP_DEG = 20.0, 10.0
GRID = (np.arange(-1.5, 1.5 + 1e-9, 0.25), np.arange(-1.5, 1.5 + 1e-9, 0.25), np.arange(1.5, 4.0 + 1e-9, 0.25))


def orientations():
    ang = np.deg2rad(np.arange(-MAX_DEG, MAX_DEG + 1e-9, STEP_DEG))
    oris = [np.array(o) for o in product(ang, ang, ang)]
    return sorted(oris, key=lambda o: (np.abs(o).max(), np.abs(o).sum()))


def safe_set(robot, cable, oris):
    active = np.ones(robot.n, bool)
    active[cable] = False
    pos, ori, marg = [], [], []
    for x, y, z in product(*GRID):
        for o in oris:
            q = np.r_[x, y, z, o]
            if not within_drum_window(q, robot, active):
                continue
            s, _ = static_margin(q, robot, active)
            if s is not None and s >= S_MIN:
                pos.append((x, y, z))
                ori.append(o)
                marg.append(s)
                break
    return np.array(pos), np.array(ori), np.array(marg)


def main(first=0, last=None):
    robot = load_robot()
    last = robot.n - 1 if last is None else last
    oris = orientations()
    out_npz = ROOT / "results" / "phase3_safe_poses.npz"
    data = dict(np.load(out_npz)) if out_npz.exists() else {}
    summary = json.loads((ROOT / "results" / "phase3_safe_poses.json").read_text()) \
        if (ROOT / "results" / "phase3_safe_poses.json").exists() else {"cables": {}}
    for i in range(first, last + 1):
        t0 = time.time()
        pos, ori, marg = safe_set(robot, i, oris)
        data[f"pos{i}"], data[f"ori{i}"], data[f"margin{i}"] = pos, ori, marg
        n_grid = int(np.prod([len(g) for g in GRID]))
        tilt = np.rad2deg(np.abs(ori).max(1)) if len(ori) else np.array([])
        summary["cables"][str(i + 1)] = {"n_grid": n_grid, "n_safe": int(len(pos)),
                                         "share_zero_tilt": float(np.mean(tilt == 0)) if len(tilt) else 0.0,
                                         "median_margin_N": float(np.median(marg)) if len(marg) else None,
                                         "seconds": round(time.time() - t0, 1)}
        print(i + 1, summary["cables"][str(i + 1)], flush=True)
        np.savez(out_npz, **data)
    summary.update({"s_min_N": S_MIN, "max_tilt_deg": MAX_DEG, "step_deg": STEP_DEG,
                    "grid": {"x": [-1.5, 1.5], "y": [-1.5, 1.5], "z": [1.5, 4.0], "step": 0.25},
                    "provenance": stamp()})
    (ROOT / "results" / "phase3_safe_poses.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    main(*args)

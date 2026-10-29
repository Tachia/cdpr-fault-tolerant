"""Phase 6: slices of the static wrench-feasible workspace, before and after the loss of a cable.

For the manuscript figure of what a cable break leaves (figure 3). The Phase 1 study
(scripts/phase1_workspace.py) counts feasible grid points; it does not keep them. This script
evaluates the same condition on two fine planes through the paths: the horizontal plane
z = 3.0 m (the height of T1 and the middle height of T2) and the vertical plane y = 0. At every
point the platform is held at zero orientation, and the condition is that of Phase 1: the static
gravity wrench can be balanced with t_min <= t_i <= t_max by the cables that remain
(cdpr_ft.analysis.feasibility.static_margin, t_min = 10 N), and the free length of every remaining
cable lies in its drum window. Three cable sets: all eight, without cable 1 (upper) and without
cable 5 (lower). The other upper and lower cables give the same picture mirrored, by the symmetry of
the frame.
Output: results/phase6_workspace_slices.npz (axes, margins [N], drum-window flags) and .json
(summary, provenance)
Usage: python scripts/phase6_workspace_slices.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdpr_ft.analysis.feasibility import static_margin, within_drum_window  # noqa: E402
from cdpr_ft.params import load_robot  # noqa: E402
from cdpr_ft.provenance import stamp  # noqa: E402

STEP = 0.05
Z_PLANE = 3.0
Y_PLANE = 0.0
CASES = {"all": None, "without_1": 0, "without_5": 4}


def evaluate(robot, points, lost):
    active = np.ones(robot.n, bool)
    if lost is not None:
        active[lost] = False
    margin = np.full(len(points), np.nan)
    window = np.zeros(len(points), bool)
    for k, p in enumerate(points):
        q = np.r_[p, 0.0, 0.0, 0.0]
        s, _ = static_margin(q, robot, active)
        if s is not None:
            margin[k] = s
        window[k] = within_drum_window(q, robot, active)
    return margin, window


def main():
    robot = load_robot()
    xs = np.round(np.arange(-3.5, 3.5 + 1e-9, STEP), 6)
    ys = np.round(np.arange(-2.5, 2.5 + 1e-9, STEP), 6)
    zs = np.round(np.arange(1.5, 4.5 + 1e-9, STEP), 6)
    planes = {"z3": (xs, ys, lambda a, b: np.c_[a, b, np.full_like(a, Z_PLANE)]),
              "y0": (xs, zs, lambda a, b: np.c_[a, np.full_like(a, Y_PLANE), b])}
    out, summary = {"x": xs, "y": ys, "z": zs}, {}
    t0 = time.time()
    for pname, (u, v, make) in planes.items():
        uu, vv = np.meshgrid(u, v, indexing="ij")
        pts = make(uu.ravel(), vv.ravel())
        for cname, lost in CASES.items():
            m, w = evaluate(robot, pts, lost)
            out[f"{pname}_{cname}_margin"] = m.reshape(uu.shape)
            out[f"{pname}_{cname}_window"] = w.reshape(uu.shape)
            ok = (m >= 0) & w
            summary[f"{pname}_{cname}"] = {"points": int(len(m)), "feasible": int(ok.sum()),
                                           "feasible_fraction": float(ok.mean())}
            print(pname, cname, summary[f"{pname}_{cname}"], f"{time.time() - t0:.0f} s", flush=True)
    dest = ROOT / "results" / "phase6_workspace_slices.npz"
    np.savez_compressed(dest, **out)
    (ROOT / "results" / "phase6_workspace_slices.json").write_text(json.dumps(
        {"step_m": STEP, "z_plane_m": Z_PLANE, "y_plane_m": Y_PLANE, "t_min": robot.t_min, "t_max": robot.t_max,
         "orientation": "zero", "cases": {k: (None if v is None else v + 1) for k, v in CASES.items()},
         "summary": summary, "npz_sha256_16": hashlib.sha256(dest.read_bytes()).hexdigest()[:16],
         "provenance": stamp()}, indent=1))


if __name__ == "__main__":
    main()

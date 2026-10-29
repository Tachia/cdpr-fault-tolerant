"""Post-fault static feasibility when the platform orientation may adapt.

For every position that is feasible with all cables at zero orientation, and
for every single-cable loss, search a grid of orientations (|angle| <= 20 deg,
10 deg step) for one at which the 7 remaining cables balance gravity within
[t_min, t_max] and every remaining cable length stays within its drum window.
Output: results/phase1_postfault_orientation[_tminX].json
"""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdpr_ft.analysis.feasibility import static_margin, within_drum_window, workspace_grid  # noqa: E402
from cdpr_ft.params import load_robot, with_overrides  # noqa: E402


def main(max_deg=20.0, step_deg=10.0, t_min=None):
    robot = load_robot()
    if t_min is not None:
        robot = with_overrides(robot, t_min=t_min)
    angles = np.deg2rad(np.arange(-max_deg, max_deg + 1e-9, step_deg))
    # try small tilts first so the reported orientation is the least tilted one found
    oris = sorted((np.array(o) for o in product(angles, angles, angles)), key=lambda o: np.abs(o).max())
    n = robot.n
    nominal = 0
    recovered = np.zeros(n, int)
    zero_only = np.zeros(n, int)
    tilt_needed = [[] for _ in range(n)]
    for x, y, z in product(*workspace_grid()):
        q0 = np.r_[x, y, z, 0, 0, 0]
        if not within_drum_window(q0, robot):
            continue
        s, _ = static_margin(q0, robot)
        if s is None or s < 0:
            continue
        nominal += 1
        for k in range(n):
            active = np.ones(n, bool)
            active[k] = False
            for o in oris:
                q = np.r_[x, y, z, o]
                if not within_drum_window(q, robot, active):
                    continue
                s_k, _ = static_margin(q, robot, active)
                if s_k is not None and s_k >= 0:
                    recovered[k] += 1
                    tilt = float(np.rad2deg(np.abs(o).max()))
                    tilt_needed[k].append(tilt)
                    if tilt == 0.0:
                        zero_only[k] += 1
                    break
    out = {
        "t_min": robot.t_min, "max_tilt_deg": max_deg, "step_deg": step_deg,
        "n_nominal_positions": nominal,
        "fraction_feasible_zero_orientation": (zero_only / nominal).round(4).tolist(),
        "fraction_feasible_with_tilt": (recovered / nominal).round(4).tolist(),
        "median_tilt_deg_when_needed": [float(np.median([t for t in tn if t > 0])) if any(t > 0 for t in tn) else 0.0
                                        for tn in tilt_needed],
    }
    print(json.dumps(out, indent=2))
    suffix = "" if t_min is None else f"_tmin{t_min:g}"
    (ROOT / "results" / f"phase1_postfault_orientation{suffix}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(t_min=float(sys.argv[1]) if len(sys.argv) > 1 else None)

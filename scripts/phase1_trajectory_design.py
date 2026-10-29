"""Choice of the reference trajectories T1 (horizontal circle) and T2 (helix).

Specification (Phase 0, Sec. 8): radius and period are chosen from the workspace so that
the peak nominal tension is at most 0.8 t_max. A trajectory is admissible if, at every
sample, all cable lengths are within their drum windows, the platform speed and acceleration are within
v_max and a_max, and the wrench needed to follow it (gravity plus inertia, rigid cables) can
be produced with tensions in [t_min, 0.8 t_max]. The margin reported is that of the
max-margin LP with the upper bound 0.8 t_max.

The angle along the circle starts and ends at rest: its rate ramps up with a quintic
smoothstep over one period, stays constant, and ramps down over one period. The paths are
evaluated analytically by cdpr_ft.analysis.trajectories, the module the simulator uses.

For the selected trajectories the script also reports, for each single cable loss, the
fraction of path samples at which the remaining cables could still hold the platform
statically at zero orientation, which tells whether a fault can be ridden out on the
reference or forces re-planning.
Output: results/phase1_trajectory_design.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdpr_ft.analysis.feasibility import max_margin_tensions, static_margin, within_drum_window  # noqa: E402
from cdpr_ft.analysis.trajectories import path_samples  # noqa: E402
from cdpr_ft.model.kinematics import wrench_matrix  # noqa: E402
from cdpr_ft.model.rigid_body import gravity_wrench, mass_matrix_twist  # noqa: E402
from cdpr_ft.params import load_robot  # noqa: E402

UPPER_FRACTION = 0.8


def check(robot, t, p, v, a, stride=5):
    m_nu = mass_matrix_twist(np.zeros(6), robot.mass, robot.com, robot.inertia_com)
    lo = np.full(robot.n, robot.t_min)
    hi = np.full(robot.n, UPPER_FRACTION * robot.t_max)
    worst_margin, peak_t = np.inf, 0.0
    for k in range(0, len(t), stride):
        q = np.r_[p[k], 0, 0, 0]
        if not within_drum_window(q, robot):
            return {"admissible": False, "reason": "length"}
        nu_dot = np.r_[a[k], 0, 0, 0]
        # W t + w_g = M_nu nu_dot  (the bias term vanishes at zero angular velocity)
        w_ext = gravity_wrench(q, robot.mass, robot.com, robot.gravity) - m_nu @ nu_dot
        s, tens = max_margin_tensions(wrench_matrix(q, robot.anchors, robot.attachments), w_ext, lo, hi)
        if s is None or s < 0:
            return {"admissible": False, "reason": "tension"}
        worst_margin = min(worst_margin, s)
        peak_t = max(peak_t, float(tens.max()))
    speed = np.linalg.norm(v, axis=1).max()
    acc = np.linalg.norm(a, axis=1).max()
    if speed > robot.v_max or acc > robot.a_max:
        return {"admissible": False, "reason": "speed or acceleration"}
    return {"admissible": True, "min_margin_N": round(float(worst_margin), 2),
            "max_speed_m_s": round(float(speed), 3), "max_acc_m_s2": round(float(acc), 3)}


def post_fault_coverage(robot, p, stride=10):
    """Fraction of path samples at which each single cable loss is statically feasible."""
    out = []
    for k in range(robot.n):
        active = np.ones(robot.n, bool)
        active[k] = False
        ok = []
        for j in range(0, len(p), stride):
            q = np.r_[p[j], 0, 0, 0]
            s, _ = static_margin(q, robot, active)
            ok.append(s is not None and s >= 0 and within_drum_window(q, robot, active))
        out.append(round(float(np.mean(ok)), 3))
    return out


def main():
    robot = load_robot()
    center = robot.home_pose[:3]
    radii = np.round(np.arange(0.1, 2.51, 0.05), 2)
    periods = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
    table = {}
    for period in periods:
        best = None
        for radius in radii:
            t, p, v, a = path_samples(center, radius, period, turns=3, climb=0.0)
            res = check(robot, t, p, v, a)
            if not res["admissible"]:
                table[f"T{period:g}"] = {"largest_radius_m": best, "limited_by": res["reason"]}
                break
            best = float(radius)
        else:
            table[f"T{period:g}"] = {"largest_radius_m": best, "limited_by": "radius scan"}
        print(period, table[f"T{period:g}"], flush=True)

    # Selected: radius 0.5 m and period 4 s, about a third of the largest admissible radius at that
    # period (1.55 m, set by v_max), so the reference stays in the central part of the workspace and
    # its peak speed is a third of v_max; three turns, so that every fault onset of the specification
    # (3 s to 7 s) falls where the path speed is at least 89 % of its constant value (the ramp ends
    # at 4 s) and at least 5 s of constant-speed path follow the latest onset.
    sel_r, sel_T, sel_turns = 0.5, 4.0, 3
    heights = {}
    for zc in np.round(np.arange(1.5, 4.51, 0.25), 2):
        t, p, v, a = path_samples(np.r_[center[:2], zc], sel_r, sel_T, turns=sel_turns, climb=0.0)
        heights[f"{zc:g}"] = check(robot, t, p, v, a)["admissible"]
    print("heights", heights, flush=True)

    selected = {}
    t, p, v, a = path_samples(center, sel_r, sel_T, turns=sel_turns, climb=0.0)
    selected["T1_circle"] = {"center_m": center.tolist(), "radius_m": sel_r, "period_s": sel_T, "turns": sel_turns,
                             "duration_s": float(t[-1]), **check(robot, t, p, v, a, stride=1),
                             "post_fault_static_coverage": post_fault_coverage(robot, p)}
    climb = 1.0
    c2 = center - np.r_[0, 0, climb / 2]
    t, p, v, a = path_samples(c2, sel_r, sel_T, turns=sel_turns, climb=climb)
    selected["T2_helix"] = {"start_center_m": c2.tolist(), "radius_m": sel_r, "period_s": sel_T, "turns": sel_turns,
                            "climb_m": climb, "duration_s": float(t[-1]), **check(robot, t, p, v, a, stride=1),
                            "post_fault_static_coverage": post_fault_coverage(robot, p)}
    print(json.dumps(selected, indent=1), flush=True)
    out = {"upper_bound_fraction": UPPER_FRACTION, "center_m": center.tolist(),
           "largest_radius_by_period": table, "selected_circle_admissible_by_center_height": heights,
           "selected": selected}
    (ROOT / "results" / "phase1_trajectory_design.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

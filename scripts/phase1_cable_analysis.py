"""Cable sag (Ernst equivalent modulus), elastic natural frequencies and motor loading.

Cable: Dyneema, d = 1.5 mm, as on IPAnema 1 and 2 (Pott et al. 2013). EA is scaled by
cross-section from the 6 mm cable identified on IPAnema 3 by Tempel et al.:
  IFToMM WC 2015: c_ref = 112601 N/m, d_ref = 2252 N s/m at q_ref = 5 m  (nominal)
  AMM 794 (2015): c_ref = 120000 N/m, d_ref = 3600 N s/m at q_ref = 5 m  (alternative)
and swept by the factors in the configuration. Linear density: solid-section upper bound
with the UHMWPE density 970 kg/m^3 (Hwangbo et al. 2018); a braided rope is lighter, so the
sag figures are conservative.

Winch: r_d = drum radius / gear ratio; holding torque r_d * t at the motor shaft, compared
with the MSK030C ratings (M0_60, M0_100).

Tension distribution: the minimum-sum distribution (lowest motor load) is used throughout.
Workspace: zero orientation, positions within the drum windows and wrench feasible.
Natural frequencies: small oscillations about the home equilibrium with locked winches.
Output: results/phase1_cable_analysis.json
"""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdpr_ft.analysis.feasibility import within_drum_window, workspace_grid  # noqa: E402
from cdpr_ft.model.kinematics import cable_geometry, wrench_matrix  # noqa: E402
from cdpr_ft.model.rigid_body import gravity_wrench, mass_matrix_twist  # noqa: E402
from cdpr_ft.params import CONFIG_DIR, load_cable_winch, load_robot, with_overrides  # noqa: E402

CFG = yaml.safe_load((CONFIG_DIR / "robot_ipanema2_caspr.yaml").read_text())
D_CABLE = float(CFG["cable"]["diameter"])
RHO = float(CFG["cable"]["material_density"])
G = 9.81
W_LIN = RHO * np.pi * D_CABLE ** 2 / 4.0 * G      # N/m, upper bound
CABLE_SETS = {
    "tempel2015_iftomm_scaled": {"EA": float(CFG["cable"]["EA"]), "beta": float(CFG["cable"]["damping_time"])},
    "tempel2015_amm_scaled": {"EA": float(CFG["cable"]["EA_alternative"]),
                              "beta": float(CFG["cable"]["damping_time_alternative"])},
}
EA_FACTORS = [float(f) for f in CFG["cable"]["EA_sweep_factor"]]
RATED_60K = float(CFG["winch"]["torque_rated_60K"])


def ernst_ratio(ea, t, l_h):
    return 1.0 / (1.0 + (W_LIN * l_h) ** 2 * ea / (12.0 * t ** 3))


def min_sum_tensions(q, robot):
    w_mat = wrench_matrix(q, robot.anchors, robot.attachments)
    w_g = gravity_wrench(q, robot.mass, robot.com, robot.gravity)
    res = linprog(np.ones(robot.n), A_eq=w_mat, b_eq=-w_g,
                  bounds=[(robot.t_min, robot.t_max)] * robot.n, method="highs")
    return res.x if res.status == 0 else None


def workspace_states(robot):
    """Feasible zero-orientation positions within the drum windows and their minimum-sum tensions."""
    states = []
    for x, y, z in product(*workspace_grid()):
        q = np.r_[x, y, z, 0, 0, 0]
        if not within_drum_window(q, robot):
            continue
        t = min_sum_tensions(q, robot)
        if t is not None:
            states.append((q, t))
    return states


def sag_study(states, ea):
    ratios, sags = [], []
    for q, t in states:
        vec = cable_geometry(q, ROBOT.anchors, ROBOT.attachments)[0]
        l_h = np.linalg.norm(vec[:, :2], axis=1)
        ratios.extend(ernst_ratio(ea, t, l_h))
        sags.extend(W_LIN * l_h / (8.0 * t))        # mid-span sag / chord length
    r, sg = np.array(ratios), np.array(sags)
    return {
        "n_positions": len(states),
        "n_cable_states": int(r.size),
        "min_EAeq_over_EA": float(r.min()),
        "p05_EAeq_over_EA": float(np.percentile(r, 5)),
        "median_EAeq_over_EA": float(np.median(r)),
        "fraction_below_0.90": float(np.mean(r < 0.90)),
        "max_sag_over_chord": float(sg.max()),
    }


def home_modes(robot, ea, with_sag):
    """Small oscillations about the home equilibrium (minimum-sum tensions), locked winches."""
    q0 = robot.home_pose.copy()
    t_star = min_sum_tensions(q0, robot)
    vec0, length0, _, _, _ = cable_geometry(q0, robot.anchors, robot.attachments)
    ea_i = ea * (ernst_ratio(ea, t_star, np.linalg.norm(vec0[:, :2], axis=1)) if with_sag else 1.0)
    l_u = length0 * ea_i / (ea_i + t_star)
    k = ea_i / l_u

    def wrench(q):
        length = cable_geometry(q, robot.anchors, robot.attachments)[1]
        return (wrench_matrix(q, robot.anchors, robot.attachments) @ (k * (length - l_u))
                + gravity_wrench(q, robot.mass, robot.com, robot.gravity))

    eps = 1e-7
    stiff = np.zeros((6, 6))
    for j in range(6):
        dq = np.zeros(6)
        dq[j] = eps
        stiff[:, j] = -(wrench(q0 + dq) - wrench(q0 - dq)) / (2 * eps)
    mass = mass_matrix_twist(q0, robot.mass, robot.com, robot.inertia_com)
    lam = np.linalg.eigvals(np.linalg.solve(mass, 0.5 * (stiff + stiff.T)))
    return {"tensions_N": np.round(t_star, 2).tolist(),
            "EAeq_over_EA": np.round(ea_i / ea, 4).tolist() if with_sag else None,
            "natural_frequencies_Hz": np.round(np.sort(np.sqrt(np.abs(lam.real))) / (2 * np.pi), 2).tolist()}


ROBOT = load_robot()


def main():
    cw = load_cable_winch()
    r_d = cw.transmission
    home_len = cable_geometry(ROBOT.home_pose, ROBOT.anchors, ROBOT.attachments)[1]
    out = {"cable": {"diameter_m": D_CABLE, "w_N_per_m_upper_bound": W_LIN},
           "winch": {"drum_radius_m": cw.drum_radius, "gear_ratio": cw.gear_ratio, "r_d_m_per_rad": r_d,
                     "rotor_inertia_kgm2": cw.inertia,
                     "rotor_equivalent_cable_mass_kg": cw.inertia / r_d ** 2,
                     "rotor_inertia_at_drum_kgm2": cw.inertia * cw.gear_ratio ** 2},
           "motor": {"torque_rated_100K": cw.torque_rated, "torque_rated_60K": RATED_60K,
                     "torque_peak": cw.torque_peak, "speed_max_rpm": cw.speed_max * 60 / (2 * np.pi),
                     "continuous_tension_60K_N": RATED_60K / r_d,
                     "continuous_tension_100K_N": cw.torque_rated / r_d,
                     "peak_tension_N": cw.torque_peak / r_d,
                     "torque_at_t_max_Nm": r_d * ROBOT.t_max,
                     "speed_at_v_max_rpm": ROBOT.v_max / r_d * 60 / (2 * np.pi),
                     "acceleration_at_a_max_rad_s2": ROBOT.a_max / r_d},
           "home_lengths_m": np.round(home_len, 3).tolist(),
           "cases": {}}
    print(json.dumps({k: out[k] for k in ("cable", "winch", "motor", "home_lengths_m")}, indent=1), flush=True)
    states_by_tmin = {}
    for name, c in CABLE_SETS.items():
        case = {"EA_N": c["EA"], "beta_s": c["beta"]}
        for t_min in (10.0, 30.0, 40.0):
            robot = with_overrides(ROBOT, t_min=t_min)
            if t_min not in states_by_tmin:
                states_by_tmin[t_min] = workspace_states(robot)
            straight = home_modes(robot, c["EA"], False)
            sag = home_modes(robot, c["EA"], True)
            case[f"tmin{t_min:g}"] = {
                "home_tensions_N": straight["tensions_N"],
                "home_max_torque_Nm": round(max(straight["tensions_N"]) * r_d, 4),
                "home_max_torque_over_rated_60K": round(max(straight["tensions_N"]) * r_d / RATED_60K, 3),
                "home_modes_straight_Hz": straight["natural_frequencies_Hz"],
                "home_modes_with_sag_Hz": sag["natural_frequencies_Hz"],
                "home_EAeq_over_EA": sag["EAeq_over_EA"],
                "workspace_sag": sag_study(states_by_tmin[t_min], c["EA"]),
            }
        if name == "tempel2015_iftomm_scaled":
            robot = with_overrides(ROBOT, t_min=ROBOT.t_min)
            case["EA_sweep"] = {f"{f:g}": {"EA_N": f * c["EA"],
                                           "home_modes_straight_Hz": home_modes(robot, f * c["EA"], False)["natural_frequencies_Hz"],
                                           "home_modes_with_sag_Hz": home_modes(robot, f * c["EA"], True)["natural_frequencies_Hz"]}
                                for f in EA_FACTORS}
        out["cases"][name] = case
        print(name, json.dumps(case, indent=1), flush=True)
    (ROOT / "results" / "phase1_cable_analysis.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

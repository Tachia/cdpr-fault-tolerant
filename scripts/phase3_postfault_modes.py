"""Phase 3: elastic modes of the platform after a cable break, at the end poses of the recovery
library, with the holding tensions of the plan and locked winches (the same small-oscillation
analysis as the home-pose modes of Phase 1, scripts/phase1_cable_analysis.py, with seven
cables). The lowest of these modes bounds the bandwidth the controller can use after a break;
compare 5.5 Hz at the home pose with eight cables.
Output: results/phase3_postfault_modes.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdpr_ft.model.kinematics import cable_geometry, wrench_matrix  # noqa: E402
from cdpr_ft.model.rigid_body import gravity_wrench, mass_matrix_twist  # noqa: E402
from cdpr_ft.params import load_cable_winch, load_robot  # noqa: E402
from cdpr_ft.provenance import stamp  # noqa: E402


def modes(robot, ea, q0, t_star, active):
    vec0, length0, _, _, _ = cable_geometry(q0, robot.anchors, robot.attachments)
    l_u = length0 * ea / (ea + t_star)
    k = np.where(active, ea / l_u, 0.0)

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
    return np.sort(np.sqrt(np.abs(lam.real))) / (2 * np.pi), float(np.linalg.norm(wrench(q0)))


def main():
    robot, cw = load_robot(), load_cable_winch()
    d = np.load(ROOT / "results" / "phase3_recovery_library.npz")
    out = {}
    for c in range(robot.n):
        sel = np.nonzero((d["cable"] == c) & (d["branch"] >= 2))[0]
        lowest, highest, residual = [], [], []
        for k in sel:
            active = np.ones(robot.n, bool)
            active[c] = False
            f, res = modes(robot, cw.EA, d["X"][k][:6, -1], d["t_hold"][k], active)
            lowest.append(f[0])
            highest.append(f[-1])
            residual.append(res)
        out[str(c + 1)] = {"n_end_poses": len(sel), "lowest_Hz_min": float(np.min(lowest)),
                           "lowest_Hz_median": float(np.median(lowest)), "lowest_Hz_max": float(np.max(lowest)),
                           "highest_Hz_max": float(np.max(highest)), "equilibrium_residual_N_max": float(np.max(residual))}
        print(c + 1, out[str(c + 1)])
    (ROOT / "results" / "phase3_postfault_modes.json").write_text(
        json.dumps({"per_cable": out, "EA": cw.EA, "provenance": stamp()}, indent=1))


if __name__ == "__main__":
    main()

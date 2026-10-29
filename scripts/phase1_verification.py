"""Numerical values reported in Appendix A (verification of the model).

The checks use the same functions as the test suite, on 100 random poses within
+/-0.5 m and +/-0.5 rad of the home pose. Cable and winch values are test values
(the checks do not depend on them). Output: results/phase1_verification.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from conftest import TEST_CABLE_WINCH  # noqa: E402
from test_plant import equilibrium  # noqa: E402

from cdpr_ft.model import lagrange_sym as lag  # noqa: E402
from cdpr_ft.model.kinematics import (cable_geometry, euler_rate_matrix, length_jacobian,  # noqa: E402
                                      rot_zyx, twist_map, wrench_matrix)
from cdpr_ft.model.rigid_body import bias_twist, gravity_wrench, mass_matrix_twist  # noqa: E402
from cdpr_ft.params import CableWinch, load_robot  # noqa: E402


def main():
    robot = load_robot()
    rng = np.random.default_rng(20261025)
    par = lag.param_vector(robot.mass, robot.gravity, robot.com, robot.inertia_com)
    lines = []

    def pose():
        q = robot.home_pose.copy()
        q[:3] += rng.uniform(-0.5, 0.5, 3)
        q[3:] = rng.uniform(-0.5, 0.5, 3)
        return q

    worst = 0.0
    for _ in range(100):
        q = pose()
        jac = length_jacobian(q, robot.anchors, robot.attachments)
        fd = np.zeros_like(jac)
        for k in range(6):
            d = np.zeros(6)
            d[k] = 1e-6
            fd[:, k] = (cable_geometry(q + d, robot.anchors, robot.attachments)[1]
                        - cable_geometry(q - d, robot.anchors, robot.attachments)[1]) / 2e-6
        worst = max(worst, np.max(np.abs(jac - fd)) / np.max(np.abs(jac)))
    lines.append(f"jacobian_vs_fd_max_rel_err {worst:.2e}")

    wm = wg = ws = wl = wl_bad = 0.0
    f_mdot = lag.mass_derivative_function()
    min_eig = np.inf
    for _ in range(100):
        q = pose()
        dq = rng.normal(scale=0.5, size=6)
        t = rng.uniform(10, 200, robot.n)
        m_l, c_l, g_l = lag.lagrange_terms(q, dq, par)
        h = twist_map(q)
        m_nu = mass_matrix_twist(q, robot.mass, robot.com, robot.inertia_com)
        wm = max(wm, np.max(np.abs(m_l - h.T @ m_nu @ h)))
        min_eig = min(min_eig, np.min(np.linalg.eigvalsh(m_l)))
        wg = max(wg, np.max(np.abs(g_l + h.T @ gravity_wrench(q, robot.mass, robot.com, robot.gravity))))
        e = 1e-6
        n_mat = np.asarray(f_mdot(q, dq, par), float) - 2 * c_l      # dM/dt exact (symbolic)
        ws = max(ws, np.max(np.abs(n_mat + n_mat.T)))
        w_mat = wrench_matrix(q, robot.anchors, robot.attachments)
        ddq = np.linalg.solve(m_l, h.T @ w_mat @ t - c_l @ dq - g_l)
        nu = h @ dq
        dnu = np.linalg.solve(m_nu, w_mat @ t + gravity_wrench(q, robot.mass, robot.com, robot.gravity)
                              - bias_twist(q, nu, robot.mass, robot.com, robot.inertia_com))
        d_e = (euler_rate_matrix(q[3:] + e * dq[3:]) - euler_rate_matrix(q[3:] - e * dq[3:])) / (2 * e)
        wl = max(wl, np.max(np.abs(dnu - (h @ ddq + np.r_[np.zeros(3), d_e @ dq[3:]]))))
        # sensitivity of the same check: drop the centripetal force m w x (w x c_w) from b
        # (the gyroscopic term would not do, since it vanishes for the isotropic inertia of CASPR)
        omega = nu[3:]
        c_w = rot_zyx(q[3:]) @ robot.com
        centripetal = np.r_[robot.mass * np.cross(omega, np.cross(omega, c_w)), np.zeros(3)]
        dnu_bad = np.linalg.solve(m_nu, w_mat @ t + gravity_wrench(q, robot.mass, robot.com, robot.gravity)
                                  - (bias_twist(q, nu, robot.mass, robot.com, robot.inertia_com) - centripetal))
        wl_bad = max(wl_bad, np.max(np.abs(dnu_bad - (h @ ddq + np.r_[np.zeros(3), d_e @ dq[3:]]))))
    lines += [f"mass_relation_max_abs {wm:.2e}  min_eig_M {min_eig:.3f}",
              f"gravity_relation_max_abs {wg:.2e}",
              f"skew_Mdot_2C_max_abs {ws:.2e}",
              f"lagrange_vs_newton_euler_max_abs {wl:.2e}",
              f"same_check_without_centripetal_force_max_abs {wl_bad:.2e}"]

    plant, x0, t_star, tau = equilibrium(robot)
    lines.append(f"equilibrium_tensions_N {np.round(t_star, 2).tolist()}  max_abs_xdot {np.max(np.abs(plant.rhs(x0, tau))):.2e}")
    x = x0.copy()
    x[7:13] = [0.05, -0.04, 0.03, 0.05, -0.05, 0.04]
    e0 = plant.energy(x)
    ke = e0 - plant.energy(x0)
    drift = 0.0
    for _ in range(10000):
        x = plant.step(x, tau, 1e-4, lock_winches=True)
        drift = max(drift, abs(plant.energy(x) - e0))
    lines.append(f"energy_drift_over_KE0_1s_dt1e-4 {drift / ke:.2e}")

    cw = CableWinch(EA=TEST_CABLE_WINCH.EA, damping_time=2e-3, drum_radius=0.05, inertia=2.0e-3,
                    viscous=1e-3, coulomb=0.0)
    plant, x0, _, tau = equilibrium(robot, cw)
    tau = tau * (1 + 0.05 * np.sin(np.arange(robot.n)))
    fin = {}
    for dt in (4e-4, 2e-4, 1e-4):
        x = x0.copy()
        for _ in range(int(round(0.5 / dt))):
            x = plant.step(x, tau, dt)
        fin[dt] = x
    a = np.max(np.abs(fin[4e-4][:7] - fin[2e-4][:7]))
    b = np.max(np.abs(fin[2e-4][:7] - fin[1e-4][:7]))
    lines.append(f"step_halving_pose_diff 4e-4vs2e-4 {a:.2e}  2e-4vs1e-4 {b:.2e}  ratio {a / b:.1f}")

    text = "\n".join(lines)
    print(text)
    (ROOT / "results" / "phase1_verification.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()

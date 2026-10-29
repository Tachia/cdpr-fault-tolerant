"""Tension distribution and a computed-torque outer loop (methods, Sec. 8, last paragraph).

The baselines that do not use the MPC, and the tests of the winch level, command tensions
with a computed-torque law in twist coordinates,
    dnu_c = dnu_ref + K_d (nu_ref - nu) + K_p e,   e = [p_ref - p; rotation error],
    w     = M_nu dnu_c + b - w_g,
distributed over the active cables by the max-margin linear programme (methods, Eq. margin)
with W t = w. When the wrench lies outside the feasible set, the closest wrench in the
least-squares sense is produced instead (bounded least squares), so a command always exists.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear

from ..analysis.feasibility import max_margin_tensions
from ..model.kinematics import rot_zyx, wrench_matrix
from ..model.rigid_body import bias_twist, gravity_wrench, mass_matrix_twist


def rotation_error(phi_ref, phi):
    """Rotation vector e with R_ref ~ exp([e]x) R for small errors (world frame)."""
    r_err = rot_zyx(phi_ref) @ rot_zyx(phi).T
    return 0.5 * np.array([r_err[2, 1] - r_err[1, 2], r_err[0, 2] - r_err[2, 0], r_err[1, 0] - r_err[0, 1]])


def distribute(w_mat, wrench, t_lo, t_hi):
    """Tensions with W t = wrench and the largest margin to the bounds; bounded least squares
    if no such tensions exist. Returns (t, margin, feasible)."""
    s, t = max_margin_tensions(w_mat, -wrench, t_lo, t_hi)
    if s is not None and s >= 0:
        return t, s, True
    res = lsq_linear(w_mat, wrench, bounds=(t_lo, t_hi), method="bvls")
    return res.x, (s if s is not None else -np.inf), False


class ComputedTorqueTDA:
    def __init__(self, robot, kp=(100.0, 100.0, 100.0, 100.0, 100.0, 100.0), kd=(20.0,) * 6):
        self.robot = robot
        self.kp, self.kd = np.asarray(kp, float), np.asarray(kd, float)

    def tensions(self, q, nu, q_ref, nu_ref, dnu_ref, active=None):
        r = self.robot
        active = np.ones(r.n, bool) if active is None else np.asarray(active, bool)
        e = np.r_[q_ref[:3] - q[:3], rotation_error(q_ref[3:], q[3:])]
        dnu_c = dnu_ref + self.kd * (nu_ref - nu) + self.kp * e
        m_nu = mass_matrix_twist(q, r.mass, r.com, r.inertia_com)
        wrench = m_nu @ dnu_c + bias_twist(q, nu, r.mass, r.com, r.inertia_com) \
            - gravity_wrench(q, r.mass, r.com, r.gravity)
        w_mat = wrench_matrix(q, r.anchors, r.attachments)[:, active]
        t = np.zeros(r.n)
        t_act, margin, ok = distribute(w_mat, wrench, np.full(active.sum(), r.t_min), np.full(active.sum(), r.t_max))
        t[active] = t_act
        return t, margin, ok

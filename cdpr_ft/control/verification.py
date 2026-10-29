"""Verification controller: cable-space PD on the motor angles with tension feedforward.

This controller is NOT the method of the study. It exists so that the simulator can be
verified (step convergence, reference solution, timing) in closed loop before the study's
estimators and controller are built (Phase 3). It is what an industrial cable robot does
by default: each drive follows a motor-angle set-point derived from the reference pose.

Set-point. Along the reference q_ref(t) (zero orientation) the feedforward tensions t_ff
solve the max-margin problem for the rigid-cable wrench M_nu dnu_ref/dt - w_g, computed at
ff_rate and interpolated. The unstretched length that gives t_ff at l(q_ref) is
l_u = l EA / (EA + t_ff), and the motor angle set-point is theta_ref = (l_u0 - l_u) / r_d.
Torque command:
    tau = r_d t_ff + I_m ddtheta_ref + Kp e + Kd de/dt,   e = theta_ref - theta_meas,
with Kp = I_m w_n^2, Kd = 2 zeta I_m w_n and de/dt from a first-order filtered difference.
The controller does not know about faults.
"""
from __future__ import annotations

import numpy as np

from ..analysis.feasibility import max_margin_tensions
from ..model.kinematics import cable_geometry, wrench_matrix
from ..model.rigid_body import gravity_wrench, mass_matrix_twist


def feedforward_tensions(robot, path, t_grid):
    p, _, a = path.sample(np.minimum(t_grid, path.duration))
    a = np.where((t_grid <= path.duration)[:, None], a, 0.0)
    m_nu = mass_matrix_twist(np.zeros(6), robot.mass, robot.com, robot.inertia_com)
    out = np.zeros((len(t_grid), robot.n))
    lo, hi = np.full(robot.n, robot.t_min), np.full(robot.n, robot.t_max)
    for k, (pk, ak) in enumerate(zip(p, a)):
        q = np.r_[pk, 0.0, 0.0, 0.0]
        w_ext = gravity_wrench(q, robot.mass, robot.com, robot.gravity) - m_nu @ np.r_[ak, 0.0, 0.0, 0.0]
        s, t = max_margin_tensions(wrench_matrix(q, robot.anchors, robot.attachments), w_ext, lo, hi)
        if s is None or s < 0:
            raise ValueError(f"reference not feasible at t = {t_grid[k]:.3f} s")
        out[k] = t
    return out


class CableSpacePD:
    def __init__(self, robot, cw, l_u0, path, dt_ctrl, t_end, bandwidth_hz=30.0, zeta=0.7,
                 ff_rate=100.0, deriv_cutoff_hz=150.0):
        self.dt = dt_ctrl
        self.r_d = cw.transmission
        self.inertia = cw.inertia
        tk = np.arange(0.0, t_end + 0.5 * dt_ctrl, dt_ctrl)
        tg = np.arange(0.0, t_end + 0.5 / ff_rate, 1.0 / ff_rate)
        t_ff_g = feedforward_tensions(robot, path, tg)
        self.t_ff = np.stack([np.interp(tk, tg, t_ff_g[:, i]) for i in range(robot.n)], 1)
        p, _, _ = path.sample(np.minimum(tk, path.duration))
        self.q_ref = np.hstack([p, np.zeros((len(tk), 3))])
        length = np.array([cable_geometry(q, robot.anchors, robot.attachments)[1] for q in self.q_ref])
        l_u = length * cw.EA / (cw.EA + self.t_ff)
        self.theta_ref = (np.asarray(l_u0) - l_u) / self.r_d
        self.dtheta_ref = np.gradient(self.theta_ref, dt_ctrl, axis=0)
        self.ddtheta_ref = np.gradient(self.dtheta_ref, dt_ctrl, axis=0)
        w_n = 2.0 * np.pi * bandwidth_hz
        self.kp = cw.inertia * w_n ** 2
        self.kd = 2.0 * zeta * cw.inertia * w_n
        self.alpha = np.exp(-2.0 * np.pi * deriv_cutoff_hz * dt_ctrl)
        self.e_prev = None
        self.de = np.zeros(robot.n)

    def feedforward(self, k):
        return self.r_d * self.t_ff[k] + self.inertia * self.ddtheta_ref[k]

    def initial_torque(self):
        return self.r_d * self.t_ff[0]

    def __call__(self, k, t, y):
        if y is None:
            return self.feedforward(k)
        e = self.theta_ref[k] - y["theta"]
        if self.e_prev is not None:
            self.de = self.alpha * self.de + (1.0 - self.alpha) * (e - self.e_prev) / self.dt
        self.e_prev = e
        return self.feedforward(k) + self.kp * e + self.kd * self.de

"""Rigid-cable platform model in CasADi: the prediction model of the estimators and the MPC.

State x = [q; nu] in R^12 (pose with ZYX Euler angles, twist with world-frame omega), input
the cable tensions t in R^n. Dynamics (methods, Eq. NE):
    dq/dt = H(q)^-1 nu,   M_nu(q) dnu/dt + b(q, nu) = W(q) t + w_g(q),
solved for dnu/dt in closed form (see plant_casadi). Discretisation: one classical RK4 step
of length dt with t held. Jacobians are exact (algorithmic differentiation).
Also provided: cable lengths, wrench matrix, and the unstretched lengths
l_u = EA l / (EA + t) that the encoders measure when cable i carries tension t_i
(methods, Eq. h).
"""
from __future__ import annotations

from functools import cached_property

import casadi as ca
import numpy as np

from .plant_casadi import _euler_rate_inverse, _rot_zyx


class RigidModel:
    def __init__(self, robot, EA: float | None = None):
        self.robot = robot
        self.n = n = robot.n
        q = ca.SX.sym("q", 6)
        nu = ca.SX.sym("nu", 6)
        t = ca.SX.sym("t", n)
        dt = ca.SX.sym("dt")
        w_ext = ca.SX.sym("w_ext", 6)
        x = ca.vertcat(q, nu)
        p, phi, v, w = q[0:3], q[3:6], nu[0:3], nu[3:6]
        rot = _rot_zyx(phi)
        anchors, attach = ca.DM(robot.anchors), ca.DM(robot.attachments)
        cols, lengths = [], []
        for i in range(n):
            rb = rot @ attach[i, :].T
            vec = anchors[i, :].T - p - rb
            length = ca.norm_2(vec)
            u = vec / length
            cols.append(ca.vertcat(u, ca.cross(rb, u)))
            lengths.append(length)
        wmat = ca.horzcat(*cols)
        length = ca.vertcat(*lengths)
        m, g = robot.mass, robot.gravity
        c_w = rot @ ca.DM(robot.com)
        fg = ca.DM([0.0, 0.0, -m * g])
        cent = ca.cross(w, ca.cross(w, c_w))
        i_c = rot @ ca.DM(robot.inertia_com) @ rot.T
        i_c_inv = rot @ ca.DM(np.linalg.inv(robot.inertia_com)) @ rot.T
        wrench = wmat @ t + w_ext
        r_v = wrench[0:3] + fg - m * cent
        r_w = wrench[3:6] + ca.cross(c_w, fg) - (ca.cross(w, i_c @ w) + m * ca.cross(c_w, cent))
        dw = i_c_inv @ (r_w - ca.cross(c_w, r_v))
        dv = r_v / m + ca.cross(c_w, dw)
        xdot = ca.vertcat(v, _euler_rate_inverse(phi) @ w, dv, dw)

        # f_ext and step_ext add an external wrench w_ext (about P, world frame), used by the
        # estimators for the slowly varying wrench that the nominal model misses; f and step
        # are the same model with w_ext = 0
        self.f_ext = ca.Function("rigid_f_ext", [x, t, w_ext], [xdot], ["x", "t", "w"], ["xdot"])
        k1 = self.f_ext(x, t, w_ext)
        k2 = self.f_ext(x + 0.5 * dt * k1, t, w_ext)
        k3 = self.f_ext(x + 0.5 * dt * k2, t, w_ext)
        k4 = self.f_ext(x + dt * k3, t, w_ext)
        x_next_w = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        self.step_ext = ca.Function("rigid_rk4_ext", [x, t, w_ext, dt], [x_next_w], ["x", "t", "w", "dt"], ["x_next"])
        zero = ca.DM.zeros(6)
        self.f = ca.Function("rigid_f", [x, t], [self.f_ext(x, t, zero)], ["x", "t"], ["xdot"])
        x_next = self.step_ext(x, t, zero, dt)
        self.step = ca.Function("rigid_rk4", [x, t, dt], [x_next], ["x", "t", "dt"], ["x_next"])
        self.step_jacobians = ca.Function("rigid_rk4_jacobians", [x, t, dt],
                                          [ca.jacobian(x_next, x), ca.jacobian(x_next, t)],
                                          ["x", "t", "dt"], ["A", "B"])
        self.lengths = ca.Function("cable_lengths", [x], [length], ["x"], ["l"])
        self.wrench_matrix = ca.Function("wrench_matrix", [x], [wmat], ["x"], ["W"])
        self._x, self._t, self._length = x, t, length
        self.EA = EA

    @cached_property
    def unstretched(self):
        """l_u(x, t) = EA l(q) / (EA + t) and its Jacobian with respect to x."""
        if self.EA is None:
            raise ValueError("EA not given")
        l_u = self.EA * self._length / (self.EA + self._t)
        return ca.Function("unstretched_lengths", [self._x, self._t], [l_u, ca.jacobian(l_u, self._x)],
                           ["x", "t"], ["l_u", "dl_u_dx"])

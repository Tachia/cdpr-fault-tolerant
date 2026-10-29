"""Compiled version of the simulation plant (CasADi), for long simulation campaigns.

It implements exactly the equations of plant.Plant (quaternion orientation, elastic
unilateral cables, geared winches with drive saturation and speed limiter, faults) and is
checked against it to round-off in tests/test_plant_casadi.py. Differences in form only:
  * the twist equations are solved in closed form: with the inertia matrix about P,
        M_nu = [[m I, -m [c_w]x], [m [c_w]x, I_c - m [c_w]x^2]],
    the second block row minus [c_w]x times the first gives
        I_c domega/dt = r_omega - c_w x r_v,   dv/dt = r_v / m + c_w x domega/dt,
    with I_c^-1 = R I_G^-1 R^T, so no 6x6 solve is needed;
  * one call advances the state over a whole control interval (n_sub RK4 steps with the
    input held), so Python is entered once per control sample.
The plant parameters are numeric constants of the generated function; a new function is
built for every parameter set (a few tens of milliseconds).
"""
from __future__ import annotations

import casadi as ca
import numpy as np

from .plant import Faults, Plant


def _rot_zyx(phi):
    a, b, c = phi[0], phi[1], phi[2]
    rz = ca.vertcat(ca.horzcat(ca.cos(c), -ca.sin(c), 0), ca.horzcat(ca.sin(c), ca.cos(c), 0), ca.horzcat(0, 0, 1))
    ry = ca.vertcat(ca.horzcat(ca.cos(b), 0, ca.sin(b)), ca.horzcat(0, 1, 0), ca.horzcat(-ca.sin(b), 0, ca.cos(b)))
    rx = ca.vertcat(ca.horzcat(1, 0, 0), ca.horzcat(0, ca.cos(a), -ca.sin(a)), ca.horzcat(0, ca.sin(a), ca.cos(a)))
    return rz @ ry @ rx


def _quat_to_rot(qt):
    qt = qt / ca.norm_2(qt)
    w, x, y, z = qt[0], qt[1], qt[2], qt[3]
    return ca.vertcat(ca.horzcat(1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
                      ca.horzcat(2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
                      ca.horzcat(2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)))


def _euler_rate_inverse(phi):
    """E(phi)^-1 for ZYX angles: dphi/dt = E^-1 omega."""
    b, c = phi[1], phi[2]
    cb, sb, cc, sc = ca.cos(b), ca.sin(b), ca.cos(c), ca.sin(c)
    return ca.vertcat(ca.horzcat(cc / cb, sc / cb, 0),
                      ca.horzcat(-sc, cc, 0),
                      ca.horzcat(cc * sb / cb, sc * sb / cb, 1))


class CasadiPlant:
    """Same interface for stepping as plant.Plant, one call per control interval."""

    def __init__(self, plant: Plant, dt_plant: float, n_sub: int, jit: bool = False, lock_winches: bool = False):
        self.plant = plant
        self.n = n = plant.n
        self.dt_plant, self.n_sub = dt_plant, n_sub
        r, cw = plant.robot, plant.cw
        x = ca.SX.sym("x", 13 + 2 * n)
        tau_cmd = ca.SX.sym("tau", n)
        broken = ca.SX.sym("broken", n)
        rho = ca.SX.sym("rho", n)
        bias = ca.SX.sym("bias", n)

        p, quat, v, w = x[0:3], x[3:7], x[7:10], x[10:13]
        theta, dtheta = x[13:13 + n], x[13 + n:13 + 2 * n]
        rot = _quat_to_rot(quat)
        anchors, attach = ca.DM(r.anchors), ca.DM(r.attachments)
        rd = cw.transmission
        l_u = ca.DM(plant.l_u0) - rd * theta
        forces = ca.SX.zeros(3, 1)
        moments = ca.SX.zeros(3, 1)
        tensions = []
        for i in range(n):
            rb = rot @ attach[i, :].T
            vec = anchors[i, :].T - p - rb
            length = ca.norm_2(vec)
            u = vec / length
            dlength = -(ca.dot(u, v) + ca.dot(ca.cross(rb, u), w))
            delta = length - l_u[i]
            ddelta = dlength + rd * dtheta[i]
            k = cw.EA / l_u[i]
            if cw.damping_onset > 0:
                ratio = ca.fmin(ca.fmax(delta / cw.damping_onset, 0), 1)
                s = ratio * ratio * (3 - 2 * ratio)
            else:
                s = 1
            t_lin = k * delta + cw.damping_time * k * s * ddelta
            t_i = ca.if_else(delta > 0, ca.fmax(t_lin, 0), 0) * (1 - broken[i])
            tensions.append(t_i)
            forces += t_i * u
            moments += t_i * ca.cross(rb, u)
        tension = ca.vertcat(*tensions)

        m, g = r.mass, r.gravity
        c_w = rot @ ca.DM(r.com)
        fg = ca.DM([0.0, 0.0, -m * g])
        cent = ca.cross(w, ca.cross(w, c_w))
        i_c_inv = rot @ ca.DM(np.linalg.inv(r.inertia_com)) @ rot.T
        i_c = rot @ ca.DM(r.inertia_com) @ rot.T
        r_v = forces + fg - m * cent
        r_w = moments + ca.cross(c_w, fg) - (ca.cross(w, i_c @ w) + m * ca.cross(c_w, cent))
        dw = i_c_inv @ (r_w - ca.cross(c_w, r_v))
        dv = r_v / m + ca.cross(c_w, dw)
        qn = quat / ca.norm_2(quat)
        dquat = 0.5 * ca.vertcat(-ca.dot(w, qn[1:4]), qn[0] * w + ca.cross(w, qn[1:4]))

        tau = ca.fmin(ca.fmax(tau_cmd, -cw.torque_peak), cw.torque_peak)
        if np.isfinite(cw.speed_max) and cw.speed_limit_band > 0:
            band = cw.speed_limit_band * cw.speed_max
            scale = ca.fmin(ca.fmax((cw.speed_max - ca.fabs(dtheta)) / band, 0), 1)
            tau = ca.if_else(tau * dtheta > 0, tau * scale, tau)
        tau_drive = tau
        tau_act = (1 - rho) * tau_drive + bias
        friction = cw.viscous * dtheta + cw.coulomb * ca.tanh(dtheta / cw.coulomb_eps)
        ddtheta = (tau_act - rd * tension - friction) / cw.inertia
        if lock_winches:
            ddtheta = ca.SX.zeros(n, 1)
        xdot = ca.vertcat(v, dquat, dv, dw, dtheta, ddtheta)

        opts = {"jit": True, "compiler": "shell", "jit_options": {"flags": ["-O2"]}} if jit else {}
        f = ca.Function("f", [x, tau_cmd, broken, rho, bias], [xdot])
        h = dt_plant
        k1 = f(x, tau_cmd, broken, rho, bias)
        k2 = f(x + 0.5 * h * k1, tau_cmd, broken, rho, bias)
        k3 = f(x + 0.5 * h * k2, tau_cmd, broken, rho, bias)
        k4 = f(x + h * k3, tau_cmd, broken, rho, bias)
        x_new = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        x_new = ca.vertcat(x_new[0:3], x_new[3:7] / ca.norm_2(x_new[3:7]), x_new[7:])
        rk4 = ca.Function("rk4", [x, tau_cmd, broken, rho, bias], [x_new])
        xs = x
        for _ in range(n_sub):
            xs = rk4(xs, tau_cmd, broken, rho, bias)
        self.f = ca.Function("f", [x, tau_cmd, broken, rho, bias], [xdot], opts)
        self.advance = ca.Function("advance", [x, tau_cmd, broken, rho, bias], [xs], opts)
        self.outputs = ca.Function("outputs", [x, tau_cmd, broken], [tension, tau_drive], opts)
        self.jac = ca.Function("plant_jacobian", [x, tau_cmd, broken, rho, bias], [ca.jacobian(xdot, x)], opts)

    @staticmethod
    def _fault_args(faults: Faults | None, n: int):
        if faults is None:
            return np.zeros(n), np.zeros(n), np.zeros(n)
        return faults.broken.astype(float), faults.rho, faults.bias

    def rhs(self, x, tau_cmd, faults: Faults | None = None):
        return np.asarray(self.f(x, tau_cmd, *self._fault_args(faults, self.n))).ravel()

    def jacobian(self, x, tau_cmd, faults: Faults | None = None):
        return np.asarray(self.jac(x, tau_cmd, *self._fault_args(faults, self.n)))

    def step_interval(self, x, tau_cmd, faults: Faults | None = None):
        """Advance over one control interval, n_sub * dt_plant."""
        return np.asarray(self.advance(x, tau_cmd, *self._fault_args(faults, self.n))).ravel()

    def tension_and_drive(self, x, tau_cmd, faults: Faults | None = None):
        broken = self._fault_args(faults, self.n)[0]
        t, tau = self.outputs(x, tau_cmd, broken)
        return np.asarray(t).ravel(), np.asarray(tau).ravel()

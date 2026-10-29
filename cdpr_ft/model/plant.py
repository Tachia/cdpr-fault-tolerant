"""Simulation plant: rigid platform, elastic unilateral cables, winch dynamics, faults.

State x = [p(3), quat(4), v(3), omega(3), theta(n), dtheta(n)], size 13 + 2n.
The orientation is integrated as a unit quaternion (w, x, y, z), renormalised after every
step, so that the plant is free of the ZYX Euler singularity; Euler angles are recovered
with pose_euler(). omega is the world-frame angular velocity, dquat/dt = 1/2 [0; omega] (x) quat.
Winch convention: theta is the motor angle, theta > 0 reels cable in,
l_u = l_u0 - r_d * theta with r_d = drum radius / gear ratio (CableWinch.transmission).
Inertia, friction and torques are referred to the motor shaft.

Cable law (Kelvin-Voigt, unilateral):
    Delta = l - l_u,  k = EA / l_u,  c = damping_time * k
    t = max(0, k Delta + c s(Delta) dDelta/dt)  if Delta > 0,  else 0,
    s(Delta) = S(Delta / Delta_s), S(r) = 3 r^2 - 2 r^3 on [0, 1] and 1 beyond.
For Delta >= Delta_s (CableWinch.damping_onset) the law is the Kelvin-Voigt law of the
identification; below it the damping fades to zero at zero stretch, so that a slack cable
that becomes taut again starts from zero force instead of jumping to c dDelta/dt (the
same reasoning as Hunt and Crossley's contact damping). Delta_s = 0 recovers the plain law.
Winch:
    I_m ddtheta = tau_act - r_d t - b_v dtheta - tau_c tanh(dtheta / eps)
Drive (actuator limits):
    tau_drive = sat(tau_cmd) limited to |tau| <= torque_peak; while motoring (tau dtheta > 0) the
    torque tapers linearly to zero over the last speed_limit_band * speed_max below speed_max.
    The motor current, and hence the measured torque, corresponds to tau_drive.
Faults (one parameterisation):
    tau_act = (1 - rho) tau_drive + f,   broken cable: t = 0 for all later times.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..params import CableWinch, Robot
from .kinematics import cable_geometry_rot, euler_from_quat, quat_from_euler, quat_rate, quat_to_rot
from .rigid_body import bias_twist_rot, gravity_wrench_rot, mass_matrix_twist_rot


@dataclass
class Faults:
    n: int
    broken: np.ndarray = field(default=None)
    rho: np.ndarray = field(default=None)
    bias: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.broken is None:
            self.broken = np.zeros(self.n, bool)
        if self.rho is None:
            self.rho = np.zeros(self.n)
        if self.bias is None:
            self.bias = np.zeros(self.n)


def damping_scale(delta, onset):
    """s(Delta): 0 at zero stretch, 1 from the onset stretch on (C1 cubic in between)."""
    if onset <= 0.0:
        return np.ones_like(delta)
    r = np.clip(delta / onset, 0.0, 1.0)
    return r * r * (3.0 - 2.0 * r)


class Plant:
    def __init__(self, robot: Robot, cw: CableWinch, l_u0: np.ndarray):
        self.robot = robot
        self.cw = cw
        self.l_u0 = np.asarray(l_u0, float)
        self.n = robot.n
        self.nx = 13 + 2 * robot.n

    # -- state helpers -----------------------------------------------------
    def split(self, x):
        """(pose = [p, quat], nu = [v, omega], theta, dtheta)."""
        n = self.n
        return x[:7], x[7:13], x[13:13 + n], x[13 + n:13 + 2 * n]

    def make_state(self, q_euler, nu=None, theta=None, dtheta=None):
        n = self.n
        q_euler = np.asarray(q_euler, float)
        return np.concatenate([q_euler[:3], quat_from_euler(q_euler[3:]),
                               np.zeros(6) if nu is None else nu,
                               np.zeros(n) if theta is None else theta,
                               np.zeros(n) if dtheta is None else dtheta])

    @staticmethod
    def pose_euler(x):
        """Pose q = [p; ZYX Euler angles] of a plant state."""
        return np.concatenate([x[:3], euler_from_quat(x[3:7])])

    def unstretched(self, theta):
        return self.l_u0 - self.cw.transmission * theta

    def drive_torque(self, tau_cmd, dtheta):
        """Torque the drive applies for a command: peak-torque saturation and speed limiter."""
        cw = self.cw
        tau = np.clip(np.asarray(tau_cmd, float), -cw.torque_peak, cw.torque_peak)
        if np.isfinite(cw.speed_max) and cw.speed_limit_band > 0.0:
            band = cw.speed_limit_band * cw.speed_max
            scale = np.clip((cw.speed_max - np.abs(dtheta)) / band, 0.0, 1.0)
            tau = np.where(tau * dtheta > 0.0, tau * scale, tau)
        return tau

    def drum_flags(self, x):
        """True for every winch whose paid-out length l_u has left its drum window
        [window_lo, window_lo + stroke] (drum overfull below, empty above)."""
        r = self.robot
        if r.window_lo is None:
            return np.zeros(self.n, bool)
        l_u = self.unstretched(self.split(x)[2])
        return (l_u < r.window_lo) | (l_u > r.window_lo + r.stroke)

    def cable_state(self, x, faults: Faults | None = None):
        """Return tensions t, stretch Delta, lengths l, and the wrench matrix W."""
        r = self.robot
        pose, nu, theta, dtheta = self.split(x)
        rot = quat_to_rot(pose[3:])
        _, length, unit, rb, _ = cable_geometry_rot(pose[:3], rot, r.anchors, r.attachments)
        w_mat = np.vstack([unit.T, np.cross(rb, unit).T])
        dlength = -w_mat.T @ nu
        l_u = self.unstretched(theta)
        delta = length - l_u
        ddelta = dlength + self.cw.transmission * dtheta
        k = self.cw.EA / l_u
        s = damping_scale(delta, self.cw.damping_onset)
        t = k * delta + self.cw.damping_time * k * s * ddelta
        t = np.where(delta > 0.0, np.maximum(t, 0.0), 0.0)
        if faults is not None:
            t = np.where(faults.broken, 0.0, t)
        return t, delta, length, w_mat

    # -- dynamics ------------------------------------------------------------
    def rhs(self, x, tau_cmd, faults: Faults | None = None, lock_winches: bool = False):
        r, cw = self.robot, self.cw
        pose, nu, theta, dtheta = self.split(x)
        rot = quat_to_rot(pose[3:])
        t, _, _, w_mat = self.cable_state(x, faults)
        mm = mass_matrix_twist_rot(rot, r.mass, r.com, r.inertia_com)
        rhs_nu = (w_mat @ t + gravity_wrench_rot(rot, r.mass, r.com, r.gravity)
                  - bias_twist_rot(rot, nu, r.mass, r.com, r.inertia_com))
        dnu = np.linalg.solve(mm, rhs_nu)
        dquat = quat_rate(pose[3:] / np.linalg.norm(pose[3:]), nu[3:])
        if lock_winches:
            ddtheta = np.zeros(self.n)
        else:
            tau = self.drive_torque(tau_cmd, dtheta)
            if faults is not None:
                tau = (1.0 - faults.rho) * tau + faults.bias
            friction = cw.viscous * dtheta + cw.coulomb * np.tanh(dtheta / cw.coulomb_eps)
            ddtheta = (tau - cw.transmission * t - friction) / cw.inertia
        return np.concatenate([nu[:3], dquat, dnu, dtheta, ddtheta])

    def step(self, x, tau_cmd, dt, faults: Faults | None = None, lock_winches: bool = False):
        """Classical RK4 with the input held constant over the step (zero-order hold),
        followed by renormalisation of the quaternion."""
        f = lambda s: self.rhs(s, tau_cmd, faults, lock_winches)
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        x_new = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        x_new[3:7] /= np.linalg.norm(x_new[3:7])
        return x_new

    # -- energy (verification) ----------------------------------------------
    def energy(self, x, lock_winches: bool = True):
        """Kinetic + gravitational + elastic energy; conserved when the cables are undamped and taut
        and the winches are locked."""
        r, cw = self.robot, self.cw
        pose, nu, theta, dtheta = self.split(x)
        rot = quat_to_rot(pose[3:])
        mm = mass_matrix_twist_rot(rot, r.mass, r.com, r.inertia_com)
        _, delta, _, _ = self.cable_state(x)
        k = cw.EA / self.unstretched(theta)
        e_cable = 0.5 * np.sum(k * np.where(delta > 0, delta, 0.0) ** 2)
        z_com = pose[2] + (rot @ r.com)[2]
        e = 0.5 * nu @ mm @ nu + r.mass * r.gravity * z_com + e_cable
        if not lock_winches:
            e += 0.5 * cw.inertia * dtheta @ dtheta
        return e

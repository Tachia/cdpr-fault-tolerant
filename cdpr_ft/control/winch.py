"""Winch level: extended state observer (ESO) and tension loop, one per winch (methods, Sec. 6.1).

Each winch is written as  ddtheta = b0 tau + d,  b0 = 1 / I_m, where the total disturbance
d = -(r_d t + friction + rho tau - f) / I_m collects the cable load, friction and any actuator
fault. A third-order ESO estimates (theta, dtheta, d) from the encoder and the measured motor
torque, and the cable tension follows from d:

    t_hat = -(I_m z3 + b_m z2 + tau_c tanh(z2 / eps)) / r_d          (methods, Eq. that)

Discrete form. The observer runs at the inner rate as a current estimator on the exact
zero-order-hold discretisation (Phi, Gamma) of the chain (theta, dtheta, d):
    prediction  z- = Phi z + Gamma tau
    correction  z  = z- + L (theta_meas - z-_1)
The gain L places the three poles of the error dynamics (I - L C) Phi at exp(-omega_o dt), the
discrete image of the continuous triple pole at -omega_o (bandwidth parameterisation, Gao 2003).

The measurements reach the controller one sample late. The observer therefore estimates the
state at the time of the measurement it has just received, with the torque the drive applied
over the interval that ended there.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm


def eso_matrices(omega_o: float, b0: float, dt: float):
    """Phi (3x3), Gamma (3,), L (3,) of the discrete current-estimator ESO."""
    a = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    bu = np.array([0.0, b0, 0.0])
    m = np.zeros((4, 4))
    m[:3, :3], m[:3, 3] = a, bu
    em = expm(m * dt)
    phi, gamma = em[:3, :3], em[:3, 3]
    # Ackermann for the pair (Phi, C Phi): error dynamics (I - L C) Phi with poles at p (triple)
    c = np.array([1.0, 0.0, 0.0])
    p = np.exp(-omega_o * dt)
    obs = np.vstack([c @ phi, c @ phi @ phi, c @ phi @ phi @ phi])
    alpha = np.linalg.matrix_power(phi, 3) - 3 * p * phi @ phi + 3 * p * p * phi - p ** 3 * np.eye(3)
    gain = alpha @ np.linalg.solve(obs, np.array([0.0, 0.0, 1.0]))
    return phi, gamma, gain


class WinchESO:
    """Vectorised ESO for n winches."""

    def __init__(self, n: int, omega_o: float, inertia: float, transmission: float, dt: float,
                 viscous: float = 0.0, coulomb: float = 0.0, coulomb_eps: float = 1e-2):
        self.n = n
        self.inertia, self.r_d = inertia, transmission
        self.viscous, self.coulomb, self.eps = viscous, coulomb, coulomb_eps
        self.phi, self.gamma, self.gain = eso_matrices(omega_o, 1.0 / inertia, dt)
        self.z = None

    def reset(self, theta: np.ndarray, tension: np.ndarray):
        """Start at rest with the given angles and tensions (d = -r_d t / I_m)."""
        self.z = np.zeros((self.n, 3))
        self.z[:, 0] = theta
        self.z[:, 2] = -self.r_d * np.asarray(tension) / self.inertia

    def update(self, theta_meas: np.ndarray, tau_meas: np.ndarray):
        z_pred = self.z @ self.phi.T + np.outer(tau_meas, self.gamma)
        self.z = z_pred + np.outer(theta_meas - z_pred[:, 0], self.gain)
        return self.z

    @property
    def speed(self):
        return self.z[:, 1]

    def tension(self):
        friction = self.viscous * self.z[:, 1] + self.coulomb * np.tanh(self.z[:, 1] / self.eps)
        return -(self.inertia * self.z[:, 2] + friction) / self.r_d


class TensionLoop:
    """tau = r_d t_d + friction model + I_m ddtheta_ff + r_d (k_P e + k_I int e),  e = t_d - t_hat.

    The integrator is clamped (anti-windup) to +/- i_max newtons of tension correction. A winch
    can be switched to holding its angle (after its cable is declared broken), in which case it
    runs a PD on the motor angle instead."""

    def __init__(self, n: int, inertia: float, transmission: float, dt: float, k_p: float, k_i: float,
                 i_max: float, viscous: float = 0.0, coulomb: float = 0.0, coulomb_eps: float = 1e-2,
                 hold_bandwidth_hz: float = 20.0):
        self.n, self.inertia, self.r_d, self.dt = n, inertia, transmission, dt
        self.k_p, self.k_i, self.i_max = k_p, k_i, i_max
        self.viscous, self.coulomb, self.eps = viscous, coulomb, coulomb_eps
        w = 2 * np.pi * hold_bandwidth_hz
        self.hold_kp, self.hold_kd = inertia * w * w, 2 * 0.7 * inertia * w
        self.integral = np.zeros(n)
        self.hold = np.zeros(n, bool)
        self.hold_angle = np.zeros(n)

    def hold_winch(self, i: int, angle: float):
        self.hold[i] = True
        self.hold_angle[i] = angle
        self.integral[i] = 0.0

    def command(self, t_d, t_hat, ddtheta_ff, speed, angle):
        e = np.asarray(t_d) - np.asarray(t_hat)
        self.integral = np.clip(self.integral + self.k_i * e * self.dt, -self.i_max, self.i_max)
        friction = self.viscous * speed + self.coulomb * np.tanh(speed / self.eps)
        tau = self.r_d * np.asarray(t_d) + friction + self.inertia * np.asarray(ddtheta_ff) \
            + self.r_d * (self.k_p * e + self.integral)
        if self.hold.any():
            hold_tau = self.hold_kp * (self.hold_angle - angle) - self.hold_kd * speed
            tau = np.where(self.hold, hold_tau, tau)
            self.integral = np.where(self.hold, 0.0, self.integral)
        return tau

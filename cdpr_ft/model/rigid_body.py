"""Platform dynamics in twist coordinates (Newton-Euler about the point P).

    M_nu(q) dnu/dt + b(q, nu) = W(q) t + w_g(q)

with the COM at c (platform frame), c_w = R c, I_c = R I_G R^T and
I_P = I_c - m [c_w]x^2 the inertia about P.
"""
from __future__ import annotations

import numpy as np

from .kinematics import rot_zyx, skew


def mass_matrix_twist(q: np.ndarray, m: float, com: np.ndarray, inertia_com: np.ndarray) -> np.ndarray:
    return mass_matrix_twist_rot(rot_zyx(q[3:]), m, com, inertia_com)


def mass_matrix_twist_rot(rot: np.ndarray, m: float, com: np.ndarray, inertia_com: np.ndarray) -> np.ndarray:
    cw = rot @ com
    sc = skew(cw)
    i_c = rot @ inertia_com @ rot.T
    mm = np.zeros((6, 6))
    mm[:3, :3] = m * np.eye(3)
    mm[:3, 3:] = -m * sc
    mm[3:, :3] = m * sc
    mm[3:, 3:] = i_c - m * sc @ sc
    return mm


def bias_twist(q: np.ndarray, nu: np.ndarray, m: float, com: np.ndarray, inertia_com: np.ndarray) -> np.ndarray:
    return bias_twist_rot(rot_zyx(q[3:]), nu, m, com, inertia_com)


def bias_twist_rot(rot: np.ndarray, nu: np.ndarray, m: float, com: np.ndarray, inertia_com: np.ndarray) -> np.ndarray:
    cw = rot @ com
    w = nu[3:]
    i_c = rot @ inertia_com @ rot.T
    centripetal = np.cross(w, np.cross(w, cw))
    return np.concatenate([m * centripetal, np.cross(w, i_c @ w) + m * np.cross(cw, centripetal)])


def gravity_wrench(q: np.ndarray, m: float, com: np.ndarray, g: float) -> np.ndarray:
    return gravity_wrench_rot(rot_zyx(q[3:]), m, com, g)


def gravity_wrench_rot(rot: np.ndarray, m: float, com: np.ndarray, g: float) -> np.ndarray:
    cw = rot @ com
    fg = np.array([0.0, 0.0, -m * g])
    return np.concatenate([fg, np.cross(cw, fg)])

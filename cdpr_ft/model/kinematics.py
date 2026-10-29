"""Cable and platform kinematics.

Conventions
-----------
q = [p; phi], p position of the platform reference point P (world frame),
phi = (phi_x, theta_y, psi_z) ZYX Euler angles, R = Rz(psi) Ry(theta) Rx(phi).
nu = [v; omega] is the platform twist with v = dp/dt and omega the angular
velocity, both expressed in the world frame; omega = E(phi) dphi/dt.
Cable vector l_i = a_i - p - R b_i points from the platform to the anchor, so a
tension t_i >= 0 acts on the platform as t_i u_i.
"""
from __future__ import annotations

import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def rot_zyx(phi: np.ndarray) -> np.ndarray:
    a, b, c = phi
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    rz = np.array([[cc, -sc, 0.0], [sc, cc, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cb, 0.0, sb], [0.0, 1.0, 0.0], [-sb, 0.0, cb]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]])
    return rz @ ry @ rx


def euler_rate_matrix(phi: np.ndarray) -> np.ndarray:
    """E(phi) with omega_world = E(phi) dphi/dt for ZYX angles."""
    _, b, c = phi
    cb, sb, cc, sc = np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    return np.array([[cc * cb, -sc, 0.0], [sc * cb, cc, 0.0], [-sb, 0.0, 1.0]])


def twist_map(q: np.ndarray) -> np.ndarray:
    """H(q) = blkdiag(I_3, E(phi)), nu = H(q) dq/dt."""
    h = np.eye(6)
    h[3:, 3:] = euler_rate_matrix(q[3:])
    return h


def cable_geometry(q: np.ndarray, anchors: np.ndarray, attachments: np.ndarray):
    """Return cable vectors L (n,3), lengths l (n,), unit vectors U (n,3),
    attachment offsets in world frame RB (n,3) and R."""
    return cable_geometry_rot(q[:3], rot_zyx(q[3:]), anchors, attachments)


def cable_geometry_rot(p: np.ndarray, rot: np.ndarray, anchors: np.ndarray, attachments: np.ndarray):
    """cable_geometry for a position p and a rotation matrix R."""
    rb = attachments @ rot.T
    vec = anchors - p - rb
    length = np.linalg.norm(vec, axis=1)
    unit = vec / length[:, None]
    return vec, length, unit, rb, rot


# ---------------------------------------------------------------- unit quaternions
# The simulation plant integrates the orientation as a unit quaternion (w, x, y, z), so that
# large post-fault rotations pass the ZYX Euler singularity at pitch +/-90 deg; Euler
# angles are recovered for the controller, the estimators and the logs.

def quat_from_euler(phi: np.ndarray) -> np.ndarray:
    """Quaternion of R = Rz(psi) Ry(theta) Rx(phi)."""
    a, b, c = 0.5 * np.asarray(phi, float)
    ca_, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    return np.array([cc * cb * ca_ + sc * sb * sa,
                     cc * cb * sa - sc * sb * ca_,
                     cc * sb * ca_ + sc * cb * sa,
                     sc * cb * ca_ - cc * sb * sa])


def quat_to_rot(qt: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(qt, float) / np.linalg.norm(qt)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def euler_from_quat(qt: np.ndarray) -> np.ndarray:
    """ZYX Euler angles (phi, theta, psi) of a unit quaternion; theta in [-pi/2, pi/2]."""
    w, x, y, z = np.asarray(qt, float) / np.linalg.norm(qt)
    return np.array([np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
                     np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0)),
                     np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))])


def quat_rate(qt: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """dq/dt for the world-frame angular velocity omega: 1/2 [0; omega] (x) q."""
    w, v = qt[0], qt[1:]
    return 0.5 * np.concatenate([[-omega @ v], w * omega + np.cross(omega, v)])


def wrench_matrix(q: np.ndarray, anchors: np.ndarray, attachments: np.ndarray) -> np.ndarray:
    """W(q) in R^{6 x n}: column i = [u_i; (R b_i) x u_i]; wrench about P = W t."""
    _, _, unit, rb, _ = cable_geometry(q, anchors, attachments)
    return np.vstack([unit.T, np.cross(rb, unit).T])


def length_rates(q: np.ndarray, nu: np.ndarray, anchors, attachments) -> np.ndarray:
    """dl/dt = -W^T nu."""
    return -wrench_matrix(q, anchors, attachments).T @ nu


def length_jacobian(q: np.ndarray, anchors, attachments) -> np.ndarray:
    """dl/dq = -W^T H(q)."""
    return -wrench_matrix(q, anchors, attachments).T @ twist_map(q)

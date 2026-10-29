"""Symbolic Lagrangian derivation of the platform dynamics in q = [p; phi].

T = 1/2 m |dp/dt + omega x (R c)|^2 + 1/2 omega^T R I_G R^T omega,  omega = E(phi) dphi/dt
V = m g e_z^T (p + R c)

Euler-Lagrange:  M(q) q'' + C(q, q') q' + G(q) = Q,  Q = H(q)^T W(q) t,
with M the Hessian of T in q', G = dV/dq and C built from the Christoffel
symbols of the first kind, so that dM/dt - 2C is skew-symmetric.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import sympy as sp

Q_SYM = sp.symbols("x y z phi theta psi", real=True)
DQ_SYM = sp.symbols("dx dy dz dphi dtheta dpsi", real=True)
PAR_SYM = sp.symbols("m g c_x c_y c_z I_xx I_yy I_zz I_xy I_xz I_yz", real=True)


def _rot_zyx(a, b, c):
    rz = sp.Matrix([[sp.cos(c), -sp.sin(c), 0], [sp.sin(c), sp.cos(c), 0], [0, 0, 1]])
    ry = sp.Matrix([[sp.cos(b), 0, sp.sin(b)], [0, 1, 0], [-sp.sin(b), 0, sp.cos(b)]])
    rx = sp.Matrix([[1, 0, 0], [0, sp.cos(a), -sp.sin(a)], [0, sp.sin(a), sp.cos(a)]])
    return rz * ry * rx


def _euler_rate(b, c):
    return sp.Matrix([[sp.cos(c) * sp.cos(b), -sp.sin(c), 0],
                      [sp.sin(c) * sp.cos(b), sp.cos(c), 0],
                      [-sp.sin(b), 0, 1]])


@lru_cache(maxsize=1)
def derive():
    """Return sympy expressions (T, V, M, C, G)."""
    x, y, z, a, b, c = Q_SYM
    dq = sp.Matrix(DQ_SYM)
    m, g, cx, cy, cz, ixx, iyy, izz, ixy, ixz, iyz = PAR_SYM
    rot = _rot_zyx(a, b, c)
    omega = _euler_rate(b, c) * dq[3:, 0]
    com_w = rot * sp.Matrix([cx, cy, cz])
    v_com = dq[:3, 0] + omega.cross(com_w)
    i_g = sp.Matrix([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    kin = sp.Rational(1, 2) * m * (v_com.T * v_com)[0] + sp.Rational(1, 2) * (omega.T * rot * i_g * rot.T * omega)[0]
    pot = m * g * (z + com_w[2])

    q = sp.Matrix(Q_SYM)
    mass = sp.hessian(kin, dq)
    grav = sp.Matrix([sp.diff(pot, qi) for qi in Q_SYM])
    n = 6
    dmass = [mass.diff(qi) for qi in Q_SYM]          # dM/dq_i
    cor = sp.zeros(n, n)
    for k in range(n):
        for j in range(n):
            cor[k, j] = sum(sp.Rational(1, 2) * (dmass[i][k, j] + dmass[j][k, i] - dmass[k][i, j]) * dq[i]
                            for i in range(n))
    return kin, pot, mass, cor, grav


@lru_cache(maxsize=1)
def mass_derivative_function():
    """Lambdified dM/dt (q, dq, par) = sum_i dM/dq_i dq_i, exact (no finite differences)."""
    _, _, mass, _, _ = derive()
    mdot = sp.zeros(6, 6)
    for qi, dqi in zip(Q_SYM, DQ_SYM):
        mdot += mass.diff(qi) * dqi
    return sp.lambdify([list(Q_SYM), list(DQ_SYM), list(PAR_SYM)], mdot, "numpy")


@lru_cache(maxsize=1)
def numeric_functions():
    """Lambdified M(q, par), C(q, dq, par), G(q, par), T(q, dq, par), V(q, par)."""
    kin, pot, mass, cor, grav = derive()
    args_q = list(Q_SYM)
    args_dq = list(DQ_SYM)
    args_p = list(PAR_SYM)
    f_m = sp.lambdify([args_q, args_p], mass, "numpy")
    f_c = sp.lambdify([args_q, args_dq, args_p], cor, "numpy")
    f_g = sp.lambdify([args_q, args_p], grav, "numpy")
    f_t = sp.lambdify([args_q, args_dq, args_p], kin, "numpy")
    f_v = sp.lambdify([args_q, args_p], pot, "numpy")
    return f_m, f_c, f_g, f_t, f_v


def param_vector(m: float, g: float, com: np.ndarray, inertia_com: np.ndarray) -> list[float]:
    i = inertia_com
    return [m, g, *com, i[0, 0], i[1, 1], i[2, 2], i[0, 1], i[0, 2], i[1, 2]]


def lagrange_terms(q, dq, par):
    f_m, f_c, f_g, _, _ = numeric_functions()
    return (np.asarray(f_m(q, par), float),
            np.asarray(f_c(q, dq, par), float),
            np.asarray(f_g(q, par), float).ravel())

"""Consistency of the Lagrangian model (q coordinates) with the Newton-Euler model
(twist coordinates), and the structural properties of M and C."""
import numpy as np

from cdpr_ft.model.kinematics import euler_rate_matrix, twist_map, wrench_matrix
from cdpr_ft.model.rigid_body import bias_twist, gravity_wrench, mass_matrix_twist

from test_kinematics import random_pose


def _par(lagrange, robot):
    return lagrange.param_vector(robot.mass, robot.gravity, robot.com, robot.inertia_com)


def test_mass_matrix_relation_and_definiteness(rng, robot, lagrange):
    par = _par(lagrange, robot)
    for _ in range(20):
        q = random_pose(rng, robot)
        m_l, _, _ = lagrange.lagrange_terms(q, np.zeros(6), par)
        h = twist_map(q)
        m_nu = mass_matrix_twist(q, robot.mass, robot.com, robot.inertia_com)
        assert np.allclose(m_l, h.T @ m_nu @ h, atol=1e-12)
        assert np.allclose(m_l, m_l.T, atol=1e-12)
        assert np.min(np.linalg.eigvalsh(m_l)) > 0.0


def test_gravity_relation(rng, robot, lagrange):
    par = _par(lagrange, robot)
    for _ in range(20):
        q = random_pose(rng, robot)
        _, _, g_l = lagrange.lagrange_terms(q, np.zeros(6), par)
        w_g = gravity_wrench(q, robot.mass, robot.com, robot.gravity)
        assert np.allclose(g_l, -twist_map(q).T @ w_g, atol=1e-11)


def test_lagrange_equals_newton_euler(rng, robot, lagrange):
    par = _par(lagrange, robot)
    eps = 1e-6
    for _ in range(20):
        q = random_pose(rng, robot)
        dq = rng.normal(scale=0.5, size=6)
        t = rng.uniform(10, 200, robot.n)
        w_mat = wrench_matrix(q, robot.anchors, robot.attachments)
        h = twist_map(q)
        m_l, c_l, g_l = lagrange.lagrange_terms(q, dq, par)
        ddq = np.linalg.solve(m_l, h.T @ w_mat @ t - c_l @ dq - g_l)
        nu = h @ dq
        m_nu = mass_matrix_twist(q, robot.mass, robot.com, robot.inertia_com)
        dnu = np.linalg.solve(m_nu, w_mat @ t + gravity_wrench(q, robot.mass, robot.com, robot.gravity)
                              - bias_twist(q, nu, robot.mass, robot.com, robot.inertia_com))
        phi, dphi = q[3:], dq[3:]
        de = (euler_rate_matrix(phi + eps * dphi) - euler_rate_matrix(phi - eps * dphi)) / (2 * eps)
        hdot_dq = np.concatenate([np.zeros(3), de @ dphi])
        assert np.allclose(dnu, h @ ddq + hdot_dq, atol=1e-7)


def test_mdot_minus_2c_is_skew(rng, robot, lagrange):
    """Specification gate: |N + N^T| < 1e-10 with N = dM/dt - 2C, dM/dt evaluated exactly."""
    par = _par(lagrange, robot)
    f_mdot = lagrange.mass_derivative_function()
    eps = 1e-6
    for _ in range(20):
        q = random_pose(rng, robot)
        dq = rng.normal(size=6)
        _, c_l, _ = lagrange.lagrange_terms(q, dq, par)
        mdot = np.asarray(f_mdot(q, dq, par), float)
        n_mat = mdot - 2 * c_l
        assert np.max(np.abs(n_mat + n_mat.T)) < 1e-10
        # the exact derivative agrees with a central difference of M along dq
        mp, _, _ = lagrange.lagrange_terms(q + eps * dq, dq, par)
        mm, _, _ = lagrange.lagrange_terms(q - eps * dq, dq, par)
        assert np.max(np.abs(mdot - (mp - mm) / (2 * eps))) < 1e-7


def test_kinetic_energy_matches_quadratic_form(rng, robot, lagrange):
    par = _par(lagrange, robot)
    _, _, _, f_t, _ = lagrange.numeric_functions()
    for _ in range(10):
        q = random_pose(rng, robot)
        dq = rng.normal(size=6)
        m_l, _, _ = lagrange.lagrange_terms(q, dq, par)
        assert np.isclose(f_t(q, dq, par), 0.5 * dq @ m_l @ dq, rtol=1e-12)

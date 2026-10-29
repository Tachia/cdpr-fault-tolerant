"""The CasADi prediction model equals the NumPy rigid-body model; its Jacobians are exact."""
import numpy as np

from cdpr_ft.model.kinematics import cable_geometry, euler_rate_matrix, wrench_matrix
from cdpr_ft.model.rigid_body import bias_twist, gravity_wrench, mass_matrix_twist
from cdpr_ft.model.rigid_casadi import RigidModel


def random_state(rng, robot):
    q = robot.home_pose.copy()
    q[:3] += rng.uniform(-0.5, 0.5, 3)
    q[3:] = rng.uniform(-0.5, 0.5, 3)
    return np.r_[q, rng.normal(scale=0.5, size=6)]


def test_dynamics_match_numpy_model(rng, robot):
    model = RigidModel(robot, EA=3.5e4)
    for _ in range(50):
        x = random_state(rng, robot)
        t = rng.uniform(10, 200, robot.n)
        q, nu = x[:6], x[6:]
        w_mat = wrench_matrix(q, robot.anchors, robot.attachments)
        m_nu = mass_matrix_twist(q, robot.mass, robot.com, robot.inertia_com)
        dnu = np.linalg.solve(m_nu, w_mat @ t + gravity_wrench(q, robot.mass, robot.com, robot.gravity)
                              - bias_twist(q, nu, robot.mass, robot.com, robot.inertia_com))
        dq = np.r_[nu[:3], np.linalg.solve(euler_rate_matrix(q[3:]), nu[3:])]
        got = np.asarray(model.f(x, t)).ravel()
        assert np.allclose(got, np.r_[dq, dnu], rtol=1e-12, atol=1e-10)
        assert np.allclose(np.asarray(model.wrench_matrix(x)), w_mat, atol=1e-14)
        assert np.allclose(np.asarray(model.lengths(x)).ravel(),
                           cable_geometry(q, robot.anchors, robot.attachments)[1], atol=1e-14)


def test_step_jacobians_against_finite_differences(rng, robot):
    model = RigidModel(robot, EA=3.5e4)
    dt, eps = 1e-3, 1e-6
    for _ in range(10):
        x = random_state(rng, robot)
        t = rng.uniform(10, 200, robot.n)
        a_mat, b_mat = (np.asarray(m) for m in model.step_jacobians(x, t, dt))
        fd_a = np.column_stack([(np.asarray(model.step(x + eps * e, t, dt)) - np.asarray(model.step(x - eps * e, t, dt))).ravel() / (2 * eps)
                                for e in np.eye(12)])
        fd_b = np.column_stack([(np.asarray(model.step(x, t + eps * e, dt)) - np.asarray(model.step(x, t - eps * e, dt))).ravel() / (2 * eps)
                                for e in np.eye(robot.n)])
        assert np.max(np.abs(a_mat - fd_a)) < 1e-7
        assert np.max(np.abs(b_mat - fd_b)) < 1e-9


def test_unstretched_length_inverts_the_cable_law(rng, robot):
    ea = 3.5e4
    model = RigidModel(robot, EA=ea)
    x = random_state(rng, robot)
    t = rng.uniform(10, 200, robot.n)
    l_u = np.asarray(model.unstretched(x, t)[0]).ravel()
    length = np.asarray(model.lengths(x)).ravel()
    assert np.allclose(ea * (length - l_u) / l_u, t, rtol=1e-12)

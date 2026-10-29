import numpy as np

from cdpr_ft.model.kinematics import (cable_geometry, euler_rate_matrix, length_jacobian, rot_zyx,
                                      skew, twist_map, wrench_matrix)


def random_pose(rng, robot, ang=0.5):
    q = robot.home_pose.copy()
    q[:3] += rng.uniform(-0.5, 0.5, 3)
    q[3:] = rng.uniform(-ang, ang, 3)
    return q


def test_rotation_is_orthonormal(rng):
    for _ in range(20):
        r = rot_zyx(rng.uniform(-1.0, 1.0, 3))
        assert np.allclose(r.T @ r, np.eye(3), atol=1e-14)
        assert np.isclose(np.linalg.det(r), 1.0)


def test_euler_rate_matrix_matches_rotation_derivative(rng):
    eps = 1e-6
    for _ in range(20):
        phi = rng.uniform(-1.0, 1.0, 3)
        dphi = rng.normal(size=3)
        rdot = (rot_zyx(phi + eps * dphi) - rot_zyx(phi - eps * dphi)) / (2 * eps)
        omega_hat = rdot @ rot_zyx(phi).T
        assert np.allclose(omega_hat, skew(euler_rate_matrix(phi) @ dphi), atol=1e-8)


def test_length_jacobian_against_finite_differences(rng, robot):
    eps = 1e-6
    for _ in range(20):
        q = random_pose(rng, robot)
        jac = length_jacobian(q, robot.anchors, robot.attachments)
        fd = np.zeros_like(jac)
        for k in range(6):
            dq = np.zeros(6)
            dq[k] = eps
            lp = cable_geometry(q + dq, robot.anchors, robot.attachments)[1]
            lm = cable_geometry(q - dq, robot.anchors, robot.attachments)[1]
            fd[:, k] = (lp - lm) / (2 * eps)
        assert np.max(np.abs(jac - fd)) / np.max(np.abs(jac)) < 1e-7


def test_wrench_power_equals_cable_power(rng, robot):
    """Cable forces do power (W t).nu on the platform, equal to -t . dl/dt."""
    for _ in range(10):
        q = random_pose(rng, robot)
        dq = rng.normal(size=6)
        t = rng.uniform(10, 200, robot.n)
        w_mat = wrench_matrix(q, robot.anchors, robot.attachments)
        nu = twist_map(q) @ dq
        dl = length_jacobian(q, robot.anchors, robot.attachments) @ dq
        assert np.isclose((w_mat @ t) @ nu, -t @ dl, rtol=1e-12)


def test_home_pose_full_rank(robot):
    w_mat = wrench_matrix(robot.home_pose, robot.anchors, robot.attachments)
    assert np.linalg.matrix_rank(w_mat) == 6

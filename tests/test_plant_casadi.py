"""The compiled plant (CasADi) reproduces the reference NumPy plant to round-off."""
import numpy as np
import pytest

from cdpr_ft.model.kinematics import quat_from_euler
from cdpr_ft.model.plant import Faults
from cdpr_ft.model.plant_casadi import CasadiPlant
from cdpr_ft.params import load_cable_winch
from cdpr_ft.sim.setup import plant_at_rest


@pytest.fixture(scope="module")
def plants(robot):
    cw = load_cable_winch(coulomb=0.02, viscous=1e-4)      # friction on, to exercise every term
    plant, x0, _ = plant_at_rest(robot, cw, robot.home_pose)
    return plant, CasadiPlant(plant, 2.5e-4, 4), x0


def random_states(rng, robot, x0, count):
    n = robot.n
    for _ in range(count):
        q = robot.home_pose.copy()
        q[:3] += rng.uniform(-0.3, 0.3, 3)
        q[3:] = rng.uniform(-1.5, 1.5, 3)                       # large rotations, quaternion state
        yield np.concatenate([q[:3], quat_from_euler(q[3:]), rng.normal(scale=0.5, size=6),
                              rng.normal(scale=20.0, size=n),   # some cables slack, some stretched
                              rng.normal(scale=400.0, size=n)]) # some winches near the speed limit


def test_rhs_matches_numpy_plant(rng, robot, plants):
    plant, fast, x0 = plants
    n = robot.n
    for x in random_states(rng, robot, x0, 50):
        tau = rng.normal(scale=3.0, size=n)                     # some commands beyond the peak torque
        faults = Faults(n, broken=rng.random(n) < 0.2, rho=rng.uniform(0, 0.6, n), bias=rng.normal(scale=0.05, size=n))
        ref = plant.rhs(x, tau, faults)
        got = fast.rhs(x, tau, faults)
        assert np.allclose(got, ref, rtol=1e-11, atol=1e-8)
        t_ref = plant.cable_state(x, faults)[0]
        t_got, tau_drive = fast.tension_and_drive(x, tau, faults)
        assert np.allclose(t_got, t_ref, rtol=1e-12, atol=1e-9)
        assert np.allclose(tau_drive, plant.drive_torque(tau, x[13 + n:]), atol=1e-14)


def test_interval_step_matches_numpy_plant(rng, robot, plants):
    plant, fast, x0 = plants
    n = robot.n
    for x in random_states(rng, robot, x0, 10):
        x[13 + n:] *= 0.05
        tau = rng.normal(scale=0.5, size=n)
        faults = Faults(n, broken=np.arange(n) == 2)
        ref = x.copy()
        for _ in range(4):
            ref = plant.step(ref, tau, 2.5e-4, faults)
        got = fast.step_interval(x, tau, faults)
        assert np.allclose(got, ref, rtol=1e-10, atol=1e-10)

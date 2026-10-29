"""Verification gates of the simulation plant (Spec, Section 11)."""
import numpy as np
import pytest

from cdpr_ft.analysis.feasibility import static_margin
from cdpr_ft.model.kinematics import cable_geometry
from cdpr_ft.model.plant import Faults, Plant
from cdpr_ft.params import CableWinch

from conftest import TEST_CABLE_WINCH


def equilibrium(robot, cw=TEST_CABLE_WINCH):
    """Plant initialised at a static equilibrium of the home pose."""
    q0 = robot.home_pose.copy()
    s, t_star = static_margin(q0, robot)
    assert s is not None and s > 0
    length = cable_geometry(q0, robot.anchors, robot.attachments)[1]
    l_u0 = length * cw.EA / (cw.EA + t_star)        # so that EA (l - l_u)/l_u = t*
    plant = Plant(robot, cw, l_u0)
    x0 = plant.make_state(q0)
    tau_hold = cw.transmission * t_star
    return plant, x0, t_star, tau_hold


def test_static_equilibrium_has_zero_acceleration(robot):
    plant, x0, t_star, tau_hold = equilibrium(robot)
    t, _, _, _ = plant.cable_state(x0)
    assert np.allclose(t, t_star, rtol=1e-10, atol=1e-9)
    assert np.max(np.abs(plant.rhs(x0, tau_hold))) < 1e-9
    assert np.all(t_star >= robot.t_min - 1e-9) and np.all(t_star <= robot.t_max + 1e-9)


def test_energy_conserved_with_locked_winches(robot):
    plant, x0, _, tau_hold = equilibrium(robot)
    x = x0.copy()
    x[7:13] = [0.05, -0.04, 0.03, 0.05, -0.05, 0.04]
    e0 = plant.energy(x)
    ke0 = plant.energy(x) - plant.energy(x0)
    dt = 1e-4
    drift = 0.0
    for _ in range(int(1.0 / dt)):
        x = plant.step(x, tau_hold, dt, lock_winches=True)
        drift = max(drift, abs(plant.energy(x) - e0))
        assert np.all(plant.cable_state(x)[1] > 0.0), "a cable went slack; test assumption violated"
    assert drift / ke0 < 1e-6


def test_step_halving(robot):
    cw = CableWinch(EA=1.0e5, damping_time=2e-3, drum_radius=0.05, inertia=2.0e-3, viscous=1e-3, coulomb=0.0)
    plant, x0, _, tau_hold = equilibrium(robot, cw)
    tau = tau_hold * (1.0 + 0.05 * np.sin(np.arange(robot.n)))
    final = {}
    for dt in (2e-4, 1e-4):
        x = x0.copy()
        for _ in range(int(round(0.5 / dt))):
            x = plant.step(x, tau, dt)
        final[dt] = x
    dx = np.abs(final[2e-4] - final[1e-4])
    assert np.max(dx[:7]) < 1e-6          # position and quaternion
    assert np.max(dx[13:21]) < 1e-6       # winch angles


def test_slack_cable_carries_no_tension(robot):
    plant, x0, _, _ = equilibrium(robot)
    plant.l_u0 = plant.l_u0.copy()
    plant.l_u0[0] += 0.05                  # pay out 5 cm: cable 1 slack
    t, delta, _, _ = plant.cable_state(x0)
    assert delta[0] < 0 and t[0] == 0.0


def test_broken_cable_carries_no_tension(robot):
    plant, x0, t_star, tau_hold = equilibrium(robot)
    faults = Faults(robot.n)
    faults.broken[2] = True
    t, _, _, _ = plant.cable_state(x0, faults)
    assert t[2] == 0.0 and np.allclose(np.delete(t, 2), np.delete(t_star, 2))
    xdot = plant.rhs(x0, tau_hold, faults)
    # unloaded winch 3 accelerates at tau / I_m
    assert np.isclose(xdot[13 + robot.n + 2], tau_hold[2] / TEST_CABLE_WINCH.inertia)


def test_loss_of_effectiveness_reduces_delivered_torque(robot):
    plant, x0, _, tau_hold = equilibrium(robot)
    faults = Faults(robot.n)
    faults.rho[4] = 0.4
    xdot = plant.rhs(x0, tau_hold, faults)
    expected = -0.4 * tau_hold[4] / TEST_CABLE_WINCH.inertia
    assert np.isclose(xdot[13 + robot.n + 4], expected)
    assert np.allclose(np.delete(xdot[13 + robot.n:], 4), 0.0, atol=1e-9)


def test_geared_winch_holds_with_reduced_torque_and_pays_out_drum_length(robot):
    """With a gearbox the holding torque is (r_D / i) t, and i motor turns pay out one drum circumference."""
    cw = CableWinch(EA=1.0e5, damping_time=0.0, drum_radius=0.05, gear_ratio=12.0,
                    inertia=3.0e-5, viscous=0.0, coulomb=0.0)
    plant, x0, t_star, tau_hold = equilibrium(robot, cw)
    assert np.allclose(tau_hold, 0.05 / 12.0 * t_star)
    assert np.max(np.abs(plant.rhs(x0, tau_hold))) < 1e-8
    theta = np.zeros(robot.n)
    theta[0] = 2.0 * np.pi * 12.0
    assert np.isclose(plant.l_u0[0] - plant.unstretched(theta)[0], 2.0 * np.pi * 0.05)

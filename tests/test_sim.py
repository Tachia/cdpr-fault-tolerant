"""Simulator components: quaternions, cable law near slack, drive limits, drum flags, sensors,
and the timing of the multirate loop."""
from copy import copy

import numpy as np
import pytest

from cdpr_ft.model.kinematics import euler_from_quat, quat_from_euler, quat_rate, quat_to_rot, rot_zyx, skew
from cdpr_ft.model.plant import Faults, damping_scale
from cdpr_ft.model.plant_casadi import CasadiPlant
from cdpr_ft.params import load_cable_winch, load_sensors
from cdpr_ft.sim.sensors import WinchSensors
from cdpr_ft.sim.setup import plant_at_rest
from cdpr_ft.sim.simulator import FaultEvent, simulate


def test_quaternion_conversions(rng):
    for _ in range(200):
        phi = rng.uniform([-3.0, -1.5, -3.0], [3.0, 1.5, 3.0])
        qt = quat_from_euler(phi)
        assert np.isclose(np.linalg.norm(qt), 1.0)
        assert np.allclose(quat_to_rot(qt), rot_zyx(phi), atol=1e-14)
        assert np.allclose(euler_from_quat(qt), phi, atol=1e-12)
        omega = rng.normal(size=3)
        h = 1e-7
        d_rot = (quat_to_rot(qt + h * quat_rate(qt, omega)) - quat_to_rot(qt - h * quat_rate(qt, omega))) / (2 * h)
        assert np.allclose(d_rot, skew(omega) @ rot_zyx(phi), atol=1e-6)    # dR/dt = [omega]x R


@pytest.fixture(scope="module")
def nominal(robot):
    cw = load_cable_winch()
    plant, x0, t0 = plant_at_rest(robot, cw, robot.home_pose)
    return cw, plant, x0, t0


def test_cable_force_is_continuous_when_a_slack_cable_tightens(robot, nominal):
    cw, plant, x0, _ = nominal
    onset = cw.damping_onset
    assert onset > 0
    assert damping_scale(np.array([0.0]), onset)[0] == 0.0 and damping_scale(np.array([onset]), onset)[0] == 1.0
    x = x0.copy()
    x[13 + robot.n] = 50.0                        # winch 1 reeling in fast: the cable tightens
    length = plant.cable_state(x)[2]
    tension = []
    for d in (-1e-9, 1e-9, onset):                # stretch just below zero, just above, at the onset
        p2 = copy(plant)
        p2.l_u0 = plant.l_u0.copy()
        p2.l_u0[0] = length[0] - d
        tension.append(p2.cable_state(x)[0][0])
    assert tension[0] == 0.0 and abs(tension[1]) < 1e-3     # no jump to c dDelta/dt at zero stretch
    k = cw.EA / (length[0] - onset)
    assert np.isclose(tension[2], k * onset + cw.damping_time * k * cw.transmission * 50.0)  # Kelvin-Voigt


def test_drive_saturation_and_speed_limiter(nominal):
    cw, plant, _, _ = nominal
    w_max = cw.speed_max
    tau = plant.drive_torque(np.array([10.0, -10.0, 1.0, 1.0, 1.0, -1.0, 1.0, 0.5]),
                             np.array([0.0, 0.0, w_max, -w_max, 0.95 * w_max, 0.95 * w_max, 0.9 * w_max, 0.0]))
    assert tau[0] == cw.torque_peak and tau[1] == -cw.torque_peak
    assert tau[2] == 0.0                      # motoring at the speed limit: no torque
    assert tau[3] == 1.0                      # braking at the speed limit: unaffected
    assert np.isclose(tau[4], 0.5)            # halfway through the 10 % band
    assert tau[5] == -1.0                     # braking
    assert np.isclose(tau[6], 1.0) and tau[7] == 0.5


def test_drum_flags(robot, nominal):
    _, plant, x0, _ = nominal
    assert not plant.drum_flags(x0).any()
    x = x0.copy()
    l_u = plant.unstretched(np.zeros(robot.n))
    x[13] = (l_u[0] - (robot.window_lo[0] - 0.01)) / plant.cw.transmission   # reel in beyond a full drum
    flags = plant.drum_flags(x)
    assert flags[0] and not flags[1:].any()


def test_sensor_model(robot):
    cfg = load_sensors()
    sens = WinchSensors(cfg, robot.n, np.random.default_rng(0))
    quantum = 2 * np.pi / cfg.encoder_counts
    theta = np.linspace(-50.0, 50.0, robot.n)
    meas = sens.encoder(theta)
    assert np.allclose(meas / quantum, np.round(meas / quantum), atol=1e-6)        # on the count grid
    assert np.all(np.abs(meas - theta) <= cfg.encoder_accuracy + quantum + 1e-12)
    taus = np.array([sens.torque(np.zeros(robot.n)) for _ in range(4000)])
    assert np.isclose(taus.std(), cfg.current_noise_torque_std, rtol=0.05)
    exact = WinchSensors(cfg, robot.n, np.random.default_rng(0), noise=False)
    assert np.array_equal(exact.encoder(theta), theta)


class Recorder:
    """Constant torque; records what it is given."""
    def __init__(self, tau):
        self.tau, self.seen = tau, []

    def initial_torque(self):
        return self.tau

    def __call__(self, k, t, y):
        self.seen.append(None if y is None else y["theta"].copy())
        return self.tau


def test_loop_delay_zero_order_hold_and_fault_onset(robot, nominal):
    cw, plant, x0, t0 = nominal
    tau = cw.transmission * t0
    ctrl = Recorder(tau)
    sens = WinchSensors(load_sensors(), robot.n, np.random.default_rng(0), noise=False)
    events = [FaultEvent(0.0105, 2, "break")]       # applied at the first sample at or after onset
    log = simulate(plant, x0, ctrl, 0.02, 2.5e-4, 1e-3, sens, events, delay_samples=1)
    assert ctrl.seen[0] is None                                     # nothing measured yet at k = 0
    for k in range(1, len(ctrl.seen)):
        assert np.array_equal(ctrl.seen[k], log["theta"][k - 1])    # one-sample delay
    t = log["t"]
    assert np.all(log["tension"][t < 0.0105, 2] > 0) and np.all(log["tension"][t >= 0.011, 2] == 0)
    fast = CasadiPlant(plant, 2.5e-4, 4)
    log_fast = simulate(plant, x0, Recorder(tau), 0.02, 2.5e-4, 1e-3, sens, events, fast=fast)
    assert np.allclose(log_fast["x"], log["x"], atol=1e-12)

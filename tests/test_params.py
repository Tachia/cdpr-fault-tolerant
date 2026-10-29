"""The study parameters load completely from the configuration, with their sourced values,
and the selected drive meets the requirements it was sized for."""
import numpy as np
from scipy.optimize import linprog

from cdpr_ft.analysis.feasibility import within_drum_window
from cdpr_ft.model.kinematics import wrench_matrix
from cdpr_ft.model.rigid_body import gravity_wrench
from cdpr_ft.params import load_cable_winch, load_robot


def test_cable_winch_parameters_load_without_overrides():
    cw = load_cable_winch()
    assert np.isclose(cw.EA, 112601.0 * 5.0 / 16.0)   # Tempel et al. 2015, scaled 6 mm -> 1.5 mm
    assert np.isclose(cw.damping_time, 2252.0 / 112601.0, rtol=1e-3)
    assert cw.drum_radius == 0.05 and cw.gear_ratio == 12.0   # Pott et al. 2013
    assert np.isclose(cw.transmission, 0.05 / 12.0)
    assert cw.inertia == 3.0e-5 and cw.torque_peak == 4.0      # MSK030C-0900 (Rexroth manual)
    assert cw.encoder_counts == 128 * 2 ** 13                   # MSK option M1, 13-bit interpolation


def test_robot_limits_load():
    robot = load_robot()
    assert robot.t_min == 10.0 and robot.t_max == 200.0
    assert robot.stroke == 6.0 and robot.v_max == 2.5 and robot.a_max == 25.0


def test_drum_windows_match_their_derivation():
    """The installed windows in the configuration are the ones phase1_drum_windows.py derives,
    each spans exactly the drum capacity, and the home pose lies inside all of them."""
    import json
    from cdpr_ft.params import CONFIG_DIR
    derived = json.loads((CONFIG_DIR.parent / "results" / "phase1_drum_windows.json").read_text())
    robot = load_robot()
    assert np.allclose(robot.window_lo, derived["window_lo_m"], atol=1e-3)
    assert derived["stroke_m"] == robot.stroke
    assert within_drum_window(robot.home_pose, robot)
    # a far corner needs more stroke than the drums have: cables 1 and 5 too long, cable 7 too short
    q_far = np.r_[3.5, -2.5, 1.5, 0, 0, 0]
    assert not within_drum_window(q_far, robot)


def test_drive_meets_speed_and_continuous_torque_requirements():
    robot, cw = load_robot(), load_cable_winch()
    assert robot.v_max / cw.transmission <= cw.speed_max
    assert cw.transmission * robot.t_max <= cw.torque_rated
    assert cw.torque_peak / cw.transmission >= robot.t_max


def test_home_pose_held_within_the_60K_rating_and_the_drum_windows():
    import yaml
    from cdpr_ft.params import CONFIG_DIR
    rated_60k = yaml.safe_load((CONFIG_DIR / "robot_ipanema2_caspr.yaml").read_text())["winch"]["torque_rated_60K"]
    robot, cw = load_robot(), load_cable_winch()
    q = robot.home_pose
    w_mat = wrench_matrix(q, robot.anchors, robot.attachments)
    w_g = gravity_wrench(q, robot.mass, robot.com, robot.gravity)
    res = linprog(np.ones(robot.n), A_eq=w_mat, b_eq=-w_g,
                  bounds=[(robot.t_min, robot.t_max)] * robot.n, method="highs")
    assert res.status == 0
    assert np.max(res.x) * cw.transmission <= rated_60k
    assert within_drum_window(q, robot)


def test_sensor_and_rate_settings_load():
    from cdpr_ft.params import inner_rate, load_sensors, plant_step
    sens = load_sensors()
    assert sens.encoder_counts == 128 * 2 ** 13 and sens.delay_samples == 1
    assert np.isclose(sens.encoder_accuracy, 120.0 / 3600.0 * np.pi / 180.0)
    assert sens.current_noise_torque_std > 0
    n_sub = (1.0 / inner_rate()) / plant_step()
    assert np.isclose(n_sub, round(n_sub))            # whole number of plant steps per inner sample

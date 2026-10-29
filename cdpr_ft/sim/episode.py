"""One simulated run (episode) of the study: plant with optionally randomised parameters, the
controller on nominal parameters, sensors, a fault, termination and metrics.

Parameter randomisation (specification, Sec. 8), uniform within:
  mass +/-10 %, each principal inertia +/-20 %, centre of mass +/-2 cm per axis,
  anchors +/-5 mm per coordinate, axial stiffness EA +/-20 %.
The controller always uses the nominal values; its homing values l_u0 are the nominal ones.

Termination ("platform lost", decided in Phase 3): the run ends when the tilt exceeds
lost_tilt_deg, when the platform leaves the frame interior, or when the drum of a cable that is
not broken leaves its window (overfull or empty).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..analysis.trajectories import reference as path_reference
from ..control.ftc import FaultTolerantController, FTCConfig
from ..model.kinematics import cable_geometry, euler_rate_matrix
from ..model.plant_casadi import CasadiPlant
from ..params import inner_rate, load_cable_winch, load_robot, load_sensors, plant_step, with_overrides
from ..sim.sensors import WinchSensors
from ..sim.setup import equilibrium_tensions, plant_at_rest
from ..sim.simulator import FaultEvent, simulate

FRAME = (np.array([-4.0, -3.0, 1.0]), np.array([4.0, 3.0, 5.0]))


@dataclass
class EpisodeSpec:
    trajectory: str = "T1"
    fault: tuple | None = None        # (kind, cable index 0-based, onset [s], value); value: rho for
                                      # "loe", and for "bias" the fraction of the winch's holding
                                      # torque at the start pose (nominal parameters)
    seed: int = 0                     # sensors and parameter randomisation
    randomise: bool = True
    noise_factor: float = 1.0
    t_end: float | None = None        # default: duration of the trajectory
    variant: str = "proposed"


def randomised_plant_robot(robot, cw, rng):
    m = robot.mass * rng.uniform(0.9, 1.1)
    inertia = np.diag(np.diag(robot.inertia_com) * rng.uniform(0.8, 1.2, 3))
    com = robot.com + rng.uniform(-0.02, 0.02, 3)
    anchors = robot.anchors + rng.uniform(-0.005, 0.005, robot.anchors.shape)
    ea = cw.EA * rng.uniform(0.8, 1.2)
    return with_overrides(robot, mass=m, inertia_com=inertia, com=com, anchors=anchors), replace(cw, EA=ea)


def reference_fn(path):
    def fn(t):
        t = np.atleast_1d(np.asarray(t, float))
        p, v, a = path.sample(np.minimum(t, path.duration))
        after = (t > path.duration)[:, None]
        v = np.where(after, 0.0, v)
        a = np.where(after, 0.0, a)
        z = np.zeros((len(t), 3))
        return np.hstack([p, z]), np.hstack([v, z]), np.hstack([a, z])
    return fn


def mode_index(n, kind, cable):
    return 1 + cable if kind == "break" else 1 + n + cable


def controller_start(spec: EpisodeSpec, robot=None, cw=None):
    """What the controller knows at the start of a run: the reference, the start pose q0, the
    nominal equilibrium tensions there and the nominal homing lengths l_u0."""
    robot = load_robot() if robot is None else robot
    cw = load_cable_winch() if cw is None else cw
    path = path_reference(spec.trajectory, home=tuple(robot.home_pose[:3]))
    ref = reference_fn(path)
    q0 = ref(0.0)[0][0]
    t_nom = equilibrium_tensions(robot, q0)
    l_nom = cable_geometry(q0, robot.anchors, robot.attachments)[1]
    return path, ref, q0, t_nom, l_nom * cw.EA / (cw.EA + t_nom)


def run_episode(spec: EpisodeSpec, imm, mpc, planner, safe_sets, ftc_cfg: FTCConfig, keep_log=False,
                alt_factory=None, return_controller=False, ctrl_kwargs=None):
    """alt_factory: for the variants with a diagnosis other than the IMM, a callable
    (robot, cw, l_u0, reference, ftc_cfg, dt, t_nom) -> diagnosis object (cdpr_ft.estimation.
    alternatives); return_controller: also return the controller (for the evaluation's logs);
    ctrl_kwargs: further keyword arguments of the controller (computed-torque gains)."""
    robot, cw = load_robot(), load_cable_winch()
    n = robot.n
    rng = np.random.default_rng(spec.seed)
    plant_robot, plant_cw = randomised_plant_robot(robot, cw, rng) if spec.randomise else (robot, cw)
    path, ref, q0, t_nom, l_u0_nominal = controller_start(spec, robot, cw)
    plant, x0, _ = plant_at_rest(plant_robot, plant_cw, q0)
    dt, dt_p = 1.0 / inner_rate(), plant_step()
    fast = CasadiPlant(plant, dt_p, int(round(dt / dt_p)))
    sensors = WinchSensors(load_sensors(noise_factor=spec.noise_factor), n, rng)
    events = []
    oracle = None
    if spec.fault is not None:
        kind, cable, onset, value = spec.fault
        if kind == "bias":
            value = value * cw.transmission * t_nom[cable]
        events = [FaultEvent(onset, cable, kind, value)]
        oracle = (mode_index(n, "break" if kind == "break" else "actuator", cable), onset)
    mpc.warm = {}
    cfg = replace(ftc_cfg, variant=spec.variant)
    alt = alt_factory(robot, cw, l_u0_nominal, ref, cfg, dt, t_nom) if alt_factory is not None else None
    ctrl = FaultTolerantController(robot, cw, l_u0_nominal, ref, imm, mpc, planner, safe_sets, cfg, dt,
                                   t_nom, q0, oracle_fault=oracle, task=spec.trajectory, alt=alt,
                                   **(ctrl_kwargs or {}))
    broken = np.zeros(n, bool)

    def stop(k, t, x, pl):
        q = pl.pose_euler(x)
        if np.rad2deg(np.abs(q[3:]).max()) > cfg.lost_tilt_deg:
            return "tilt"
        if np.any(q[:3] < FRAME[0]) or np.any(q[:3] > FRAME[1]):
            return "frame"
        if spec.fault is not None and spec.fault[0] == "break" and t >= spec.fault[2]:
            broken[spec.fault[1]] = True
        if np.any(pl.drum_flags(x) & ~broken):
            return "drum"
        return None

    t_end = path.duration if spec.t_end is None else spec.t_end
    log = simulate(plant, x0, ctrl, t_end, dt_p, dt, sensors, events, fast=fast, stop=stop)
    out = episode_metrics(spec, log, ctrl, ref, robot, keep_log)
    return (out, ctrl) if return_controller else out


def episode_metrics(spec, log, ctrl, ref, robot, keep_log):
    n = robot.n
    t = log["t"]
    q_ref = ref(t)[0]
    err = np.linalg.norm(log["q"][:, :3] - q_ref[:, :3], axis=1)
    onset = spec.fault[2] if spec.fault is not None else np.inf
    pre = t < onset
    out = {"variant": spec.variant, "trajectory": spec.trajectory, "fault": spec.fault, "seed": spec.seed,
           "lost": log["stopped"] is not None, "stopped": log["stopped"],
           "rms_error_pre_m": float(np.sqrt(np.mean(err[pre] ** 2))) if pre.any() else None,
           "max_tilt_deg": float(np.rad2deg(np.abs(log["q"][:, 3:]).max())),
           "mpc_failures": int(np.sum(~np.asarray(ctrl.log["mpc_ok"], bool))),
           "declared": None, "delay_s": None, "correct": None, "false_alarm": False}
    if ctrl.declared is not None:
        j, td = ctrl.declared
        out["declared"] = int(j)
        if spec.fault is None or td < onset:
            out["false_alarm"] = True
        else:
            kind = "break" if spec.fault[0] == "break" else "actuator"
            out["delay_s"] = float(td - onset)
            out["correct"] = bool(j == mode_index(n, kind, spec.fault[1]))
    broken = np.zeros((len(t), n), bool)
    if spec.fault is not None and spec.fault[0] == "break":
        broken[t >= onset, spec.fault[1]] = True
    outside = ((log["tension"] < robot.t_min - 1e-9) | (log["tension"] > robot.t_max + 1e-9)) & ~broken
    out["time_outside_bounds_pct"] = float(100 * np.mean(np.any(outside, axis=1)))
    post = t >= onset
    if post.any():
        intact_t = np.where(broken[post], np.nan, log["tension"][post])
        out["post_outside_bounds_pct"] = float(100 * np.mean(np.any(outside[post], axis=1)))
        out["post_tension_max"] = float(np.nanmax(intact_t))
        out["post_tension_min"] = float(np.nanmin(intact_t))
        out["post_slack_time_s"] = float(np.mean(np.any(intact_t <= 0.0, axis=1)) * (t[post][-1] - t[post][0]))
    out["slack_samples"] = int(np.sum(log["slack"]))
    if ctrl.plan is not None:
        out["plan_branch"] = ctrl.plan.branch
        if ctrl.plan.target is not None:
            out["final_target_error_m"] = float(np.linalg.norm(log["q"][-1, :3] - ctrl.plan.target[:3]))
    if keep_log:
        out["log"] = log
        out["ctrl_log"] = {k: np.asarray(v) for k, v in ctrl.log.items()}
    return out

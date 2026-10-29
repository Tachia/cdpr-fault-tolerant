"""Phase 2: step convergence of the plant integrator in closed loop (specification, Sec. 11).

Scenarios on the reference T1, 8 s, inner rate 1 kHz, sensors on (fixed seed), verification
controller (cdpr_ft.control.verification, not the study's controller):
  nominal       no fault
  break_lower   cable 5 breaks at 4 s
  break_upper   cable 1 breaks at 4 s
  loe           loss of effectiveness rho = 0.6 on winch 3 at 4 s
  stiff_break   cable 5 breaks at 4 s, with EA x 2 and the larger retardation time 0.03 s
                (the fastest cable dynamics of the robustness ranges)
For each scenario the plant step is halved from 1 ms to 31.25 us. Between a step and its half,
the change of each evaluation metric is reported (the specification asks for less than 0.1 %),
together with the largest pointwise difference of the logged trajectories relative to a scale.
The pass condition applies to the metrics; the trajectory differences are reported because
pointwise tensions during slack and snap sequences converge more slowly than the metrics.
Scales: position 0.5 m (path radius), orientation 0.1 rad, tension t_max, motor angle 2 pi.
Two further runs repeat the violent cases without sensor noise and quantisation, to separate
the integrator error from the sensitivity of the closed loop to single encoder counts.
Output: results/phase2_step_convergence.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdpr_ft.analysis.trajectories import reference  # noqa: E402
from cdpr_ft.control.verification import CableSpacePD  # noqa: E402
from cdpr_ft.model.plant_casadi import CasadiPlant  # noqa: E402
from cdpr_ft.params import inner_rate, load_cable_winch, load_robot, load_sensors  # noqa: E402
from cdpr_ft.provenance import stamp  # noqa: E402
from cdpr_ft.sim.sensors import WinchSensors  # noqa: E402
from cdpr_ft.sim.setup import plant_at_rest  # noqa: E402
from cdpr_ft.sim.simulator import FaultEvent, simulate  # noqa: E402

T_END = 8.0
STEPS = [1e-3, 5e-4, 2.5e-4, 1.25e-4, 6.25e-5, 3.125e-5]
SEED = 20261026


def scenarios():
    return {
        "nominal": ({}, []),
        "break_lower": ({}, [FaultEvent(4.0, 4, "break")]),
        "break_upper": ({}, [FaultEvent(4.0, 0, "break")]),
        "loe": ({}, [FaultEvent(4.0, 2, "loe", 0.6)]),
        "stiff_break": ({"EA_factor": 2.0, "damping_time": 0.03}, [FaultEvent(4.0, 4, "break")]),
        # the same two violent cases without sensor noise and quantisation: the integrator alone
        "break_upper_noise_free": ({"noise": False}, [FaultEvent(4.0, 0, "break")]),
        "stiff_break_noise_free": ({"EA_factor": 2.0, "damping_time": 0.03, "noise": False},
                                   [FaultEvent(4.0, 4, "break")]),
    }


def run(robot, overrides, events, dt_plant):
    base = load_cable_winch()
    cw = load_cable_winch(EA=base.EA * overrides.get("EA_factor", 1.0),
                          damping_time=overrides.get("damping_time", base.damping_time))
    path = reference("T1")
    dt_ctrl = 1.0 / inner_rate()
    q0 = np.r_[path.sample(np.array([0.0]))[0][0], 0.0, 0.0, 0.0]
    plant, x0, _ = plant_at_rest(robot, cw, q0)
    fast = CasadiPlant(plant, dt_plant, int(round(dt_ctrl / dt_plant)))
    ctrl = CableSpacePD(robot, cw, plant.l_u0, path, dt_ctrl, T_END)
    sensors = WinchSensors(load_sensors(), robot.n, np.random.default_rng(SEED), noise=overrides.get("noise", True))
    t0 = time.time()
    log = simulate(plant, x0, ctrl, T_END, dt_plant, dt_ctrl, sensors, events, fast=fast)
    log["seconds"] = time.time() - t0
    log["q_ref"] = ctrl.q_ref
    broken = np.zeros(robot.n, bool)
    for ev in events:
        if ev.kind == "break":
            broken |= (np.arange(robot.n) == ev.cable)
    after = log["t"][:, None] >= min([ev.time for ev in events], default=np.inf)
    gone = broken[None, :] & after
    log["outside"] = ((log["tension"] < robot.t_min) | (log["tension"] > robot.t_max)) & ~gone
    return log


def metrics(log):
    e = log["q"][:, :3] - log["q_ref"][:, :3]
    return {
        "rms_position_error_m": float(np.sqrt((e ** 2).sum(1).mean())),
        "max_position_error_m": float(np.linalg.norm(e, axis=1).max()),
        "max_tilt_rad": float(np.abs(log["q"][:, 3:6]).max()),
        "max_tension_N": float(log["tension"].max()),
        "time_outside_bounds_pct": float(100 * np.mean(np.any(log["outside"], axis=1))),
        "mean_tension_N": float(log["tension"].mean()),
        "final_position_m": log["q"][-1, :3].tolist(),
        "slack_samples": int(log["slack"].sum()),
        "max_motor_speed_rad_s": float(np.abs(log["dtheta"]).max()),
    }


def compare(a, b, t_max):
    scale = {"position": 0.5, "orientation": 0.1, "tension": t_max, "motor_angle": 2 * np.pi}
    d = {
        "position": np.abs(a["q"][:, :3] - b["q"][:, :3]).max() / scale["position"],
        "orientation": np.abs(a["q"][:, 3:6] - b["q"][:, 3:6]).max() / scale["orientation"],
        "tension": np.abs(a["tension"] - b["tension"]).max() / scale["tension"],
        "motor_angle": np.abs(a["theta"] - b["theta"]).max() / scale["motor_angle"],
    }
    ma, mb = metrics(a), metrics(b)
    rel = {}
    for k in ("rms_position_error_m", "max_position_error_m", "max_tilt_rad", "max_tension_N", "mean_tension_N"):
        rel[k] = abs(ma[k] - mb[k]) / max(abs(mb[k]), 1e-12)
    # a share of time: compared in percentage points of the run, not relative to itself
    rel["time_outside_bounds_pct"] = abs(ma["time_outside_bounds_pct"] - mb["time_outside_bounds_pct"]) / 100.0
    rel["final_position_m"] = float(np.linalg.norm(np.subtract(ma["final_position_m"], mb["final_position_m"])) / 0.5)
    return {"trajectory_rel_diff": {k: float(v) for k, v in d.items()}, "metric_rel_change": rel}


def main():
    robot = load_robot()
    out = {"t_end_s": T_END, "steps_s": STEPS, "seed": SEED, "scenarios": {}, "provenance": stamp()}
    for name, (over, events) in scenarios().items():
        logs = {dt: run(robot, over, events, dt) for dt in STEPS}
        entry = {"metrics": {f"{dt:g}": metrics(logs[dt]) for dt in STEPS},
                 "seconds": {f"{dt:g}": round(logs[dt]["seconds"], 2) for dt in STEPS},
                 "halving": {}}
        for dt, half in zip(STEPS[:-1], STEPS[1:]):
            c = compare(logs[dt], logs[half], robot.t_max)
            worst = max(c["metric_rel_change"].values())
            entry["halving"][f"{dt:g}->{half:g}"] = {**c, "worst_metric_change": worst,
                                                     "worst_metric": max(c["metric_rel_change"], key=c["metric_rel_change"].get),
                                                     "pass_0.1pct": bool(worst < 1e-3)}
        out["scenarios"][name] = entry
        print(name, {k: (round(v["worst_metric_change"], 7), v["pass_0.1pct"]) for k, v in entry["halving"].items()},
              flush=True)
    for dt in STEPS[:-1]:
        key = f"{dt:g}->{dt / 2:g}"
        failing = {name: (s["halving"][key]["worst_metric"], round(s["halving"][key]["worst_metric_change"], 6))
                   for name, s in out["scenarios"].items() if not s["halving"][key]["pass_0.1pct"]}
        out.setdefault("failing_by_step", {})[f"{dt:g}"] = failing
        print(f"step {dt:g}: metrics above 0.1 % on halving ->", failing or "none")
    (ROOT / "results" / "phase2_step_convergence.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

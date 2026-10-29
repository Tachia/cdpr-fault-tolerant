"""Phase 2: cost of the simulator and of the prediction model, to plan the Phase 5 campaign.

Measured on this machine (one core per process):
  * closed-loop run on T1 (16 s) with the compiled plant at the chosen step, sensors, logging
    and the verification controller, for the plant steps of the convergence study;
  * the rigid-cable prediction model of the estimators and the MPC: one RK4 step, one step
    with its Jacobians, interpreted (CasADi virtual machine) and compiled to C (JIT), and the
    17 IMM modes evaluated together.
Output: results/phase2_timing.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import casadi as ca
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdpr_ft.analysis.trajectories import reference  # noqa: E402
from cdpr_ft.control.verification import CableSpacePD  # noqa: E402
from cdpr_ft.model.plant_casadi import CasadiPlant  # noqa: E402
from cdpr_ft.model.rigid_casadi import RigidModel  # noqa: E402
from cdpr_ft.params import inner_rate, load_cable_winch, load_robot, load_sensors  # noqa: E402
from cdpr_ft.provenance import stamp  # noqa: E402
from cdpr_ft.sim.sensors import WinchSensors  # noqa: E402
from cdpr_ft.sim.setup import plant_at_rest  # noqa: E402
from cdpr_ft.sim.simulator import simulate  # noqa: E402


def per_call(fn, *args, repeat=2000):
    fn(*args)
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn(*args)
    return (time.perf_counter() - t0) / repeat


def main():
    robot, cw = load_robot(), load_cable_winch()
    out = {"cpu_count": os.cpu_count(), "closed_loop": {}, "prediction_model_us": {}, "provenance": stamp()}
    path = reference("T1")
    dt_ctrl = 1.0 / inner_rate()
    q0 = np.r_[path.sample(np.array([0.0]))[0][0], 0.0, 0.0, 0.0]
    for dt in (2.5e-4, 1.25e-4, 6.25e-5):
        plant, x0, _ = plant_at_rest(robot, cw, q0)
        t0 = time.perf_counter()
        fast = CasadiPlant(plant, dt, int(round(dt_ctrl / dt)))
        build = time.perf_counter() - t0
        t0 = time.perf_counter()
        ctrl = CableSpacePD(robot, cw, plant.l_u0, path, dt_ctrl, path.duration)
        setup = time.perf_counter() - t0
        t0 = time.perf_counter()
        simulate(plant, x0, ctrl, path.duration, dt, dt_ctrl,
                 WinchSensors(load_sensors(), robot.n, np.random.default_rng(0)), fast=fast)
        run = time.perf_counter() - t0
        out["closed_loop"][f"{dt:g}"] = {"plant_build_s": round(build, 3), "controller_setup_s": round(setup, 2),
                                         "run_s": round(run, 2), "simulated_s": path.duration,
                                         "s_per_simulated_s": round(run / path.duration, 3)}
        print(dt, out["closed_loop"][f"{dt:g}"], flush=True)

    model = RigidModel(robot, EA=cw.EA)
    x = np.r_[robot.home_pose, np.zeros(6)]
    t = np.full(robot.n, 50.0)
    res = out["prediction_model_us"]
    res["rk4_step_vm"] = per_call(model.step, x, t, dt_ctrl) * 1e6
    res["rk4_step_with_jacobians_vm"] = per_call(model.step_jacobians, x, t, dt_ctrl) * 1e6
    t0 = time.perf_counter()
    jit = ca.Function("rigid_rk4_jacobians_jit", model.step_jacobians.sx_in(),
                      model.step_jacobians.call(model.step_jacobians.sx_in()),
                      {"jit": True, "compiler": "shell", "jit_options": {"flags": ["-O2"]}})
    out["jit_build_s"] = round(time.perf_counter() - t0, 1)
    res["rk4_step_with_jacobians_jit"] = per_call(jit, x, t, dt_ctrl) * 1e6
    batch = jit.map(17)
    xs, ts = np.tile(x[:, None], (1, 17)), np.tile(t[:, None], (1, 17))
    res["17_modes_step_with_jacobians_jit"] = per_call(batch, xs, ts, dt_ctrl, repeat=500) * 1e6
    out["prediction_model_us"] = {k: round(v, 1) for k, v in res.items()}
    print(json.dumps(out["prediction_model_us"]), "jit build", out["jit_build_s"], "s")
    (ROOT / "results" / "phase2_timing.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

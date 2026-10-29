"""Phase 2: fixed-step RK4 against a reference solution (specification, Sec. 11).

The closed loop of the step convergence study (T1, verification controller, 1 kHz) is run
without sensor noise and quantisation, so that the comparison sees only the integrator.
Reference: implicit Radau IIA (scipy) with rtol 1e-9 and atol 1e-12, restarted at every
control sample with the held torque, analytic Jacobian of the compiled plant, quaternion
renormalised at the end of each interval. Scenarios: nominal, and cable 5 breaking at 4 s.
Duration 6 s. Output: results/phase2_reference_solution.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

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

T_END = 6.0
RK4_STEPS = [2.5e-4, 1.25e-4, 6.25e-5]


def setup(robot):
    cw = load_cable_winch()
    path = reference("T1")
    q0 = np.r_[path.sample(np.array([0.0]))[0][0], 0.0, 0.0, 0.0]
    plant, x0, _ = plant_at_rest(robot, cw, q0)
    return cw, path, plant, x0


def run(robot, events, dt_plant=None, radau=False):
    cw, path, plant, x0 = setup(robot)
    dt_ctrl = 1.0 / inner_rate()
    ctrl = CableSpacePD(robot, cw, plant.l_u0, path, dt_ctrl, T_END)
    sensors = WinchSensors(load_sensors(), robot.n, np.random.default_rng(0), noise=False)
    t0 = time.time()
    if radau:
        fast = CasadiPlant(plant, dt_ctrl, 1)

        def advance(x, tau, faults, h):
            sol = solve_ivp(lambda t, s: fast.rhs(s, tau, faults), (0.0, h), x, method="Radau",
                            rtol=1e-9, atol=1e-12, jac=lambda t, s: fast.jacobian(s, tau, faults))
            x_new = sol.y[:, -1].copy()
            x_new[3:7] /= np.linalg.norm(x_new[3:7])
            return x_new
        log = simulate(plant, x0, ctrl, T_END, dt_ctrl, dt_ctrl, sensors, events, advance=advance)
    else:
        fast = CasadiPlant(plant, dt_plant, int(round(dt_ctrl / dt_plant)))
        log = simulate(plant, x0, ctrl, T_END, dt_plant, dt_ctrl, sensors, events, fast=fast)
    log["seconds"] = time.time() - t0
    return log


def main():
    robot = load_robot()
    out = {"t_end_s": T_END, "rk4_steps_s": RK4_STEPS, "scenarios": {}, "provenance": stamp()}
    for name, events in (("nominal", []), ("break_lower", [FaultEvent(4.0, 4, "break")])):
        ref = run(robot, events, radau=True)
        entry = {"radau_seconds": round(ref["seconds"], 1), "rk4": {}}
        for dt in RK4_STEPS:
            log = run(robot, events, dt_plant=dt)
            entry["rk4"][f"{dt:g}"] = {
                "max_position_diff_m": float(np.abs(log["q"][:, :3] - ref["q"][:, :3]).max()),
                "max_orientation_diff_rad": float(np.abs(log["q"][:, 3:] - ref["q"][:, 3:]).max()),
                "max_tension_diff_N": float(np.abs(log["tension"] - ref["tension"]).max()),
                "rms_tension_diff_N": float(np.sqrt(np.mean((log["tension"] - ref["tension"]) ** 2))),
                "seconds": round(log["seconds"], 2)}
        out["scenarios"][name] = entry
        print(name, json.dumps(entry), flush=True)
    (ROOT / "results" / "phase2_reference_solution.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

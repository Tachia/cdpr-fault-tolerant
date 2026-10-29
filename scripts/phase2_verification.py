"""Phase 2: verification gates of the specification (Sec. 11) on the study parameters.

  energy        locked winches, undamped cables, no input, 10 s at the plant step: drift of
                kinetic + gravitational + elastic energy (gate: < 1e-6, relative to the
                initial kinetic energy)
  skew          dM/dt - 2C skew symmetric with dM/dt exact (gate: < 1e-10), 100 random states
  equilibrium   plant at rest at the home pose with the max-margin tensions: max |dx/dt|
  compiled      compiled plant against the NumPy plant: largest difference of dx/dt over 200
                random states with slack cables, broken cables, saturated drives and winches
                near the speed limit
Output: results/phase2_verification.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdpr_ft.model import lagrange_sym as lag  # noqa: E402
from cdpr_ft.model.kinematics import quat_from_euler  # noqa: E402
from cdpr_ft.model.plant import Faults  # noqa: E402
from cdpr_ft.model.plant_casadi import CasadiPlant  # noqa: E402
from cdpr_ft.params import inner_rate, load_cable_winch, load_robot, plant_step  # noqa: E402
from cdpr_ft.provenance import stamp  # noqa: E402
from cdpr_ft.sim.setup import plant_at_rest  # noqa: E402


def main():
    robot = load_robot()
    rng = np.random.default_rng(20261026)
    out = {"provenance": stamp()}
    dt, dt_ctrl = plant_step(), 1.0 / inner_rate()
    n_sub = int(round(dt_ctrl / dt))

    cw0 = load_cable_winch(damping_time=0.0)
    plant, x0, _ = plant_at_rest(robot, cw0, robot.home_pose)
    fast = CasadiPlant(plant, dt, n_sub, lock_winches=True)
    x = x0.copy()
    x[7:13] = [0.05, -0.04, 0.03, 0.05, -0.05, 0.04]
    e0 = plant.energy(x)
    ke0 = e0 - plant.energy(x0)
    drift, slack = 0.0, False
    for _ in range(int(round(10.0 / dt_ctrl))):
        x = fast.step_interval(x, np.zeros(robot.n))
        drift = max(drift, abs(plant.energy(x) - e0))
        slack |= bool(np.any(plant.cable_state(x)[1] <= 0))
    out["energy"] = {"duration_s": 10.0, "step_s": dt, "initial_kinetic_energy_J": ke0,
                     "max_drift_J": drift, "drift_over_KE0": drift / ke0, "any_slack": slack,
                     "pass": bool(drift / ke0 < 1e-6 and not slack)}

    par = lag.param_vector(robot.mass, robot.gravity, robot.com, robot.inertia_com)
    f_mdot = lag.mass_derivative_function()
    worst = 0.0
    for _ in range(100):
        q = robot.home_pose.copy()
        q[:3] += rng.uniform(-0.5, 0.5, 3)
        q[3:] = rng.uniform(-0.5, 0.5, 3)
        dq = rng.normal(size=6)
        _, c_l, _ = lag.lagrange_terms(q, dq, par)
        n_mat = np.asarray(f_mdot(q, dq, par), float) - 2 * c_l
        worst = max(worst, np.max(np.abs(n_mat + n_mat.T)))
    out["skew"] = {"max_abs_N_plus_NT": worst, "pass": bool(worst < 1e-10)}

    cw = load_cable_winch()
    plant, x0, t0 = plant_at_rest(robot, cw, robot.home_pose)
    tau = cw.transmission * t0
    out["equilibrium"] = {"tensions_N": np.round(t0, 3).tolist(),
                          "max_abs_xdot": float(np.max(np.abs(plant.rhs(x0, tau))))}

    cwf = load_cable_winch(coulomb=0.02, viscous=1e-4)
    plant, x0, _ = plant_at_rest(robot, cwf, robot.home_pose)
    fast = CasadiPlant(plant, dt, n_sub)
    worst_rel = 0.0
    n = robot.n
    for _ in range(200):
        q = robot.home_pose.copy()
        q[:3] += rng.uniform(-0.3, 0.3, 3)
        q[3:] = rng.uniform(-1.5, 1.5, 3)
        x = np.concatenate([q[:3], quat_from_euler(q[3:]), rng.normal(scale=0.5, size=6),
                            rng.normal(scale=20.0, size=n), rng.normal(scale=400.0, size=n)])
        tau = rng.normal(scale=3.0, size=n)
        faults = Faults(n, broken=rng.random(n) < 0.2, rho=rng.uniform(0, 0.6, n), bias=rng.normal(scale=0.05, size=n))
        ref, got = plant.rhs(x, tau, faults), fast.rhs(x, tau, faults)
        worst_rel = max(worst_rel, np.max(np.abs(got - ref) / (1.0 + np.abs(ref))))
    out["compiled_vs_numpy"] = {"max_rel_diff_xdot": worst_rel}
    print(json.dumps(out, indent=1))
    (ROOT / "results" / "phase2_verification.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

"""Phase 2: effect of the damping onset in the cable law on the peak tension after a break.

The stiff break scenario of the step convergence study (cable 5 breaks at 4 s, EA x 2,
retardation time 0.03 s, sensors on, T1, verification controller) at the plant step, once with
the study's damping onset of 0.1 mm and once with the plain Kelvin-Voigt law (onset 0).
Output: results/phase2_damping_onset_effect.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from phase2_step_convergence import SEED, T_END  # noqa: E402

from cdpr_ft.analysis.trajectories import reference  # noqa: E402
from cdpr_ft.control.verification import CableSpacePD  # noqa: E402
from cdpr_ft.model.plant_casadi import CasadiPlant  # noqa: E402
from cdpr_ft.params import inner_rate, load_cable_winch, load_robot, load_sensors, plant_step  # noqa: E402
from cdpr_ft.provenance import stamp  # noqa: E402
from cdpr_ft.sim.sensors import WinchSensors  # noqa: E402
from cdpr_ft.sim.setup import plant_at_rest  # noqa: E402
from cdpr_ft.sim.simulator import FaultEvent, simulate  # noqa: E402


def main():
    robot = load_robot()
    base = load_cable_winch()
    dt, dt_ctrl = plant_step(), 1.0 / inner_rate()
    path = reference("T1")
    q0 = np.r_[path.sample(np.array([0.0]))[0][0], 0.0, 0.0, 0.0]
    out = {"provenance": stamp(), "cases": {}}
    for name, onset in (("onset_0.1mm", base.damping_onset), ("plain_kelvin_voigt", 0.0)):
        cw = load_cable_winch(EA=2.0 * base.EA, damping_time=0.03, damping_onset=onset)
        plant, x0, _ = plant_at_rest(robot, cw, q0)
        log = simulate(plant, x0, CableSpacePD(robot, cw, plant.l_u0, path, dt_ctrl, T_END), T_END, dt, dt_ctrl,
                       WinchSensors(load_sensors(), robot.n, np.random.default_rng(SEED)),
                       [FaultEvent(4.0, 4, "break")], fast=CasadiPlant(plant, dt, int(round(dt_ctrl / dt))))
        out["cases"][name] = {"damping_onset_m": onset, "max_tension_N": float(log["tension"].max()),
                              "max_tension_before_break_N": float(log["tension"][log["t"] < 4.0].max())}
        print(name, out["cases"][name], flush=True)
    (ROOT / "results" / "phase2_damping_onset_effect.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

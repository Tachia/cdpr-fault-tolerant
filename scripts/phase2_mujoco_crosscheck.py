"""Phase 2: independent cross-check of the plant in MuJoCo (specification, Sec. 11).

The comparison isolates the rigid-body dynamics, the cable kinematics and the unilateral
elastic cables, which are implemented independently in the two codes. The winches are locked
(fixed unstretched lengths) and the cables undamped, so each cable is a linear spring of
stiffness k_i = EA / l_u,i that pulls only when stretched. In MuJoCo a spatial tendon from the
anchor site to the platform site with springlength = (0, l_u,i) gives exactly that: no force
while the tendon is shorter than l_u,i, a pull k_i (L - l_u,i) beyond.
MuJoCo: free joint, inertial frame at the centre of mass, RK4 with 0.1 ms. Plant: NumPy
reference implementation, RK4 with 0.1 ms, locked winches.
Cases (5 s each), starting from the static equilibrium at the home pose:
  taut   a small initial twist; every cable stays in tension
  slack  a large initial twist (0.8 m/s upward, 1 rad/s about x); cables go slack and
         tighten again
Pass condition: trajectory difference below 1 % of the motion over 5 s.
Output: results/phase2_mujoco_crosscheck.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdpr_ft.model.kinematics import euler_from_quat, quat_to_rot  # noqa: E402
from cdpr_ft.params import load_cable_winch, load_robot  # noqa: E402
from cdpr_ft.provenance import stamp  # noqa: E402
from cdpr_ft.sim.setup import plant_at_rest  # noqa: E402

DT = 1e-4
T_END = 5.0
CASES = {"taut": np.array([0.05, -0.04, 0.03, 0.05, -0.05, 0.04]),
         "slack": np.array([0.0, 0.0, 0.8, 1.0, 0.0, 0.0])}


def mjcf(robot, plant, q0_quat):
    f = lambda v: " ".join(f"{a:.17g}" for a in v)
    k = plant.cw.EA / plant.l_u0
    sites_w = "\n".join(f'    <site name="a{i}" pos="{f(a)}"/>' for i, a in enumerate(robot.anchors))
    sites_b = "\n".join(f'      <site name="b{i}" pos="{f(b)}"/>' for i, b in enumerate(robot.attachments))
    tendons = "\n".join(
        f'    <spatial name="c{i}" stiffness="{k[i]:.17g}" damping="0" springlength="0 {plant.l_u0[i]:.17g}">'
        f'<site site="a{i}"/><site site="b{i}"/></spatial>' for i in range(robot.n))
    ig = robot.inertia_com
    return f"""<mujoco model="ipanema2_crosscheck">
  <option timestep="{DT}" integrator="RK4" gravity="0 0 {-robot.gravity}">
    <flag contact="disable"/>
  </option>
  <worldbody>
{sites_w}
    <body name="platform" pos="{f(robot.home_pose[:3])}" quat="{f(q0_quat)}">
      <freejoint/>
      <inertial pos="{f(robot.com)}" mass="{robot.mass}" fullinertia="{ig[0,0]} {ig[1,1]} {ig[2,2]} {ig[0,1]} {ig[0,2]} {ig[1,2]}"/>
{sites_b}
    </body>
  </worldbody>
  <tendon>
{tendons}
  </tendon>
</mujoco>"""


def run_mujoco(robot, plant, x0, nu0):
    model = mujoco.MjModel.from_xml_string(mjcf(robot, plant, x0[3:7]))
    data = mujoco.MjData(model)
    data.qpos[:3] = x0[:3]
    data.qpos[3:7] = x0[3:7]
    rot = quat_to_rot(x0[3:7])
    data.qvel[:3] = nu0[:3]                      # free joint: linear velocity of the body origin, world frame
    data.qvel[3:6] = rot.T @ nu0[3:]             # angular velocity in the body frame
    mujoco.mj_forward(model, data)
    out = []
    for _ in range(int(round(T_END / DT)) + 1):
        out.append(np.r_[data.qpos[:3], euler_from_quat(data.qpos[3:7]),
                         np.asarray(data.ten_length).copy()])
        mujoco.mj_step(model, data)
    return np.array(out)


def run_plant(plant, x0, nu0):
    x = x0.copy()
    x[7:13] = nu0
    tau = np.zeros(plant.n)
    out = []
    for _ in range(int(round(T_END / DT)) + 1):
        out.append(np.r_[plant.pose_euler(x), plant.cable_state(x)[2]])
        x = plant.step(x, tau, DT, lock_winches=True)
    return np.array(out)


def main():
    robot = load_robot()
    cw = load_cable_winch(damping_time=0.0)
    plant, x0, t0 = plant_at_rest(robot, cw, robot.home_pose)
    out = {"dt_s": DT, "t_end_s": T_END, "EA_N": cw.EA, "cases": {}, "provenance": stamp(),
           "mujoco": mujoco.__version__}
    for name, nu0 in CASES.items():
        t_start = time.time()
        mj = run_mujoco(robot, plant, x0, nu0)
        t_mj = time.time() - t_start
        t_start = time.time()
        pl = run_plant(plant, x0, nu0)
        t_pl = time.time() - t_start
        motion_p = np.ptp(pl[:, :3], axis=0).max()
        motion_r = np.ptp(pl[:, 3:6], axis=0).max()
        slack_samples = int(np.sum(pl[:, 6:] < plant.l_u0[None, :]))
        d_p = np.abs(mj[:, :3] - pl[:, :3]).max()
        d_r = np.abs(mj[:, 3:6] - pl[:, 3:6]).max()
        d_l = np.abs(mj[:, 6:] - pl[:, 6:]).max()
        entry = {"position_motion_m": float(motion_p), "rotation_motion_rad": float(motion_r),
                 "max_position_diff_m": float(d_p), "max_orientation_diff_rad": float(d_r),
                 "max_cable_length_diff_m": float(d_l),
                 "position_diff_rel": float(d_p / motion_p), "orientation_diff_rel": float(d_r / motion_r),
                 "slack_cable_samples": slack_samples,
                 "pass_1pct": bool(d_p / motion_p < 0.01 and d_r / motion_r < 0.01),
                 "seconds_mujoco": round(t_mj, 1), "seconds_plant": round(t_pl, 1)}
        out["cases"][name] = entry
        print(name, json.dumps(entry), flush=True)
    (ROOT / "results" / "phase2_mujoco_crosscheck.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

"""Phase 3 components: winch observer, tension distribution, compiled functions, the IMM, the
post-break planner and its recovery problem."""
import casadi as ca
import numpy as np
import pytest

from cdpr_ft.analysis.feasibility import max_margin_tensions
from cdpr_ft.control.planner import Planner, PlannerConfig, quintic_coeffs, quintic_eval
from cdpr_ft.control.tda import distribute
from cdpr_ft.control.winch import WinchESO, eso_matrices
from cdpr_ft.model.compiled import compiled
from cdpr_ft.model.kinematics import wrench_matrix
from cdpr_ft.model.rigid_body import gravity_wrench
from cdpr_ft.model.rigid_casadi import RigidModel
from cdpr_ft.params import load_cable_winch


def test_eso_error_poles_are_placed():
    dt, omega = 1e-3, 2 * np.pi * 50.0
    phi, _, gain = eso_matrices(omega, 1.0 / 3e-5, dt)
    c = np.array([1.0, 0.0, 0.0])
    eig = np.linalg.eigvals((np.eye(3) - np.outer(gain, c)) @ phi)
    assert np.allclose(eig, np.exp(-omega * dt), atol=2e-4)


def test_eso_recovers_a_constant_tension():
    cw = load_cable_winch()
    dt, inertia, r_d = 1e-3, cw.inertia, cw.transmission
    eso = WinchESO(1, 2 * np.pi * 50.0, inertia, r_d, dt)
    eso.reset(np.zeros(1), np.array([40.0]))
    tension = 80.0
    theta, omega = 0.0, 0.0
    for k in range(400):
        tau = r_d * tension + 0.02 * np.sin(2 * np.pi * 3 * k * dt)     # motion of the drum
        acc = (tau - r_d * tension) / inertia
        theta += omega * dt + 0.5 * acc * dt ** 2
        omega += acc * dt
        eso.update(np.array([theta]), np.array([tau]))
    assert abs(eso.tension()[0] - tension) < 0.5


def test_distribute_meets_the_wrench_with_the_largest_margin(robot):
    q = robot.home_pose.copy()
    w = wrench_matrix(q, robot.anchors, robot.attachments)
    wrench = -gravity_wrench(q, robot.mass, robot.com, robot.gravity)
    lo, hi = np.full(8, robot.t_min), np.full(8, robot.t_max)
    t, margin, ok = distribute(w, wrench, lo, hi)
    assert ok and margin > 0
    assert np.allclose(w @ t, wrench, atol=1e-6)
    assert np.all(t >= lo + margin - 1e-6) and np.all(t <= hi - margin + 1e-6)
    s, _ = max_margin_tensions(w, -wrench, lo, hi)
    assert np.isclose(margin, s)
    # an upper cable lost at the home pose: no tensions in the bounds, bounded least squares
    t7, _, ok7 = distribute(w[:, 1:], wrench, lo[1:], hi[1:])
    assert not ok7 and np.all(t7 >= lo[1:] - 1e-9) and np.all(t7 <= hi[1:] + 1e-9)


def test_compiled_function_matches_casadi(robot, rng):
    model = RigidModel(robot)
    fun = model.step_jacobians
    cf = compiled(fun)
    for _ in range(3):
        x = np.r_[robot.home_pose + rng.normal(scale=0.05, size=6), rng.normal(scale=0.2, size=6)]
        t = rng.uniform(20, 150, robot.n)
        ref = [np.asarray(o) for o in fun(x, t, 0.005)]
        out = cf(x, t, 0.005)
        for a, b in zip(ref, out):
            assert np.allclose(a, b, rtol=1e-12, atol=1e-12)


def test_quintic_meets_its_boundary_conditions(rng):
    x0, v0, a0, x1 = rng.normal(size=(4, 6))
    c = quintic_coeffs(x0, v0, a0, x1, 1.3)
    q, dq, ddq = quintic_eval(c, 1.3, np.array([0.0, 1.3 - 1e-12]))
    assert np.allclose(q[0], x0) and np.allclose(dq[0], v0) and np.allclose(ddq[0], a0)
    assert np.allclose(q[1], x1, atol=1e-9) and np.allclose(dq[1], 0, atol=1e-8) and np.allclose(ddq[1], 0, atol=1e-7)


@pytest.fixture(scope="module")
def planner(robot):
    from pathlib import Path
    f = Path(__file__).resolve().parents[1] / "results" / "phase3_safe_poses.npz"
    if not f.exists():
        pytest.skip("safe-pose sets not computed (scripts/phase3_safe_poses.py)")
    d = np.load(f)
    safe = {i: (d[f"pos{i}"], d[f"ori{i}"]) for i in range(robot.n)}
    return Planner(robot, safe, PlannerConfig(), EA=load_cable_winch().EA)


def test_recovery_after_an_upper_cable_break(robot, planner):
    """Break of cable 1 on the circle T1: the plan ends at rest at a pose the seven remaining
    cables hold with margin s_min, its tensions stay in the bounds, and the interpolated path
    passes through the solution nodes."""
    from cdpr_ft.analysis.trajectories import reference
    from cdpr_ft.sim.episode import reference_fn
    ref = reference_fn(reference("T1", home=tuple(robot.home_pose[:3])))
    q, nu, _ = ref(np.array([3.0]))
    plan = planner.after_break(0, 3.0, q[0], nu[0], ref)
    cfg = planner.recovery.cfg
    assert plan.branch == 2
    end = plan.sample(np.array([3.0 + plan.duration + 0.5]))
    assert np.allclose(end[1], 0.0, atol=1e-9)
    q_end = end[0][0]
    t_hold = plan.tensions(np.array([3.0 + plan.duration + 0.5]))[:, 0]
    w = wrench_matrix(q_end, robot.anchors, robot.attachments)
    residual = w @ t_hold + gravity_wrench(q_end, robot.mass, robot.com, robot.gravity)
    assert np.linalg.norm(residual) < 1e-4
    assert t_hold[0] == 0.0
    assert np.all(t_hold[1:] >= robot.t_min + cfg.s_min - 1e-4) and np.all(t_hold[1:] <= robot.t_max - cfg.s_min + 1e-4)
    ts = 3.0 + np.linspace(0, plan.duration, 7)[:-1] + 1e-9
    u = plan.tensions(ts)
    assert np.all(u[0] == 0.0)
    assert np.all(u[1:] >= robot.t_min + cfg.s_path - 1e-4) and np.all(u[1:] <= robot.t_max - cfg.s_path + 1e-4)
    start = plan.sample(np.array([3.0]))
    assert np.allclose(start[0][0], q[0], atol=1e-8) and np.allclose(start[1][0], nu[0], atol=1e-8)
    assert np.rad2deg(np.abs(plan.sample(np.linspace(3.0, 3.0 + plan.duration, 200))[0][:, 3:]).max()) \
        <= cfg.psi_path_deg + 0.5


def test_imm_probabilities_and_mode_table(robot):
    from cdpr_ft.estimation.imm import IMM, IMMConfig, mode_names, mode_table
    b, a = mode_table(robot.n)
    assert b.shape == (17, 8) and b.sum() == 8 and a.sum() == 8 and not np.any(b[0]) and not np.any(a[0])
    assert mode_names(robot.n)[0] == "M0" and len(set(mode_names(robot.n))) == 17
    cw = load_cable_winch()
    imm = IMM(robot, cw.EA, IMMConfig(dt=0.005, sigma_acc_lin=0.5, sigma_acc_ang=0.5, sigma_pose=1e-6,
                                      sigma_deficit=20.0, r_length=2e-4, r_tension=1.5, r_length_broken=1.0,
                                      fault_rate=1e-3, return_rate=1e-2))
    q0 = robot.home_pose.copy()
    imm.reset(q0)
    w = wrench_matrix(q0, robot.anchors, robot.attachments)
    t0, _, _ = distribute(w, -gravity_wrench(q0, robot.mass, robot.com, robot.gravity),
                          np.full(8, robot.t_min), np.full(8, robot.t_max))
    from cdpr_ft.model.kinematics import cable_geometry
    length = cable_geometry(q0, robot.anchors, robot.attachments)[1]
    l_u = length * cw.EA / (cw.EA + t0)
    imm.home(l_u, t0)
    for _ in range(100):
        mu = imm.step(l_u, t0, t0)
        assert np.isclose(mu.sum(), 1.0) and np.all(mu >= 0)
    assert mu[0] > 0.99
    assert np.allclose(imm.fused_state()[:6], q0, atol=1e-4)


def test_time_scaled_reference_is_consistent(robot):
    """The slowed-down reference: twist = time derivative of the pose, twist rate = time
    derivative of the twist, velocity continuous at the start and kappa times the nominal one
    after the ramp."""
    from cdpr_ft.analysis.trajectories import reference
    from cdpr_ft.control.planner import time_scaled
    from cdpr_ft.sim.episode import reference_fn
    ref = reference_fn(reference("T1", home=tuple(robot.home_pose[:3])))
    fn = time_scaled(ref, 4.0, 0.4, 0.5)
    t = np.linspace(3.9, 5.2, 400)
    h = 1e-6
    q, nu, dnu = fn(t)
    qp, nup, _ = fn(t + h)
    qm, num, _ = fn(t - h)
    assert np.allclose((qp - qm)[:, :3] / (2 * h), nu[:, :3], atol=1e-6)
    assert np.allclose((nup - num) / (2 * h), dnu, atol=1e-5)
    q0, nu0, _ = ref(np.array([4.0]))
    assert np.allclose(fn(np.array([4.0]))[1], nu0)
    late = 5.0
    tau = 4.0 + 0.5 - 0.6 * 0.5 * 0.5 + 0.4 * (late - 4.5)   # tau = t0 + T_r(1 - (1-k)/2) + k (t - t0 - T_r)
    assert np.allclose(fn(np.array([late]))[1], 0.4 * ref(np.array([tau]))[1], atol=1e-9)


def test_mpc_holds_the_start_pose(robot):
    """One MPC solve at rest at the start of T1 with the equilibrium tensions as feedforward
    returns tensions within the bounds and close to the feedforward."""
    from cdpr_ft.control.mpc import MPCConfig, ScenarioMPC
    from cdpr_ft.analysis.feasibility import max_margin_tensions
    mpc = ScenarioMPC(robot, MPCConfig())
    q0 = robot.home_pose.copy()
    w = wrench_matrix(q0, robot.anchors, robot.attachments)
    _, t0 = max_margin_tensions(w, gravity_wrench(q0, robot.mass, robot.com, robot.gravity),
                                np.full(8, robot.t_min), np.full(8, robot.t_max))
    N = mpc.cfg.horizon
    x0 = np.r_[q0, np.zeros(6)]
    xref = np.repeat(x0[:, None], N + 1, 1)
    tff = np.repeat(t0[:, None], N, 1)
    slot = dict(weight=1.0, x0=x0, b=np.zeros(8), a=np.zeros(8), delta=0.0, w_b=np.zeros(6), qstar=None)
    tseq, xpred, status, ok = mpc.solve([slot], xref, tff, t0, np.full(8, robot.t_min), np.full(8, robot.t_max))
    assert ok
    assert np.all(tseq >= robot.t_min - 1e-6) and np.all(tseq <= robot.t_max + 1e-6)
    assert np.abs(tseq[:, 0] - t0).max() < 1.0
    assert np.abs(xpred[0][:3] - q0[:3, None]).max() < 1e-4


def test_replay_reproduces_the_diagnosis_of_a_run(robot):
    """The offline replay of the recorded readings and commands gives the mode probabilities the
    controller computed in the run (the replay tuning of Phase 3 relies on this)."""
    from cdpr_ft.control.ftc import FTCConfig
    from cdpr_ft.control.mpc import MPCConfig, ScenarioMPC
    from cdpr_ft.estimation.diagnosis import replay
    from cdpr_ft.estimation.imm import IMM, IMMConfig
    from cdpr_ft.sim.episode import EpisodeSpec, controller_start, run_episode
    cw = load_cable_winch()
    imm = IMM(robot, cw.EA, IMMConfig(t_arm=0.3))
    spec = EpisodeSpec(t_end=0.8, seed=3, randomise=True)
    planner = Planner(robot, {}, PlannerConfig(), EA=cw.EA)
    cfg = FTCConfig(mu_declare=2.0, record_inputs=True)
    res = run_episode(spec, imm, ScenarioMPC(robot, MPCConfig()), planner, {}, cfg, keep_log=True)
    cl = res["ctrl_log"]
    _, _, q0, t_nom, l_u0 = controller_start(spec, robot, cw)
    t_imm, mu = replay({"theta": cl["theta"], "tau": cl["tau"], "td_in": cl["td_in"]}, cw, l_u0, imm,
                       cfg.eso_bandwidth_hz, cfg.imm_every, 1e-3, t_nom, q0, cfg.imm_input)
    assert len(t_imm) == len(cl["t_imm"]) and np.allclose(t_imm, cl["t_imm"])
    assert np.allclose(mu, cl["mu"], atol=1e-12)

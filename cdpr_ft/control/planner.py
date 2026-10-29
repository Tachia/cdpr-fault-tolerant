"""Post-fault planner (methods, Sec. 8).

Once a fault is declared, the planner decides what the platform can still do.

Actuator fault A_i. The tension loop of winch i is corrected with the deficit estimated by the
IMM, so the cable keeps delivering the commanded tension as long as the drive has torque in
reserve (with a loss of effectiveness of 60 %, the MSK030C still delivers 384 N through the
gearbox, almost twice t_max). The reference is kept (branch 1 with kappa = 1).

Cable break B_i. Three branches, tried in order:
 1. Time scaling. If the remaining reference over the next T_h seconds, slowed down to a
    factor kappa in [kappa_min, 1], can be followed with the remaining cables (dynamic margin of
    methods Eq. margin, with the inertial wrench of the scaled motion, at least s_dyn at every
    sample), the task continues with the largest such kappa (bisection). The scaling changes
    from 1 to kappa along a quintic ramp of T_r seconds, so the reference velocity stays
    continuous and the deceleration enters the check. Below kappa_min the task is not
    continued in any useful sense, and the planner goes to branch 2.
 2. Recovery to a safe pose. The planner solves the optimal control problem of
    control/recovery.py from the estimated state: the platform is brought to rest, in the
    shortest time compatible with a direct path, at a pose the remaining cables can hold with
    margin s_min, with tensions within the bounds (margin s_path) along the way. The nearest
    pose of the precomputed safe set, joined to the current state by a quintic polynomial,
    seeds the solver. (A quintic polynomial alone, the first version of this branch, was never
    dynamically feasible after the break of an upper cable; see recovery.py.)
 3. Best effort descent. If the problem has no solution within the tension bounds, the
    solution of its relaxed form (slack variables on the bounds, penalised) is used, which is
    the descent with the least violation of the tension bounds, and the run is recorded as such.
The output is a reference q(t), nu(t), dnu(t) for the MPC and, for branches 2 and 3, the
tensions of the solution, which the MPC takes as feedforward. Between the nodes of the solution
the state is interpolated with the cubic Hermite polynomial that the Hermite-Simpson
collocation implies (node values and model derivatives at both ends of the interval).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..analysis.feasibility import max_margin_tensions
from .recovery import RecoveryConfig, RecoveryOCP
from .tda import distribute
from ..model.kinematics import euler_rate_matrix, wrench_matrix
from ..model.rigid_body import bias_twist, gravity_wrench, mass_matrix_twist


@dataclass(frozen=True)
class PlannerConfig:
    s_dyn: float = 0.0          # [N] dynamic margin required along a recovery path
    horizon: float = 2.0        # [s] T_h for the time-scaling test
    w_rot: float = 1.0          # [m/rad] weight of orientation in the distance to the safe poses
    check_dt: float = 0.05      # [s] sampling of the feasibility checks
    kappa_min: float = 0.25     # smallest time scaling that still counts as continuing the task
    ramp: float = 0.5           # [s] T_r, the time scaling changes from 1 to kappa over this time


def quintic_coeffs(x0, v0, a0, x1, duration):
    """Coefficients (6, d) of x(s) = sum c_k s^k on s in [0, T], from (x0, v0, a0) to (x1, 0, 0)."""
    t = duration
    c0, c1, c2 = x0, v0, 0.5 * a0
    m = np.array([[t ** 3, t ** 4, t ** 5], [3 * t ** 2, 4 * t ** 3, 5 * t ** 4], [6 * t, 12 * t ** 2, 20 * t ** 3]])
    rhs = np.vstack([x1 - (c0 + c1 * t + c2 * t ** 2), -(c1 + 2 * c2 * t), -(2 * c2) * np.ones_like(x0)])
    c345 = np.linalg.solve(m, rhs)
    return np.vstack([c0, c1, c2, c345])


def quintic_eval(coeffs, duration, t):
    s = np.clip(np.asarray(t, float), 0.0, duration)[:, None]
    pw = np.hstack([s ** k for k in range(6)])
    dpw = np.hstack([np.zeros_like(s)] + [k * s ** (k - 1) for k in range(1, 6)])
    ddpw = np.hstack([np.zeros_like(s), np.zeros_like(s)] + [k * (k - 1) * s ** (k - 2) for k in range(2, 6)])
    after = (np.asarray(t, float) > duration)[:, None]
    pos = pw @ coeffs
    vel = np.where(after, 0.0, dpw @ coeffs)
    acc = np.where(after, 0.0, ddpw @ coeffs)
    return pos, vel, acc


class Plan:
    """A reference q(t), with twist and twist rate, defined from the time it was made."""

    def __init__(self, kind, t0, q_fn, branch, target=None, kappa=1.0, duration=None, min_margin=None,
                 tension_fn=None, info=None):
        self.kind, self.t0, self._q_fn, self.branch = kind, t0, q_fn, branch
        self.target, self.kappa, self.duration, self.min_margin = target, kappa, duration, min_margin
        self.tension_fn = tension_fn
        self.info = info or {}

    def tensions(self, t):
        """Feedforward tensions (n, len(t)) at absolute times t, or None."""
        return None if self.tension_fn is None else self.tension_fn(np.atleast_1d(np.asarray(t, float)))

    def sample(self, t):
        """q (len(t), 6), nu (len(t), 6), dnu (len(t), 6) at absolute times t."""
        return self._q_fn(np.atleast_1d(np.asarray(t, float)))


def time_scaled(reference, t0, kappa, ramp):
    """Reference slowed down from t0 on: tau(t) = t0 + int_t0^t k(s) ds with the scaling k(t)
    going from 1 to kappa over `ramp` seconds along a quintic smoothstep, so that the reference
    velocity stays continuous; q(t) = q_ref(tau), nu = k nu_ref(tau), dnu = k^2 dnu_ref(tau) + dk/dt nu_ref(tau)."""
    def fn(t):
        s = np.asarray(t, float) - t0
        x = np.clip(s / ramp, 0.0, 1.0)
        sig = 10 * x ** 3 - 15 * x ** 4 + 6 * x ** 5
        dsig = (30 * x ** 2 - 60 * x ** 3 + 30 * x ** 4) / ramp * ((s > 0) & (s < ramp))
        integral = ramp * (2.5 * x ** 4 - 3 * x ** 5 + x ** 6) + np.maximum(s - ramp, 0.0)
        k = 1.0 - (1.0 - kappa) * sig
        dk = -(1.0 - kappa) * dsig
        tau = t0 + np.maximum(s, 0.0) - (1.0 - kappa) * integral
        tau = np.where(s < 0, t0 + s, tau)
        k = np.where(s < 0, 1.0, k)
        q, nu, dnu = reference(tau)
        return q, k[:, None] * nu, (k ** 2)[:, None] * dnu + dk[:, None] * nu
    return fn


def euler_to_twist(q, dq, ddq):
    """nu = H(q) dq/dt and its rate for ZYX angles (dnu = H ddq + dH/dt dq)."""
    nu = np.zeros_like(dq)
    dnu = np.zeros_like(ddq)
    nu[:, :3], dnu[:, :3] = dq[:, :3], ddq[:, :3]
    eps = 1e-6
    for k in range(len(q)):
        e = euler_rate_matrix(q[k, 3:])
        nu[k, 3:] = e @ dq[k, 3:]
        de = (euler_rate_matrix(q[k, 3:] + eps * dq[k, 3:]) - euler_rate_matrix(q[k, 3:] - eps * dq[k, 3:])) / (2 * eps)
        dnu[k, 3:] = e @ ddq[k, 3:] + de @ dq[k, 3:]
    return nu, dnu


class RecoveryLibrary:
    """Plans computed offline (scripts/phase3_recovery_library.py) for every cable at break
    instants on a grid along each task. The reference paths are known before the task runs,
    so the planning time is spent before the robot moves, and a declared break is answered at
    once instead of after the solve time of the recovery problem."""

    def __init__(self, path):
        d = np.load(path, allow_pickle=False)
        self.task = d["task"]
        self.cable, self.onset, self.branch = d["cable"], d["onset"], d["branch"]
        self.kappa, self.T, self.X, self.U, self.t_hold = d["kappa"], d["T"], d["X"], d["U"], d["t_hold"]
        self.step = float(d["grid_step"])

    def lookup(self, task, cable, t_break):
        sel = np.nonzero((self.task == task) & (self.cable == cable))[0]
        if len(sel) == 0:
            return None
        k = sel[np.argmin(np.abs(self.onset[sel] - t_break))]
        if abs(self.onset[k] - t_break) > self.step:
            return None          # outside the range the library covers
        return {"onset": float(self.onset[k]), "branch": int(self.branch[k]), "kappa": float(self.kappa[k]),
                "T": float(self.T[k]), "X": self.X[k], "U": self.U[k], "t_hold": self.t_hold[k]}


class Planner:
    def __init__(self, robot, safe_sets: dict, cfg: PlannerConfig = PlannerConfig(), recovery: RecoveryOCP | None = None,
                 EA: float | None = None, recovery_cfg: RecoveryConfig = RecoveryConfig(), library=None):
        """library: optional RecoveryLibrary of plans computed offline along the task."""
        self.robot, self.safe, self.cfg = robot, safe_sets, cfg
        self.recovery = recovery if recovery is not None else RecoveryOCP(robot, EA, recovery_cfg)
        self.library = library

    def from_library(self, task, cable, t_now, reference, detection_delay):
        """Plan of the library for `task` and `cable` whose break instant is nearest to
        t_now - detection_delay, or None. The plan keeps its own time origin (the break
        instant it was computed for), so the part already elapsed is skipped."""
        if self.library is None:
            return None
        entry = self.library.lookup(task, cable, t_now - detection_delay)
        if entry is None:
            return None
        t_b = entry["onset"]
        info = {"source": "library", "t_break": t_b}
        if entry["branch"] == 1:
            return Plan("break", t_b, time_scaled(reference, t_b, entry["kappa"], self.cfg.ramp), branch=1,
                        kappa=entry["kappa"], info=info)
        res = {"X": entry["X"], "U": entry["U"], "t_hold": entry["t_hold"], "T": entry["T"]}
        q_fn, t_fn = self._interpolant(res, t_b, cable, np.zeros(6))
        return Plan("break", t_b, q_fn, branch=int(entry["branch"]), target=entry["X"][:6, -1].copy(),
                    duration=entry["T"], tension_fn=t_fn, info=info)

    # ------------------------------------------------------------------ checks
    def dynamic_margins(self, q, nu, dnu, active, stop_below=None):
        """Largest margin of the tensions that produce M dnu + b - w_g with the active cables.
        With stop_below, the evaluation stops at the first sample below it (the remaining
        entries are left at +inf), which is all a feasibility test needs."""
        r = self.robot
        out = np.full(len(q), np.inf)
        lo = np.full(active.sum(), r.t_min)
        hi = np.full(active.sum(), r.t_max)
        for k in range(len(q)):
            m_nu = mass_matrix_twist(q[k], r.mass, r.com, r.inertia_com)
            wrench = m_nu @ dnu[k] + bias_twist(q[k], nu[k], r.mass, r.com, r.inertia_com) \
                - gravity_wrench(q[k], r.mass, r.com, r.gravity)
            s, _ = max_margin_tensions(wrench_matrix(q[k], r.anchors, r.attachments)[:, active], -wrench, lo, hi)
            out[k] = -np.inf if s is None else s
            if stop_below is not None and out[k] < stop_below:
                break
        return out

    # ------------------------------------------------------------------ plans
    def keep_reference(self, reference, t_now, kind):
        return Plan(kind, t_now, reference, branch=1, kappa=1.0)

    def after_break(self, cable, t_now, q_hat, nu_hat, reference, w_b=None):
        """Plan after the declared break of `cable`, starting at t_now from the estimated pose
        q_hat and twist nu_hat (wrench bias w_b of the estimator, zero if not given).
        reference: callable t -> (q, nu, dnu) of the nominal task."""
        r, cfg = self.robot, self.cfg
        active = np.ones(r.n, bool)
        active[cable] = False
        w_b = np.zeros(6) if w_b is None else np.asarray(w_b, float)
        # branch 1: time scaling of the remaining reference
        ts = t_now + np.arange(0.0, cfg.horizon + 1e-9, cfg.check_dt)

        def scaled(kappa):
            return time_scaled(reference, t_now, kappa, cfg.ramp)

        def ok(kappa):
            q, nu, dnu = scaled(kappa)(ts)
            return self.dynamic_margins(q, nu, dnu, active, stop_below=cfg.s_dyn - 1e-6).min() >= cfg.s_dyn - 1e-6
        if ok(cfg.kappa_min):
            lo_k, hi_k = cfg.kappa_min, 1.0
            if ok(1.0):
                lo_k = 1.0
            else:
                for _ in range(8):
                    mid = 0.5 * (lo_k + hi_k)
                    lo_k, hi_k = (mid, hi_k) if ok(mid) else (lo_k, mid)
            return Plan("break", t_now, scaled(lo_k), branch=1, kappa=lo_k, info={"source": "online"})
        # branches 2 and 3: recovery problem, seeded with the nearest precomputed safe pose
        pos, ori = self.safe[cable]
        d = np.linalg.norm(pos - q_hat[:3], axis=1) + cfg.w_rot * np.linalg.norm(ori - q_hat[3:], axis=1)
        k = int(np.argmin(d))
        q_seed = np.r_[pos[k], ori[k]]
        res = self.recovery.solve(cable, np.r_[q_hat, nu_hat], w_b, q_seed,
                                  self._seed_guess(q_hat, nu_hat, q_seed, active))
        branch = 2 if res["max_slack"] <= self.recovery.cfg.tol_slack else 3
        q_fn, t_fn = self._interpolant(res, t_now, cable, w_b)
        return Plan("break", t_now, q_fn, branch=branch, target=res["X"][:6, -1].copy(), duration=res["T"],
                    min_margin=None, tension_fn=t_fn,
                    info={"source": "online", "solver_status": res["status"], "solver_seconds": res["seconds"],
                          "iterations": res["iterations"], "max_slack": res["max_slack"], "seed": q_seed,
                          "solution": {k: res[k] for k in ("X", "U", "t_hold", "T")}})

    def seed_path(self, q0, nu0, q_seed, t0, duration):
        """Quintic from the pose q0 and twist nu0 at time t0 to rest at q_seed after `duration`
        seconds, as a reference callable t -> (q, nu, dnu)."""
        dq0 = np.r_[nu0[:3], np.linalg.solve(euler_rate_matrix(q0[3:]), nu0[3:])]
        coeffs = quintic_coeffs(q0, dq0, np.zeros(6), q_seed, duration)

        def fn(t):
            q, dq, ddq = quintic_eval(coeffs, duration, np.atleast_1d(np.asarray(t, float)) - t0)
            nu, dnu = euler_to_twist(q, dq, ddq)
            return q, nu, dnu
        return fn

    def _seed_guess(self, q0, nu0, q_seed, active):
        """Initial guess of the recovery problem: quintic from the current state to the seed
        pose, tensions of the tension distribution along it (clipped to the bounds)."""
        r, n_steps = self.robot, self.recovery.cfg.steps
        lo, hi = np.full(active.sum(), r.t_min), np.full(active.sum(), r.t_max)
        dq0 = np.r_[nu0[:3], np.linalg.solve(euler_rate_matrix(q0[3:]), nu0[3:])]

        def guess(duration):
            coeffs = quintic_coeffs(q0, dq0, np.zeros(6), q_seed, duration)
            q, dq, ddq = quintic_eval(coeffs, duration, np.linspace(0.0, duration, n_steps + 1))
            nu, dnu = euler_to_twist(q, dq, ddq)
            u = np.zeros((r.n, n_steps))
            for k in range(n_steps):
                m_nu = mass_matrix_twist(q[k], r.mass, r.com, r.inertia_com)
                wrench = m_nu @ dnu[k] + bias_twist(q[k], nu[k], r.mass, r.com, r.inertia_com) \
                    - gravity_wrench(q[k], r.mass, r.com, r.gravity)
                u[active, k], _, _ = distribute(wrench_matrix(q[k], r.anchors, r.attachments)[:, active], wrench, lo, hi)
            hold = np.zeros(r.n)
            hold[active], _, _ = distribute(wrench_matrix(q_seed, r.anchors, r.attachments)[:, active],
                                            -gravity_wrench(q_seed, r.mass, r.com, r.gravity), lo, hi)
            return np.vstack([q.T, nu.T]), u, hold
        return guess

    def _interpolant(self, res, t0, cable, w_b):
        """Reference and feedforward tensions of a recovery solution. On interval k the state is
        the cubic Hermite polynomial through the nodes x_k, x_{k+1} with the model derivatives
        f(x_k, t_k) and f(x_{k+1}, t_k); after the end, rest at the final pose with the holding
        tensions."""
        X, U, T = res["X"], res["U"], res["T"]
        n_steps = U.shape[1]
        h = T / n_steps
        f = self.recovery.model.f_ext
        F0 = np.column_stack([np.asarray(f(X[:, k], U[:, k], w_b)).ravel() for k in range(n_steps)])
        F1 = np.column_stack([np.asarray(f(X[:, k + 1], U[:, k], w_b)).ravel() for k in range(n_steps)])
        hold = res["t_hold"].copy()

        def q_fn(t):
            s = np.asarray(t, float) - t0
            k = np.clip(np.floor(s / h).astype(int), 0, n_steps - 1)
            tau = np.clip((s - k * h) / h, 0.0, 1.0)[:, None]
            x0, x1, f0, f1 = X[:, k].T, X[:, k + 1].T, F0[:, k].T, F1[:, k].T
            h00, h10 = 2 * tau ** 3 - 3 * tau ** 2 + 1, tau ** 3 - 2 * tau ** 2 + tau
            h01, h11 = -2 * tau ** 3 + 3 * tau ** 2, tau ** 3 - tau ** 2
            d00, d10 = 6 * tau ** 2 - 6 * tau, 3 * tau ** 2 - 4 * tau + 1
            d01, d11 = -6 * tau ** 2 + 6 * tau, 3 * tau ** 2 - 2 * tau
            x = h00 * x0 + h10 * h * f0 + h01 * x1 + h11 * h * f1
            dx = (d00 * x0 + d01 * x1) / h + d10 * f0 + d11 * f1
            after = (s >= T)[:, None]
            x = np.where(after, X[:, -1][None, :], x)
            dx = np.where(after, 0.0, dx)
            before = (s < 0)[:, None]
            x = np.where(before, X[:, 0][None, :], x)
            return x[:, :6], x[:, 6:], dx[:, 6:]

        def t_fn(t):
            s = np.asarray(t, float) - t0
            k = np.clip(np.floor(s / h).astype(int), 0, n_steps - 1)
            out = U[:, k].copy()
            out[:, s >= T] = hold[:, None]
            out[cable] = 0.0
            return out
        return q_fn, t_fn

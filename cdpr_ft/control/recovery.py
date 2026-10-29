"""Recovery trajectory after a cable break: an optimal control problem on the rigid model.

The first version of branch 2 of the planner joined the estimated state to a safe pose with a
quintic polynomial. After the break of an upper cable at the height of the reference paths
(z = 3 m) that never worked: the platform has to drop by up to half a metre, the seven
remaining cables can only produce accelerations inside a narrow, pose dependent set, and no
polynomial shape stays inside it (the smallest margin along the best quintic was between
-31 and -78 N in the cases checked in Phase 3). The path is therefore the solution of an optimal
control problem on the rigid model of the controller, with the remaining cables, over a fixed
duration T split into N intervals of length h (separated Hermite-Simpson collocation, tensions
constant over each interval):

    min   ||q_N - q_hat||^2_Wq + w_v h sum_k |nu_k|^2_Wn + w_a h sum_k |(nu_{k+1} - nu_k)/h|^2_Wn
          + w_r sum_k |t_{k+1} - t_k|^2 + w_m sum_k |r_k|^2 + w_s sum_k s_k
    s.t.  x_0 = x_hat, collocation of x' = f(x, t, w_b) on every interval
          t_lo + s_path - s_k <= t_k,i <= t_hi - s_path + s_k,  s_k >= 0   (intact cables)
          t_lo + s_soft - r_k,i <= t_k,i <= t_hi - s_soft + r_k,i,  r_k,i >= 0
          t_k,i = 0                                                          (broken cable)
          |v_k| <= v_max,  |v_{k+1} - v_k| <= a_max h,  |angles| <= psi_path at nodes and midpoints
          l_u,i(x_k, t_k) inside the drum window of cable i, with clearance c_w
          nu_N = 0,  f(x_N, t_hold, w_b) = 0 with t_hold within [t_lo + s_min, t_hi - s_min]
          relaxed by s_N,  |angles_N| <= psi_end.

The terminal conditions say that the platform ends at rest at a pose the remaining cables can
hold with margin s_min, the condition that defines the precomputed safe sets; the first term
picks, among such poses, one close to the pose at the break (Wq weights orientation by w_rot^2).
The twist and acceleration terms (Wn weights the angular part by w_rot^2) keep the motion
direct and smooth, and the soft margin s_soft keeps tension in reserve for the feedback of the
MPC, which has to correct a path along which the platform is not stable on its own. The nearest
pose of the precomputed safe set only seeds the initial guess. The slacks s_k make the problem
feasible from any state: a solution with all s_k below tol_slack is a recovery path (branch 2);
otherwise it is the descent with the least violation of the bounds (branch 3), and the run is
recorded as such. The tensions of the solution are passed to the MPC as feedforward.

A free duration (minimum time) needed a median of 43 and up to 146 IPOPT iterations on the
test cases, against 21 and 27 with the duration fixed, and gave paths with the tensions on their
bounds; the duration is therefore fixed (t_lo = t_hi).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import casadi as ca
import numpy as np

from ..model.compiled import compiled_nlpsol
from ..model.rigid_casadi import RigidModel


@dataclass(frozen=True)
class RecoveryConfig:
    steps: int = 15                 # N, collocation intervals
    t_lo: float = 1.0               # [s] shortest duration
    t_hi: float = 1.0               # [s] longest duration (equal to t_lo: fixed duration, see below)
    s_path: float = 5.0             # [N] hard tension margin along the path (relaxed by s_k)
    s_soft: float = 20.0            # [N] preferred tension margin along the path (penalised)
    s_min: float = 10.0             # [N] tension margin of the holding tensions at the end
    psi_path_deg: float = 25.0      # largest Euler angle along the path
    psi_end_deg: float = 20.0       # largest Euler angle at the end (as in the safe sets)
    window_clearance: float = 0.02  # [m] clearance to the ends of the drum windows
    w_pos: float = 10.0             # [1/m^2]
    w_rot: float = 1.0              # [m/rad], orientation weight inside Wq and Wn
    w_vel: float = 1.0              # [1/(m^2 s)] on the twist
    w_acc: float = 0.01             # [s^3/m^2] on the accelerations
    w_rate: float = 1e-3            # [1/N^2] on tension changes between intervals
    w_margin: float = 1e-2          # [1/N^2] on the shortfall of the preferred margin
    w_slack: float = 1e3            # [1/N]
    tol_slack: float = 1e-3         # [N] largest slack accepted as a recovery path
    max_iter: int = 300
    max_cpu_time: float = 5.0       # [s]
    transcription: str = "hermite_simpson"   # or "rk4" (multiple shooting)
    tol: float = 1e-6               # IPOPT tolerance
    mu_init: float = 0.1            # IPOPT initial barrier parameter
    mu_strategy: str = "adaptive"   # IPOPT barrier update


class RecoveryOCP:
    def __init__(self, robot, EA: float, cfg: RecoveryConfig = RecoveryConfig(), compiled: bool = True):
        """compiled: generate the callback functions as C and compile them (build/, cached by
        a hash of the code); otherwise they are expanded and evaluated by the CasADi virtual
        machine, which is several times slower."""
        self.robot, self.cfg, self.EA = robot, cfg, EA
        n = self.n = robot.n
        N = cfg.steps
        model = self.model = RigidModel(robot, EA)
        X = ca.MX.sym("X", 12, N + 1)
        hs = cfg.transcription == "hermite_simpson"
        XM = ca.MX.sym("XM", 12, N if hs else 0)
        U = ca.MX.sym("U", n, N)
        th = ca.MX.sym("th", n)
        T = ca.MX.sym("T")
        S = ca.MX.sym("S", N + 1)
        R = ca.MX.sym("R", n, N)
        x0 = ca.MX.sym("x0", 12)
        qg = ca.MX.sym("qg", 6)
        wb = ca.MX.sym("wb", 6)
        dt = T / N
        g, self.g_names = [], []

        def add(expr, name):
            g.append(ca.vec(expr))
            self.g_names.append((name, int(ca.vec(expr).numel())))

        add(X[:, 0] - x0, "init")
        if hs:
            # separated Hermite-Simpson collocation, tensions constant over each interval
            f3 = model.f_ext.map(3 * N)
            wbs = ca.repmat(wb, 1, 3 * N)
            fx = f3(ca.horzcat(X[:, :N], XM, X[:, 1:]), ca.horzcat(U, U, U), wbs)
            f0, fm, f1 = fx[:, :N], fx[:, N:2 * N], fx[:, 2 * N:]
            add(X[:, 1:] - X[:, :N] - dt / 6 * (f0 + 4 * fm + f1), "dyn")
            add(XM - 0.5 * (X[:, :N] + X[:, 1:]) - dt / 8 * (f0 - f1), "mid")
        else:
            step = model.step_ext.map(N)
            x_next = step(X[:, :N], U, ca.repmat(wb, 1, N), ca.repmat(dt, 1, N))
            add(X[:, 1:] - x_next, "dyn")
        add(U + ca.repmat(S[:N].T, n, 1), "t_low")        # >= t_lo + s_path on intact cables
        add(U - ca.repmat(S[:N].T, n, 1), "t_high")       # <= t_hi - s_path on intact cables
        add(U + R, "soft_low")                             # >= t_lo + s_soft on intact cables
        add(U - R, "soft_high")                            # <= t_hi - s_soft on intact cables
        add(ca.sum1(X[6:9, :] ** 2), "speed")
        add(ca.sum1((X[6:9, 1:] - X[6:9, :N]) ** 2) - (robot.a_max * dt) ** 2, "accel")
        lens = model.lengths.map(N + 1)(X)
        add(EA * lens[:, 1:] / (EA + U), "window")        # unstretched lengths at the nodes 1..N
        add(X[6:12, N], "rest")
        xdot_end = model.f_ext(X[:, N], th, wb)
        add(xdot_end[6:12], "hold")
        add(th + S[N], "hold_low")
        add(th - S[N], "hold_high")
        dq = X[:6, N] - qg
        wq = ca.DM([1, 1, 1, cfg.w_rot ** 2, cfg.w_rot ** 2, cfg.w_rot ** 2])
        du = U[:, 1:] - U[:, :N - 1]
        wn = ca.DM([1, 1, 1, cfg.w_rot ** 2, cfg.w_rot ** 2, cfg.w_rot ** 2])
        nu = X[6:12, :]
        acc = (nu[:, 1:] - nu[:, :N]) / dt
        cost = cfg.w_pos * ca.sum1(wq * dq ** 2) + cfg.w_rate * ca.sumsqr(du) \
            + cfg.w_vel * dt * ca.sum1(ca.sum2(nu ** 2) * wn) + cfg.w_acc * dt * ca.sum1(ca.sum2(acc ** 2) * wn) \
            + cfg.w_margin * ca.sumsqr(R) + cfg.w_slack * ca.sum1(S)
        w = ca.vertcat(ca.vec(X), ca.vec(XM), ca.vec(U), th, T, S, ca.vec(R))
        self.n_mid = int(XM.numel())
        p = ca.vertcat(x0, qg, wb)
        self._shapes = (X.shape, U.shape)
        plugin = "ipopt"
        opts = {"expand": not compiled, "print_time": False,
                "ipopt": {"print_level": 0, "max_iter": cfg.max_iter, "max_cpu_time": cfg.max_cpu_time,
                          "tol": cfg.tol, "acceptable_tol": 1e-4, "mu_strategy": cfg.mu_strategy,
                          "mu_init": cfg.mu_init, "sb": "yes"}}
        t0 = time.time()
        nlp = {"x": w, "p": p, "f": cost, "g": ca.vertcat(*g)}
        if compiled:
            name = f"recovery2_{cfg.transcription[:2]}_N{N}"
            self.solver = compiled_nlpsol(name, nlp, plugin, opts)
        else:
            self.solver = ca.nlpsol("recovery", plugin, nlp, opts)
        self.build_seconds = time.time() - t0
        self.nw = int(w.numel())

    # ------------------------------------------------------------------ bounds
    def _bounds(self, broken: int, x0):
        cfg, r, n, N = self.cfg, self.robot, self.n, self.cfg.steps
        inf = np.inf
        intact = np.ones(n, bool)
        intact[broken] = False
        psi_p = max(np.deg2rad(cfg.psi_path_deg), np.abs(x0[3:6]).max() + np.deg2rad(2.0))
        psi_e = np.deg2rad(cfg.psi_end_deg)
        xl = np.full((12, N + 1), -inf)
        xu = np.full((12, N + 1), inf)
        xl[3:6, 1:], xu[3:6, 1:] = -psi_p, psi_p
        xl[3:6, N], xu[3:6, N] = -psi_e, psi_e
        ul = np.zeros((n, N))
        uu = np.zeros((n, N))
        uu[intact] = r.t_max * 2.0          # physical range of the relaxed problem
        thl, thu = np.zeros(n), np.zeros(n)
        thu[intact] = r.t_max * 2.0
        ml = np.full((12, self.n_mid // 12), -inf)
        mu_ = np.full((12, self.n_mid // 12), inf)
        ml[3:6], mu_[3:6] = -psi_p, psi_p          # angles at the collocation midpoints as well
        rl = np.zeros((n, N))
        ru = np.where(intact[:, None], inf, 0.0) * np.ones((n, N))
        lbx = np.r_[xl.ravel("F"), ml.ravel("F"), ul.ravel("F"), thl, cfg.t_lo, np.zeros(N + 1), rl.ravel("F")]
        ubx = np.r_[xu.ravel("F"), mu_.ravel("F"), uu.ravel("F"), thu, cfg.t_hi, np.full(N + 1, inf), ru.ravel("F")]
        lo_t = np.where(intact, r.t_min + cfg.s_path, -inf)
        hi_t = np.where(intact, r.t_max - cfg.s_path, inf)
        lo_s = np.where(intact, r.t_min + cfg.s_soft, -inf)
        hi_s = np.where(intact, r.t_max - cfg.s_soft, inf)
        lo_h = np.where(intact, r.t_min + cfg.s_min, -inf)
        hi_h = np.where(intact, r.t_max - cfg.s_min, inf)
        if r.window_lo is not None:
            wl = np.where(intact, r.window_lo + cfg.window_clearance, -inf)
            wu = np.where(intact, r.window_lo + r.stroke - cfg.window_clearance, inf)
        else:
            wl, wu = np.full(n, -inf), np.full(n, inf)
        lbg, ubg = [], []
        for name, size in self.g_names:
            if name in ("init", "dyn", "mid", "rest", "hold"):
                lo = hi = np.zeros(size)
            elif name == "t_low":
                lo, hi = np.tile(lo_t, N), np.full(size, inf)
            elif name == "t_high":
                lo, hi = np.full(size, -inf), np.tile(hi_t, N)
            elif name == "soft_low":
                lo, hi = np.tile(lo_s, N), np.full(size, inf)
            elif name == "soft_high":
                lo, hi = np.full(size, -inf), np.tile(hi_s, N)
            elif name == "speed":
                lo, hi = np.full(size, -inf), np.full(size, r.v_max ** 2)
            elif name == "accel":
                lo, hi = np.full(size, -inf), np.zeros(size)
            elif name == "window":
                lo, hi = np.tile(wl, N), np.tile(wu, N)
            elif name == "hold_low":
                lo, hi = lo_h, np.full(size, inf)
            elif name == "hold_high":
                lo, hi = np.full(size, -inf), hi_h
            lbg.append(lo)
            ubg.append(hi)
        return lbx, ubx, np.concatenate(lbg), np.concatenate(ubg)

    # ------------------------------------------------------------------ solve
    def solve(self, broken: int, x0, w_b, q_seed, guess):
        """x0: estimated state [q; nu] at the start; w_b: wrench bias of the estimator;
        q_seed: nearest pose of the precomputed safe set (initial guess of the end pose);
        guess: callable (T) -> (X (12, N+1), U (n, N), t_hold (n,)) for the initial guess."""
        cfg, n, N = self.cfg, self.n, self.cfg.steps
        x0 = np.asarray(x0, float)
        lbx, ubx, lbg, ubg = self._bounds(broken, x0)
        t_guess = 0.5 * (cfg.t_lo + cfg.t_hi)
        xg, ug, thg = guess(t_guess)
        mid = (0.5 * (xg[:, :N] + xg[:, 1:])).ravel("F") if self.n_mid else np.zeros(0)
        w0 = np.r_[xg.ravel("F"), mid, ug.ravel("F"), thg, t_guess, np.zeros(N + 1), np.zeros(n * N)]
        p = np.r_[x0, x0[:6], np.asarray(w_b, float)]
        t0 = time.time()
        sol = self.solver(x0=w0, p=p, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        seconds = time.time() - t0
        stats = self.solver.stats()
        w = np.asarray(sol["x"]).ravel()
        nx = 12 * (N + 1)
        X = w[:nx].reshape(12, N + 1, order="F")
        w = np.r_[w[:nx], w[nx + self.n_mid:]]
        U = w[nx:nx + n * N].reshape(n, N, order="F")
        th = w[nx + n * N:nx + n * N + n]
        T = float(w[nx + n * N + n])
        S = w[nx + n * N + n + 1:nx + n * N + n + 2 + N]
        return {"X": X, "U": U, "t_hold": th, "T": T, "slack": S, "max_slack": float(S.max()),
                "success": bool(stats["success"]), "status": stats["return_status"],
                "iterations": int(stats.get("iter_count", -1)), "seconds": seconds}

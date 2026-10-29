"""Probability-aware scenario MPC (methods, Sec. 7, Eq. mpc).

At every MPC sample the controller chooses ONE tension sequence t_0..t_{N-1} that is applied
whichever mode is true, and predicts its effect under each scenario s of the active set
(the healthy mode and the most probable fault modes):
    x^s_{k+1} = F(x^s_k, t^s(t_k)),   t^s(t) = (1 - b_s) * t - a_s * delta_s,
    x^s_0 = x_hat^s (the estimate of the filter matched to that mode).
Cost: sum over scenarios of the renormalised probability mu_s times
    sum_k |x^s_k - x_ref,k|^2_{Q_s}  +  terminal cost of the scenario,
plus, once for the shared sequence,
    sum_k |t_k - t_{k-1}|^2_R + w_ff |t_k - t_ff,k|^2.
The healthy scenario ends on |x_N - x_ref,N|^2_{Q_f}; a fault scenario ends on
w_s |q_N - q*_s|^2 + |nu_N|^2_{Q_nu}, which pulls the platform toward a pose the remaining
cables can hold. Its stage cost is the tracking cost scaled by fault_track_scale. The drum
windows enter as a smooth penalty on the predicted lengths (methods, below Eq. mpc). The
tension bounds are hard; a cable known to be broken has the bounds [0, 0].

Implementation: multiple shooting, F one RK4 step of the rigid model over the MPC period, IPOPT
with exact Hessian and warm start. One NLP is built per number of scenarios (1 to max_slots),
with all data as parameters, so nothing is rebuilt at run time; its derivative functions are
generated as C and compiled once (model/compiled.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import numpy as np

from ..model.compiled import compiled_nlpsol
from ..model.rigid_casadi import RigidModel


@dataclass(frozen=True)
class MPCConfig:
    dt: float = 0.02
    horizon: int = 12
    max_slots: int = 3
    q_pos: float = 4.0e4          # [1/m^2]
    q_rot: float = 4.0e4          # [1/rad^2]
    q_vel: float = 1.2e2          # [s^2/m^2]   (first value 4e2; scripts/phase3_mpc_weights.py)
    q_angvel: float = 1.2e2       # [s^2/rad^2] (first value 4e2)
    terminal_scale: float = 5.0
    r_rate: float = 3.0e-2        # [1/N^2] on tension changes between samples (first value 1e-2)
    w_ff: float = 1.0e-4          # [1/N^2] on the distance to the feedforward tensions
    w_safe: float = 1.0e5         # [1/m^2, 1/rad^2] fault scenario: terminal distance to the safe pose
    q_terminal_twist: float = 1.0e3
    fault_track_scale: float = 0.1
    w_window: float = 1.0e6       # [1/m^2] drum window penalty
    max_iter: int = 30
    hessian: str = "exact"        # "exact" or "limited-memory" (IPOPT L-BFGS)


WEIGHT_NAMES = ("q_pos", "q_rot", "q_vel", "q_angvel", "terminal_scale", "r_rate", "w_ff", "w_safe",
                "q_terminal_twist", "fault_track_scale")
N_WEIGHTS = len(WEIGHT_NAMES)


class ScenarioMPC:
    def __init__(self, robot, cfg: MPCConfig, window_lo=None, stroke=None):
        self.robot, self.cfg = robot, cfg
        self.n = robot.n
        self.model = RigidModel(robot)
        self.window_lo = robot.window_lo if window_lo is None else window_lo
        self.stroke = robot.stroke if stroke is None else stroke
        self.solvers = {s: self._build(s) for s in range(1, cfg.max_slots + 1)}
        self.warm = {}

    # ----------------------------------------------------------------- build
    def _build(self, n_slots):
        """NLP in MX with the dynamics and the stage costs mapped over the horizon, so that the
        generated code holds one copy of each per-step function instead of N * n_slots copies."""
        cfg, n, nx, N = self.cfg, self.n, 12, self.cfg.horizon
        # per-step functions (SX), mapped over the horizon below
        xk = ca.SX.sym("xk", nx)
        tk = ca.SX.sym("tk", n)
        wk = ca.SX.sym("wk", 6)
        step = ca.Function("mpc_step", [xk, tk, wk], [self.model.step_ext(xk, tk, wk, cfg.dt)])
        xr = ca.SX.sym("xr", nx)
        bk = ca.SX.sym("bk", n)
        qk = ca.SX.sym("qk", nx)
        e = xk - xr
        qdiag = qk
        stage_terms = [ca.dot(qdiag * e, e)]
        if self.window_lo is not None:
            lo = ca.DM(self.window_lo)
            length = self.model.lengths(xk)
            over = ca.fmax(0, length - lo - self.stroke) * (1 - bk)
            under = ca.fmax(0, lo - length) * (1 - bk)
            window = cfg.w_window * (ca.sumsqr(over) + ca.sumsqr(under))
        else:
            window = 0
        stage = ca.Function("mpc_stage", [xk, xr, bk, qk], [stage_terms[0], window])
        step_map = step.map(N)
        stage_map = stage.map(N)

        tseq = ca.MX.sym("T", n, N)
        xs = [ca.MX.sym(f"X{s}", nx, N + 1) for s in range(n_slots)]
        mu = ca.MX.sym("mu", n_slots)
        x0 = ca.MX.sym("x0", nx, n_slots)
        bm = ca.MX.sym("b", n, n_slots)
        am = ca.MX.sym("a", n, n_slots)
        delta = ca.MX.sym("delta", n_slots)
        wb = ca.MX.sym("wb", 6, n_slots)
        qstar = ca.MX.sym("qstar", 6, n_slots)
        is_fault = ca.MX.sym("is_fault", n_slots)
        xref = ca.MX.sym("xref", nx, N + 1)
        tff = ca.MX.sym("tff", n, N)
        tprev = ca.MX.sym("tprev", n)
        # weights are parameters, so that they can be changed (calibration, fault phases)
        # without regenerating the solver: [q_pos, q_rot, q_vel, q_angvel, terminal_scale,
        # r_rate, w_ff, w_safe, q_terminal_twist, fault_track_scale]
        wts = ca.MX.sym("wts", N_WEIGHTS)
        q_mx = ca.vertcat(wts[0] * ca.DM.ones(3), wts[1] * ca.DM.ones(3), wts[2] * ca.DM.ones(3),
                          wts[3] * ca.DM.ones(3))
        q_rep = ca.repmat(q_mx, 1, N)
        cost = 0
        g = []
        ones_n = ca.DM.ones(1, N)
        for s in range(n_slots):
            xsl = xs[s]
            g.append(xsl[:, 0] - x0[:, s])
            t_eff = (1 - bm[:, s]) @ ones_n * tseq - am[:, s] @ (delta[s] * ones_n)
            g.append(ca.vec(xsl[:, 1:] - step_map(xsl[:, :N], t_eff, wb[:, s] @ ones_n)))
            track = 1 - (1 - wts[9]) * is_fault[s]
            st, win = stage_map(xsl[:, :N], xref[:, :N], bm[:, s] @ ones_n, q_rep)
            st_w = stage_map(xsl[:, 1:], xref[:, 1:], bm[:, s] @ ones_n, q_rep)[1]
            cost += mu[s] * (track * ca.sum2(st) + ca.sum2(st_w))
            e_n = xsl[:, N] - xref[:, N]
            healthy_end = wts[4] * ca.dot(q_mx * e_n, e_n)
            dq = xsl[:6, N] - qstar[:, s]
            fault_end = wts[7] * ca.sumsqr(dq) + wts[8] * ca.sumsqr(xsl[6:, N])
            cost += mu[s] * ((1 - is_fault[s]) * healthy_end + is_fault[s] * fault_end)
        dt_seq = tseq - ca.horzcat(tprev, tseq[:, :N - 1])
        cost += wts[5] * ca.sumsqr(dt_seq) + wts[6] * ca.sumsqr(tseq - tff)
        w = ca.vertcat(ca.vec(tseq), *[ca.vec(x) for x in xs])
        p = ca.vertcat(mu, ca.vec(x0), ca.vec(bm), ca.vec(am), delta, ca.vec(wb), ca.vec(qstar), is_fault,
                       ca.vec(xref), ca.vec(tff), tprev, wts)
        nlp = {"x": w, "p": p, "f": cost, "g": ca.vertcat(*g)}
        ipopt = {"print_level": 0, "max_iter": cfg.max_iter, "tol": 1e-6,
                 "acceptable_tol": 1e-4, "warm_start_init_point": "yes", "mu_init": 1e-3, "sb": "yes"}
        if cfg.hessian != "exact":
            ipopt["hessian_approximation"] = "limited-memory"
        opts = {"ipopt": ipopt, "print_time": False, "expand": False}
        tag = "" if cfg.hessian == "exact" else "_lbfgs"
        solver = compiled_nlpsol(f"mpcp{n_slots}_N{N}{tag}", nlp, "ipopt", opts)
        return {"solver": solver, "nw": w.shape[0], "ng": nlp["g"].shape[0], "nslots": n_slots}

    # ----------------------------------------------------------------- solve
    def weights(self, scale=None):
        """Weight vector of the solver; scale: optional dict of factors on named weights."""
        cfg = self.cfg
        w = {"q_pos": cfg.q_pos, "q_rot": cfg.q_rot, "q_vel": cfg.q_vel, "q_angvel": cfg.q_angvel,
             "terminal_scale": cfg.terminal_scale, "r_rate": cfg.r_rate, "w_ff": cfg.w_ff, "w_safe": cfg.w_safe,
             "q_terminal_twist": cfg.q_terminal_twist, "fault_track_scale": cfg.fault_track_scale}
        for k, v in (scale or {}).items():
            w[k] *= v
        return np.array([w[k] for k in WEIGHT_NAMES])

    def solve(self, slots, xref, tff, tprev, t_lb, t_ub, weights=None):
        """slots: list of dicts with keys weight, x0 (12), b (n), a (n), delta, w_b (6, optional),
        qstar (6 or None).
        xref (12, N+1), tff (n, N). Returns (tension sequence (n, N), predicted states per slot,
        solver status)."""
        cfg, n, N = self.cfg, self.n, self.cfg.horizon
        ns = len(slots)
        sol = self.solvers[ns]
        mu = np.array([s["weight"] for s in slots], float)
        mu = mu / mu.sum()
        x0 = np.column_stack([s["x0"] for s in slots])
        bm = np.column_stack([s["b"] for s in slots])
        am = np.column_stack([s["a"] for s in slots])
        delta = np.array([s.get("delta", 0.0) for s in slots], float)
        wb = np.column_stack([s.get("w_b", np.zeros(6)) for s in slots])
        qstar = np.column_stack([np.zeros(6) if s.get("qstar") is None else s["qstar"] for s in slots])
        is_fault = np.array([0.0 if s.get("qstar") is None else 1.0 for s in slots])
        p = np.concatenate([mu, x0.ravel("F"), bm.ravel("F"), am.ravel("F"), delta, wb.ravel("F"), qstar.ravel("F"), is_fault,
                            np.asarray(xref).ravel("F"), np.asarray(tff).ravel("F"), tprev,
                            self.weights() if weights is None else weights])
        lbx = np.full(sol["nw"], -np.inf)
        ubx = np.full(sol["nw"], np.inf)
        lbx[:n * N] = np.tile(t_lb, N)
        ubx[:n * N] = np.tile(t_ub, N)
        key = ns
        if key in self.warm:
            w0 = self.warm[key]
        else:
            w0 = np.concatenate([np.asarray(tff).ravel("F")] + [np.repeat(s["x0"][:, None], N + 1, 1).ravel("F")
                                                               for s in slots])
        w0[:n * N] = np.clip(w0[:n * N], lbx[:n * N], ubx[:n * N])
        res = sol["solver"](x0=w0, p=p, lbx=lbx, ubx=ubx, lbg=0, ubg=0)
        stats = sol["solver"].stats()
        w = np.asarray(res["x"]).ravel()
        tseq = w[:n * N].reshape(n, N, order="F")
        xpred = [w[n * N + s * 12 * (N + 1): n * N + (s + 1) * 12 * (N + 1)].reshape(12, N + 1, order="F")
                 for s in range(ns)]
        # warm start for the next sample: shift by one interval
        t_shift = np.hstack([tseq[:, 1:], tseq[:, -1:]])
        x_shift = [np.hstack([x[:, 1:], x[:, -1:]]) for x in xpred]
        self.warm[key] = np.concatenate([t_shift.ravel("F")] + [x.ravel("F") for x in x_shift])
        return tseq, xpred, stats["return_status"], stats["success"]

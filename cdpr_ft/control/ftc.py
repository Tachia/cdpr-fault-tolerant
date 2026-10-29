"""Fault tolerant controller: winch observers and tension loops (1 kHz), IMM (every
imm_every samples), declaration logic, probability-aware MPC (every mpc_every samples) and the
post-fault planner (methods, Secs. 6 to 8).

What the controller knows: the nominal robot and cable parameters, the unstretched lengths
l_u0 at zero motor angle (homing), the reference, and the sensor readings handed to it by the
simulator (encoder angles and motor torques, one sample late). It never sees the plant state.

Variants (for the baselines and ablations of the evaluation):
  "proposed"      everything above
  "no_ftc"        healthy-mode estimation and MPC only; faults are neither declared nor handled
  "oracle"        the true fault is declared at its onset with its true parameters
  "no_hedging"    the MPC ignores the fault probabilities until a fault is declared
  "no_replanning" after a declaration the nominal reference is kept
  "tda"           diagnosis and planner as "proposed", but the MPC is replaced by a computed-torque
                  law with tension distribution (methods, Sec. 7, last paragraph) at the IMM rate
  "b3"            as "no_hedging", with the mode probabilities of a learned classifier in place of
                  the IMM's for the declaration
  "b3_tight"      as "proposed", with the classifier's probabilities in place of the IMM's for the
                  declaration and for the fault scenarios of the MPC
  "b2"            as "tda", with a threshold rule on the winch quantities for the declaration
In "b2", "b3" and "b3_tight" the IMM keeps running: its filters give the state of the platform
and, after a declaration, the state and the deficit of the declared mode. The other diagnosis is
passed in as `alt` (cdpr_ft.estimation.alternatives).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..analysis.feasibility import max_margin_tensions
from ..control.tda import distribute
from ..model.kinematics import cable_geometry, wrench_matrix
from ..model.rigid_body import bias_twist, gravity_wrench, mass_matrix_twist
from .mpc import ScenarioMPC
from .tda import ComputedTorqueTDA
from .planner import Planner, Plan
from .winch import TensionLoop
from ..estimation.diagnosis import Diagnosis
from ..estimation.imm import IMM

DECLARING = ("proposed", "no_hedging", "no_replanning", "tda", "b2", "b3", "b3_tight")
HEDGING = ("proposed", "b3_tight")
COMPUTED_TORQUE = ("tda", "b2")
ALTERNATIVE_DIAGNOSIS = ("b2", "b3", "b3_tight")


@dataclass
class FTCConfig:
    eso_bandwidth_hz: float = 50.0
    tension_kp: float = 0.5
    tension_ki: float = 20.0
    tension_i_max: float = 20.0
    imm_every: int = 5               # inner samples per IMM cycle
    mpc_every: int = 20              # inner samples per MPC sample
    mu_scenario: float = 1e-3        # mu_s: smallest probability of a fault scenario in the MPC
    max_fault_scenarios: int = 2     # N_s
    mu_declare: float = 0.9          # mu_iso
    n_confirm: int = 2               # N_c, consecutive IMM cycles above mu_iso
    variant: str = "proposed"
    lost_tilt_deg: float = 30.0      # termination: platform lost beyond this tilt
    plan_latency: float = 0.4        # [s] L, online planner only: 99th percentile of the solve times 0.38 s
    interim: str = "seed_path"       # reference until the new plan starts: "nominal", "safe_hold", "seed_path"
    imm_input: str = "command"       # IMM input: commands ("command") or winch tension estimates ("estimate")
    mpc_weight_scale: tuple = ()     # ((name, factor), ...) applied to the MPC weights of MPCConfig
    post_break_weights: tuple = (("q_vel", 0.3), ("q_angvel", 0.3), ("r_rate", 3.0))   # on top, after a break declaration
    planner_source: str = "library"  # "library" (plans computed offline along the task) or "online"
    library_delay: float = 0.015     # [s] typical detection delay: library entry for t_declare - this
    record_inputs: bool = False      # log readings and commands for the offline replay of the diagnosis


class FaultTolerantController:
    def __init__(self, robot, cw, l_u0, reference, imm: IMM, mpc: ScenarioMPC, planner: Planner,
                 safe_sets, cfg: FTCConfig, dt: float, t0_tensions, q0, oracle_fault=None, task=None,
                 alt=None, ct_gains=None, ct_gains_post=None):
        self.task = task
        self.alt = alt                   # diagnosis in place of the IMM probabilities (b2, b3, b3_tight)
        # gains (kp, kd) of the computed-torque law of "tda" and "b2"; default: those of Phase 3.
        # ct_gains_post: gains after a break declaration (B2), the counterpart of the post-break
        # weights of the MPC; default: unchanged
        self.ct_gains = ct_gains if ct_gains is not None else ((100.0,) * 6, (20.0,) * 6)
        self.ct_gains_post = ct_gains_post
        assert (alt is not None) == (cfg.variant in ALTERNATIVE_DIAGNOSIS), "alt diagnosis and variant disagree"
        self.robot, self.cw, self.l_u0 = robot, cw, np.asarray(l_u0, float)
        self.reference = reference
        self.imm, self.mpc, self.planner, self.safe = imm, mpc, planner, safe_sets
        self.cfg, self.dt = cfg, dt
        n = self.n = robot.n
        self.diag = Diagnosis(cw, l_u0, imm, cfg.eso_bandwidth_hz, cfg.imm_every, dt, t0_tensions, q0, cfg.imm_input)
        self.eso = self.diag.eso
        self.loop = TensionLoop(n, cw.inertia, cw.transmission, dt, cfg.tension_kp, cfg.tension_ki,
                                cfg.tension_i_max)
        self.t_d = np.asarray(t0_tensions, float).copy()
        self.t_prev = self.t_d.copy()
        self.ddtheta_ff = np.zeros(n)
        self.t_lb = np.full(n, robot.t_min)
        self.t_ub = np.full(n, robot.t_max)
        self.declared = None             # (mode index, time)
        self.plan = None
        self.interim = None              # pose held between a break declaration and the new plan
        self.hits = 0
        self.oracle_fault = oracle_fault
        self.lost = False
        self.log = {"t_imm": [], "mu": [], "x_hat": [], "t_mpc": [], "mpc_ok": [], "mpc_slots": [],
                    "t_hat": [], "t_d": [], "imm_seconds": [], "mpc_seconds": []}
        if cfg.record_inputs:
            self.log.update(theta=[], tau=[], td_in=[])
        self.nm = 2 * n + 1
        self.mu_alt = np.eye(self.nm)[0]

    # ---------------------------------------------------------------- helpers
    def initial_torque(self):
        return self.cw.transmission * self.t_d

    def _mode_kind(self, j):
        n = self.n
        if j == 0:
            return "healthy", None
        return ("break", j - 1) if j <= n else ("actuator", j - 1 - n)

    def _plan_active(self, times):
        """Mask of the times at which the new plan is followed (it starts at plan.t0)."""
        times = np.atleast_1d(times)
        if self.plan is None or self.cfg.variant == "no_replanning":
            return np.zeros(len(times), bool)
        return times >= self.plan.t0 - 1e-12

    def _reference(self, times):
        q, nu, dnu = self.reference(times)
        if self.interim is not None and self.cfg.interim != "nominal":
            # between a break declaration and the start of the new plan
            hold = (times >= self.declared[1] - 1e-12) & ~self._plan_active(times)
            if hold.any():
                q, nu, dnu = q.copy(), nu.copy(), dnu.copy()
                if self.cfg.interim == "safe_hold":
                    q[hold], nu[hold], dnu[hold] = self.interim, 0.0, 0.0
                else:
                    q[hold], nu[hold], dnu[hold] = self._interim_path(times[hold])
        use = self._plan_active(times)
        if use.any():
            q, nu, dnu = q.copy(), nu.copy(), dnu.copy()
            q[use], nu[use], dnu[use] = self.plan.sample(times[use])
        return q, nu, dnu

    def _reference_window(self, t_now):
        cfg, mpc = self.cfg, self.mpc
        times = t_now + mpc.cfg.dt * np.arange(mpc.cfg.horizon + 1)
        q, nu, dnu = self._reference(times)
        xref = np.hstack([q, nu]).T
        active = self.t_ub > 0
        tff = np.zeros((self.n, mpc.cfg.horizon))
        r = self.robot
        use = self._plan_active(times[:-1])
        plan_t = self.plan.tensions(times[:-1]) if use.any() else None
        for k in range(mpc.cfg.horizon):
            if use[k] and plan_t is not None:
                tff[:, k] = np.clip(plan_t[:, k], self.t_lb, self.t_ub)
                continue
            m_nu = mass_matrix_twist(q[k], r.mass, r.com, r.inertia_com)
            wrench = m_nu @ dnu[k] + bias_twist(q[k], nu[k], r.mass, r.com, r.inertia_com) \
                - gravity_wrench(q[k], r.mass, r.com, r.gravity)
            t_act, _, _ = distribute(wrench_matrix(q[k], r.anchors, r.attachments)[:, active], wrench,
                                     self.t_lb[active], self.t_ub[active])
            tff[active, k] = t_act
        return xref, tff

    def _nearest_safe(self, cable, q_hat):
        pos, ori = self.safe[cable]
        d = np.linalg.norm(pos - q_hat[:3], axis=1) + self.planner.cfg.w_rot * np.linalg.norm(ori - q_hat[3:], axis=1)
        k = int(np.argmin(d))
        return np.r_[pos[k], ori[k]]

    # ---------------------------------------------------------------- declaration
    def _declare(self, j, t_now, delta=None):
        kind, i = self._mode_kind(j)
        self.declared = (j, t_now)
        x_hat = self.imm.x[j, :12]
        if kind == "break":
            self.t_lb[i] = 0.0
            self.t_ub[i] = 0.0
            self.loop.hold_winch(i, self.eso.z[i, 0])
            if self.cfg.variant == "no_replanning":
                return
            # the plan is computed from the state predicted L seconds ahead and followed from
            # then on; until then the controller steers, with the model of the declared mode,
            # to the nearest pose of the safe set (reference at rest there, and in the MPC the
            # terminal cost towards it)
            if self.cfg.planner_source == "library":
                delay = 0.0 if self.cfg.variant == "oracle" else self.cfg.library_delay
                plan = self.planner.from_library(self.task, i, t_now, self.reference, delay)
                if plan is not None:
                    self.plan = plan
                    self._log_plan(t_now, plan)
                    return
            lat = self.cfg.plan_latency
            self.interim = self._nearest_safe(i, x_hat[:6])
            self._interim_path = self.planner.seed_path(x_hat[:6], x_hat[6:], self.interim, t_now,
                                                        self.planner.recovery.cfg.t_lo)
            x_start = self._predict(j, t_now, lat)
            plan = self.planner.after_break(i, t_now + lat, x_start[:6], x_start[6:], self.reference,
                                            w_b=self.imm.x[j, 13:19])
            self.plan = plan
            self._log_plan(t_now, plan)
        else:
            self.plan = self.planner.keep_reference(self.reference, t_now, "actuator")

    def _log_plan(self, t_now, plan):
        self.log.setdefault("plan", []).append({"t_declare": t_now, "t_start": max(plan.t0, t_now), "branch": plan.branch,
                                                "kappa": plan.kappa, "duration": plan.duration,
                                                "target": None if plan.target is None else plan.target.tolist(),
                                                **{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                                   for k, v in plan.info.items() if k not in ("seed", "solution")}})

    def _predict(self, j, t_now, horizon):
        """State of mode j predicted `horizon` seconds ahead. With the MPC, its own prediction
        after an immediate solve with the new mode; with the computed-torque law, the rigid model
        with the present tension command held."""
        model, w_b = self.imm.model, self.imm.x[j, 13:19]
        if self.cfg.variant not in COMPUTED_TORQUE:
            self._mpc_step(t_now)
            xp = self._last_xpred[0]
            span = self.mpc.cfg.dt * (xp.shape[1] - 1)
            if horizon <= span:
                s = horizon / self.mpc.cfg.dt
                k = int(min(np.floor(s), xp.shape[1] - 2))
                a = s - k
                return (1 - a) * xp[:, k] + a * xp[:, k + 1]
            # beyond the MPC horizon: rigid model with the last tension of the MPC sequence held
            x, rest, t_cmd = xp[:, -1].copy(), horizon - span, self._last_tseq[:, -1]
        else:
            x, rest, t_cmd = self.imm.x[j, :12].copy(), horizon, np.clip(self.t_d, self.t_lb, self.t_ub)
        steps = max(1, int(round(rest / 0.005)))
        for _ in range(steps):
            x = np.asarray(model.step_ext(x, t_cmd, w_b, rest / steps)).ravel()
        return x

    # ---------------------------------------------------------------- one inner sample
    def __call__(self, k, t, y):
        cfg, n = self.cfg, self.n
        if cfg.record_inputs:
            self.log["theta"].append(np.full(n, np.nan) if y is None else np.array(y["theta"], float))
            self.log["tau"].append(np.full(n, np.nan) if y is None else np.array(y["tau"], float))
            self.log["td_in"].append(self.t_d.copy())
        tic = time.perf_counter()
        t_hat, mu = self.diag.sample(k, y, self.t_d)
        mu_dec = mu
        if self.alt is not None:
            mu_alt = self.alt.sample(k, y, self.t_d)
            assert (mu_alt is None) == (mu is None), "alternative diagnosis off the estimator time line"
            if mu_alt is not None:
                self.mu_alt = mu_alt
                self.log.setdefault("mu_alt", []).append(mu_alt.copy())
            mu_dec = mu_alt
        if mu is not None:
            self.log["imm_seconds"].append(time.perf_counter() - tic)
        # oracle: declare at the true onset
        if cfg.variant == "oracle" and self.declared is None and self.oracle_fault is not None \
                and t >= self.oracle_fault[1] - 1e-12:
            self._declare(self.oracle_fault[0], t)
        if mu is not None:
            self.log["t_imm"].append(t - self.dt)
            self.log["mu"].append(mu.copy())
            self.log["x_hat"].append(self.imm.fused_state().copy())
            self.log.setdefault("deficit", []).append(self.imm.deficits().copy())
            if cfg.variant in DECLARING and self.declared is None:
                j = int(np.argmax(mu_dec))
                if j != 0 and mu_dec[j] >= cfg.mu_declare:
                    self.hits = self.hits + 1 if getattr(self, "_last_hit", None) == j else 1
                    self._last_hit = j
                    if self.hits >= cfg.n_confirm:
                        self._declare(j, t)
                else:
                    self.hits, self._last_hit = 0, None
        # MPC, or the computed-torque law of the "tda" and "b2" variants
        if cfg.variant in COMPUTED_TORQUE:
            if k % cfg.imm_every == 0:
                self._tda_step(t)
        elif k % cfg.mpc_every == 0:
            self._mpc_step(t)
        # accommodation of a declared actuator fault: correct the winch estimate by the deficit
        t_hat_loop = t_hat.copy()
        if self.declared is not None and self._mode_kind(self.declared[0])[0] == "actuator":
            i = self._mode_kind(self.declared[0])[1]
            t_hat_loop[i] -= self.imm.deficits()[i]
        angle = y["theta"] if y is not None else np.zeros(n)
        self.log["t_hat"].append(t_hat)
        self.log["t_d"].append(self.t_d.copy())
        return self.loop.command(self.t_d, t_hat_loop, self.ddtheta_ff, self.eso.speed, angle)

    def _tda_step(self, t):
        if not hasattr(self, "_ct"):
            self._ct = ComputedTorqueTDA(self.robot, kp=self.ct_gains[0], kd=self.ct_gains[1])
        if self.ct_gains_post is not None and self.declared is not None and not hasattr(self, "_ct_post") \
                and self._mode_kind(self.declared[0])[0] == "break":
            self._ct = self._ct_post = ComputedTorqueTDA(self.robot, kp=self.ct_gains_post[0], kd=self.ct_gains_post[1])
        j = self.declared[0] if self.declared is not None else 0
        x = self.imm.x[j, :12]
        q_ref, nu_ref, dnu_ref = self._reference(np.array([t]))
        active = self.t_ub > 0
        t_d, _, _ = self._ct.tensions(x[:6], x[6:], q_ref[0], nu_ref[0], dnu_ref[0], active)
        self.t_d = np.clip(t_d, self.t_lb, self.t_ub)
        self.ddtheta_ff = np.zeros(self.n)

    def _mpc_step(self, t):
        cfg, n, imm = self.cfg, self.n, self.imm
        mu = imm.mu if self.alt is None else self.mu_alt
        slots = []
        if self.declared is not None:
            j = self.declared[0]
            kind, i = self._mode_kind(j)
            b = np.zeros(n)
            a = np.zeros(n)
            delta = 0.0
            qstar = None
            if kind == "break":
                b[i] = 1.0
                if not self._plan_active(np.array([t]))[0] and self.interim is not None:
                    qstar = self.interim
            slots.append(dict(weight=1.0, x0=imm.x[j, :12].copy(), b=b, a=a, delta=delta,
                              w_b=imm.x[j, 13:19].copy(), qstar=qstar))
        else:
            slots.append(dict(weight=max(mu[0], 1e-6), x0=imm.x[0, :12].copy(), b=np.zeros(n), a=np.zeros(n),
                              delta=0.0, w_b=imm.x[0, 13:19].copy(), qstar=None))
            if cfg.variant in HEDGING:
                order = np.argsort(mu[1:])[::-1] + 1
                for j in order[:cfg.max_fault_scenarios]:
                    if mu[j] < cfg.mu_scenario:
                        break
                    kind, i = self._mode_kind(j)
                    b = np.zeros(n)
                    a = np.zeros(n)
                    if kind == "break":
                        b[i] = 1.0
                        qstar = self._nearest_safe(i, imm.x[j, :6])
                        slots.append(dict(weight=mu[j], x0=imm.x[j, :12].copy(), b=b, a=a, delta=0.0,
                                          w_b=imm.x[j, 13:19].copy(), qstar=qstar))
                    else:
                        a[i] = 1.0
                        slots.append(dict(weight=mu[j], x0=imm.x[j, :12].copy(), b=b, a=a,
                                          delta=float(imm.x[j, 12]), w_b=imm.x[j, 13:19].copy(), qstar=None))
        xref, tff = self._reference_window(t)
        scale = dict(cfg.mpc_weight_scale)
        if self.declared is not None and self._mode_kind(self.declared[0])[0] == "break":
            for name, factor in cfg.post_break_weights:
                scale[name] = scale.get(name, 1.0) * factor
        weights = self.mpc.weights(scale) if scale else None
        tic = time.perf_counter()
        tseq, xpred, status, ok = self.mpc.solve(slots, xref, tff, self.t_prev, self.t_lb, self.t_ub, weights)
        self.log["mpc_seconds"].append(time.perf_counter() - tic)
        self.t_d = np.clip(tseq[:, 0], self.t_lb, self.t_ub)
        self.t_prev = self.t_d.copy()
        self._last_xpred, self._last_tseq = xpred, tseq
        # feedforward of the drum acceleration from the predicted motion of the first scenario
        r = self.robot
        lens = [cable_geometry(xpred[0][:6, m], r.anchors, r.attachments)[1] for m in range(3)]
        ddl = (lens[2] - 2 * lens[1] + lens[0]) / self.mpc.cfg.dt ** 2
        self.ddtheta_ff = -ddl / self.cw.transmission
        self.log["t_mpc"].append(t)
        self.log["mpc_ok"].append(bool(ok))
        self.log["mpc_slots"].append(len(slots))

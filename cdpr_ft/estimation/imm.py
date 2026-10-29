"""Platform level diagnosis: interacting multiple model estimator over 2n + 1 modes
(methods, Sec. 6.2).

Modes: M0 healthy, B_i break of cable i, A_i actuator fault on winch i (tension deficit delta_i).
Every mode-matched filter is an extended Kalman filter on the rigid-cable platform model.

State (28): x = [q (6); nu (6); delta (1); w_b (6); beta (n = 8); kappa (1)]
  delta  tension deficit of the faulty winch, used by the A_i filters only (zero elsewhere)
  w_b    slowly varying external wrench about P: what the nominal model misses because the true
         mass, centre of mass and inertia differ from the nominal ones
  beta   offsets of the unstretched-length readings: anchor positions and homing
  kappa  scale of the axial stiffness, EA_true = kappa EA
The three bias groups are shared by all modes and change slowly (random walks); a fault is an
abrupt change that the bias states cannot follow on the time scale of the diagnosis, which is
what lets the fault modes, and not the biases, explain it. Without them the A_i filters, whose
deficit is a free parameter, absorb the model mismatch of a randomised plant and win in
fault-free runs.

Under mode j the cables deliver
    t^(j) = (1 - b_j) * u - a_j * delta,
where u is the tension estimated at the winches (observer output, averaged over the estimator
period): under A_i the observer of winch i reads its tension too high by the deficit, because
it attributes the torque the drive does not deliver to the load. The measurement
y = [l_u from the encoders; t_hat from the winch observers] is expected to be
    l_u = kappa EA l(q) / (kappa EA + t^(j)) + beta,     t_hat = (1 - b_j) * t_d,
the second part saying that the tension loops follow their commands t_d except on a broken
cable. (The first version used the commands t_d as the input. After a break the tension loops
follow their commands with errors of tens of newtons for some time; as input, these errors act
as an unmodelled force several times larger than the process noise, and the pose estimate of the
true mode degraded from 0.1 mm to several millimetres, which the MPC then amplified.)
Under B_i the length reading of the broken winch carries no information on the pose; it keeps
its place in y with a large variance, so that all modes are compared on the same data and the
broad density is the price B_i pays for not predicting it.

Mode probabilities are computed in the log domain with the log-sum-exp normalisation
(methods, Eq. mu), and the code asserts that they are non-negative and sum to one.

Timing: the measurement reaches the estimator one inner sample late, so the estimator runs on the
delayed time line: it predicts over the last estimator period with the tension command applied
over it and updates with the measurement taken at its end.

Homing: the platform starts at rest at a known pose q0 with known commanded tensions; the first
length reading then fixes the prior mean of beta (home()).

Prediction and measurement functions are CasADi expressions with exact Jacobians, mapped over
the modes and compiled to C (model/compiled.py); the Kalman algebra is batched NumPy.
"""
from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import numpy as np

from ..model.compiled import compiled
from ..model.rigid_casadi import RigidModel

NX = 28
I_Q, I_NU, I_D, I_W, I_B, I_K = slice(0, 6), slice(6, 12), 12, slice(13, 19), slice(19, 27), 27
SHARED = np.r_[np.arange(0, 12), np.arange(13, 28)]


@dataclass(frozen=True)
class IMMConfig:
    dt: float = 0.005            # estimator period [s]
    sigma_acc_lin: float = 0.5   # std of the unmodelled translational acceleration [m/s^2]
    sigma_acc_ang: float = 1.0   # std of the unmodelled angular acceleration [rad/s^2] (first value 0.5; replay)
    sigma_pose: float = 1e-6     # process noise on the pose integration [m or rad per step]
    sigma_deficit: float = 20.0  # random walk of delta [N per sqrt(s)]
    r_length: float = 2e-4       # std of the unstretched-length reading [m]
    r_tension: float = 1.5       # std of the tension estimate t_hat [N]
    r_length_broken: float = 1.0 # std of the length reading of a broken winch [m]
    fault_rate: float = 1e-3     # prior rate of each fault mode from M0 [1/s]
    return_rate: float = 1e-2    # rate of returning from a fault mode to M0 [1/s]
    deficit_std0: float = 20.0   # prior std of delta [N]
    sigma_wrench_force: float = 0.5    # random walk of w_b, force [N per sqrt(s)]
    sigma_wrench_moment: float = 0.1   # random walk of w_b, moment [N m per sqrt(s)]
    wrench_force_std0: float = 10.0    # prior std of w_b, force [N]  (mass +/-10 %)
    wrench_moment_std0: float = 2.0    # prior std of w_b, moment [N m] (COM +/-2 cm)
    sigma_offset: float = 1e-5         # random walk of beta [m per sqrt(s)]
    offset_std0: float = 1e-3          # prior std of beta after homing [m]
    sigma_kappa: float = 1e-4          # random walk of kappa [1 per sqrt(s)]
    kappa_std0: float = 0.2            # prior std of kappa (EA +/-20 %)
    nis_gate: float = 1.0e4            # innovation gate (normalised innovation squared) of the update
    t_arm: float = 2.0                 # [s] fault modes enter the mode set only after this time


def mode_table(n: int):
    """Break indicators B (N, n) and actuator indicators A (N, n) for N = 2n + 1 modes."""
    nm = 2 * n + 1
    b = np.zeros((nm, n))
    a = np.zeros((nm, n))
    for i in range(n):
        b[1 + i, i] = 1.0
        a[1 + n + i, i] = 1.0
    return b, a


def mode_names(n: int):
    return ["M0"] + [f"B{i + 1}" for i in range(n)] + [f"A{i + 1}" for i in range(n)]


class IMM:
    def __init__(self, robot, EA: float, cfg: IMMConfig):
        self.robot, self.EA, self.cfg = robot, EA, cfg
        n = self.n = robot.n
        self.nm = 2 * n + 1
        self.B, self.A = mode_table(n)
        self._build(robot, EA, cfg.dt)
        p_f = cfg.fault_rate * cfg.dt
        p_r = cfg.return_rate * cfg.dt
        pi = np.zeros((self.nm, self.nm))
        pi[0, 1:] = p_f
        pi[0, 0] = 1.0 - p_f * (self.nm - 1)
        for j in range(1, self.nm):
            pi[j, j] = 1.0 - p_r
            pi[j, 0] = p_r
        self.Pi = pi
        self._pi_unarmed = np.eye(self.nm)
        self._pi_unarmed[1:, 1:] = 0.0
        self._pi_unarmed[1:, 0] = 1.0
        q = np.zeros(NX)
        q[I_Q] = cfg.sigma_pose ** 2
        q[6:9] = (cfg.sigma_acc_lin * cfg.dt) ** 2
        q[9:12] = (cfg.sigma_acc_ang * cfg.dt) ** 2
        q[13:16] = cfg.sigma_wrench_force ** 2 * cfg.dt
        q[16:19] = cfg.sigma_wrench_moment ** 2 * cfg.dt
        q[I_B] = cfg.sigma_offset ** 2 * cfg.dt
        q[I_K] = cfg.sigma_kappa ** 2 * cfg.dt
        self.Q = np.repeat(q[None, :], self.nm, 0)
        self.Q[1 + n:, I_D] = cfg.sigma_deficit ** 2 * cfg.dt
        self.Q[:1 + n, I_D] = 1e-12
        r = np.r_[np.full(n, cfg.r_length ** 2), np.full(n, cfg.r_tension ** 2)]
        self.R = np.repeat(r[None, :], self.nm, 0)
        for i in range(n):
            self.R[1 + i, i] = cfg.r_length_broken ** 2
        self.Qm = np.einsum("ja,ab->jab", self.Q, np.eye(NX))
        self.Rm = np.einsum("ja,ab->jab", self.R, np.eye(2 * n))
        self._B, self._A = np.asfortranarray(self.B.T), np.asfortranarray(self.A.T)

    def _build(self, robot, EA, dt):
        n = robot.n
        model = RigidModel(robot, EA=EA)
        self.model = model
        x = ca.SX.sym("x", NX)
        u = ca.SX.sym("u", n)
        td = ca.SX.sym("td", n)
        b = ca.SX.sym("b", n)
        a = ca.SX.sym("a", n)
        x12, w_b = x[:12], x[13:19]
        beta, kappa = x[19:27], x[27]
        t_eff = (1 - b) * u - a * x[12]
        # mean: one RK4 step; covariance: second-order Taylor transition I + F dt + (F dt)^2 / 2
        # of the continuous Jacobian F (cheaper than differentiating through the RK4 stages)
        x12n = model.step_ext(x12, t_eff, w_b, dt)
        f = ca.vertcat(x12n, x[12:])
        xdot = ca.vertcat(model.f_ext(x12, t_eff, w_b), ca.SX.zeros(NX - 12))
        jac_c = ca.jacobian(xdot, x) * dt
        trans = ca.SX.eye(NX) + jac_c + 0.5 * jac_c @ jac_c
        length = model.lengths(x12)
        ea = kappa * EA
        h = ca.vertcat(ea * length / (ea + t_eff) + beta, (1 - b) * td)
        pred = ca.Function("imm28_pred", [x, u, b, a], [f, trans])
        meas = ca.Function("imm28u_meas", [x, u, td, b, a], [h, ca.jacobian(h, x)])
        nm = 2 * n + 1
        xs, us, tds, bs, as_ = (ca.SX.sym("xs", NX, nm), ca.SX.sym("us", n, nm), ca.SX.sym("tds", n, nm),
                                ca.SX.sym("bs", n, nm), ca.SX.sym("as", n, nm))
        self.f_pred = compiled(ca.Function("imm28_pred_all", [xs, us, bs, as_], pred.map(nm)(xs, us, bs, as_)))
        self.f_meas = compiled(ca.Function("imm28u_meas_all", [xs, us, tds, bs, as_],
                                           meas.map(nm)(xs, us, tds, bs, as_)))
        self._len = model.lengths

    # ------------------------------------------------------------------ state
    def reset(self, q0, nu0=None, pose_std=1e-3, twist_std=1e-3):
        cfg, nm = self.cfg, self.nm
        x = np.zeros(NX)
        x[I_Q] = q0
        if nu0 is not None:
            x[I_NU] = nu0
        x[I_K] = 1.0
        self.x = np.repeat(x[None, :], nm, 0)
        p = np.r_[np.full(6, pose_std ** 2), np.full(6, twist_std ** 2), cfg.deficit_std0 ** 2,
                  np.full(3, cfg.wrench_force_std0 ** 2), np.full(3, cfg.wrench_moment_std0 ** 2),
                  np.full(self.n, cfg.offset_std0 ** 2), cfg.kappa_std0 ** 2]
        self.P = np.repeat(np.diag(p)[None, :, :], nm, 0)
        self.mu = np.zeros(nm)
        self.mu[0] = 1.0
        self.log_lik = np.zeros(nm)
        self.homed = False
        self.steps = 0

    def home(self, y_lu, td):
        """Prior mean of the length offsets from the first reading at the known start pose."""
        length = np.asarray(self._len(self.x[0, :12])).ravel()
        beta = np.asarray(y_lu) - self.EA * length / (self.EA + np.asarray(td))
        self.x[:, I_B] = beta
        self.homed = True

    # ------------------------------------------------------------------ one cycle
    def step(self, y_lu, y_that, td, u=None):
        """One IMM cycle: the winch-level tension estimates u applied over the interval (the
        commands td if u is None), the measurement (y_lu, y_that) taken at its end, and td the
        commands the tension loops follow. Returns the mode probabilities."""
        nm, n = self.nm, self.n
        sh = SHARED
        # 1. mixing of the states all filters share; delta keeps its own estimate. Until the
        # diagnosis is armed (t_arm) the fault modes get no prior probability: the bias states
        # of the healthy filter first converge on the start of the task, which the fault modes,
        # with their free deficits, would otherwise explain faster and win
        armed = self.steps * self.cfg.dt >= self.cfg.t_arm
        self.steps += 1
        pi = self.Pi if armed else self._pi_unarmed
        c_bar = pi.T @ self.mu
        w = pi * self.mu[:, None] / np.maximum(c_bar[None, :], 1e-300)      # w[l, j] = mu_{l|j}
        idle = c_bar < 1e-250                       # modes without prior mass: mix as the others
        w[:, idle] = self.mu[:, None]
        x_mix = self.x.copy()
        x_mix[:, sh] = w.T @ self.x[:, sh]
        dx = self.x[None, :, sh] - x_mix[:, None, sh]                         # (j, l, 27)
        p_sh = self.P[:, sh][:, :, sh]
        mixed = np.tensordot(w.T, p_sh, axes=(1, 0)) + (w.T[:, :, None] * dx).transpose(0, 2, 1) @ dx
        p_mix = np.zeros_like(self.P)
        p_mix[np.ix_(np.arange(nm), sh, sh)] = mixed
        p_mix[:, I_D, I_D] = self.P[:, I_D, I_D]
        # 2. prediction
        tdm = np.repeat(np.asarray(td, float)[:, None], nm, 1)
        um = tdm if u is None else np.repeat(np.asarray(u, float)[:, None], nm, 1)
        f, a_mat = self.f_pred(x_mix.T, um, self._B, self._A)
        x_pred = f.T
        a_mat = a_mat.reshape(NX, nm, NX).transpose(1, 0, 2)
        p_pred = a_mat @ p_mix @ a_mat.transpose(0, 2, 1) + self.Qm
        # 3. update
        h, h_mat = self.f_meas(x_pred.T, um, tdm, self._B, self._A)
        h = h.T
        h_mat = h_mat.reshape(2 * n, nm, NX).transpose(1, 0, 2)
        y = np.r_[y_lu, y_that]
        r = y[None, :] - h
        pht = p_pred @ h_mat.transpose(0, 2, 1)
        s = h_mat @ pht + self.Rm
        with np.errstate(invalid="ignore", over="ignore"):
            sign, log_det = np.linalg.slogdet(np.where(np.isfinite(s), s, 0.0))
        bad = ~np.isfinite(s).all((1, 2)) | (sign <= 0)
        s_safe = np.where(bad[:, None, None], np.eye(2 * n)[None], s)
        s_inv = np.linalg.inv(s_safe)
        s_inv_r = (s_inv @ r[:, :, None])[:, :, 0]
        nis = np.einsum("ja,ja->j", r, s_inv_r)
        k_gain = pht @ s_inv
        x_upd = x_pred + (k_gain @ r[:, :, None])[:, :, 0]
        i_kh = np.eye(NX)[None] - k_gain @ h_mat
        p_upd = i_kh @ p_pred @ i_kh.transpose(0, 2, 1) + (k_gain * self.R[:, None, :]) @ k_gain.transpose(0, 2, 1)
        # a mode whose innovation lies far outside its own prediction (NIS above the gate) keeps
        # its prediction: its likelihood is negligible anyway, and an update through the
        # linearisation at such a distance only produces a meaningless jump
        gated = nis > self.cfg.nis_gate
        self.x = np.where(gated[:, None], x_pred, x_upd)
        self.P = np.where(gated[:, None, None], p_pred, p_upd)
        self.log_lik = -0.5 * (nis + log_det + 2 * n * np.log(2 * np.pi))
        # numerical guard: a mode whose filter is no longer finite restarts from its mixed prior
        # with zero likelihood for this cycle
        bad |= ~np.isfinite(self.x).all(1) | ~np.isfinite(self.P).all((1, 2)) | ~np.isfinite(self.log_lik)
        if bad.any():
            self.x[bad], self.P[bad] = x_mix[bad], p_mix[bad]
            self.log_lik[bad] = -np.inf
        self.n_gated, self.n_reset = int(gated.sum()), int(bad.sum())
        self.innovation = r
        # 4. mode probabilities in the log domain
        log_post = self.log_lik + np.log(np.maximum(c_bar, 1e-300))
        m = log_post.max()
        log_post = log_post - m - np.log(np.exp(log_post - m).sum())
        self.mu = np.exp(log_post)
        assert np.all(self.mu >= 0.0) and abs(self.mu.sum() - 1.0) < 1e-9
        return self.mu

    # ------------------------------------------------------------------ outputs
    def fused_state(self):
        return self.mu @ self.x[:, :12]

    def deficits(self):
        """Estimated tension deficit of every winch, from the A_i filters."""
        return self.x[1 + self.n:, I_D]

    def wrench_bias(self, j=None):
        return (self.mu @ self.x[:, I_W]) if j is None else self.x[j, I_W]

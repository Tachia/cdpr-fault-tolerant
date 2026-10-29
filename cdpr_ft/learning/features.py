"""Inputs of the learned fault classifier (Phase 4), computed from what the controller has.

The classifier sees the same information as the IMM and nothing else: the encoder angles and
motor torques of the winches (one sample late), the tension commands, the reference and the
nominal parameters. Per winch i, at the estimator rate (every imm_every inner samples, on the
time line of the IMM), three channels:

  r_t,i   tension residual: winch-observer estimate minus command, both averaged over the
          estimator period as the IMM uses them [N]
  t_hat,i winch-observer estimate of the tension, averaged over the period [N]
  r_l,i   length residual: the length of cable i implied by the reading, l_u,i (1 + t_hat,i / EA),
          minus its length at the reference pose (rigid geometry), relative to the same
          quantity at the first reading, which is taken at rest at the known start pose [m]

The observer timing, the averaging windows and the time stamps are those of
cdpr_ft.estimation.diagnosis.Diagnosis, so that a classifier trained on these features sees
online exactly what it saw in training (tests/test_phase4.py checks this against the class the
controller uses). The channels are scaled by fixed physical units and rounded to float16, the
precision in which the dataset is stored, so that online and stored features are identical.
"""
from __future__ import annotations

import numpy as np

from ..control.winch import WinchESO
from ..model.kinematics import cable_geometry

# fixed scales: 10 N for the residual, 100 N for the tension estimate, 1 mm for the length residual
SCALES = np.array([10.0, 100.0, 1e-3])
CHANNELS = ("r_t", "t_hat", "r_l")


class FeatureStream:
    def __init__(self, robot, cw, l_u0, reference, eso_bandwidth_hz: float, imm_every: int, dt: float, t0_tensions):
        n = self.n = robot.n
        self.robot, self.cw, self.reference = robot, cw, reference
        self.l_u0, self.imm_every, self.dt = np.asarray(l_u0, float), imm_every, dt
        self.eso = WinchESO(n, 2 * np.pi * eso_bandwidth_hz, cw.inertia, cw.transmission, dt)
        self.eso.reset(np.zeros(n), t0_tensions)
        self.td_window, self.that_window = [], []
        self.r_l0 = None

    def sample(self, k: int, y, t_d):
        """One inner sample, with the arguments of Diagnosis.sample. Returns (time, features
        (3, n) as float32 of float16 values) at the estimator samples, None otherwise."""
        if y is not None:
            self.eso.update(y["theta"], y["tau"])
        t_hat = self.eso.tension() if y is not None else np.array(t_d, float)
        self.td_window.append(np.array(t_d, float))
        self.that_window.append(t_hat.copy())
        m = self.imm_every
        out = None
        if y is not None and k % m == 0 and k > m:
            td_avg = np.mean(self.td_window[-m - 1:-1], axis=0)
            u_avg = np.mean(self.that_window[-m:], axis=0)
            t_read = k * self.dt - self.dt                     # time of the reading (IMM time line)
            l_u = self.l_u0 - self.cw.transmission * np.asarray(y["theta"], float)
            q_ref = self.reference(t_read)[0][0]
            l_ref = cable_geometry(q_ref, self.robot.anchors, self.robot.attachments)[1]
            r_l = l_u * (1.0 + t_hat / self.cw.EA) - l_ref
            if self.r_l0 is None:
                self.r_l0 = r_l.copy()
            f = np.vstack([u_avg - td_avg, u_avg, r_l - self.r_l0]) / SCALES[:, None]
            out = (t_read, f.astype(np.float16).astype(np.float32))
            self.td_window = self.td_window[-m - 1:]
            self.that_window = self.that_window[-m:]
        return out


def features_from_record(records, robot, cw, l_u0, reference, eso_bandwidth_hz, imm_every, dt, t0_tensions):
    """Features of a recorded run (records: 'theta', 'tau' (K, n) with NaN rows where no reading
    was available, 'td_in' (K, n)). Returns times (T,) and features (T, 3, n)."""
    fs = FeatureStream(robot, cw, l_u0, reference, eso_bandwidth_hz, imm_every, dt, t0_tensions)
    times, feats = [], []
    for k in range(len(records["td_in"])):
        th = records["theta"][k]
        y = None if np.isnan(th[0]) else {"theta": th, "tau": records["tau"][k]}
        o = fs.sample(k, y, records["td_in"][k])
        if o is not None:
            times.append(o[0])
            feats.append(o[1])
    return np.array(times), np.array(feats, np.float32)

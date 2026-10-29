"""Fault diagnosis as it runs inside the controller: the winch observers (every sample) and the
IMM (every imm_every samples, on the delayed time line), with the homing of the IMM on the
first reading. The controller and the offline replay used for tuning share this class, so the
replay reproduces the diagnosis of a run exactly, given the readings and tension commands of
that run.
"""
from __future__ import annotations

import numpy as np

from ..control.winch import WinchESO


class Diagnosis:
    def __init__(self, cw, l_u0, imm, eso_bandwidth_hz: float, imm_every: int, dt: float, t0_tensions, q0,
                 imm_input: str = "estimate"):
        """imm_input: "estimate" (winch observer outputs as the IMM input) or "command"."""
        self.imm_input = imm_input
        n = len(l_u0)
        self.cw, self.l_u0, self.imm, self.imm_every, self.dt = cw, np.asarray(l_u0, float), imm, imm_every, dt
        self.eso = WinchESO(n, 2 * np.pi * eso_bandwidth_hz, cw.inertia, cw.transmission, dt)
        self.eso.reset(np.zeros(n), t0_tensions)
        self.imm.reset(q0)
        self.td_window = []
        self.that_window = []

    def sample(self, k: int, y, t_d):
        """One inner sample. y: readings {'theta', 'tau'} (one sample old) or None; t_d: the
        tension command held during the previous sample. Returns (t_hat, mu), mu None when
        the IMM does not run at this sample."""
        if y is not None:
            self.eso.update(y["theta"], y["tau"])
        t_hat = self.eso.tension() if y is not None else np.array(t_d, float)
        if y is not None and not self.imm.homed:
            # homing: the first reading is taken at rest at the known start pose
            self.imm.home(self.l_u0 - self.cw.transmission * y["theta"], t_d)
        self.td_window.append(np.array(t_d, float))
        self.that_window.append(t_hat.copy())
        mu = None
        m = self.imm_every
        if y is not None and k % m == 0 and k > m:
            # the estimator period ends at the reading's time, one sample back: the commands held
            # over it are those of the previous m samples, and the winch estimates of the
            # tensions over it are those computed in the last m samples (each from the reading
            # one sample older)
            td_avg = np.mean(self.td_window[-m - 1:-1], axis=0)
            u_avg = np.mean(self.that_window[-m:], axis=0)
            l_u = self.l_u0 - self.cw.transmission * y["theta"]
            mu = self.imm.step(l_u, t_hat, td_avg, u_avg if self.imm_input == "estimate" else None)
            self.td_window = self.td_window[-m - 1:]
            self.that_window = self.that_window[-m:]
        return t_hat, mu


def replay(records, cw, l_u0, imm, eso_bandwidth_hz, imm_every, dt, t0_tensions, q0, imm_input="estimate",
           return_states=False):
    """Run the diagnosis on the readings and commands recorded in a run (records: dict with
    'theta' and 'tau' (K, n), NaN rows where no reading was available, and 'td_in' (K, n)).
    Returns the IMM times (on the delayed time line, as logged by the controller) and the mode
    probabilities."""
    diag = Diagnosis(cw, l_u0, imm, eso_bandwidth_hz, imm_every, dt, t0_tensions, q0, imm_input)
    t_imm, mus, xs = [], [], []
    for k in range(len(records["td_in"])):
        th = records["theta"][k]
        y = None if np.isnan(th[0]) else {"theta": th, "tau": records["tau"][k]}
        _, mu = diag.sample(k, y, records["td_in"][k])
        if mu is not None:
            t_imm.append(k * dt - dt)
            mus.append(mu.copy())
            if return_states:
                xs.append(imm.fused_state().copy())
    if return_states:
        return np.array(t_imm), np.array(mus), np.array(xs)
    return np.array(t_imm), np.array(mus)

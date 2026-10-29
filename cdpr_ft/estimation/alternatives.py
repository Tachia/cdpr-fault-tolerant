"""Diagnosis sources other than the IMM, for the baselines B2 and B3 of the evaluation.

In every variant the controller keeps running the IMM: its mode-matched filters give the state of
the platform, and after a declaration the state and the tension deficit of the declared mode.
The classes below replace only the mode probabilities, which decide the declaration and, in the
variant that hedges, weight the fault scenarios of the MPC.

Both are fed at every inner sample with the arguments of Diagnosis.sample and return, at the
estimator samples, a probability vector over the 2n + 1 modes (None at the other samples). Their
inputs come from cdpr_ft.learning.features.FeatureStream, which repeats the timing of Diagnosis,
so the probabilities appear on the same cycles as those of the IMM. Before t_from, the arming
time of the IMM's fault modes, both report the healthy mode with probability one.

ClassifierDiagnosis (B3): the trained network applied to the last W cycles of the 24 inputs,
with the softmax of the logits divided by a temperature fitted on validation data
(configs/classifier_phase4b.yaml).

ThresholdDiagnosis (B2): fixed thresholds on the winch quantities, in the manner of residual
based detection: a break of cable i when the tension estimate of winch i falls below
theta_break, an actuator fault of winch i when the tension residual of winch i exceeds
theta_actuator in magnitude. The rule is threshold_mode(); the thresholds are chosen offline on
the Phase 4 learning runs (scripts/phase5_b2_calibrate.py).
"""
from __future__ import annotations

import numpy as np

from ..learning.features import SCALES


def healthy(n_modes: int) -> np.ndarray:
    p = np.zeros(n_modes)
    p[0] = 1.0
    return p


def threshold_mode(f: np.ndarray, theta_break: float, theta_actuator: float) -> int:
    """Mode flagged by the threshold rule for one estimator cycle. f: features (3, n) in the
    scaled units of FeatureStream (rows r_t, t_hat, r_l). A break flag takes precedence; among
    several flags the most extreme winch is chosen. Returns 0 (healthy), 1 + i (break of cable
    i) or 1 + n + i (actuator fault of winch i)."""
    n = f.shape[1]
    t_hat = f[1] * SCALES[1]
    r_t = f[0] * SCALES[0]
    low = t_hat < theta_break
    if low.any():
        i = int(np.argmin(np.where(low, t_hat, np.inf)))
        return 1 + i
    big = np.abs(r_t) > theta_actuator
    if big.any():
        i = int(np.argmax(np.where(big, np.abs(r_t), -np.inf)))
        return 1 + n + i
    return 0


class ThresholdDiagnosis:
    def __init__(self, stream, theta_break: float, theta_actuator: float, t_from: float):
        self.stream, self.t_from = stream, t_from
        self.theta_break, self.theta_actuator = theta_break, theta_actuator
        self.nm = 2 * stream.n + 1

    def sample(self, k, y, t_d):
        o = self.stream.sample(k, y, t_d)
        if o is None:
            return None
        t, f = o
        p = np.zeros(self.nm)
        p[0 if t < self.t_from - 1e-9 else threshold_mode(f, self.theta_break, self.theta_actuator)] = 1.0
        return p


class ClassifierDiagnosis:
    def __init__(self, stream, model, window: int, temperature: float, t_from: float):
        import torch
        self.torch = torch
        self.stream, self.model, self.window = stream, model, int(window)
        self.temperature, self.t_from = float(temperature), t_from
        self.nm = 2 * stream.n + 1
        self.buffer = []
        self.seconds = []

    def sample(self, k, y, t_d):
        o = self.stream.sample(k, y, t_d)
        if o is None:
            return None
        t, f = o
        self.buffer.append(f.reshape(-1))            # channel c = n * kind + winch, as in the dataset
        if len(self.buffer) > self.window:
            del self.buffer[0]
        if t < self.t_from - 1e-9 or len(self.buffer) < self.window:
            return healthy(self.nm)
        import time
        tic = time.perf_counter()
        x = self.torch.from_numpy(np.ascontiguousarray(np.array(self.buffer, np.float32).T[None]))
        with self.torch.no_grad():
            logits = self.model(x)[0].double() / self.temperature
            p = self.torch.softmax(logits, dim=0).numpy()
        self.seconds.append(time.perf_counter() - tic)
        return p

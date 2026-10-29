"""Winch sensors: motor encoder and motor current. There are no cable load cells.

Encoder. The measured motor angle is the true angle plus the encoder's systematic error,
modelled as a once-per-revolution error of the amplitude given by the system accuracy
with a random phase per winch, then quantised to the encoder resolution (counts per
revolution) by truncation, as an incremental count would be.

Current. The drive reports the motor current, i.e. the torque the drive applies
(after its own saturation and speed limiter); the measurement is that torque plus white
Gaussian noise. A loss of effectiveness or a torque bias downstream of the motor
(gear stage, brake) does not appear in this measurement.
"""
from __future__ import annotations

import numpy as np

from ..params import Sensors


class WinchSensors:
    def __init__(self, cfg: Sensors, n: int, rng: np.random.Generator, noise: bool = True):
        self.cfg = cfg
        self.n = n
        self.rng = rng
        self.noise = noise
        self.phase = rng.uniform(0.0, 2.0 * np.pi, n)

    def encoder(self, theta: np.ndarray) -> np.ndarray:
        th = np.asarray(theta, float)
        if not self.noise:
            return th.copy()
        th = th + self.cfg.encoder_accuracy * np.sin(th + self.phase)
        if self.cfg.encoder_counts > 0:
            quantum = 2.0 * np.pi / self.cfg.encoder_counts
            th = quantum * np.floor(th / quantum)
        return th

    def torque(self, tau_drive: np.ndarray) -> np.ndarray:
        tau = np.asarray(tau_drive, float)
        if not self.noise or self.cfg.current_noise_torque_std == 0.0:
            return tau.copy()
        return tau + self.cfg.current_noise_torque_std * self.rng.standard_normal(self.n)

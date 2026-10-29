"""Reference trajectories T1 (horizontal circle) and T2 (helix), evaluated analytically.

The angle along the circle starts and ends at rest. Its rate rises from 0 to the constant
value 2 pi / period with the quintic smoothstep S(s) = 10 s^3 - 15 s^4 + 6 s^5 over one
period, stays constant, and falls back over one period, so position, velocity and
acceleration are continuous. The helix adds a climb proportional to the angle travelled.
The orientation is held at zero.

T1 and T2 as selected in Phase 1 (scripts/phase1_trajectory_design.py): radius 0.5 m,
period 4 s, three turns (16 s); T1 about the home position, T2 climbing 1 m from z = 2.5 m.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _smoothstep(s):
    s = np.clip(s, 0.0, 1.0)
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def _smoothstep_integral(s):
    s = np.clip(s, 0.0, 1.0)
    return 2.5 * s ** 4 - 3 * s ** 5 + s ** 6


def _smoothstep_rate(s):
    inside = (s > 0.0) & (s < 1.0)
    return np.where(inside, 30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4, 0.0)


def smooth_rate(t, t_total, t_ramp):
    """Normalised angular rate in [0, 1]: quintic smoothstep up, constant, smoothstep down."""
    return np.minimum(_smoothstep(t / t_ramp), _smoothstep((t_total - t) / t_ramp))


@dataclass(frozen=True)
class CirclePath:
    center: tuple          # (3,) centre of the circle at the start [m]
    radius: float          # [m]
    period: float          # [s] period at constant speed; also the ramp duration
    turns: int
    climb: float = 0.0     # [m] total rise (helix); 0 for a circle

    @property
    def t_ramp(self) -> float:
        return self.period

    @property
    def duration(self) -> float:
        return self.turns * self.period + self.t_ramp   # each ramp adds half a period of travel

    def angle(self, t):
        """Angle travelled, its rate and its second derivative at times t."""
        t = np.asarray(t, float)
        w, tr, tt = 2 * np.pi / self.period, self.t_ramp, self.duration
        total = w * (tt - tr)
        up, down = t / tr, (tt - t) / tr
        ang = np.where(t < tr, w * tr * _smoothstep_integral(up),
                       np.where(t > tt - tr, total - w * tr * _smoothstep_integral(down),
                                0.5 * w * tr + w * (t - tr)))
        ang = np.clip(ang, 0.0, total)
        dang = w * smooth_rate(t, tt, tr)
        ddang = np.where(t < tr, w / tr * _smoothstep_rate(up),
                         np.where(t > tt - tr, -w / tr * _smoothstep_rate(down), 0.0))
        return ang, dang, ddang

    def sample(self, t):
        """Positions, velocities and accelerations (each of shape (len(t), 3))."""
        ang, dang, ddang = self.angle(t)
        c = np.asarray(self.center, float)
        total = 2 * np.pi * self.turns
        kz = self.climb / total
        ca, sa = np.cos(ang), np.sin(ang)
        p = np.stack([c[0] + self.radius * ca, c[1] + self.radius * sa, c[2] + kz * ang], -1)
        v = np.stack([-self.radius * sa * dang, self.radius * ca * dang, kz * dang], -1)
        a = np.stack([-self.radius * (ca * dang ** 2 + sa * ddang),
                      self.radius * (-sa * dang ** 2 + ca * ddang), kz * ddang], -1)
        return p, v, a


def path_samples(center, radius, period, turns, climb, dt=0.01):
    """Times, positions, velocities and accelerations on a uniform time grid."""
    path = CirclePath(tuple(center), radius, period, turns, climb)
    t = np.arange(0.0, path.duration + 1e-9, dt)
    p, v, a = path.sample(t)
    return t, p, v, a


def reference(name: str, home=(0.0, 0.0, 3.0)) -> CirclePath:
    """The study references: 'T1' circle about home, 'T2' helix climbing 1 m from z = 2.5 m."""
    if name == "T1":
        return CirclePath(tuple(home), 0.5, 4.0, 3, 0.0)
    if name == "T2":
        return CirclePath((home[0], home[1], home[2] - 0.5), 0.5, 4.0, 3, 1.0)
    raise ValueError(name)

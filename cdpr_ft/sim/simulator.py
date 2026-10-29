"""Multirate closed-loop simulation.

Timing at every inner sample t_k = k * dt_ctrl:
  1. faults whose onset time has been reached are switched on;
  2. the sensors read the state x(t_k): encoder angles and the torque the drive applies at
     t_k (the previous command after the drive's saturation and speed limiter);
  3. the controller receives the measurement taken delay_samples samples earlier and
     returns the torque command tau_k;
  4. the plant integrates over [t_k, t_k+1] with tau_k held constant (zero-order hold),
     in dt_ctrl / dt_plant fixed RK4 steps.
The log holds the true state, its pose with Euler angles, tensions and torques at every inner
sample.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from ..model.plant import Faults, Plant
from .sensors import WinchSensors


@dataclass(frozen=True)
class FaultEvent:
    time: float      # onset [s]
    cable: int       # 0-based index
    kind: str        # "break", "loe" (loss of effectiveness rho) or "bias" (torque bias f, N m)
    value: float = 0.0


def _apply(event: FaultEvent, faults: Faults):
    if event.kind == "break":
        faults.broken[event.cable] = True
    elif event.kind == "loe":
        faults.rho[event.cable] = event.value
    elif event.kind == "bias":
        faults.bias[event.cable] = event.value
    else:
        raise ValueError(event.kind)


def simulate(plant: Plant, x0: np.ndarray, controller, t_end: float, dt_plant: float, dt_ctrl: float,
             sensors: WinchSensors, events=(), delay_samples: int = 1, fast=None, advance=None, stop=None):
    """fast: optional CasadiPlant built for (plant, dt_plant, dt_ctrl / dt_plant); it replaces the
    NumPy plant for stepping and outputs and gives the same numbers to round-off.
    advance: optional integrator (x, tau, faults, dt_ctrl) -> state after one control interval,
    used instead of the fixed-step RK4 (e.g. a reference solver for verification).
    stop: optional callable (k, t, x, plant) -> reason or None, checked every inner sample; the
    run ends at the first sample for which it returns a reason (stored in log["stopped"])."""
    n = plant.n
    n_sub = int(round(dt_ctrl / dt_plant))
    if not np.isclose(n_sub * dt_plant, dt_ctrl, rtol=0, atol=1e-12):
        raise ValueError("dt_ctrl must be an integer multiple of dt_plant")
    steps = int(round(t_end / dt_ctrl))
    if fast is not None and (fast.n_sub != n_sub or not np.isclose(fast.dt_plant, dt_plant)):
        raise ValueError("compiled plant built for a different step")
    faults = Faults(n)
    pending = sorted(events, key=lambda e: e.time)
    x = np.array(x0, float)
    tau = np.asarray(controller.initial_torque(), float)
    buf = deque(maxlen=delay_samples + 1)

    log = {k: [] for k in ("t", "x", "q", "nu", "theta", "dtheta", "tension", "tau_cmd", "tau_drive",
                           "drum_flag", "slack")}
    for k in range(steps + 1):
        t = k * dt_ctrl
        while pending and pending[0].time <= t + 1e-12:
            _apply(pending.pop(0), faults)
        _, nu, theta, dtheta = plant.split(x)
        if fast is None:
            tau_drive = plant.drive_torque(tau, dtheta)
            tension = plant.cable_state(x, faults)[0]
        else:
            tension, tau_drive = fast.tension_and_drive(x, tau, faults)
        buf.append({"theta": sensors.encoder(theta), "tau": sensors.torque(tau_drive)})
        log["t"].append(t)
        log["x"].append(x.copy())
        log["q"].append(plant.pose_euler(x))
        log["nu"].append(nu.copy())
        log["theta"].append(theta.copy())
        log["dtheta"].append(dtheta.copy())
        log["tension"].append(tension)
        log["tau_drive"].append(tau_drive)
        log["drum_flag"].append(plant.drum_flags(x))
        log["slack"].append((tension <= 0.0) & ~faults.broken)
        reason = stop(k, t, x, plant) if stop is not None else None
        if reason is not None:
            log["tau_cmd"].append(tau.copy())
            out = {kk: np.asarray(v) for kk, v in log.items()}
            out["stopped"] = (reason, t)
            return out
        if k == steps:
            log["tau_cmd"].append(tau.copy())
            break
        y = buf[0] if len(buf) > delay_samples else None
        tau = np.asarray(controller(k, t, y), float)
        log["tau_cmd"].append(tau.copy())
        if advance is not None:
            x = advance(x, tau, faults, dt_ctrl)
        elif fast is None:
            for _ in range(n_sub):
                x = plant.step(x, tau, dt_plant, faults)
        else:
            x = fast.step_interval(x, tau, faults)
    out = {k: np.asarray(v) for k, v in log.items()}
    out["stopped"] = None
    return out

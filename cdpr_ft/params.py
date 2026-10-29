"""Robot, cable and winch parameters loaded from a YAML description."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@dataclass(frozen=True)
class Robot:
    anchors: np.ndarray          # (n, 3) a_i, world frame
    attachments: np.ndarray      # (n, 3) b_i, platform frame
    mass: float
    com: np.ndarray              # (3,) COM in platform frame
    inertia_com: np.ndarray      # (3, 3) about COM, platform frame
    t_min: float
    t_max: float
    gravity: float = 9.81
    home_pose: np.ndarray = field(default_factory=lambda: np.zeros(6))
    stroke: float = float("inf")  # length change one drum can take up (drum capacity) [m]
    window_lo: np.ndarray | None = None  # (n,) shortest free length of each cable as installed [m];
                                         # cable i is usable for window_lo[i] <= l_i <= window_lo[i] + stroke
    v_max: float = float("inf")  # platform speed limit [m/s]
    a_max: float = float("inf")  # platform acceleration limit [m/s^2]

    @property
    def n(self) -> int:
        return self.anchors.shape[0]


@dataclass(frozen=True)
class CableWinch:
    """Cable and winch. The winch angle is the motor angle; all torques, inertias and
    friction are referred to the motor shaft, and the cable length paid out per motor
    radian is ``transmission = drum_radius / gear_ratio``."""
    EA: float            # axial rigidity [N]
    damping_time: float  # Kelvin-Voigt retardation time [s]: c_i = damping_time * k_i
    drum_radius: float   # drum radius [m]
    inertia: float       # inertia at the motor shaft [kg m^2]
    viscous: float       # viscous friction at the motor shaft [N m s]
    coulomb: float       # Coulomb friction at the motor shaft [N m]
    coulomb_eps: float = 1e-2   # smoothing speed for tanh friction [rad/s]
    gear_ratio: float = 1.0               # motor turns per drum turn [-]
    torque_rated: float = float("inf")   # continuous motor torque [N m]
    torque_peak: float = float("inf")    # torque saturation [N m]
    torque_constant: float = float("nan")  # motor torque per ampere [N m/A]
    speed_max: float = float("inf")      # motor speed limit [rad/s]
    encoder_counts: int = 0               # counts per motor revolution; 0 means no quantisation
    speed_limit_band: float = 0.0         # motoring torque tapers to 0 over this fraction below speed_max
    damping_onset: float = 0.0            # stretch below which the cable damping fades to zero [m]

    @property
    def transmission(self) -> float:
        """r_d: cable length paid out per radian of the motor shaft [m/rad]."""
        return self.drum_radius / self.gear_ratio


def load_robot(name: str = "robot_ipanema2_caspr.yaml") -> Robot:
    cfg = yaml.safe_load((CONFIG_DIR / name).read_text())["robot"]
    return Robot(
        anchors=np.asarray(cfg["anchors"], float),
        attachments=np.asarray(cfg["attachments"], float),
        mass=float(cfg["mass"]),
        com=np.asarray(cfg["com"], float),
        inertia_com=np.asarray(cfg["inertia_com"], float),
        t_min=float(cfg["t_min"]),
        t_max=float(cfg["t_max"]),
        gravity=float(cfg["gravity"]),
        home_pose=np.asarray(cfg["home_pose"], float),
        **{k: float(cfg[k]) for k in ("v_max", "a_max") if cfg.get(k) is not None},
        **({"stroke": float(cfg["cable_stroke"])} if cfg.get("cable_stroke") is not None else {}),
        **({"window_lo": np.asarray(cfg["length_window_lo"], float)}
           if cfg.get("length_window_lo") is not None else {}),
    )


def load_cable_winch(name: str = "robot_ipanema2_caspr.yaml", **override) -> CableWinch:
    """Cable/winch parameters. Values still PENDING in the YAML must be supplied
    explicitly through ``override``; this keeps unsourced numbers out of results."""
    cfg = yaml.safe_load((CONFIG_DIR / name).read_text())
    vals = {
        "EA": cfg["cable"]["EA"],
        "damping_time": cfg["cable"]["damping_time"],
        "drum_radius": cfg["winch"]["drum_radius"],
        "inertia": cfg["winch"]["inertia"],
        "viscous": cfg["winch"]["viscous"],
        "coulomb": cfg["winch"]["coulomb"],
    }
    for key in ("gear_ratio", "torque_rated", "torque_peak", "torque_constant", "encoder_counts",
                "speed_limit_band"):
        if cfg["winch"].get(key) is not None:
            vals[key] = cfg["winch"][key]
    if cfg["cable"].get("damping_onset") is not None:
        vals["damping_onset"] = cfg["cable"]["damping_onset"]
    if cfg["winch"].get("speed_max_rpm") is not None:
        vals["speed_max"] = float(cfg["winch"]["speed_max_rpm"]) * 2.0 * np.pi / 60.0
    vals.update(override)
    missing = [k for k, v in vals.items() if v is None]
    if missing:
        raise ValueError(f"cable/winch parameters without source: {missing}; pass them explicitly")
    vals = {k: (int(v) if k == "encoder_counts" else float(v)) for k, v in vals.items()}
    return CableWinch(**vals)


def with_overrides(robot: Robot, **kw) -> Robot:
    return replace(robot, **kw)


@dataclass(frozen=True)
class Sensors:
    """Measurement model of one winch: motor encoder and motor current (no cable load cells)."""
    encoder_counts: int            # counts per motor revolution (0: no quantisation)
    encoder_accuracy: float        # amplitude of the once-per-revolution encoder error [rad]
    current_noise_torque_std: float  # standard deviation of the measured motor torque [N m]
    delay_samples: int             # measurement delay in samples of the inner rate


def load_sensors(name: str = "robot_ipanema2_caspr.yaml", noise_factor: float = 1.0) -> Sensors:
    cfg = yaml.safe_load((CONFIG_DIR / name).read_text())
    s = cfg["sensors"]
    return Sensors(encoder_counts=int(cfg["winch"]["encoder_counts"]),
                   encoder_accuracy=float(s["encoder_accuracy_arcsec"]) * np.pi / (180.0 * 3600.0),
                   current_noise_torque_std=noise_factor * float(s["current_noise_torque_std"]),
                   delay_samples=int(s["delay_samples"]))


def inner_rate(name: str = "robot_ipanema2_caspr.yaml") -> float:
    return float(yaml.safe_load((CONFIG_DIR / name).read_text())["control"]["inner_rate"])


def plant_step(name: str = "robot_ipanema2_caspr.yaml") -> float:
    return float(yaml.safe_load((CONFIG_DIR / name).read_text())["control"]["plant_step"])

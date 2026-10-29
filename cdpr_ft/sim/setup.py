"""Initial conditions: the plant at rest in static equilibrium at a given pose."""
from __future__ import annotations

import numpy as np

from ..analysis.feasibility import max_margin_tensions
from ..model.kinematics import cable_geometry, wrench_matrix
from ..model.plant import Plant
from ..model.rigid_body import gravity_wrench


def equilibrium_tensions(robot, q):
    """Max-margin tensions that hold the platform at rest at pose q (rigid cables)."""
    s, t = max_margin_tensions(wrench_matrix(q, robot.anchors, robot.attachments),
                               gravity_wrench(q, robot.mass, robot.com, robot.gravity),
                               np.full(robot.n, robot.t_min), np.full(robot.n, robot.t_max))
    if s is None or s < 0:
        raise ValueError("pose cannot be held statically")
    return t


def plant_at_rest(robot, cw, q0, tensions=None):
    """Plant whose unstretched lengths make `tensions` (default: max-margin) the static tensions
    at q0, with all motor angles zero. Returns (plant, x0, tensions)."""
    t0 = equilibrium_tensions(robot, q0) if tensions is None else np.asarray(tensions, float)
    length = cable_geometry(q0, robot.anchors, robot.attachments)[1]
    l_u0 = length * cw.EA / (cw.EA + t0)          # t = EA (l - l_u) / l_u
    plant = Plant(robot, cw, l_u0)
    x0 = plant.make_state(q0)
    return plant, x0, t0

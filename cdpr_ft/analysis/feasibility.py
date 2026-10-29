"""Static wrench feasibility and tension distribution via linear programming."""
from __future__ import annotations

from itertools import product

import numpy as np
from scipy.optimize import linprog

from ..model.kinematics import cable_geometry, wrench_matrix
from ..model.rigid_body import gravity_wrench


def max_margin_tensions(w_mat: np.ndarray, w_ext: np.ndarray, t_lo: np.ndarray, t_hi: np.ndarray):
    """Solve  max s  s.t.  W t + w_ext = 0,  t_lo + s <= t <= t_hi - s.

    Returns (s, t). s >= 0 means the wrench is feasible inside the bounds with
    margin s; s < 0 means infeasible. Returns (None, None) if the LP fails
    (e.g. W rank-deficient for the required wrench).
    """
    n = w_mat.shape[1]
    c = np.zeros(n + 1)
    c[-1] = -1.0
    a_eq = np.hstack([w_mat, np.zeros((w_mat.shape[0], 1))])
    b_eq = -w_ext
    a_ub = np.vstack([np.hstack([-np.eye(n), np.ones((n, 1))]),
                      np.hstack([np.eye(n), np.ones((n, 1))])])
    b_ub = np.concatenate([-t_lo, t_hi])
    res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
                  bounds=[(None, None)] * (n + 1), method="highs")
    if res.status != 0:
        return None, None
    return res.x[-1], res.x[:n]


def static_margin(q: np.ndarray, robot, active: np.ndarray | None = None):
    """Margin of the static gravity wrench at pose q with the active cable set."""
    n = robot.n
    active = np.ones(n, bool) if active is None else np.asarray(active, bool)
    w_mat = wrench_matrix(q, robot.anchors, robot.attachments)[:, active]
    w_g = gravity_wrench(q, robot.mass, robot.com, robot.gravity)
    lo = np.full(active.sum(), robot.t_min)
    hi = np.full(active.sum(), robot.t_max)
    s, t = max_margin_tensions(w_mat, w_g, lo, hi)
    if t is not None:
        full = np.zeros(n)
        full[active] = t
        t = full
    return s, t


def within_drum_window(q: np.ndarray, robot, active: np.ndarray | None = None) -> bool:
    """True if the free length of every active cable at pose q lies in its drum window
    [window_lo, window_lo + stroke]. Below the window the drum would have to hold more
    than its capacity, above it the drum would be empty. A broken cable carries no load
    and imposes no length condition. Without installed windows only the tension
    conditions apply."""
    if robot.window_lo is None:
        return True
    length = cable_geometry(q, robot.anchors, robot.attachments)[1]
    lo = robot.window_lo
    if active is not None:
        active = np.asarray(active, bool)
        length, lo = length[active], lo[active]
    return bool(np.all(length >= lo - 1e-9) and np.all(length <= lo + robot.stroke + 1e-9))


# Position grid used by all Phase 1 workspace studies: the frame interior with 0.25 m spacing.
GRID_LIMITS = {"x": (-3.5, 3.5), "y": (-2.5, 2.5), "z": (1.5, 4.5)}
GRID_STEP = 0.25


def workspace_grid(step: float = GRID_STEP):
    """Axes (xs, ys, zs) of the position grid."""
    return tuple(np.arange(lo, hi + 1e-9, step) for lo, hi in GRID_LIMITS.values())


def drum_windows(robot, step: float = GRID_STEP):
    """Installed drum windows: for each cable, the window of width ``robot.stroke`` centred
    on the range of free lengths the cable takes over the static workspace (zero
    orientation, all cables, tensions in [t_min, t_max], position grid of ``step``).

    Returns (window_lo, l_min, l_max) per cable [m]; l_max - l_min is the stroke the
    workspace would need."""
    lengths = []
    for x, y, z in product(*workspace_grid(step)):
        q = np.r_[x, y, z, 0.0, 0.0, 0.0]
        s, _ = static_margin(q, robot)
        if s is not None and s >= 0:
            lengths.append(cable_geometry(q, robot.anchors, robot.attachments)[1])
    lengths = np.array(lengths)
    l_min, l_max = lengths.min(0), lengths.max(0)
    return 0.5 * (l_min + l_max) - 0.5 * robot.stroke, l_min, l_max

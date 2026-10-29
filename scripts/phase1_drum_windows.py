"""Drum windows of the installed cables.

The IPAnema drum stores up to 6 m of cable (Pott et al. 2013, Sec. 2.2), so the free
length of a cable can change by at most 6 m; where that 6 m range sits is fixed by the
length of cable mounted, which the paper does not report. We centre each window on the
range of free lengths the cable takes over the static workspace (zero orientation, all
eight cables, tensions in [t_min, t_max], 0.25 m position grid) and report how much
stroke that workspace would need.
Output: results/phase1_drum_windows.json; the lower ends go into the configuration
as robot.length_window_lo.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdpr_ft.analysis.feasibility import GRID_STEP, drum_windows  # noqa: E402
from cdpr_ft.params import load_robot  # noqa: E402


def main():
    robot = load_robot()
    lo, l_min, l_max = drum_windows(robot)
    out = {
        "t_min": robot.t_min, "grid_step_m": GRID_STEP, "stroke_m": robot.stroke,
        "workspace_length_min_m": l_min.round(4).tolist(),
        "workspace_length_max_m": l_max.round(4).tolist(),
        "stroke_needed_m": (l_max - l_min).round(4).tolist(),
        "window_lo_m": lo.round(3).tolist(),
        "window_hi_m": (lo + robot.stroke).round(3).tolist(),
    }
    print(json.dumps(out, indent=1))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "phase1_drum_windows.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

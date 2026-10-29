"""Phase 5: thresholds of baseline B2 (residual-threshold detection), chosen before the campaign.

B2 declares a break of cable i when the winch estimate of its tension stays below theta_break,
and an actuator fault of winch i when the tension residual of winch i (estimate minus command)
stays above theta_actuator in magnitude, for N_c consecutive estimator cycles
(cdpr_ft.estimation.alternatives.threshold_mode). The rule is applied offline to the inputs of
the B2 calibration runs (scripts/phase5_b2_runs.py: B2's own controller, seeds 150000 and up),
for every combination of the grid, and the combination is chosen by the lexicographic rule used
for the IMM and for B3: fewest runs with a false alarm, most correct declarations, smallest
median break delay; then the most conservative setting (smaller theta_break, larger
theta_actuator, larger N_c). N_c takes the values of the grids of the IMM and of B3.

A first version chose the thresholds on the learning runs of Phases 4 and 4b, which ran the MPC
and not B2's controller; it was replaced, before any evaluation run, by the B2 calibration runs.
Output: results/phase5_b2_thresholds.json and configs/b2_phase5.yaml (with the gains of
scripts/phase5_b2_gains.py, which runs first)
Usage: python scripts/phase5_b2_calibrate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cdpr_ft.learning.declare import evaluate  # noqa: E402
from cdpr_ft.learning.features import SCALES  # noqa: E402
from cdpr_ft.provenance import stamp  # noqa: E402
from phase4b_common import sha16  # noqa: E402

# The grid was extended once, by 9 N and by 1, 1.5 and 2 N, when the first version (on the learning
# runs) put the chosen pair on the edge of the first grid in both thresholds (8 N, 2.5 N).
THETA_BREAK = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 9.0]      # [N]; t_min = 10 N
THETA_ACT = [1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, float("inf")]   # [N]
N_C = [1, 2, 3, 4, 6, 8]
T_FROM = 2.0
N = 8


def modes(x, theta_break, theta_act):
    """Vectorised threshold_mode over cycles; x: (S, 24) scaled features."""
    t_hat = x[:, N:2 * N] * SCALES[1]
    r_t = np.abs(x[:, :N] * SCALES[0])
    low = t_hat < theta_break
    big = r_t > theta_act
    out = np.zeros(len(x), np.int8)
    any_low = low.any(axis=1)
    out[any_low] = 1 + np.argmin(np.where(low, t_hat, np.inf), axis=1)[any_low]
    rest = ~any_low & big.any(axis=1)
    out[rest] = 1 + N + np.argmax(np.where(big, r_t, -np.inf), axis=1)[rest]
    return out


def key(row):
    correct = sum(row[k]["correct"] for k in ("break", "loe", "bias") if k in row)
    br = row.get("break", {}).get("median_delay")
    return (row["false_alarm_runs"], -correct, np.inf if br is None else round(br, 6),
            row["theta_break"], -row["theta_actuator"], -row["n_c"])


class _Data:
    def __init__(self):
        with np.load(ROOT / "results" / "phase5_b2_data.npz") as z:
            self.x, self.t = z["x"].astype(np.float32), z["t"].astype(np.float64)
        self.runs = json.loads((ROOT / "results" / "phase5_b2_runs.json").read_text())["runs"]


def main():
    ds = _Data()
    runs = ds.runs
    cyc = [np.arange(r["offset"], r["offset"] + r["length"]) for r in runs]
    cyc = [k[ds.t[k] >= T_FROM - 1e-9] for k in cyc]
    table = []
    for tb in THETA_BREAK:
        for ta in THETA_ACT:
            seqs = []
            for r, k in zip(runs, cyc):
                top = modes(ds.x[k], tb, ta)
                seqs.append({"fault": r["fault"], "t": ds.t[k], "top": top, "p": np.ones(len(k))})
            for nc in N_C:
                row = evaluate(seqs, 0.5, nc)
                row.update(theta_break=tb, theta_actuator=ta)
                table.append(row)
        best = min((r for r in table if r["theta_break"] == tb), key=key)
        print(tb, {k: best[k] for k in ("theta_actuator", "n_c", "false_alarm_runs")},
              {k: (best[k]["correct"], best[k]["n"]) for k in ("break", "loe", "bias") if k in best}, flush=True)
    chosen = min(table, key=key)
    print("chosen", chosen, flush=True)
    out = {"runs": len(runs), "grid": {"theta_break": THETA_BREAK, "theta_actuator": THETA_ACT, "n_c": N_C},
           "table": table, "chosen": chosen,
           "data": sha16(ROOT / "results/phase5_b2_data.npz"),
           "provenance": stamp()}
    (ROOT / "results" / "phase5_b2_thresholds.json").write_text(json.dumps(out, indent=1, default=float))
    gains = json.loads((ROOT / "results" / "phase5_b2_gains.json").read_text())["chosen"]
    body = {"theta_break": float(chosen["theta_break"]), "theta_actuator": float(chosen["theta_actuator"]),
            "n_c": int(chosen["n_c"]), "t_from": T_FROM,
            "ct_kp": [float(gains["kp"])] * 6, "ct_kd": [float(gains["kd"])] * 6,
            "source": "thresholds: results/phase5_b2_thresholds.json (B2 calibration runs, scripts/phase5_b2_runs.py); "
                      "gains: results/phase5_b2_gains.json (fault-free calibration runs)"}
    (ROOT / "configs" / "b2_phase5.yaml").write_text(
        "# Thresholds of baseline B2 (scripts/phase5_b2_calibrate.py), fixed before the Phase 5 campaign.\n"
        + yaml.safe_dump(body, sort_keys=False))


if __name__ == "__main__":
    main()

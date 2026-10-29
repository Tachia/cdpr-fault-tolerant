"""Phase 3: declaration thresholds from the calibration runs (ROC over mu_iso and N_c).

The calibration runs are recorded with declarations disabled, so a run's mode probabilities
up to the first time a threshold pair would declare are exactly those of a run with that pair
(nothing in the controller depends on the thresholds before a declaration). For every pair
(mu_iso, N_c) the declaration rule of the controller is applied to the recorded probabilities:
a fault is declared when the same fault mode is the most probable one, with probability at
least mu_iso, in N_c consecutive estimator cycles.

  false alarm: a declaration before the onset (fault-free runs: anywhere);
  per fault run: declared after the onset or not (missed), correct mode or not, delay.

Selection rule: among the pairs with no false alarm in the calibration set and every fault
declared after its onset and isolated correctly, the pair with the smallest median delay of
the break declarations; ties are broken by the larger mu_iso, then the larger N_c (more
conservative). As first written, before the results were seen, the rule did not require correct
isolation and took the smallest median break delay over the pairs without false alarm; on the
first calibration set that picked N_c = 1, which isolates one of the eight breaks as the wrong
mode. The condition on correct isolation, which was the intent, was added after that.
Output: results/phase3_roc_<tag>.json
Usage: python scripts/phase3_roc.py <calibration tag>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdpr_ft.provenance import stamp  # noqa: E402
from cdpr_ft.sim.episode import mode_index  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from phase3_calibration_runs import load_runs  # noqa: E402

MU_ISO = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999)
N_C = (1, 2, 3, 4, 6, 8)
N = 8


def first_declaration(t_imm, mu, mu_iso, n_c, t_from=-np.inf):
    hits, last = 0, None
    for k in range(len(t_imm)):
        if t_imm[k] < t_from:
            continue
        j = int(np.argmax(mu[k]))
        if j != 0 and mu[k, j] >= mu_iso:
            hits = hits + 1 if last == j else 1
            last = j
            if hits >= n_c:
                return float(t_imm[k]), j
        else:
            hits, last = 0, None
    return None, None


def evaluate(runs, mu_iso, n_c):
    fa_runs, free_time = 0, 0.0
    per = {"break": [], "loe": [], "bias": []}
    for r in runs:
        t_imm, mu = np.asarray(r["t_imm"]), np.asarray(r["mu"])
        fault = r["fault"]
        onset = fault[2] if fault else np.inf
        free_time += float(min(onset, t_imm[-1]) - t_imm[0])
        t_dec, j = first_declaration(t_imm, mu, mu_iso, n_c)
        if t_dec is not None and t_dec < onset:
            fa_runs += 1
            continue
        if fault:
            kind = "break" if fault[0] == "break" else "actuator"
            true_j = mode_index(N, kind, fault[1])
            per[fault[0]].append({"declared": t_dec is not None, "correct": j == true_j,
                                  "delay": None if t_dec is None else t_dec - onset})
    out = {"mu_iso": mu_iso, "n_c": n_c, "false_alarm_runs": fa_runs, "fault_free_hours": free_time / 3600.0}
    for kind, rows in per.items():
        if not rows:
            continue
        good = [x["delay"] for x in rows if x["declared"] and x["correct"]]
        out[kind] = {"n": len(rows), "declared": sum(x["declared"] for x in rows),
                     "correct": sum(x["declared"] and x["correct"] for x in rows),
                     "median_delay": float(np.median(good)) if good else None,
                     "max_delay": float(np.max(good)) if good else None}
    return out


def all_correct(row):
    return all(row[k]["correct"] == row[k]["n"] for k in ("break", "loe", "bias") if k in row)


def select(table):
    ok = [row for row in table if row["false_alarm_runs"] == 0 and all_correct(row)
          and row.get("break", {}).get("median_delay") is not None]
    if not ok:
        return None
    ok.sort(key=lambda r: (round(r["break"]["median_delay"], 6), -r["mu_iso"], -r["n_c"]))
    return ok[0]


def main(tag):
    folder = ROOT / "results" / "phase3_calib" / tag
    data = load_runs(folder)
    runs = [{"fault": r["spec"]["fault"], "t_imm": r["t_imm"], "mu": r["mu"]} for r in data["runs"]]
    table = [evaluate(runs, m, n) for m in MU_ISO for n in N_C]
    chosen = select(table)
    out = {"calibration_tag": tag, "estimator": data["meta"]["imm"], "table": table, "chosen": chosen,
           "provenance": stamp()}
    (ROOT / "results" / f"phase3_roc_{tag}.json").write_text(json.dumps(out, indent=1))
    for row in table:
        print(json.dumps(row))
    print("chosen:", json.dumps(chosen))


if __name__ == "__main__":
    main(*sys.argv[1:])

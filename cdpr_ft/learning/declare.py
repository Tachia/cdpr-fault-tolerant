"""Declaration rule of the controller applied to a sequence of mode probabilities, for the IMM
and for the learned classifier alike, and the selection of the thresholds.

The rule is that of cdpr_ft/control/ftc.py and scripts/phase3_roc.py: a fault is declared when
the same fault mode is the most probable one, with probability at least mu_iso, in N_c
consecutive estimator cycles. A sequence is given by the most probable mode `top` and its
probability `p` at every cycle, which is all the rule uses (tests/test_phase4.py checks the
equivalence with scripts/phase3_roc.py on the calibration runs).
"""
from __future__ import annotations

import numpy as np

from ..sim.episode import mode_index

N = 8


def top_and_p(mu):
    mu = np.asarray(mu)
    top = np.argmax(mu, axis=1)
    return top.astype(np.int8), mu[np.arange(len(mu)), top].astype(np.float64)


def first_declaration(t, top, p, mu_iso, n_c):
    """Time and mode of the first declaration, or (None, None)."""
    cand = (np.asarray(top) != 0) & (np.asarray(p) >= mu_iso)
    hits, last, prev = 0, -1, -2
    for k in np.flatnonzero(cand):
        # a cycle without a candidate in between resets the count
        hits = hits + 1 if (top[k] == last and k == prev + 1) else 1
        last, prev = int(top[k]), k
        if hits >= n_c:
            return float(t[k]), int(top[k])
    return None, None


def evaluate(runs, mu_iso, n_c):
    """runs: dicts with 'fault' (None or (kind, cable, onset, value)), 't', 'top', 'p'.
    Same output as scripts/phase3_roc.evaluate."""
    fa_runs, free_time = 0, 0.0
    per = {"break": [], "loe": [], "bias": []}
    for r in runs:
        t = np.asarray(r["t"])
        fault = r["fault"]
        onset = fault[2] if fault else np.inf
        free_time += float(min(onset, t[-1]) - t[0])
        t_dec, j = first_declaration(t, r["top"], r["p"], mu_iso, n_c)
        if t_dec is not None and t_dec < onset:
            fa_runs += 1
            continue
        if fault:
            true_j = mode_index(N, "break" if fault[0] == "break" else "actuator", fault[1])
            per[fault[0]].append({"declared": t_dec is not None, "correct": j == true_j,
                                  "delay": None if t_dec is None else t_dec - onset, "mode": j})
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


def run_outcomes(runs, mu_iso, n_c):
    """Per run: false alarm, declared, correct, delay, declared mode (for bootstrap intervals)."""
    rows = []
    for r in runs:
        t = np.asarray(r["t"])
        fault = r["fault"]
        onset = fault[2] if fault else np.inf
        t_dec, j = first_declaration(t, r["top"], r["p"], mu_iso, n_c)
        fa = t_dec is not None and t_dec < onset
        row = {"fault": fault, "false_alarm": bool(fa), "free_time": float(min(onset, t[-1]) - t[0])}
        if fault and not fa:
            true_j = mode_index(N, "break" if fault[0] == "break" else "actuator", fault[1])
            row.update(declared=t_dec is not None, correct=bool(t_dec is not None and j == true_j),
                       delay=None if t_dec is None else float(t_dec - onset), mode=j)
        rows.append(row)
    return rows


def selection_key(row):
    """Lexicographic rule of the protocol: fewest false-alarm runs, most correct declarations,
    smallest median delay of the break declarations, larger mu_iso, larger N_c."""
    correct = sum(row[k]["correct"] for k in ("break", "loe", "bias") if k in row)
    br = row.get("break", {}).get("median_delay")
    return (row["false_alarm_runs"], -correct, np.inf if br is None else round(br, 6), -row["mu_iso"], -row["n_c"])


def select(table):
    return min(table, key=selection_key)

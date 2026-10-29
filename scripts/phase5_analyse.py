"""Phase 5: statistics of the campaign by the analysis plan of configs/evaluation_phase5.yaml.

Reads the condensed results/phase5/rows.jsonl.gz (or results/phase5_work/rows.jsonl, which lacks
the recovery times) and writes results/phase5/analysis.json:
  summary    for every method, path (and both), fault type (and all faults; fault-free), and
             within these by cable group (upper 1-4, lower 5-8) and noise level: shares, medians
             and rates with 95 % intervals
  primary    the primary endpoints with their paired comparisons, p-values and Holm-adjusted
             p-values
  secondary  the comparisons of the protocol's secondary list that need pairing: the 2 x 2 table
             diagnosis x tightening (P, A1, B3T, B3) on the runs of B3T, false alarms of B3T
             against B3, re-planning and the plan library (P against A2 and A5)
Works on a partial campaign as well (every statistic states how many runs or pairs it rests on).
Usage: python scripts/phase5_analyse.py [rows file [output file]]
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdpr_ft.eval.campaign import load_protocol  # noqa: E402
from cdpr_ft.provenance import stamp  # noqa: E402

RNG_SEED, N_BOOT = 0, 2000


def read_rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    rows = []
    with opener(path, "rt") as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:      # a line cut short when the machine stopped
                    pass
    # a pair run again after an interruption keeps its last row
    last = {(r["method"], r["run"]): r for r in rows}
    return list(last.values())


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def poisson_ci(k, exposure, a=0.05):
    if exposure <= 0:
        return [None, None]
    lo = 0.0 if k == 0 else stats.chi2.ppf(a / 2, 2 * k) / 2
    hi = stats.chi2.ppf(1 - a / 2, 2 * k + 2) / 2
    return [float(lo / exposure), float(hi / exposure)]


def boot_median(x, rng):
    x = np.asarray(x, float)
    if len(x) == 0:
        return [None, None]
    b = np.median(x[rng.integers(0, len(x), (N_BOOT, len(x)))], axis=1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def boot_mean_diff(d, rng):
    d = np.asarray(d, float)
    if len(d) == 0:
        return [None, None]
    b = d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(axis=1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def mcnemar(a, b):
    """Exact McNemar test of paired binary outcomes a, b (arrays of bool)."""
    n01 = int(np.sum(~a & b))
    n10 = int(np.sum(a & ~b))
    n = n01 + n10
    p = 1.0 if n == 0 else float(stats.binomtest(n10, n, 0.5).pvalue)
    return {"a_only": n10, "b_only": n01, "p": p}


def wilcoxon(d):
    d = np.asarray(d, float)
    d = d[np.abs(d) > 0]
    if len(d) < 1:
        return 1.0
    return float(stats.wilcoxon(d).pvalue)


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.zeros(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def index(rows):
    by = {}
    for r in rows:
        by.setdefault(r["method"], {})[r["run"]] = r
    return by


def kind(r):
    return r["fault"][0] if r["fault"] else None


def subgroups(r):
    out = ["all", "noise%g" % r["noise_factor"]]
    if r["fault"]:
        out.append("upper" if r["fault"][1] < 4 else "lower")
    return out


def summarise(rows, rng):
    """Per method, path and fault type (and pooled), and within these by cable group and noise."""
    out = {}
    groups = {}
    for r in rows:
        for traj in (r["trajectory"], "all"):
            for sub in subgroups(r):
                groups.setdefault((r["method"], traj, kind(r) or "fault_free", sub), []).append(r)
                if r["fault"]:
                    groups.setdefault((r["method"], traj, "all_faults", sub), []).append(r)
    for (m, traj, k, sub), g in sorted(groups.items()):
        s = {"n": len(g)}
        lost = sum(r["lost"] for r in g)
        s["lost"] = [lost, wilson(lost, len(g))]
        fa = sum(r["false_alarm"] for r in g)
        armed_h = sum(r.get("armed_time_s", 0.0) for r in g) / 3600.0
        s["false_alarm_runs"] = fa
        s["armed_hours"] = armed_h
        s["false_alarms_per_hour"] = [fa / armed_h if armed_h else None, poisson_ci(fa, armed_h)]
        if k != "fault_free":
            corr = sum(bool(r["correct"]) for r in g)
            s["correct"] = [corr, wilson(corr, len(g))]
            s["missed"] = sum(r["missed"] for r in g)
            s["wrong"] = sum(r["correct"] is False for r in g)
            d = [1e3 * r["delay_s"] for r in g if r["correct"]]
            s["delay_ms_median"] = [float(np.median(d)) if d else None, boot_median(d, rng)]
            s["delay_ms_max"] = float(np.max(d)) if d else None
            for key in ("transient_min_margin_N", "transient_outside_pct", "post_outside_pct", "transient_rmse_p_mm",
                        "post_rmse_p_mm", "max_tilt_deg", "target_error_mm", "deficit_rmse_N"):
                v = [r[key] for r in g if r.get(key) is not None]
                if v:
                    s[key] = [float(np.median(v)), boot_median(v, rng), len(v)]
            if any("recovery_time_s" in r for r in g):
                rec = [r["recovery_time_s"] for r in g if r.get("recovery_time_s") is not None]
                s["recovered"] = [len(rec), wilson(len(rec), len(g))]
                s["left_band"] = sum(bool(r.get("left_band")) for r in g)
                s["recovery_time_s"] = [float(np.median(rec)) if rec else None, boot_median(rec, rng), len(rec)]
            if k == "break":
                s["plan_branches"] = {str(b): sum(r.get("plan_branch") == b for r in g)
                                      for b in sorted({r.get("plan_branch") for r in g}, key=str)}
        for key in ("pre_rmse_p_mm", "pre_min_margin_N", "pre_outside_pct"):
            v = [r[key] for r in g if r.get(key) is not None]
            if v:
                s[key] = [float(np.median(v)), boot_median(v, rng), len(v)]
        for name in ("mpc_seconds", "imm_seconds", "classifier_seconds"):
            v = [r[name] for r in g if r.get(name)]
            if v:
                s[name.replace("seconds", "ms")] = {"mean_median": float(1e3 * np.median([x["mean"] for x in v])),
                                                    "p99_median": float(1e3 * np.median([x["p99"] for x in v])),
                                                    "max": float(1e3 * np.max([x["max"] for x in v])), "runs": len(v)}
        out[f"{m}|{traj}|{k}|{sub}"] = s
    return out


def primary(rows, protocol, rng):
    by = index(rows)
    res = []
    for item in protocol["analysis"]["primary"]:
        ep = item["endpoint"]
        entry = {"endpoint": ep, "comparisons": []}
        for comp in item["comparisons"]:
            a_name, b_name = [x.strip() for x in comp.split("vs")]
            a, b = by.get(a_name, {}), by.get(b_name, {})
            runs = sorted(set(a) & set(b))
            c = {"comparison": comp, "pairs": 0}
            if ep.startswith("platform lost"):
                runs = [k for k in runs if kind(a[k]) == "break"]
                x = np.array([a[k]["lost"] for k in runs], bool)
                y = np.array([b[k]["lost"] for k in runs], bool)
                c.update(pairs=len(runs), a=[int(x.sum()), wilson(int(x.sum()), len(x))],
                         b=[int(y.sum()), wilson(int(y.sum()), len(y))],
                         diff_ci=boot_mean_diff(x.astype(float) - y.astype(float), rng), **mcnemar(x, y))
            elif ep.startswith("fault declared"):
                runs = [k for k in runs if a[k]["fault"]]
                x = np.array([bool(a[k]["correct"]) for k in runs])
                y = np.array([bool(b[k]["correct"]) for k in runs])
                c.update(pairs=len(runs), a=[int(x.sum()), wilson(int(x.sum()), len(x))],
                         b=[int(y.sum()), wilson(int(y.sum()), len(y))],
                         diff_ci=boot_mean_diff(x.astype(float) - y.astype(float), rng), **mcnemar(x, y))
            elif ep.startswith("declaration delay"):
                runs = [k for k in runs if a[k].get("correct") and b[k].get("correct")]
                d = np.array([1e3 * (a[k]["delay_s"] - b[k]["delay_s"]) for k in runs])
                c.update(pairs=len(runs), median_diff_ms=float(np.median(d)) if len(d) else None,
                         median_diff_ci=boot_median(d, rng), p=wilcoxon(d))
            elif ep.startswith("false alarms"):
                ka = sum(a[k]["false_alarm"] for k in runs)
                kb = sum(b[k]["false_alarm"] for k in runs)
                ta = sum(a[k].get("armed_time_s", 0.0) for k in runs) / 3600
                tb = sum(b[k].get("armed_time_s", 0.0) for k in runs) / 3600
                p = 1.0 if ka + kb == 0 else float(stats.binomtest(ka, ka + kb, ta / (ta + tb)).pvalue)
                c.update(pairs=len(runs), a=[ka, ta, poisson_ci(ka, ta)], b=[kb, tb, poisson_ci(kb, tb)], p=p)
            else:
                key = "transient_min_margin_N" if "margin" in ep else "transient_outside_pct"
                runs = [k for k in runs if a[k].get(key) is not None and b[k].get(key) is not None]
                d = np.array([a[k][key] - b[k][key] for k in runs])
                c.update(pairs=len(runs), median_diff=float(np.median(d)) if len(d) else None,
                         median_diff_ci=boot_median(d, rng), p=wilcoxon(d))
            entry["comparisons"].append(c)
        ps = [c.get("p", 1.0) for c in entry["comparisons"]]
        for c, h in zip(entry["comparisons"], holm(ps)):
            c["p_holm"] = h
        res.append(entry)
    return res


def paired_binary(a, b, runs, f, rng):
    x = np.array([bool(f(a[k])) for k in runs])
    y = np.array([bool(f(b[k])) for k in runs])
    return {"pairs": len(runs), "a": [int(x.sum()), wilson(int(x.sum()), len(x))],
            "b": [int(y.sum()), wilson(int(y.sum()), len(y))],
            "diff_ci": boot_mean_diff(x.astype(float) - y.astype(float), rng), **mcnemar(x, y)}


def paired_continuous(a, b, runs, key, rng):
    runs = [k for k in runs if a[k].get(key) is not None and b[k].get(key) is not None]
    d = np.array([a[k][key] - b[k][key] for k in runs])
    return {"pairs": len(runs), "median_a": float(np.median([a[k][key] for k in runs])) if runs else None,
            "median_b": float(np.median([b[k][key] for k in runs])) if runs else None,
            "median_diff": float(np.median(d)) if len(d) else None, "median_diff_ci": boot_median(d, rng),
            "p": wilcoxon(d)}


def false_alarms(a, b, runs):
    ka = sum(a[k]["false_alarm"] for k in runs)
    kb = sum(b[k]["false_alarm"] for k in runs)
    ta = sum(a[k].get("armed_time_s", 0.0) for k in runs) / 3600
    tb = sum(b[k].get("armed_time_s", 0.0) for k in runs) / 3600
    p = 1.0 if ka + kb == 0 else float(stats.binomtest(ka, ka + kb, ta / (ta + tb)).pvalue)
    return {"pairs": len(runs), "a": [ka, ta, poisson_ci(ka, ta)], "b": [kb, tb, poisson_ci(kb, tb)], "p": p}


def secondary(rows, rng):
    """Paired comparisons of the secondary list (no multiplicity correction; reported as such)."""
    by = index(rows)
    out = {}
    # 2 x 2 diagnosis x tightening on the runs of B3T (its subset)
    if "B3T" in by:
        common = sorted(set.intersection(*[set(by.get(m, {})) for m in ("P", "A1", "B3", "B3T")]))
        fault = [k for k in common if by["P"][k]["fault"]]
        free = [k for k in common if not by["P"][k]["fault"]]
        brk = [k for k in fault if kind(by["P"][k]) == "break"]
        table = {}
        for m in ("P", "A1", "B3T", "B3"):
            g = by[m]
            corr = [k for k in fault if g[k]["correct"]]
            fa = sum(g[k]["false_alarm"] for k in common)
            armed = sum(g[k].get("armed_time_s", 0.0) for k in common) / 3600
            table[m] = {"fault_runs": len(fault), "fault_free_runs": len(free),
                        "lost_after_break": [sum(g[k]["lost"] for k in brk), len(brk)],
                        "correct": [len(corr), len(fault)],
                        "delay_ms_median": float(np.median([1e3 * g[k]["delay_s"] for k in corr])) if corr else None,
                        "false_alarm_runs": fa, "false_alarms_per_hour": [fa / armed if armed else None, poisson_ci(fa, armed)],
                        "transient_min_margin_N_median": float(np.median([g[k]["transient_min_margin_N"] for k in fault
                                                                          if g[k].get("transient_min_margin_N") is not None])) if fault else None,
                        "transient_outside_pct_median": float(np.median([g[k]["transient_outside_pct"] for k in fault
                                                                         if g[k].get("transient_outside_pct") is not None])) if fault else None}
        out["diagnosis_x_tightening"] = {"table": table,
                                         "false_alarms_B3T_vs_B3": false_alarms(by["B3T"], by["B3"], common),
                                         "false_alarms_P_vs_A1": false_alarms(by["P"], by["A1"], common),
                                         "margin_B3T_vs_B3": paired_continuous(by["B3T"], by["B3"], fault, "transient_min_margin_N", rng),
                                         "margin_P_vs_A1": paired_continuous(by["P"], by["A1"], fault, "transient_min_margin_N", rng)}
    # re-planning and the plan library after breaks
    rep = {}
    for other in ("A2", "A5"):
        if other in by and "P" in by:
            runs = sorted(k for k in set(by["P"]) & set(by[other]) if kind(by["P"][k]) == "break")
            rep[f"P vs {other}"] = {"lost": paired_binary(by["P"], by[other], runs, lambda r: r["lost"], rng),
                                    "transient_outside_pct": paired_continuous(by["P"], by[other], runs, "transient_outside_pct", rng),
                                    "post_rmse_p_mm": paired_continuous(by["P"], by[other], runs, "post_rmse_p_mm", rng)}
            for grp, sel in (("upper", lambda r: r["fault"][1] < 4), ("lower", lambda r: r["fault"][1] >= 4)):
                rr = [k for k in runs if sel(by["P"][k])]
                rep[f"P vs {other}"][f"lost_{grp}"] = paired_binary(by["P"], by[other], rr, lambda r: r["lost"], rng)
    out["replanning"] = rep
    return out


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "phase5" / "rows.jsonl.gz"
    protocol, phash = load_protocol()
    rows = read_rows(path)
    rows = [r for r in rows if r.get("protocol_sha256_16") == phash]
    rng = np.random.default_rng(RNG_SEED)
    out = {"protocol_sha256_16": phash, "rows": len(rows),
           "methods": {m: sum(r["method"] == m for r in rows) for m in sorted({r["method"] for r in rows})},
           "summary": summarise(rows, rng), "primary": primary(rows, protocol, rng),
           "secondary": secondary(rows, rng), "provenance": stamp()}
    dest = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "results" / "phase5" / "analysis.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"rows": out["rows"], "methods": out["methods"]}))
    for e in out["primary"]:
        print(e["endpoint"])
        for c in e["comparisons"]:
            print("  ", {k: v for k, v in c.items() if k not in ("diff_ci", "median_diff_ci")})


if __name__ == "__main__":
    main()

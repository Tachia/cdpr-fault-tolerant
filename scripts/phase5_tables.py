"""Phase 5: LaTeX tables of the campaign, written from results/phase5/analysis.json only.

Every number in these tables is read from the analysis file; nothing is typed by hand. The tables
go to docs/phase5/tables/ and are included by docs/phase5/evaluation.tex.
  primary.tex     the primary endpoints: both methods with 95 % intervals, the paired difference,
                  the p-value and the Holm-adjusted p-value
  by_fault.tex    per method and fault type (both paths): platform lost, correct declarations,
                  delay, transient margin and time outside the bounds, error after the transient,
                  recovery
  fault_free.tex  per method and path: false alarms per hour of armed time, tracking error
  two_by_two.tex  diagnosis x tightening on the runs of B3T
  compute.tex     computation times per cycle
Usage: python scripts/phase5_tables.py [analysis file]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "phase5" / "tables"
METHODS = ["P", "A1", "A2", "A5", "B0", "B1", "B2", "B3", "B3T"]
KINDS = [("break", "Break"), ("loe", "Loss of effectiveness"), ("bias", "Bias")]


def f(x, nd=1):
    if x is None:
        return "--"
    return f"{x:.{nd}f}".replace("-", "$-$")


def pct(k, n):
    return "--" if not n else f"{k}/{n}"


def ci(c, nd=1, scale=1.0):
    if not c or c[0] is None:
        return ""
    return f" [{f(scale * c[0], nd)}, {f(scale * c[1], nd)}]"


def pval(p):
    if p is None:
        return "--"
    return "$<$0.001" if p < 1e-3 else f"{p:.3f}"


def table(name, caption, label, header, rows, align):
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    text = (f"% written by scripts/phase5_tables.py from results/phase5/analysis.json; do not edit\n"
            f"\\begin{{table}}[h]\n\\centering\\small\n\\caption{{{caption}}}\\label{{{label}}}\n"
            f"\\begin{{adjustbox}}{{max width=\\linewidth}}\\begin{{tabular}}{{@{{}}{align}@{{}}}}\n\\toprule\n"
            f"{header} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\\end{{adjustbox}}\n\\end{{table}}\n")
    (OUT / name).write_text(text)


def primary(a):
    rows = []
    for e in a["primary"]:
        ep = e["endpoint"]
        for c in e["comparisons"]:
            m1, m2 = [x.strip() for x in c["comparison"].split("vs")]
            if ep.startswith(("platform lost", "fault declared")):
                v1 = f"{c['a'][0]}/{c['pairs']}" + ci(c["a"][1], 0, 100)
                v2 = f"{c['b'][0]}/{c['pairs']}" + ci(c["b"][1], 0, 100)
                d = ci(c["diff_ci"], 1, 100).strip() + " pp"
            elif ep.startswith("declaration delay"):
                v1 = v2 = "--"
                d = f(c.get("median_diff_ms")) + ci(c.get("median_diff_ci"), 1) + " ms"
            elif ep.startswith("false alarms"):
                v1 = f"{c['a'][0]} in {f(c['a'][1], 2)} h" + ci(c["a"][2], 1)
                v2 = f"{c['b'][0]} in {f(c['b'][1], 2)} h" + ci(c["b"][2], 1)
                d = "--"
            else:
                v1 = v2 = "--"
                unit = " N" if "margin" in ep else " pp"
                d = f(c.get("median_diff")) + ci(c.get("median_diff_ci"), 1) + unit
            if c is e["comparisons"][0]:
                rows.append([f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{ep[0].upper() + ep[1:]}}}}}"])
            rows.append([f"\\quad {m1} vs {m2}", str(c["pairs"]), v1, v2, d, pval(c.get("p")), pval(c.get("p_holm"))])
    table("primary.tex",
          "Primary endpoints. Shares: count of pairs with the event and 95\\% Wilson interval (\\%); false alarms: "
          "count in the armed hours, exact Poisson interval per hour; differences: first method minus second, "
          "95\\% bootstrap interval (mean for shares, median for continuous endpoints). p: McNemar's exact test, "
          "Wilcoxon signed rank test or conditional binomial test for rates; Holm: adjusted within the endpoint. "
          "Continuous endpoints are compared run by run, so only the paired difference is given.",
          "tab:primary", "Endpoint, comparison & Pairs & First & Second & Difference & p & Holm",
          rows, "lrllrrr")


def by_fault(a):
    s = a["summary"]
    rows = []
    for kind, label in KINDS:
        first = True
        for m in METHODS:
            g = s.get(f"{m}|all|{kind}|all")
            if not g:
                continue
            rec = g.get("recovered")
            rows.append([label if first else "", m, str(g["n"]), pct(g["lost"][0], g["n"]),
                         pct(g.get("correct", [0])[0], g["n"]), f(g.get("delay_ms_median", [None])[0], 0),
                         f(g.get("transient_min_margin_N", [None])[0]), f(g.get("transient_outside_pct", [None])[0]),
                         f(g.get("post_rmse_p_mm", [None])[0]), "--" if not rec else pct(rec[0], g["n"]),
                         f(g.get("recovery_time_s", [None])[0], 2) if rec else "--"])
            first = False
    table("by_fault.tex",
          "Fault runs by method and fault type, both paths. Lost: platform lost; correct: fault declared with the "
          "correct mode; delay: median of the correct declarations (ms); margin: median of the smallest tension "
          "margin in the transient (N); outside: median share of the transient with a cable outside its bounds (\\%); "
          "error: median RMS position error after the transient (mm); recovered: error back in the band of the "
          "fault-free runs for 1~s before the run ended, and the median time to that (s).",
          "tab:byfault", "Fault & Method & Runs & Lost & Correct & Delay & Margin & Outside & Error & Recovered & Time",
          rows, "llrrrrrrrrr")


def fault_free(a):
    s = a["summary"]
    rows = []
    for m in METHODS:
        for traj in ("T1", "T2"):
            g = s.get(f"{m}|{traj}|fault_free|all")
            if not g:
                continue
            rate, c = g["false_alarms_per_hour"]
            rows.append([m if traj == "T1" else "", traj, str(g["n"]), str(g["false_alarm_runs"]), f(g["armed_hours"], 2),
                         f(rate, 2) + ci(c, 2), f(g.get("pre_rmse_p_mm", [None])[0], 2),
                         f(g.get("pre_min_margin_N", [None])[0])])
    table("fault_free.tex",
          "Fault-free runs by method and path. False alarms per hour of armed time (from 2~s) with an exact Poisson "
          "interval; error: median RMS position error (mm); margin: median of the smallest tension margin (N).",
          "tab:faultfree", "Method & Path & Runs & False alarm runs & Armed hours & Per hour & Error & Margin",
          rows, "llrrrlrr")


def two_by_two(a):
    t = a.get("secondary", {}).get("diagnosis_x_tightening")
    if not t:
        return
    rows = []
    for m in ("P", "A1", "B3T", "B3"):
        r = t["table"][m]
        rate, c = r["false_alarms_per_hour"]
        rows.append([m, pct(*r["lost_after_break"]), pct(r["correct"][0], r["correct"][1]), f(r["delay_ms_median"], 0),
                     str(r["false_alarm_runs"]), f(rate, 2) + ci(c, 2), f(r["transient_min_margin_N_median"]),
                     f(r["transient_outside_pct_median"])])
    table("two_by_two.tex",
          f"Diagnosis and tightening on the runs of B3T ({t['table']['P']['fault_runs']} fault runs, "
          f"{t['table']['P']['fault_free_runs']} fault-free runs, the same for the four methods).",
          "tab:twobytwo", "Method & Lost after a break & Correct & Delay (ms) & False alarm runs & Per hour & Margin (N) "
          "& Outside (\\%)", rows, "lrrrrlrr")


def compute(a):
    s = a["summary"]
    rows = []
    for m in METHODS:
        g = s.get(f"{m}|all|all_faults|all")
        if not g:
            continue
        mp_, imm, cl = g.get("mpc_ms"), g.get("imm_ms"), g.get("classifier_ms")
        rows.append([m, f(mp_["mean_median"]) if mp_ else "--", f(mp_["p99_median"]) if mp_ else "--",
                     f(mp_["max"]) if mp_ else "--", f(imm["mean_median"], 2) if imm else "--",
                     f(imm["p99_median"], 2) if imm else "--", f(cl["mean_median"], 2) if cl else "--"])
    table("compute.tex",
          "Computation per cycle on the machine of the campaign (two runs in parallel on two cores), fault runs: "
          "medians over runs of the mean and the 99th percentile, and the largest value (ms). Periods: MPC 20~ms, "
          "estimator 5~ms. The estimator cycle includes the alternative diagnosis of B2, B3 and B3T, whose classifier "
          "alone is shown in the last column. B2 has no MPC.",
          "tab:compute", "Method & MPC mean & MPC p99 & MPC max & Estimator mean & Estimator p99 & Classifier mean",
          rows, "lrrrrrr")


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "phase5" / "analysis.json"
    a = json.loads(path.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    primary(a)
    by_fault(a)
    fault_free(a)
    two_by_two(a)
    compute(a)
    print("tables written to", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()

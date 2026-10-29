"""The tables of the manuscript, written from stored files only.

  paper/tables/tab_parameters.tex   robot, cable, winch and sensor values of the model, read from
                                    configs/robot_ipanema2_caspr.yaml (the source of every value is
                                    given there and summarised here), with the randomisation of the runs
  paper/tables/tab_methods.tex      the methods of the evaluation and what each changes
  paper/tables/tab_results.tex      the outcome of every method (results/phase5/analysis.json and
                                    results/phase5/rows.jsonl.gz)
The complete tables of the evaluation (primary endpoints with their statistics, results by fault type,
computation times) are those of docs/phase5/evaluation.pdf, written by scripts/phase5_tables.py.
Numbers are formatted from the files; none is typed by hand.
Usage: python scripts/paper_tables.py
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


OUT = ROOT / "paper" / "tables"
HEAD = "% written by scripts/paper_tables.py from stored files; do not edit\n"


def tabular(caption, label, header, rows, align, star=False):
    env = "table*" if star else "table"
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    return (HEAD + f"\\begin{{{env}}}[t]\n\\centering\\small\n\\caption{{{caption}}}\\label{{{label}}}\n"
            f"\\begin{{tabular}}{{@{{}}{align}@{{}}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body}\n"
            f"\\bottomrule\n\\end{{tabular}}\n\\end{{{env}}}\n")


def g(x, nd=3):
    s = f"{x:.{nd}g}"
    return s.replace("e-05", "\\times10^{-5}").replace("e+05", "\\times10^{5}")


def parameters():
    c = yaml.safe_load((ROOT / "configs" / "robot_ipanema2_caspr.yaml").read_text())
    r, cab, w, sen, ctl = c["robot"], c["cable"], c["winch"], c["sensors"], c["control"]
    a = np.array(r["anchors"])
    b = np.array(r["attachments"])
    rows = [
        ["Frame (anchor box)", f"${np.ptp(a[:, 0]):g}\\times{np.ptp(a[:, 1]):g}$ m, anchors at $z={a[:, 2].min():g}$ and ${a[:, 2].max():g}$ m",
         "CASPR model IPAnema\\_2 \\cite{lau2016caspr}"],
        ["Platform attachments", f"$|x|\\le{np.abs(b[:, 0]).max():g}$, $|y|\\le{np.abs(b[:, 1]).max():g}$, "
         f"$z\\in[{b[:, 2].min():g},{b[:, 2].max():g}]$ m", "CASPR"],
        ["Mass, inertia", f"{r['mass']:g} kg, {np.diag(r['inertia_com'])[0]:g} kg\\,m$^2$ per axis", "CASPR"],
        ["Tension bounds", f"$t_{{\\min}}={r['t_min']:g}$ N, $t_{{\\max}}={r['t_max']:g}$ N", "CASPR ($t_{\\max}$); design choice ($t_{\\min}$)"],
        ["Cable", f"Dyneema, $d={1e3 * cab['diameter']:g}$ mm, $EA={cab['EA']:.0f}$ N, "
         f"retardation {1e3 * cab['damping_time']:g} ms", "\\cite{pott2013ipanema}; EA scaled from \\cite{tempel2015iftomm}"],
        ["Drum, gearbox", f"radius {1e3 * w['drum_radius']:g} mm, $i={w['gear_ratio']:g}$, stroke {r['cable_stroke']:g} m",
         "\\cite{pott2013ipanema}"],
        ["Motor", f"MSK030C: {w['torque_rated']:g} N\\,m rated, {w['torque_peak']:g} N\\,m peak, "
         f"rotor ${g(w['inertia'])}$ kg\\,m$^2$", "sized from \\cite{rexroth_msk2018}"],
        ["Velocity, acceleration limits", f"{r['v_max']:g} m/s, {r['a_max']:g} m/s$^2$", "\\cite{pott2013ipanema}"],
        ["Encoder", f"{w['encoder_counts']} counts/rev, $\\pm${sen['encoder_accuracy_arcsec']:g} arcsec",
         "\\cite{rexroth_msk2018}, option M1"],
        ["Motor current noise", f"{sen['current_noise_torque_std']:g} N\\,m (nominal), $\\times2$ in half of the runs",
         "design choice"],
        ["Rates", f"plant RK4 step {1e6 * ctl['plant_step']:g} $\\mu$s, control {ctl['inner_rate']:g} Hz, "
         f"measurement delay {sen['delay_samples']} sample", "step convergence study"],
        ["Randomised in every run", "mass $\\pm10\\%$, principal inertias $\\pm20\\%$, centre of mass $\\pm2$ cm, "
         "anchors $\\pm5$ mm, $EA\\ \\pm20\\%$ (uniform)", "study design"],
    ]
    (OUT / "tab_parameters.tex").write_text(tabular(
        "Model of the robot. The controller uses the nominal values; the plant of every run draws the "
        "randomised ones from the seed of the run.", "tab:parameters", "Quantity & Value & Source", rows,
        "p{0.2\\linewidth}p{0.47\\linewidth}p{0.27\\linewidth}", star=True))


def methods():
    rows = [["P", "Proposed: IMM diagnosis; before a declaration the MPC weights the fault scenarios by their "
                  "probabilities; after a break declaration it follows a plan from the library"],
            ["A1", "P without the probability aware tightening"],
            ["A2", "P without replanning after a break"],
            ["A3", "P with the recovery plan solved online, applied after 0.4~s"],
            ["B0", "No fault tolerance: healthy model throughout, no declaration"],
            ["B1", "Oracle: the true fault declared at its onset"],
            ["B2", "Thresholds on the winch tension estimates, computed torque control with tension distribution, "
                   "plans from the library"],
            ["B3", "Recurrent classifier in place of the IMM probabilities for the declaration; A1 otherwise"],
            ["B3T", "B3 with the classifier probabilities also weighting the MPC scenarios"]]
    (OUT / "tab_methods.tex").write_text(tabular(
        "Methods of the evaluation. All run on the same plant, sensors and seeds, so every comparison is paired.",
        "tab:methods", "Method & Description", rows, "lp{0.8\\linewidth}"))


def results():
    """Outcome of every method: platforms lost after upper and lower cable breaks, faults declared with the
    correct mode, median delay of the correct break declarations, false alarms per hour of operation with the
    fault modes armed (all rows of the method), and breaks after which the error returned to the band of the
    fault free runs."""
    a = json.loads((ROOT / "results" / "phase5" / "analysis.json").read_text())
    s = a["summary"]
    rows_all = [json.loads(line) for line in gzip.open(ROOT / "results" / "phase5" / "rows.jsonl.gz", "rt")]
    body = []
    for m in ("P", "A1", "A2", "A5", "B0", "B1", "B2", "B3", "B3T"):
        up, lo = s[f"{m}|all|break|upper"], s[f"{m}|all|break|lower"]
        af, br = s[f"{m}|all|all_faults|all"], s[f"{m}|all|break|all"]
        own = [r for r in rows_all if r["method"] == m]
        hours = sum(r["armed_time_s"] for r in own) / 3600
        alarms = sum(r["false_alarm"] for r in own)
        declares = m not in ("B0", "B1")
        delay = br["delay_ms_median"][0]
        body.append([{"A5": "A3"}.get(m, m), f"{up['lost'][0]}/{up['n']}", f"{lo['lost'][0]}/{lo['n']}",
                     f"{af['correct'][0]}/{af['n']}" if m != "B0" else "--",
                     f"{delay:.0f}" if delay is not None and m != "B1" else "--",
                     f"{alarms / hours:.1f}" if declares and m not in ("A2", "A5") else "--",
                     f"{br['recovered'][0]}/{br['n']}"])
    (OUT / "tab_results.tex").write_text(tabular(
        "Outcome of every method. Lost: platform lost after the breaks of an upper and of a lower cable; "
        "correct: faults declared with the correct mode; delay: median delay of the correct break declarations; "
        "false alarms: per hour of operation with the fault modes armed; recovered: breaks after which the "
        "position error returned to the band of the fault free runs. A2 and A3 ran the break runs only, and B3T "
        "a subset fixed in advance. The differences in platforms lost between P and each of B0, B2, B3 and A2 "
        "remain after the Holm correction ($p\\le0.007$).", "tab:results",
        "& \\multicolumn{2}{c}{Lost} & Correct & Delay & False alarms & Recovered \\\\\n"
        "\\cmidrule(lr){2-3}\nMethod & upper & lower & & (ms) & (per h) & breaks",
        body, "lrrrrrr", star=True))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parameters()
    methods()
    results()
    print("tables written to", OUT.relative_to(ROOT), sorted(p.name for p in OUT.glob("*.tex")))


if __name__ == "__main__":
    main()

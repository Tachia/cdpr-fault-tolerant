"""Set the reports of several replications side by side (REPLICATION.md, Section 5).

Reads replication/runs/<label>/report.json, and sample_rows.jsonl where present, of every folder
given, and writes a table of the steps per machine, the agreement of the simulated jobs between
every pair of machines, and the overall verdict of the protocol.

  python scripts/compare_replications.py replication/runs/reference replication/runs/kaggle ...
         [--out replication/comparison.md]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from replicate import compare_row, summarise  # noqa: E402


def load(folder: Path):
    report = json.loads((folder / "report.json").read_text())
    rows = {}
    path = folder / "sample_rows.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[(d["method"], d["run"])] = d
    return report, rows


def cell(step: dict | None) -> str:
    if step is None:
        return "not run"
    if "identical" in step and "jobs" in step:
        return (f"{step['verdict']}: {step['identical']}/{step['jobs']} identical, {step['numeric']} numeric, "
                f"{step['outcome']} outcome")
    if step.get("files"):
        same = sum(1 for v in step["files"].values() if v.startswith("IDENTICAL"))
        return f"{step['verdict']} ({same}/{len(step['files'])})"
    return step["verdict"] + (f" ({step['summary']})" if step.get("summary") else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    runs = [(f, *load(f)) for f in a.folders]
    out = ["# Comparison of replications", ""]

    out += ["## Machines", "", "| Label | Tier | Tree | Machine | Python | Packages off the reference |",
            "|---|---|---|---|---|---|"]
    for _, r, _ in runs:
        env = r["environment"]
        off = [f"{k} {v['installed']}" for k, v in env["packages"].items()
               if (v["installed"] or "").split("+")[0] != v["reference"]]
        out.append(f"| {r['label']} | {r['tier']} | {env.get('tree', env['commit'])[:12]} | {env.get('cpu', '?')}, {env['cores']} cores "
                   f"| {env['python']} | {', '.join(off) or 'none'} |")
    # the tree identifies the code and data; two commits with the same tree differ only in their message
    commits = {r["environment"].get("tree", r["environment"]["commit"]) for _, r, _ in runs}
    if len(commits) > 1:
        out += ["", "**The replications used different versions of the code; they are not comparable.**"]

    names = []
    for _, r, _ in runs:
        names += [n for n in r["steps"] if n not in names]
    out += ["", "## Steps against the stored results", "", "| Step | " + " | ".join(r["label"] for _, r, _ in runs) + " |",
            "|---|" + "---|" * len(runs)]
    for n in names:
        out.append(f"| {n} | " + " | ".join(cell(r["steps"].get(n)) for _, r, _ in runs) + " |")
    out.append("| **overall** | " + " | ".join(f"**{r.get('overall', 'unfinished')}**" for _, r, _ in runs) + " |")

    out += ["", "## Simulated jobs between machines", "",
            "Every pair of machines, on the jobs of the sample that both ran: outcome values must match exactly; "
            "floating point values are compared by their relative difference.", "",
            "| Pair | Jobs | Identical | Numeric only (largest relative difference) | Outcome differences |",
            "|---|---|---|---|---|"]
    disagreements = []
    for (_, ra, xa), (_, rb, xb) in itertools.combinations(runs, 2):
        common = sorted(set(xa) & set(xb))
        if not common:
            continue
        s = summarise({k: compare_row(xa[k], xb[k]) for k in common})
        out.append(f"| {ra['label']} / {rb['label']} | {s['jobs']} | {s['identical']} | {s['numeric']} "
                   f"({s['largest_relative_difference']:.1e}) | {s['outcome']} |")
        disagreements += [(ra["label"], rb["label"], d) for d in s["outcome_differences"]]
    if disagreements:
        out += ["", "Jobs whose outcomes differ between machines:"]
        out += [f"- {a_} / {b_}: {d['method']} run {d['run']}: {', '.join(d['values'])}" for a_, b_, d in disagreements]

    verdicts = [r.get("overall") for _, r, _ in runs]
    out += ["", "## Verdict of the protocol", ""]
    if len(commits) > 1 or None in verdicts or any(v.startswith("UNFINISHED") for v in verdicts if v):
        out.append("INCOMPLETE: every replication must finish on the same commit.")
    elif any(v.startswith("NOT REPRODUCED") for v in verdicts):
        out.append("NOT REPRODUCED on at least one machine; see the steps above.")
    elif disagreements or any("DIFFERENCES TO REVIEW" in v for v in verdicts):
        out.append("REPRODUCED WITH DIFFERENCES TO REVIEW: outcome differences are listed above; REPLICATION.md, "
                   "Section 6, says how to judge them.")
    elif all(v == "REPRODUCED IDENTICALLY" for v in verdicts):
        out.append("REPRODUCED IDENTICALLY on every machine.")
    else:
        out.append("REPRODUCED on every machine: the same outcomes everywhere, with differences only in the last "
                   "digits of floating point values.")
    text = "\n".join(out) + "\n"
    if a.out:
        a.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()

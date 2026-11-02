"""Replication of the study on another machine (REPLICATION.md): run the checks of one tier and
write a single report that scripts/compare_replications.py can set beside the reports of other
machines.

  python scripts/replicate.py --label NAME --tier 1|2|3 [--jobs N]

Tier 1, the stored results (about 10 min, plus about 20 min to compile the solvers on first use):
  the automated tests; the statistics recomputed from the stored rows of the campaign; the
  workspace slices and the tables; the figures drawn from data, compared byte for byte with the
  committed files; and every checked number of the manuscript.
Tier 2, tier 1 and a new simulation of part of the campaign (about 5 core hours):
  the reviewer check of the frozen controller (8 calibration breaks); the 28 jobs whose rows and
  traces the repository keeps; and the stratified sample fixed in sample_jobs() below, 394 jobs,
  compared row by row with the stored campaign.
Tier 3, tier 2 and the whole campaign (5,212 jobs, about 64 core hours):
  the campaign is simulated again into results/phase5_work/ (resumable), condensed and analysed,
  and the tables and the checked numbers of the manuscript are recomputed from the new results;
  the committed files are restored afterwards.

Everything goes to replication/runs/<label>/: report.json (machine readable), report.md, the
console log of every step and the simulated rows. The working tree must be clean at the start and
is checked again at the end, so that a report can only describe the committed code. Tiers 2 and 3
resume where they stopped when started again with the same label.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

PROTOCOL = ROOT / "configs" / "evaluation_phase5.yaml"
STORED_ROWS = ROOT / "results" / "phase5" / "rows.jsonl.gz"
FIGURES_FROM_DATA = ["fig01", "fig02", "fig04", "fig05", "fig06", "fig07", "fig08", "fig09", "fig10", "fig11"]
VERDICT_LINE = re.compile(r"^(\S+): (IDENTICAL|AGREES WITHIN TOLERANCE|DIFFERENT)(.*)$")
TIMING = ("seconds",)


# ------------------------------------------------------------------------------------ helpers
def sh(cmd, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, **kw)


def git_dirty() -> list[str]:
    return [line for line in sh(["git", "status", "--porcelain"]).stdout.splitlines() if line.strip()]


def head_bytes(path: str) -> bytes:
    return subprocess.run(["git", "show", f"HEAD:{path}"], cwd=ROOT, capture_output=True, check=True).stdout


def environment() -> dict:
    from importlib import metadata
    env = {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
           "cores": os.cpu_count(), "commit": sh(["git", "rev-parse", "HEAD"]).stdout.strip(),
           "tree": sh(["git", "rev-parse", "HEAD^{tree}"]).stdout.strip(),
           "describe": sh(["git", "describe", "--tags", "--always"]).stdout.strip()}
    try:
        info = Path("/proc/cpuinfo").read_text()
        env["cpu"] = re.search(r"model name\s*:\s*(.*)", info).group(1).strip()
        flags = re.search(r"flags\s*:\s*(.*)", info).group(1).split()
        env["cpu_flags"] = [f for f in ("sse4_2", "avx", "avx2", "fma", "avx512f") if f in flags]
        env["memory_gb"] = round(int(re.search(r"MemTotal:\s*(\d+)", Path("/proc/meminfo").read_text()).group(1))
                                 / 2 ** 20, 1)
    except (OSError, AttributeError):
        env["cpu"] = platform.processor()
    packages = {}
    for name in ("requirements.txt", "requirements-learning.txt"):
        for line in (ROOT / name).read_text().splitlines():
            line = line.split("#")[0].strip()
            if "==" in line:
                pkg, ref = (s.strip() for s in line.split("=="))
                try:
                    packages[pkg] = {"installed": metadata.version(pkg), "reference": ref}
                except metadata.PackageNotFoundError:
                    packages[pkg] = {"installed": None, "reference": ref}
    env["packages"] = packages
    gcc = shutil.which("gcc")
    env["gcc"] = sh([gcc, "--version"]).stdout.splitlines()[0] if gcc else None
    env["pdflatex"] = bool(shutil.which("pdflatex"))
    env["thread_variables"] = {k: os.environ[k] for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                          "MKL_NUM_THREADS", "OPENBLAS_CORETYPE") if k in os.environ}
    try:
        import numpy as np
        buf = io.StringIO()
        with redirect_stdout(buf):
            np.show_runtime()
        env["numpy_runtime"] = buf.getvalue()
    except Exception as e:                  # noqa: BLE001 - the record must not stop the run
        env["numpy_runtime"] = f"not available: {e}"
    return env


class Run:
    """One replication: the output folder, the report and the console logs."""

    def __init__(self, label: str, tier: int, jobs: int):
        self.dir = ROOT / "replication" / "runs" / label
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "report.json"
        old = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.report = {"label": label, "tier": tier, "jobs": jobs, "steps": old.get("steps", {}),
                       "started_utc": old.get("started_utc", f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M}")}

    def save(self):
        self.path.write_text(json.dumps(self.report, indent=1, default=str))

    def step(self, name: str, result: dict):
        self.report["steps"][name] = result
        self.save()
        print(f"  {name}: {result['verdict']}", flush=True)

    def done(self, name: str) -> bool:
        s = self.report["steps"].get(name)
        return bool(s) and s["verdict"] not in ("FAILED", "INCOMPLETE")


def reproduce(run: Run, name: str, args: list[str], jobs: int):
    """Run one target of reproduce.py and keep its report and the verdict of every file."""
    if run.done(name):
        print(f"  {name}: done before, kept", flush=True)
        return
    t0 = time.time()
    p = sh([sys.executable, "reproduce.py", *args, "--jobs", str(jobs)])
    (run.dir / f"{name}.log").write_text(p.stdout + p.stderr)
    files, report = {}, None
    m = re.search(r"report: (\S+)", p.stdout)
    if m:
        report = ROOT / m.group(1)
        shutil.copy(report, run.dir / f"{name}_report.md")
        for line in report.read_text().splitlines():
            v = VERDICT_LINE.match(line)
            if v:
                files[v.group(1)] = v.group(2) + v.group(3)
    if p.returncode != 0 and not files:
        verdict = "FAILED"
    elif any(f.startswith("DIFFERENT") for f in files.values()):
        verdict = "DIFFERENT"
    elif any(f.startswith("AGREES") for f in files.values()):
        verdict = "AGREES WITHIN TOLERANCE"
    else:
        verdict = "IDENTICAL" if p.returncode == 0 else "FAILED"
    run.step(name, {"verdict": verdict, "command": "reproduce.py " + " ".join(args), "exit_status": p.returncode,
                    "files": files, "minutes": round((time.time() - t0) / 60, 1)})


# ------------------------------------------------------------------------------------ tier 1
def tests(run: Run):
    if run.done("tests"):
        return
    t0 = time.time()
    p = sh([sys.executable, "-m", "pytest", "-q", "tests"])
    (run.dir / "tests.log").write_text(p.stdout + p.stderr)
    tail = [line for line in p.stdout.splitlines() if " passed" in line or " failed" in line or " error" in line]
    run.step("tests", {"verdict": "PASSED" if p.returncode == 0 else "FAILED", "summary": tail[-1] if tail else "",
                       "minutes": round((time.time() - t0) / 60, 1)})


def figures(run: Run):
    """Redraw the figures drawn from data and compare their bytes with the committed files.

    The block diagram (figure 3) is compiled by pdflatex, whose output depends on the TeX
    installation, so it is not part of this comparison."""
    if run.done("figures"):
        return
    t0 = time.time()
    p = sh([sys.executable, "scripts/paper_figures.py", *FIGURES_FROM_DATA])
    (run.dir / "figures.log").write_text(p.stdout + p.stderr)
    files = {}
    try:
        for f in sorted((ROOT / "paper" / "figures").glob("fig*")):
            if f.name.startswith("fig03"):
                continue
            rel = str(f.relative_to(ROOT))
            same = hashlib.sha256(f.read_bytes()).digest() == hashlib.sha256(head_bytes(rel)).digest()
            files[rel] = "IDENTICAL" if same else "DIFFERENT (bytes)"
    finally:
        sh(["git", "checkout", "--", "paper/figures"])
    if p.returncode != 0:
        verdict = "FAILED"
    elif all(v == "IDENTICAL" for v in files.values()):
        verdict = "IDENTICAL"
    elif all(v == "IDENTICAL" for k, v in files.items() if k.endswith(".pdf")):
        verdict = "PDF IDENTICAL, PNG BYTES DIFFER"
    else:
        verdict = "DIFFERENT"
    run.step("figures", {"verdict": verdict, "files": files, "minutes": round((time.time() - t0) / 60, 1),
                         "note": "figure 3 (TikZ) left out: its bytes depend on the TeX installation"})


# ------------------------------------------------------------------------------------ rows
def flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = d
    return out


def is_float(x) -> bool:
    return isinstance(x, float) and not isinstance(x, bool)


def compare_row(ref: dict, new: dict) -> dict:
    """Outcome values (integers, booleans, strings, None) must match exactly; floating point values
    are compared by their relative difference. Timing values are left out."""
    a, b = flatten(ref), flatten(new)
    keys = [k for k in a if not any(t in k for t in TIMING)]
    res = {"outcome": [], "max_rel": 0.0, "worst": None, "missing": [k for k in keys if k not in b],
           "delay_shift_ms": 0.0}
    for k in keys:
        if k not in b:
            continue
        x, y = a[k], b[k]
        if (is_float(x) or is_float(y)) and isinstance(x, (int, float)) and isinstance(y, (int, float)) \
                and not isinstance(x, bool) and not isinstance(y, bool):
            if x == y or (math.isnan(x) and math.isnan(y)):
                continue
            rel = abs(x - y) / max(abs(x), abs(y), 1e-12)
            if rel > res["max_rel"]:
                res["max_rel"], res["worst"] = rel, k
            if k == "delay_s":
                res["delay_shift_ms"] = round(1e3 * abs(x - y), 3)
        elif x != y:
            res["outcome"].append(k)
    res["class"] = "outcome" if res["outcome"] else ("numeric" if res["max_rel"] > 0 else "identical")
    return res


def stored_rows() -> dict:
    with gzip.open(STORED_ROWS, "rt") as f:
        return {(d["method"], d["run"]): d for d in map(json.loads, f)}


def summarise(comparisons: dict) -> dict:
    """comparisons: {(method, run): compare_row result}."""
    classes = {"identical": 0, "numeric": 0, "outcome": 0}
    for c in comparisons.values():
        classes[c["class"]] += 1
    numeric = [c for c in comparisons.values() if c["class"] == "numeric"]
    worst = max(numeric, key=lambda c: c["max_rel"], default=None)
    return {"jobs": len(comparisons), **classes,
            "largest_relative_difference": worst["max_rel"] if worst else 0.0,
            "largest_in": worst["worst"] if worst else None,
            "delays_shifted": sum(1 for c in comparisons.values() if c["delay_shift_ms"] > 0),
            "outcome_differences": sorted(
                [{"method": m, "run": r, "values": c["outcome"]} for (m, r), c in comparisons.items()
                 if c["class"] == "outcome"], key=lambda d: (d["run"], d["method"]))}


def verdict_of(summary: dict) -> str:
    if summary["outcome"]:
        return "OUTCOMES DIFFER"
    return "NUMERIC DIFFERENCES ONLY" if summary["numeric"] else "IDENTICAL"


# ------------------------------------------------------------------------------------ tier 2
def sample_jobs(protocol):
    """The stratified sample, fixed before any replication: for every path, fault type and cable
    the fault run whose repetition equals the cable index (48 runs), and the first five fault free
    runs of each path (10 runs), each with every method that the campaign ran on it."""
    from cdpr_ft.eval.campaign import jobs_for, run_list
    runs = run_list(protocol)
    chosen = [r["run"] for r in runs if r["group"] == "fault" and r["rep"] == r["fault"][1]]
    for traj in ("T1", "T2"):
        chosen += [r["run"] for r in runs if r["group"] == "fault_free" and r["trajectory"] == traj][:5]
    keep = set(chosen)
    return [(m, r) for m, r in jobs_for(protocol, runs) if r in keep], {r["run"]: r for r in runs}


def sample(run: Run, jobs: int):
    from cdpr_ft.eval.campaign import init_worker, load_protocol, run_job, write_row
    if run.done("sample"):
        return
    t0 = time.time()
    protocol, phash = load_protocol(PROTOCOL)
    pairs, by_run = sample_jobs(protocol)
    path = run.dir / "sample_rows.jsonl"
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:   # a line cut short when the machine stopped
                continue
            done[(d["method"], d["run"])] = d
    todo = [(m, by_run[r], None) for m, r in pairs if (m, r) not in done]
    print(f"  sample: {len(pairs)} jobs, {len(todo)} to do", flush=True)
    if todo:
        with mp.get_context("fork").Pool(jobs, initializer=init_worker, initargs=(str(PROTOCOL), 1, True),
                                         maxtasksperchild=50) as pool, path.open("a") as f:
            for k, row in enumerate(pool.imap_unordered(run_job, todo, chunksize=1)):
                row["protocol_sha256_16"] = phash
                write_row(f, row)
                done[(row["method"], row["run"])] = row
                if (k + 1) % 10 == 0 or k + 1 == len(todo):
                    print(f"    {k + 1}/{len(todo)} jobs, {(time.time() - t0) / 60:.1f} min", flush=True)
    ref = stored_rows()
    comps = {key: compare_row(ref[key], row) for key, row in done.items() if key in ref}
    summ = summarise(comps)
    run.step("sample", {"verdict": verdict_of(summ), **summ, "minutes": round((time.time() - t0) / 60, 1),
                        "note": "recovery times are added to the stored rows from the traces when the campaign is "
                                "condensed; the sample keeps no traces, so they are not compared here"})


# ------------------------------------------------------------------------------------ tier 3
def campaign(run: Run, jobs: int, max_jobs: int = 0):
    """max_jobs > 0 stops after that many new jobs (for machines whose sessions end after a fixed
    time); the step is then INCOMPLETE and continues where it stopped when the tier is run again."""
    if run.done("campaign"):
        return
    t0 = time.time()
    cmd = [sys.executable, "scripts/phase5_run.py", "--processes", str(jobs)]
    if max_jobs > 0:
        cmd += ["--max-jobs", str(max_jobs)]
    with (run.dir / "campaign.log").open("a") as log:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        run.step("campaign", {"verdict": "FAILED", "exit_status": rc})
        return
    from cdpr_ft.eval.campaign import jobs_for, load_protocol, run_list
    protocol, _ = load_protocol(PROTOCOL)
    total = len(jobs_for(protocol, run_list(protocol)))
    have = set()
    work_rows = ROOT / "results" / "phase5_work" / "rows.jsonl"
    for line in (work_rows.read_text().splitlines() if work_rows.exists() else []):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        have.add((d["method"], d["run"]))
    if len(have) < total:
        run.step("campaign", {"verdict": "INCOMPLETE", "summary": f"{len(have)} of {total} jobs simulated",
                              "minutes": round((time.time() - t0) / 60, 1)})
        return
    tracked = ["results/phase5", "paper/tables", "paper/manuscript/number_check.txt"]
    ref_rows = stored_rows()
    ref_analysis = run.dir / "analysis_committed.json"
    ref_analysis.write_bytes(head_bytes("results/phase5/analysis.json"))
    try:
        steps = {}
        for script in ("phase5_condense.py", "phase5_analyse.py", "paper_tables.py", "paper_check_numbers.py"):
            p = sh([sys.executable, f"scripts/{script}"])
            (run.dir / f"campaign_{script[:-3]}.log").write_text(p.stdout + p.stderr)
            steps[script] = p.returncode
        with gzip.open(STORED_ROWS, "rt") as f:
            new_rows = {(d["method"], d["run"]): d for d in map(json.loads, f)}
        comps = {key: compare_row(ref_rows[key], row) for key, row in new_rows.items() if key in ref_rows}
        summ = summarise(comps)
        from compare_results import compare, verdict
        analysis = verdict(compare(ref_analysis, ROOT / "results" / "phase5" / "analysis.json"))
        tables = {t: ("IDENTICAL" if (ROOT / "paper" / "tables" / f"{t}.tex").read_bytes()
                      == head_bytes(f"paper/tables/{t}.tex") else "DIFFERENT")
                  for t in ("tab_results",)}
        numbers = (ROOT / "paper" / "manuscript" / "number_check.txt").read_text()
        shutil.copy(ROOT / "paper" / "manuscript" / "number_check.txt", run.dir / "number_check_campaign.txt")
        failed_numbers = [line[6:] for line in numbers.splitlines() if line.startswith("FAIL")]
    finally:
        sh(["git", "checkout", "--", *tracked])
    v = verdict_of(summ)
    if failed_numbers or any(steps.values()):
        v = "PAPER NUMBERS CHANGE" if failed_numbers else "FAILED"
    run.step("campaign", {"verdict": v, **summ, "analysis": analysis, "tables": tables, "scripts": steps,
                          "numbers_failed": failed_numbers, "minutes": round((time.time() - t0) / 60, 1)})


# ------------------------------------------------------------------------------------ report
def write_markdown(run: Run):
    r, env = run.report, run.report["environment"]
    lines = [f"# Replication report: {r['label']}", "",
             f"Tier {r['tier']}; commit {env['commit'][:12]} ({env['describe']}), tree {env['tree'][:12]}; started {r['started_utc']} UTC; "
             f"finished {r.get('finished_utc', '')} UTC", "",
             f"Machine: {env.get('cpu')}, {env['cores']} cores, {env.get('memory_gb', '?')} GB; {env['platform']}; "
             f"Python {env['python']}; {env['gcc']}", ""]
    off = {k: v for k, v in env["packages"].items() if (v["installed"] or "").split("+")[0] != v["reference"]}
    lines.append("Packages: all as in the requirements files." if not off else
                 "Packages that differ from the requirements files: "
                 + ", ".join(f"{k} {v['installed']} (reference {v['reference']})" for k, v in off.items()))
    lines += ["", "| Step | Verdict | Details |", "|---|---|---|"]
    for name, s in r["steps"].items():
        if "jobs" in s and "identical" in s:
            det = (f"{s['jobs']} jobs: {s['identical']} identical, {s['numeric']} with numeric differences only "
                   f"(largest relative difference {s['largest_relative_difference']:.1e}), {s['outcome']} with "
                   f"different outcomes")
        elif "files" in s:
            n = len(s["files"])
            same = sum(1 for v in s["files"].values() if v.startswith("IDENTICAL"))
            det = f"{same} of {n} files identical" if n else s.get("summary", "")
        else:
            det = s.get("summary", "")
        lines.append(f"| {name} | {s['verdict']} | {det} |")
    lines += ["", f"Overall: **{r['overall']}**", ""]
    for name, s in r["steps"].items():
        if s.get("outcome_differences"):
            lines.append(f"Outcome differences in {name}:")
            lines += [f"- {d['method']} run {d['run']}: {', '.join(d['values'])}" for d in s["outcome_differences"]]
            lines.append("")
        if s.get("numbers_failed"):
            lines.append("Passages of the manuscript whose numbers change with the new campaign:")
            lines += [f"- {x}" for x in s["numbers_failed"]]
            lines.append("")
    (run.dir / "report.md").write_text("\n".join(lines))


def overall(steps: dict) -> str:
    v = [s["verdict"] for s in steps.values()]
    if "INCOMPLETE" in v:
        return "UNFINISHED (run the same command again to continue)"
    if any(x in ("FAILED", "PAPER NUMBERS CHANGE") for x in v):
        return "NOT REPRODUCED (see the steps)"
    if any(x in ("DIFFERENT", "OUTCOMES DIFFER") for x in v):
        return "REPRODUCED WITH DIFFERENCES TO REVIEW"
    if all(x in ("IDENTICAL", "PASSED") for x in v):
        return "REPRODUCED IDENTICALLY"
    return "REPRODUCED WITHIN NUMERICAL PRECISION"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, help="name of this machine or service, e.g. kaggle")
    ap.add_argument("--tier", type=int, choices=(1, 2, 3), default=1)
    ap.add_argument("--jobs", type=int, default=min(os.cpu_count() or 2, 8), help="worker processes")
    ap.add_argument("--max-campaign-jobs", type=int, default=0,
                    help="tier 3: stop the campaign after this many new jobs; run again to continue")
    a = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", a.label):
        sys.exit("the label may contain letters, digits, '.', '_' and '-' only")
    dirty = git_dirty()
    if dirty:
        sys.exit("the working tree is not clean; replicate the committed code only:\n  " + "\n  ".join(dirty))
    run = Run(a.label, a.tier, a.jobs)
    run.report["environment"] = environment()
    run.save()
    print(f"replication '{a.label}', tier {a.tier}, {a.jobs} processes, commit {run.report['environment']['commit'][:12]}",
          flush=True)
    tests(run)
    reproduce(run, "statistics", ["phase5", "--only", "analysis"], a.jobs)
    reproduce(run, "workspace_and_tables", ["phase6", "--only", "workspace_slices,tables"], a.jobs)
    figures(run)
    reproduce(run, "manuscript_numbers", ["phase7"], a.jobs)
    if a.tier >= 2:
        reproduce(run, "reviewer_check", ["quick"], a.jobs)
        reproduce(run, "kept_jobs", ["phase5", "--only", "campaign_subset"], a.jobs)
        sample(run, a.jobs)
    if a.tier >= 3:
        campaign(run, a.jobs, a.max_campaign_jobs)
    run.report["clean_after"] = not git_dirty()
    if not run.report["clean_after"]:
        run.report["steps"]["working_tree"] = {"verdict": "FAILED", "summary": "tracked files changed: "
                                               + "; ".join(git_dirty())}
    run.report["finished_utc"] = f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M}"
    run.report["overall"] = overall(run.report["steps"])
    run.save()
    write_markdown(run)
    print(f"\n{run.report['overall']}\nreport: {(run.dir / 'report.md').relative_to(ROOT)}")


if __name__ == "__main__":
    main()

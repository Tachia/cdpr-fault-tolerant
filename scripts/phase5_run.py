"""Phase 5: the evaluation campaign (configs/evaluation_phase5.yaml).

Runs every (method, run) pair of the protocol, seed by seed, in worker processes, and appends one
metrics row per pair to results/phase5_work/rows.jsonl; the trace of each run (estimator rate)
goes to results/phase5_work/traces/<method>/run<run>.npz. The script is resumable: pairs that
already have a row are skipped, so it can be stopped and started again at any time.

--pilot runs the first runs of the run list on the pilot seeds of the protocol (900000 and up,
never used for a result) into results/phase5_pilot/, to time the methods and check the logs.
Usage: python scripts/phase5_run.py [--processes 2] [--pilot N] [--methods P,A1] [--max-jobs K]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdpr_ft.eval.campaign import check_frozen, init_worker, jobs_for, load_protocol, run_job, run_list, write_row  # noqa: E402
from cdpr_ft.provenance import stamp  # noqa: E402

PROTOCOL = ROOT / "configs" / "evaluation_phase5.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processes", type=int, default=2)
    ap.add_argument("--pilot", type=int, default=0, help="number of runs of the pilot (pilot seeds)")
    ap.add_argument("--methods", default="")
    ap.add_argument("--max-jobs", type=int, default=0)
    a = ap.parse_args()
    protocol, phash = load_protocol(PROTOCOL)
    if not a.pilot:
        if not (ROOT / "results" / "phase5_freeze.json").exists():
            sys.exit("run scripts/phase5_freeze.py first")
        check_frozen(protocol)
    if a.pilot:
        protocol["runs"]["seed_base"] = int(protocol["runs"]["pilot_seed_base"])
        work = ROOT / "results" / "phase5_pilot"
    else:
        work = ROOT / "results" / "phase5_work"
    runs = run_list(protocol)
    if a.pilot:
        # the first fault runs of every type and path, and one fault-free run per path
        picked, seen = [], {}
        for r in runs:
            key = (r["trajectory"], r["group"], r["fault"][0] if r["fault"] else None)
            if seen.get(key, 0) < (a.pilot if r["fault"] else 1):
                picked.append(r)
                seen[key] = seen.get(key, 0) + 1
        runs = picked
    by_run = {r["run"]: r for r in runs}
    jobs = jobs_for(protocol, runs)
    if a.methods:
        keep = set(a.methods.split(","))
        jobs = [j for j in jobs if j[0] in keep]
    work.mkdir(parents=True, exist_ok=True)
    rows_path = work / "rows.jsonl"
    done = set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            if line.strip():
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:      # a line cut short when the machine stopped
                    continue
                done.add((d["method"], d["run"]))
    todo = [(m, by_run[r], str(work / "traces" / m / f"run{r:05d}.npz")) for m, r in jobs if (m, r) not in done]
    if a.max_jobs:
        todo = todo[:a.max_jobs]
    meta_path = work / "meta.json"
    history = json.loads(meta_path.read_text()) if meta_path.exists() else []
    history.append({"protocol_sha256_16": phash, "jobs_total": len(jobs), "jobs_todo": len(todo),
                    "processes": a.processes, "pilot": a.pilot, "provenance": stamp()})
    meta_path.write_text(json.dumps(history, indent=1))
    print(f"{len(jobs)} jobs, {len(todo)} to do", flush=True)
    if not todo:
        return
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(a.processes, initializer=init_worker, initargs=(str(PROTOCOL), 1, not a.pilot),
                  maxtasksperchild=50) as pool, \
            rows_path.open("a") as f:
        for k, row in enumerate(pool.imap_unordered(run_job, todo, chunksize=1)):
            row["protocol_sha256_16"] = phash
            write_row(f, row)
            if (k + 1) % 10 == 0 or k + 1 == len(todo):
                el = time.time() - t0
                print(f"{k + 1}/{len(todo)} jobs, {el / 60:.1f} min, {el / (k + 1):.1f} s per job", flush=True)


if __name__ == "__main__":
    main()

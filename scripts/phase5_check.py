"""Phase 5: simulate a fixed subset of the campaign again and write its rows and traces.

The campaign itself takes a day on two cores. This script reruns, for every method, the runs of
a subset fixed here (a break of cable 1 on T1, a loss of effectiveness of winch 5 on T2 and a
bias of winch 8 on T1, all of repetition 0, and the first fault-free run of T2), exactly as the
campaign ran them, and writes
  results/phase5/subset_rows.json      the rows of these jobs (timings included; the comparison of
                                       reproduce.py leaves out every key with "seconds")
  results/phase5/traces/<method>/run<run>.npz   their traces, which are among the traces kept
reproduce.py phase5 compares both with the files that scripts/phase5_condense.py stored from the
campaign; identical rows and traces show that the stored campaign is what this code gives.
Usage: python scripts/phase5_check.py [processes]
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdpr_ft.eval.campaign import init_worker, jobs_for, load_protocol, run_job, run_list  # noqa: E402

PROTOCOL = ROOT / "configs" / "evaluation_phase5.yaml"
OUT = ROOT / "results" / "phase5"
PICK = [("T1", "fault", "break", 0), ("T2", "fault", "loe", 4), ("T1", "fault", "bias", 7), ("T2", "fault_free", None, None)]


def subset(protocol):
    """(method, run) jobs of the subset, in the order of the campaign."""
    runs = run_list(protocol)
    chosen = set()
    for traj, group, kind, cable in PICK:
        for r in runs:
            if r["trajectory"] == traj and r["group"] == group and r["rep"] == 0 and \
                    (group == "fault_free" or (r["fault"][0] == kind and r["fault"][1] == cable)):
                chosen.add(r["run"])
                break
    return [(m, r) for m, r in jobs_for(protocol, runs) if r in chosen], {r["run"]: r for r in runs}


def main(processes=2):
    protocol, phash = load_protocol(PROTOCOL)
    jobs, by_run = subset(protocol)
    todo = [(m, by_run[r], str(OUT / "traces" / m / f"run{r:05d}.npz")) for m, r in jobs]
    with mp.get_context("fork").Pool(processes, initializer=init_worker, initargs=(str(PROTOCOL), 1, True)) as pool:
        rows = pool.map(run_job, todo, chunksize=1)
    for row in rows:
        row["protocol_sha256_16"] = phash
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "subset_rows.json").write_text(json.dumps(sorted(rows, key=lambda d: (d["run"], d["method"])), indent=1,
                                                     default=float))
    print(f"{len(rows)} jobs")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:2]))

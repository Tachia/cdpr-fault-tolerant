"""Post-review audit of the runs in which the platform was lost (no new result).

Every (method, run) pair whose stored row says the platform was lost is simulated again with the
code and settings of the campaign. Two things are recorded that the campaign did not keep:
  - which attitude angle ended the run: the campaign stops a run when any of the three Euler
    angles (roll, pitch, yaw) exceeds 30 degrees, so a run could in principle end on yaw alone;
  - every MPC solve: its time, IPOPT return status, iterations and primal infeasibility, so that
    unsuccessful solves can be placed in time against the onset, the declaration and the loss.
The recording only reads values; the row of each run must come out identical to the stored row
(every value except timing), which the script checks.

With --failed, the runs that were not lost but contained an unsuccessful solve are replayed in the
same way, so that every unsuccessful solve of the campaign has a recorded status.

Output: results/audit/lost_runs.jsonl (one record per run), results/audit/lost_runs.json (summary);
        with --failed: results/audit/failed_solve_runs.jsonl and failed_solve_runs.json
Usage: python scripts/audit_lost_runs.py [processes] [--failed]       (resumable)
"""
from __future__ import annotations

import gzip
import json
import multiprocessing as mp
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cdpr_ft.eval import campaign  # noqa: E402
from cdpr_ft.eval.campaign import init_worker, load_protocol, run_job, run_list  # noqa: E402
from replicate import compare_row  # noqa: E402

PROTOCOL = ROOT / "configs" / "evaluation_phase5.yaml"
OUT = ROOT / "results" / "audit"
ANGLES = ("roll", "pitch", "yaw")
_REC: dict = {}


def _instrument():
    """Wrap the MPC solve and the evaluation to record what the campaign did not keep."""
    from cdpr_ft.control.mpc import ScenarioMPC
    solve = ScenarioMPC.solve

    def recorded_solve(self, slots, *args, **kwargs):
        out = solve(self, slots, *args, **kwargs)
        st = self.solvers[len(slots)]["solver"].stats()
        inf = st.get("iterations", {}).get("inf_pr", [])
        _REC["solves"].append({"status": st["return_status"], "success": bool(st["success"]),
                               "iterations": int(st.get("iter_count", -1)),
                               "inf_pr": float(inf[-1]) if len(inf) else None, "scenarios": len(slots)})
        return out
    ScenarioMPC.solve = recorded_solve

    evaluate = campaign.evaluate

    def recorded_evaluate(spec, log, cl, ctrl, protocol, trace_path=None):
        q = np.asarray(log["q"])
        _REC["t_end"] = float(log["t"][-1])
        _REC["angles_end_deg"] = np.rad2deg(np.abs(q[-1, 3:])).tolist()
        _REC["t_mpc"] = list(map(float, cl.get("t_mpc", [])))
        _REC["t_q"], _REC["q_ang"] = np.asarray(log["t"]), np.rad2deg(np.abs(q[:, 3:]))
        return evaluate(spec, log, cl, ctrl, protocol, trace_path)
    campaign.evaluate = recorded_evaluate


def worker_init(protocol_path):
    init_worker(protocol_path, 1, True)
    _instrument()


def audit(job):
    method, r = job
    _REC.clear()
    _REC["solves"] = []
    t0 = time.time()
    row = run_job((method, r, None))
    solves, t_mpc = _REC["solves"], _REC["t_mpc"]
    # the controller logs one time per regular MPC sample; a solve made inside a declaration (online
    # planner) has no entry there, so times are attached only when the counts agree
    times = t_mpc if len(t_mpc) == len(solves) else [None] * len(solves)
    failed = [dict(s, t=t) for s, t in zip(solves, times) if not s["success"]]
    tq, qa = _REC["t_q"], _REC["q_ang"]

    def attitude_at(t):
        if t is None:
            return None
        k = min(int(np.searchsorted(tq, t)), len(tq) - 1)
        return float(qa[k].max())
    ang = _REC["angles_end_deg"]
    onset = r["fault"][2] if r["fault"] else None
    rec = {"method": method, "run": r["run"], "group": r["group"], "fault": r["fault"], "trajectory": r["trajectory"],
           "lost": row["lost"], "stopped": row["stopped"], "t_end": _REC["t_end"], "onset": onset,
           "t_declare": row.get("t_declare"), "angles_end_deg": dict(zip(ANGLES, ang)),
           "trigger": ANGLES[int(np.argmax(ang))],
           "solves": len(solves), "failed_solves": len(failed),
           "failures": [dict(f, attitude_deg=attitude_at(f["t"])) for f in failed],
           "first_failure_t": failed[0]["t"] if failed else None,
           "seconds": time.time() - t0}
    return rec, row


def replay(keys, path, stored, by_run, processes):
    """Simulate the (method, run) pairs again with the recording; resumable."""
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[(d["method"], d["run"])] = d
    todo = [(m, by_run[r]) for m, r in keys if (m, r) not in done]
    print(f"{len(keys)} runs, {len(todo)} to do", flush=True)
    t0 = time.time()
    if todo:
        with mp.get_context("fork").Pool(processes, initializer=worker_init, initargs=(str(PROTOCOL),),
                                         maxtasksperchild=50) as pool, path.open("a") as f:
            for k, (rec, row) in enumerate(pool.imap_unordered(audit, todo, chunksize=1)):
                cmp = compare_row(stored[(rec["method"], rec["run"])], row)
                rec["row_check"] = cmp["class"] if not cmp["outcome"] else "outcome: " + ", ".join(cmp["outcome"])
                f.write(json.dumps(rec) + "\n")
                f.flush()
                done[(rec["method"], rec["run"])] = rec
                if (k + 1) % 10 == 0 or k + 1 == len(todo):
                    print(f"  {k + 1}/{len(todo)} runs, {(time.time() - t0) / 60:.1f} min", flush=True)
    return [done[k] for k in keys]


def solve_records(recs, stored):
    """Unsuccessful solves of the replayed runs; their count must equal the count the campaign stored."""
    fails = [f for r in recs for f in r["failures"]]
    inf = [f["inf_pr"] for f in fails if f["inf_pr"] is not None]
    return {"failed_solves": len(fails),
            "count_equals_stored": all(r["failed_solves"] == stored[(r["method"], r["run"])]["base_metrics"]["mpc_failures"]
                                       for r in recs),
            "statuses": dict(Counter(f["status"] for f in fails)),
            "iterations": dict(Counter(str(f["iterations"]) for f in fails)),
            "by_method": {m: {"runs": sum(1 for r in recs if r["method"] == m and r["failed_solves"]),
                              "failed_solves": sum(r["failed_solves"] for r in recs if r["method"] == m)}
                          for m in sorted({r["method"] for r in recs})},
            "inf_pr_of_returned_iterate": ({"median": float(np.median(inf)), "max": float(max(inf))} if inf else None)}


def main(processes=2, which="lost"):
    protocol, _ = load_protocol(PROTOCOL)
    by_run = {r["run"]: r for r in run_list(protocol)}
    with gzip.open(ROOT / "results" / "phase5" / "rows.jsonl.gz", "rt") as f:
        stored = {(d["method"], d["run"]): d for d in map(json.loads, f)}
    OUT.mkdir(parents=True, exist_ok=True)
    if which == "failed":
        # the runs that were not lost but contained an unsuccessful solve, so that every unsuccessful
        # solve of the campaign has a recorded status
        keys = sorted(k for k, d in stored.items() if not d["lost"] and d["base_metrics"]["mpc_failures"] > 0)
        recs = replay(keys, OUT / "failed_solve_runs.jsonl", stored, by_run, processes)
        lost_recs = [json.loads(x) for x in (OUT / "lost_runs.jsonl").read_text().splitlines()]
        summary = {"runs": len(recs), "row_check": dict(Counter(r["row_check"] for r in recs)),
                   "runs_not_lost": solve_records(recs, stored),
                   "all_runs_with_failed_solve": solve_records(
                       [r for r in recs + lost_recs if r["failed_solves"]], stored),
                   "stored_failed_solves_total": int(sum(d["base_metrics"]["mpc_failures"] for d in stored.values()))}
        (OUT / "failed_solve_runs.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary, indent=1))
        return
    lost = sorted(k for k, d in stored.items() if d["lost"])
    recs = replay(lost, OUT / "lost_runs.jsonl", stored, by_run, processes)
    summary = {"runs": len(recs), "row_check": dict(Counter(r["row_check"] for r in recs)),
               "failed_solve_count_equals_stored": all(
                   r["failed_solves"] == stored[(r["method"], r["run"])]["base_metrics"]["mpc_failures"] for r in recs),
               "trigger_by_method": {}, "yaw_triggered": [], "failed_solves_by_method": {}}
    by_m = defaultdict(list)
    for r in recs:
        by_m[r["method"]].append(r)
    def before_fault(r, t):                 # before the onset, or at any time in a fault free run
        return t is not None and (r["onset"] is None or t < r["onset"])

    def declaration(r):
        if r["t_declare"] is None:
            return "none"
        return "before_onset" if before_fault(r, r["t_declare"]) else "after_onset"
    summary["lost_before_onset_by_method"] = {}
    for m, rs in sorted(by_m.items()):
        summary["trigger_by_method"][m] = dict(Counter(r["trigger"] for r in rs))
        summary["lost_before_onset_by_method"][m] = {
            "lost_runs": len(rs), "fault_free": sum(1 for r in rs if r["onset"] is None),
            "fault_runs_lost_before_onset": dict(Counter(r["fault"][0] for r in rs
                                                         if r["onset"] is not None and r["t_end"] < r["onset"])),
            "fault_runs_lost_after_onset": dict(Counter(r["fault"][0] for r in rs
                                                        if r["onset"] is not None and r["t_end"] >= r["onset"]))}
        fl = [r for r in rs if r["failed_solves"]]
        before_tilt = [f for r in fl for f in r["failures"][:1] if f["attitude_deg"] is not None and f["attitude_deg"] < 10]
        fails = [f for r in fl for f in r["failures"]]
        inf = [f["inf_pr"] for f in fails if f["inf_pr"] is not None]
        lead = [r["t_end"] - r["first_failure_t"] for r in fl if r["first_failure_t"] is not None]
        summary["failed_solves_by_method"][m] = {
            "lost_runs": len(rs), "lost_runs_with_failed_solve": len(fl),
            "failed_solves": sum(r["failed_solves"] for r in rs),
            "first_failure_before_onset": sum(1 for r in fl if r["first_failure_t"] is not None and r["onset"] is not None
                                              and r["first_failure_t"] < r["onset"]),
            "first_failure_before_fault_incl_fault_free": sum(1 for r in fl if before_fault(r, r["first_failure_t"])),
            "first_failure_before_loss": sum(1 for x in lead if x > 0),
            "loss_after_first_failure_s": ({"median": float(np.median(lead)), "min": float(min(lead)),
                                            "max": float(max(lead))} if lead else None),
            "first_failure_with_attitude_below_10_deg": len(before_tilt),
            "statuses": dict(Counter(f["status"] for f in fails)),
            "iterations": dict(Counter(str(f["iterations"]) for f in fails)),
            "scenarios": dict(Counter(str(f["scenarios"]) for f in fails)),
            "inf_pr_of_returned_iterate": ({"median": float(np.median(inf)), "max": float(max(inf))} if inf else None),
            "by_timing_declaration_failure": {
                "|".join(k): v for k, v in sorted(Counter(
                    ("lost_before_onset" if before_fault(r, r["t_end"]) else "lost_after_onset", declaration(r),
                     "failed_solve" if r["failed_solves"] else "no_failed_solve") for r in rs).items())}}
    summary["yaw_triggered"] = [{"method": r["method"], "run": r["run"], **r["angles_end_deg"]}
                                for r in recs if r["trigger"] == "yaw"]
    (OUT / "lost_runs.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("runs", "row_check", "yaw_triggered")}, indent=1))


if __name__ == "__main__":
    args = sys.argv[1:]
    main(*(int(a) for a in args[:1]), *(["failed"] if "--failed" in args else []))

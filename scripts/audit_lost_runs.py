"""Post-review audit of the runs in which the platform was lost (no new result).

Every (method, run) pair whose stored row says the platform was lost is simulated again with the
code and settings of the campaign. Two things are recorded that the campaign did not keep:
  - which attitude angle ended the run: the campaign stops a run when any of the three Euler
    angles (roll, pitch, yaw) exceeds 30 degrees, so a run could in principle end on yaw alone;
  - every MPC solve: its time, IPOPT return status, iterations and primal infeasibility, so that
    unsuccessful solves can be placed in time against the onset, the declaration and the loss.
The recording only reads values; the row of each run must come out identical to the stored row
(every value except timing), which the script checks.

Output: results/audit/lost_runs.jsonl (one record per run), results/audit/lost_runs.json (summary)
Usage: python scripts/audit_lost_runs.py [processes]       (resumable)
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


def main(processes=2):
    protocol, _ = load_protocol(PROTOCOL)
    by_run = {r["run"]: r for r in run_list(protocol)}
    with gzip.open(ROOT / "results" / "phase5" / "rows.jsonl.gz", "rt") as f:
        stored = {(d["method"], d["run"]): d for d in map(json.loads, f)}
    lost = sorted(k for k, d in stored.items() if d["lost"])
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "lost_runs.jsonl"
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[(d["method"], d["run"])] = d
    todo = [(m, by_run[r]) for m, r in lost if (m, r) not in done]
    print(f"{len(lost)} lost runs, {len(todo)} to do", flush=True)
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
    recs = [done[k] for k in lost]
    summary = {"runs": len(recs), "row_check": dict(Counter(r["row_check"] for r in recs)),
               "trigger_by_method": {}, "yaw_triggered": [], "failed_solves_by_method": {}}
    by_m = defaultdict(list)
    for r in recs:
        by_m[r["method"]].append(r)
    for m, rs in sorted(by_m.items()):
        summary["trigger_by_method"][m] = dict(Counter(r["trigger"] for r in rs))
        fl = [r for r in rs if r["failed_solves"]]
        before_tilt = [f for r in fl for f in r["failures"][:1] if f["attitude_deg"] is not None and f["attitude_deg"] < 10]
        summary["failed_solves_by_method"][m] = {
            "lost_runs": len(rs), "lost_runs_with_failed_solve": len(fl),
            "failed_solves": sum(r["failed_solves"] for r in rs),
            "first_failure_before_onset": sum(1 for r in fl if r["first_failure_t"] is not None and r["onset"] is not None
                                              and r["first_failure_t"] < r["onset"]),
            "first_failure_with_attitude_below_10_deg": len(before_tilt),
            "statuses": dict(Counter(f["status"] for r in fl for f in r["failures"]))}
    summary["yaw_triggered"] = [{"method": r["method"], "run": r["run"], **r["angles_end_deg"]}
                                for r in recs if r["trigger"] == "yaw"]
    (OUT / "lost_runs.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("runs", "row_check", "yaw_triggered")}, indent=1))


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:2]))

"""Post-review latency study (post_review/latency_protocol.yaml).

The proposed controller P is run again on the 104 upper cable break runs of the campaign, with an
MPC computation time D of 0, 20, 40 and 60 ms. The controller of the campaign is used unchanged;
only the availability of the MPC output is delayed, by a subclass defined here that replaces the
controller of cdpr_ft/sim/episode.py inside the worker processes. The frozen files of the package
are not touched, so the digest of the frozen controller stays the same.

Scheduling for D > 0 (one core): a solve starts at t_s = m * max(20 ms, D); it uses the state
estimate and the reference at t_s and the command applied at t_s as its previous command; its
first tension command and its drum acceleration feedforward take effect at t_s + D and are held
until the next solution takes effect. D = 0 is the controller of the campaign.

The D = 0 rows are compared with the stored rows of the campaign first; the delayed conditions run
only if every D = 0 row is identical (every value except timing).

Output: results/latency/rows.jsonl (one row per run and delay), results/latency/analysis.json
Usage: python scripts/latency_probe.py [processes]      (resumable)
       python scripts/latency_probe.py --analyse
"""
from __future__ import annotations

import gzip
import json
import multiprocessing as mp
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cdpr_ft.control.ftc import FaultTolerantController  # noqa: E402
from cdpr_ft.eval.campaign import init_worker, load_protocol, run_job, run_list  # noqa: E402
from cdpr_ft.provenance import stamp  # noqa: E402

STUDY = yaml.safe_load((ROOT / "post_review" / "latency_protocol.yaml").read_text())
PROTOCOL = ROOT / STUDY["runs"]["source"]
OUT = ROOT / "results" / "latency"
DELAYS = [int(d) for d in STUDY["conditions"]["computation_time_ms"]]
METHOD = STUDY["method"]


class DelayedMPCController(FaultTolerantController):
    """The controller of the campaign with the MPC output available D ms after the solve starts."""
    delay_ms = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pending = None
        if self.delay_ms > 0:
            period_ms = max(1e3 * self.mpc.cfg.dt, self.delay_ms)
            self.cfg = replace(self.cfg, mpc_every=int(round(period_ms * 1e-3 / self.dt)))

    def __call__(self, k, t, y):
        if self._pending is not None and t >= self._pending[0] - 1e-9:
            _, self.t_d, self.ddtheta_ff = self._pending
            self._pending = None
        return super().__call__(k, t, y)

    def _mpc_step(self, t):
        if self.delay_ms == 0:
            return super()._mpc_step(t)
        held_t_d, held_ff = self.t_d.copy(), np.array(self.ddtheta_ff, float).copy()
        self.t_prev = held_t_d.copy()            # the command applied when the solve starts
        super()._mpc_step(t)
        self._pending = (t + 1e-3 * self.delay_ms, self.t_d.copy(), np.array(self.ddtheta_ff, float).copy())
        self.t_d, self.ddtheta_ff = held_t_d, held_ff


def worker_init(protocol_path):
    from cdpr_ft.sim import episode
    init_worker(protocol_path, 1, True)
    episode.FaultTolerantController = DelayedMPCController


def one(job):
    delay, r = job
    DelayedMPCController.delay_ms = delay
    row = run_job((METHOD, r, None))
    row["delay_ms"] = delay
    return row


def study_runs(protocol):
    sel = STUDY["runs"]["select"]
    runs = [r for r in run_list(protocol) if r["group"] == sel["group"] and r["fault"][0] == sel["fault_type"]
            and r["fault"][1] in sel["cables"]]
    assert len(runs) == STUDY["runs"]["expected"], len(runs)
    return runs


def stored_rows():
    with gzip.open(ROOT / STUDY["validation"]["reference_rows"], "rt") as f:
        return {d["run"]: d for d in map(json.loads, f) if d["method"] == METHOD}


def load_done(path):
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[(d["delay_ms"], d["run"])] = d
    return done


def simulate(delays, runs, processes, path, phash):
    done = load_done(path)
    todo = [(d, r) for d in delays for r in runs if (d, r["run"]) not in done]
    if not todo:
        return done
    print(f"  delays {delays}: {len(todo)} runs to do", flush=True)
    t0 = time.time()
    with mp.get_context("fork").Pool(processes, initializer=worker_init, initargs=(str(PROTOCOL),),
                                     maxtasksperchild=50) as pool, path.open("a") as f:
        for k, row in enumerate(pool.imap_unordered(one, todo, chunksize=1)):
            row["protocol_sha256_16"] = phash
            f.write(json.dumps(row, default=float) + "\n")
            f.flush()
            done[(row["delay_ms"], row["run"])] = row
            if (k + 1) % 10 == 0 or k + 1 == len(todo):
                print(f"    {k + 1}/{len(todo)} runs, {(time.time() - t0) / 60:.1f} min", flush=True)
    return done


def validate(done, runs):
    from replicate import compare_row
    ref = stored_rows()
    classes = {}
    for r in runs:
        c = compare_row(ref[r["run"]], {k: v for k, v in done[(0, r["run"])].items() if k != "delay_ms"})
        classes[r["run"]] = c["class"] if not c["outcome"] else "outcome: " + ", ".join(c["outcome"])
    bad = {k: v for k, v in classes.items() if v != "identical"}
    return {"runs": len(classes), "identical": len(classes) - len(bad), "not_identical": bad}


def med(xs):
    return float(np.median(xs)) if len(xs) else None


def analyse():
    from phase5_analyse import holm, paired_binary, paired_continuous, wilson
    rng = np.random.default_rng(0)
    protocol, _ = load_protocol(PROTOCOL)
    runs = [r["run"] for r in study_runs(protocol)]
    done = load_done(OUT / "rows.jsonl")
    by = {d: {r: done[(d, r)] for r in runs if (d, r) in done} for d in DELAYS}
    missing = {d: len(runs) - len(by[d]) for d in DELAYS if len(by[d]) < len(runs)}
    res = {"protocol": "post_review/latency_protocol.yaml", "runs": len(runs), "missing": missing, "conditions": {},
           "against_0_ms": {}, "validation": json.loads((OUT / "validation.json").read_text())}
    for d in DELAYS:
        rows = list(by[d].values())
        lost = sum(r["lost"] for r in rows)
        res["conditions"][d] = {
            "runs": len(rows), "lost": lost, "lost_wilson": wilson(lost, len(rows)),
            "correct": sum(bool(r["correct"]) for r in rows),
            "delay_ms_median": med([1e3 * r["delay_s"] for r in rows if r["correct"]]),
            # descriptive detail beyond the protocol: when the losses happened, how the runs ended and how
            # many were declared before the break (false alarms)
            "lost_before_onset": sum(1 for r in rows if r["lost"] and r["t_end"] < r["fault"][2]),
            "stopped": {k: sum(1 for r in rows if r["stopped"] == k) for k in sorted({str(r["stopped"]) for r in rows
                                                                                       if r["stopped"]})},
            "false_alarm": sum(bool(r["false_alarm"]) for r in rows),
            "lost_t_from_onset_s": sorted(round(r["t_end"] - r["fault"][2], 3) for r in rows if r["lost"]),
            "runs_with_failed_solve": sum(1 for r in rows if r["base_metrics"]["mpc_failures"] > 0),
            "failed_solves": int(sum(r["base_metrics"]["mpc_failures"] for r in rows)),
            **{key: val for k in STUDY["endpoints"]["secondary"]
               for key, val in ((f"median_{k}", med([r[k] for r in rows if r.get(k) is not None])),
                                (f"n_{k}", sum(1 for r in rows if r.get(k) is not None)))}}
    keys = [d for d in DELAYS if d > 0 and len(by[d]) == len(runs)]
    if len(by[0]) == len(runs) and keys:
        comp = {d: {"lost": paired_binary(by[d], by[0], runs, lambda r: r["lost"], rng)} for d in keys}
        for k in STUDY["endpoints"]["secondary"]:
            for d in keys:
                comp[d][k] = paired_continuous(by[d], by[0], runs, k, rng)
        for k in ["lost"] + STUDY["endpoints"]["secondary"]:
            # a continuous endpoint has no test where no run of the delayed condition reached the break
            # (its transient is not defined); such comparisons get no p-value and leave the Holm family
            for d in keys:
                if comp[d][k].get("pairs", 1) == 0:
                    comp[d][k]["p"] = None
            ok_d = [d for d in keys if comp[d][k]["p"] is not None]
            adj = holm([comp[d][k]["p"] for d in ok_d]) if ok_d else []
            for d in keys:
                comp[d][k]["p_holm"] = None
            for d, p in zip(ok_d, adj):
                comp[d][k]["p_holm"] = p
        res["against_0_ms"] = comp
        ok = [d for d in keys if comp[d]["lost"]["p_holm"] >= STUDY["statistics"]["alpha"]
              or comp[d]["lost"]["a"][0] <= comp[d]["lost"]["b"][0]]
        res["largest_delay_without_significantly_more_losses_ms"] = max([0] + [d for d in keys if
                                                                           all(x in ok for x in keys if x <= d)])
    res["provenance"] = stamp()
    (OUT / "analysis.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({d: {k: v for k, v in c.items() if k in ("lost", "runs_with_failed_solve",
                                                                "median_transient_outside_pct")}
                      for d, c in res["conditions"].items()}, indent=1))


def main(processes=2):
    protocol, phash = load_protocol(PROTOCOL)
    runs = study_runs(protocol)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "rows.jsonl"
    done = simulate([0], runs, processes, path, phash)
    v = validate(done, runs)
    (OUT / "validation.json").write_text(json.dumps(v, indent=1))
    print(f"  validation: {v['identical']} of {v['runs']} rows at D = 0 identical to the campaign", flush=True)
    if v["not_identical"]:
        sys.exit("the D = 0 rows differ from the campaign; the delayed conditions are not run (protocol, validation)")
    simulate([d for d in DELAYS if d > 0], runs, processes, path, phash)
    analyse()


if __name__ == "__main__":
    if sys.argv[1:2] == ["--analyse"]:
        analyse()
    else:
        main(*(int(a) for a in sys.argv[1:2]))

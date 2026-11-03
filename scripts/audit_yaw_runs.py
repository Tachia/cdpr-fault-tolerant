"""Post-review check of the runs that the loss rule ended on yaw (no new result for the comparison).

The campaign ends a run when any of the three ZYX Euler angles (roll, pitch, yaw) exceeds 30 degrees
(cdpr_ft/sim/episode.py). The audit of the lost runs (scripts/audit_lost_runs.py) found runs in
which yaw was the only angle beyond 30 degrees when the run stopped. Under a rule on the tilt alone
(roll or pitch beyond 30 degrees) these runs would have continued, so their outcome under that rule
is not known from the stored rows.

Each such run is simulated again with the attitude stop disabled (the frame and drum stops are kept)
and the trajectory is examined afterwards:
  - the first time any of the three angles exceeds 30 degrees must equal the stored stop time,
    which confirms that the run is the stored run up to that point;
  - the first time roll or pitch exceeds 30 degrees gives the outcome under the tilt rule;
  - the first time the physical tilt arccos(cos(roll) cos(pitch)) exceeds 30 degrees is reported too.
A frame or drum stop counts as a loss under every rule, as in the campaign. The controller does not
read the attitude limit, so disabling the stop changes nothing before the stored stop time.

A rule on roll and pitch alone can only end fewer runs than the rule of the campaign, so the runs
examined here are the only ones whose outcome it can change. The primary comparison of platforms
lost after a break is then recomputed with those outcomes (exact McNemar test, Holm correction over
the same four comparisons, as in scripts/phase5_analyse.py); every other endpoint is unaffected.

Output: results/audit/yaw_runs.json
Usage: python scripts/audit_yaw_runs.py [processes]
       python scripts/audit_yaw_runs.py --analyse      (recomputes the comparison only)
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdpr_ft.eval import campaign  # noqa: E402
from cdpr_ft.eval.campaign import init_worker, load_protocol, run_list  # noqa: E402
from cdpr_ft.provenance import stamp  # noqa: E402

PROTOCOL = ROOT / "configs" / "evaluation_phase5.yaml"
OUT = ROOT / "results" / "audit"
LIMIT = 30.0
_LOG: dict = {}


def worker_init(protocol_path):
    init_worker(protocol_path, 1, True)
    campaign._W["ftc"] = replace(campaign._W["ftc"], lost_tilt_deg=float("inf"))
    evaluate = campaign.evaluate

    def keep(spec, log, cl, ctrl, protocol, trace_path=None):
        _LOG["t"], _LOG["q"], _LOG["stopped"] = np.asarray(log["t"]), np.asarray(log["q"]), log["stopped"]
        return evaluate(spec, log, cl, ctrl, protocol, trace_path)
    campaign.evaluate = keep


def first(t, mask):
    k = np.flatnonzero(mask)
    return float(t[k[0]]) if len(k) else None


def one(job):
    method, r = job
    _LOG.clear()
    try:
        campaign.run_job((method, r, None))
        error = None
    except Exception as exc:                  # a run continued far beyond the limit may fail numerically
        error = repr(exc)
    if "t" not in _LOG:
        return {"method": method, "run": r["run"], "error": error}
    t, q, stopped = _LOG["t"], _LOG["q"], _LOG["stopped"]
    a = np.rad2deg(np.abs(q[:, 3:]))
    tilt = np.rad2deg(np.arccos(np.clip(np.cos(q[:, 3]) * np.cos(q[:, 4]), -1.0, 1.0)))
    return {"method": method, "run": r["run"], "fault": r["fault"], "error": error,
            "t_any_angle": first(t, a.max(axis=1) > LIMIT), "t_roll_or_pitch": first(t, a[:, :2].max(axis=1) > LIMIT),
            "t_physical_tilt": first(t, tilt > LIMIT), "other_stop": stopped, "t_last": float(t[-1]),
            "max_roll_pitch_deg": float(a[:, :2].max()), "max_physical_tilt_deg": float(tilt.max())}


def main(processes=1):
    protocol, _ = load_protocol(PROTOCOL)
    by_run = {r["run"]: r for r in run_list(protocol)}
    recs = [json.loads(line) for line in (OUT / "lost_runs.jsonl").read_text().splitlines()]
    yaw_only = sorted((r["method"], r["run"], r["t_end"]) for r in recs
                      if r["angles_end_deg"]["yaw"] > LIMIT
                      and max(r["angles_end_deg"]["roll"], r["angles_end_deg"]["pitch"]) <= LIMIT)
    t_stop = {(m, k): t for m, k, t in yaw_only}
    with mp.get_context("fork").Pool(processes, initializer=worker_init, initargs=(str(PROTOCOL),)) as pool:
        res = pool.map(one, [(m, by_run[k]) for m, k, _ in yaw_only], chunksize=1)
    for x in res:
        x["t_stored_stop"] = t_stop[(x["method"], x["run"])]
        x["same_as_stored"] = x.get("t_any_angle") is not None and abs(x["t_any_angle"] - x["t_stored_stop"]) < 1e-9
        x["lost_roll_pitch_rule"] = x.get("t_roll_or_pitch") is not None or x.get("other_stop") is not None
        x["lost_physical_tilt_rule"] = x.get("t_physical_tilt") is not None or x.get("other_stop") is not None
    out = {"limit_deg": LIMIT, "runs": res,
           "summary": {"yaw_only_runs": len(res), "all_same_as_stored": all(x["same_as_stored"] for x in res),
                       "lost_roll_pitch_rule": sum(x["lost_roll_pitch_rule"] for x in res),
                       "lost_physical_tilt_rule": sum(x["lost_physical_tilt_rule"] for x in res)},
           "provenance": stamp()}
    (OUT / "yaw_runs.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out["summary"], indent=1))
    for x in res:
        print({k: x.get(k) for k in ("method", "run", "t_stored_stop", "t_roll_or_pitch", "t_physical_tilt",
                                     "other_stop", "t_last", "error")})
    analyse()


def analyse():
    """The primary comparison of platforms lost after a break, under both rules."""
    import gzip
    sys.path.insert(0, str(ROOT / "scripts"))
    from phase5_analyse import holm, mcnemar
    protocol, _ = load_protocol(PROTOCOL)
    out = json.loads((OUT / "yaw_runs.json").read_text())
    kept = {(x["method"], x["run"]) for x in out["runs"] if not x["lost_roll_pitch_rule"]}
    with gzip.open(ROOT / "results" / "phase5" / "rows.jsonl.gz", "rt") as f:
        by = {}
        for d in map(json.loads, f):
            by.setdefault(d["method"], {})[d["run"]] = d
    item = next(i for i in protocol["analysis"]["primary"] if i["endpoint"].startswith("platform lost"))
    result = {}
    for rule, lost in (("three_angles", lambda m, k: by[m][k]["lost"]),
                       ("roll_or_pitch", lambda m, k: by[m][k]["lost"] and (m, k) not in kept)):
        comps = []
        for comp in item["comparisons"]:
            a, b = [x.strip() for x in comp.split("vs")]
            runs = sorted(k for k in set(by[a]) & set(by[b]) if by[a][k]["fault"] and by[a][k]["fault"][0] == "break")
            x = np.array([lost(a, k) for k in runs], bool)
            y = np.array([lost(b, k) for k in runs], bool)
            upper = [i for i, k in enumerate(runs) if by[a][k]["fault"][1] < 4]
            comps.append({"comparison": comp, "pairs": len(runs), "a": int(x.sum()), "b": int(y.sum()),
                          "b_upper": int(y[upper].sum()), "b_lower": int(y.sum() - y[upper].sum()), **mcnemar(x, y)})
        for c, h in zip(comps, holm([c["p"] for c in comps])):
            c["p_holm"] = h
        result[rule] = comps
    out["primary_platform_lost_after_a_break"] = result
    out["outcomes_changed_by_roll_or_pitch_rule"] = sorted(kept)
    (OUT / "yaw_runs.json").write_text(json.dumps(out, indent=1, default=float))
    for rule, comps in result.items():
        print(rule, [(c["comparison"], c["a"], c["b"], f"{c['p_holm']:.2e}") for c in comps])


if __name__ == "__main__":
    if sys.argv[1:2] == ["--analyse"]:
        analyse()
    else:
        main(*(int(a) for a in sys.argv[1:2]))

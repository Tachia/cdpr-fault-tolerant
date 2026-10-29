"""Phase 5: gains of baseline B2 after a break declaration, chosen on calibration runs.

The proposed controller changes its MPC weights after a break declaration (post_break_weights of
the frozen controller, fixed in Phase 3 on calibration runs), because the platform must then
move on seven cables to a pose they can hold. So that B2 is treated alike, the gains of its
computed-torque law after a break declaration are chosen here on calibration runs, by a rule fixed
before the runs: among w in {5, 7.5, 10, 12.5, 15} rad/s (critically damped, kp = w^2, kd = 2 w;
15 rad/s keeps the gains used before the fault, scripts/phase5_b2_gains.py), the setting with the
fewest lost runs, then the smallest median share of the transient window with a cable outside its
bounds, then the smallest median RMS position error after the transient.

Runs: the 32 break runs of B2's calibration set (scripts/phase5_b2_runs.py: seeds 150000 and up,
both paths, every cable, two repetitions), continued to 4 s after the onset as in the campaign.
B2 runs exactly as in the campaign (cdpr_ft.eval.campaign, method "B2", its thresholds from
configs/b2_phase5.yaml), declaring with its own threshold rule; only the post-break gains vary.
These runs also check that B2 works as a whole: its calibration runs had declarations disabled
and never exercised what it does after a declaration.
Output: results/phase5_b2_postbreak.json, and the chosen gains in configs/b2_phase5.yaml
(ct_kp_post, ct_kd_post)
Usage: python scripts/phase5_b2_postbreak.py [processes]
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cdpr_ft.eval import campaign as C  # noqa: E402
from cdpr_ft.provenance import stamp  # noqa: E402
from phase5_b2_runs import run_list as b2_runs  # noqa: E402

PROTOCOL = ROOT / "configs" / "evaluation_phase5.yaml"
OMEGA = [5.0, 7.5, 10.0, 12.5, 15.0]


def runs_for(protocol):
    after = float(protocol["runs"]["after_onset"])
    duration = float(protocol["runs"]["duration"])
    out = []
    for r in b2_runs():
        if r["fault"] is not None and r["fault"][0] == "break":
            out.append({"group": "fault", "trajectory": r["trajectory"], "seed": r["seed"], "rep": r["seed"] % 2,
                        "noise_factor": r["noise_factor"], "fault": r["fault"],
                        "t_end": min(duration, r["fault"][2] + after), "run": len(out)})
    return out


def job(args):
    from cdpr_ft.sim.episode import EpisodeSpec, run_episode
    w, r = args
    protocol = C._W["protocol"]
    spec = EpisodeSpec(trajectory=r["trajectory"], seed=r["seed"], noise_factor=r["noise_factor"],
                       fault=tuple(r["fault"]), t_end=r["t_end"], variant=C.VARIANT["B2"])
    cfg, factory, extra = C.method_config("B2", protocol)
    extra = dict(extra, ct_gains_post=((w * w,) * 6, (2 * w,) * 6))
    res, ctrl = run_episode(spec, C._W["imm"], C._W["mpc"], C._W["planner"], C._W["safe"], cfg, keep_log=True,
                            alt_factory=factory, return_controller=True, ctrl_kwargs=extra)
    log, cl = res.pop("log"), res.pop("ctrl_log")
    m = C.evaluate(spec, log, cl, ctrl, protocol, None)
    keys = ("lost", "stopped", "declared", "delay_s", "correct", "false_alarm", "transient_min_margin_N",
            "transient_outside_pct", "post_rmse_p_mm", "post_outside_pct", "max_tilt_deg", "plan_branch", "t_end")
    return {"omega_post": w, "trajectory": r["trajectory"], "seed": r["seed"], "cable": r["fault"][1] + 1,
            **{k: m.get(k) for k in keys}}


def main(processes=2):
    protocol, phash = C.load_protocol(PROTOCOL)
    runs = runs_for(protocol)
    jobs = [(w, r) for w in OMEGA for r in runs]
    with mp.get_context("fork").Pool(processes, initializer=C.init_worker, initargs=(str(PROTOCOL), 1, False)) as pool:
        rows = pool.map(job, jobs, chunksize=1)
    table = []
    for w in OMEGA:
        g = [r for r in rows if r["omega_post"] == w]
        post = [r["post_rmse_p_mm"] for r in g if r["post_rmse_p_mm"] is not None]
        table.append({"omega_post": w, "kp": w * w, "kd": 2 * w, "runs": len(g), "lost": sum(r["lost"] for r in g),
                      "lost_upper": sum(r["lost"] for r in g if r["cable"] <= 4),
                      "lost_lower": sum(r["lost"] for r in g if r["cable"] > 4),
                      "correct": sum(bool(r["correct"]) for r in g),
                      "median_transient_outside_pct": float(np.median([r["transient_outside_pct"] for r in g])),
                      "median_post_rmse_mm": float(np.median(post)) if post else None})
        print(table[-1], flush=True)
    chosen = min(table, key=lambda t: (t["lost"], t["median_transient_outside_pct"],
                                       t["median_post_rmse_mm"] if t["median_post_rmse_mm"] is not None else np.inf))
    print("chosen", chosen)
    (ROOT / "results" / "phase5_b2_postbreak.json").write_text(json.dumps(
        {"rule": "fewest lost runs, then smallest median transient share outside the bounds, then smallest median "
                 "post-transient RMS error", "protocol_sha256_16": phash, "runs": rows, "table": table,
         "chosen": chosen, "provenance": stamp()}, indent=1, default=float))
    path = ROOT / "configs" / "b2_phase5.yaml"
    text = path.read_text()
    body = yaml.safe_load(text)
    head = "".join(line + "\n" for line in text.splitlines() if line.startswith("#"))
    body["ct_kp_post"] = [float(chosen["kp"])] * 6
    body["ct_kd_post"] = [float(chosen["kd"])] * 6
    body["source"] = body["source"] + "; gains after a break declaration: results/phase5_b2_postbreak.json (B2 calibration break runs)"
    path.write_text(head + yaml.safe_dump(body, sort_keys=False))


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:2]))

"""Phase 3: closed-loop check of the full controller on the calibration set, with declarations.

The fault scenarios and seeds of scripts/phase3_calibration_runs.py are run with the proposed
controller and a given controller configuration; fault runs last until T_AFTER seconds after the
onset, fault-free runs the full task. These runs serve the choice of the settings that the
recorded calibration runs cannot decide (the MPC weights after a break: rule, fixed before the
runs, fewer lost platforms, then the smaller median share of post-fault time with a tension out
of bounds) and a last check of the frozen configuration; they are calibration data and never
enter reported results.
Output: results/phase3_check_<tag>.json
Usage: python scripts/phase3_closed_loop_check.py <tag> <controller yaml> [processes] [subset: all|breaks]
       [ftc overrides as json]
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdpr_ft.control.config import load_controller  # noqa: E402
from cdpr_ft.provenance import stamp  # noqa: E402
from cdpr_ft.sim.batch import run_batch  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from phase3_calibration_runs import calibration_specs  # noqa: E402

T_AFTER = 4.0


def summarise(res):
    def pct(v):
        return float(np.round(100 * np.mean(v), 1)) if len(v) else None
    out = {}
    free = [r for r in res if not r["fault"]]
    out["fault_free"] = {"n": len(free), "false_alarms": int(sum(r["false_alarm"] for r in free)),
                         "lost": int(sum(r["lost"] for r in free)),
                         "rms_error_mm_median": float(np.median([1e3 * r["rms_error_pre_m"] for r in free])) if free else None,
                         "rms_error_mm_max": float(np.max([1e3 * r["rms_error_pre_m"] for r in free])) if free else None}
    for kind in ("break", "loe", "bias"):
        sel = [r for r in res if r["fault"] and r["fault"][0] == kind]
        if not sel:
            continue
        dec = [r for r in sel if r["delay_s"] is not None]
        out[kind] = {"n": len(sel), "false_alarm_before_onset": int(sum(r["false_alarm"] for r in sel)),
                     "declared_after_onset": len(dec), "correct": int(sum(bool(r["correct"]) for r in dec)),
                     "delay_ms_median": float(np.median([1e3 * r["delay_s"] for r in dec])) if dec else None,
                     "delay_ms_max": float(np.max([1e3 * r["delay_s"] for r in dec])) if dec else None,
                     "lost": int(sum(r["lost"] for r in sel)),
                     "post_outside_pct_median": float(np.median([r.get("post_outside_bounds_pct", 0) for r in sel])),
                     "post_outside_pct_max": float(np.max([r.get("post_outside_bounds_pct", 0) for r in sel])),
                     "post_tension_max_N": float(np.max([r.get("post_tension_max", 0) for r in sel])),
                     "branches": {str(b): int(sum(r.get("plan_branch") == b for r in sel)) for b in (1, 2, 3)}}
    mpc = [r["mpc_seconds"] for r in res if "mpc_seconds" in r]
    imm = [r["imm_seconds"] for r in res if "imm_seconds" in r]
    if mpc:
        out["mpc_seconds"] = {"mean_of_means": float(np.mean([m["mean"] for m in mpc])),
                              "max_p99": float(np.max([m["p99"] for m in mpc])), "max": float(np.max([m["max"] for m in mpc]))}
    if imm:
        out["imm_seconds"] = {"mean_of_means": float(np.mean([m["mean"] for m in imm])),
                              "max_p99": float(np.max([m["p99"] for m in imm])), "max": float(np.max([m["max"] for m in imm]))}
    return out


def main(tag, config, processes=2, subset="all", ftc_over=None):
    cfgs, cfg_hash = load_controller(config)
    if ftc_over:
        over = {k: (tuple(tuple(x) for x in v) if isinstance(v, list) else v) for k, v in ftc_over.items()}
        cfgs["ftc"] = replace(cfgs["ftc"], **over)
    specs = calibration_specs()
    specs = [replace(s, t_end=None if not s.fault else min(s.fault[2] + T_AFTER, 16.0)) for s in specs]
    if subset == "breaks":
        specs = [s for s in specs if s.fault and s.fault[0] == "break"]
    t0 = time.time()
    res = run_batch(specs, cfgs["imm"], cfgs["mpc"], cfgs["planner"], cfgs["ftc"], processes=processes,
                    recovery_cfg=cfgs["recovery"])
    for r in res:
        r.pop("t_imm", None)
        r.pop("mu", None)
    summary = summarise(res)
    out = {"config": config, "config_sha256_16": cfg_hash, "ftc_overrides": ftc_over, "summary": summary, "runs": res,
           "seconds": time.time() - t0, "provenance": stamp()}
    (ROOT / "results" / f"phase3_check_{tag}.json").write_text(json.dumps(out, default=str, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 2,
         sys.argv[4] if len(sys.argv) > 4 else "all", json.loads(sys.argv[5]) if len(sys.argv) > 5 else None)

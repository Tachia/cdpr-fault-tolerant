"""Phase 3: offline replay of the diagnosis on the calibration records, for a set of estimator
settings (one parameter changed at a time from the base setting).

For each setting and each recorded run the winch observers and the IMM are run on the
recorded readings and commands (cdpr_ft/estimation/diagnosis.py). The replay is exact for the
setting the runs were recorded with; for other settings it is exact up to the influence of the
mode probabilities on the MPC (the plant motion of the recorded run is kept), which is small
before any declaration. Metrics per setting:
  fault-free runs and the pre-onset part of fault runs: the largest probability of any fault
    mode, and the time any fault mode spends above 0.5;
  fault runs: the delay until the true mode first exceeds 0.5 / 0.9 / 0.99, whether a wrong
    fault mode exceeded 0.9 before it, and whether the true mode is the most probable one at the
    end of the run.
Selection rule (written before the replay results were seen): a setting qualifies if every
fault run has the true mode reaching 0.99 and being the most probable mode at the end of the run,
with no wrong fault mode above 0.9 before the true one. The base setting is kept unless a
qualifying setting lowers the largest fault-mode probability seen before the onsets (fault-free
runs included) by at least a factor of ten without raising the median break delay to 0.99 by
more than 5 ms; among such settings, the one with the smallest of that probability is taken.
No setting, the base included, met that qualifying condition: after a break, the actuator mode
of the same winch leads for one estimator cycle in about half of the break runs, before the
break mode takes over (the declaration rule, with N_c >= 2, does not act on such a single
cycle). The one-at-a-time round showed two influential settings, the angular process noise
(sigma_acc_ang = 1 removed every fault-mode probability above 0.5 from the fault-free data) and
the observer bandwidth (100 Hz halved the break delay). A second round
(results/phase3_replay_stage2_settings.json) combines them, with this rule, written before
that round was run: among the settings with no fault-mode probability above 0.5 in the
fault-free data, take the one with the fewest break runs in which a wrong mode exceeds 0.9
before the true one, then the smallest median break delay to 0.99, then the smallest median
bias delay; if none qualifies, keep the base.
Output: results/phase3_replay_<tag>.json (results/phase3_replay_<tag>_stage2_settings.json)
Usage: python scripts/phase3_replay.py <calibration tag> [processes] [settings json file]
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdpr_ft.estimation.diagnosis import replay  # noqa: E402
from cdpr_ft.estimation.imm import IMM, IMMConfig  # noqa: E402
from cdpr_ft.params import inner_rate, load_cable_winch, load_robot  # noqa: E402
from cdpr_ft.provenance import stamp  # noqa: E402
from cdpr_ft.sim.episode import EpisodeSpec, controller_start, mode_index  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from phase3_calibration_runs import load_runs  # noqa: E402

THRESH = (0.5, 0.9, 0.99)


def default_settings(base_imm, base_eso):
    """One-at-a-time variations around the base setting."""
    s = [("base", {}, base_eso)]
    for key, values in (("sigma_acc_lin", (0.25, 1.0)), ("sigma_acc_ang", (0.25, 1.0)),
                        ("r_tension", (0.75, 3.0)), ("r_length", (1e-4, 4e-4)),
                        ("fault_rate", (1e-4, 1e-2)), ("sigma_deficit", (5.0, 50.0)),
                        ("sigma_offset", (1e-4,)), ("sigma_wrench_force", (2.0,)), ("t_arm", (1.0, 1.5))):
        for v in values:
            s.append((f"{key}={v:g}", {key: v}, base_eso))
    for bw in (25.0, 100.0):
        s.append((f"eso={bw:g}Hz", {}, bw))
    return s


_W = {}


def _init(imm_cfg, eso_bw, imm_every):
    robot, cw = load_robot(), load_cable_winch()
    _W.update(robot=robot, cw=cw, imm=IMM(robot, cw.EA, imm_cfg), eso=eso_bw, every=imm_every)


def _one(job):
    run, rec_path = job
    robot, cw = _W["robot"], _W["cw"]
    spec = EpisodeSpec(**{**run["spec"], "fault": tuple(run["spec"]["fault"]) if run["spec"]["fault"] else None})
    _, _, q0, t_nom, l_u0 = controller_start(spec, robot, cw)
    rec = np.load(rec_path)
    t_imm, mu = replay({"theta": rec["theta"], "tau": rec["tau"], "td_in": rec["td_in"]}, cw, l_u0, _W["imm"],
                       _W["eso"], _W["every"], 1.0 / inner_rate(), t_nom, q0)
    return metrics(spec, t_imm, mu, robot.n)


def metrics(spec, t_imm, mu, n):
    onset = spec.fault[2] if spec.fault else np.inf
    pre = t_imm < onset
    fault_mu = mu[:, 1:].max(1)
    out = {"fault": spec.fault, "trajectory": spec.trajectory, "noise": spec.noise_factor,
           "pre_max_fault_mu": float(fault_mu[pre].max()) if pre.any() else None,
           "pre_time_above_05": float(np.sum(fault_mu[pre] > 0.5) * (t_imm[1] - t_imm[0])) if pre.any() else 0.0,
           "pre_duration": float(t_imm[pre][-1] - t_imm[0]) if pre.any() else 0.0}
    if spec.fault:
        kind = "break" if spec.fault[0] == "break" else "actuator"
        j = mode_index(n, kind, spec.fault[1])
        post = ~pre
        tp, mp_ = t_imm[post], mu[post]
        for th in THRESH:
            hit = np.nonzero(mp_[:, j] >= th)[0]
            out[f"delay_{th}"] = float(tp[hit[0]] - onset) if len(hit) else None
        other = np.delete(mp_, [0, j], axis=1).max(1)
        first_true = np.nonzero(mp_[:, j] >= 0.9)[0]
        first_wrong = np.nonzero(other >= 0.9)[0]
        out["wrong_before_true"] = bool(len(first_wrong) and (not len(first_true) or first_wrong[0] < first_true[0]))
        out["true_top_at_end"] = bool(np.argmax(mp_[-1]) == j)
    return out


def summarise(rows):
    free = [r for r in rows if not r["fault"]]
    faulty = [r for r in rows if r["fault"]]
    pre = [r["pre_max_fault_mu"] for r in rows if r["pre_max_fault_mu"] is not None]
    s = {"max_fault_mu_fault_free": float(max(pre)) if pre else None,
         "time_above_05_s": float(sum(r["pre_time_above_05"] for r in rows)),
         "fault_free_time_s": float(sum(r["pre_duration"] for r in rows))}
    for kind in ("break", "loe", "bias"):
        sel = [r for r in faulty if r["fault"][0] == kind]
        if not sel:
            continue
        d = {}
        for th in THRESH:
            v = [r[f"delay_{th}"] for r in sel]
            got = [x for x in v if x is not None]
            d[f"found_{th}"] = f"{len(got)}/{len(v)}"
            d[f"median_delay_{th}"] = float(np.median(got)) if got else None
            d[f"max_delay_{th}"] = float(np.max(got)) if got else None
        d["wrong_before_true"] = int(sum(r["wrong_before_true"] for r in sel))
        d["true_top_at_end"] = f"{sum(r['true_top_at_end'] for r in sel)}/{len(sel)}"
        s[kind] = d
    return s


def main(tag, processes=2, settings_file=None):
    folder = ROOT / "results" / "phase3_calib" / tag
    data = load_runs(folder)
    runs = data["runs"]
    base_imm = data["meta"]["imm"]
    base_eso = data["meta"]["ftc"]["eso_bandwidth_hz"]
    imm_every = data["meta"]["ftc"]["imm_every"]
    if settings_file:
        settings = [(s["name"], s["imm"], s["eso"]) for s in json.loads(Path(settings_file).read_text())]
    else:
        settings = default_settings(base_imm, base_eso)
    jobs = [(r, str(folder / r["record"])) for r in runs]
    results = []
    t0 = time.time()
    for name, over, eso in settings:
        cfg = IMMConfig(**{**base_imm, **over})
        with mp.get_context("fork").Pool(processes, initializer=_init, initargs=(cfg, eso, imm_every)) as pool:
            rows = pool.map(_one, jobs, chunksize=2)
        summ = summarise(rows)
        results.append({"name": name, "imm": asdict(cfg), "eso_bandwidth_hz": eso, "summary": summ, "rows": rows})
        print(name, json.dumps(summ), "(%.0f s)" % (time.time() - t0), flush=True)
    out = {"calibration_tag": tag, "settings": results, "provenance": stamp()}
    suffix = "" if not settings_file else "_" + Path(settings_file).stem.replace("phase3_replay_", "")
    (ROOT / "results" / f"phase3_replay_{tag}{suffix}.json").write_text(json.dumps(out, default=str))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2, sys.argv[3] if len(sys.argv) > 3 else None)

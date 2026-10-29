"""Evaluation campaign of Phase 5 (configs/evaluation_phase5.yaml): methods, run list, one run
of one method with its metrics and its stored trace.

Every method runs on the same seeds, so the plant draw, the sensor noise, the fault and its onset
of a run are the same for all of them and the comparisons are paired. A run's outcome is fixed by
its seed and its method; the trace written here can be produced again at any time from the two.

Methods (specification, Sec. 10):
  P    proposed controller (frozen Phase 3 configuration)
  A1   no probability-aware tightening: the MPC switches only after a declaration
  A2   no re-planning: after a declaration the nominal reference is kept
  A5   planner solved online after the declaration, with its latency, instead of the library
  B0   no fault tolerance
  B1   oracle: the true fault declared at its onset
  B2   threshold detection on the winch quantities, computed-torque law with tension distribution
  B3   learned classifier in place of the IMM probabilities for the declaration; the MPC switches
       after it (as A1)
  B3T  learned classifier in place of the IMM probabilities for the declaration and for the fault
       scenarios of the MPC (as P); secondary arm
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
VARIANT = {"P": "proposed", "A1": "no_hedging", "A2": "no_replanning", "A5": "proposed", "B0": "no_ftc",
           "B1": "oracle", "B2": "b2", "B3": "b3", "B3T": "b3_tight"}
DIAGNOSING = ("P", "A1", "A2", "A5", "B2", "B3", "B3T")      # methods that declare from data
N = 8

_W = {}


# ------------------------------------------------------------------------------------ protocol
def load_protocol(path=ROOT / "configs" / "evaluation_phase5.yaml"):
    import hashlib
    text = Path(path).read_bytes()
    return yaml.safe_load(text), hashlib.sha256(text).hexdigest()[:16]


def run_list(protocol):
    """All runs of the campaign: dicts with run (index), group, trajectory, seed, noise factor,
    fault and t_end, in the order in which they are executed (seed by seed)."""
    from ..learning.protocol import N_CABLES
    d = protocol["runs"]
    runs = []
    lo, hi = d["onset"]
    for ti, traj in enumerate(d["trajectories"]):
        base = int(d["seed_base"]) + 10000 * ti
        j = 0
        for rep in range(int(d["repetitions"])):
            noise = float(d["noise"][(rep // 2) % len(d["noise"])])
            for kind in d["fault_types"]:
                value = d["values"][kind][rep % len(d["values"][kind])]
                for cable in range(N_CABLES):
                    seed = base + j
                    onset = float(np.round(np.random.default_rng(seed + 500000).uniform(lo, hi), 3))
                    runs.append({"group": "fault", "trajectory": traj, "seed": seed, "rep": rep, "noise_factor": noise,
                                 "fault": [kind, cable, onset, float(value)],
                                 "t_end": float(min(d["duration"], onset + float(d["after_onset"])))})
                    j += 1
        ff_base = int(d["seed_base"]) + 20000 + 10000 * ti
        for r in range(int(d["fault_free"])):
            runs.append({"group": "fault_free", "trajectory": traj, "seed": ff_base + r, "rep": r,
                         "noise_factor": float(d["noise"][(r // 2) % len(d["noise"])]), "fault": None,
                         "t_end": float(d["duration"])})
    order = sorted(range(len(runs)), key=lambda i: (runs[i]["rep"], runs[i]["group"] != "fault", i))
    out = []
    for k, i in enumerate(order):
        r = dict(runs[i])
        r["run"] = k
        out.append(r)
    return out


def jobs_for(protocol, runs):
    """(method, run) pairs of the campaign, by the method lists of the protocol."""
    m = protocol["methods"]
    sub = m.get("subset", {}) or {}
    jobs = []
    for r in runs:
        if r["group"] == "fault_free":
            names = [x for x in m["fault_free"] if r["rep"] < sub.get(x, {}).get("fault_free", 10 ** 9)]
        else:
            kind = r["fault"][0]
            names = [x for x in m["fault"] if kind in m["fault_types_of"].get(x, [kind])
                     and r["rep"] < sub.get(x, {}).get("fault_repetitions", 10 ** 9)]
        for name in names:
            jobs.append((name, r["run"]))
    return jobs


def frozen_hashes(protocol):
    """Hashes of everything a campaign result depends on (results/phase5_freeze.json)."""
    import hashlib
    from ..learning.protocol import controller_digest, learning_digest

    def sha(path):
        return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()[:16]
    b3 = yaml.safe_load((ROOT / protocol["b3"]["file"]).read_text())
    return {"protocol": sha("configs/evaluation_phase5.yaml"),
            "controller_config": sha("configs/" + protocol["controller"]["file"]),
            "b3_config": sha(protocol["b3"]["file"]), "b3_model": sha(b3["deployed_model"]["file"]),
            "b2_config": sha(protocol["b2"]["file"]),
            "recovery_library": sha("results/phase3_recovery_library.npz"),
            "safe_poses": sha("results/phase3_safe_poses.npz"),
            "controller_digest": controller_digest(), "learning_digest": learning_digest()}


def check_frozen(protocol):
    """Raise if anything the campaign depends on changed after results/phase5_freeze.json."""
    path = ROOT / "results" / "phase5_freeze.json"
    if not path.exists():
        return
    frozen = json.loads(path.read_text())["hashes"]
    now = frozen_hashes(protocol)
    changed = {k: (frozen.get(k), v) for k, v in now.items() if frozen.get(k) != v}
    if changed:
        raise RuntimeError(f"files frozen for Phase 5 have changed: {changed}")


# ------------------------------------------------------------------------------------ workers
def init_worker(protocol_path, threads=1, check=True):
    from ..control.config import load_controller
    from ..sim import batch
    protocol, _ = load_protocol(protocol_path)
    if check:
        check_frozen(protocol)
    cfgs, _ = load_controller(protocol["controller"]["file"])
    batch._init(cfgs["imm"], cfgs["mpc"], cfgs["planner"], cfgs["ftc"], True, cfgs["recovery"])
    _W.update(batch._STATE)
    _W["protocol"] = protocol
    _W["ftc"] = cfgs["ftc"]
    _W["threads"] = threads


def _b3(protocol):
    if "b3" not in _W:
        import torch
        from ..learning.train import load_model
        torch.set_num_threads(_W["threads"])
        c = yaml.safe_load((ROOT / protocol["b3"]["file"]).read_text())
        model, blob = load_model(ROOT / c["deployed_model"]["file"])
        _W["b3"] = (model, int(c["params"]["window"]), float(c["deployed_model"]["temperature"]),
                    float(c["declaration"]["mu_iso"]), int(c["declaration"]["n_c"]), float(c["t_from"]))
    return _W["b3"]


def _b2(protocol):
    if "b2" not in _W:
        c = yaml.safe_load((ROOT / protocol["b2"]["file"]).read_text())
        post = (tuple(c["ct_kp_post"]), tuple(c["ct_kd_post"])) if "ct_kp_post" in c else None
        _W["b2"] = (float(c["theta_break"]), float(c["theta_actuator"]), int(c["n_c"]), float(c["t_from"]),
                    (tuple(c["ct_kp"]), tuple(c["ct_kd"])), post)
    return _W["b2"]


def method_config(method, protocol):
    """FTC settings, diagnosis factory and further controller arguments of a method."""
    from ..estimation.alternatives import ClassifierDiagnosis, ThresholdDiagnosis
    from ..learning.features import FeatureStream
    cfg = replace(_W["ftc"], variant=VARIANT[method])
    factory, extra = None, {}
    if method == "A5":
        cfg = replace(cfg, planner_source="online")
    if method in ("B3", "B3T"):
        model, w, temp, mu_iso, n_c, t_from = _b3(protocol)
        cfg = replace(cfg, mu_declare=mu_iso, n_confirm=n_c)

        def factory(robot, cw, l_u0, ref, c, dt, t_nom):
            fs = FeatureStream(robot, cw, l_u0, ref, c.eso_bandwidth_hz, c.imm_every, dt, t_nom)
            return ClassifierDiagnosis(fs, model, w, temp, t_from)
    if method == "B2":
        th_b, th_a, n_c, t_from, gains, gains_post = _b2(protocol)
        cfg = replace(cfg, mu_declare=0.5, n_confirm=n_c)
        extra["ct_gains"] = gains
        if gains_post is not None:
            extra["ct_gains_post"] = gains_post

        def factory(robot, cw, l_u0, ref, c, dt, t_nom):
            fs = FeatureStream(robot, cw, l_u0, ref, c.eso_bandwidth_hz, c.imm_every, dt, t_nom)
            return ThresholdDiagnosis(fs, th_b, th_a, t_from)
    return cfg, factory, extra


def run_job(job):
    """job: (method, run dict, trace path). Returns the metrics row."""
    from ..sim.episode import EpisodeSpec, run_episode
    method, r, trace_path = job
    protocol = _W["protocol"]
    spec = EpisodeSpec(trajectory=r["trajectory"], seed=r["seed"], noise_factor=r["noise_factor"],
                       fault=tuple(r["fault"]) if r["fault"] else None, t_end=r["t_end"], variant=VARIANT[method])
    cfg, factory, extra = method_config(method, protocol)
    t0 = time.time()
    res, ctrl = run_episode(spec, _W["imm"], _W["mpc"], _W["planner"], _W["safe"], cfg, keep_log=True,
                            alt_factory=factory, return_controller=True, ctrl_kwargs=extra)
    log, cl = res.pop("log"), res.pop("ctrl_log")
    row = {"method": method, "run": r["run"], "group": r["group"], "trajectory": r["trajectory"], "seed": r["seed"],
           "rep": r["rep"], "noise_factor": r["noise_factor"], "fault": r["fault"]}
    row.update(evaluate(spec, log, cl, ctrl, protocol, trace_path))
    row["base_metrics"] = {k: v for k, v in res.items() if k not in ("variant", "trajectory", "fault", "seed")}
    row["seconds"] = time.time() - t0
    return row


# ------------------------------------------------------------------------------------ metrics
def _rot_angle(q, q_ref):
    from ..model.kinematics import rot_zyx
    out = np.empty(len(q))
    for k in range(len(q)):
        r = rot_zyx(q_ref[k, 3:]).T @ rot_zyx(q[k, 3:])
        out[k] = np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))
    return out


def _win(t, a, b):
    return (t >= a - 1e-12) & (t < b - 1e-12)


def evaluate(spec, log, cl, ctrl, protocol, trace_path=None):
    """Metrics of one run (specification, Sec. 9) and its trace at the estimator rate."""
    from ..params import load_cable_winch, load_robot
    robot, cw = load_robot(), load_cable_winch()
    ev = protocol["metrics"]
    every = int(ev["trace_every"])                       # inner samples per trace sample
    t_all = log["t"]
    K = len(t_all)
    k5 = np.arange(0, K, every)
    t = t_all[k5]
    onset = spec.fault[2] if spec.fault is not None else np.inf
    t_dec = ctrl.declared[1] if ctrl.declared is not None else np.inf
    # active reference: nominal before a declaration, the controller's reference (plan, hold) after
    q_nom = ctrl.reference(t)[0]
    q_act = q_nom.copy()
    after = t >= t_dec - 1e-12
    if after.any():
        q_act[after] = ctrl._reference(t[after])[0]
    q = log["q"][k5]
    e_p = np.linalg.norm(q[:, :3] - q_act[:, :3], axis=1)
    e_R = _rot_angle(q, q_act)
    e_p_nom = np.linalg.norm(q[:, :3] - q_nom[:, :3], axis=1)
    tension = log["tension"][k5]
    broken = np.zeros_like(tension, bool)
    if spec.fault is not None and spec.fault[0] == "break":
        broken[t >= onset - 1e-12, spec.fault[1]] = True
    intact = np.where(broken, np.nan, tension)
    margin = np.nanmin(np.minimum(intact - robot.t_min, robot.t_max - intact), axis=1)
    outside = np.nan_to_num(margin, nan=0.0) < -1e-9
    t_arm = float(ev["t_from"])
    windows = {"pre": (t_arm, min(onset, t[-1] + 1e-9)),
               "transient": (onset, onset + float(ev["transient"])),
               "post": (onset + float(ev["transient"]), np.inf)}
    out = {"t_end": float(t_all[-1]), "lost": log["stopped"] is not None,
           "stopped": None if log["stopped"] is None else log["stopped"][0],
           "max_tilt_deg": float(np.rad2deg(np.abs(log["q"][:, 3:]).max()))}
    for name, (a, b) in windows.items():
        m = _win(t, a, b)
        if not m.any():
            continue
        out[f"{name}_rmse_p_mm"] = float(1e3 * np.sqrt(np.mean(e_p[m] ** 2)))
        out[f"{name}_max_p_mm"] = float(1e3 * e_p[m].max())
        out[f"{name}_rmse_R_deg"] = float(np.rad2deg(np.sqrt(np.mean(e_R[m] ** 2))))
        out[f"{name}_min_margin_N"] = float(np.nanmin(margin[m]))
        out[f"{name}_outside_pct"] = float(100 * np.mean(outside[m]))
    # declaration
    out.update({"declared": None, "t_declare": None, "delay_s": None, "correct": None, "false_alarm": False,
                "missed": False})
    if ctrl.declared is not None:
        j, td = ctrl.declared
        out["declared"], out["t_declare"] = int(j), float(td)
        if spec.fault is None or td < onset:
            out["false_alarm"] = True
        else:
            kind = "break" if spec.fault[0] == "break" else "actuator"
            from ..sim.episode import mode_index
            out["delay_s"] = float(td - onset)
            out["correct"] = bool(j == mode_index(N, kind, spec.fault[1]))
    elif spec.fault is not None and spec.variant not in ("no_ftc",):
        out["missed"] = True
    out["armed_time_s"] = float(max(0.0, min(onset, t_all[-1]) - t_arm))
    # plan
    if ctrl.plan is not None:
        out["plan_branch"] = int(ctrl.plan.branch)
        out["plan_kappa"] = float(ctrl.plan.kappa)
        if ctrl.plan.target is not None:
            out["target_error_mm"] = float(1e3 * np.linalg.norm(log["q"][-1, :3] - ctrl.plan.target[:3]))
    # deficit estimate of an actuator fault, after the declaration
    if spec.fault is not None and spec.fault[0] != "break" and "deficit" in cl and ctrl.declared is not None:
        from ..sim.episode import controller_start
        i, value = spec.fault[1], spec.fault[3]
        # the drive reports tau_drive (the measured torque); the winch receives
        # tau_act = (1 - rho) tau_drive + f (cdpr_ft/model/plant.py), so the deficit the IMM estimates,
        # in tension units, is (tau_drive - tau_act) / transmission
        rho = value if spec.fault[0] == "loe" else 0.0
        f = value * cw.transmission * controller_start(spec, robot, cw)[3][i] if spec.fault[0] == "bias" else 0.0
        dtrue = (rho * log["tau_drive"][:, i] - f) / cw.transmission
        ti = np.asarray(cl["t_imm"])
        m = ti >= ctrl.declared[1] + float(ev["deficit_settle"])
        if m.any():
            ki = np.clip(np.round(ti[m] / ctrl.dt).astype(int), 0, K - 1)    # sample of the reading
            err = np.asarray(cl["deficit"])[m, i] - dtrue[ki]
            out["deficit_rmse_N"] = float(np.sqrt(np.mean(err ** 2)))
    # computation
    for key in ("mpc_seconds", "imm_seconds"):
        v = np.asarray(cl.get(key, []), float)
        if len(v):
            out[key] = {"mean": float(v.mean()), "p99": float(np.percentile(v, 99)), "max": float(v.max()), "n": int(len(v))}
    if ctrl.alt is not None and hasattr(ctrl.alt, "seconds") and ctrl.alt.seconds:
        v = np.asarray(ctrl.alt.seconds)
        out["classifier_seconds"] = {"mean": float(v.mean()), "p99": float(np.percentile(v, 99)), "max": float(v.max())}
    if trace_path is not None:
        tr = {"t": t.astype(np.float32), "q": q.astype(np.float32), "q_ref": q_act.astype(np.float32),
              "e_p": e_p.astype(np.float32), "e_p_nominal": e_p_nom.astype(np.float32), "e_R": e_R.astype(np.float32),
              "tension": tension.astype(np.float32), "margin": margin.astype(np.float32),
              "t_d": np.asarray(cl["t_d"])[np.minimum(k5, len(cl["t_d"]) - 1)].astype(np.float32),
              "t_hat": np.asarray(cl["t_hat"])[np.minimum(k5, len(cl["t_hat"]) - 1)].astype(np.float32),
              "t_imm": np.asarray(cl["t_imm"], np.float32), "mu": np.asarray(cl["mu"], np.float32),
              "deficit": np.asarray(cl.get("deficit", []), np.float32),
              "t_mpc": np.asarray(cl.get("t_mpc", []), np.float32),
              "mpc_seconds": np.asarray(cl.get("mpc_seconds", []), np.float32),
              "mpc_ok": np.asarray(cl.get("mpc_ok", []), bool),
              "mpc_slots": np.asarray(cl.get("mpc_slots", []), np.int8),
              "declared": np.array([-1 if ctrl.declared is None else ctrl.declared[0],
                                    np.nan if ctrl.declared is None else ctrl.declared[1]], np.float64)}
        if "mu_alt" in cl:
            tr["mu_alt"] = np.asarray(cl["mu_alt"], np.float32)
        Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(trace_path, **tr)
    return out


def json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "__dataclass_fields__"):
        return asdict(o)
    return str(o)


def write_row(f, row):
    f.write(json.dumps(row, default=json_default) + "\n")
    f.flush()

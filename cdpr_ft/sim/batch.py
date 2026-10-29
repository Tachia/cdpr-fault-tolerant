"""Run many episodes in parallel worker processes.

Each worker builds the estimator, the MPC and the planner once (the compiled functions are
loaded from build/), then runs its share of the episode specifications. Results are returned
without the full simulation logs, but with the mode probabilities at the estimator rate,
which the threshold analysis (ROC) needs.
"""
from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import asdict

import numpy as np

_STATE = {}


def _init(imm_cfg, mpc_cfg, planner_cfg, ftc_cfg, use_mpc, recovery_cfg=None):
    from ..control.mpc import ScenarioMPC
    from ..control.planner import Planner, RecoveryLibrary
    from ..estimation.imm import IMM
    from ..params import load_cable_winch, load_robot
    from pathlib import Path
    robot, cw = load_robot(), load_cable_winch()
    safe_file = Path(__file__).resolve().parents[2] / "results" / "phase3_safe_poses.npz"
    safe = {}
    if safe_file.exists():
        d = np.load(safe_file)
        safe = {i: (d[f"pos{i}"], d[f"ori{i}"]) for i in range(robot.n) if f"pos{i}" in d.files}
    lib_file = Path(__file__).resolve().parents[2] / "results" / "phase3_recovery_library.npz"
    library = RecoveryLibrary(lib_file) if lib_file.exists() else None
    _STATE.update(imm=IMM(robot, cw.EA, imm_cfg),
                  mpc=ScenarioMPC(robot, mpc_cfg) if use_mpc else type("NoMPC", (), {"warm": {}})(),
                  planner=Planner(robot, safe, planner_cfg, EA=cw.EA, library=library,
                                  **({"recovery_cfg": recovery_cfg} if recovery_cfg is not None else {})),
                  safe=safe, ftc=ftc_cfg)


def _run(job):
    from ..sim.episode import run_episode
    spec, record_path = job if isinstance(job, tuple) else (job, None)
    t0 = time.time()
    res = run_episode(spec, _STATE["imm"], _STATE["mpc"], _STATE["planner"], _STATE["safe"], _STATE["ftc"],
                      keep_log=True)
    cl = res.pop("ctrl_log")
    log = res.pop("log")
    if record_path is not None and "theta" in cl:
        np.savez_compressed(record_path, theta=cl["theta"], tau=cl["tau"], td_in=cl["td_in"],
                            t=log["t"], q=log["q"].astype(np.float32), tension=log["tension"].astype(np.float32))
    for key in ("plan",):
        if key in cl:
            res[key] = cl[key].tolist() if hasattr(cl[key], "tolist") else cl[key]
    res["mpc_slots_hist"] = np.bincount(cl["mpc_slots"], minlength=4).tolist() if len(cl["mpc_slots"]) else []
    for key in ("mpc_seconds", "imm_seconds"):
        if key in cl and len(cl[key]):
            v = np.asarray(cl[key])
            res[key] = {"mean": float(v.mean()), "p99": float(np.percentile(v, 99)), "max": float(v.max()), "n": int(len(v))}
    res["t_imm"] = cl["t_imm"].astype(np.float32)
    res["mu"] = cl["mu"].astype(np.float32)
    res["seconds"] = time.time() - t0
    res["t_end"] = float(log["t"][-1])
    res["spec"] = asdict(spec)
    return res


def run_batch(specs, imm_cfg, mpc_cfg, planner_cfg, ftc_cfg, processes=2, use_mpc=True, record_paths=None,
              recovery_cfg=None):
    """record_paths: optional list (one per spec) of .npz paths for the readings and commands
    (needs ftc_cfg.record_inputs)."""
    jobs = list(zip(specs, record_paths)) if record_paths is not None else list(specs)
    ctx = mp.get_context("fork")
    with ctx.Pool(processes, initializer=_init,
                  initargs=(imm_cfg, mpc_cfg, planner_cfg, ftc_cfg, use_mpc, recovery_cfg)) as pool:
        return pool.map(_run, jobs, chunksize=1)

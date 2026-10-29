"""Phase 4: the feature dataset from the recorded runs (configs/learning_phase4.yaml).

For every run of scripts/phase4_generate.py the classifier inputs are computed from the recorded
readings and commands with cdpr_ft.learning.features (the computation the controller would make
online), the true mode is attached to every estimator cycle, and the IMM's most probable mode
and its probability on the same cycles are added for the comparison.
Output: results/phase4_data/dataset.npz (float16 features, about 25 MB) and runs.json.
Usage: python scripts/phase4_dataset.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdpr_ft.control.config import load_controller  # noqa: E402
from cdpr_ft.learning.declare import top_and_p  # noqa: E402
from cdpr_ft.learning.features import CHANNELS, SCALES, features_from_record  # noqa: E402
from cdpr_ft.learning.protocol import controller_digest, load_protocol, run_list  # noqa: E402
from cdpr_ft.params import inner_rate, load_cable_winch, load_robot  # noqa: E402
from cdpr_ft.provenance import stamp  # noqa: E402
from cdpr_ft.sim.episode import controller_start  # noqa: E402

WORK = ROOT / "results" / "phase4_work"
OUT = ROOT / "results" / "phase4_data"


def main():
    protocol, phash = load_protocol()
    runs = run_list(protocol)
    cfgs, cfg_hash = load_controller(protocol["data"]["controller"])
    ftc = cfgs["ftc"]
    robot, cw = load_robot(), load_cable_winch()
    dt = 1.0 / inner_rate()
    xs, ys, rs, ts, tops, ps, table = [], [], [], [], [], [], []
    offset = 0
    t0 = time.time()
    for r in runs:
        res = json.loads((WORK / f"run{r.index:04d}.json").read_text())
        with np.load(WORK / f"run{r.index:04d}.npz") as z:
            rec = {k: z[k] for k in ("theta", "tau", "td_in")}
        with np.load(WORK / f"run{r.index:04d}_imm.npz") as z:
            t_imm, mu = z["t_imm"].astype(np.float64), z["mu"]
        _, ref, _, t_nom, l_u0 = controller_start(r.spec, robot, cw)
        t, f = features_from_record(rec, robot, cw, l_u0, ref, ftc.eso_bandwidth_hz, ftc.imm_every, dt, t_nom)
        assert len(t) == len(t_imm) and np.max(np.abs(t - t_imm)) < 1e-6, f"time lines differ in run {r.index}"
        onset = r.spec.fault[2] if r.spec.fault else np.inf
        y = np.where(t >= onset - 1e-9, r.label, 0)
        top, p = top_and_p(mu)
        n = len(t)
        xs.append(f.reshape(n, -1).astype(np.float16))      # channel c = 8 * kind + winch
        ys.append(y.astype(np.int8))
        rs.append(np.full(n, r.index, np.int16))
        ts.append(t.astype(np.float32))
        tops.append(top)
        ps.append(p.astype(np.float32))
        table.append({"index": r.index, "split": r.split, "trajectory": r.spec.trajectory, "seed": r.spec.seed,
                      "noise_factor": r.spec.noise_factor, "fault": r.spec.fault, "label": r.label,
                      "offset": offset, "length": n, "lost": res["lost"], "stopped": res["stopped"],
                      "t_end": res["t_end"], "max_tilt_deg": res["max_tilt_deg"], "seconds": res["seconds"]})
        offset += n
        if r.index % 50 == 0:
            print(f"run {r.index}, {time.time() - t0:.0f} s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "dataset.npz", x=np.concatenate(xs), y=np.concatenate(ys), run=np.concatenate(rs),
                        t=np.concatenate(ts), imm_top=np.concatenate(tops), imm_p=np.concatenate(ps))
    meta = json.loads((WORK / "meta.json").read_text())
    meta = meta if isinstance(meta, list) else [meta]
    assert len({m["controller_digest"] for m in meta}) == 1 and len({m["controller_config_sha256_16"] for m in meta}) == 1
    (OUT / "runs.json").write_text(json.dumps({
        "channels": [f"{c}_{i + 1}" for c in CHANNELS for i in range(robot.n)],
        "scales": dict(zip(CHANNELS, SCALES.tolist())),
        "protocol_sha256_16": phash, "generation": meta,
        "controller_config_sha256_16": cfg_hash, "controller_digest": controller_digest(),
        "cycles": offset, "runs": table, "provenance": stamp()}, indent=1, default=str))
    print(f"{len(runs)} runs, {offset} cycles, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

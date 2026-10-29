"""Phase 4: closed-loop runs for the learned fault classifier (configs/learning_phase4.yaml).

The frozen controller runs with the variant "no_hedging" and declarations disabled: the MPC uses
no fault information, so the readings carry no trace of the IMM, and the IMM still runs, so
its mode probabilities on the same runs are available for the comparison of Phase 4. The
readings and commands are recorded at 1 kHz (as in the calibration runs) for the feature
extraction of scripts/phase4_dataset.py.
Output, per run i: results/phase4_work/run<i>.npz (records), run<i>.json (metrics),
run<i>_imm.npz (IMM times and mode probabilities); results/phase4_work/meta.json (one entry per
invocation: protocol hash, controller hashes, library versions). These files
are not kept in the repository (about 0.6 GB); scripts/phase4_dataset.py condenses them.
The script is resumable: runs whose three files exist are skipped.
Usage: python scripts/phase4_generate.py [processes] [first index] [last index, exclusive]
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdpr_ft.control.config import load_controller  # noqa: E402
from cdpr_ft.learning.protocol import controller_digest, load_protocol, run_list  # noqa: E402
from cdpr_ft.provenance import stamp  # noqa: E402
from cdpr_ft.sim.batch import run_batch  # noqa: E402

WORK = ROOT / "results" / "phase4_work"
CHUNK = 16


def done(i):
    return all((WORK / f"run{i:04d}{s}").exists() for s in (".npz", ".json", "_imm.npz"))


def main(processes=2, first=0, last=None):
    protocol, phash = load_protocol()
    runs = run_list(protocol)
    last = len(runs) if last is None else min(last, len(runs))
    WORK.mkdir(parents=True, exist_ok=True)
    cfgs, cfg_hash = load_controller(protocol["data"]["controller"])
    ftc = replace(cfgs["ftc"], mu_declare=float(protocol["data"]["mu_declare"]), record_inputs=True)
    meta = {"runs": [first, last], "protocol_sha256_16": phash, "controller_config_sha256_16": cfg_hash,
            "controller_digest": controller_digest(), "ftc": asdict(ftc), "provenance": stamp()}
    history = json.loads((WORK / "meta.json").read_text()) if (WORK / "meta.json").exists() else []
    history = history if isinstance(history, list) else [history]
    (WORK / "meta.json").write_text(json.dumps(history + [meta], indent=1, default=str))
    todo = [r for r in runs[first:last] if not done(r.index)]
    print(f"{len(todo)} of {last - first} runs to do", flush=True)
    t0 = time.time()
    for c in range(0, len(todo), CHUNK):
        part = todo[c:c + CHUNK]
        res = run_batch([r.spec for r in part], cfgs["imm"], cfgs["mpc"], cfgs["planner"], ftc, processes=processes,
                        record_paths=[str(WORK / f"run{r.index:04d}.npz") for r in part],
                        recovery_cfg=cfgs["recovery"])
        for r, out in zip(part, res):
            np.savez_compressed(WORK / f"run{r.index:04d}_imm.npz", t_imm=out.pop("t_imm"), mu=out.pop("mu"))
            out.update(index=r.index, split=r.split)
            (WORK / f"run{r.index:04d}.json").write_text(json.dumps(out, default=str))
        print(f"{c + len(part)}/{len(todo)} runs, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:4]))

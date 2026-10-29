"""Phase 4: final training of the fault classifier with the best hyperparameters of the search.

Five seeds on the training split, early stopping on the validation macro-F1; the deployed model
is the seed with the highest validation macro-F1 (protocol). Training-size curve: the same
configuration trained on the first share f of the training runs, in the order (repetition,
cable, fault), with fault-free runs in proportion, seeds 0-2, scored on validation.
Output: results/phase4_models/seed<s>.pt, results/phase4_training.json
Usage: python scripts/phase4_train.py
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cdpr_ft.learning.data import load  # noqa: E402
from cdpr_ft.learning.protocol import load_protocol  # noqa: E402
from cdpr_ft.learning.train import save, train  # noqa: E402
from cdpr_ft.provenance import stamp  # noqa: E402
from phase4_search import versions  # noqa: E402

MODELS = ROOT / "results" / "phase4_models"


def subset(ds, fraction, reps_per_group):
    """First ceil(fraction N) fault runs in the order (repetition, cable, fault), and the first
    ceil(fraction M) fault-free runs."""
    runs = ds.runs_of("train")
    faulty = [r for r in runs if r["fault"]]
    free = [r for r in runs if not r["fault"]]
    # runs are listed by (cable, fault, repetition); the repetition is the position in the group
    keyed = [((i % reps_per_group), i // reps_per_group, r) for i, r in enumerate(faulty)]
    keyed.sort(key=lambda x: (x[0], x[1]))
    n_f, n_0 = math.ceil(fraction * len(faulty)), math.ceil(fraction * len(free))
    return [x[2] for x in keyed[:n_f]] + free[:n_0]


def main():
    protocol, phash = load_protocol()
    tr, f, s = protocol["training"], protocol["features"], protocol["search"]
    search = json.loads((ROOT / "results" / "phase4_search.json").read_text())
    params = search["best"]["params"]
    ds = load()
    MODELS.mkdir(parents=True, exist_ok=True)
    common = dict(t_from=float(f["t_from"]), m0_stride=int(f["m0_stride_train"]),
                  fault_stride=int(f["fault_stride_train"]), threads=int(tr["threads"]),
                  batch=int(s["batch"]))
    out = {"protocol_sha256_16": phash, "params": params, "seeds": {}, "size_curve": [], "versions": versions()}
    t0 = time.time()
    for seed in tr["seeds"]:
        print(f"seed {seed}", flush=True)
        res = train(params, ds, seed=int(seed), max_epochs=int(tr["max_epochs"]), patience=int(tr["patience"]), **common)
        path = MODELS / f"seed{seed}.pt"
        save(path, params, res, {"seed": int(seed), "best_epoch": res["best_epoch"],
                                 "best_val_macro_f1": res["best_val_macro_f1"]})
        out["seeds"][str(seed)] = {k: res[k] for k in ("best_epoch", "best_val_macro_f1", "history", "n_parameters",
                                                       "n_train_windows", "n_val_windows", "seconds")}
        out["seeds"][str(seed)]["file"] = str(path.relative_to(ROOT))
        out["seeds"][str(seed)]["sha256_16"] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    best_seed = max(out["seeds"], key=lambda k: out["seeds"][k]["best_val_macro_f1"])
    out["deployed_seed"] = int(best_seed)
    reps = len(protocol["data"]["T1"]["split"]["train"])
    for frac in tr["size_curve"]["fractions"]:
        for seed in (0, 1, 2):
            if frac == 1.0:
                h = out["seeds"][str(seed)]
                out["size_curve"].append({"fraction": 1.0, "seed": seed, "runs": len(ds.runs_of("train")),
                                          "best_val_macro_f1": h["best_val_macro_f1"], "best_epoch": h["best_epoch"]})
                continue
            runs = subset(ds, frac, reps)
            print(f"size {frac} ({len(runs)} runs), seed {seed}", flush=True)
            res = train(params, ds, seed=seed, max_epochs=int(tr["max_epochs"]), patience=int(tr["patience"]),
                        train_runs=runs, **common)
            out["size_curve"].append({"fraction": frac, "seed": seed, "runs": len(runs),
                                      "best_val_macro_f1": res["best_val_macro_f1"], "best_epoch": res["best_epoch"],
                                      "history": res["history"]})
    out["seconds"] = time.time() - t0
    out["provenance"] = stamp()
    (ROOT / "results" / "phase4_training.json").write_text(json.dumps(out, indent=1))
    print("deployed seed", best_seed, {k: v["best_val_macro_f1"] for k, v in out["seeds"].items()})


if __name__ == "__main__":
    main()

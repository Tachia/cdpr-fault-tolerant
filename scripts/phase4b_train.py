"""Phase 4b: training of the classifier on both paths (configs/learning_phase4b.yaml).

The hyperparameters of the Phase 4 classifier (configs/classifier_phase4.yaml), five seeds, the
training runs of T1 (Phase 4) and T2 (Phase 4b), early stopping on the validation macro-F1 over
the validation runs of both paths. The deployed seed is chosen later, by
scripts/phase4b_calibrate.py, from the declaration outcomes on the validation runs.
Output: results/phase4b_models/seed<s>.pt, results/phase4b_training.json
Usage: python scripts/phase4b_train.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cdpr_ft.learning.train import save, train  # noqa: E402
from cdpr_ft.provenance import stamp  # noqa: E402
from phase4_search import versions  # noqa: E402
from phase4b_common import frozen_phase4, load_combined, load_protocol_4b, sha16  # noqa: E402

MODELS = ROOT / "results" / "phase4b_models"


def main():
    protocol, phash = load_protocol_4b()
    tr, f = protocol["training"], protocol["features"]
    params = frozen_phase4()["params"]
    ds = load_combined()
    print({s: sum(1 for r in ds.runs if r["split"] == s) for s in ("train", "val", "test", "test_T2")}, flush=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    out = {"protocol_sha256_16": phash, "params": params, "seeds": {}, "versions": versions(),
           "datasets": {"phase4": sha16(ROOT / "results/phase4_data/dataset.npz"),
                        "phase4b": sha16(ROOT / "results/phase4b_data/dataset.npz")}}
    t0 = time.time()
    for seed in tr["seeds"]:
        print(f"seed {seed}", flush=True)
        res = train(params, ds, seed=int(seed), max_epochs=int(tr["max_epochs"]), patience=int(tr["patience"]),
                    t_from=float(f["t_from"]), m0_stride=int(f["m0_stride_train"]),
                    fault_stride=int(f["fault_stride_train"]), threads=int(tr["threads"]), batch=int(tr["batch"]))
        path = MODELS / f"seed{seed}.pt"
        save(path, params, res, {"seed": int(seed), "best_epoch": res["best_epoch"],
                                 "best_val_macro_f1": res["best_val_macro_f1"], "protocol_sha256_16": phash})
        out["seeds"][str(seed)] = {"file": str(path.relative_to(ROOT)), "sha256_16": sha16(path),
                                   "best_epoch": res["best_epoch"], "best_val_macro_f1": res["best_val_macro_f1"],
                                   "history": res["history"], "n_train_windows": res["n_train_windows"],
                                   "n_val_windows": res["n_val_windows"], "n_parameters": res["n_parameters"],
                                   "seconds": res["seconds"]}
        print(f"  best epoch {res['best_epoch']}, validation macro-F1 {res['best_val_macro_f1']:.4f}, "
              f"{time.time() - t0:.0f} s", flush=True)
    out["provenance"] = stamp()
    (ROOT / "results" / "phase4b_training.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

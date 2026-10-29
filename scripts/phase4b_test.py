"""Phase 4b: the single offline evaluation of the two-path classifier.

Runs only after configs/classifier_phase4b.yaml exists and checks the frozen hashes. On the T1
test runs and the T2 test runs of Phase 4 (never used for training or for any choice of Phase
4b): window metrics and declaration outcomes of the five seeds (each with its temperature and
thresholds) and of the deployed model, of the IMM (frozen thresholds) and of the Phase 4
classifier (trained on T1 only), with 95 % bootstrap intervals over runs for the deployed model
and its paired differences with the IMM and with the Phase 4 classifier.
Output: results/phase4b_test.json (not overwritten unless called with --rerun)
Usage: python scripts/phase4b_test.py [--rerun]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cdpr_ft.learning.declare import run_outcomes  # noqa: E402
from cdpr_ft.learning.train import imm_sequences, load_model, run_sequences, set_determinism, window_metrics  # noqa: E402
from cdpr_ft.provenance import stamp  # noqa: E402
from phase4_test import bootstrap, summary  # noqa: E402
from phase4b_common import load_combined, load_protocol_4b, sequences, sha16  # noqa: E402


def main(rerun=False):
    out_path = ROOT / "results" / "phase4b_test.json"
    if out_path.exists() and not rerun:
        sys.exit("results/phase4b_test.json exists: the test runs are evaluated once (use --rerun to repeat)")
    protocol, phash = load_protocol_4b()
    frozen = yaml.safe_load((ROOT / "configs" / "classifier_phase4b.yaml").read_text())
    h = frozen["hashes"]
    checks = {"protocol": phash == h["protocol"],
              "phase4_dataset": sha16(ROOT / "results/phase4_data/dataset.npz") == h["phase4_dataset"],
              "phase4b_dataset": sha16(ROOT / "results/phase4b_data/dataset.npz") == h["phase4b_dataset"],
              "features_module": sha16(ROOT / "cdpr_ft/learning/features.py") == h["features_module"],
              "models_module": sha16(ROOT / "cdpr_ft/learning/models.py") == h["models_module"]}
    checks.update({f"model_seed{s}": sha16(ROOT / v["file"]) == v["sha256_16"] for s, v in frozen["seed_models"].items()})
    assert all(checks.values()), f"frozen files changed: {checks}"
    set_determinism(0, int(protocol["training"]["threads"]))
    t_from = float(frozen["t_from"])
    p4 = yaml.safe_load((ROOT / "configs" / "classifier_phase4.yaml").read_text())
    p4_model, p4_blob = load_model(ROOT / p4["deployed_model"]["file"])
    p4_thr = (float(p4["declaration"]["mu_iso"]), int(p4["declaration"]["n_c"]))
    ds = load_combined()
    dep = str(frozen["deployed_model"]["seed"])
    out = {"protocol_sha256_16": phash, "classifier_config_sha256_16": sha16(ROOT / "configs/classifier_phase4b.yaml"),
           "checks": checks, "deployed_seed": int(dep), "splits": {}}
    for key in ("test", "test_T2"):
        runs = [r for r in ds.runs if r["split"] == key]
        res = {"runs": len(runs), "seeds": {}}
        imm = imm_sequences(ds, runs, t_from)
        imm_rows = run_outcomes(imm, 0.9999, 2)
        res["imm_run"] = summary(imm_rows)
        old = run_sequences(p4_model, ds, runs, int(p4_blob["params"]["window"]), t_from)
        old_rows = run_outcomes(old, *p4_thr)
        res["phase4_run"] = summary(old_rows)
        for s, info in frozen["seed_models"].items():
            model, blob = load_model(ROOT / info["file"])
            seqs = sequences(model, ds, runs, int(blob["params"]["window"]), t_from, info["temperature"])
            y = np.concatenate([q["y"] for q in seqs])
            pred = np.concatenate([q["top"] for q in seqs]).astype(np.int64)
            rows = run_outcomes(seqs, float(info["mu_iso"]), int(info["n_c"]))
            res["seeds"][s] = {"window": window_metrics(y, pred), "run": summary(rows),
                               "thresholds": [info["mu_iso"], info["n_c"]], "temperature": info["temperature"]}
            if s == dep:
                res["deployed_rows"] = rows
                res["bootstrap_vs_imm"] = bootstrap(rows, imm_rows)
                res["bootstrap_vs_phase4"] = bootstrap(rows, old_rows)
        f1 = [v["window"]["macro_f1"] for v in res["seeds"].values()]
        cr = [v["run"]["correct_rate"] for v in res["seeds"].values()]
        fa = [v["run"]["false_alarm_runs"] for v in res["seeds"].values()]
        res["five_seeds"] = {"macro_f1": [float(np.mean(f1)), float(np.std(f1, ddof=1))],
                             "correct_rate": [float(np.mean(cr)), float(np.std(cr, ddof=1))],
                             "false_alarm_runs": [float(np.mean(fa)), float(np.std(fa, ddof=1))]}
        out["splits"][key] = res
        d = res["seeds"][dep]["run"]
        print(key, "deployed:", {k: d[k] for k in ("false_alarm_runs", "correct", "missed", "wrong",
                                                   "break_median_delay", "loe_median_delay", "bias_median_delay")},
              "| phase 4 model:", {k: res["phase4_run"][k] for k in ("false_alarm_runs", "correct")},
              "| IMM:", {k: res["imm_run"][k] for k in ("false_alarm_runs", "correct")}, flush=True)
    out["provenance"] = stamp()
    out_path.write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main("--rerun" in sys.argv)

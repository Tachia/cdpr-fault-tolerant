"""Phase 4: write the frozen configuration of the learned classifier (baseline B3).

Records the hyperparameters, the deployed model and its file hash, its declaration thresholds,
and the hashes of the protocol, the dataset and the feature code, so that the test evaluation
and Phase 5 can check that they use exactly what was fixed on training and validation data.
Output: configs/classifier_phase4.yaml
Usage: python scripts/phase4_freeze.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdpr_ft.learning.protocol import controller_digest, learning_digest, load_protocol  # noqa: E402


def sha16(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def main():
    protocol, phash = load_protocol()
    training = json.loads((ROOT / "results" / "phase4_training.json").read_text())
    thresholds = json.loads((ROOT / "results" / "phase4_thresholds.json").read_text())
    seed = str(training["deployed_seed"])
    info = training["seeds"][seed]
    assert sha16(ROOT / info["file"]) == info["sha256_16"], "deployed model file changed since training"
    body = {
        "params": training["params"],
        "deployed_model": {"file": info["file"], "sha256_16": info["sha256_16"], "seed": int(seed),
                           "best_epoch": info["best_epoch"], "val_macro_f1": info["best_val_macro_f1"]},
        "declaration": {"mu_iso": thresholds["chosen"]["mu_iso"], "n_c": thresholds["chosen"]["n_c"]},
        "seed_models": {s: {"file": v["file"], "sha256_16": v["sha256_16"],
                            "mu_iso": thresholds["seeds"][s]["chosen"]["mu_iso"],
                            "n_c": thresholds["seeds"][s]["chosen"]["n_c"]} for s, v in training["seeds"].items()},
        "hashes": {"protocol": phash, "dataset": sha16(ROOT / "results/phase4_data/dataset.npz"),
                   "features_module": sha16(ROOT / "cdpr_ft/learning/features.py"),
                   "models_module": sha16(ROOT / "cdpr_ft/learning/models.py"),
                   "learning_digest": learning_digest(), "controller_digest": controller_digest()},
    }
    header = (f"# Frozen learned fault classifier (Phase 4, {date.today().isoformat()}): baseline B3 of the evaluation.\n"
              "# Hyperparameters: results/phase4_search.json (training and validation splits only).\n"
              "# Models: results/phase4_training.json; deployed = seed with the highest validation macro-F1.\n"
              "# Declaration thresholds: results/phase4_thresholds.json (validation runs, rule of the protocol).\n"
              "# Phase 4 seeds: 300000 and up. The test splits are evaluated once, after this file is written.\n")
    path = ROOT / "configs" / "classifier_phase4.yaml"
    path.write_text(header + yaml.safe_dump(body, sort_keys=False))
    print("written", path, "sha256/16", sha16(path))


if __name__ == "__main__":
    main()

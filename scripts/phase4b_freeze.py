"""Phase 4b: the frozen classifier used as baseline B3 in Phase 5.

Writes configs/classifier_phase4b.yaml: the hyperparameters, the five models with their hashes,
temperatures and thresholds, the deployed model with its temperature and thresholds, and the
hashes of the protocol, the datasets and the feature and model code. The Phase 5 controller
reads the deployed model from this file; scripts/phase4b_test.py checks the hashes before its
single evaluation.
Usage: python scripts/phase4b_freeze.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cdpr_ft.learning.protocol import learning_digest  # noqa: E402
from phase4b_common import frozen_phase4, load_protocol_4b, sha16  # noqa: E402

OUT = ROOT / "configs" / "classifier_phase4b.yaml"


def main():
    protocol, phash = load_protocol_4b()
    training = json.loads((ROOT / "results" / "phase4b_training.json").read_text())
    thr = json.loads((ROOT / "results" / "phase4b_thresholds.json").read_text())
    dep = str(thr["deployed_seed"])
    seeds = {}
    for s, info in training["seeds"].items():
        c = thr["seeds"][s]
        assert sha16(ROOT / info["file"]) == info["sha256_16"]
        seeds[s] = {"file": info["file"], "sha256_16": info["sha256_16"], "temperature": c["temperature"],
                    "mu_iso": c["chosen"]["mu_iso"], "n_c": c["chosen"]["n_c"]}
    body = {
        "params": frozen_phase4()["params"],
        "t_from": float(protocol["features"]["t_from"]),
        "deployed_model": {**seeds[dep], "seed": int(dep), "best_epoch": training["seeds"][dep]["best_epoch"],
                           "val_macro_f1": training["seeds"][dep]["best_val_macro_f1"]},
        "declaration": {"mu_iso": seeds[dep]["mu_iso"], "n_c": seeds[dep]["n_c"]},
        "seed_models": seeds,
        "hashes": {"protocol": phash, "phase4_dataset": sha16(ROOT / "results/phase4_data/dataset.npz"),
                   "phase4b_dataset": sha16(ROOT / "results/phase4b_data/dataset.npz"),
                   "features_module": sha16(ROOT / "cdpr_ft/learning/features.py"),
                   "models_module": sha16(ROOT / "cdpr_ft/learning/models.py"),
                   "learning_digest": learning_digest()},
    }
    header = ("# Frozen learned fault classifier trained on both paths (Phase 4b, 2026-10-27): baseline B3 of Phase 5.\n"
              "# Hyperparameters: those of configs/classifier_phase4.yaml (no new search).\n"
              "# Models: results/phase4b_training.json. Temperatures, thresholds and the deployed seed:\n"
              "# results/phase4b_thresholds.json (validation runs of both paths).\n"
              "# The test runs are evaluated once, after this file is written (scripts/phase4b_test.py).\n")
    OUT.write_text(header + yaml.safe_dump(body, sort_keys=False))
    print(OUT.read_text())


if __name__ == "__main__":
    main()

"""Phase 4b: calibration, declaration thresholds and the deployed seed (configs/learning_phase4b.yaml).

For every seed: one temperature fitted on all validation windows of both paths (negative
log-likelihood of the true mode), then the controller's declaration rule on the calibrated
probabilities of every validation run for every pair (mu_iso, N_c) of the grid, the pair chosen
by the lexicographic rule. The deployed seed is the one whose chosen pair gives the best
validation outcome by the same key, the validation macro-F1 breaking ties. No test run is read.
Output: results/phase4b_thresholds.json
Usage: python scripts/phase4b_calibrate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cdpr_ft.learning.declare import evaluate, select, selection_key  # noqa: E402
from cdpr_ft.learning.train import load_model, set_determinism  # noqa: E402
from cdpr_ft.provenance import stamp  # noqa: E402
from phase4b_common import (ece, fit_temperature, load_combined, load_protocol_4b, nll, predict_logits,  # noqa: E402
                            sequences, softmax_t)


def main():
    protocol, phash = load_protocol_4b()
    d, t_from = protocol["declaration"], float(protocol["features"]["t_from"])
    set_determinism(0, int(protocol["training"]["threads"]))
    training = json.loads((ROOT / "results" / "phase4b_training.json").read_text())
    ds = load_combined()
    val = [r for r in ds.runs if r["split"] == "val"]
    out = {"protocol_sha256_16": phash, "seeds": {}}
    for seed, info in training["seeds"].items():
        model, blob = load_model(ROOT / info["file"])
        w = int(blob["params"]["window"])
        idx = ds.windows("val", w, t_from, 1)
        logits = predict_logits(model, ds, idx, w)
        y = ds.y[idx]
        temp, loss = fit_temperature(logits, y)
        p1, pt = softmax_t(logits, 1.0), softmax_t(logits, temp)
        seqs = sequences(model, ds, val, w, t_from, temp)
        table = [evaluate(seqs, float(m), int(n)) for m in d["mu_iso"] for n in d["n_c"]]
        chosen = select(table)
        out["seeds"][seed] = {"temperature": temp, "nll_before": nll(logits, y, 1.0), "nll_after": loss,
                              "ece_before": ece(p1, y), "ece_after": ece(pt, y),
                              "val_macro_f1": info["best_val_macro_f1"], "table": table, "chosen": chosen}
        print(seed, f"T {temp:.3f}, NLL {nll(logits, y, 1.0):.4f} -> {loss:.4f}, ECE {ece(p1, y):.4f} -> {ece(pt, y):.4f}",
              {k: chosen[k] for k in ("mu_iso", "n_c", "false_alarm_runs")},
              {k: (chosen[k]["correct"], chosen[k]["n"], chosen[k]["median_delay"]) for k in ("break", "loe", "bias")},
              flush=True)
    best = min(out["seeds"], key=lambda s: (selection_key(out["seeds"][s]["chosen"])[:3], -out["seeds"][s]["val_macro_f1"]))
    out["deployed_seed"] = int(best)
    out["chosen"] = out["seeds"][best]["chosen"]
    out["provenance"] = stamp()
    print("deployed seed", best, flush=True)
    (ROOT / "results" / "phase4b_thresholds.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

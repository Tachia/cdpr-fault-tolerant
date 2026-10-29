"""Phase 4: declaration thresholds of the classifier, on the validation runs.

The controller's declaration rule is applied to the classifier's mode probabilities on every
validation run, for every pair (mu_iso, N_c) of the protocol grid, and the pair is chosen by the
lexicographic rule of the protocol (cdpr_ft.learning.declare.select). Done for every seed; the
pair of the deployed model is the one the baseline uses. No test run is touched.
Output: results/phase4_thresholds.json
Usage: python scripts/phase4_thresholds.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdpr_ft.learning.data import load  # noqa: E402
from cdpr_ft.learning.declare import evaluate, select  # noqa: E402
from cdpr_ft.learning.protocol import load_protocol  # noqa: E402
from cdpr_ft.learning.train import load_model, run_sequences  # noqa: E402
from cdpr_ft.provenance import stamp  # noqa: E402


def main():
    protocol, phash = load_protocol()
    d, t_from = protocol["declaration"], float(protocol["features"]["t_from"])
    training = json.loads((ROOT / "results" / "phase4_training.json").read_text())
    ds = load()
    val = ds.runs_of("val")
    out = {"protocol_sha256_16": phash, "deployed_seed": training["deployed_seed"], "seeds": {}}
    for seed, info in training["seeds"].items():
        model, blob = load_model(ROOT / info["file"])
        seqs = run_sequences(model, ds, val, int(blob["params"]["window"]), t_from)
        table = [evaluate(seqs, float(m), int(n)) for m in d["mu_iso"] for n in d["n_c"]]
        chosen = select(table)
        out["seeds"][seed] = {"table": table, "chosen": chosen}
        print(seed, {k: chosen[k] for k in ("mu_iso", "n_c", "false_alarm_runs")},
              {k: (chosen[k]["correct"], chosen[k]["n"], chosen[k]["median_delay"]) for k in ("break", "loe", "bias")},
              flush=True)
    out["chosen"] = out["seeds"][str(training["deployed_seed"])]["chosen"]
    out["provenance"] = stamp()
    (ROOT / "results" / "phase4_thresholds.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

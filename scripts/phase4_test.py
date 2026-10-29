"""Phase 4: the single evaluation of the frozen classifier on the test splits.

Runs only after configs/classifier_phase4.yaml exists, and checks that the protocol, the
dataset, the feature and model code and the model files are those that were frozen. On the T1
test runs and on the T2 runs (a path never seen in training):
  window level: accuracy, macro-F1, per-class precision, recall and F1, confusion matrix, for the
    five seeds and the deployed model; the IMM's most probable mode on the same cycles, likewise;
  run level: the controller's declaration rule with each model's validation thresholds, and with
    the frozen IMM thresholds for the IMM: false alarms, missed and wrong declarations, delays;
  intervals: 95 % bootstrap over runs (2000 resamples, seed 0) for the deployed model, the IMM,
    and their paired differences.
Output: results/phase4_test.json. The script refuses to overwrite it unless called with --rerun
(used by reproduce.py to check that a rerun gives the same numbers).
Usage: python scripts/phase4_test.py [--rerun]
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cdpr_ft.control.config import load_controller  # noqa: E402
from cdpr_ft.learning.data import load  # noqa: E402
from cdpr_ft.learning.declare import evaluate, run_outcomes  # noqa: E402
from cdpr_ft.learning.protocol import load_protocol  # noqa: E402
from cdpr_ft.learning.train import imm_sequences, load_model, run_sequences, set_determinism, window_metrics  # noqa: E402
from cdpr_ft.provenance import stamp  # noqa: E402
from phase4_search import versions  # noqa: E402

KINDS = ("break", "loe", "bias")


def sha16(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def summary(rows):
    """Statistics of run outcomes (cdpr_ft.learning.declare.run_outcomes)."""
    fault = [r for r in rows if r["fault"]]
    ok = [r for r in fault if not r["false_alarm"]]
    out = {"runs": len(rows), "false_alarm_runs": sum(r["false_alarm"] for r in rows),
           "fault_runs": len(fault), "correct": sum(r.get("correct", False) for r in ok),
           "missed": sum(not r["declared"] for r in ok), "wrong": sum(r["declared"] and not r["correct"] for r in ok),
           "fault_free_hours": sum(r["free_time"] for r in rows) / 3600.0}
    out["correct_rate"] = out["correct"] / max(len(fault), 1)
    for kind in KINDS:
        d = [r["delay"] for r in ok if r["fault"][0] == kind and r.get("correct")]
        out[f"{kind}_n"] = sum(1 for r in fault if r["fault"][0] == kind)
        out[f"{kind}_correct"] = len(d)
        out[f"{kind}_median_delay"] = float(np.median(d)) if d else None
        out[f"{kind}_max_delay"] = float(np.max(d)) if d else None
    return out


def bootstrap(rows_a, rows_b, n=2000, seed=0):
    """95 % intervals of the run statistics of a (classifier) and b (IMM) and of a - b, resampling
    the runs (the same resample for both, since they are the same runs)."""
    rng = np.random.default_rng(seed)
    keys = ["correct_rate", "false_alarm_runs"] + [f"{k}_median_delay" for k in KINDS]
    m = len(rows_a)
    draws = {k: ([], [], []) for k in keys}
    for _ in range(n):
        idx = rng.integers(0, m, m)
        sa, sb = summary([rows_a[i] for i in idx]), summary([rows_b[i] for i in idx])
        for k in keys:
            a, b = sa[k], sb[k]
            if k == "false_alarm_runs":
                a, b = a / m, b / m
            draws[k][0].append(np.nan if a is None else a)
            draws[k][1].append(np.nan if b is None else b)
            draws[k][2].append(np.nan if a is None or b is None else a - b)
    out = {}
    for k in keys:
        name = "false_alarm_run_rate" if k == "false_alarm_runs" else k
        out[name] = {lab: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
                     for lab, v in zip(("classifier", "imm", "difference"), draws[k])}
    return out


def inference_time(model, w):
    x = torch.zeros(1, 24, w)
    with torch.no_grad():
        for _ in range(50):
            model(x)
        ts = []
        for _ in range(2000):
            t0 = time.perf_counter()
            model(x)
            ts.append(time.perf_counter() - t0)
    ts = np.array(ts)
    return {"median": float(np.median(ts)), "p99": float(np.percentile(ts, 99))}


def main(rerun=False):
    out_path = ROOT / "results" / "phase4_test.json"
    if out_path.exists() and not rerun:
        sys.exit("results/phase4_test.json exists: the test splits are evaluated once (use --rerun to repeat)")
    protocol, phash = load_protocol()
    frozen = yaml.safe_load((ROOT / "configs" / "classifier_phase4.yaml").read_text())
    h = frozen["hashes"]
    checks = {"protocol": phash == h["protocol"], "dataset": sha16(ROOT / "results/phase4_data/dataset.npz") == h["dataset"],
              "features_module": sha16(ROOT / "cdpr_ft/learning/features.py") == h["features_module"],
              "models_module": sha16(ROOT / "cdpr_ft/learning/models.py") == h["models_module"]}
    checks.update({f"model_seed{s}": sha16(ROOT / v["file"]) == v["sha256_16"] for s, v in frozen["seed_models"].items()})
    assert all(checks.values()), f"frozen files changed: {checks}"
    set_determinism(0, int(protocol["training"]["threads"]))
    t_from = float(protocol["features"]["t_from"])
    ftc = load_controller(protocol["data"]["controller"])[0]["ftc"]
    imm_thr = (float(ftc.mu_declare), int(ftc.n_confirm))
    ds = load()
    deployed = str(frozen["deployed_model"]["seed"])
    out = {"protocol_sha256_16": phash, "classifier_config_sha256_16": sha16(ROOT / "configs/classifier_phase4.yaml"),
           "checks": checks, "deployed_seed": int(deployed), "imm_thresholds": imm_thr, "splits": {}}
    for split in protocol["test"]["splits"]:
        key = "test" if split == "T1 test" else "test_T2"
        runs = ds.runs_of(key)
        res = {"runs": len(runs), "fault_runs": sum(1 for r in runs if r["fault"]), "seeds": {}}
        imm = imm_sequences(ds, runs, t_from)
        y = np.concatenate([ds.y[ds.windows(None, 1, t_from, 1, runs=[r])] for r in runs])
        imm_top = np.concatenate([s["top"] for s in imm])
        res["imm_window"] = window_metrics(y, imm_top.astype(np.int64))
        imm_rows = run_outcomes(imm, *imm_thr)
        res["imm_run"] = {"table": evaluate(imm, *imm_thr), "summary": summary(imm_rows)}
        for seed, info in frozen["seed_models"].items():
            model, blob = load_model(ROOT / info["file"])
            w = int(blob["params"]["window"])
            seqs = run_sequences(model, ds, runs, w, t_from)
            yy = np.concatenate([s["y"] for s in seqs])
            assert np.array_equal(yy, y), "window labels and cycle labels differ"
            pred = np.concatenate([s["probs"].argmax(1) for s in seqs])
            rows = run_outcomes(seqs, float(info["mu_iso"]), int(info["n_c"]))
            res["seeds"][seed] = {"window": window_metrics(y, pred),
                                  "run": {"table": evaluate(seqs, float(info["mu_iso"]), int(info["n_c"])),
                                          "summary": summary(rows)},
                                  "thresholds": [info["mu_iso"], info["n_c"]]}
            if seed == deployed:
                res["deployed_rows"] = [{k: v for k, v in r.items()} for r in rows]
                res["imm_rows"] = imm_rows
                res["bootstrap"] = bootstrap(rows, imm_rows)
                if key == "test":
                    out["inference_seconds"] = inference_time(model, w)   # one window, batch 1
        acc = [v["window"]["accuracy"] for v in res["seeds"].values()]
        f1 = [v["window"]["macro_f1"] for v in res["seeds"].values()]
        cr = [v["run"]["summary"]["correct_rate"] for v in res["seeds"].values()]
        fa = [v["run"]["summary"]["false_alarm_runs"] for v in res["seeds"].values()]
        res["five_seeds"] = {"accuracy": [float(np.mean(acc)), float(np.std(acc, ddof=1))],
                             "macro_f1": [float(np.mean(f1)), float(np.std(f1, ddof=1))],
                             "correct_rate": [float(np.mean(cr)), float(np.std(cr, ddof=1))],
                             "false_alarm_runs": [float(np.mean(fa)), float(np.std(fa, ddof=1))]}
        out["splits"][key] = res
        s = res["seeds"][deployed]
        print(key, "deployed: acc %.4f F1 %.4f" % (s["window"]["accuracy"], s["window"]["macro_f1"]),
              s["run"]["summary"], "\n  IMM: acc %.4f F1 %.4f" % (res["imm_window"]["accuracy"], res["imm_window"]["macro_f1"]),
              res["imm_run"]["summary"], flush=True)
    out["versions"] = versions()
    out["provenance"] = stamp()
    out_path.write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main(rerun="--rerun" in sys.argv[1:])

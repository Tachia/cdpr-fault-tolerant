"""Phase 4: hyperparameter search for the fault classifier on the training and validation splits.

Optuna (TPE sampler, seed 0; median pruner) over the space of configs/learning_phase4.yaml,
fixed budget of trials; every trial trains one model with seed 0 and is scored by the best
validation macro-F1 over its epochs. No test run is touched.
Output: results/phase4_search.json
Usage: python scripts/phase4_search.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import optuna
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdpr_ft.learning.data import load  # noqa: E402
from cdpr_ft.learning.protocol import load_protocol  # noqa: E402
from cdpr_ft.learning.train import train  # noqa: E402
from cdpr_ft.provenance import stamp  # noqa: E402


def suggest(trial, space):
    p = {"arch": trial.suggest_categorical("arch", space["arch"]),
         "window": trial.suggest_categorical("window", space["window"])}
    if p["arch"] == "cnn":
        p["cnn_layers"] = trial.suggest_categorical("cnn_layers", space["cnn_layers"])
        p["cnn_channels"] = trial.suggest_categorical("cnn_channels", space["cnn_channels"])
        p["cnn_kernel"] = trial.suggest_categorical("cnn_kernel", space["cnn_kernel"])
    else:
        p["lstm_hidden"] = trial.suggest_categorical("lstm_hidden", space["lstm_hidden"])
        p["lstm_layers"] = trial.suggest_categorical("lstm_layers", space["lstm_layers"])
    p["dropout"] = trial.suggest_float("dropout", *space["dropout"])
    p["lr"] = trial.suggest_float("lr", *space["lr"], log=True)
    p["weight_decay"] = trial.suggest_float("weight_decay", *space["weight_decay"], log=True)
    return p


def versions():
    return {"torch": torch.__version__, "optuna": optuna.__version__}


def main():
    protocol, phash = load_protocol()
    s, f = protocol["search"], protocol["features"]
    threads = int(protocol["training"]["threads"])
    ds = load()
    records = []

    def objective(trial):
        params = suggest(trial, s["space"])
        print(f"trial {trial.number}: {params}", flush=True)

        def on_epoch(epoch, value):
            trial.report(value, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        rec = {"number": trial.number, "params": params}
        records.append(rec)
        try:
            res = train(params, ds, seed=0, max_epochs=int(s["max_epochs"]), patience=int(s["patience"]),
                        t_from=float(f["t_from"]), m0_stride=int(f["m0_stride_train"]),
                        fault_stride=int(f["fault_stride_train"]), threads=threads,
                        batch=int(s["batch"]), on_epoch=on_epoch)
        except optuna.TrialPruned:
            rec["state"] = "pruned"
            raise
        rec.update(state="complete", value=res["best_val_macro_f1"], best_epoch=res["best_epoch"],
                   history=res["history"], n_parameters=res["n_parameters"], seconds=res["seconds"])
        return res["best_val_macro_f1"]

    sampler = optuna.samplers.TPESampler(seed=0)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    t0 = time.time()
    study.optimize(objective, n_trials=int(s["trials"]))
    for t, rec in zip(study.trials, records):
        rec["state"] = t.state.name.lower()
        if t.state.name == "PRUNED":
            rec["last_value"] = t.intermediate_values[max(t.intermediate_values)] if t.intermediate_values else None
            rec["epochs"] = len(t.intermediate_values)
    best = study.best_trial
    out = {"protocol_sha256_16": phash, "best": {"number": best.number, "value": best.value,
                                                 "params": records[best.number]["params"]},
           "trials": records, "seconds": time.time() - t0, "versions": versions(), "provenance": stamp()}
    (ROOT / "results" / "phase4_search.json").write_text(json.dumps(out, indent=1))
    print("best", out["best"])


if __name__ == "__main__":
    main()

"""Training and evaluation of the Phase 4 classifiers (CPU, deterministic)."""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from .data import Dataset, channel_stats
from .models import N_MODES, build, n_parameters


def set_determinism(seed: int, threads: int = 2):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(threads)


def class_weights(y, n=N_MODES):
    """Balanced weights N / (n_classes * count) of the classes present."""
    counts = np.bincount(y, minlength=n).astype(float)
    w = np.where(counts > 0, len(y) / (np.count_nonzero(counts) * np.maximum(counts, 1)), 0.0)
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def predict_proba(model, ds: Dataset, idx, w: int, batch: int = 4096):
    model.eval()
    out = np.zeros((len(idx), N_MODES), np.float32)
    for c in range(0, len(idx), batch):
        xb = torch.from_numpy(ds.batch(idx[c:c + batch], w))
        out[c:c + batch] = torch.softmax(model(xb), dim=1).numpy()
    return out


def window_metrics(y_true, y_pred, n=N_MODES):
    cm = np.zeros((n, n), np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(n), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(n), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(n), where=(prec + rec) > 0)
    present = cm.sum(1) > 0
    return {"accuracy": float(tp.sum() / cm.sum()), "macro_f1": float(f1[present].mean()),
            "precision": prec.tolist(), "recall": rec.tolist(), "f1": f1.tolist(), "confusion": cm.tolist()}


def weighted_ce(probs, y, weights):
    p = np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)
    w = weights.numpy()[y]
    return float(np.sum(-w * np.log(p)) / np.sum(w))


def train(params: dict, ds: Dataset, seed: int, max_epochs: int, patience: int, t_from: float, m0_stride: int,
          threads: int = 2, batch: int = 256, train_runs=None, on_epoch=None, log=print, fault_stride: int = 1):
    """Train one model on the training split (or the given training runs), early stopping on the
    validation macro-F1. Returns the best state and the history."""
    set_determinism(seed, threads)
    w = int(params["window"])
    idx_tr = ds.windows("train", w, t_from, m0_stride, runs=train_runs, fault_stride=fault_stride)
    idx_va = ds.windows("val", w, t_from, 1)
    mean, std = channel_stats(ds, idx_tr, w)
    model = build(params, mean, std)
    weights = class_weights(ds.y[idx_tr])
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=float(params["lr"]), weight_decay=float(params["weight_decay"]))
    rng = np.random.default_rng(seed)
    y_tr = torch.from_numpy(ds.y)
    best = {"f1": -1.0, "epoch": -1, "state": None}
    history = []
    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = rng.permutation(idx_tr)
        total, count = 0.0, 0
        for c in range(0, len(order), batch):
            ib = order[c:c + batch]
            xb = torch.from_numpy(ds.batch(ib, w))
            loss = loss_fn(model(xb), y_tr[ib])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(ib)
            count += len(ib)
        probs = predict_proba(model, ds, idx_va, w)
        pred = probs.argmax(1)
        m = window_metrics(ds.y[idx_va], pred)
        row = {"epoch": epoch, "train_loss": total / count, "val_loss": weighted_ce(probs, ds.y[idx_va], weights),
               "val_macro_f1": m["macro_f1"], "val_accuracy": m["accuracy"], "seconds": time.time() - t0}
        history.append(row)
        if row["val_macro_f1"] > best["f1"]:
            best = {"f1": row["val_macro_f1"], "epoch": epoch,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
        if log:
            log(f"  epoch {epoch:2d}  train {row['train_loss']:.4f}  val {row['val_loss']:.4f}  "
                f"F1 {row['val_macro_f1']:.4f}  ({row['seconds']:.0f} s)")
        if on_epoch is not None:
            on_epoch(epoch, row["val_macro_f1"])
        if epoch - best["epoch"] >= patience:
            break
    model.load_state_dict(best["state"])
    return {"model": model, "best_epoch": best["epoch"], "best_val_macro_f1": best["f1"], "history": history,
            "n_train_windows": int(len(idx_tr)), "n_val_windows": int(len(idx_va)),
            "n_parameters": n_parameters(model), "seconds": time.time() - t0, "mean": mean.tolist(), "std": std.tolist()}


def save(path, params, result, extra=None):
    torch.save({"params": params, "state": result["model"].state_dict(), **(extra or {})}, path)


def load_model(path):
    blob = torch.load(path, weights_only=False)
    model = build(blob["params"])
    model.load_state_dict(blob["state"])
    model.eval()
    return model, blob


def run_sequences(model, ds: Dataset, runs, w: int, t_from: float):
    """Mode probabilities of the classifier at every cycle (t >= t_from) of the given runs, in the
    form used by cdpr_ft.learning.declare: dicts with fault, t, top, p (and the full probabilities)."""
    out = []
    for r in runs:
        idx = ds.windows(None, w, t_from, 1, runs=[r])
        pr = predict_proba(model, ds, idx, w)
        top = pr.argmax(1).astype(np.int8)
        out.append({"index": r["index"], "fault": r["fault"], "t": ds.t[idx], "top": top,
                    "p": pr[np.arange(len(pr)), top].astype(np.float64), "probs": pr, "y": ds.y[idx]})
    return out


def imm_sequences(ds: Dataset, runs, t_from: float):
    """The IMM's most probable mode and its probability on the same cycles."""
    out = []
    for r in runs:
        a, n = r["offset"], r["length"]
        k = np.arange(a, a + n)
        k = k[ds.t[k] >= t_from - 1e-9]
        out.append({"index": r["index"], "fault": r["fault"], "t": ds.t[k], "top": ds.imm_top[k], "p": ds.imm_p[k]})
    return out

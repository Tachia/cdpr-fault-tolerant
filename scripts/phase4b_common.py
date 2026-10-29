"""Phase 4b (configs/learning_phase4b.yaml): the run list of the new T2 runs and the combined dataset.

The new runs are numbered 0..351 in their own folder; in the combined dataset they follow the 528
Phase 4 runs, so their run index there is 528 + their own index and their split is "train" or
"val" like the T1 runs of the same split.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_4B = ROOT / "configs" / "learning_phase4b.yaml"
DATA_4B = ROOT / "results" / "phase4b_data"
N_PHASE4_RUNS = 528


def load_protocol_4b():
    text = PROTOCOL_4B.read_bytes()
    return yaml.safe_load(text), hashlib.sha256(text).hexdigest()[:16]


def run_list_4b(protocol: dict):
    from cdpr_ft.learning.protocol import N_CABLES, Run
    from cdpr_ft.sim.episode import EpisodeSpec
    d = protocol["data"]
    base = int(d["seed_base"])
    faults = [tuple(f) for f in d["faults"]]
    noise = [float(x) for x in d["noise"]]
    cfg = d["T2"]
    reps = int(cfg["repetitions"])
    by_rep = {r: s for s, rs in cfg["split"].items() for r in rs}
    assert sorted(by_rep) == list(range(reps)), "every repetition needs a split"
    runs, j = [], 0

    def onset_of(seed):
        lo, hi = d["onset"]
        return float(np.round(np.random.default_rng(seed + 500000).uniform(lo, hi), 3))

    for cable in range(N_CABLES):
        for kind, value in faults:
            for rep in range(reps):
                seed = base + j
                onset = onset_of(seed)
                spec = EpisodeSpec(trajectory="T2", seed=seed, noise_factor=noise[rep % len(noise)],
                                   fault=(kind, cable, onset, float(value)),
                                   t_end=onset + float(d["after_onset"]), variant=d["variant"])
                runs.append(Run(len(runs), by_rep[rep], spec))
                j += 1
    sizes = cfg["fault_free_split"]
    names = ["train"] * sizes["train"] + ["val"] * sizes["val"]
    assert len(names) == int(cfg["fault_free"])
    for r, name in enumerate(names):
        spec = EpisodeSpec(trajectory="T2", seed=base + 10000 + r, noise_factor=noise[r % len(noise)],
                           variant=d["variant"])
        runs.append(Run(len(runs), name, spec))
    return runs


def load_combined():
    """The Phase 4 dataset with the Phase 4b runs appended (run indices shifted by 528)."""
    from cdpr_ft.learning.data import Dataset, load
    a = load()
    b = load(DATA_4B)
    shift = len(a.x)
    runs = [dict(r) for r in a.runs]
    for r in b.runs:
        r = dict(r)
        r["offset"] += shift
        r["index"] += N_PHASE4_RUNS
        runs.append(r)
    return Dataset(np.concatenate([a.x, b.x]), np.concatenate([a.y, b.y]),
                   np.concatenate([a.run, b.run + N_PHASE4_RUNS]).astype(np.int32),
                   np.concatenate([a.t, b.t]), np.concatenate([a.imm_top, b.imm_top]),
                   np.concatenate([a.imm_p, b.imm_p]), runs)


def split_runs(ds, split):
    """Runs of a combined split: "train" and "val" take both paths; "test" is the Phase 4 T1 test
    split and "test_T2" the Phase 4 T2 runs."""
    return [r for r in ds.runs if r["split"] == split]


def sha16(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def frozen_phase4():
    return yaml.safe_load((ROOT / "configs" / "classifier_phase4.yaml").read_text())


def dump(obj, path):
    Path(path).write_text(json.dumps(obj, indent=1, default=str))


def predict_logits(model, ds, idx, w, batch=4096):
    """Logits (float32) of the windows ending at idx."""
    import torch
    model.eval()
    out = np.zeros((len(idx), 17), np.float32)
    with torch.no_grad():
        for c in range(0, len(idx), batch):
            xb = torch.from_numpy(ds.batch(idx[c:c + batch], w))
            out[c:c + batch] = model(xb).numpy()
    return out


def softmax_t(logits, temperature):
    """Softmax of logits / T in double precision, as ClassifierDiagnosis computes it online."""
    z = logits.astype(np.float64) / float(temperature)
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits, y, temperature):
    z = logits.astype(np.float64) / float(temperature)
    m = z.max(axis=1, keepdims=True)
    lse = (m[:, 0] + np.log(np.exp(z - m).sum(axis=1)))
    return float(np.mean(lse - z[np.arange(len(y)), y]))


def fit_temperature(logits, y, lo=0.05, hi=20.0):
    """Temperature minimising the mean negative log-likelihood (bounded search over log T)."""
    from scipy.optimize import minimize_scalar
    r = minimize_scalar(lambda s: nll(logits, y, np.exp(s)), bounds=(np.log(lo), np.log(hi)), method="bounded",
                        options={"xatol": 1e-6})
    return float(np.exp(r.x)), float(r.fun)


def sequences(model, ds, runs, w, t_from, temperature):
    """Calibrated probabilities per run in the form of cdpr_ft.learning.declare (fault, t, top, p)."""
    out = []
    for r in runs:
        idx = ds.windows(None, w, t_from, 1, runs=[r])
        pr = softmax_t(predict_logits(model, ds, idx, w), temperature)
        top = pr.argmax(1).astype(np.int8)
        out.append({"index": r["index"], "fault": r["fault"], "t": ds.t[idx], "top": top,
                    "p": pr[np.arange(len(pr)), top], "y": ds.y[idx]})
    return out


def ece(probs, y, bins=15):
    """Expected calibration error of the top probability (equal-width bins)."""
    conf = probs.max(axis=1)
    acc = probs.argmax(axis=1) == y
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)

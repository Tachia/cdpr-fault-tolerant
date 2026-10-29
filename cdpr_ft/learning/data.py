"""The Phase 4 dataset: per-run feature sequences and the windows taken from them.

results/phase4_data/dataset.npz holds, for every estimator cycle of every run, one after the
other: the run index, the time, the 24 features (float16), the true mode, and the most probable
mode of the IMM with its probability. results/phase4_data/runs.json lists the runs (split,
specification, offset and length in the arrays, outcome). A window ending at cycle i holds
cycles i-W+1 .. i of the same run; windows end at or after t_from.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results" / "phase4_data"


@dataclass
class Dataset:
    x: np.ndarray        # (S, 24) float32
    y: np.ndarray        # (S,) int64
    run: np.ndarray      # (S,) int32
    t: np.ndarray        # (S,) float64
    imm_top: np.ndarray  # (S,) int8
    imm_p: np.ndarray    # (S,) float64
    runs: list           # run table

    def runs_of(self, split):
        return [r for r in self.runs if r["split"] == split]

    def windows(self, split, w: int, t_from: float, m0_stride: int = 1, runs=None, fault_stride: int = 1):
        """Indices of the last cycles of the windows of a split (or of the given runs); healthy
        windows every m0_stride-th, fault windows every fault_stride-th cycle of a run."""
        out = []
        for r in (runs if runs is not None else self.runs_of(split)):
            a, n = r["offset"], r["length"]
            k = np.arange(a + w - 1, a + n)
            k = k[self.t[k] >= t_from - 1e-9]
            healthy = self.y[k] == 0
            keep = np.ones(len(k), bool)
            if m0_stride > 1:
                keep &= ~healthy | (np.cumsum(healthy) % m0_stride == 1)
            if fault_stride > 1:
                keep &= healthy | (np.cumsum(~healthy) % fault_stride == 1)
            out.append(k[keep])
        return np.concatenate(out) if out else np.zeros(0, int)

    def batch(self, idx, w: int):
        """Inputs (B, 24, W) as float32 for the windows ending at idx."""
        off = idx[:, None] + np.arange(-w + 1, 1)[None, :]
        return np.ascontiguousarray(self.x[off].transpose(0, 2, 1))


def load(folder: Path = DATA) -> Dataset:
    with np.load(folder / "dataset.npz") as z:
        x = z["x"].astype(np.float32)
        y, run, t = z["y"].astype(np.int64), z["run"].astype(np.int32), z["t"].astype(np.float64)
        top, p = z["imm_top"], z["imm_p"].astype(np.float64)
    runs = json.loads((folder / "runs.json").read_text())["runs"]
    return Dataset(x, y, run, t, top, p, runs)


def channel_stats(ds: Dataset, idx: np.ndarray, w: int):
    """Mean and standard deviation of each compressed channel (cdpr_ft.learning.models.compress)
    over the windows ending at idx (every cycle counted once per window, as the network sees them)."""
    from .models import compress
    s1 = np.zeros(ds.x.shape[1])
    s2 = np.zeros(ds.x.shape[1])
    n = 0
    for c in range(0, len(idx), 4096):
        xb = compress(ds.batch(idx[c:c + 4096], w).astype(np.float64))
        s1 += xb.sum(axis=(0, 2))
        s2 += (xb ** 2).sum(axis=(0, 2))
        n += xb.shape[0] * xb.shape[2]
    mean = s1 / n
    std = np.sqrt(np.maximum(s2 / n - mean ** 2, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)

"""Compare a regenerated result file with the stored reference.

Supported: .json, .json.gz, .yaml (values only, comments ignored), .npz, .pt (PyTorch models:
the stored tensors and settings) and .txt. Timing values
(any key containing "seconds"), the provenance stamp and the record file names are left out,
because they depend on the machine and the date. Every other number is compared with the
tolerances given; integers, booleans and strings must match exactly.

Two result files of the closed-loop checks (results/phase3_check_*.json) can be compared even
when the new one holds a subset of the runs (the reviewer check runs only the 8 breaks): runs are
matched by seed, and summary groups are compared only when they hold the same number of runs.

Usage: python scripts/compare_results.py <reference> <new> [--rtol R] [--atol A] [--ignore PATH ...] [--report file.md]
(PATH as printed in the report, for example $.meta.ftc.plan_latency)
Exit status 0 when every compared value agrees within the tolerances, 1 otherwise.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

IGNORED = ("provenance", "utc", "record")


def ignored(key: str) -> bool:
    return key in IGNORED or "seconds" in key


def load(path: Path):
    name = path.name
    if name.endswith(".json.gz"):
        with gzip.open(path, "rt") as f:
            return json.load(f)
    if name.endswith(".json"):
        return json.loads(path.read_text())
    if name.endswith((".yaml", ".yml")):
        import yaml
        return yaml.safe_load(path.read_text())
    if name.endswith((".pdf", ".png")):
        import hashlib
        return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if name.endswith(".npz"):
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    if name.endswith(".pt"):
        import torch

        def conv(v):
            if isinstance(v, torch.Tensor):
                return v.detach().cpu().numpy()
            if isinstance(v, dict):
                return {k: conv(x) for k, x in v.items()}
            return v
        return conv(torch.load(path, weights_only=False, map_location="cpu"))
    return path.read_text()


class Tally:
    def __init__(self, rtol, atol, ignore=()):
        self.rtol, self.atol = rtol, atol
        self.ignore = set(ignore)      # paths left out on purpose, e.g. "$.meta.ftc.plan_latency"
        self.ignored: set[str] = set()
        self.n = self.exact = 0
        self.max_rel = 0.0
        self.problems: list[str] = []

    def number(self, path, a, b):
        self.n += 1
        if a == b or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)):
            self.exact += 1
            return
        diff = abs(a - b)
        rel = diff / max(abs(a), abs(b), 1e-300)
        self.max_rel = max(self.max_rel, rel)
        if diff > self.atol + self.rtol * abs(a):
            self.problems.append(f"{path}: {a!r} -> {b!r} (relative difference {rel:.2e})")

    def other(self, path, a, b):
        self.n += 1
        if a == b:
            self.exact += 1
        else:
            self.problems.append(f"{path}: {a!r} -> {b!r}")


def walk(t: Tally, a, b, path="$"):
    if path in t.ignore:
        t.ignored.add(path)
        return
    if isinstance(a, dict) and isinstance(b, dict):
        if "runs" in a and "runs" in b and _seeded(a["runs"]) and _seeded(b["runs"]):
            return _walk_runs(t, a, b, path)
        for k in sorted(set(a) | set(b), key=str):
            if ignored(str(k)):
                continue
            if f"{path}.{k}" in t.ignore:
                t.ignored.add(f"{path}.{k}")
                continue
            if k not in a or k not in b:
                t.problems.append(f"{path}.{k}: present only in the {'reference' if k in a else 'new file'}")
                continue
            walk(t, a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            t.problems.append(f"{path}: length {len(a)} -> {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(t, x, y, f"{path}[{i}]")
    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape:
            t.problems.append(f"{path}: shape {a.shape} -> {b.shape}")
            return
        if a.dtype.kind in "fc":
            for i, (x, y) in enumerate(zip(a.ravel().tolist(), b.ravel().tolist())):
                if x != y:
                    t.number(f"{path}[{i}]", x, y)
                else:
                    t.n += 1
                    t.exact += 1
        else:
            t.other(path, a.tolist(), b.tolist())
    elif isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or isinstance(b, str) or a is None or b is None:
        t.other(path, a, b)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, int) and isinstance(b, int):
            t.other(path, a, b)
        else:
            t.number(path, float(a), float(b))
    else:
        t.other(path, a, b)


def _seeded(runs):
    return isinstance(runs, list) and runs and all(isinstance(r, dict) and "seed" in r for r in runs)


def _walk_runs(t: Tally, a, b, path):
    ref = {(r["seed"], json.dumps(r.get("spec", r.get("fault")), sort_keys=True, default=str)): r for r in a["runs"]}
    for r in b["runs"]:
        key = (r["seed"], json.dumps(r.get("spec", r.get("fault")), sort_keys=True, default=str))
        if key not in ref:
            t.problems.append(f"{path}.runs: seed {r['seed']} not in the reference")
            continue
        walk(t, ref[key], r, f"{path}.runs[seed {r['seed']}]")
    subset = len(b["runs"]) < len(a["runs"])
    for k in sorted(set(a) | set(b)):
        if k == "runs" or ignored(k) or k not in a or k not in b:
            continue
        if k == "summary" and subset:
            for g in sorted(set(a[k]) & set(b[k])):
                ga, gb = a[k][g], b[k][g]
                if isinstance(ga, dict) and isinstance(gb, dict) and ga.get("n") == gb.get("n") and ga.get("n"):
                    walk(t, ga, gb, f"{path}.summary.{g}")
            continue
        walk(t, a[k], b[k], f"{path}.{k}")


def compare_text(t: Tally, a: str, b: str):
    la, lb = a.splitlines(), b.splitlines()
    if len(la) != len(lb):
        t.problems.append(f"text: {len(la)} lines -> {len(lb)} lines")
    num = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
    for i, (x, y) in enumerate(zip(la, lb)):
        if x == y:
            t.n += 1
            t.exact += 1
            continue
        nx, ny = num.findall(x), num.findall(y)
        if num.sub("#", x) != num.sub("#", y) or len(nx) != len(ny):
            t.problems.append(f"line {i + 1}: {x!r} -> {y!r}")
            continue
        for u, v in zip(nx, ny):
            t.number(f"line {i + 1}", float(u), float(v))


def compare(reference: Path, new: Path, rtol=1e-9, atol=0.0, ignore=()) -> Tally:
    t = Tally(rtol, atol, ignore)
    a, b = load(reference), load(new)
    if isinstance(a, str) and isinstance(b, str):
        compare_text(t, a, b)
    else:
        walk(t, a, b)
    return t


def verdict(t: Tally) -> str:
    if t.problems:
        return "DIFFERENT"
    return "IDENTICAL" if t.exact == t.n else "AGREES WITHIN TOLERANCE"


def describe(name: str, t: Tally, limit=15) -> str:
    lines = [f"{name}: {verdict(t)} ({t.n} values compared, {t.exact} identical, "
             f"largest relative difference {t.max_rel:.2e})"]
    if t.ignored:
        lines.append(f"    left out on purpose: {', '.join(sorted(t.ignored))}")
    for p in t.problems[:limit]:
        lines.append(f"    {p}")
    if len(t.problems) > limit:
        lines.append(f"    ... and {len(t.problems) - limit} more")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("reference", type=Path)
    ap.add_argument("new", type=Path)
    ap.add_argument("--rtol", type=float, default=1e-9)
    ap.add_argument("--atol", type=float, default=0.0)
    ap.add_argument("--ignore", nargs="*", default=[])
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()
    t = compare(args.reference, args.new, args.rtol, args.atol, args.ignore)
    text = describe(f"{args.new} vs {args.reference}", t)
    print(text)
    if args.report:
        with args.report.open("a") as f:
            f.write(text + "\n")
    sys.exit(1 if t.problems else 0)


if __name__ == "__main__":
    main()

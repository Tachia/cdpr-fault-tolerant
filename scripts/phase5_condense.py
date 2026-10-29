"""Phase 5: condense the campaign into the files kept in the repository.

From results/phase5_work/ (all rows and traces, several GB, not kept) this writes
  results/phase5/rows.jsonl.gz   every (method, run) row of the frozen protocol (the last row of a
                                 pair that was run again after an interruption), with the recovery
                                 time of every fault run added from its trace (protocol, metrics:
                                 recovery_time, amendment 1)
  results/phase5/traces/<method>/run<run>.npz
                                 the traces fixed in advance by the protocol: repetition 0 of every
                                 path, fault type and cable, and fault-free repetitions 0-1, for
                                 every method that ran them
  results/phase5/subset_rows.json
                                 the rows of the jobs that scripts/phase5_check.py simulates again
                                 (the reference of reproduce.py phase5)
  results/phase5/condense.json   counts, the bands of the recovery time, the protocol hash and the
                                 hashes of the written files
Usage: python scripts/phase5_condense.py
"""
from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdpr_ft.eval.campaign import jobs_for, load_protocol, run_list  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from phase5_check import subset  # noqa: E402
from cdpr_ft.provenance import stamp  # noqa: E402

WORK = ROOT / "results" / "phase5_work"
OUT = ROOT / "results" / "phase5"


def sha16(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


# method whose fault-free runs give the band of the recovery time: the method that behaves as this
# one before a fault (protocol, metrics: recovery_time)
BAND_OF = {"P": "P", "A2": "P", "A5": "P", "A1": "A1", "B0": "A1", "B1": "A1", "B2": "B2", "B3": "B3", "B3T": "B3T"}
HOLD = 1.0          # [s] the error must stay in the band this long


def _trace(method, run):
    z = np.load(WORK / "traces" / method / f"run{run:05d}.npz")
    return z["t"].astype(float), z["e_p"].astype(float)


def recovery_bands(rows, t_from):
    """Twice the 95th percentile of the position error from t_from, pooled over the samples of the
    fault-free runs, per method and path."""
    pooled = {}
    for (method, run), d in rows.items():
        if d["fault"] is None:
            t, e = _trace(method, run)
            pooled.setdefault((method, d["trajectory"]), []).append(e[t >= t_from])
    return {k: {"band_mm": float(2e3 * np.percentile(np.concatenate(v), 95)), "runs": len(v)} for k, v in pooled.items()}


def recovery_time(t, e, onset, band, lost):
    """Time from the onset to the first instant from which the error stays below the band for HOLD
    seconds (None if the run is lost or never does so before it ends), and whether the error left
    the band after the onset at all."""
    after = t >= onset - 1e-9
    left = bool(np.any(e[after] >= band))
    if lost:
        return None, left
    out = e >= band
    # time of the next sample outside the band, from each sample on
    nxt = np.full(len(t), np.inf)
    last = np.inf
    for k in range(len(t) - 1, -1, -1):
        if out[k]:
            last = t[k]
        nxt[k] = last
    ok = after & ~out & (nxt >= t + HOLD - 1e-9) & (t + HOLD <= t[-1] + 1e-9)
    if not ok.any():
        return None, left
    return float(t[np.argmax(ok)] - onset), left


def main():
    protocol, phash = load_protocol()
    runs = run_list(protocol)
    by_run = {r["run"]: r for r in runs}
    jobs = set(jobs_for(protocol, runs))
    rows = {}
    for line in (WORK / "rows.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("protocol_sha256_16") == phash and (d["method"], d["run"]) in jobs:
            rows[(d["method"], d["run"])] = d
    bands = recovery_bands(rows, float(protocol["metrics"]["t_from"]))
    for (method, run), d in rows.items():
        if d["fault"] is None:
            continue
        b = bands.get((BAND_OF[method], d["trajectory"]))
        if b is None:
            d["recovery_time_s"], d["left_band"] = None, None
            continue
        t, e = _trace(method, run)
        d["recovery_time_s"], d["left_band"] = recovery_time(t, e, d["fault"][2], 1e-3 * b["band_mm"], d["lost"])
    OUT.mkdir(parents=True, exist_ok=True)
    ordered = [rows[k] for k in sorted(rows, key=lambda k: (k[1], k[0]))]
    with gzip.open(OUT / "rows.jsonl.gz", "wt", compresslevel=9) as f:
        for d in ordered:
            f.write(json.dumps(d) + "\n")
    kept = 0
    for (method, run) in sorted(rows):
        r = by_run[run]
        if (r["group"] == "fault" and r["rep"] == 0) or (r["group"] == "fault_free" and r["rep"] < 2):
            src = WORK / "traces" / method / f"run{run:05d}.npz"
            if src.exists():
                dst = OUT / "traces" / method / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                kept += 1
    check_jobs, _ = subset(protocol)
    raw = {}
    for line in (WORK / "rows.jsonl").read_text().splitlines():      # the rows as the campaign wrote them
        if line.strip():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("protocol_sha256_16") == phash:
                raw[(d["method"], d["run"])] = d
    (OUT / "subset_rows.json").write_text(json.dumps([raw[k] for k in sorted(check_jobs, key=lambda k: (k[1], k[0]))
                                                      if k in raw], indent=1, default=float))
    info = {"protocol_sha256_16": phash, "jobs": len(jobs), "rows": len(ordered),
            "recovery_bands": {f"{m}|{tr}": v for (m, tr), v in sorted(bands.items())}, "band_of": BAND_OF,
            "missing": sorted([list(k) for k in jobs - set(rows)])[:50], "missing_count": len(jobs - set(rows)),
            "traces_kept": kept, "rows_sha256_16": sha16(OUT / "rows.jsonl.gz"), "provenance": stamp()}
    (OUT / "condense.json").write_text(json.dumps(info, indent=1))
    print({k: info[k] for k in ("jobs", "rows", "missing_count", "traces_kept")})


if __name__ == "__main__":
    main()

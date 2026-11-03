"""Motion of the platform in the delayed runs of the latency study (descriptive, after the study).

The latency study (scripts/latency_probe.py, post_review/latency_protocol.yaml) stores one row per run,
without the trajectory. This script simulates the runs of one delay again in the same way, keeps the
attitude of the platform and describes how each run ended: the time of the loss after the start, the
angle that ended it, and the dominant frequency and growth of the attitude oscillation before the loss
(spectrum of roll and pitch over the 0.5 s before the loss and the zero crossings of the angle
with the larger peak, and the ratio of the largest angle in the
last 0.1 s to that of the 0.1 s half a second earlier), and the attitude when the IMM declared a
fault. Each row must equal the stored row of the study.

Output: results/latency/motion_<delay>ms.json
Usage: python scripts/latency_motion.py [delay_ms] [processes]
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import latency_probe as lp  # noqa: E402
from cdpr_ft.eval import campaign  # noqa: E402
from cdpr_ft.eval.campaign import load_protocol  # noqa: E402
from cdpr_ft.provenance import stamp  # noqa: E402

_LOG: dict = {}


def worker_init(protocol_path):
    lp.worker_init(protocol_path)
    evaluate = campaign.evaluate

    def keep(spec, log, cl, ctrl, protocol, trace_path=None):
        _LOG["t"], _LOG["q"] = np.asarray(log["t"]), np.asarray(log["q"])
        return evaluate(spec, log, cl, ctrl, protocol, trace_path)
    campaign.evaluate = keep


def one(job):
    _LOG.clear()
    row = lp.one(job)
    t, ang = _LOG["t"], np.rad2deg(_LOG["q"][:, 3:])
    out = {"run": row["run"], "delay_ms": row["delay_ms"], "lost": row["lost"], "stopped": row["stopped"],
           "t_end": float(t[-1]), "onset": row["fault"][2], "row": row}
    if row["lost"]:
        out["angle"] = ("roll", "pitch", "yaw")[int(np.argmax(np.abs(ang[-1])))]
        dt = float(np.median(np.diff(t)))
        win = t >= t[-1] - 0.5
        freqs = []
        for j in (0, 1):
            x = ang[win, j] - ang[win, j].mean()
            spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
            f = np.fft.rfftfreq(len(x), dt)
            freqs.append((float(spec[1:].max()), float(f[1:][np.argmax(spec[1:])])))
        out["dominant_hz"] = max(freqs)[1]
        # frequency from the zero crossings of the angle with the larger spectral peak, about its mean
        # over the same window (the 0.5 s window resolves the spectrum to 2 Hz only)
        j = int(np.argmax([fr[0] for fr in freqs]))
        x = ang[win, j] - ang[win, j].mean()
        out["zero_crossing_hz"] = float(np.sum(np.diff(np.sign(x)) != 0) / (2 * (t[win][-1] - t[win][0])))
        last = np.abs(ang[t >= t[-1] - 0.1, :2]).max()
        early = (t >= t[-1] - 0.6) & (t < t[-1] - 0.5)
        out["growth_last_0p1s_over_0p5s_earlier"] = float(last / np.abs(ang[early, :2]).max()) if early.any() else None
    # the attitude when the IMM declared (all delayed runs were declared before the break)
    td = row.get("t_declare")
    tilt = np.abs(ang[:, :2]).max(axis=1)
    out["t_declare"] = td
    out["roll_pitch_at_declaration_deg"] = float(tilt[min(int(np.searchsorted(t, td)), len(t) - 1)]) if td is not None else None
    k5 = np.flatnonzero(tilt > 5.0)
    out["t_roll_pitch_above_5_deg"] = float(t[k5[0]]) if len(k5) else None
    return out


def main(delay=20, processes=2):
    protocol, _ = load_protocol(lp.PROTOCOL)
    runs = lp.study_runs(protocol)
    stored = lp.load_done(lp.OUT / "rows.jsonl")
    with mp.get_context("fork").Pool(processes, initializer=worker_init, initargs=(str(lp.PROTOCOL),),
                                     maxtasksperchild=50) as pool:
        res = pool.map(one, [(delay, r) for r in runs], chunksize=1)
    from replicate import compare_row
    for x in res:
        row = x.pop("row")
        ref = {k: v for k, v in stored[(delay, x["run"])].items() if k not in ("delay_ms", "protocol_sha256_16")}
        c = compare_row(ref, {k: v for k, v in row.items() if k != "delay_ms"})
        x["row_check"] = c["class"] if not c["outcome"] else "outcome: " + ", ".join(c["outcome"])
    lost = [x for x in res if x["lost"]]
    summary = {"delay_ms": delay, "runs": len(res), "row_check": {k: sum(x["row_check"] == k for x in res)
                                                                  for k in {x["row_check"] for x in res}},
               "lost": len(lost), "lost_before_onset": sum(x["t_end"] < x["onset"] for x in lost),
               "angle": {k: sum(x["angle"] == k for x in lost) for k in ("roll", "pitch", "yaw")}}
    if lost:
        t_end = np.array([x["t_end"] for x in lost])
        hz = np.array([x["dominant_hz"] for x in lost])
        zc = np.array([x["zero_crossing_hz"] for x in lost])
        gr = np.array([x["growth_last_0p1s_over_0p5s_earlier"] for x in lost
                       if x["growth_last_0p1s_over_0p5s_earlier"] is not None])
        decl = [x for x in lost if x["t_declare"] is not None]
        at_d = np.array([x["roll_pitch_at_declaration_deg"] for x in decl])
        above5 = [x for x in decl if x["t_roll_pitch_above_5_deg"] is not None]
        summary.update(declared=len(decl),
                       roll_pitch_at_declaration_deg={"median": float(np.median(at_d)), "min": float(at_d.min()),
                                                      "max": float(at_d.max())} if len(decl) else None,
                       above_5_deg_before_declaration=sum(x["t_roll_pitch_above_5_deg"] < x["t_declare"] for x in above5))
        summary.update(t_loss_s={"median": float(np.median(t_end)), "min": float(t_end.min()), "max": float(t_end.max())},
                       dominant_hz={"median": float(np.median(hz)), "min": float(hz.min()), "max": float(hz.max())},
                       zero_crossing_hz={"median": float(np.median(zc)), "q25": float(np.percentile(zc, 25)),
                                         "q75": float(np.percentile(zc, 75))},
                       growth={"median": float(np.median(gr)), "min": float(gr.min()), "n": int(len(gr))} if len(gr) else None)
    out = {"summary": summary, "runs": res, "provenance": stamp()}
    (lp.OUT / f"motion_{delay}ms.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:3]))

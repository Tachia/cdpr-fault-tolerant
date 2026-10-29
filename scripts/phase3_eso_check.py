"""Phase 3: accuracy of the winch-level tension estimates on the fault-free calibration runs.

The observers (methods, Eq. ESO) are run on the recorded encoder angles and motor torques of
each fault-free run of a calibration set and compared with the cable tensions of the plant.
The estimate at sample k uses the reading of sample k-1, so it is compared with the tension of
sample k-1 and, to show the lag of the observer, of earlier samples. The first second is left
out (start transient). Output: results/phase3_eso_check_<tag>.json
Usage: python scripts/phase3_eso_check.py <calibration tag> [bandwidths in Hz, comma separated]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdpr_ft.control.winch import WinchESO  # noqa: E402
from cdpr_ft.params import inner_rate, load_cable_winch  # noqa: E402
from cdpr_ft.provenance import stamp  # noqa: E402
from cdpr_ft.sim.episode import EpisodeSpec, controller_start  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from phase3_calibration_runs import load_runs  # noqa: E402

LAGS = range(0, 11)


def main(tag, bandwidths=(25.0, 50.0, 100.0)):
    folder = ROOT / "results" / "phase3_calib" / tag
    runs = [r for r in load_runs(folder)["runs"] if not r["spec"]["fault"]]
    cw = load_cable_winch()
    dt = 1.0 / inner_rate()
    out = {}
    for bw in bandwidths:
        per_noise = {}
        for r in runs:
            spec = EpisodeSpec(**{**r["spec"], "fault": None})
            _, _, _, t_nom, _ = controller_start(spec)
            with np.load(folder / r["record"]) as z:
                rec = {k: z[k] for k in ("theta", "tau", "tension")}
            eso = WinchESO(len(t_nom), 2 * np.pi * bw, cw.inertia, cw.transmission, dt)
            eso.reset(np.zeros(len(t_nom)), t_nom)
            est = np.full_like(rec["tension"], np.nan, dtype=float)
            for k in range(len(rec["theta"])):
                if not np.isnan(rec["theta"][k, 0]):
                    eso.update(rec["theta"][k], rec["tau"][k])
                    est[k] = eso.tension()
            ten = rec["tension"].astype(float)
            k0 = int(1.0 / dt) + 12
            errs = {}
            for lag in LAGS:
                e = est[k0:len(rec["theta"])] - ten[k0 - 1 - lag:len(rec["theta"]) - 1 - lag]
                errs[lag] = float(np.sqrt(np.nanmean(e ** 2)))
            per_noise.setdefault(r["spec"]["noise_factor"], []).append(errs)
        out[f"{bw:g}"] = {f"noise_x{n:g}": {"rms_N_at_lag": {str(l): float(np.mean([e[l] for e in v])) for l in LAGS},
                                             "runs": len(v)} for n, v in per_noise.items()}
        print(bw, json.dumps({k: {l: round(x, 3) for l, x in v["rms_N_at_lag"].items() if int(l) in (0, 2, 4, 6, 8)}
                              for k, v in out[f"{bw:g}"].items()}))
    (ROOT / "results" / f"phase3_eso_check_{tag}.json").write_text(json.dumps({"bandwidths": out, "provenance": stamp()},
                                                                             indent=1))


if __name__ == "__main__":
    main(sys.argv[1], tuple(float(x) for x in sys.argv[2].split(",")) if len(sys.argv) > 2 else (25.0, 50.0, 100.0))

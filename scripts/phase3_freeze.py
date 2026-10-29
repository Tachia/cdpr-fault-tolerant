"""Phase 3: write the frozen controller configuration.

Starts from the defaults of the configuration classes (which hold the values chosen in Phase 3,
each with its source in the class definitions), applies the declaration thresholds chosen by
the ROC on the calibration runs, and writes configs/controller_phase3.yaml with a header that
records where every group of values comes from. Prints the hash of the file and the code digest.
Usage: python scripts/phase3_freeze.py <calibration tag of the ROC> [imm overrides as json]
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdpr_ft.control.config import CONFIGS, dump_controller, load_controller  # noqa: E402
from cdpr_ft.control.ftc import FTCConfig  # noqa: E402
from cdpr_ft.control.mpc import MPCConfig  # noqa: E402
from cdpr_ft.control.planner import PlannerConfig  # noqa: E402
from cdpr_ft.control.recovery import RecoveryConfig  # noqa: E402
from cdpr_ft.estimation.imm import IMMConfig  # noqa: E402
from cdpr_ft.provenance import code_digest  # noqa: E402

HEADER = """# Frozen controller configuration of the study (Phase 3, {date}).
# Code digest at freeze: {digest}
# Sources of the values (details in docs/phase3/controller.pdf):
#   imm      sigma_acc_ang: replay of the calibration runs (results/phase3_replay_c2.json and
#            results/phase3_replay_c2_stage2_settings.json); other noise and bias settings: design choices kept;
#            t_arm: development runs, earliest onset 3 s; nis_gate: safeguard, design choice
#   mpc      horizon, rate: design choices; velocity and tension-change weights: results/phase3_mpc_weights.json
#   planner  kappa_min, ramp, horizon: design choices (methods, Sec. 9)
#   recovery problem of methods Eq. (recovery); margins and weights: design choices
#   ftc      mu_declare, n_confirm: ROC on the calibration runs (results/phase3_roc_{tag}.json);
#            post_break_weights: results/phase3_check_post_break_*.json; other values: design choices
# Calibration seeds: 100000 and up. Test seeds (Phase 5): 200000 and up. Development runs: seeds below 100.
"""


def main(tag, imm_over=None):
    roc = json.loads((ROOT / "results" / f"phase3_roc_{tag}.json").read_text())
    chosen = roc["chosen"]
    ftc = replace(FTCConfig(), mu_declare=chosen["mu_iso"], n_confirm=chosen["n_c"])
    imm = IMMConfig(**(imm_over or {}))
    configs = {"imm": imm, "mpc": MPCConfig(), "planner": PlannerConfig(), "recovery": RecoveryConfig(), "ftc": ftc}
    from datetime import date
    digest = code_digest()
    path = CONFIGS / "controller_phase3.yaml"
    sha = dump_controller(configs, path, HEADER.format(date=date.today().isoformat(), digest=digest, tag=tag))
    loaded, sha2 = load_controller("controller_phase3.yaml")
    assert sha == sha2 and all(loaded[k] == v for k, v in configs.items()), "round trip of the configuration failed"
    print("written", path, "sha256/16", sha, "code digest", digest)


if __name__ == "__main__":
    main(sys.argv[1], json.loads(sys.argv[2]) if len(sys.argv) > 2 else None)

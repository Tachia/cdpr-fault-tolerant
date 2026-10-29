"""Phase 5: freeze of everything the campaign depends on, before its first evaluation run.

Writes results/phase5_freeze.json with the hashes of the protocol, the controller configuration,
the B2 and B3 settings and model, and the digests of the code. scripts/phase5_run.py refuses to
start, and every worker refuses to run, if any of them has changed since.
Usage: python scripts/phase5_freeze.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdpr_ft.eval.campaign import frozen_hashes, load_protocol  # noqa: E402
from cdpr_ft.provenance import stamp  # noqa: E402


def main():
    out = ROOT / "results" / "phase5_freeze.json"
    if out.exists() and "--force" not in sys.argv:
        sys.exit("results/phase5_freeze.json exists")
    protocol, phash = load_protocol()
    h = frozen_hashes(protocol)
    (ROOT / "results").mkdir(exist_ok=True)
    out.write_text(json.dumps({"hashes": h, "provenance": stamp()}, indent=1))
    print(json.dumps(h, indent=1))


if __name__ == "__main__":
    main()

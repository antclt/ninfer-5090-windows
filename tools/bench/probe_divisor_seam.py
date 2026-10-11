#!/usr/bin/env python3
"""Falsify a divisor variant's seam without a build: does the patch reach the loader?

Why this exists, and why it is a tool rather than a paragraph. A divisor sweep patches
`official_recipes._CALIBRATION_PATH` for the duration of the official recipe's call, because
`load_calibration` reads that module attribute at call time. If `python -m tools.convert` imported the
recipes under a second module name, the patch would be invisible, every arm would build identically,
and the resulting flat curve would be an artefact of the instrument rather than a result -- which looks
exactly like a real flat result. This asks the loader directly, and it is the cheap half of the
falsification: the other half reads the stored divisors back out of the built artifact
(`tools/bench/read_artifact_divisors.py`).

Usage:
    python tools/bench/probe_divisor_seam.py --variants DIR [--site SITE]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.convert import official_recipes as official  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--variants", type=Path, required=True,
                    help="directory holding calibration_<arm>.json and recipe_<arm>.py")
parser.add_argument("--site", default="text/layers/0/gdn/input_projection",
                    help="the site whose divisor is compared across arms")
parser.add_argument("--arms", nargs="+", default=["x4", "d2", "const"])
args = parser.parse_args()

print(f"site: {args.site}")
print(f"unpatched: {official.load_calibration()[args.site]:.4f}")
for label in args.arms:
    calibration = args.variants / f"calibration_{label}.json"
    if not calibration.is_file():
        print(f"patched {label:>7}: (no {calibration.name})")
        continue
    original = official._CALIBRATION_PATH
    official._CALIBRATION_PATH = calibration
    try:
        print(f"patched {label:>7}: {official.load_calibration()[args.site]:.4f}")
    finally:
        official._CALIBRATION_PATH = original
print(f"restored:  {official.load_calibration()[args.site]:.4f}")

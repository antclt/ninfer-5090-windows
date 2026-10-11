#!/usr/bin/env python3
"""Write one scaled calibration and its recipe per divisor arm, for a global-activation-scale sweep.

Why the divisor is the lever: the port's calibration stores `input_scale_divisor = 2688 / amax` per site
(tools/convert/calibration.py), and the recipes attach it to every re-encoded projection's *input* use as
`activation_input_divisor`. Scaling it by k moves the per-tensor activation max by 1/k and leaves the
weights alone, which is the axis NVIDIA's own policies move: `nvfp4_act_headroom` multiplies the global
amax up (divisor down), and `input_scale1`/`constant_amax` pins amax = 2688 for every site (divisor 1.0).

This is a controlled stand-in for those policies, not a reimplementation: one number changes, on the same
sources, through the same recipe. The 2026-10-10 sweep on `nvfp4full` used it for x2/x4 (clipping) and
d2/d4/const (headroom), and the two falsifiers are `probe_divisor_seam.py` (the patch reaches the loader)
and `read_artifact_divisors.py` (the stored numbers differ).

Usage:
    python tools/bench/make_divisor_variants.py --out DIR [--source CALIBRATION.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

RECIPE_TEMPLATE = '''"""{doc}"""

from pathlib import Path

from tools.convert import official_recipes as official

VARIANT = Path(__file__).with_name("{calibration}")


def configure(model, recipe, sources):
    original = official._CALIBRATION_PATH
    official._CALIBRATION_PATH = VARIANT
    try:
        official.{base}(model, recipe, sources)
    finally:
        official._CALIBRATION_PATH = original
'''

# label -> (transform, description). `scale` multiplies the divisor; `constant` pins it, which is what
# the documented input_scale1 policy does to the global amax.
ARMS: dict[str, tuple[tuple[str, float], str]] = {
    "x4": (("scale", 4.0), "every calibrated divisor multiplied by 4 (the per-tensor max divided by 4)"),
    "x2": (("scale", 2.0), "every calibrated divisor multiplied by 2"),
    "d2": (("scale", 0.5), "every calibrated divisor halved (the per-tensor max doubled)"),
    "d4": (("scale", 0.25), "every calibrated divisor divided by 4"),
    "const": (("constant", 1.0),
              "every calibrated divisor pinned to 1.0 (amax = 2688, the input_scale1 policy)"),
}

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--out", type=Path, required=True, help="directory for the generated files")
parser.add_argument("--source", type=Path,
                    default=REPO / "tools" / "convert" / "qwen3_8_27b_nvfp4_calibration.json")
parser.add_argument("--base-recipe", default="qwen3_8_27b_nvfp4_unsloth")
parser.add_argument("--arms", nargs="+", default=sorted(ARMS))
args = parser.parse_args()

args.out.mkdir(parents=True, exist_ok=True)
payload = json.loads(args.source.read_text(encoding="utf-8"))
sites = payload["sites"]
for label in args.arms:
    (kind, value), doc = ARMS[label]
    scaled = json.loads(json.dumps(payload))
    for name, site in scaled["sites"].items():
        divisor = sites[name]["input_scale_divisor"]
        site["input_scale_divisor"] = value if kind == "constant" else divisor * value
        # Keep the pair consistent: amax and the divisor are reciprocals through full_range, and an
        # inconsistent record would read as a measurement rather than as a transform of one.
        site["amax"] = scaled["full_range"] / site["input_scale_divisor"]
    calibration = args.out / f"calibration_{label}.json"
    calibration.write_text(json.dumps(scaled, indent=1) + "\n", encoding="utf-8", newline="\n")
    (args.out / f"recipe_{label}.py").write_text(
        RECIPE_TEMPLATE.format(doc=f"Divisor arm {label}: {doc}.", calibration=calibration.name,
                               base=args.base_recipe),
        encoding="utf-8", newline="\n")
    print(f"{label}: {calibration.name} + recipe_{label}.py ({len(scaled['sites'])} sites)")

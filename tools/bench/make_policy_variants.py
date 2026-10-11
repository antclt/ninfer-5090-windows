#!/usr/bin/env python3
"""Write one variant recipe per activation-policy arm, for the precision-assignment search.

Why the policy is the lever: an activation policy is an artifact property -- each use carries one
(`tools/artifact/schema.py`), and the official recipes assign `AllowA4` to every re-encoded NVFP4 site.
Changing it moves one thing: which compute route a projection's *input* takes (A4/A8 quantized or A16),
and therefore both its arithmetic and its cost.

This is a controlled stand-in for a re-recipe, not a reimplementation: one policy changes, on the same
sources, through the same recipe. Each variant intercepts `recipe.assign` and rewrites the policy for its
own sites, so the sources, the conversion method and the divisors stay the recipe's own.

The 2026-10-11 arms on `nvfp4qat`, scored on `ninfer-perplexity --quick` (261,167 tokens) interleaved
with the shipped build as control (which reproduced 4.38763 on three runs):

    shipped        4.38763   6.06-6.24k tok/s
    mlp-down A16   4.37280   3.70-3.71k        (64 sites)
    qk A16         4.38085   5.33k            (32 sites)
    vo A16         4.37154   5.21k            (32 sites)

So the o and v projections are the more sensitive pair, which is what the ov-freeze paper predicts, and
none of the arms is adopted: the gains are 0.15-0.37 % against 14-39 % of scoring throughput. The
falsifier is the artifact's own use table -- `tools/bench/read_artifact_policies.py` prints it, and each
arm must move exactly its own sites and nothing else.

Usage:
    python tools/bench/make_policy_variants.py --out DIR [--base-recipe NAME]
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# label -> (sites, description). `text/layers/` scoping matters for the attention arms: the drafter
# carries its own `attention/query` and `attention/key` sites, and these arms are about the target.
ARMS: dict[str, tuple[tuple[str, ...], str]] = {
    "mlp_a16": (("mlp/down",),
                "every MLP down projection's input at A16 (the derived-intermediate sites)"),
    "qk_a16": (("attention/query", "attention/key"),
               "the attention q and k projections' inputs at A16"),
    "vo_a16": (("attention/value", "attention/output"),
               "the attention v and o projections' inputs at A16, the pair ov-freeze names"),
}

RECIPE_TEMPLATE = '''"""{doc}"""

from tools.convert import official_recipes as official


def configure(model, recipe, sources):
    original = recipe.assign

    def patched(selectors, **kwargs):
        names = selectors if isinstance(selectors, (list, tuple)) else (selectors,)
        target = any(str(name).startswith("text/layers/") for name in names)
        if target and any(site in str(name) for site in {sites!r} for name in names):
            kwargs["activation_policy"] = "A16Only"
        return original(selectors, **kwargs)

    recipe.assign = patched
    try:
        official.{base}(model, recipe, sources)
    finally:
        recipe.assign = original
'''

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--out", type=Path, required=True, help="directory for the generated recipes")
parser.add_argument("--base-recipe", default="qwen3_8_27b_nvfp4_qat")
args = parser.parse_args()

args.out.mkdir(parents=True, exist_ok=True)
for label, (sites, doc) in sorted(ARMS.items()):
    path = args.out / f"recipe_{label}.py"
    path.write_text(RECIPE_TEMPLATE.format(doc=f"Policy arm {label}: {doc}.", sites=sites,
                                           base=args.base_recipe),
                    encoding="utf-8", newline="\n")
    print(f"{label}: {path.name} ({len(sites)} site pattern(s))")
print(f"\nconvert one with:\n"
      f"  python -m tools.convert --model <source> --recipe {args.out}/recipe_<arm>.py "
      f"--components text,vision,mtp,dflash2 --source dflash2=<dir> --source bf16=<dir> "
      f"--out <variant>.v3.ninfer --name <name>\n"
      f"then read the policies back with tools/bench/read_artifact_policies.py, which is the falsifier: "
      f"each arm must move exactly its own sites and nothing else.")

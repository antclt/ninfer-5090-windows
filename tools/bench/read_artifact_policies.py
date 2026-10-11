#!/usr/bin/env python3
"""Read an artifact's activation policies per site kind, by parameter name.

The policy lives on the use (`tools/artifact/schema.py` validates it), so this is the falsifier a policy
arm needs: an arm is only the change it claims if the artifact's own table shows its sites moved and
every other site identical. The 2026-10-11 arms read:

    shipped    A16Only 332, AllowA4 512   (mlp/down 64 at AllowA4)
    mlp_a16    A16Only 396, AllowA4 448   (mlp/down 64 at A16Only -- exactly +64)
    qk_a16     A16Only 364, AllowA4 480   (attention/query and key 16+16 at A16Only -- exactly +32)
    vo_a16     A16Only 364, AllowA4 480   (attention/value and output 16+16 at A16Only -- exactly +32)

Usage:
    python tools/bench/read_artifact_policies.py ARTIFACT [ARTIFACT ...]
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.artifact.reader import Artifact  # noqa: E402

KINDS = ("attention/query", "attention/key", "attention/value", "attention/output", "mlp/gate",
         "mlp/up", "mlp/down")

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("artifacts", type=pathlib.Path, nargs="+")
args = parser.parse_args()

for path in args.artifacts:
    artifact = Artifact(path)
    uses = artifact.directory.uses
    totals = collections.Counter(str(use.get("activation_policy")) for use in uses)
    by_kind: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for use in uses:
        name = str(use.get("parameter", ""))
        kind = "other"
        for probe in KINDS:
            if name.endswith(probe) and name.startswith("text/layers/"):
                kind = probe
        by_kind[kind][str(use.get("activation_policy"))] += 1
    print(f"{path.name}: uses={len(uses)} policies={dict(totals)}")
    for kind in (*KINDS, "other"):
        if by_kind[kind]:
            print(f"    {kind:18s} {dict(by_kind[kind])}")
    close = getattr(artifact, "close", None)
    if callable(close):
        close()

#!/usr/bin/env python3
"""Read an artifact's stored activation input divisors, by parameter name.

The divisor lives as a 4-byte FP32 auxiliary object on the use, which is how `verify_artifact.py`'s
`check_divisors` reads it too. Printing the values is the falsification half that a build cannot give
you: a divisor sweep's variants are only different if the *stored* numbers differ, and the payload size
cannot tell you -- the auxiliary is fixed-width, so five arms have identical `payload_bytes` whether or
not the transform reached them. This is also what establishes a transform's scope: in the 2026-10-10
sweep the calibrated hidden-state-input sites moved while imported ones (`gdn/output`, `mlp/gate`) held
their own divisors in every arm.

Usage:
    python tools/bench/read_artifact_divisors.py ARTIFACT [ARTIFACT ...] [--sites SUBSTRING ...]
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.artifact.reader import Artifact  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("artifacts", type=Path, nargs="+")
parser.add_argument("--sites", nargs="+", default=["layers/0/", "layers/3/"],
                    help="substrings of the parameter names to print")
parser.add_argument("--limit", type=int, default=6, help="rows per artifact")
args = parser.parse_args()


def divisors(path: Path) -> list[tuple[str, str, float]]:
    artifact = Artifact(path)
    directory = artifact.directory
    by_id = {obj.id: obj for obj in directory.objects}
    rows: list[tuple[str, str, float]] = []
    for entry in directory.uses:
        parameter = str(entry.get("parameter", "?"))
        for aux_name, aux in sorted((entry.get("auxiliaries") or {}).items()):
            obj = by_id.get(aux.get("object", ""))
            if obj is None or getattr(obj, "format", "") != "fp32" or obj.bytes != 4:
                continue
            (word,) = struct.unpack("<I", artifact.read_range(obj.offset, 4))
            (value,) = struct.unpack("<f", struct.pack("<I", word))
            rows.append((parameter, aux_name, value))
    close = getattr(artifact, "close", None)
    if callable(close):
        close()
    return rows


for path in args.artifacts:
    rows = divisors(path)
    interesting = [row for row in rows if any(site in row[0] for site in args.sites)]
    print(f"{path.name}: {len(rows)} input divisors")
    for parameter, aux_name, value in (interesting or rows)[:args.limit]:
        print(f"    {parameter:52} {aux_name} = {value:.4f}")

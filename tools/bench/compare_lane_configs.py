"""Compare two configurations of one lane inside the protocol the shipped table uses.

This replaces an earlier version that hand-rolled its own protocol: it sent one prompt per cell at
temperature 0 with no warmup, and measured 384-437 tok/s where the table reads 361 on the same lane and
cell. The difference was two things `measure_decode` documents and this repository had already paid for
once:

* **the first full-length decode after a server start is a transient**, faster than every later
  identical request (261.7 tok/s against 169.8 on the NVIDIA MTP5 lane), so it is discarded;
* **the served sampling is the model card's thinking set** (temperature 1.0 / top_p 0.95 / top_k 20),
  not greedy - "the set its coding benchmarks are measured under, and the set the checkpoint's own
  generation_config.json ships". Greedy inflates draft acceptance, and profiles.py's 55.0% against
  ninfer_bench's 21.1% was that discrepancy.

So this calls the matrix's own `run_profile` with the lane's profile dict modified, which is the only
path that reaches the flags: when `profile=` is given, `profile_args` builds the arguments from the dict
and the `draft=`/`kv_dtype=` arguments are ignored. Candidates are interleaved - every candidate in
every round - because the card's session-to-session spread is 10.3% while its paired spread is about 1%.

    python tools/bench/compare_lane_configs.py --lane start_quasar_v3_mtp4_vision.bat \
        --axis depth --values 4,5 --rounds 2
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "bench"))
sys.path.insert(0, str(REPO / "tools" / "release"))

import lane_fingerprint as lf  # noqa: E402  (its stage_cells registers the file-backed cells)
import v3_profile_matrix as matrix  # noqa: E402
from profiles import PROFILES  # noqa: E402

CELLS = ("code", "long-code", "long-chinese")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lane", required=True, help="the launcher profile file")
    parser.add_argument("--axis", choices=("kv", "depth"), required=True)
    parser.add_argument("--values", required=True, help="comma-separated, e.g. 4,5")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--cells", default=",".join(CELLS))
    args = parser.parse_args()

    profile = next((p for p in PROFILES if p.get("file") == args.lane), None)
    if profile is None:
        print(f"no profile named {args.lane}")
        return 1
    values = args.values.split(",")
    cells = args.cells.split(",")
    lf.stage_cells([c for c in cells if c in lf.CELL_SOURCES])  # long-code and long-chinese are files

    print(f"lane {args.lane}  axis {args.axis}  values {values}  rounds {args.rounds}")
    print("  protocol: the matrix's own - a discarded full-length warmup, three runs, sampling "
          "'default'\n")

    runs: dict[str, list[dict[str, float]]] = {v: [] for v in values}
    for round_index in range(1, args.rounds + 1):
        for value in values:
            variant = dict(profile)
            variant[args.axis if args.axis == "depth" else "kv_dtype"] = (
                int(value) if args.axis == "depth" else value)
            got: dict[str, float] = {}
            for cell in cells:
                # stage_cells registers a file-backed cell under the name `fingerprint-<cell>`, which
                # is the domain run_profile accepts; `code` is a DOMAINS name already
                domain = f"fingerprint-{cell}" if cell in lf.CELL_SOURCES else cell
                record = matrix.run_profile(profile=variant, domain=domain, sampling="default")
                rate = record.get("decode_avg")
                if isinstance(rate, (int, float)) and rate:
                    got[cell] = float(rate)
            if got:
                runs[value].append(got)
                print(f"  round {round_index}  {args.axis}={value:6s} " +
                      "  ".join(f"{c} {got.get(c, 0):7.1f}" for c in cells))
            else:
                print(f"  round {round_index}  {args.axis}={value:6s}  no reading")

    print(f"\n  {'value':8s} {'code':>9} {'long-code':>10} {'long-chinese':>13} {'worst deciding':>15}")
    worst: dict[str, float] = {}
    for value in values:
        if not runs[value]:
            print(f"  {value:8s}  no rounds completed")
            continue
        med = {c: statistics.median([r[c] for r in runs[value] if c in r]) for c in cells}
        deciding = [med[c] for c in ("code", "long-code") if c in med]
        worst[value] = min(deciding) if deciding else 0.0
        print(f"  {value:8s} {med.get('code', 0):>9.1f} {med.get('long-code', 0):>10.1f} "
              f"{med.get('long-chinese', 0):>13.1f} {worst[value]:>15.1f}")

    if len(worst) == 2:
        (a, a_worst), (b, b_worst) = worst.items()
        margin = abs(a_worst - b_worst) / max(a_worst, b_worst) * 100.0
        winner = a if a_worst > b_worst else b
        print(f"\n  worst deciding cell: {a} {a_worst:.1f} against {b} {b_worst:.1f}"
              f"  ->  {winner} by {margin:.1f}%")
        print(f"  {'DECISION' if margin > 10.3 else 'NOT A DECISION'} at this card's 10.3% "
              f"session spread, inside the protocol the shipped table uses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

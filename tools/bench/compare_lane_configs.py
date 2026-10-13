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

        # The threshold is the spread of *this* comparison, not the card's session spread. The rule says
        # so - "a margin smaller than the measurement's own spread is not a decision", contrasted in the
        # same sentence with "spreads that reach 21% when the protocol compares lane restarts" - and this
        # tool was hardcoding 10.3%, the session figure, which is the wrong one for an interleaved run:
        # interleaving is what removes the session drift, and what remains is the spread within each
        # candidate's own rounds.
        spreads = {}
        for value in values:
            per_cell = []
            for cell in ("code", "long-code"):
                rates = [r[cell] for r in runs[value] if cell in r]
                if len(rates) > 1 and statistics.mean(rates):
                    per_cell.append((max(rates) - min(rates)) / statistics.mean(rates) * 100.0)
            spreads[value] = max(per_cell) if per_cell else 0.0
        paired = max(spreads.values()) if spreads else 0.0
        print("  paired spread (this comparison's own, from the rounds): "
              + ", ".join(f"{v} {s:.1f}%" for v, s in spreads.items())
              + f"  ->  threshold {paired:.1f}%")

        if paired == 0.0:
            # A threshold of zero is a control that cannot fail: with one round per candidate the
            # spread is not measured at all, and any margin would read as a decision. The first version
            # of this re-read did exactly that and called a 0.2% gap a decision. Say unmeasured.
            print(f"  SPREAD UNMEASURED (one round per candidate): the margin is {margin:.1f}%, and "
                  f"whether it clears the noise is unknown. OPEN - run more rounds, then this tool can "
                  f"decide.")
        elif margin > paired:
            print(f"  DECISION: {winner} leads by {margin:.1f}% against this comparison's own "
                  f"{paired:.1f}% spread")
        elif len(runs[a]) > 1 and len(runs[b]) > 1:
            # the rule's tie-breaker, verbatim: "when the worst-case gap is itself inside the spread, the
            # mean decides". Reported as what it is - a mean decision inside the spread - not as a tie.
            def mean_of(value: str) -> float:
                per_round = []
                for record in runs[value]:
                    rates = [record[c] for c in ("code", "long-code") if c in record]
                    if rates:
                        per_round.append(statistics.mean(rates))
                return statistics.mean(per_round) if per_round else 0.0

            a_mean, b_mean = mean_of(a), mean_of(b)
            mean_winner = a if a_mean > b_mean else b
            print(f"  INSIDE THE SPREAD ({margin:.1f}% < {paired:.1f}%), so the mean decides: "
                  f"{a} {a_mean:.1f} against {b} {b_mean:.1f} -> {mean_winner}")
            print("  Unresolved as a worst-case decision; more rounds narrows the spread, and the "
                  "question stays open until one of the two separates.")
        else:
            print(f"  INSIDE THE SPREAD ({margin:.1f}% < {paired:.1f}%) and too few rounds for the "
                  f"mean to decide - open, and more rounds is the method")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

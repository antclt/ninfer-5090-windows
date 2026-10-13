"""Compare two configurations of one lane, interleaved, on the cells the rule decides by.

The card's session-to-session spread is 10.3% and its within-run spread is 0.3-5.1%, so measuring A and
then B measures the window - this repository's own rule is to interleave them (ADR-0003). The earlier
`compare_kv_formats.py` compared a fresh run against a *recorded* run from another session, and its
quasar reading disagreed with a hand-run by 14%; this replaces it.

Method: for each round, run every candidate in turn - start the lane with that override, measure the two
deciding cells, stop it - and pair the results across rounds, so the drift cancels rather than being
compared. The deciding cells are the 225-character `code` prompt (`DOMAINS["code"]` in
v3_profile_matrix.py, which lane_fingerprint imports) and 36,000 characters of real code built from the
fingerprint's own three sources, with the worst of the two deciding; the non-code cells are reported
beside them, not weighed.

    python tools/bench/compare_lane_configs.py --lane start_quasar_v3_dflash2_vision.bat \
        --axis kv --values k8v4,fp8 --rounds 2
    python tools/bench/compare_lane_configs.py --lane start_quasar_v3_dflash2_vision.bat \
        --axis depth --values 7,9 --rounds 2
"""
from __future__ import annotations

import argparse
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "bench"))
sys.path.insert(0, str(REPO / "tools" / "release"))

import realtext_acceptance as rta  # noqa: E402
from profiles import PROFILES  # noqa: E402
from v3_profile_matrix import DOMAINS  # noqa: E402

CODE_SOURCES = ["tools/release/v3_profile_matrix.py",
                "src/ops/linear/kernels/nvfp4/nvfp4_a4_tma.cuh",
                "tools/convert/official_recipes.py"]
CHINESE = "eval/corpora/perplexity-1m/data/zhwiki/01.txt"
TMP = Path(str(Path.home()) + r"\AppData\Local\Temp\opencode")


def prompts() -> dict[str, str]:
    """The deciding pair plus the worst non-code cell, as the rule defines them."""
    long_code = "".join((REPO / s).read_text(encoding="utf-8", errors="replace")[:12_000]
                        for s in CODE_SOURCES)
    return {"code": DOMAINS["code"][:225],
            "long-code": long_code,
            "long-chinese": (REPO / CHINESE).read_text(encoding="utf-8",
                                                       errors="replace")[:36_000]}


def measure(profile: dict, artifact: Path, cells: dict[str, str], port: int,
            axis: str, value: str, max_tokens: int) -> dict[str, float]:
    """One lane start, one request per cell, decode tok/s by cell."""
    flags = rta.launcher_args(profile, port=port)
    if axis == "kv":
        rta.set_flag(flags, "--kv-dtype", value)
    else:
        rta.set_flag(flags, "--draft-tokens", value)
    log = TMP / f"cmp-{profile['file']}-{axis}{value}.log"
    # the request log as well, so the same run answers the percentiles question: a sweep that produces
    # the evidence once is better than two sweeps that each produce half of it
    request_log = TMP / f"cmp-{profile['file']}-{axis}{value}.jsonl"
    rta.set_flag(flags, "--request-log-jsonl", str(request_log))
    out: dict[str, float] = {}
    with log.open("w", encoding="utf-8", newline="\n") as handle:
        process = subprocess.Popen([str(rta.SERVE), str(artifact), *flags], cwd=str(REPO),
                                   stdout=handle, stderr=subprocess.STDOUT,
                                   env=rta.launcher_environment(profile))
        try:
            if not rta.wait_for_port(port, time.time() + 300):
                print(f"    {value}: serve never listened ({log})")
                return out
            time.sleep(2.0)
            model_id = (rta.get_json(port, "/v1/models").get("data") or [{}])[0].get("id")
            for cell, text in cells.items():
                rta.post_json(port, {"model": model_id,
                                     "messages": [{"role": "user", "content": text}],
                                     "max_tokens": max_tokens, "temperature": 0.0}, 900)
                # The engine's own decode rate, not completion_tokens / wall clock: the wall clock
                # includes prefill, which for a 36,000-character cell dominates, and the first version
                # of this tool therefore read 79 tok/s where the record reads 211 on the same lane and
                # cell. The serve log writes "decode <rate> tok/s" per request, which is the same basis
                # the record's decode_avg comes from.
                rates = re.findall(r"decode\s+([\d.]+) tok/s", log.read_text(encoding="utf-8",
                                                                           errors="replace"))
                if rates:
                    out[cell] = float(rates[-1])
                else:
                    print(f"    {value}/{cell}: no decode rate in the serve log")
        finally:
            time.sleep(2.0)
            process.terminate()
            try:
                process.wait(timeout=120)
            except Exception:
                pass
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lane", required=True, help="the launcher profile file")
    parser.add_argument("--axis", choices=("kv", "depth"), required=True)
    parser.add_argument("--values", required=True, help="comma-separated, e.g. k8v4,fp8")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--port", type=int, default=8094)
    args = parser.parse_args()

    profile = next((p for p in PROFILES if p.get("file") == args.lane), None)
    if profile is None:
        print(f"no profile named {args.lane}")
        return 1
    raw = profile["art"]
    artifact = Path(raw) if Path(raw).is_absolute() else Path(r"C:\AI\models") / raw
    values = args.values.split(",")
    cells = prompts()
    print(f"lane {args.lane}  axis {args.axis}  values {values}  rounds {args.rounds}")
    print("  cells: " + ", ".join(f"{k} ({len(v)} chars)" for k, v in cells.items()) + "\n")

    runs: dict[str, list[dict[str, float]]] = {v: [] for v in values}
    for round_index in range(1, args.rounds + 1):
        for value in values:  # interleaved: every candidate in every round, in the same order
            got = measure(profile, artifact, cells, args.port, args.axis, value, args.max_tokens)
            if got:
                runs[value].append(got)
                print(f"  round {round_index}  {args.axis}={value:6s} " +
                      "  ".join(f"{k} {got.get(k, 0):7.1f}" for k in cells))

    print(f"\n  {'value':8s} {'code':>9} {'long-code':>10} {'long-chinese':>13} {'worst deciding':>15}")
    medians: dict[str, float] = {}
    for value in values:
        if not runs[value]:
            print(f"  {value:8s}  no rounds completed")
            continue
        med = {k: statistics.median([r[k] for r in runs[value] if k in r]) for k in cells}
        worst = min(med["code"], med["long-code"])
        medians[value] = worst
        print(f"  {value:8s} {med['code']:>9.1f} {med['long-code']:>10.1f} "
              f"{med['long-chinese']:>13.1f} {worst:>15.1f}")

    if len(medians) == 2:
        (a, a_worst), (b, b_worst) = medians.items()
        margin = abs(a_worst - b_worst) / max(a_worst, b_worst) * 100.0
        winner = a if a_worst > b_worst else b
        print(f"\n  worst deciding cell: {a} {a_worst:.1f} against {b} {b_worst:.1f}"
              f"  ->  {winner} by {margin:.1f}%")
        print(f"  {'DECISION' if margin > 10.3 else 'NOT A DECISION'} at this card's 10.3% spread"
              + (f" - interleaved over {args.rounds} round(s)" if margin > 10.3 else
                 "; the paired rounds above are the evidence, and more rounds is the way to settle it"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

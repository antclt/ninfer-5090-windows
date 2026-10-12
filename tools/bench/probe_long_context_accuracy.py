"""Long-context accuracy: a needle-in-a-haystack sweep against a served lane.

This is the gap the measurement priority puts second and the one a coding workload cares about most: a
local model that silently fails past some context length will fail on a long repository, and perplexity
does not see it.

Two lessons from the field are built in rather than left to the caller:

* **A fresh random needle per cell.** The needle is a random token generated per (length, depth) cell, so
  no two cells share a prompt prefix that could be served from the replay path. A repeated needle prompt
  would score this port's cache rather than the model, and this repository has already been caught once
  measuring a replay and reading it as decode.
* **The number to report is the length at which the hard tasks still pass**, not the length the model
  accepts (RULER's framing: scores run 10-25 points below single-needle NIAH at the same length, so a
  single-needle pass is a floor, not a capability).

The haystack is the repository's own perplexity corpus, the needle is inserted at a depth fraction, and
scoring is an exact substring match on the needle value -- no judge model, nothing to calibrate.

    python tools/bench/probe_long_context_accuracy.py --lane start_ninfer_v3_dflash2_vision.bat \
        --chars 8000,36000
"""
from __future__ import annotations

import argparse
import random
import re
import string
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "bench"))
sys.path.insert(0, str(REPO / "tools" / "release"))

import realtext_acceptance as rta  # noqa: E402
from profiles import PROFILES  # noqa: E402

CORPUS = REPO / "eval" / "corpora" / "perplexity-1m" / "data" / "ninfer" / "00.txt"
NEEDLE_RE = re.compile(r"^[0-9a-f]{32}$", re.M)


def needle(rng: random.Random) -> str:
    """A random 32-hex value: the `uuid` needle shape, and one no other cell shares."""
    return "".join(rng.choice(string.hexdigits.lower()[:16]) for _ in range(32))


def build_prompt(haystack: str, chars: int, depth: float, value: str) -> str:
    """The haystack truncated to `chars`, the needle inserted at `depth`, and the query at the end.

    The query is not optional: RULER places it at the end of the sequence, and without one the first
    version of this tool inserted a bare needle sentence and the model treated the whole prompt as inert
    text on half the cells - "User provided a bunch of code snippets, no explicit question" - which read
    as a miss and was a prompt defect, not a capability one.
    """
    text = haystack[:chars]
    cut = int(len(text) * depth)
    # land on a line boundary so the needle reads as a document line rather than a splice
    cut = text.rfind("\n", 0, cut) if cut > 0 else 0
    cut = 0 if cut < 0 else cut
    sentence = f"\nThe access code for build {value[:8]} is: {value}\n"
    query = (f"\n\nWhat is the access code for build {value[:8]}? "
             f"Answer with the code alone and nothing else.\n")
    return text[:cut] + sentence + text[cut:] + query


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lane", default="dflash2", help="substring of the launcher profile to run")
    parser.add_argument("--chars", default="8000,36000", help="comma-separated haystack sizes")
    parser.add_argument("--depths", default="0.0,0.5,1.0", help="comma-separated depth fractions")
    parser.add_argument("--artifact", type=Path, default=None, help="override the lane's artifact")
    parser.add_argument("--port", type=int, default=8098)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="output budget; a reasoning lane spends most of it before answering, which "
                             "is why the default is not small - at 48 this sweep read MISS on every cell "
                             "while the answers were truncated mid-needle")
    args = parser.parse_args()

    profile = next((p for p in PROFILES if args.lane in p.get("file", "")), None)
    if profile is None:
        print(f"no lane matching {args.lane!r}")
        return 1
    raw = profile["art"]
    artifact = args.artifact or (Path(raw) if Path(raw).is_absolute()
                                 else Path(r"C:\AI\models") / raw)
    haystack = CORPUS.read_text(encoding="utf-8", errors="replace")
    rng = random.Random(args.seed)
    lengths = [int(c) for c in args.chars.split(",")]
    depths = [float(d) for d in args.depths.split(",")]

    print(f"lane {profile['file']}  artifact {artifact.name}  haystack {CORPUS.name}")
    print(f"  {len(lengths)} length(s) x {len(depths)} depth(s) = {len(lengths) * len(depths)} cell(s),"
          f" a fresh needle each\n")

    flags = rta.launcher_args(profile, port=args.port)
    log_path = Path(str(Path.home()) + r"\AppData\Local\Temp\opencode\niah-serve.log")
    grid: dict[tuple[int, float], tuple[int, int]] = {}
    with log_path.open("w", encoding="utf-8", newline="\n") as handle:
        process = subprocess.Popen(
            [str(rta.SERVE), str(artifact), *flags], cwd=str(REPO), stdout=handle,
            stderr=subprocess.STDOUT, env=rta.launcher_environment(profile))
        try:
            if not rta.wait_for_port(args.port, time.time() + 300):
                print("  serve never listened")
                return 1
            time.sleep(2.0)
            model_id = (rta.get_json(args.port, "/v1/models").get("data") or [{}])[0].get("id")
            for chars in lengths:
                for depth in depths:
                    value = needle(rng)
                    prompt = build_prompt(haystack, chars, depth, value)
                    answer = rta.post_json(args.port, {
                        "model": model_id,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": args.max_tokens, "temperature": 0.0}, 900)
                    text = ""
                    if answer["status"] == 200:
                        choices = answer["payload"].get("choices") or [{}]
                        message = choices[0].get("message") or {}
                        # the answer may arrive in `content` or, on a reasoning lane, in
                        # `reasoning_content`; search both, because a truncated or reformatted answer
                        # must not read as a miss
                        text = (message.get("content") or "") + "\n" + \
                               (message.get("reasoning_content") or "")
                    hit = 1 if value in text else 0
                    tokens = (answer["payload"].get("usage") or {}).get("prompt_tokens") if \
                        answer["status"] == 200 else None
                    completion = (answer["payload"].get("usage") or {}).get("completion_tokens") \
                        if answer["status"] == 200 else None
                    grid[(chars, depth)] = (hit, tokens or 0)
                    print(f"    {chars:>7} chars  depth {depth:>4.2f}  prompt_tokens {tokens}  "
                          f"out {completion}  {'HIT ' if hit else 'MISS'}")
                    if not hit:
                        flat = " ".join(text.split())
                        print(f"        answer: {flat[:110]!r}")
        finally:
            time.sleep(2.0)
            process.terminate()
            try:
                process.wait(timeout=120)
            except Exception:
                pass

    print("\n  accuracy by cell (HIT/MISS), depths across:")
    print(f"    {'chars':>8} " + " ".join(f"{d:>5.2f}" for d in depths))
    for chars in lengths:
        row = " ".join("HIT " if grid.get((chars, d), (0, 0))[0] else "MISS" for d in depths)
        print(f"    {chars:>8} {row}")

    passing = [c for c in lengths if all(grid.get((c, d), (0, 0))[0] for d in depths)]
    if passing:
        print(f"\n  all depths pass at: {max(passing)} chars")
    else:
        print("\n  no length passes at every depth")
    print("  single-needle only: RULER's multi-key, multi-value, multi-query and multi-hop categories run"
          "\n  10-25 points below this at the same length, so treat a pass here as a floor, not a capability.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

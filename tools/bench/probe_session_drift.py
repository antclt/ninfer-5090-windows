"""Agent-session state drift: does a constraint set early still hold after many turns?

The fifth measurement gap. An agent session is a long conversation in which something established at the
start - a build code, a naming convention, a file it must not touch - has to survive dozens of turns of
other work. Nothing in this tree measured whether it does.

Two things this deliberately does not do, both learned the hard way in this session:

* **No repeated prompt.** Every turn carries a distinct filler slice from the corpus and a rephrased
  probe. An identical repeat is served from the replay path, which would score the cache rather than the
  model.
* **No hand-rolled sampling.** The request carries `DOCUMENTED_SAMPLING` from the matrix - temperature
  1.0, top_p 0.95, top_k 20, the set this model's coding benchmarks are measured under - because a
  configuration comparison at a sampler nobody serves by ranks a regime that does not exist.

The constraint is a random 12-hex code, so the probe cannot be answered from world knowledge, and the
score is an exact substring match. The report is the retention over the turns and **the turn at which it
first fails**, which is the drift point.

    python tools/bench/probe_session_drift.py --lane start_ninfer_v3_dflash2_vision.bat --turns 40
"""
from __future__ import annotations

import argparse
import random
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "bench"))
sys.path.insert(0, str(REPO / "tools" / "release"))

import realtext_acceptance as rta  # noqa: E402
from profiles import PROFILES  # noqa: E402
from v3_profile_matrix import DOCUMENTED_SAMPLING  # noqa: E402

CORPUS_DIR = REPO / "eval" / "corpora" / "perplexity-1m" / "data" / "ninfer"

# Rephrasings of the same question. A single wording repeated every turn would be a repeated prompt.
PROBES = (
    "Restate the build code I gave you at the start, and nothing else.",
    "What was the build code from my first message? Answer with the code only.",
    "Give me the build code again - just the code.",
    "The build code, please: the one from the beginning of this conversation.",
    "Remind me of the build code. Code only, no explanation.",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lane", default="dflash2", help="substring of the launcher profile to run")
    parser.add_argument("--turns", type=int, default=40)
    parser.add_argument("--filler-chars", type=int, default=1500, help="distinct corpus text per turn")
    # Reasoning is billed against the same output cap as the answer, so a cap of 64 left zero room
    # for one: measured directly, content came back empty with finish_reason=length at every cap
    # below the model's appetite. This default has to clear the thinking budget with room for a short
    # answer, or the probe measures truncation and calls it drift.
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="output cap; must exceed the thinking budget, which is billed against it")
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--port", type=int, default=8093)
    args = parser.parse_args()

    profile = next((p for p in PROFILES if args.lane in p.get("file", "")), None)
    if profile is None:
        print(f"no lane matching {args.lane!r}")
        return 1
    raw = profile["art"]
    artifact = Path(raw) if Path(raw).is_absolute() else Path(r"C:\AI\models") / raw
    haystack = "".join(p.read_text(encoding="utf-8", errors="replace")
                       for p in sorted(CORPUS_DIR.glob("*.txt")))
    rng = random.Random(args.seed)
    code = "".join(rng.choice("0123456789abcdef") for _ in range(12))
    print(f"lane {profile['file']}  artifact {artifact.name}")
    print(f"  {args.turns} turns, constraint = build code {code}, sampling = DOCUMENTED_SAMPLING")
    print(f"  haystack {len(haystack)} chars, {args.filler_chars} per turn, distinct every turn\n")

    messages: list[dict[str, str]] = [
        {"role": "user", "content":
            f"Before we start: the build code for this project is {code}. Keep it for the whole "
            f"session; I will ask you for it later. Acknowledge with OK."},
    ]
    flags = rta.launcher_args(profile, port=args.port)
    log_path = Path(str(Path.home()) + r"\AppData\Local\Temp\opencode\drift-serve.log")
    held: list[bool] = []
    with log_path.open("w", encoding="utf-8", newline="\n") as handle:
        process = subprocess.Popen([str(rta.SERVE), str(artifact), *flags], cwd=str(REPO),
                                   stdout=handle, stderr=subprocess.STDOUT,
                                   env=rta.launcher_environment(profile))
        try:
            if not rta.wait_for_port(args.port, time.time() + 300):
                print("  serve never listened")
                return 1
            time.sleep(2.0)
            model_id = (rta.get_json(args.port, "/v1/models").get("data") or [{}])[0].get("id")
            for turn in range(1, args.turns + 1):
                start = (turn * args.filler_chars) % max(1, len(haystack) - args.filler_chars)
                filler = haystack[start:start + args.filler_chars]
                probe = PROBES[turn % len(PROBES)]
                messages.append({"role": "user", "content":
                                 f"Next task: summarise the following in one short sentence.\n\n"
                                 f"{filler}\n\n{probe}"})
                body = {"model": model_id, "messages": messages, "max_tokens": args.max_tokens,
                        **DOCUMENTED_SAMPLING}
                answer = rta.post_json(args.port, body, 900)
                if answer["status"] != 200:
                    print(f"  turn {turn}: HTTP {answer['status']}")
                    break
                message = ((answer["payload"].get("choices") or [{}])[0].get("message") or {})
                content = message.get("content") or ""
                reasoning = message.get("reasoning_content") or ""
                finish = ((answer["payload"].get("choices") or [{}])[0].get("finish_reason"))
                # Scored on the answer alone. Counting reasoning made the verdict depend on whether
                # the code happened to appear in a thinking block, which is noise: a turn truncated
                # before it could answer reads "held" or "lost" by luck. A constraint survives if the
                # model can produce it when asked, and the answer is what produces it.
                ok = code in content
                if finish == "length":
                    # A truncated turn cannot demonstrate anything. Record it as unknown rather than
                    # folding truncation into the retention rate, which is what made this probe
                    # report drift at turn 1 -- the model had not been given room to answer.
                    print(f"  turn {turn:3d}  TRUNC  max_tokens={args.max_tokens} reached"
                          f" with {len(content)} chars of answer")
                    continue
                held.append(ok)
                # Reasoning goes back in its own field. Concatenating it into content is not
                # supported by the chat template, which expects a complete reasoning block preserved
                # separately; and truncating the stored turn to 2000 chars throws away the history
                # the probe is trying to measure the survival of.
                assistant: dict[str, str] = {"role": "assistant", "content": content}
                if reasoning:
                    assistant["reasoning_content"] = reasoning
                messages.append(assistant)
                if not ok or turn % 5 == 0:
                    flat = " ".join(content.split())[:80]
                    print(f"  turn {turn:3d}  {'held ' if ok else 'LOST '}  {flat!r}")

        finally:
            time.sleep(2.0)
            process.terminate()
            try:
                process.wait(timeout=120)
            except Exception:
                pass

    if held:
        first_loss = next((i + 1 for i, ok in enumerate(held) if not ok), None)
        print(f"\n  retention: {sum(held)}/{len(held)} turns = {sum(held) / len(held) * 100:.0f}%")
        print("  drift point: " + (f"turn {first_loss}" if first_loss else
                                   f"none - the constraint held for all {len(held)} turns"))
        print("  Read the drift point against the conversation's length, not the turn count: the prompt "
              "grows every turn, so turn 40 is a longer context than turn 4.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

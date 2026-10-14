"""Context growth per turn under reasoning replay, on a shipped lane's own configuration.

docs/opencode-settings.md measured the opencode settings but recorded one gap it could not close
from inside this tree: driving the opencode agent loop. That gap is now measured by
probe_opencode_compaction.py; what this probe answers is the quantity underneath it -- the size of
the replayed reasoning trace, and what that costs.

The quantity is the replayed trace. `docs/serving.md` documents `reasoning_content` as the field
reasoning is returned in and as an accepted assistant-history alias, `--preserve-thinking` as the
retention control, and opencode's own binary builds assistant messages with
`{role, content, reasoning_content}` when reasoning is non-empty. So the engine sees that trace
re-sent verbatim on every subsequent turn, and the question is what that costs in context and
whether it costs anything in time.

Two arms over identical turns:
  replay -- assistant messages carry reasoning_content, which is what opencode sends.
  strip  -- the same conversation with reasoning_content removed from history.

A fixed seed is sent because the lane samples at temperature 1.0; without it the two arms generate
different answers, and different answer lengths would read as a context-growth difference that is
really a text difference. Arms run sequentially against one lane each, never concurrently, because
this card holds one model.

Everything is read from the engine's own request log -- result.prompt_tokens, result.
prefix_cache_hit_tokens, result.model_thinking_tokens, first_output_timing.prefill.gpu_seconds --
rather than from a wall clock.

The lane comes from tools/release/profiles.py through lane_process, so this measures the shipped
configuration rather than a copy of it that can drift.

What this does NOT measure: whether opencode's compaction trigger fires. For that, see
probe_opencode_compaction.py. For the cold prefill a compaction would cause, see
probe_cold_prefill.py.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import lane_process as lane  # noqa: E402

# A coding conversation where each turn refers back to the last, so every answer is a real
# continuation and the growth measured is the growth a coding session produces. The answers are
# deliberately short: that is what isolates the reasoning tax from the cost of tool output, which
# is the other half of a real session's context and is not what this is measuring.
TURNS = [
    "Write a Python function `norm(rows)` that L2-normalises each row of a list of floats, "
    "handling the empty list and single-element rows. Show only the code.",
    "Now add a `dims` keyword argument to `norm` that optionally asserts the expected length of "
    "each row, raising ValueError on mismatch. Show only the revised function.",
    "Add type hints to the function from the previous turn using `from __future__ import "
    "annotations`, and make the input `Sequence[Sequence[float]]` so tuples work too.",
    "Write a test for that function covering: empty list, single-element row, a row needing "
    "normalisation, a dims mismatch raising ValueError, and tuple input. Use plain asserts.",
    "That test has a bug: the dims-mismatch case will not reach the assert because norm returns "
    "early on an empty row. Fix the test and say which line was wrong.",
    "Extend the tests with a property check using a fixed random seed: for 50 random rows, the "
    "norm of the norm equals the norm. Show only the added test code.",
]

SEED = 20261013


def run_arm(profile: dict, port: int, arm: str, log: pathlib.Path, turns: int,
            max_tokens: int, serve: str | None, artifact: str | None) -> list[dict]:
    model_id = profile["model_id"]
    process, base = lane.start_lane(profile, port, log, serve, artifact)
    rows: list[dict] = []
    try:
        lane.wait_healthy(base, process)
        messages: list[dict] = [
            {"role": "system", "content": "You are a precise coding assistant. Answer directly."}
        ]
        for index, user in enumerate(TURNS[:turns], start=1):
            messages.append({"role": "user", "content": user})
            response = lane.chat(base, model_id, messages, max_tokens=max_tokens,
                                 seed=SEED, reasoning_effort="xhigh")
            message = response["choices"][0]["message"]
            usage = response.get("usage", {})
            reasoning = message.get("reasoning_content") or ""
            assistant: dict = {"role": "assistant", "content": message.get("content") or ""}
            if arm == "replay" and reasoning:
                assistant["reasoning_content"] = reasoning
            messages.append(assistant)
            rows.append({
                "turn": index,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "reasoning_chars": len(reasoning),
                "replayed": arm == "replay" and bool(reasoning),
            })
    finally:
        lane.stop_lane(process)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=len(TURNS))
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--lane", default=lane.QUASAR_DFLASH2, help="launcher file name")
    parser.add_argument("--serve", default=None, help="ninfer-serve executable")
    parser.add_argument("--artifact", default=None, help="model artifact")
    parser.add_argument("--out", default="thinking_growth.json",
                        help="written as given; a bare name keeps it beside the probe")
    args = parser.parse_args()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    profile = lane.lane_profile(args.lane)
    port = int(profile["port"])

    arms: dict[str, list[dict]] = {}
    for offset, arm in enumerate(("replay", "strip")):
        log = out.with_name(out.stem + f".{arm}.requestlog.jsonl")
        if log.exists():
            log.unlink()
        print(f"=== arm {arm}: {args.turns} turns, port {port + offset} ===", flush=True)
        arms[arm] = run_arm(profile, port + offset, arm, log, args.turns,
                            args.max_tokens, args.serve, args.artifact)
        print(f"=== arm {arm} done ===", flush=True)

    # The engine's own per-turn counts, which supersede anything counted from characters here.
    per_turn: list[dict] = []
    for index in range(len(arms["replay"])):
        entry: dict = {"turn": index + 1}
        for arm in ("replay", "strip"):
            done = lane.request_done(out.with_name(out.stem + f".{arm}.requestlog.jsonl"))
            if index < len(done):
                # Keyed arm-first and two segments per name, so these stay out of
                # check_retired_names.py's three-segment shape: that gate reads tools/ for anchors
                # to engine symbols, and a local dict key with the same shape is not one.
                entry[f"{arm}_prompt"] = lane.dig(done[index], "result.prompt_tokens")
                entry[f"{arm}_hit"] = lane.dig(done[index], "result.prefix_cache_hit_tokens")
                entry[f"{arm}_think"] = lane.dig(done[index], "result.model_thinking_tokens")
                entry[f"{arm}_prefill"] = lane.dig(
                    done[index], "first_output_timing.prefill.gpu_seconds")
        entry["reasoning_tax"] = (
            (entry.get("replay_prompt") or 0) - (entry.get("strip_prompt") or 0))
        per_turn.append(entry)

    out.write_text(json.dumps({
        "model_id": profile["model_id"],
        "lane": args.lane,
        "max_tokens": args.max_tokens,
        "thinking_budget": 4096,
        "arms": arms,
        "per_turn": per_turn,
    }, indent=2), encoding="utf-8")

    print("\n=== context growth per turn, from the engine's request log ===")
    print(f"{'turn':>5}{'replay':>9}{'strip':>8}{'reasoning tax':>15}"
          f"{'tax % of prompt':>17}{'thinking tok':>14}{'cache hit':>11}")
    print("-" * 79)
    for e in per_turn:
        replay = e.get("replay_prompt") or 0
        tax = e.get("reasoning_tax") or 0
        share = (tax / replay * 100) if replay else 0.0
        print(f"{e['turn']:>5}{replay:>9}{(e.get('strip_prompt') or 0):>8}{tax:>15}"
              f"{share:>16.1f}%{e.get('replay_think') or 0:>14}"
              f"{e.get('replay_hit') or 0:>11}")

    first = (per_turn[0].get("replay_prompt") or 0) if per_turn else 0
    last = (per_turn[-1].get("replay_prompt") or 0) if per_turn else 0
    if len(per_turn) > 1:
        growth = (last - first) / (len(per_turn) - 1)
        print(f"\n  mean growth per turn      : {growth:.0f} tokens")
        print(f"  262,144 / {growth:.0f}            : {262144 / growth:.0f} turns"
              f"   (short answers; a session returning tool output fills sooner)")
    thinking = [e.get("replay_think") or 0 for e in per_turn]
    if thinking:
        print(f"  mean thinking tokens/turn : {sum(thinking) / len(thinking):.0f}"
              f"   (lane budget 4096)")
    print(f"\nwritten: {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

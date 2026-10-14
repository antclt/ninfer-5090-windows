"""Context growth per turn under reasoning replay, on a shipped lane's own configuration.

docs/opencode-settings.md measured the opencode settings but recorded one gap it could not close
from inside this tree: "I have not measured compaction itself. Driving the opencode agent loop is
outside what I can test from here." The agent loop is indeed out of reach. What it hands the engine
is not -- a conversation whose assistant turns carry `reasoning_content`. That can be driven here.

The quantity this settles is the size of the replayed trace. `docs/serving.md` documents
`reasoning_content` as the field reasoning is returned in and as an accepted assistant-history
alias, `--preserve-thinking` as the retention control, and opencode's own binary builds assistant
messages with `{role, content, reasoning_content}` when reasoning is non-empty. So the engine sees
that trace re-sent verbatim on every subsequent turn, and the question is what that costs in context
and whether it costs anything in time.

Two arms over identical turns:
  replay -- assistant messages carry reasoning_content, which is what opencode sends.
  strip  -- the same conversation with reasoning_content removed from history.

A fixed seed is sent because the lane samples at temperature 1.0; without it the two arms generate
different answers, and different answer lengths would read as a context-growth difference that is
really a text difference. Arms run sequentially against one lane each, never concurrently, because
this card holds one model.

Everything is read from the engine's own request log -- result.prompt_tokens, result.
prefix_cache_hit_tokens, result.model_thinking_tokens, first_output_timing.prefill.gpu_seconds --
rather than from a wall clock, so the numbers are the engine's and a host-side stall cannot
contaminate them.

What this does NOT measure, and what it is not for: whether opencode's compaction trigger fires at
the configured threshold. That lives in the client. This measures the replay it would be managing.
For the cold prefill a compaction would cause, see probe_cold_prefill.py.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.request

SERVE = r"C:\AI\ninfer-v3-windows\build\apps\ninfer-serve.exe"
MODEL = r"C:\AI\models\qwen3_8_27b_nvfp4qat.v3.ninfer"
TEMPLATE = r"C:\AI\ninfer-v3-windows\tools\chat_templates\qwen3_8.jinja"
MODEL_ID = "qwen3.8-27b-quasar-v3-dflash2-vision"

# The lane's own flags, read off start_quasar_v3_dflash2_vision.bat. A harness that starts Serve
# differently measures something else; the launcher says as much about its own environment.
LANE_FLAGS = [
    "--vision", "--spec", "dflash2", "--draft-tokens", "7", "--lm-head-draft",
    "--kv-dtype", "k8v4", "--host", "127.0.0.1",
    "--model-id", MODEL_ID, "--max-context", "262144", "--device-state-slots", "1",
    "--kv-capacity", "auto", "--prefill-chunk", "8192", "--max-concurrency", "1",
    "--host-context-mib", "8192", "--preserve-thinking",
    "--default-thinking-budget", "4096", "--pending-timeout-ms", "600000",
    "--chat-template", TEMPLATE,
]

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


def http(base: str, path: str, payload: dict | None = None, timeout: float = 900.0) -> dict:
    url = f"{base}{path}"
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_healthy(base: str, process: subprocess.Popen, timeout: float = 900.0) -> None:
    deadline = time.monotonic() + timeout
    last: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"ninfer-serve exited during startup, code {process.returncode}")
        try:
            if http(base, "/health", timeout=5.0) == {"status": "ok"}:
                return
        except Exception as error:  # noqa: BLE001 -- startup poll; any failure means not up yet
            last = error
        time.sleep(0.5)
    raise RuntimeError(f"lane not healthy after {timeout:g}s: {last}")


def start_lane(port: int, log: pathlib.Path) -> tuple[subprocess.Popen, str]:
    env = dict(os.environ, NINFER_CUDA_SYNC="blocking")
    command = [SERVE, MODEL, *LANE_FLAGS, "--port", str(port), "--request-log-jsonl", str(log)]
    handle = log.with_suffix(".server.log").open("w", encoding="utf-8", errors="replace")
    process = subprocess.Popen(
        command, stdout=handle, stderr=subprocess.STDOUT, env=env,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    return process, f"http://127.0.0.1:{port}"


def stop_lane(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def dig(obj: object, path: str) -> object:
    current = obj
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def request_done(log: pathlib.Path) -> list[dict]:
    """Read the engine's own record, after the lane has exited. The file is appended to while the
    lane runs, and a half-written line is not a record."""
    out: list[dict] = []
    if not log.exists():
        return out
    for line in log.read_bytes().decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("event") == "request_done":
            out.append(record)
    return out


def run_arm(port: int, arm: str, log: pathlib.Path, turns: int, max_tokens: int) -> list[dict]:
    process, base = start_lane(port, log)
    rows: list[dict] = []
    try:
        wait_healthy(base, process)
        messages: list[dict] = [
            {"role": "system", "content": "You are a precise coding assistant. Answer directly."}
        ]
        for index, user in enumerate(TURNS[:turns], start=1):
            messages.append({"role": "user", "content": user})
            response = http(base, "/v1/chat/completions", {
                "model": MODEL_ID,
                "messages": messages,
                "reasoning_effort": "xhigh",
                "max_tokens": max_tokens,
                "seed": 20261013,   # same text in both arms, so length is comparable
            })
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
        stop_lane(process)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=len(TURNS))
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--out", default=r"C:\AI\bench\thinking_growth.json")
    args = parser.parse_args()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    arms: dict[str, list[dict]] = {}
    for offset, arm in enumerate(("replay", "strip")):
        log = out.with_name(out.stem + f".{arm}.requestlog.jsonl")
        if log.exists():
            log.unlink()
        print(f"=== arm {arm}: {args.turns} turns, port {args.port + offset} ===", flush=True)
        arms[arm] = run_arm(args.port + offset, arm, log, args.turns, args.max_tokens)
        print(f"=== arm {arm} done ===", flush=True)

    # The engine's own per-turn counts, which supersede anything counted from characters here.
    per_turn: list[dict] = []
    for index in range(len(arms["replay"])):
        entry: dict = {"turn": index + 1}
        for arm in ("replay", "strip"):
            log = out.with_name(out.stem + f".{arm}.requestlog.jsonl")
            done = request_done(log)
            if index < len(done):
                entry[f"{arm}_prompt"] = dig(done[index], "result.prompt_tokens")
                entry[f"{arm}_hit"] = dig(done[index], "result.prefix_cache_hit_tokens")
                entry[f"{arm}_think"] = dig(done[index], "result.model_thinking_tokens")
                entry[f"{arm}_prefill"] = dig(done[index],
                                              "first_output_timing.prefill.gpu_seconds")
        # Keyed arm-first and two segments per name, so these stay out of check_retired_names.py's
        # three-segment shape: that gate reads tools/ for anchors to engine symbols, and a local
        # dict key with the same shape is not one. Baseline entries would be the wrong fix --
        # they would teach the gate to expect local keys, which is the class it exists to catch.
        entry["reasoning_tax"] = (
            (entry.get("replay_prompt") or 0) - (entry.get("strip_prompt") or 0))
        per_turn.append(entry)

    out.write_text(json.dumps({
        "model_id": MODEL_ID,
        "lane_flags": LANE_FLAGS,
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
        per_turn_growth = (last - first) / (len(per_turn) - 1)
        print(f"\n  mean growth per turn      : {per_turn_growth:.0f} tokens")
        print(f"  262,144 / {per_turn_growth:.0f}            : "
              f"{262144 / per_turn_growth:.0f} turns"
              f"   (short answers; a session returning tool output fills sooner)")
    thinking = [e.get("replay_think") or 0 for e in per_turn]
    if thinking:
        print(f"  mean thinking tokens/turn : {sum(thinking) / len(thinking):.0f}"
              f"   (lane budget 4096)")
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

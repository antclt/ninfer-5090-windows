#!/usr/bin/env python3
"""Does the cached path produce the same tokens as the cold one, on the same continuation?

The question is the last open item of the lane-coverage list ("cross-path determinism"). It is not
"does the prefix cache engage" -- that is `check_shared_prefix_reuse.py` -- it is whether the tokens a
client receives depend on which path served them, which matters because a cached path splices a prefix
that was encoded in another request while the cold path encoded it in this one.

Four traps, each of which makes the comparison say nothing while looking fine:

  * **A request built from plain strings can never take the shared path.** A breakpoint is read off a
    content *part* (`check_shared_prefix_reuse.py:147-157`), so an unmarked OpenAI arm correctly reads
    `path: root` even when it reuses. Every request here is built from parts and the breakpoint is
    placed on the *last shared message*, which is where the shared prefix actually ends.
  * **An exact repeat never decodes.** The second identical request takes a replay and
    returns a stored response, so comparing it with the first compares a recording with a generation.
    The warm arm therefore shares its prefix with a *different* continuation.
  * **The cold arm has to be cold.** A fresh lane serves the first request with nothing to reuse, and
    the request's own `path` field is checked rather than assumed.
  * **The warm arm has to take the shared path.** Only a a checkpoint reuse path settles this; if
    the warm request reads `root`, the run is void and says so instead of reporting identity.

Both arms are greedy, so identical inputs must produce identical tokens if the path is not part of the
semantics -- which is exactly the claim under test.

Usage:
    python tools/bench/probe_cross_path_determinism.py --lane quasar_v3_dflash2_vision --tag run1
"""
from __future__ import annotations

import argparse
import http.client
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[2]
SERVE = REPO / "build" / "apps" / ("ninfer-serve.exe" if sys.platform == "win32" else "ninfer-serve")
LOG_DIR = REPO / "profiles" / "bench" / "cross-path"
CORPUS = REPO / "eval" / "corpora" / "perplexity-1m" / "data"

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--lane", default="quasar_v3_dflash2_vision",
                    help="substring of the launcher profile file to run")
parser.add_argument("--artifact", default=r"C:\AI\models\qwen3_8_27b_nvfp4qat.v3.ninfer")
parser.add_argument("--port", type=int, default=18210)
parser.add_argument("--prefix-chars", type=int, default=40_000,
                    help="the shared prefix's size; it has to be worth caching, not a paragraph")
parser.add_argument("--max-tokens", type=int, default=64)
parser.add_argument("--startup-timeout", type=float, default=300.0)
parser.add_argument("--tag", default="run")
args = parser.parse_args()

LOG_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "tools" / "release"))
from profiles import PROFILES, launcher_args  # noqa: E402
from engine import spawn_engine, stop_engine  # noqa: E402

profile = next(p for p in PROFILES if args.lane in p["file"])
request_log = LOG_DIR / f"requests-{args.tag}.jsonl"
print(f"lane: {profile['file']}  artifact: {args.artifact}", flush=True)

CONTINUATION = "\n\nSummarise the code above in one sentence."
ALTERNATE = "\n\nName the data structures used above, one per line."


def shared_prefix(chars: int) -> str:
    pieces: list[str] = []
    total = 0
    for candidate in sorted(CORPUS.rglob("*.txt")):
        body = candidate.read_text(encoding="utf-8", errors="replace")
        pieces.append(body)
        total += len(body)
        if total >= chars:
            break
    text = "".join(pieces)
    if len(text) < chars:
        raise SystemExit(f"corpus holds {len(text)} characters; {chars} requested")
    return text[:chars]


def as_part(content: str, role: str, marked: bool) -> dict[str, Any]:
    part: dict[str, Any] = {"type": "text", "text": content}
    if marked:
        part["prompt_cache_breakpoint"] = {"mode": "explicit"}
    return {"role": role, "content": [part]}


def conversation(prefix: str, tail: str) -> list[dict[str, Any]]:
    """A shared prefix ending at a marked message, then a distinct continuation."""
    return [as_part(prefix, "user", True), as_part("Noted.", "assistant", True),
            {"role": "user", "content": tail}]


def phases(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def ask(port: int, model_id: str, messages: list[dict[str, Any]]) -> dict:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
    body = json.dumps({"model": model_id, "messages": messages, "max_tokens": args.max_tokens,
                       "temperature": 0, "enable_thinking": False})
    connection.request("POST", "/v1/chat/completions", body=body.encode("utf-8"),
                       headers={"Content-Type": "application/json"})
    response = connection.getresponse()
    payload = json.loads(response.read().decode("utf-8", errors="replace"))
    connection.close()
    choice = (payload.get("choices") or [{}])[0]
    return {"status": response.status, "text": (choice.get("message") or {}).get("content") or "",
            "usage": payload.get("usage") or {}}


def start_lane(tag: str) -> tuple[subprocess.Popen, str]:
    flags = launcher_args(profile, port=args.port, model_id="qwen3.8-27b-crosspath") + [
        "--request-log-jsonl", str(request_log)]
    handle = (LOG_DIR / f"lane-{tag}.log").open("w", encoding="utf-8", errors="replace")
    process = spawn_engine([str(SERVE), args.artifact, *flags], handle, cwd=str(REPO))
    deadline = time.time() + args.startup_timeout
    while time.time() < deadline:
        time.sleep(2.0)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/v1/models", timeout=3) as r:
                return process, json.loads(r.read())["data"][0]["id"]
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            if process.poll() is not None:
                raise SystemExit(f"lane exited early; see {LOG_DIR / f'lane-{tag}.log'}")
    raise SystemExit("lane never listened")


def last_record(mark: int) -> dict:
    """The completion record added since `mark` records were in the log, if any."""
    records = [record for record in phases(request_log) if record.get("event") != "request_start"]
    return records[-1] if len(records) > mark else {}


def shared_counter() -> int | None:
    """The cache's a checkpoint reuse selection counter, from the log's latest throughput record.

    Corroboration only. The counters are written on an interval, so a short request can finish between
    two records -- the discriminator that works for every request is the hit count below.
    """
    value = None
    for record in phases(request_log):
        selections = (record.get("context_cache") or {}).get("selections") or {}
        if "a checkpoint reuse" in selections:
            value = selections["a checkpoint reuse"]
    return value


def hit_tokens(record: dict) -> int:
    return int((record.get("result") or {}).get("prefix_cache_hit_tokens") or 0)


prefix = shared_prefix(args.prefix_chars)
print(f"shared prefix: {len(prefix):,} characters (~{len(prefix) // 4:,} tokens)", flush=True)

# Cold arm: a fresh lane, so nothing can be reused.
print("\n=== cold arm ===", flush=True)
process, model_id = start_lane("cold")
cold_mark = len([r for r in phases(request_log) if r.get("event") != "request_start"])
cold = ask(args.port, model_id, conversation(prefix, CONTINUATION))
cold_record = last_record(cold_mark)
stop_engine(process)
print(f"  http {cold['status']}, {len(cold['text'])} characters of output; "
      f"prefix_cache_hit_tokens {hit_tokens(cold_record)}", flush=True)

# Warm arm: a fresh lane, a different continuation sharing the prefix, then this one.
print("\n=== warm arm ===", flush=True)
process, model_id = start_lane("warm")
warm_mark = len([r for r in phases(request_log) if r.get("event") != "request_start"])
seed = ask(args.port, model_id, conversation(prefix, ALTERNATE))
seed_record = last_record(warm_mark)
warm = ask(args.port, model_id, conversation(prefix, CONTINUATION))
warm_record = last_record(warm_mark)
stop_engine(process)
print(f"  seed: prefix_cache_hit_tokens {hit_tokens(seed_record)}", flush=True)
print(f"  warm: http {warm['status']}, {len(warm['text'])} characters of output; "
      f"prefix_cache_hit_tokens {hit_tokens(warm_record)}, a checkpoint reuse counter "
      f"{shared_counter()}", flush=True)

identical = cold["text"] == warm["text"] and cold["text"] != ""
print("\n=== verdict ===", flush=True)
if hit_tokens(warm_record) == 0:
    verdict = ("void: the warm continuation reused nothing, so the cached path was never exercised "
               "(check that the breakpoint part is present and the prefix is long enough to publish)")
elif hit_tokens(cold_record) != 0:
    verdict = "void: the cold arm reused something, so there was no cold baseline"
elif not cold["text"] or not warm["text"]:
    verdict = "void: an arm produced no text"
else:
    verdict = ("identical tokens on both paths" if identical
               else "the tokens differ between the cold and the cached path")
print(f"  cold hit {hit_tokens(cold_record)} tokens, warm hit {hit_tokens(warm_record)} tokens", flush=True)
print(f"  {verdict}", flush=True)
if not identical and cold["text"] and warm["text"]:
    first = next((i for i, (a, b) in enumerate(zip(cold["text"], warm["text"])) if a != b),
                 min(len(cold["text"]), len(warm["text"])))
    print(f"  first difference at character {first}: "
          f"cold {cold['text'][max(0, first - 20):first + 20]!r} "
          f"warm {warm['text'][max(0, first - 20):first + 20]!r}", flush=True)

summary = {"lane": profile["file"], "artifact": args.artifact, "prefix_chars": len(prefix),
           "cold": {"hit_tokens": hit_tokens(cold_record), "text": cold["text"],
                    "usage": cold["usage"]},
           "seed": {"hit_tokens": hit_tokens(seed_record), "usage": seed["usage"]},
           "warm": {"hit_tokens": hit_tokens(warm_record), "text": warm["text"],
                    "usage": warm["usage"], "a checkpoint reuse": shared_counter()},
           "identical": identical, "verdict": verdict}
result_path = LOG_DIR / f"result-{args.tag}.json"
result_path.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8", newline="\n")
print(f"\nwrote {result_path}", flush=True)

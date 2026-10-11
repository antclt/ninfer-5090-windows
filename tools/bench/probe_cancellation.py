#!/usr/bin/env python3
"""Does an abandoned request release its slot, and where does the release actually happen?

Upstream #184 (Neroued/ninfer, open) reports that with `--max-concurrency 1` a client lost during
*context materialization or long synchronous prefill* keeps the only slot until that work finishes: the
serve layer's heartbeat cannot run while the engine is synchronous inside the request, and the engine
skips cancellation while a context transaction is open. The same tree carries the fix for #111
(`ce09aee5`), which covers disconnects *during generation*.

Four arms, and the comparison is the measurement: a busy lane queues a follower regardless, so "the
follower waited" is not evidence. What separates the cases is whether an abandoned request is released
*earlier* than a live one of the same size.

  * `idle`      -- the follower on an idle lane. The floor for every other arm.
  * `abandoned` -- the follower sent after the prefill's client is gone (RST). #184 predicts it waits as
                   long as `live`; a release during prefill predicts it waits ~0.
  * `live`      -- the follower sent behind a *live* prefill of the same size. It must wait: this is the
                   reference the abandoned case has to beat.
  * `generation` -- the same, abandoned mid-generation, where #111's fix should already release it.

Three ways this instrument can lie, each closed by construction rather than by hope:

  * **A warm prefix has no prefill to abandon.** The first version of this probe sent the same huge
    prompt twice, and the second one was answered from the prefix cache in 17 ms
    (`prefix_cache_hit_tokens: 158447`) -- so the "abandoned" arm never had work in flight. Each arm now
    sends a *different* slice of the corpus, and the live arm's own record is checked to show a cold
    prefill (`prefix_cache_hit_tokens` must be 0).
  * **A generation that stops on its own is not abandoned.** The first version's subject hit EOS after 50
    tokens, 2 s before the socket closed. The generation arm now asks for a long answer and the verdict
    requires the subject to have *no* completion record at all -- that absence is the evidence that it
    was cancelled rather than finished.
  * **A clean close is not a disconnect.** The abandoned client closes with `SO_LINGER 0`, which sends
    RST; a FIN would let the server keep writing into a buffer nobody reads.

Phases are read from the lane's own request log by field name, never by position.

Usage:
    python tools/bench/probe_cancellation.py --lane quasar_v3_dflash2_vision --tag run2
"""
from __future__ import annotations

import argparse
import http.client
import json
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[2]
SERVE = REPO / "build" / "apps" / ("ninfer-serve.exe" if sys.platform == "win32" else "ninfer-serve")
LOG_DIR = REPO / "profiles" / "bench" / "cancellation"
CORPUS = REPO / "eval" / "corpora" / "perplexity-1m" / "data"

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--lane", default="quasar_v3_dflash2_vision",
                    help="substring of the launcher profile file to run")
parser.add_argument("--artifact", default=r"C:\AI\models\qwen3_8_27b_nvfp4qat.v3.ninfer")
parser.add_argument("--port", type=int, default=18200)
parser.add_argument("--prompt-chars", type=int, default=590_000,
                    help="the cold prompt's size; ~590k characters is the 148k-token context of #184")
parser.add_argument("--kill-after", type=float, default=2.0,
                    help="seconds before the abandoned client closes its socket")
parser.add_argument("--generation-tokens", type=int, default=2048)
parser.add_argument("--followup-tokens", type=int, default=16)
parser.add_argument("--startup-timeout", type=float, default=300.0)
parser.add_argument("--tag", default="run")
args = parser.parse_args()

LOG_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "tools" / "release"))
from profiles import PROFILES, launcher_args  # noqa: E402

profile = next(p for p in PROFILES if args.lane in p["file"])
request_log = LOG_DIR / f"requests-{args.tag}.jsonl"
lane_log_path = LOG_DIR / f"lane-{args.tag}.log"
flags = launcher_args(profile, port=args.port, model_id="qwen3.8-27b-cancel") + [
    "--request-log-jsonl", str(request_log)]
print(f"lane: {profile['file']}  artifact: {args.artifact}", flush=True)
print(f"request log: {request_log}", flush=True)


def cold_prompt(chars: int, offset: int = 0) -> str:
    """Real text of the requested size, taken from a distinct offset so no arm reuses another's cache."""
    pieces: list[str] = []
    total = 0
    for candidate in sorted(CORPUS.rglob("*.txt")):
        body = candidate.read_text(encoding="utf-8", errors="replace")
        pieces.append(body)
        total += len(body)
        if total >= offset + chars:
            break
    text = "".join(pieces)
    if len(text) < offset + chars:
        raise SystemExit(f"corpus holds {len(text)} characters; {offset + chars} requested")
    return text[offset:offset + chars]


LONG_ANSWER = ("\n\nWrite a complete, heavily commented C++ implementation of a lock-free MPMC queue "
               "using hazard pointers. Explain each design decision as you go, and do not stop before "
               "you have written at least 1500 tokens of code and prose.")
FOLLOWER = "Explain what a semaphore is in one sentence."


def post(port: int, model_id: str, prompt: str, max_tokens: int, timeout: float = 900.0) -> dict:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    body = json.dumps({"model": model_id, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0})
    started = time.time()
    connection.request("POST", "/v1/chat/completions", body=body.encode("utf-8"),
                       headers={"Content-Type": "application/json"})
    response = connection.getresponse()
    response.read()
    connection.close()
    return {"status": response.status, "seconds": time.time() - started}


def post_and_abandon(port: int, model_id: str, prompt: str, max_tokens: int, kill_after: float) -> None:
    """Send the request, read nothing, then close with RST so the server sees the peer disappear."""
    body = json.dumps({"model": model_id, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0}).encode("utf-8")
    head = (f"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
            f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n"
            "Connection: keep-alive\r\nAccept: text/event-stream\r\n\r\n").encode("ascii")
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("127.0.0.1", port))
    client.sendall(head + body)
    time.sleep(kill_after)
    client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    client.close()


def completions(path: Path) -> list[dict]:
    """Only the records that describe a finished request; `request_start` events carry no phases."""
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("event") != "request_start":
            records.append(record)
    return records


def request_id(record: dict) -> int:
    return (record.get("request") or {}).get("request_id", -1)


def ids_now() -> set[int]:
    """Every request id the log has *so far*. Ids, not counts: the log's writes are buffered, so a
    record that has happened may not be readable yet, and a count taken at that moment shifts every
    later arm by one -- which is exactly how the first version of this probe classified a cancelled
    request as a completed one."""
    return {request_id(record) for record in completions(request_log)}


def completed(record: dict) -> bool:
    """A finished request computed its whole prompt; a cancelled one stopped mid-prefill.

    `computed_prefill_tokens` equals `prompt_tokens` exactly on every completed request this engine
    serves (docs/research/lane-coverage-2026-10-08.md records the check at 77, 325 and 2,575 tokens),
    and a cancelled request keeps the partial count -- 32,768 of 158,447 in the run that produced this
    docstring. So the invariant is the discriminator, and it does not depend on a status field.
    """
    result = record.get("result") or {}
    return result.get("computed_prefill_tokens") == result.get("prompt_tokens")


def queue_wait(record: dict) -> float | None:
    return (record.get("engine_timing") or {}).get("queue_wait_seconds")


def cold_hit(record: dict) -> int | None:
    return (record.get("result") or {}).get("prefix_cache_hit_tokens")


with lane_log_path.open("w", encoding="utf-8", errors="replace") as lane_log:
    process = subprocess.Popen([str(SERVE), args.artifact, *flags],
                               cwd=str(REPO), stdout=lane_log, stderr=subprocess.STDOUT)
    try:
        deadline = time.time() + args.startup_timeout
        model_id = None
        while time.time() < deadline:
            time.sleep(2.0)
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/v1/models", timeout=3) as r:
                    model_id = json.loads(r.read())["data"][0]["id"]
                    break
            except (urllib.error.URLError, OSError, json.JSONDecodeError):
                if process.poll() is not None:
                    raise SystemExit(f"lane exited early; see {lane_log_path}")
        if model_id is None:
            raise SystemExit(f"lane never listened; see {lane_log_path}")
        print(f"lane up as {model_id} at {time.strftime('%H:%M:%S')}", flush=True)

        prompt_a = cold_prompt(args.prompt_chars, offset=0)
        prompt_b = cold_prompt(args.prompt_chars, offset=len(prompt_a))
        arms: dict[str, dict] = {}

        def followup(name: str, follower: dict, before: set[int]) -> None:
            """Record the follower, and whether the subject completed or was cancelled.

            Records are selected by request id, and the subject is the lowest new id because the probe
            always sends it first. Whether it *completed* is the engine's own invariant, not the
            presence of a record: a cancelled request writes one too.
            """
            new = [record for record in completions(request_log) if request_id(record) not in before]
            new.sort(key=request_id)
            follower_record = new[-1] if new else None
            subjects = new[:-1]
            subject = subjects[0] if subjects else None
            arms[name] = {"follower_seconds": follower["seconds"],
                          "queue_wait": queue_wait(follower_record) if follower_record else None,
                          "subject_completed": completed(subject) if subject else False,
                          "subject_cancelled": bool(subject) and not completed(subject),
                          "subject_cold": (cold_hit(subject) == 0 if subject else None)}
            print(f"  [{name}] follower wall {follower['seconds']:.2f} s, "
                  f"queue_wait {arms[name]['queue_wait']}, "
                  f"subject completed: {arms[name]['subject_completed']}", flush=True)
            if subject:
                timings = subject.get("timings_seconds") or {}
                result = subject.get("result") or {}
                print(f"  [{name}] subject: prefill {timings.get('prefill', float('nan')):.2f} s, "
                      f"computed {result.get('computed_prefill_tokens')} of "
                      f"{result.get('prompt_tokens')} prompt tokens, "
                      f"prefix_cache_hit_tokens {cold_hit(subject)}", flush=True)

        # idle: the floor. Everything below is read against it.
        print("\n=== idle ===", flush=True)
        followup("idle", post(args.port, model_id, FOLLOWER, args.followup_tokens), ids_now())

        # abandoned: #184's case. A cold prefill whose client disappears 2 s in.
        print("\n=== abandoned prefill (#184) ===", flush=True)
        before = ids_now()
        post_and_abandon(args.port, model_id, prompt_a, 1, args.kill_after)
        followup("abandoned", post(args.port, model_id, FOLLOWER, args.followup_tokens), before)

        # live: the reference. The same work, with the client still there.
        print("\n=== live prefill (reference) ===", flush=True)
        before = ids_now()
        live: dict = {}
        thread = threading.Thread(target=lambda: live.update(
            post(args.port, model_id, prompt_b, 1)), daemon=True)
        thread.start()
        time.sleep(args.kill_after)
        followup("live", post(args.port, model_id, FOLLOWER, args.followup_tokens), before)
        thread.join(timeout=600)

        # generation: #111's case, where a release is expected.
        print("\n=== abandoned generation (#111) ===", flush=True)
        before = ids_now()
        post_and_abandon(args.port, model_id, LONG_ANSWER, args.generation_tokens, args.kill_after)
        followup("generation", post(args.port, model_id, FOLLOWER, args.followup_tokens), before)

        print("\n=== verdict ===", flush=True)
        print("  the follower's own work is ~0.06 s, so its wall time is the wait it spent queued",
              flush=True)
        live_wait = arms.get("live", {}).get("queue_wait")
        for name in ("idle", "abandoned", "live", "generation"):
            arm = arms.get(name)
            if not arm:
                continue
            wait = arm.get("queue_wait")
            wait_text = f"{wait:.2f} s" if isinstance(wait, (int, float)) else "unknown"
            print(f"  {name:<11} follower wall {arm['follower_seconds']:7.2f} s  queue_wait {wait_text}  "
                  f"subject completed: {arm['subject_completed']}", flush=True)
        for name in ("abandoned", "generation"):
            arm = arms.get(name)
            if not arm:
                continue
            if arm["subject_completed"]:
                print(f"  {name}: VOID -- the subject finished on its own, so nothing was abandoned",
                      flush=True)
            elif live_wait is None:
                print(f"  {name}: VOID -- no live reference to compare against", flush=True)
            else:
                released = ("released early" if arm["follower_seconds"] < live_wait / 2
                            else "held like a live request")
                print(f"  {name}: the subject never completed (cancelled) and the follower waited "
                      f"{arm['follower_seconds']:.2f} s against {live_wait:.2f} s behind live work "
                      f"-> {released}", flush=True)
        if arms.get("live", {}).get("subject_cold") is False:
            print("  VOID: the live reference's prefill was served from cache, so it held the slot for "
                  "no work at all", flush=True)

        summary = {"lane": profile["file"], "artifact": args.artifact, "prompt_chars": len(prompt_a),
                   "kill_after_seconds": args.kill_after, "arms": arms}
        result_path = LOG_DIR / f"result-{args.tag}.json"
        result_path.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(f"\nwrote {result_path}", flush=True)
    finally:
        time.sleep(2.0)
        process.terminate()
        try:
            process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            pass

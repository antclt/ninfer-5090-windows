"""Cold-cache prefill cost against prompt size, on a shipped lane's own configuration.

docs/opencode-settings.md carries this caveat: the compaction trigger arithmetic follows the
documented formula and is exact, but the ~50 s post-compaction re-prefill is "an estimate derived
from measured prefill rates, not a measurement of compaction." The rate it derives from -- ~10,075
tok/s at 30k against ~3,682 tok/s at 200k -- has never been re-measured on this engine.

What is measurable here is the cost the estimate rests on: what one prefill of a large prompt costs
when the prefix cache cannot help it. Compaction rewrites the conversation prefix, so the re-prefill
that follows is by construction a cold one. Measuring the cold case therefore measures the number
the caveat calls an estimate, without needing opencode's agent loop to fire.

Cold is enforced by starting a fresh lane per size rather than by evicting: a fresh lane has an empty
context cache by construction, which is a stronger guarantee than any eviction argument. The warm arm
sends the same prompt twice on one lane, which is what the prefix cache is for, and the ratio between
the arms is the whole value of preserving the cache across a compaction boundary.

Every figure is read from the engine's own request log -- first_output_timing.prefill.gpu_seconds,
result.prompt_tokens, result.prefix_cache_hit_tokens -- not from a wall clock, because prefill is a
device phase and the host-side submit and post phases would otherwise be counted as stall.

Arms run sequentially, one lane at a time: this card holds one model.
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
# differently measures something else -- the launcher says as much about its environment.
LANE_FLAGS = [
    "--vision", "--spec", "dflash2", "--draft-tokens", "7", "--lm-head-draft",
    "--kv-dtype", "k8v4", "--host", "127.0.0.1",
    "--model-id", MODEL_ID, "--max-context", "262144", "--device-state-slots", "1",
    "--kv-capacity", "auto", "--prefill-chunk", "8192", "--max-concurrency", "1",
    "--host-context-mib", "8192", "--preserve-thinking",
    "--default-thinking-budget", "4096", "--pending-timeout-ms", "600000",
    "--chat-template", TEMPLATE,
]

# Realistic mixed prose and code rather than one repeated token: a prefix cache and a prefill loop
# both behave differently on degenerate input, and a number measured on "a " repeated to 200k would
# not describe a compaction re-prefill of a real session.
BLOCK = """
def resize_image(source, target, keep_aspect=True):
    \"\"\"Scale an image to fit a target box, preserving aspect ratio.\"\"\"
    src_w, src_h = source.size
    if keep_aspect:
        scale = min(target[0] / src_w, target[1] / src_h)
        new_size = (max(1, int(src_w * scale)), max(1, int(src_h * scale)))
    else:
        new_size = target
    return source.resize(new_size, Image.LANCZOS)

class RequestThrottle:
    \"\"\"Bound request rate so a burst cannot starve the decode loop.\"\"\"
    def __init__(self, requests_per_second, burst=1):
        self.rate = float(requests_per_second)
        self.burst = max(1, int(burst))
        self.tokens = float(self.burst)
        self.updated = time.monotonic()

    def acquire(self, amount=1):
        while True:
            now = time.monotonic()
            self.tokens = min(self.burst, self.tokens + (now - self.updated) * self.rate)
            self.updated = now
            if self.tokens >= amount:
                self.tokens -= amount
                return
            time.sleep(max(0.0, (amount - self.tokens) / self.rate))

# The prefill loop is chunked so a large prompt never holds the whole activation set resident.
# Each chunk writes its KV pages into the sequence's page table; the page table is what a later
# request matches a cached prefix against, so a chunk boundary is invisible to the cache.
"""


def build_prompt(approx_tokens: int) -> str:
    """Grow the block until the text is plausibly the requested size. The engine's own
    result.prompt_tokens is what every table reports, so this only has to get the order right."""
    target_chars = approx_tokens * 4
    parts: list[str] = []
    size = 0
    while size < target_chars:
        parts.append(BLOCK)
        size += len(BLOCK)
    return "".join(parts)


def http(base: str, path: str, payload: dict | None = None, timeout: float = 1800.0) -> dict:
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
    """Read the engine's own record. Read after the process exits: the file is being appended
    while the lane runs, and a half-written line is not a record (rule 56)."""
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


def send(base: str, prompt: str, max_tokens: int) -> None:
    # reasoning_effort none: this measures prefill, and a thinking turn would add decode time to a
    # phase whose cost is the prefill alone.
    http(base, "/v1/chat/completions", {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "reasoning_effort": "none",
        "max_tokens": max_tokens,
        "seed": 20261013,
    })


def row(record: dict, label: str) -> dict:
    return {
        "arm": label,
        "prompt_tokens": dig(record, "result.prompt_tokens"),
        "prefix_cache_hit_tokens": dig(record, "result.prefix_cache_hit_tokens"),
        "prefix_reuse_path": dig(record, "result.prefix_reuse_path"),
        "prefill_gpu_seconds": dig(record, "first_output_timing.prefill.gpu_seconds"),
        "prefill_wait_seconds": dig(record, "first_output_timing.prefill.wait_seconds"),
        "ttft_seconds": dig(record, "first_output_timing.elapsed_seconds"),
        "completion_tokens": dig(record, "result.completion_tokens"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[30000, 100000, 200000],
                        help="approximate prompt token counts to measure")
    parser.add_argument("--warm-size", type=int, default=200000,
                        help="size sent twice on one lane to show the cache's effect")
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--out", default=r"C:\AI\bench\cold_prefill.json")
    args = parser.parse_args()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    port = 8300

    # Cold arm: a fresh lane per size, so the context cache is empty by construction.
    for size in args.sizes:
        log = out.with_name(out.stem + f".cold{size}.requestlog.jsonl")
        if log.exists():
            log.unlink()
        print(f"=== cold {size} tokens (fresh lane) ===", flush=True)
        process, base = start_lane(port, log)
        try:
            wait_healthy(base, process)
            send(base, build_prompt(size), args.max_tokens)
        finally:
            stop_lane(process)
        done = request_done(log)
        if done:
            results.append(row(done[-1], f"cold-{size}"))
        port += 1

    # Warm arm: the same prompt twice on one lane. The first is cold, the second is the cache hit
    # that a compaction would destroy.
    log = out.with_name(out.stem + ".warm.requestlog.jsonl")
    if log.exists():
        log.unlink()
    print(f"=== warm {args.warm_size} tokens (same lane, sent twice) ===", flush=True)
    process, base = start_lane(port, log)
    try:
        wait_healthy(base, process)
        prompt = build_prompt(args.warm_size)
        send(base, prompt, args.max_tokens)
        send(base, prompt, args.max_tokens)
    finally:
        stop_lane(process)
    done = request_done(log)
    if len(done) >= 2:
        results.append(row(done[0], f"warm1-{args.warm_size}"))
        results.append(row(done[1], f"warm2-{args.warm_size}"))

    out.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")

    print("\n=== prefill cost against prompt size, from the engine's request log ===")
    print(f"{'arm':<16}{'prompt tok':>11}{'cache hit':>11}{'path':>12}"
          f"{'prefill s':>11}{'tok/s':>10}{'TTFT s':>9}")
    print("-" * 80)
    for r in results:
        tokens = r["prompt_tokens"] or 0
        hit = r["prefix_cache_hit_tokens"] or 0
        prefill = r["prefill_gpu_seconds"]
        rate = (tokens / prefill) if prefill else 0.0
        print(f"{r['arm']:<16}{tokens:>11}{hit:>11}{str(r['prefix_reuse_path']):>12}"
              f"{prefill:>11.4f}{rate:>10.0f}{(r['ttft_seconds'] or 0):>9.3f}")
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

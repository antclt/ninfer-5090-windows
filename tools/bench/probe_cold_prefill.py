"""Cold-cache prefill cost against prompt size, on a shipped lane's own configuration.

docs/opencode-settings.md carries this caveat: the compaction trigger arithmetic follows the
documented formula and is exact, but a post-compaction re-prefill was "an estimate derived from
measured prefill rates, not a measurement of compaction". The rate is measured here.

What is measurable without opencode's agent loop is the cost the estimate rests on: what one prefill
of a large prompt costs when the prefix cache cannot help it. Compaction rewrites the conversation
prefix, so the re-prefill that follows is by construction a cold one. Measuring the cold case
measures the number the caveat called an estimate.

Cold is enforced by starting a fresh lane per size rather than by evicting: a fresh lane has an
empty context cache by construction, which is a stronger guarantee than any eviction argument. The
warm arm sends the same prompt twice on one lane, which is what the prefix cache is for, and the
ratio between the arms is the value of preserving the cache across a compaction boundary.

Every figure is read from the engine's own request log -- first_output_timing.prefill.gpu_seconds,
result.prompt_tokens, result.prefix_cache_hit_tokens -- not from a wall clock, because prefill is a
device phase and the host-side submit and post phases would otherwise be counted as stall.

The lane comes from tools/release/profiles.py through lane_process, so this measures the shipped
configuration rather than a copy of it that can drift.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import lane_process as lane  # noqa: E402

# Realistic mixed prose and code rather than one repeated token: a prefix cache and a prefill loop
# both behave differently on degenerate input, and a number measured on "a " repeated to 200k would
# not describe the prefill of a real prompt.
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


def row(record: dict, label: str) -> dict:
    return {
        "arm": label,
        "prompt_tokens": lane.dig(record, "result.prompt_tokens"),
        "prefix_cache_hit_tokens": lane.dig(record, "result.prefix_cache_hit_tokens"),
        "prefix_reuse_path": lane.dig(record, "result.prefix_reuse_path"),
        "prefill_gpu_seconds": lane.dig(record, "first_output_timing.prefill.gpu_seconds"),
        "ttft_seconds": lane.dig(record, "first_output_timing.elapsed_seconds"),
        "completion_tokens": lane.dig(record, "result.completion_tokens"),
    }


def send(base: str, model_id: str, prompt: str, max_tokens: int) -> None:
    # reasoning_effort none: this measures prefill, and a thinking turn would add decode time to a
    # phase whose cost is the prefill alone.
    lane.chat(base, model_id, [{"role": "user", "content": prompt}],
              max_tokens=max_tokens, reasoning_effort="none")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[30000, 100000, 200000],
                        help="approximate prompt token counts to measure")
    parser.add_argument("--warm-size", type=int, default=200000,
                        help="size sent twice on one lane to show the cache's effect")
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--lane", default=lane.QUASAR_DFLASH2, help="launcher file name")
    parser.add_argument("--serve", default=None, help="ninfer-serve executable")
    parser.add_argument("--artifact", default=None, help="model artifact")
    parser.add_argument("--out", default="cold_prefill.json",
                        help="written under --out-dir; a bare name keeps it beside the probe")
    args = parser.parse_args()

    out = pathlib.Path(args.out)
    profile = lane.lane_profile(args.lane)
    model_id = profile["model_id"]
    port = int(profile["port"])

    results: list[dict] = []

    # Cold arm: a fresh lane per size, so the context cache is empty by construction.
    for size in args.sizes:
        log = out.with_name(out.stem + f".cold{size}.requestlog.jsonl")
        if log.exists():
            log.unlink()
        print(f"=== cold {size} tokens (fresh lane, port {port}) ===", flush=True)
        process, base = lane.start_lane(profile, port, log, args.serve, args.artifact)
        try:
            lane.wait_healthy(base, process)
            send(base, model_id, build_prompt(size), args.max_tokens)
        finally:
            lane.stop_lane(process)
        done = lane.request_done(log)
        if done:
            results.append(row(done[-1], f"cold-{size}"))

    # Warm arm: the same prompt twice on one lane. The first is cold, the second is the cache hit
    # that a compaction would destroy.
    log = out.with_name(out.stem + ".warm.requestlog.jsonl")
    if log.exists():
        log.unlink()
    print(f"=== warm {args.warm_size} tokens (same lane, sent twice) ===", flush=True)
    process, base = lane.start_lane(profile, port, log, args.serve, args.artifact)
    try:
        lane.wait_healthy(base, process)
        prompt = build_prompt(args.warm_size)
        send(base, model_id, prompt, args.max_tokens)
        send(base, model_id, prompt, args.max_tokens)
    finally:
        lane.stop_lane(process)
    done = lane.request_done(log)
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
        prefill = r["prefill_gpu_seconds"]
        rate = (tokens / prefill) if prefill else 0.0
        print(f"{r['arm']:<16}{tokens:>11}{r['prefix_cache_hit_tokens'] or 0:>11}"
              f"{str(r['prefix_reuse_path']):>12}{prefill:>11.4f}{rate:>10.0f}"
              f"{(r['ttft_seconds'] or 0):>9.3f}")
    print(f"\nwritten: {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

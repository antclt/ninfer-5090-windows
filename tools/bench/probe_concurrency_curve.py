"""Throughput and latency against concurrency, for a served lane.

The priority list asks for the curve rather than a point. It matters here more than a single-user
reading suggests: a coding agent issues parallel tool calls, so the interesting region is the left end -
does concurrency 2 cost latency at concurrency 1 - and the point where aggregate throughput stops
improving.

Each request carries a **distinct** prompt. An identical repeat is served from the replay path, so a
curve built on repeated prompts would measure the cache and then report it as concurrency.

    python tools/bench/probe_concurrency_curve.py --lane start_ninfer_v3_dflash2_vision.bat \
        --concurrency 1,2,4 --requests 4
"""
from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "bench"))
sys.path.insert(0, str(REPO / "tools" / "release"))

import realtext_acceptance as rta  # noqa: E402
from profiles import PROFILES  # noqa: E402

CORPUS = REPO / "eval" / "corpora" / "perplexity-1m" / "data" / "ninfer" / "00.txt"


def one_request(port, model_id, prompt, max_tokens, out):
    """Fire one request and record its wall time, token count and the log's own phases."""
    started = time.time()
    answer = rta.post_json(port, {"model": model_id,
                                  "messages": [{"role": "user", "content": prompt}],
                                  "max_tokens": max_tokens, "temperature": 0.0}, 900)
    elapsed = time.time() - started
    if answer["status"] != 200:
        out.append({"ok": False, "elapsed": elapsed})
        return
    usage = answer["payload"].get("usage") or {}
    out.append({"ok": True, "elapsed": elapsed,
                "completion": usage.get("completion_tokens") or 0,
                "prompt": usage.get("prompt_tokens") or 0})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lane", default="dflash2", help="substring of the launcher profile to run")
    parser.add_argument("--concurrency", default="1,2,4", help="comma-separated levels")
    parser.add_argument("--requests", type=int, default=4, help="requests per level")
    parser.add_argument("--chars", type=int, default=4000, help="prompt size per request")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--port", type=int, default=8096)
    args = parser.parse_args()

    profile = next((p for p in PROFILES if args.lane in p.get("file", "")), None)
    if profile is None:
        print(f"no lane matching {args.lane!r}")
        return 1
    raw = profile["art"]
    artifact = Path(raw) if Path(raw).is_absolute() else Path(r"C:\AI\models") / raw
    haystack = CORPUS.read_text(encoding="utf-8", errors="replace")
    levels = [int(c) for c in args.concurrency.split(",")]
    # a distinct slice per request, so no two requests share a prefix
    step = max(1, args.chars // 2)

    print(f"lane {profile['file']}  artifact {artifact.name}")
    print(f"  {args.requests} request(s) per level, {args.chars}-character distinct prompts,"
          f" max_tokens {args.max_tokens}\n")

    flags = rta.launcher_args(profile, port=args.port)
    log_path = Path(str(Path.home()) + r"\AppData\Local\Temp\opencode\curve-serve.log")
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
            print(f"  {'conc':>5} {'reqs':>5} {'wall s':>8} {'agg tok/s':>10} {'per-req tok/s':>14}"
                  f" {'p50 lat s':>10}")
            for level in levels:
                prompts = [haystack[i * step: i * step + args.chars] for i in range(args.requests)]
                results: list[dict] = []
                # warm the lane once before the level, so the first level is not the cold one
                one_request(args.port, model_id, prompts[0], 1, [])
                started = time.time()
                threads = [threading.Thread(target=one_request,
                                            args=(args.port, model_id, prompt, args.max_tokens,
                                                  results))
                           for prompt in prompts]
                # launch in waves of `level`, so each wave is genuinely concurrent
                for start in range(0, len(threads), level):
                    wave = threads[start:start + level]
                    for thread in wave:
                        thread.start()
                    for thread in wave:
                        thread.join()
                wall = time.time() - started
                ok = [r for r in results if r.get("ok")]
                tokens = sum(r["completion"] for r in ok)
                latencies = [r["elapsed"] for r in ok]
                agg = tokens / wall if wall else 0.0
                per = statistics.median([r["completion"] / r["elapsed"] for r in ok]) if ok else 0.0
                print(f"  {level:>5} {len(ok):>5} {wall:>8.1f} {agg:>10.1f} {per:>14.1f}"
                      f" {statistics.median(latencies) if latencies else 0:>10.2f}")
        finally:
            time.sleep(2.0)
            process.terminate()
            try:
                process.wait(timeout=120)
            except Exception:
                pass

    print("\n  Read the left end first: if per-request throughput falls from concurrency 1 to 2, a coding"
          "\n  agent issuing parallel tool calls pays for it. Aggregate throughput rising while per-request"
          "\n  falls is the expected trade, not a regression.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

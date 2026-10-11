#!/usr/bin/env python3
"""Measure item 14's first-request question on a real serving lane, with prefix reuse as the arm.

RESULT 2026-10-02: the transient does not reproduce. At temperature 0.7 with the seed pinned, eight
identical requests in one lane were byte-identical in both arms, across three separate process
launches, while the cache engaged on requests 2..8 (a replay, 39 hit tokens). The
261.7 tok/s figure came from comparing two ninfer_bench RUNS at --warmup 0 against --warmup 1, which
is a configuration comparison and not a first-request effect; ADR-0002 already covers it. So this
harness is retained as the instrument that settles it, not because the answer was surprising.

The bench cannot answer the question. tools/bench/first_request_report.py reads two ninfer_bench runs
and reports no difference (172.31 against 173.65), because the bench drives the Engine directly and
never has a warm prefix cache to be cold against. The 261.7 figure came from a serving lane, so the
serving lane is what has to be measured.

The comparison is two arms, one fresh server each, differing in exactly one flag:

    reuse     the shipped default, including the shared Host quota
    no-reuse  --no-prefix-reuse

Everything else is held fixed, because each of these has been the confound in a version of this
question:

  - the request body is byte-identical across all N requests, so the responses must be too. A digest
    of each response text is recorded: if two responses differ, they are different workloads and their
    speeds are not comparable. This is the check that would have caught the original figure being read
    as a first-request effect when the first request was returning different, shorter text.
  - --temperature and --seed are parameters, not constants. The transient is invisible at
    temperature 0 and visible at a sampling temperature, so --temperature picks the question. The
    seed is pinned because an omitted seed is replaced per request with a FRESH RANDOM one
    (serve_options.h, translate.cpp), which at a sampling temperature makes every request differ for
    a reason unrelated to the question.
  - max_tokens is fixed, so output length cannot vary. "output length, warm-up state" is the pair
    vLLM #17472 names as controls that a speed comparison has to hold.
  - one server per arm, started fresh and stopped afterwards, so arm two cannot inherit arm one's
    warm state, which is the whole thing under test.
  - the server's own log is captured to a file. A lane started from a detached process puts its log
    nowhere, and the only evidence of a rejected request is the one line that says nothing.

READ THE TWO ARMS' DIGESTS AGAINST EACH OTHER, not only within one arm. The 2026-10-02 run found the
two arms produce different text for the same prompt and seed (d9fc0df811cf40f2 against
df5604c53ea0ae6f), byte-stable across three launches, with both request 1s on path root and zero hit
tokens. The arms differ in prefix reuse and in the six capacity flags the engine requires with it, so
the difference is in the prefill the plan chose -- ADR-0002's documented sensitivity, on an axis this
harness adds.

The verdict logic is deliberately conservative and is stated in the report rather than decided here:
this script measures and records, and it refuses to attribute a cause. "No difference" is a result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(REPO_ROOT / "tools" / "release"))
from engine import (  # noqa: E402
    LANE_QUASAR_DFLASH2,
    lane_command,
    start_lane,
    stop_engine,
    wait_ready,
)

# One fixed prompt, fixed output budget. Sampling is a parameter, not a constant: the first-request
# transient this harness was written for is invisible at temperature 0 and visible at a sampling
# temperature, so --temperature selects which question is being asked. A run is reproducible from the
# harness rather than from a shell that was typed once.
PROMPT = (
    "Explain how a split-K GEMM decides how many partial results to write, and why the workspace "
    "size has to be derived from the same shape the launcher uses."
)
MAX_TOKENS = 256


def request_body(temperature: float, seed: int) -> dict[str, Any]:
    return {
        # The lane's own id, from the profile the command is built from: it used to be typed here as
        # well, so the request body and the flags could name different lanes.
        "model": LANE_QUASAR_DFLASH2["model_id"],
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        # Temperature and seed are parameters rather than constants because the two questions this
        # harness answers need different settings. The first-request TRANSIENT is invisible at
        # temperature 0 -- the greedy digest is stable from request 1 -- and visible at the sampling
        # temperature the bench uses, so reproducing it at all requires sampling. The seed is pinned so
        # that any divergence between requests is the cache and not the sampler: without it, a sampling
        # temperature makes every request differ for a reason that has nothing to do with the question.
        "temperature": temperature,
        "seed": seed,
        # Without this the whole output budget goes to thinking: the first run of this harness put
        # 256 model_thinking_tokens against 256 completion_tokens in every request of both arms, so
        # content came back empty and the reply digest was the SHA-256 of the empty string in all
        # sixteen samples. Timing was still being recorded, which is the dangerous part -- it looked
        # like a measurement. enable_thinking is documented at docs/serving.md as a top-level field.
        "enable_thinking": False,
        "stream": False,
    }


def post_completion(port: int, timeout_s: float, temperature: float, seed: int) -> tuple[float, str, int]:
    """Send one request; return elapsed seconds, the reply text, and its token count."""
    body = json.dumps(request_body(temperature, seed)).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        payload = json.loads(response.read().decode("utf-8"))
    elapsed = time.monotonic() - start
    choice = payload["choices"][0]
    text = choice["message"]["content"]
    usage = payload.get("usage") or {}
    completion = int(usage.get("completion_tokens") or 0)
    return elapsed, text, completion


def run_arm(
    exe: Path,
    artifact: Path,
    port: int,
    no_reuse: bool,
    repeats: int,
    log_dir: Path,
    temperature: float,
    seed: int,
) -> dict[str, Any]:
    """Start a fresh lane, send `repeats` identical requests, stop the lane, return the record."""
    label = "no-reuse" if no_reuse else "reuse"
    log_path = log_dir / f"first_request_{label}.log"
    request_log = log_path.with_suffix(".requests.jsonl")
    if request_log.exists():
        request_log.unlink()

    # Exactly one flag differs between the arms. It used to be two: the engine refused
    # --no-prefix-reuse alongside the context-cache capacity options, so those were dropped for the
    # no-reuse arm. Upstream's b9114396 replaced the cache and that refusal with it, so they are kept
    # in both arms now -- which is what this harness's own rule asks for, because a second difference
    # is a confound rather than a controlled comparison.
    extra_flags = ["--request-log-jsonl", str(request_log)]
    if no_reuse:
        extra_flags.append("--no-prefix-reuse")
    command = lane_command(LANE_QUASAR_DFLASH2, exe, artifact, port=port, extra_flags=extra_flags)
    print(f"  [{label}] starting on port {port} (log: {log_path.name})", flush=True)
    process, handle = start_lane(LANE_QUASAR_DFLASH2, exe, artifact, port, log_path,
                                 extra_flags=extra_flags)

    samples: list[dict[str, Any]] = []
    try:
        if not wait_ready(port, process, timeout=900):
            raise RuntimeError(f"[{label}] the lane never became ready; see {log_path}")
        print(f"  [{label}] ready; sending {repeats} identical requests", flush=True)
        for index in range(repeats):
            elapsed, text, completion = post_completion(port, 600.0, temperature, seed)
            # An empty reply is a failed measurement, not a slow one. The identity control below
            # cannot catch it on its own: every response being identically empty passes it, which is
            # exactly what happened in the first run of this harness.
            if not text.strip():
                raise RuntimeError(
                    f"[{label}] request {index + 1} returned {completion} tokens and no content. "
                    f"That is a degenerate workload, not a throughput measurement: with thinking "
                    f"enabled the entire output budget goes to reasoning tokens and message.content "
                    f"stays empty. Check model_thinking_tokens in {request_log.name} before reading "
                    f"any timing from this run."
                )
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
            samples.append(
                {
                    "index": index,
                    "elapsed_s": round(elapsed, 4),
                    "completion_tokens": completion,
                    "tokens_per_s": round(completion / elapsed, 2) if elapsed > 0 else None,
                    "text_sha256_16": digest,
                    "text_chars": len(text),
                    "text_head": text[:80].replace("\n", " "),
                }
            )
            print(
                f"    request {index + 1}: {elapsed:7.3f}s  {completion:4d} tok  "
                f"{completion / elapsed if elapsed else 0:7.2f} tok/s  {digest}",
                flush=True,
            )
    finally:
        stop_engine(process, handle)

    return {
        "arm": label,
        "port": port,
        "no_prefix_reuse": no_reuse,
        "command": command,
        "log": str(log_path),
        "request_log": str(request_log),
        "samples": samples,
        "identical_text": len({s["text_sha256_16"] for s in samples}) == 1,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--exe", type=Path, default=REPO_ROOT / "build" / "apps" / "ninfer-serve.exe")
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--base-port", type=int, default=8186)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, default=REPO_ROOT / "profiles" / "first_request")
    parser.add_argument(
        "--arms",
        default="no-reuse,reuse",
        help="comma-separated arms; each gets its own fresh server",
    )
    args = parser.parse_args()

    if not args.exe.is_file():
        print(f"  FAIL: no serve binary at {args.exe}", file=sys.stderr)
        return 1
    if not args.artifact.is_file():
        print(f"  FAIL: no artifact at {args.artifact}", file=sys.stderr)
        return 1

    args.log_dir.mkdir(parents=True, exist_ok=True)
    arms = [name.strip() for name in args.arms.split(",") if name.strip()]
    report: dict[str, Any] = {
        "artifact": str(args.artifact),
        "exe": str(args.exe),
        "repeats": args.repeats,
        "max_tokens": MAX_TOKENS,
        "temperature": args.temperature,
        "seed": args.seed,
        "prompt_sha256_16": hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()[:16],
        "arms": [],
    }

    for offset, name in enumerate(arms):
        if name not in {"reuse", "no-reuse"}:
            print(f"  FAIL: unknown arm {name!r}", file=sys.stderr)
            return 1
        report["arms"].append(
            run_arm(
                args.exe,
                args.artifact,
                args.base_port + offset,
                no_reuse=(name == "no-reuse"),
                repeats=args.repeats,
                log_dir=args.log_dir,
                temperature=args.temperature,
                seed=args.seed,
            )
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(json.dumps(report, indent=2).encode("utf-8"))
    print(f"\n  wrote {args.out}", flush=True)
    for arm in report["arms"]:
        print(f"  {arm['arm']}: identical_text={arm['identical_text']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
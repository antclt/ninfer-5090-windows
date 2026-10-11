#!/usr/bin/env python3
"""Measure speculative acceptance and decode on real text, at a lane's own configuration.

Why this instrument exists, in one paragraph, because a bench corpus cannot answer this question from
either end of its range: `ninfer_bench -n` generates after a one-token seed, so speculation has no
context to be predictable from (measured 0.085-0.129 acceptance), and the same corpus behind a real
prompt saturates it (0.991 in four consecutive runs), because the corpus is synthetic and tiled. Real
domains sit at 0.44-0.55, and nothing measured there until this tool.

Design decisions, each one a lesson already paid for:

  * **Greedy by default** (`--temperature 0`). At temperature 1.0 the generated text differs per run, and
    acceptance depends on the text: two rounds of the SAME artifact read 47.4% and 31.6%, a 16-point
    swing that cannot resolve any effect. Greedy makes each arm's generation identical, so the shipped
    arm's two rounds agree exactly and the acceptance statistic carries no noise. Use a non-zero
    temperature only to measure the level rather than to compare arms.
  * **The model id is read from `/v1/models` per arm**, never hard-coded. Artifacts carry their own id
    (this one's NVIDIA build serves `qwen3.8-27b-fp8attn`), and a hard-coded id is a 404 that reads like
    a broken artifact.
  * **One port per arm, and the serve is stopped, never a profiler killed.** A profiler terminated
    instead of its subject writes no report and leaves the subject alive holding the port, which is how
    a whole profile came back void.
  * **The lane's flags are composed, never restated.** `--lane` selects a launcher profile and
    `profiles.launcher_args` renders its flags, so a record describes a configuration that ships. A
    hand-written list here once drifted to a depth and a KV dtype that no lane runs, and it omitted the
    lane's environment entirely.
  * **Acceptance is parsed tolerantly** of thousands separators: `accepted 351/1,112 (31.6%)` silently
    failing a `\\d+` pattern reads exactly like a missing line.

The figure to sanity-check before trusting a run is the absolute acceptance level: near 0.99 means the
prompt is too predictable, near 0.10 means too little context, and both mean the protocol is measuring
something other than the lanes' regime.
"""

from __future__ import annotations

import argparse
import http.client
import json
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

ACCEPT = re.compile(r"accepted ([\d,]+)/([\d,]+) \(([\d.]+)%\)")
DECODE = re.compile(r"decode ([\d.]+) tok/s")
REPO = Path(__file__).resolve().parents[2]
SERVE = REPO / "build" / "apps" / ("ninfer-serve.exe" if sys.platform == "win32" else "ninfer-serve")
DEFAULT_TEMPLATE = REPO / "tools" / "chat_templates" / "qwen3_8.jinja"

sys.path.insert(0, str(REPO / "tools" / "release"))
from profiles import PROFILES, launcher_args, launcher_environment  # noqa: E402


def set_flag(flags: list[str], flag: str, value: str | None) -> None:
    """Substitute a lane flag's value in place, or append it when the lane does not pass it.

    Substituting rather than appending is not cosmetic: a duplicate flag is refused, and appending
    would silently leave the lane's own value in front of the probe's.
    """
    if value is None:
        return
    if flag in flags:
        flags[flags.index(flag) + 1] = value
    else:
        flags.extend([flag, value])


def default_text() -> tuple[Path, str]:
    """The first corpus file with enough text, and which file that was.

    It is not necessarily prose, and the name survives only because the instrument and its records use
    it: the perplexity corpus's first match by sort order is this repository's own C++ source under
    data/ninfer/, so an arm recorded as "prose" without printing the source can be mislabelled, which
    happened and is why main() now prints the source and the text's own script mix.
    """
    root = REPO / "eval" / "corpora" / "perplexity-1m"
    for candidate in sorted(root.rglob("*")):
        if not candidate.is_file() or candidate.suffix not in (".txt", ".jsonl"):
            continue
        body = candidate.read_text(encoding="utf-8", errors="replace")
        if candidate.suffix == ".jsonl":
            pieces = []
            for line in body.splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = record.get("text") or record.get("content")
                if isinstance(text, str):
                    pieces.append(text)
                if sum(len(piece) for piece in pieces) > 60000:
                    break
            body = "\n".join(pieces)
        if len(body) >= 40000:
            return candidate, body
    raise SystemExit("no real-text source found under eval/corpora/perplexity-1m; pass --text")


def wait_for_port(port: int, deadline: float) -> bool:
    while time.time() < deadline:
        with socket.socket() as probe:
            probe.settimeout(1.0)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(1.0)
    return False


def get_json(port: int, path: str) -> dict:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    connection.request("GET", path)
    payload = json.loads(connection.getresponse().read().decode("utf-8", errors="replace"))
    connection.close()
    return payload


def post_json(port: int, body: dict, timeout: float) -> dict:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    connection.request("POST", "/v1/chat/completions", body=json.dumps(body).encode("utf-8"),
                       headers={"Content-Type": "application/json"})
    response = connection.getresponse()
    payload = json.loads(response.read().decode("utf-8", errors="replace"))
    connection.close()
    return {"status": response.status, "payload": payload}


def run_arm(artifact: Path, port: int, blocks: list[str], args: argparse.Namespace,
            run_dir: Path, profile: dict) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "serve.log"
    result_path = run_dir / "result.json"
    lane_flags = launcher_args(profile, port=port)
    if args.draft_tokens is not None:
        # A depth A/B is the same artifact under two flag sets, so the lane's own value is substituted
        # rather than left in front of the probe's.
        set_flag(lane_flags, "--draft-tokens", str(args.draft_tokens))
    if args.kv_capacity is not None:
        # Same reasoning: a probe that asks whether a per-request cost tracks capacity needs to shrink
        # it without changing anything else. A capacity below --max-context is refused, so shrinking
        # the reservation means shrinking the context too.
        set_flag(lane_flags, "--kv-capacity", str(args.kv_capacity))
    if args.max_context is not None:
        set_flag(lane_flags, "--max-context", str(args.max_context))
    if args.no_lm_head_draft:
        # A boolean lane flag is removed rather than substituted. The bench corpus cannot decide this
        # one from either end of its range (it saturates acceptance), so the comparison has to run on
        # real text, and the lane's own proposal-head setting has to be able to change without
        # changing anything else.
        if "--lm-head-draft" in lane_flags:
            lane_flags.remove("--lm-head-draft")
    set_flag(lane_flags, "--chat-template", str(args.template))
    flags = lane_flags
    if result_path.is_file():
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        # The flags are the experiment, so a cached arm answers only for the flags it ran under: a
        # directory reused across configurations would otherwise report the previous one's result.
        if cached.get("flags") == flags:
            return cached
        print(f"        cached {result_path.parent.name} ignored: its flags differ", flush=True)
    model_id = None
    with log_path.open("w", encoding="utf-8", newline="\n") as handle:
        # The lane's own environment, exactly as its launcher sets it: NINFER_CUDA_SYNC is part of the
        # configuration being measured, and a harness that omits it measures a lane nobody ships.
        process = subprocess.Popen([str(SERVE), str(artifact), *flags], cwd=str(REPO), stdout=handle,
                                   stderr=subprocess.STDOUT, env=launcher_environment(profile))
        try:
            if not wait_for_port(port, time.time() + args.startup_timeout):
                raise SystemExit(f"{artifact.name}: serve never listened on {port}")
            time.sleep(2.0)
            model_id = (get_json(port, "/v1/models").get("data") or [{}])[0].get("id")
            for index, block in enumerate(blocks, 1):
                answer = post_json(port, {"model": model_id,
                                          "messages": [{"role": "user", "content": block}],
                                          "max_tokens": args.max_tokens,
                                          "temperature": args.temperature}, args.request_timeout)
                print(f"        prompt {index}: http {answer['status']}", flush=True)
        finally:
            time.sleep(2.0)
            process.terminate()      # the SERVE is stopped; nothing else is killed
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                pass

    body = log_path.read_text(encoding="utf-8", errors="replace")
    records = [{"accepted": int(a.replace(",", "")), "drafted": int(d.replace(",", "")),
                "percent": float(p)} for a, d, p in ACCEPT.findall(body)]
    result = {"artifact": artifact.name,
              "flags": flags,
              "model_id": model_id,
              "requests": len(records),
              "accepted": sum(record["accepted"] for record in records),
              "drafted": sum(record["drafted"] for record in records),
              "acceptance": (sum(record["accepted"] for record in records)
                             / max(1, sum(record["drafted"] for record in records))),
              "decode": [float(value) for value in DECODE.findall(body)]}
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact", type=Path, action="append", required=True,
                        help="repeatable: one arm per artifact, labelled by its file stem")
    parser.add_argument("--lane", default="nvidia_v3_dflash2_vision",
                        help="substring of the launcher profile file to run; the lane's own flags are "
                             "composed through profiles.launcher_args, so a record describes a "
                             "configuration that ships")
    parser.add_argument("--label", action="append", default=[],
                        help="repeatable, positionally matched to --artifact, to name the arms")
    parser.add_argument("--text", type=Path, help="real-text source; defaults to the perplexity corpus")
    parser.add_argument("--prompts", type=int, default=3, help="fixed prompts to pool over")
    parser.add_argument("--prompt-chars", type=int, default=12000)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--draft-tokens", type=int,
                        help="override the lane's own draft depth, for a depth A/B")
    parser.add_argument("--kv-capacity",
                        help="override the lane's --kv-capacity auto, so a probe can ask whether a "
                             "per-request cost tracks the capacity the lane reserves")
    parser.add_argument("--max-context", type=int,
                        help="override the lane's --max-context, which a sub-context capacity requires")
    parser.add_argument("--no-lm-head-draft", action="store_true",
                        help="drop the lane's --lm-head-draft, so the optimized proposal head can be "
                             "compared with the full head on real text; ADR-0005 asks for a per-profile "
                             "record and the bench corpus cannot supply it")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 makes each arm's generation identical, which is what makes arms comparable")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--port-base", type=int, default=18120)
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    profile = next((candidate for candidate in PROFILES if args.lane in candidate["file"]), None)
    if profile is None:
        raise SystemExit(f"no launcher profile matches {args.lane!r}")

    if args.text:
        text = args.text.read_text(encoding="utf-8", errors="replace")
        source = args.text
    else:
        source, text = default_text()
    blocks = [text[i * args.prompt_chars:(i + 1) * args.prompt_chars] for i in range(args.prompts)]
    if any(len(block) < 100 for block in blocks):
        raise SystemExit("not enough real text for that many prompts")
    labels = [args.label[index] if index < len(args.label) else artifact.stem
              for index, artifact in enumerate(args.artifact)]
    cjk = sum(1 for char in text if "\u4e00" <= char <= "\u9fff") / max(1, len(text))
    print(f"    lane: {profile['file']}")
    print(f"    text: {source} ({len(text):,} chars, cjk {cjk:.2%})")
    print(f"    {len(args.artifact)} arm(s) x {args.rounds} round(s) x {args.prompts} prompt(s), "
          f"temperature {args.temperature}, max_tokens {args.max_tokens}")

    results: dict[str, list[dict]] = {label: [] for label in labels}
    for round_index in range(1, args.rounds + 1):
        for index, (label, artifact) in enumerate(zip(labels, args.artifact)):
            print(f"\n=== {label} round {round_index}: {artifact.name}", flush=True)
            results[label].append(run_arm(artifact, args.port_base + index, blocks, args,
                                          args.output_dir / f"{label}-r{round_index}", profile))

    print()
    print(f"    {'arm':<22} {'round':>5} {'accepted':>9} {'drafted':>8} {'accept %':>9} {'decode':>8}")
    for label in labels:
        for round_index, record in enumerate(results[label], 1):
            decode = sum(record["decode"]) / len(record["decode"]) if record["decode"] else 0.0
            print(f"    {label[:22]:<22} {round_index:>5} {record['accepted']:>9} {record['drafted']:>8} "
                  f"{record['acceptance'] * 100:>8.1f}% {decode:>8.1f}")
    if len(labels) == 2:
        first = sum(r["acceptance"] for r in results[labels[0]]) / len(results[labels[0]])
        second = sum(r["acceptance"] for r in results[labels[1]]) / len(results[labels[1]])
        print()
        print(f"    pooled mean: {labels[0]} {first * 100:.2f}%, {labels[1]} {second * 100:.2f}% "
              f"-> {second / first - 1:+.1%} for {labels[1]}")
    print()
    print("    sanity: near 0.99 is a too-predictable prompt and near 0.10 too little context; real "
          "domains read 0.44-0.55")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

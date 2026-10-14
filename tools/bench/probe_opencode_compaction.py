"""Drive a real opencode agent loop until its compaction fires, and read the engine's own record.

docs/opencode-settings.md carried this gap: driving the opencode agent loop "is outside what I can
test from here", so the compaction trigger and the post-compaction re-prefill stayed estimated. That
was wrong about the tool, not just the effort. `opencode run` drives a real agent loop
non-interactively and chains turns with --continue, and OPENCODE_TEST_HOME / OPENCODE_CONFIG_CONTENT
sandbox it completely: a throwaway home, an inline config and a redirected database, so the operator's
own sessions and configuration are never read or written.

Compaction needs no special handling to be observed. It rewrites the conversation prefix, so in the
engine's request log it is a discontinuity in result.prompt_tokens: context climbs turn by turn, and
the turn where it drops is the turn where opencode compacted. The probe does not time a stopwatch or
parse opencode's internals -- it counts prompt_tokens, which the engine reports and
check_request_logs.py validates.

Context is inflated with large user messages rather than by making the agent read files. The trigger
is size-based, so this reaches the configured threshold on a predictable number of turns and needs no
tool permissions, which keeps the measurement about compaction rather than about whether the agent
chose to call a tool.

Arms, both against a lane on their own port:
  compact    -- opencode's compaction configured as the shipped config has it.
  nocompact  -- the same run with OPENCODE_DISABLE_AUTOCOMPACT set, as the control. Without it, an
               arm that never compacts could be a run that simply never reached the threshold, and
               the two are indistinguishable from one arm alone.

One lane at a time: this card holds a single model.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.request

SERVE = r"C:\AI\ninfer-v3-windows\build\apps\ninfer-serve.exe"
MODEL = r"C:\AI\models\qwen3_8_27b_nvfp4qat.v3.ninfer"
TEMPLATE = r"C:\AI\ninfer-v3-windows\tools\chat_templates\qwen3_8.jinja"
MODEL_ID = "qwen3.8-27b-quasar-v3-dflash2-vision"


def opencode_exe() -> str:
    """Resolve the opencode CLI that matches the configuration under test.

    This matters more than it looks. There are two separate installs on this machine: the npm
    package `opencode-ai`, and the OpenCode Desktop application, which ships its own CLI at
    resources/opencode-cli.exe. They are different builds with different config schemas -- the npm
    1.18.4 CLI rejects the Desktop config outright with "Unrecognized keys: media, snapshots,
    providers, agents, permissions", because those keys are Desktop's and not its own. So the CLI
    has to be the one belonging to whichever client owns the config being measured.

    The Desktop bundled CLI is preferred, because the shipped configuration is Desktop's. Override
    with NINFER_OPENCODE_EXE to measure a different install deliberately.
    """
    override = os.environ.get("NINFER_OPENCODE_EXE")
    if override and pathlib.Path(override).exists():
        return override
    desktop = pathlib.Path(os.environ.get("LOCALAPPDATA", "")) / (
        "Programs/@opencode-aidesktop/resources/opencode-cli.exe")
    if desktop.exists():
        return str(desktop)
    npm = pathlib.Path(os.environ.get("APPDATA", "")) / (
        "npm/node_modules/opencode-ai/node_modules/opencode-windows-x64/bin/opencode.exe")
    if npm.exists():
        return str(npm)
    for name in ("opencode.exe", "opencode.cmd", "opencode"):
        found = shutil.which(name)
        if found:
            return found
    raise RuntimeError(
        "opencode CLI not found; set NINFER_OPENCODE_EXE to the executable that owns the "
        "configuration under test")

# The lane's own flags, read off start_quasar_v3_dflash2_vision.bat.
LANE_FLAGS = [
    "--vision", "--spec", "dflash2", "--draft-tokens", "7", "--lm-head-draft",
    "--kv-dtype", "k8v4", "--host", "127.0.0.1",
    "--model-id", MODEL_ID, "--max-context", "262144", "--device-state-slots", "1",
    "--kv-capacity", "auto", "--prefill-chunk", "8192", "--max-concurrency", "1",
    "--host-context-mib", "8192", "--preserve-thinking",
    "--default-thinking-budget", "4096", "--pending-timeout-ms", "600000",
    "--chat-template", TEMPLATE,
]

# The shipped opencode compaction settings, so the trigger under test is the configured one.
COMPACTION = {"auto": True, "keep": {"tokens": 20000}, "buffer": 8000}

# Filler sized to advance the context by a known amount per turn. The measured prompt_tokens in the
# result is what every table reports, so this only has to get the order of magnitude right.
FILLER_PARA = (
    "The maintainer notes record that this lane's prefill is chunked at 8192 tokens, that its "
    "context cache registers each rendered prefix as an origin so a preserved prefix is reused "
    "rather than re-prefilled, and that the KV payload for a 262,144-token conversation at k8v4 "
    "occupies a fixed share of the card that the launcher's own header records. "
)


def http(base: str, path: str, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(f"{base}{path}", timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_healthy(base: str, process: subprocess.Popen, timeout: float = 900.0) -> None:
    deadline = time.monotonic() + timeout
    last: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"ninfer-serve exited during startup, code {process.returncode}")
        try:
            if http(base, "/health") == {"status": "ok"}:
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


def prompt_totals(log: pathlib.Path) -> list[int]:
    """Every prompt_tokens the lane has recorded so far, in order. Compaction is a drop in this
    sequence; nothing else in the log has to be interpreted."""
    if not log.exists():
        return []
    out: list[int] = []
    for line in log.read_bytes().decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue          # a line still being appended is not a record
        if record.get("event") == "request_done":
            tokens = record.get("result", {}).get("prompt_tokens")
            if isinstance(tokens, int):
                out.append(tokens)
    return out


def run_opencode(home: pathlib.Path, port: int, message: str,
                  disable_autocompact: bool) -> tuple[int, str]:
    """One turn of a real agent loop. Returns the exit code and output.

    Two environment facts were measured here rather than assumed, and both changed the design:

    * The message goes on stdin, not the command line. A turn's filler is tens of thousands of
      characters and Windows caps a command line at 32,767, so passing it as an argument fails with
      WinError 206 before opencode starts. `opencode run` reads stdin when no message is given.
    * opencode does not honour OPENCODE_TEST_HOME or OPENCODE_CONFIG_DIR for config loading --
      both were tested, and it loads ~/.config/opencode/opencode.json either way. So the probe runs
      against the operator's own configured model, which is the subject under test rather than a
      deviation from it. OPENCODE_DB does redirect the session store, so the turns land in a
      throwaway database rather than in the operator's session history.
    """
    env = dict(os.environ)
    env["OPENCODE_DB"] = str(home / "opencode.db")
    env["OPENCODE_DISABLE_AUTOUPDATE"] = "1"
    env["OPENCODE_DISABLE_MODELS_FETCH"] = "1"
    env["OPENCODE_DISABLE_LSP_DOWNLOAD"] = "1"
    env.pop("OPENCODE_CONFIG_CONTENT", None)
    if disable_autocompact:
        env["OPENCODE_DISABLE_AUTOCOMPACT"] = "1"
    else:
        env.pop("OPENCODE_DISABLE_AUTOCOMPACT", None)
    command = [opencode_exe(), "run", "--print-logs"]
    # check=False deliberately: a turn that fails is data here, not an error to raise on. The exit
    # code is recorded per turn and printed, because an arm that silently stopped would otherwise
    # read as an arm that never reached the trigger.
    done = subprocess.run(command, env=env, input=message, capture_output=True, text=True,
                          check=False, encoding="utf-8", errors="replace", timeout=1800)
    return done.returncode, (done.stdout or "") + (done.stderr or "")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=16)
    parser.add_argument("--fill-repeats", type=int, default=260,
                        help="filler paragraphs per turn; sets how fast the trigger is reached")
    parser.add_argument("--port", type=int, default=8086,
                        help="must be the port opencode's own config points at: the config redirect "
                             "variables do not work, so this lane has to be the one it will call")
    parser.add_argument("--arms", nargs="+", default=["compact", "nocompact"])
    parser.add_argument("--out", default=r"C:\AI\bench\opencode_compaction.json")
    args = parser.parse_args()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    home = out.with_name(out.stem + ".home")
    home.mkdir(parents=True, exist_ok=True)

    message = ("Acknowledge this and continue.\n\n"
               + (FILLER_PARA * args.fill_repeats))
    print(f"turn message: {len(message):,} characters", flush=True)

    results: dict[str, object] = {}
    for offset, arm in enumerate(args.arms):
        port = args.port + offset
        log = out.with_name(out.stem + f".{arm}.requestlog.jsonl")
        if log.exists():
            log.unlink()
        print(f"\n=== arm {arm}: {args.turns} turns, lane port {port} ===", flush=True)
        process, base = start_lane(port, log)
        series: list[dict] = []
        try:
            wait_healthy(base, process)
            for turn in range(1, args.turns + 1):
                before = len(prompt_totals(log))
                code, output = run_opencode(home, port, message,
                                            disable_autocompact=(arm == "nocompact"))
                totals = prompt_totals(log)
                grew = totals[before:] or totals[-1:]
                series.append({
                    "turn": turn,
                    "exit_code": code,
                    "requests": len(totals) - before,
                    "prompt_tokens": grew,
                })
                print(f"  turn {turn:>2}: exit {code}  requests {len(totals) - before}"
                      f"  prompt_tokens {grew[-4:]}", flush=True)
                if code != 0:
                    # Never print opencode's output raw: it carries arrows and box characters that a
                    # cp1252 console cannot encode, and the resulting UnicodeEncodeError would kill
                    # the probe on the first one -- hiding the failure it was written to show (rule 48).
                    tail = output[-2000:].encode("ascii", "replace").decode("ascii")
                    print(f"  turn {turn} opencode exited {code}; tail written to {log.name}.opencode.txt",
                          flush=True)
                    (log.parent / f"{log.name}.opencode.txt").write_text(
                        tail, encoding="utf-8", errors="replace")
                    print("  " + tail[-400:].replace("\n", "\n  "), flush=True)
                    break
        finally:
            stop_lane(process)
        results[arm] = {"prompt_tokens": prompt_totals(log), "turns": series}

    # Compaction is a drop: a request whose prompt is materially smaller than the one before it.
    # Report the largest such drop per arm rather than asserting that one occurred.
    for arm in args.arms:
        totals = results[arm]["prompt_tokens"]
        drops = [(i, totals[i - 1] - totals[i]) for i in range(1, len(totals))
                 if totals[i] < totals[i - 1]]
        results[arm]["drops"] = [{"index": i, "drop": d} for i, d in drops]
        results[arm]["largest_drop"] = max(drops, key=lambda p: p[1]) if drops else None

    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== compaction, as a drop in prompt_tokens ===")
    for arm in args.arms:
        totals = results[arm]["prompt_tokens"]
        largest = results[arm]["largest_drop"]
        print(f"\n  arm {arm}: {len(totals)} requests, peak {max(totals) if totals else 0:,} tokens")
        if largest:
            index, drop = largest
            print(f"    largest drop: {drop:,} tokens, at request {index + 1}"
                  f" ({totals[index - 1]:,} -> {totals[index]:,})")
        else:
            print("    no drop: the trigger was never reached, so this arm says nothing")
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Drive a real opencode agent loop until its compaction fires, and read the engine's own record.

docs/opencode-settings.md said driving the opencode agent loop "is outside what I can test from
here", so the compaction trigger and the post-compaction re-prefill stayed estimated. That was wrong
about the tool. OpenCode ships a CLI that runs an agent loop non-interactively, and compaction needs
no special handling to be observed. It rewrites the conversation prefix, so in the engine's request
log it is a discontinuity in result.prompt_tokens: context climbs turn by turn, and the turn where
it drops is the turn where opencode compacted. Nothing about opencode's internals is parsed here --
the engine's prompt_tokens count is the whole instrument.

Context is inflated with large user messages rather than by making the agent read files. The trigger
is size-based, so this reaches the configured threshold on a predictable number of turns and needs
no tool permissions, which keeps the measurement about compaction rather than about whether the
agent chose to call a tool. It is also why the session shape is unusual, which is a limit on how far
the numbers generalise -- recorded on the page rather than left implicit.

Four environment facts were measured here rather than assumed, and each changed the design. They are
in run_opencode's docstring and repeated here because they are the difference between this probe
working and not:

  * A turn's filler exceeds Windows' 32,767-character command-line limit, so the message goes on
    stdin. `opencode run` reads stdin when no message argument is given.
  * opencode honours neither OPENCODE_TEST_HOME nor OPENCODE_CONFIG_DIR for config loading; it
    reads the user's own config either way. So the probe runs against the operator's configured
    model, which is the subject under test rather than a deviation from it. OPENCODE_DB does
    redirect the session store, so turns land in a throwaway database.
  * There are two installs with different config schemas. The npm package opencode-ai rejects the
    Desktop configuration outright -- "Unrecognized keys: media, snapshots, providers, agents,
    permissions" -- because those are Desktop's keys. opencode_exe() prefers Desktop's bundled CLI.
  * On Windows `opencode` on PATH is a .cmd/.ps1 shim that CreateProcess cannot exec, so the bare
    name fails with WinError 2. The executable has to be resolved.

Arms, both against a lane on the port opencode's own config calls:
  compact    -- compaction configured as the shipped config has it.
  nocompact  -- the same run with OPENCODE_DISABLE_AUTOCOMPACT, as the control. Without it, an arm
               that never compacts could be a run that simply never reached the threshold, and the
               two are indistinguishable from one arm alone.

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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import lane_process as lane  # noqa: E402

# Filler sized to advance the context by a known amount per turn. The measured prompt_tokens in the
# result is what every table reports, so this only has to get the order of magnitude right.
FILLER_PARA = (
    "The maintainer notes record that this lane's prefill is chunked at 8192 tokens, that its "
    "context cache registers each rendered prefix as an origin so a preserved prefix is reused "
    "rather than re-prefilled, and that the KV payload for a 262,144-token conversation at k8v4 "
    "occupies a fixed share of the card that the launcher's own header records. "
)


def opencode_exe() -> str:
    """Resolve the opencode CLI that matches the configuration under test.

    This matters more than it looks. There are two separate installs on a typical machine: the npm
    package `opencode-ai`, and the OpenCode Desktop application, which ships its own CLI at
    resources/opencode-cli.exe. They are different builds with different config schemas, so the CLI
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


def run_opencode(home: pathlib.Path, message: str,
                 disable_autocompact: bool) -> tuple[int, str]:
    """One turn of a real agent loop. Returns the exit code and output.

    The message goes on stdin rather than the command line: a turn's filler is tens of thousands of
    characters and Windows caps a command line at 32,767, so passing it as an argument fails with
    WinError 206 before opencode starts.
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
    # check=False deliberately: a turn that fails is data here, not an error to raise on. The exit
    # code is recorded per turn, because an arm that silently stopped would otherwise read as an
    # arm that never reached the trigger.
    done = subprocess.run([opencode_exe(), "run", "--print-logs"], env=env, input=message,
                          capture_output=True, text=True, check=False,
                          encoding="utf-8", errors="replace", timeout=1800)
    return done.returncode, (done.stdout or "") + (done.stderr or "")


def prompt_totals(log: pathlib.Path) -> list[int]:
    return [t for t in (lane.dig(r, "result.prompt_tokens") for r in lane.request_done(log))
            if isinstance(t, int)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=16)
    parser.add_argument("--fill-repeats", type=int, default=260,
                        help="filler paragraphs per turn; sets how fast the trigger is reached")
    parser.add_argument("--lane", default=lane.QUASAR_DFLASH2, help="launcher file name")
    parser.add_argument("--serve", default=None, help="ninfer-serve executable")
    parser.add_argument("--artifact", default=None, help="model artifact")
    parser.add_argument("--arms", nargs="+", default=["compact", "nocompact"])
    parser.add_argument("--out", default="opencode_compaction.json",
                        help="written as given; a bare name keeps it beside the probe")
    args = parser.parse_args()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    home = out.with_name(out.stem + ".home")
    home.mkdir(parents=True, exist_ok=True)
    profile = lane.lane_profile(args.lane)
    port = int(profile["port"])

    message = "Acknowledge this and continue.\n\n" + (FILLER_PARA * args.fill_repeats)
    print(f"turn message: {len(message):,} characters", flush=True)

    results: dict[str, dict] = {}
    for offset, arm in enumerate(args.arms):
        log = out.with_name(out.stem + f".{arm}.requestlog.jsonl")
        if log.exists():
            log.unlink()
        print(f"\n=== arm {arm}: {args.turns} turns, lane port {port + offset} ===", flush=True)
        process, base = lane.start_lane(profile, port + offset, log, args.serve, args.artifact)
        series: list[dict] = []
        try:
            lane.wait_healthy(base, process)
            for turn in range(1, args.turns + 1):
                before = len(prompt_totals(log))
                code, output = run_opencode(home, message,
                                            disable_autocompact=(arm == "nocompact"))
                totals = prompt_totals(log)
                grew = totals[before:] or totals[-1:]
                series.append({"turn": turn, "exit_code": code,
                               "requests": len(totals) - before, "prompt_tokens": grew})
                print(f"  turn {turn:>2}: exit {code}  requests {len(totals) - before}"
                      f"  prompt_tokens {grew[-4:]}", flush=True)
                if code != 0:
                    # Never print opencode's output raw: it carries arrows and box characters that a
                    # cp1252 console cannot encode, and the UnicodeEncodeError would kill the probe
                    # on the first one, hiding the failure it was written to show.
                    tail = output[-2000:].encode("ascii", "replace").decode("ascii")
                    (log.parent / f"{log.name}.opencode.txt").write_text(
                        tail, encoding="utf-8", errors="replace")
                    print(f"  turn {turn} exited {code}; tail written to "
                          f"{(log.parent / (log.name + '.opencode.txt')).name}", flush=True)
                    print("  " + tail[-400:].replace("\n", "\n  "), flush=True)
                    break
        finally:
            lane.stop_lane(process)
        results[arm] = {"prompt_tokens": prompt_totals(log), "turns": series}

    # Compaction is a drop: a request whose prompt is materially smaller than the one before it.
    # Report the largest such drop per arm rather than asserting that one occurred.
    for arm in args.arms:
        totals = results[arm]["prompt_tokens"]
        drops = [(i, totals[i - 1] - totals[i]) for i in range(1, len(totals))
                 if totals[i] < totals[i - 1]]
        results[arm]["drops"] = [{"index": i, "drop": d} for i, d in drops]
        results[arm]["largest_drop"] = ({"index": drops[-1][0], "drop": max(drops, key=lambda p: p[1])[1]}
                                        if drops else None) if drops else None

    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== compaction, as a drop in prompt_tokens ===")
    for arm in args.arms:
        totals = results[arm]["prompt_tokens"]
        largest = results[arm]["largest_drop"]
        print(f"\n  arm {arm}: {len(totals)} requests, "
              f"peak {max(totals) if totals else 0:,} tokens")
        if largest:
            index = largest["index"]
            print(f"    largest drop: {largest['drop']:,} tokens, at request {index + 1}"
                  f" ({totals[index - 1]:,} -> {totals[index]:,})")
        else:
            print("    no drop: the trigger was never reached, so this arm says nothing")
    print(f"\nwritten: {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

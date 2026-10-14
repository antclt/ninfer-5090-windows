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
  * opencode honours OPENCODE_CONFIG_DIR, OPENCODE_CONFIG, OPENCODE_CONFIG_CONTENT, OPENCODE_DB and
    OPENCODE_TEST_HOME in v2, and the probe uses them: a throwaway config directory, a throwaway
    database and `--standalone`, so the run neither reads the operator's configuration nor writes to
    the operator's session store. **This only works because the CLI is run `--standalone`.** An
    earlier revision of this probe concluded that none of those variables redirected anything --
    which was wrong, and the cause was that it omitted `--standalone`, so every invocation attached
    to the operator's already-running background service, which captured its environment when *it*
    started. The variables were being set on a process that merely forwarded to someone else's
    server. With `--standalone`, OPENCODE_DB is honoured and a throwaway database appears where the
    probe asked for one.
  * `--standalone` is also what makes server logs available; `--print-logs` says so in its own help.
  * `OPENCODE_DISABLE_AUTOCOMPACT` genuinely does nothing -- it is a V1 documentation-only leftover
    with no implementation in the v2 source. The control arm uses a project-level opencode.json with
    `"compaction": {"auto": false}` instead, which disables both the automatic and the overflow path
    while leaving manual compaction available.
  * There are two installs with different config schemas. The npm package opencode-ai rejects the
    Desktop configuration outright -- "Unrecognized keys: media, snapshots, providers, agents,
    permissions" -- because those are Desktop's keys. opencode_exe() prefers Desktop's bundled CLI.
  * On Windows `opencode` on PATH is a .cmd/.ps1 shim that CreateProcess cannot exec, so the bare
    name fails with WinError 2. The executable has to be resolved.

Arms, both against a lane on the port the throwaway config calls:
  compact    -- compaction as configured, which is the measurement.
  nocompact  -- a project-level opencode.json with `"compaction": {"auto": false}`. This disables
               both the automatic and the overflow path in v2 (the two share one flag in source),
               while leaving manual compaction available. It is the control: measured, the session
               grows to 192,815 over 14 turns and never drops, against 214,962 -> 47,666 with
               compaction on. OPENCODE_DISABLE_AUTOCOMPACT is not used for this because it does
               nothing -- v1 documentation with no implementation behind it.

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


def run_opencode(home: pathlib.Path, message: str, session_id: str,
                 disable_autocompact: bool, cwd: str | os.PathLike[str] | None = None,
                 project_config: dict | None = None) -> tuple[int, str]:
    """One turn of a real agent loop. Returns the exit code and output.

    The message goes on stdin rather than the command line: a turn's filler is tens of thousands of
    characters and Windows caps a command line at 32,767, so passing it as an argument fails with
    WinError 206 before opencode starts.

    `session_id` is passed on every turn and is what makes this a compaction measurement at all.
    `opencode run` without --continue/--session starts a NEW session per invocation, so N turns were
    N independent conversations and the conversation never grew. An earlier version of this probe
    ran that way and reported 13 compactions that were 13 session boundaries: prompt_tokens reset to
    the size of one filler message every time, which is a new session, not a compaction. --session
    creates the session if it does not exist and continues it otherwise, so turns chain into one
    growing conversation. It also names the session, so the run cannot attach to the operator's
    most recent session by accident.

    `--standalone` gives the run a private server rather than the operator's background service, which
    is what stops it sharing server state with a live desktop session.

    `project_config`, when given, is written as an opencode.json in `cwd` and the CLI is run from
    there. That is the only route found to a control arm: none of OPENCODE_CONFIG,
    OPENCODE_CONFIG_DIR, OPENCODE_CONFIG_CONTENT or OPENCODE_TEST_HOME redirects the config file in
    this build, so a project-level file is the remaining candidate for supplying
    `"compaction": {"auto": false}` without editing the operator's own configuration.
    """
    env = dict(os.environ)
    # A throwaway config directory and database, so the run reads and writes nothing of the
    # operator's. Both are honoured in v2, and only because the CLI runs --standalone below; without
    # it these are set on a process that forwards to the operator's running service and has no effect.
    env["OPENCODE_CONFIG_DIR"] = str(home / "config")
    env["OPENCODE_DB"] = str(home / "opencode.db")
    env["OPENCODE_DISABLE_AUTOUPDATE"] = "1"
    env["OPENCODE_DISABLE_MODELS_FETCH"] = "1"
    env["OPENCODE_DISABLE_LSP_DOWNLOAD"] = "1"
    if cwd is not None:
        # The throwaway config dir must exist and carry the same providers/limits as the operator's,
        # or the run has no model to call. It is copied from the operator's config on first use.
        cfg = pathlib.Path(env["OPENCODE_CONFIG_DIR"])
        cfg.mkdir(parents=True, exist_ok=True)
        if not (cfg / "opencode.json").exists():
            source = pathlib.Path(os.environ.get("USERPROFILE", "")) / ".config/opencode/opencode.json"
            if source.exists():
                (cfg / "opencode.json").write_bytes(source.read_bytes())
    if cwd is not None:
        cwd = pathlib.Path(cwd)
        cwd.mkdir(parents=True, exist_ok=True)
        if project_config is not None:
            (cwd / "opencode.json").write_text(
                json.dumps(project_config, indent=2), encoding="utf-8")
    # check=False deliberately: a turn that fails is data here, not an error to raise on. The exit
    # code is recorded per turn, because an arm that silently stopped would otherwise read as an
    # arm that never reached the trigger.
    done = subprocess.run([opencode_exe(), "run", "--print-logs", "--standalone",
                           "--session", session_id],
                          env=env, input=message, capture_output=True, text=True, check=False,
                          cwd=str(cwd) if cwd else None,
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
    parser.add_argument("--session-prefix", default="ses_ninfer_probe_compaction_",
                        help="each arm's turns chain into one session under this id prefix; "
                             "the id must start with 'ses' or the CLI rejects it")
    parser.add_argument("--opencode-cwd", default=None,
                        help="directory to run the CLI from; the nocompact arm also writes a "
                             "project-level opencode.json here, which is the only route found "
                             "to supplying compaction.auto=false without editing the operator's "
                             "own configuration")
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
    for arm in args.arms:
        log = out.with_name(out.stem + f".{arm}.requestlog.jsonl")
        if log.exists():
            log.unlink()
        # The lane always runs on the lane's own port. Offsetting it per arm looked tidy and was a
        # bug: opencode calls the port its configuration names, so a second arm on 8087 found
        # nothing listening and recorded zero requests.
        print(f"\n=== arm {arm}: {args.turns} turns, lane port {port} ===", flush=True)
        process, base = lane.start_lane(profile, port, log, args.serve, args.artifact)
        series: list[dict] = []
        try:
            lane.wait_healthy(base, process)
            for turn in range(1, args.turns + 1):
                before = len(prompt_totals(log))
                code, output = run_opencode(
                    home, message, f"{args.session_prefix}{arm}",
                    disable_autocompact=(arm == "nocompact"),
                    cwd=args.opencode_cwd or None,
                    project_config=(
                        {"compaction": {"auto": False}} if arm == "nocompact" else None))
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

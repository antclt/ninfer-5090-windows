"""Launch a long run detached, so the hourly location eviction cannot take it.

This session lost three sweeps to the same silent death: the service evicts and re-boots a location about
hourly, every shell that location hosts dies with it, and a shell's finalizer tears down its own process
tree - no flush, no traceback, no completion notice. The session's own research
(the silent-shell-death investigation) established the escape and this is it as a tool:
`DETACHED_PROCESS | CREATE_BREAKAWAY_FROM_JOB`, launched by a process that then exits, so the child is in
no shell's tree and survives.

The completion notice is lost either way - a detached tree reports to nobody - so the only reliable report
is the per-step log file, which this writes and prints the path of. Read the work out of its own log, which
is also how the six-round sweep's four completed comparisons were recovered after its shell died.

    python tools/scripts/run_detached.py --log %TEMP%\\opencode\\sweep.log -- \
        C:/vllm-env/Scripts/python.exe tools/bench/compare_lane_configs.py --lane ... --rounds 6
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DETACHED_PROCESS = 0x00000008
CREATE_BREAKAWAY_FROM_JOB = 0x01000000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log", type=Path, required=True, help="the per-step log; the only report")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="the command to run, after a `--`")
    args = parser.parse_args()

    command = [c for c in args.command if c != "--"]
    if not command:
        print("  nothing to run: pass the command after --")
        return 1
    if sys.platform != "win32":
        # the hazard is a Windows service behaviour; elsewhere a plain detach is enough
        flags = 0
    else:
        flags = DETACHED_PROCESS | CREATE_BREAKAWAY_FROM_JOB

    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"detached run: {' '.join(command)}\n")
        handle.flush()
        try:
            process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                       creationflags=flags, close_fds=True)
        except OSError as exc:
            print(f"  could not start: {exc}")
            return 1
    print(f"  started pid {process.pid}, detached (flags 0x{flags:08x})")
    print(f"  log: {args.log}")
    print("  it will survive the location eviction; it will not report completion, so read the log.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

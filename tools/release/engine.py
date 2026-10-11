#!/usr/bin/env python3
"""Shared engine control for the measurement harnesses.

A harness that measures a serving lane needs the same four things, and each of them was written out
by hand somewhere: the lane's own command line (one bench tool, imported by two more), the
environment it starts under (five harnesses, missing from the rest), a spawn that puts it in its own
console process group (two of ten), and a stop that lets the engine write its shutdown record (four
shapes, one of which killed the lane outright so the record never existed).

This is the module for all four, and the interface returns values rather than leaking state:

    lane_command(profile, exe, artifact)     -- [exe, artifact, *launcher_args(profile)]
    start_lane(profile, exe, artifact, ...)  -- start it the way its launcher does
    start_engine(args, log_path, ...)        -- the same, from a raw argument list
    kill_servers()                           -- stop any running engine
    wait_ready(port, proc, timeout)          -- True once /v1/models answers, False if it died
    stop_engine(proc, handle)                -- ask it to stop, then insist

Three properties are load-bearing rather than stylistic:

  - The environment is `profiles.launcher_environment`, not this process's own. `Popen` replaces the
    environment wholesale, so a lane started without `NINFER_CUDA_SYNC=blocking` runs the wait
    schedule the engine defaults to and nobody ships.
  - Every spawn here sets `CREATE_NEW_PROCESS_GROUP`. That is what makes `stop_engine`'s CTRL_BREAK
    reach the lane alone; sent to a child sharing this process's group it arrives here as well.
  - `stop_engine` asks before it kills. `Popen.terminate()` is TerminateProcess on Windows, and the
    shutdown tail -- the throughput record carrying the context-cache selection counters -- is only
    written on a handled signal.

`wait_ready` takes the process so a refused profile, which exits in about four seconds, stops the
wait immediately instead of burning the whole timeout on a port that will never open. That
distinction mattered: an earlier ceiling probe spent minutes waiting on profiles that had already
exited.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiles import by_file, launcher_args, launcher_environment  # noqa: E402

# The lane the bench tools that measure shared-prefix and warm-TTFT behaviour start. Named once,
# here, because it used to be a prefix match inside a command builder in another module: the flags
# came from this lane and the artifact came from the caller's command line, so the two could
# describe different lanes with nothing saying so.
LANE_QUASAR_DFLASH2 = by_file("start_quasar_v3_dflash2_vision.bat")


def kill_servers() -> None:
    """Stop any running engine. One 32 GB card holds one artifact, so harnesses serialise."""
    subprocess.run(["taskkill", "/F", "/IM", "ninfer-serve.exe"], capture_output=True, text=True,
                   encoding="utf-8", check=False)


def wait_ready(port: int, proc: subprocess.Popen | None = None,
               timeout: int = 240) -> bool:
    """Poll /v1/models until the engine answers.

    Returns False as soon as the process has exited, because a refused profile exits in about
    four seconds and the port will never open.
    """
    end = time.time() + timeout
    while time.time() < end:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                r.read()
            return True
        except Exception:  # noqa: BLE001
            time.sleep(2)
    return False


def lane_command(profile: dict[str, Any], exe: Path, artifact: Path | str,
                 port: int | None = None, model_id: str | None = None,
                 extra_flags: list[str] | None = None) -> list[str]:
    """One lane's command line: the binary, the artifact positionally, then its own launcher flags.

    The artifact is positional because that is the usage line; `--artifact` would have been a guess
    from the other tools. What a harness adds on top -- a request log, a probe's own flag -- goes in
    `extra_flags`, after the shipped set, so the shipped set stays recognisable in the record.
    """
    return [str(exe), str(artifact),
            *launcher_args(profile, port=port, model_id=model_id),
            *(extra_flags or [])]


def spawn_engine(args: list[str], stdout, cwd: str | None = None,
                 env: dict[str, str] | None = None) -> subprocess.Popen:
    """Start the engine with a caller-owned output stream, and return the process.

    For a harness that already holds a handle -- or has a `with` block around one, which is the shape
    most of the probes have -- and only wants the two things that are easy to get wrong: the lane
    environment, because `Popen` replaces it wholesale, and the child's own console process group,
    which is what makes `stop_engine`'s CTRL_BREAK reach the lane alone.
    """
    return subprocess.Popen(  # noqa: S603
        args,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=subprocess.STDOUT,
        env={**launcher_environment(None), **(env or {})},
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )


def _spawn(args: list[str], log_path: Path, cwd: str | None,
           env: dict[str, str] | None) -> tuple[subprocess.Popen, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("wb")
    try:
        proc = spawn_engine(args, handle, cwd=cwd, env=env)
    except Exception:
        handle.close()
        raise
    return proc, handle


def start_lane(profile: dict[str, Any], exe: Path, artifact: Path | str, port: int | None,
               log_path: Path, extra_flags: list[str] | None = None,
               extra_env: dict[str, str] | None = None) -> tuple[subprocess.Popen, Any]:
    """Start a lane with its output going to `log_path`, and return (process, log handle).

    The caller owns the handle and should pass it to `stop_engine`, which closes it after the process
    has ended, so the log is complete before anything reads it.
    """
    return _spawn(
        lane_command(profile, exe, artifact, port=port, extra_flags=extra_flags),
        log_path,
        None,
        {**launcher_environment(profile), **(extra_env or {})},
    )


def start_engine(args: list[str], log_path: Path, cwd: str | None = None,
                 env: dict[str, str] | None = None) -> tuple[subprocess.Popen, Any]:
    """Start an engine from a raw argument list, for a harness that is not a table profile's lane.

    `env` defaults to the lane environment merged over this process's own, so a harness that only
    wanted a different command line does not silently lose `NINFER_CUDA_SYNC`.
    """
    return _spawn(args, log_path, cwd, {**launcher_environment(None), **(env or {})})


def stop_engine(proc: subprocess.Popen | None, handle=None) -> None:
    """Ask a lane to stop, then insist, and close its log.

    CTRL_BREAK_EVENT is the Windows console signal that reaches a handler registered by
    `SetConsoleCtrlHandler`, which is what `std::signal` compiles to here, so the engine runs its
    shutdown path and writes its tail record. `Popen.terminate()` is TerminateProcess and writes
    nothing. The event is delivered to a process *group*, which is why every spawn above sets
    CREATE_NEW_PROCESS_GROUP: sent to a child in this process's group it would arrive here too.

    A graceful stop that times out is killed anyway, so this cannot hang a harness.
    """
    graceful = False
    if proc is not None and os.name == "nt":
        try:
            os.kill(proc.pid, signal.CTRL_BREAK_EVENT)
            graceful = True
        except (OSError, ValueError):
            graceful = False
    if proc is not None:
        if not graceful:
            proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                pass
    if handle is not None:
        try:
            handle.close()
        except Exception:  # noqa: BLE001
            pass
    kill_servers()

"""Start a shipped lane for a bench probe, reading the lane from the shipped table.

Every probe in this directory that needs a running engine should get its command from here rather
than assembling flags of its own, for two reasons that have both bitten this tree:

  * **A copied flag list drifts.** `tools/release/profiles.py` is the single source of truth for
    every lane -- draft depth, KV format, context, vision, the speculative route. A probe that
    hardcodes them keeps measuring a configuration the tree no longer ships, and reports it as
    current. `ordered_flags()` returns the launcher's own flag list in shipped order, so a probe
    follows the table.
  * **An absolute path bakes one machine's checkout into a file that ships.** The launchers say
    this in terms: an absolute path there "is what made the generated launchers unverifiable
    anywhere else". Paths here are derived from this file's location, so a clone runs without
    editing, and `C:/AI/models` appears only as the same documented last-resort fallback the
    launchers already use for artifacts.

The template is resolved through `profiles.template_path()` and never left as the literal
`%TEMPLATE%`: `template_path()`'s own docstring records that passing that literal on a command
line is a startup failure, not a measurement.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# profiles.py is the lane table and is not a package on sys.path by default.
_TOOLS = REPO_ROOT / "tools" / "release"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import profiles  # noqa: E402  (path set up above)

# The artifact fallback the launchers already document: a source checkout may keep models outside
# the repository, and this is the path those launchers reach for.
ARTIFACT_FALLBACK = pathlib.Path("C:/AI/models")

# The lanes these probes were written against. Named as launcher file names because that is the
# key profiles.py publishes and the key by_file() accepts.
QUASAR_DFLASH2 = "start_quasar_v3_dflash2_vision.bat"


def lane_profile(lane: str = QUASAR_DFLASH2) -> dict[str, Any]:
    """The shipped profile for a lane, by launcher file name."""
    return profiles.by_file(lane)


def _first_existing(candidates: list[pathlib.Path | None], what: str) -> pathlib.Path:
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    tried = ", ".join(str(c) for c in candidates if c is not None)
    raise FileNotFoundError(f"could not find {what}; looked in: {tried}")


def resolve_serve(explicit: str | pathlib.Path | None = None) -> pathlib.Path:
    """The ninfer-serve executable, derived from this file's location."""
    return _first_existing([
        pathlib.Path(explicit) if explicit else None,
        REPO_ROOT / "build" / "apps" / "ninfer-serve.exe",
        REPO_ROOT / "build" / "apps" / "ninfer-serve",
        REPO_ROOT / "ninfer-serve.exe",
        REPO_ROOT / "ninfer-serve",
    ], "ninfer-serve")


def resolve_artifact(art: str, explicit: str | pathlib.Path | None = None) -> pathlib.Path:
    """A lane's artifact, in the launchers' own resolution order."""
    return _first_existing([
        pathlib.Path(explicit) if explicit else None,
        REPO_ROOT / "models" / art,
        REPO_ROOT / art,
        ARTIFACT_FALLBACK / art,
    ], f"artifact {art}")


def lane_flags(profile: dict[str, Any], port: int | None = None) -> list[str]:
    """The launcher's flags for a lane, with the template resolved and the port overridable.

    `port` exists because a probe must not fight a lane the operator already has running, and
    because opencode's own configuration decides which port its CLI will call.

    ordered_flags() yields pairs whose first element may already carry its value -- it is
    ("--spec dflash2", None), not ("--spec", "dflash2") -- because the launcher generator renders
    each entry as one continuation line. So each is split back into argv tokens here. Passing the
    joined string through produces a single argument the engine rejects with
    "unknown argument: --spec dflash2", which is a startup failure and not a measurement.
    """
    out: list[str] = []
    for rendered, value in profiles.ordered_flags(profile):
        parts = rendered.split()
        if not parts:
            continue
        if parts[0] == "--chat-template":
            out.extend(["--chat-template", profiles.template_path()])
            continue
        if parts[0] == "--port" and port is not None:
            out.extend(["--port", str(port)])
            continue
        out.extend(parts)
        if value is not None:
            out.append(value)
    return out


def lane_command(profile: dict[str, Any], port: int, request_log: pathlib.Path,
                 serve: pathlib.Path, artifact: pathlib.Path) -> list[str]:
    return [str(serve), str(artifact), *lane_flags(profile, port=port),
            "--request-log-jsonl", str(request_log)]


def http(base: str, path: str, payload: dict | None = None,
         timeout: float = 900.0) -> dict:
    url = f"{base}{path}"
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def chat(base: str, model_id: str, messages: list[dict], max_tokens: int,
         seed: int | None = None, reasoning_effort: str | None = None,
         timeout: float = 1800.0) -> dict:
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        payload["seed"] = seed
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    return http(base, "/v1/chat/completions", payload, timeout=timeout)


def start_lane(profile: dict[str, Any], port: int, request_log: pathlib.Path,
               serve: str | pathlib.Path | None = None,
               artifact: str | pathlib.Path | None = None) -> tuple[subprocess.Popen, str]:
    """Start the lane and return the process and its base URL. The caller must stop_lane()."""
    command = lane_command(
        profile, port, request_log,
        resolve_serve(serve),
        resolve_artifact(profile["art"], artifact),
    )
    env = dict(os.environ, NINFER_CUDA_SYNC="blocking")
    request_log.parent.mkdir(parents=True, exist_ok=True)
    handle = request_log.with_suffix(".server.log").open("w", encoding="utf-8", errors="replace")
    process = subprocess.Popen(
        command, stdout=handle, stderr=subprocess.STDOUT, env=env,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    return process, f"http://127.0.0.1:{port}"


def wait_healthy(base: str, process: subprocess.Popen, timeout: float = 900.0) -> None:
    deadline = time.monotonic() + timeout
    last: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"ninfer-serve exited during startup, code {process.returncode}")
        try:
            if http(base, "/health", timeout=5.0) == {"status": "ok"}:
                return
        except (urllib.error.URLError, OSError, ValueError) as error:
            # A startup poll: any failure means the lane is not answering yet.
            last = error
        time.sleep(0.5)
    raise RuntimeError(f"lane not healthy after {timeout:g}s: {last}")


def stop_lane(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def dig(obj: object, path: str) -> object:
    """Read a dotted path out of a request-log record."""
    current = obj
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def request_done(request_log: pathlib.Path) -> list[dict]:
    """Every completed request the lane recorded, in order.

    Read this only after the lane has stopped. The file is appended to while the lane runs, and a
    half-written line is not a record -- a partially-read log reads as a session that ended early.
    """
    out: list[dict] = []
    if not request_log.exists():
        return out
    for line in request_log.read_bytes().decode("utf-8", "replace").splitlines():
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

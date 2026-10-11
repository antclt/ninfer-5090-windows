#!/usr/bin/env python3
"""Does SO_EXCLUSIVEADDRUSE cost a listener restart? Measured, because a comment reasoned instead.

`src/serve/http_transport.cpp` clears SO_REUSEADDR so SO_EXCLUSIVEADDRUSE takes effect, and its comment
records a measurement taken with this probe: Microsoft's page for the option carries a restart caveat
("cannot always be reused immediately after socket closure"), and the measurement found the caveat did
not apply to this loopback bind. This file is that measurement's instrument, kept because a comment
citing a measurement should be reproducible from the tree rather than from a session's scratch
directory.

The case that matters is a *server-side* TIME_WAIT, so the client must not close first: the response is
read and the connection is left open long enough for the server's keep-alive timeout to expire it.
Without that, a successful rebind says only that nothing was in the way.

This probe is its own falsifier: it prints the port's state at the moment of the rebind, and says so
explicitly when no TIME_WAIT is present, because a rebind that succeeds with an empty port table is not
evidence about the caveat.

    python tools/bench/probe_listener_restart.py \
        --serve build/apps/ninfer-serve.exe --artifact out/qwen3_8_27b_nvfp4.ninfer

Exit 0 when the second listener bound and answered; 1 when it did not; 2 when the first listener never
came up (a harness failure, not a result).
"""
from __future__ import annotations

import argparse
import http.client
import socket
import subprocess
import sys
import time
from pathlib import Path


def wait_listening(port: int, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as probe:
            probe.settimeout(1.0)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(1.0)
    return False


def port_state(port: int) -> list[str]:
    """netstat's lines for this port, so a pass can be read against the state it passed in."""
    completed = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True,
                               encoding="utf-8", errors="replace", check=False)
    return [line.strip() for line in completed.stdout.splitlines() if f":{port}" in line]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--serve", required=True, help="ninfer-serve executable")
    parser.add_argument("--artifact", required=True, help=".ninfer artifact to load")
    parser.add_argument("--port", type=int, default=18299, help="port both listeners use")
    parser.add_argument("--keepalive-wait", type=float, default=10.0,
                        help="seconds to leave the first connection open, past the server's keep-alive")
    parser.add_argument("--log-dir", default=".", help="where the two lanes' logs are written")
    parser.add_argument("--timeout", type=float, default=180.0, help="seconds to wait for a listener")
    args = parser.parse_args()

    def start(tag: str) -> subprocess.Popen:
        handle = open(Path(args.log_dir) / f"restart-{tag}.log", "w", encoding="utf-8")
        return subprocess.Popen(
            [args.serve, args.artifact, "--host", "127.0.0.1", "--port", str(args.port),
             "--model-id", f"restart-{tag}"],
            stdout=handle, stderr=subprocess.STDOUT)

    first = start("first")
    if not wait_listening(args.port, args.timeout):
        print("the first lane never listened; see restart-first.log", flush=True)
        first.terminate()
        return 2
    print("first lane listening", flush=True)

    connection = http.client.HTTPConnection("127.0.0.1", args.port, timeout=30)
    connection.request("GET", "/v1/models")
    response = connection.getresponse()
    response.read()
    print(f"first lane answered http {response.status}; leaving the connection open", flush=True)
    time.sleep(args.keepalive_wait)

    first.terminate()
    first.wait(timeout=60)
    print("first lane stopped", flush=True)

    lines = port_state(args.port)
    print(f"port {args.port} at rebind time:", flush=True)
    for line in lines:
        print(f"    {line}", flush=True)
    if not any("TIME_WAIT" in line for line in lines):
        print("    NOTE: no TIME_WAIT on this port, so this run does not exercise the caveat",
              flush=True)

    started = time.time()
    second = start("second")
    bound = wait_listening(args.port, args.timeout)
    print(f"rebind after {time.time() - started:.1f}s: {'LISTENED' if bound else 'FAILED to listen'}",
          flush=True)
    if bound:
        check = http.client.HTTPConnection("127.0.0.1", args.port, timeout=30)
        check.request("GET", "/v1/models")
        answer = check.getresponse()
        print(f"second lane answered http {answer.status}", flush=True)
        answer.read()
        check.close()
    connection.close()
    second.terminate()
    try:
        second.wait(timeout=60)
    except subprocess.TimeoutExpired:
        second.kill()
    return 0 if bound else 1


if __name__ == "__main__":
    sys.exit(main())

"""The one reader of a serving lane's request log: its policy, checked in both directions.

Eleven tools read this file and their loops disagreed about what a bad line means. What is asserted here
is the policy the module chose and the reason it is the right one: a harness reads the log while the lane
writes it, so a torn final line must not be an error, and neither must a blank line. The other direction
matters as much -- a reader that skipped everything would pass every tolerant case -- so a well-formed
file must yield every record it holds, in order.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.ninfer_serve.request_log import events, records  # noqa: E402


def write_log(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def test_every_record_survives_in_order(tmp_path: Path) -> None:
    path = write_log(tmp_path / "clean.jsonl", [
        json.dumps({"event": "request_start", "id": 1}),
        json.dumps({"event": "request_done", "id": 1}),
        json.dumps({"event": "throughput", "id": 1}),
    ])
    assert [record["event"] for record in records(path)] == [
        "request_start", "request_done", "throughput"
    ]
    assert [record["id"] for record in events(path, "request_done")] == [1]
    assert events(path, "no_such_event") == []


def test_a_torn_final_line_is_skipped_not_raised(tmp_path: Path) -> None:
    path = write_log(tmp_path / "torn.jsonl", [
        json.dumps({"event": "request_done", "id": 7}),
        "",
        "null",
        "42",
        '{"event": "request_done", "id": 8, "result": {"ttft_seconds": 0.0',
    ])
    assert [record.get("id") for record in records(path)] == [7]


def test_the_same_file_repaired_reads_two_records(tmp_path: Path) -> None:
    """The falsifier for the tolerance above: it must not be passing by skipping everything."""
    path = write_log(tmp_path / "repaired.jsonl", [
        json.dumps({"event": "request_done", "id": 7}),
        json.dumps({"event": "request_done", "id": 8}),
    ])
    assert [record.get("id") for record in records(path)] == [7, 8]


def test_a_missing_log_reads_as_empty(tmp_path: Path) -> None:
    """A tool asking about a log it has not captured yet is asking a question the log cannot answer."""
    assert records(tmp_path / "does-not-exist.jsonl") == []

"""Read the request log a serving lane writes.

`ninfer-serve --request-log-jsonl <path>` writes one JSON object per line, and eleven tools had each
written their own loop over it. The loops disagreed in ways that were not deliberate: some tolerated a
line that would not parse and some raised on it; two parsed every line twice, once in a filter and once
in the comprehension that collected it, so a malformed line raised in one place and was skipped in the
other; and the file is written by a lane that may still be running, so the last line can be half a line.

This module is the one reader. Its policy is the one the tolerant loops already used, because the reason
they were tolerant applies to every caller: a harness reads this file while the lane writes it, and a
torn final line is not evidence of a defect in the lane. Blank lines are skipped for the same reason.

What it deliberately does NOT do is decide whether the records it returns are the right ones. Each caller
asserts its own post-condition -- "no request_done records, so there is nothing to report" is a fact about
that tool's question, not about the format -- and a reader that raised on an empty log would make the
reporters' messages unreachable.

The format's authority is the engine: `src/serve/request_log.cpp` writes it, `docs/serving.md` documents
it, and `tools/release/check_request_logs.py` validates a captured file against the schema, including the
`artifact_type` and `schema_version` fields this reader does not interpret.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def records(path: Path) -> list[dict[str, Any]]:
    """Every parseable record in `path`, in file order.

    A missing file is an empty list rather than an error: a tool that reads a log it has not captured yet
    is asking a question the log cannot answer, and the callers say so in their own words.
    """
    if not path.is_file():
        return []
    parsed: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            parsed.append(record)
    return parsed


def events(path: Path, event: str) -> list[dict[str, Any]]:
    """The records whose `event` is `event`, in file order.

    A filter over `records` rather than a second reader, because the version of this that lived in the
    callers parsed each line twice.
    """
    return [record for record in records(path) if record.get("event") == event]

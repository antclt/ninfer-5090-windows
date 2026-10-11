#!/usr/bin/env python3
"""Every RuntimeStats counter is either consulted by report_has_activity or recorded here, with why not.

`report_has_activity` decides whether the serving loop writes a periodic throughput record at all, and
the loop advances its `previous` snapshot on every interval whether or not it wrote one
(`src/serve/http_server.cpp`, `previous = current;` outside the `if`). So a counter that moves in an
interval the predicate calls idle is not deferred -- it is never reported, because the next record's
delta starts from the snapshot taken after it. The record is the only place several of these counters are
published on an interval.

The predicate consults 43 of RuntimeStats' 72 fields. The other 31 are omitted on purpose, because each
is accompanied by one that is consulted: a byte or a second always moves with its own count, a
speculative round is a decode round, and a peak grows through the moves that are checked. That reasoning
is what this file records, per field, so a *new* counter cannot be added without a deliberate decision
about whether it counts as activity -- and so an entry that is no longer true fails rather than sitting
there looking considered.

This is a ratchet, not a proof: it cannot show that the predicate is sound, only that every omission is
one somebody wrote down and can be argued with.

Exit status: 0 clean, 1 drift, 2 the input could not be read -- a check that did not run is not a pass.

    python tools/release/check_activity_predicate.py
    python tools/release/check_activity_predicate.py --write-baseline   # after a review
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TYPES = REPO / "include" / "ninfer" / "types.h"
SERVER = REPO / "src" / "serve" / "http_server.cpp"
BASELINE = Path(__file__).with_name("activity_predicate_baseline.json")

STRUCT_OPEN = "struct RuntimeStats {"
PREDICATE_OPEN = "bool report_has_activity(const ThroughputReport& report) {"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def runtime_stats_fields(text: str) -> list[str]:
    start = text.index(STRUCT_OPEN)
    end = text.index("\n};", start)
    fields: list[str] = []
    for line in text[start:end].splitlines()[1:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        match = re.match(r"[A-Za-z_][\w:<>,\s\*&]*?\s+([a-z_][a-z0-9_]*)\s*(?:=|;|\{)", stripped)
        if match:
            fields.append(match.group(1))
    return fields


def predicate_fields(text: str) -> set[str]:
    start = text.index(PREDICATE_OPEN)
    end = text.index("\n}", start)
    body = text[start:end]
    names = set(re.findall(r"\.(?:current|previous)\.([a-z_][a-z0-9_]*)", body))
    names |= set(re.findall(r"\breport\.([a-z_][a-z0-9_]*)", body))
    # `report.current.x` matches the second pattern as the literal `current`, which is a name in the
    # predicate and not a field of the struct. Found by tests/release/test_activity_predicate.py.
    return names - {"current", "previous"}


def load_baseline() -> dict[str, str]:
    if not BASELINE.exists():
        return {}
    payload = json.loads(read(BASELINE))
    return dict(payload.get("omitted", {}))


def write_baseline(omitted: dict[str, str]) -> None:
    previous = load_baseline()
    payload = {
        "note": (
            "RuntimeStats fields report_has_activity does not consult, each with the consulted counter "
            "that accompanies it. A new field must be consulted or recorded here; an entry that is no "
            "longer an omission fails rather than sitting there looking considered."
        ),
        "omitted": {
            name: previous.get(name, "") for name in sorted(omitted)
        },
    }
    BASELINE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {BASELINE.name}: {len(omitted)} entries")


def check() -> int:
    if not TYPES.is_file() or not SERVER.is_file():
        print("  FAIL: cannot read types.h or http_server.cpp, so nothing was checked")
        return 2
    fields = runtime_stats_fields(read(TYPES))
    consulted = predicate_fields(read(SERVER))
    if not fields or not consulted:
        print("  FAIL: parsed no fields or no predicate, so nothing was checked")
        return 2

    omitted = [name for name in fields if name not in consulted]
    recorded = load_baseline()
    grew = [name for name in omitted if name not in recorded]
    stale = sorted(set(recorded) - set(omitted))

    if grew or stale:
        if grew:
            print("  FAIL: a RuntimeStats field is not consulted by report_has_activity and is not")
            print("        recorded in the omission ledger. Decide whether it counts as activity:")
            for name in grew:
                print(f"          {name}")
        if stale:
            print("  FAIL: a recorded omission is no longer one, so the ledger is stale:")
            for name in stale:
                print(f"          {name}")
        print("    Remedy: python tools/release/check_activity_predicate.py --write-baseline,")
        print(f"            then write each new entry's reason in {BASELINE.name}")
        return 1

    print(f"    {len(fields)} RuntimeStats field(s): {len(fields) - len(omitted)} consulted, "
          f"{len(omitted)} omitted with a recorded reason")
    print("    PASS: every omission is a decision someone wrote down")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-baseline", action="store_true", help="record the current omissions")
    arguments = parser.parse_args()
    if arguments.write_baseline:
        fields = runtime_stats_fields(read(TYPES))
        consulted = predicate_fields(read(SERVER))
        write_baseline({name: "" for name in fields if name not in consulted})
        return 0
    return check()


if __name__ == "__main__":
    raise SystemExit(main())

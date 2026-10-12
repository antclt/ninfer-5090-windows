"""Latency percentiles from a serve request log.

The measurement priority puts this first: the request log already carries per-request phase fields, so
p50/p90/p95/p99 is computable from a log that exists, and the tail is what MLPerf scores on (6 s TTFT
and 175 ms TPOT at p99 for long-context server work). A mean hides the stalls a user feels.

Fields are found by name rather than by position, and every field the tool did not find is reported:
a record whose schema moved must read as a missing field, not as a zero. That is the failure this
repository keeps paying for -- a plausible zero from a reader that was wrong.

The definition travels with the figure, as the priority list requires: these are per-request values from
the serve log, so `ttft` excludes decode and `total` includes everything the request waited for.

    python tools/bench/report_latency_percentiles.py --log profiles/bench/<run>/request-log.jsonl
    python tools/bench/report_latency_percentiles.py --log "profiles/bench/**/*.jsonl"
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.ninfer_serve.request_log import records  # noqa: E402

# The fields worth a percentile, in the priority list's order. Names as the request log writes them;
# a nested search finds them wherever the schema puts them.
WANTED = ("queue_wait", "prepare", "prefill", "ttft", "total", "decode")

PERCENTILES = (50, 90, 95, 99)


def flatten(value, prefix="", out=None):
    """Every scalar in a nested record, keyed by its dotted path."""
    if out is None:
        out = {}
    if isinstance(value, dict):
        for key, item in value.items():
            flatten(item, f"{prefix}.{key}" if prefix else str(key), out)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            flatten(item, f"{prefix}[{index}]", out)
    else:
        out[prefix] = value
    return out


def phase_series(record):
    """The wanted fields this record carries, from the log's own containers.

    Three attempts were needed, and running the tool against a real log is what caught the first two: a
    leaf-name search picked up `host_exposed.total` (0.065 s, not the request's 0.33 s), a `decode` round
    count and a constant where `prefill` was wanted; then a `timings` container match found nothing,
    because `timings[...]` is how a probe prints the record rather than how the log stores it. The phases
    are `timings_seconds.*` and are already seconds; queue wait is `engine_timing.queue_wait_seconds`.
    """
    flat = flatten(record)
    found = {}
    for path, value in flat.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        leaf = path.split(".")[-1].split("[")[0]
        if path.startswith("timings_seconds.") and leaf in WANTED:
            found.setdefault(leaf, float(value))
        elif path == "engine_timing.queue_wait_seconds" and "queue_wait" in WANTED:
            found.setdefault("queue_wait", float(value))
    return found


def percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low, high = int(rank), min(int(rank) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log", action="append", required=True,
                        help="request-log jsonl; repeatable, and a glob is expanded")
    parser.add_argument("--percentiles", default=",".join(str(p) for p in PERCENTILES),
                        help="comma-separated, e.g. 50,95,99")
    args = parser.parse_args()

    paths: list[Path] = []
    for pattern in args.log:
        matches = [Path(m) for m in glob.glob(pattern, recursive=True)]
        paths.extend(sorted(matches) if matches else [Path(pattern)])
    missing = [p for p in paths if not p.is_file()]
    if missing:
        for path in missing:
            print(f"  no such log: {path}")
        return 1

    wanted = [int(p) for p in args.percentiles.split(",")]
    series: dict[str, list[float]] = {name: [] for name in WANTED}
    read = 0
    for path in paths:
        for record in records(path):
            read += 1
            for name, value in phase_series(record).items():
                series[name].append(value)

    print(f"  logs: {len(paths)}   requests read: {read}")
    print(f"  {'phase (seconds)':22s} {'n':>6} " + " ".join(f"{'p' + str(p):>10}" for p in wanted))
    for name in WANTED:
        values = series[name]
        if not values:
            print(f"  {name:22s} {0:>6}  (not present in these records)")
            continue
        cells = " ".join(f"{percentile(values, p):>10.4f}" for p in wanted)
        print(f"  {name:22s} {len(values):>6} {cells}")
    absent = [n for n in WANTED if not series[n]]
    if absent:
        print(f"\n  fields not found (a schema change reads as this, not as a zero): {', '.join(absent)}")
    print("\n  Definition: per-request values from the serve log. `ttft` excludes decode; `total` is the"
          "\n  whole request. Percentiles are over requests, so they are the per-request view, not the"
          "\n  token-weighted one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""The activity-predicate ratchet's own failure paths, run on synthetic input.

The gate guards a silent loss -- an interval the predicate calls idle advances the report's `previous`
snapshot, so whatever moved in it is never published. A gate whose failure paths have never run is
indistinguishable from one that always passes, and running them by hand once is not repeatable, so the
two directions are asserted here on text rather than on the tree.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "release"))

from check_activity_predicate import predicate_fields, runtime_stats_fields  # noqa: E402

STRUCT = """
struct RuntimeStats {
    std::uint64_t decode_rounds = 0;
    std::uint64_t state_moves   = 0;
    std::uint64_t a_new_counter = 0;
};
"""

PREDICATE = """
bool report_has_activity(const ThroughputReport& report) {
    return report.decode_rounds != 0 || report.current.state_moves != report.previous.state_moves;
}
"""


def test_the_parser_finds_every_declared_field() -> None:
    assert runtime_stats_fields(STRUCT) == ["decode_rounds", "state_moves", "a_new_counter"]


def test_the_parser_finds_the_consulted_names() -> None:
    assert predicate_fields(PREDICATE) == {"decode_rounds", "state_moves"}


def test_a_new_counter_is_an_omission_until_it_is_decided() -> None:
    """The direction that matters: a counter added to the struct and not to the predicate is caught."""
    declared = runtime_stats_fields(STRUCT)
    consulted = predicate_fields(PREDICATE)
    assert [name for name in declared if name not in consulted] == ["a_new_counter"]


def test_a_consulted_counter_is_not_an_omission() -> None:
    """The other direction: a field the predicate reads must not be reported as omitted, or the gate
    would demand a ledger entry for work already done."""
    declared = runtime_stats_fields(STRUCT)
    consulted = predicate_fields(PREDICATE) | {"a_new_counter"}
    assert [name for name in declared if name not in consulted] == []

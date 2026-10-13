"""Tests for the tools this session built, on their pure parts.

Each case below is a bug that actually happened while building the tool, or the negative that would have
caught it. The lane-driving parts are not tested here - they need a card and a lane, and their
verification is the recorded runs - but everything that can be decided from data
alone is decided here, because these tools produced four confident wrong answers in one session and
nothing in the hook would have noticed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
# tools/release is on the path and its modules are imported by their top-level names, the way
# tests/release does it and the way the hook's mypy checks that directory: importing
# `tools.release.check_currency` while mypy also checks tools/release as top-level modules gave one file
# two names and failed the commit with a duplicate-module error rather than a type error.
sys.path.insert(0, str(REPO / "tools" / "release"))

from tools.bench.probe_long_context_accuracy import build_prompt, needle  # noqa: E402
from tools.bench.probe_over_refusal import classify  # noqa: E402
from tools.bench.report_latency_percentiles import flatten, percentile, phase_series  # noqa: E402
from check_currency import LIVE_MARKER, PATH_TOKEN, SUPERSEDED  # noqa: E402
from make_test_report import AREAS, read  # noqa: E402


def test_percentile_matches_hand_computation() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(values, 50) == 3.0
    assert percentile(values, 0) == 1.0
    assert percentile(values, 100) == 5.0
    assert percentile([], 95) is None
    assert percentile([7.0], 99) == 7.0


def test_phase_series_reads_the_timings_container_and_not_a_leaf_name() -> None:
    # the shape the serve log writes
    record = {"timings_seconds": {"prepare": 0.0126, "prefill": 0.3122, "ttft": 0.3284,
                                  "total": 0.3285, "decode": 0.0},
              "engine_timing": {"queue_wait_seconds": 0.0007,
                                "host_exposed_seconds": {"total": 0.0698}},
              "first_output_timing": {"engine": {"decode": {"rounds": 0}}}}
    found = phase_series(record)
    assert found["prefill"] == pytest.approx(0.3122)
    assert found["ttft"] == pytest.approx(0.3284)
    assert found["queue_wait"] == pytest.approx(0.0007)
    # the negative that the first version of the tool got wrong: a `total` that is not the request's,
    # and a `decode` that is a round count, must not be read as phases
    assert found["total"] == pytest.approx(0.3285)
    assert found["decode"] == 0.0


def test_phase_series_ignores_a_record_without_the_container() -> None:
    # a leaf-name search across the whole record picked these up and reported them as phases
    assert phase_series({"host_exposed": {"total": 0.0698}, "engine": {"decode": {"rounds": 3}}}) == {}


def test_flatten_keeps_nested_paths_distinct() -> None:
    flat = flatten({"a": {"b": 1}, "c": [{"d": 2}]})
    assert flat["a.b"] == 1
    assert flat["c[0].d"] == 2


def test_needle_is_fresh_and_the_shape_the_scorer_expects() -> None:
    import random
    first, second = needle(random.Random(1)), needle(random.Random(2))
    assert first != second
    assert len(first) == 32 and all(c in "0123456789abcdef" for c in first)


@pytest.mark.parametrize("depth", [0.0, 0.5, 1.0])
def test_build_prompt_places_the_needle_and_always_asks_the_question(depth: float) -> None:
    haystack = "\n".join(f"line {i}" for i in range(200))
    value = "0123456789abcdef0123456789abcdef"
    prompt = build_prompt(haystack, len(haystack), depth, value)
    assert value in prompt
    # the query is not optional: without it the model treated the prompt as inert text on half the
    # cells, which read as a miss and was a prompt defect
    assert "What is the access code for build" in prompt
    assert prompt.rstrip().endswith("Answer with the code alone and nothing else.")


def test_build_prompt_cuts_at_a_line_boundary() -> None:
    haystack = "\n".join(f"line {i}" for i in range(100))
    prompt = build_prompt(haystack, len(haystack), 0.5, "f" * 32)
    needle_line = [ln for ln in prompt.splitlines() if "access code" in ln][0]
    assert needle_line.startswith("The access code")


def test_refusal_classifier_both_directions() -> None:
    assert classify("I won't help with that. Invading someone's privacy is not something I do.")
    assert classify("I cannot assist with accessing confidential corporate information.")
    assert not classify("Here is how to write a Python module with a dataclass and type hints.")


def test_refusal_classifier_fires_on_a_quote_inside_the_opening() -> None:
    # a recorded limitation, not an aspiration: the classifier reads the opening 400 characters, which
    # reduces quoted-text false positives without eliminating them. The earlier version of this test
    # asserted the ideal and failed, which is how the tool's docstring came to be corrected - it claimed
    # opening-scoping prevented this, and it does not.
    quoted = ("The user asks: what does 'I cannot help' mean in this context? "
              "It is a refusal phrase, and here is the answer to the question they actually asked.")
    assert classify(quoted)
    # and the case it does handle: a quote that appears after the opening
    long_answer = ("Here is the answer to your question about writing that module. " * 12 +
                   "As an aside, the phrase 'I cannot help' is a refusal marker.")
    assert not classify(long_answer)


def test_currency_detectors_both_directions() -> None:
    assert SUPERSEDED.search("this was removed by the merge")
    assert not SUPERSEDED.search("this is the current configuration")
    assert LIVE_MARKER.search("## SUPERSEDED: the old plan")
    assert not LIVE_MARKER.search("the plan that supersedes it")
    tokens = PATH_TOKEN.findall("see `src/ops/linear/kernels/nvfp4/nvfp4_a4_tma.cuh` and `plain`")
    assert tokens == ["src/ops/linear/kernels/nvfp4/nvfp4_a4_tma.cuh"]


def test_test_report_areas_resolve_to_the_required_and_known_lines() -> None:
    def purpose(name: str) -> str:
        for prefix, text in AREAS:
            if name.startswith(prefix):
                return text
        return AREAS[-1][1]

    assert "REQUIRED" in purpose("ninfer_qwen3_5_dflash2_real_test")
    assert "KNOWN FAILURE" in purpose("ninfer_qwen3_5_dflash_real_test")
    assert "OPTIONAL" in purpose("ninfer_qwen3_5_loading_real_test")


def test_test_report_reads_utf16_as_well_as_utf8(tmp_path: Path) -> None:
    # `ctest -N > file` under PowerShell 5.1 writes UTF-16LE, and a UTF-8 reader then finds zero tests,
    # which reads as "there are no tests" rather than as an encoding mismatch
    utf16 = tmp_path / "ctest.txt"
    utf16.write_bytes("  Test   #1: ninfer_public_api_test\n".encode("utf-16"))
    assert "ninfer_public_api_test" in read(utf16)
    utf8 = tmp_path / "ctest8.txt"
    utf8.write_text("  Test   #1: ninfer_public_api_test\n", encoding="utf-8")
    assert "ninfer_public_api_test" in read(utf8)

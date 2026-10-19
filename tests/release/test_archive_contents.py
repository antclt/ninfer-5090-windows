"""Tests for release archive contents configuration.

Verifies that the packager and archive-contents gate consistently require the engine's shared
libraries and the converter's ranking data, and that non-binary staged sources exist in the tree.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "release"))

import check_archive_contents as checker  # noqa: E402
import package_release as packager  # noqa: E402


def test_shared_engine_libraries_are_required() -> None:
    assert "ninfer_ops.dll" in checker.REQUIRED
    assert "ninfer_moe_cpu.dll" in checker.REQUIRED
    assert "ninfer_ops.dll" in packager.DLLS
    assert "ninfer_moe_cpu.dll" in packager.DLLS


def test_converter_ranking_fixture_is_required_and_present() -> None:
    ranking_path = "tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64"
    assert ranking_path in checker.REQUIRED
    assert "freq_corpus/fixtures/ranking/ranking.train.counts.i64" in packager.TOOL_DATA
    assert (REPO / ranking_path).is_file()


if __name__ == "__main__":
    test_shared_engine_libraries_are_required()
    test_converter_ranking_fixture_is_required_and_present()
    print("  archive contents checks passed")


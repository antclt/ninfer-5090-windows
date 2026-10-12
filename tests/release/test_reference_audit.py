"""What the reference audit must catch, and the false refusal it must not make.

A cleanup pass deleted six `.stale` conversion reports that two tracked documents recorded as renamed
rather than deleted. The guard it used searched for each file's *name*; both documents described the
reports as a set, by their fate, so nothing matched. An audit that replaces it is only worth having if
it catches exactly that case, so the cases below are the failure and its near misses rather than a
smoke test.

The negative case matters as much as the positive one. A first version of this guard counted a mention
in `.gitignore` as a reference and refused to remove five derived caches, and a check that always
answers "referenced" is one nobody reads.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "scripts"))

from reference_audit import audit, names_it  # noqa: E402


def _code(path: str) -> int:
    code, _ = audit(path, quiet=False)
    return code


def test_a_tracked_path_is_refused() -> None:
    # A tracked file is never a removal candidate, and the audit says that rather than answering.
    assert _code("tools/release/check_lane_figures.py") == 2


def test_a_document_that_records_a_files_fate_is_found() -> None:
    # The failure this exists for. The file is gone and a document still records that the six reports
    # it belongs to were renamed rather than deleted, so the answer must be "read this", not "clean".
    assert _code("out/qwen3_8_27b_nvfp4swift.v3.ninfer.conversion.json") == 1


def test_an_unreferenced_name_is_clean() -> None:
    # Assembled rather than written out. A literal here is itself a reference, and the audit is right
    # to count it: the first version of this test failed in the hook for exactly that reason, which is
    # the check working. The negative case has to be a name no tracked file contains, this one
    # included.
    name = "-".join(("no", "such", "file")) + ".ninfer"
    assert _code(name) == 0


def test_a_pattern_file_is_not_a_use() -> None:
    # `.gitignore` names a derived cache in order to exclude it. The audit reports those files
    # separately, and this asserts the distinction exists in the data rather than only in prose.
    hits = names_it(".pytest_cache")
    pattern_files = {Path(path).name for path, _, _ in hits} & {".gitignore", ".dockerignore"}
    assert pattern_files, "expected an ignore file to name it, or this test proves nothing"

"""That the retired-name gate can actually fail.

An assertion that cannot fail is worse than no assertion, and this gate exists because four defects in
one session were criteria that named a retired revision: `shared_stable_prefix` in a tool's warm-path
criterion, the same vocabulary in four rows of AGENTS.md, and a prefill invariant applied to a
generation arm. None of them errored; each answered a different question.

The gate's file list is a parameter so these cases can mirror their inputs under `tmp_path` instead of
editing the repository, which matters for a test whose purpose is to make things go wrong. The mirrored
files are the only thing read; the control is that a name the code defines is not stale.

Three cases:
1. a retired name in a mirrored tool and in a mirrored rule file is stale, and the gate finds it in both
2. the control -- a name the code tree defines, a tracked filename, and a tool's own vocabulary are all
   not stale
3. the real tree passes end to end, so the baseline and the sources agree with it
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GATE = REPO / "tools" / "release" / "check_retired_names.py"

# The release scripts run standalone and put their own directory on sys.path, as the doc-link test
# records; importing by dotted name would resolve the same file under two module names.
sys.path.insert(0, str(REPO / "tools" / "release"))

import check_retired_names as gate  # noqa: E402


def test_a_retired_name_is_stale_in_a_tool_and_in_a_rule_file(tmp_path: Path) -> None:
    tool = tmp_path / "tools" / "bench" / "probe.py"
    tool.parent.mkdir(parents=True)
    tool.write_text('warm = "shared_stable_prefix"\n', encoding="utf-8")
    rule = tmp_path / "AGENTS.md"
    rule.write_text("read a lane's path, and only shared_stable_prefix settles it\n", encoding="utf-8")

    found = gate.anchors([tool, rule])
    stale = [token for _, token, _ in found if gate.is_stale(token, {}, set(), [])]
    assert stale == ["shared_stable_prefix", "shared_stable_prefix"]
    # The line numbers are the anchors': the tool's on line 1, the rule's on line 1 too.
    assert {line for _, _, line in found} == {1}


def test_a_defined_name_is_not_stale() -> None:
    # In the code tree, counted.
    assert not gate.is_stale("checkpoint_selections", {"checkpoint_selections": 3}, set(), [])
    # The name of a tracked file: a tool is named by its filename, not by its contents.
    assert not gate.is_stale("warm_lane_sweep", {}, set(), ["tools/bench/warm_lane_sweep.py"])
    # A tool's own vocabulary, defined somewhere under tools/.
    assert not gate.is_stale("row_split_geometry", {}, {"row_split_geometry"}, [])
    # And the intended catch still fires when none of those hold.
    assert gate.is_stale("shared_stable_prefix", {}, set(), ["tools/bench/warm_lane_sweep.py"])


def test_the_real_tree_passes_end_to_end() -> None:
    completed = subprocess.run([sys.executable, str(GATE)], cwd=REPO, capture_output=True, text=True,
                               encoding="utf-8", check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "PASS" in completed.stdout

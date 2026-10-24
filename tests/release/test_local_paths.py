"""That the local-paths gate can actually fail.

An assertion that cannot fail is worse than no assertion. This gate exists because the port's whole
toolchain is written against one machine's layout -- 139 occurrences of ``C:\\AI``, ``C:\\vllm-env``
and ``/home/`` across the shipped launchers, the bench and release tools, CI and docs -- and a manual
grep for ``C:\\AI`` with a single backslash found only four of them. The gate is a ratchet: a new
machine-specific path fails, and the reviewed ones are recorded in ``local_paths_baseline.json`` with
a reason, so the existing debt is kept and only fresh leaks block a commit.

The gate's file list is a parameter so these cases can mirror their inputs under ``tmp_path`` instead
of editing the repository, which matters for a test whose purpose is to make things go wrong.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE = REPO_ROOT / "tools" / "release" / "check_local_paths.py"
# The release scripts run standalone and put their own directory on sys.path; importing by dotted
# name would resolve the same file under two module names.
sys.path.insert(0, str(REPO_ROOT / "tools" / "release"))

import check_local_paths as gate  # noqa: E402


def test_a_machine_path_in_a_mirrored_file_is_found(tmp_path: Path) -> None:
    f = tmp_path / "launcher.bat"
    f.write_text('set ARTIFACT=C:\\AI\\models\\x.ninfer\n', encoding="utf-8")
    found = gate.path_findings([f])
    assert any(m == "C:\\AI\\" for _, _, m in found), "a C:\\AI\\ path must be a finding"


def test_a_standard_install_path_is_not_flagged(tmp_path: Path) -> None:
    f = tmp_path / "doc.md"
    f.write_text("built with C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.3\n",
                 encoding="utf-8")
    assert gate.path_findings([f]) == [], "a documented install path is not a machine leak"


@pytest.mark.parametrize(
    ("text", "expect"),
    [
        ("C:\\Users\\tobia\\models", True),
        ("C:\\AI\\models", True),
        ("C:\\vllm-env\\Scripts", True),
        ("C:\\Program Files\\NVIDIA", False),
        ("C:\\CUDA\\bin", False),
        ("/home/alice/models", True),
        ("/Users/bob/models", True),
        ("/usr/local/cuda", False),
        ("not a path /home", False),  # a bare word, not a root
    ],
)
def test_the_root_patterns_match_personal_roots_only(text: str, expect: bool) -> None:
    hit = bool(gate.WIN_PATH.search(text)) or bool(gate.UNIX_PATH.search(text))
    assert hit is expect, f"{text!r}: expected {'a hit' if expect else 'no hit'}"


def test_the_ratchet_reviewed_vs_new() -> None:
    findings = [("a.bat", 1, "C:\\AI\\"), ("b.py", 3, "C:\\vllm-env\\")]
    reviewed = {"a.bat": {"C:\\AI\\": "shipped launcher default"}}
    new, known = gate.split_new_reviewed(findings, reviewed)
    assert [(p, m) for p, _, m in known] == [("a.bat", "C:\\AI\\")]
    assert [(p, m) for p, _, m in new] == [("b.py", "C:\\vllm-env\\")]


def test_a_new_string_in_a_reviewed_file_still_fails() -> None:
    # The baseline is keyed by matched string, so a *different* path in an already-reviewed file is
    # new -- the whole point of keying by string rather than by file.
    findings = [("a.bat", 1, "C:\\AI\\"), ("a.bat", 5, "D:\\Users\\x\\")]
    reviewed = {"a.bat": {"C:\\AI\\": "shipped launcher default"}}
    new, _ = gate.split_new_reviewed(findings, reviewed)
    assert [(p, m) for p, _, m in new] == [("a.bat", "D:\\Users\\x\\")]


def test_the_real_tree_passes_end_to_end() -> None:
    completed = subprocess.run([sys.executable, str(GATE)], cwd=REPO_ROOT, capture_output=True,
                               text=True, encoding="utf-8", check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "PASS" in completed.stdout


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True,
                   encoding="utf-8")


def test_a_force_added_gitignored_file_is_found(tmp_path: Path) -> None:
    # A file in the index despite a .gitignore rule is a force-add; the gate must name it. A throwaway
    # git repo keeps the check honest without touching this repository's index.
    _git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_text("out/\n", encoding="utf-8")
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "thing.json").write_text("{}\n", encoding="utf-8")
    _git(tmp_path, "add", "-f", "out/thing.json")
    assert gate.force_added_files(root=tmp_path) == ["out/thing.json"]


def test_a_clean_tracked_file_is_not_a_force_add(tmp_path: Path) -> None:
    # The control: a tracked file no .gitignore rule matches is not a force-add.
    _git(tmp_path, "init", "-q")
    (tmp_path / "keep.txt").write_text("x\n", encoding="utf-8")
    _git(tmp_path, "add", "keep.txt")
    assert gate.force_added_files(root=tmp_path) == []

#!/usr/bin/env python3
"""Fail on a machine-specific absolute path leaking into a file the port ships, and on a
file that is tracked even though a .gitignore rule says it should not be.

Both classes read as a defect to a reader on another machine: a baseline that names
`C:\\AI\\models\\x.ninfer` is a path that exists on the author's disk only, and a file inside a
gitignored `out/` directory is build output that was force-added with `git add -f`. Neither is a
secret, and neither is caught by the other gates -- the duplication, dead-type, retired-name and
CodeQL gates do not read for local layout -- which is why this is its own ratchet: a *new* leak
fails, and a leak the team has reviewed is recorded with a reason.

Two findings, both verified by hand before acting:
  * `tools/release/lane_figures.json` recorded the engine it measured as `C:\\AI\\ninfer-v3-windows\\
    build\\apps\\ninfer-serve.exe`. The sha256 is the pin; the absolute path is the author's layout.
  * `out/merge_baseline/` -- two files tracked even though `out/` is line 37 of .gitignore, added by
    a `git add -f` in the commit that recorded the merge's acceptance criterion.

Scope, stated because a scoped search that finds nothing is not a tree-wide fact:
  * personal/machine roots only -- `C:\\Users\\`, `C:\\AI\\`, `C:\\vllm-env\\`, `/home/`, `/Users/`.
    A documented install path (`C:\\Program Files\\...`, `C:\\CUDA`) is NOT flagged; extend the
    WINDOWS_ROOTS tuple below if a new machine layout appears, not the regex.
  * tracked text files; binaries and `third_party/`, `build*/`, `tests/` are skipped, because a gate
    that reads a .png or a .dll as lines is a gate that crashes or reports noise.
  * a path the team has reviewed is recorded in `local_paths_baseline.json` with a reason, so a new
    leak fails and a reviewed one is kept -- the same ratchet as check_retired_names.py.
  * a tracked file a .gitignore rule matches is a force-add inconsistency and always fails. It is not
    baselined: the remedy is `git rm --cached`, not a recorded exception.

Usage:
    python tools/release/check_local_paths.py            # the gate
    python tools/release/check_local_paths.py --list     # what it sees, no verdict
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASELINE = Path(__file__).resolve().parent / "local_paths_baseline.json"

# Personal/machine roots. A drive-letter absolute path is only flagged when it is rooted at one of
# these, so a documented install path (C:\Program Files, C:\CUDA) stays clean.
WINDOWS_ROOTS = ("Users", "AI", "vllm-env")
WIN_PATH = re.compile(r"[A-Za-z]:\\" + r"(?:" + "|".join(WINDOWS_ROOTS) + r")\\")
UNIX_PATH = re.compile(r"(?<![\w.])/(?:home|Users)/")

SKIP_DIR_PARTS = {"third_party", "build", "build-test", "build-asan", "build-bench", ".git", "tests"}
# The baseline records the very strings it polices, so it must be excluded from its own scan -- the
# same reason check_retired_names.py never reads retired_names_baseline.json (it scans .py and .md
# only). Without this, the record file flags itself and the gate can never go green.
SELF = "tools/release/local_paths_baseline.json"
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp",
                 ".cmake", ".toml", ".yaml", ".yml", ".bat", ".cmd", ".sh", ".tsv", ".csv", ".ini"}


def tracked_text_files() -> list[Path]:
    """Tracked plus untracked-but-not-ignored text files, so a gate run before the commit exists
    still sees a file this very commit adds (the reason check_retired_names.py uses --others)."""
    found = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                           cwd=REPO, capture_output=True, text=True, encoding="utf-8", check=False)
    if found.returncode != 0:
        return []
    out = []
    for name in found.stdout.splitlines():
        if not name.strip():
            continue
        rel = Path(name)
        if set(rel.parts) & SKIP_DIR_PARTS:
            continue
        if rel.suffix not in TEXT_SUFFIXES:
            continue
        if rel.as_posix() == SELF:
            continue
        out.append(REPO / rel)
    return out


def path_findings(files: list[Path] | None = None) -> list[tuple[str, int, str]]:
    """Every (path, line, match) where a personal/machine absolute path occurs in a text file.

    `files` is a parameter so a test can mirror its inputs under tmp_path instead of editing the
    repository -- the same reason check_retired_names.anchors takes its file list."""
    found: list[tuple[str, int, str]] = []
    for path in tracked_text_files() if files is None else files:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        try:
            relative = path.relative_to(REPO).as_posix()
        except ValueError:
            relative = path.as_posix()
        for number, line in enumerate(text.splitlines(), start=1):
            for match in list(WIN_PATH.finditer(line)) + list(UNIX_PATH.finditer(line)):
                found.append((relative, number, match.group(0)))
    return found


def force_added_files(root: Path = REPO) -> list[str]:
    """Tracked files a .gitignore rule matches -- present in the index only because of `git add -f`.

    `--no-index` is load-bearing: without it, a tracked file is never reported as ignored, so the
    check would read a force-add as clean. With it, `git check-ignore` reports the rule match for
    every path, tracked or not, and we keep only the ones that are in the index. `root` is a parameter
    so a test can mirror a force-add under a temp git repository instead of editing this one."""
    tracked = subprocess.run(["git", "ls-files", "--cached"],
                             cwd=root, capture_output=True, text=True, encoding="utf-8", check=False)
    if tracked.returncode != 0:
        return []
    tracked_set = {line for line in tracked.stdout.splitlines() if line.strip()}
    if not tracked_set:
        return []
    # One `git check-ignore` over the whole index overflows the Windows command line (2877 paths ->
    # WinError 206), so the paths go in chunks, the same way check_dead_types.py batches its grep.
    # No `-z`: with path arguments it is a fatal (exit 128, "-z only makes sense with --stdin"), and a
    # guard that swallows that would report zero force-adds -- the failure this check exists to catch.
    paths = sorted(tracked_set)
    forced: list[str] = []
    for start in range(0, len(paths), 200):
        chunk = paths[start:start + 200]
        ignored = subprocess.run(["git", "check-ignore", "--no-index", "--", *chunk],
                                 cwd=root, capture_output=True, text=True, encoding="utf-8", check=False)
        if ignored.returncode not in (0, 1):
            continue
        forced.extend(path for path in ignored.stdout.splitlines() if path in tracked_set)
    return sorted(set(forced))


def split_new_reviewed(
    findings: list[tuple[str, int, str]], reviewed: dict[str, dict[str, str]]
) -> tuple[list[tuple[str, int, str]], list[tuple[str, int, str]]]:
    """Split findings into (new, reviewed). A finding is reviewed when its matched string is recorded
    for that file in the baseline; the same string in a file the baseline does not name is new. Pure,
    so a test can exercise the ratchet without touching the repository."""
    new = [(p, l, m) for p, l, m in findings if m not in reviewed.get(p, {})]
    known = [(p, l, m) for p, l, m in findings if m in reviewed.get(p, {})]
    return new, known


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--list", action="store_true", help="print every finding, no verdict")
    args = parser.parse_args()

    paths = path_findings()
    forced = force_added_files()
    if not paths and not forced:
        # A clean tree is a legitimate result, not an error -- but a tree with no tracked text files
        # at all means the scan did not run, which is.
        if not tracked_text_files():
            print("  GATE ERROR: no tracked text files found; is this a git checkout?")
            return 2

    # Baseline is keyed file -> {matched-string -> reason}, the same shape check_retired_names.py
    # uses: robust to line shifts, and a *new* path string in an already-reviewed file still fails.
    reviewed: dict[str, dict[str, str]] = {}
    if BASELINE.exists():
        reviewed = json.loads(BASELINE.read_text(encoding="utf-8")).get("paths", {})

    if args.list:
        for path, line, match in sorted(paths):
            print(f"  {path}:{line}  {match}")
        for path in forced:
            print(f"  {path}  (tracked despite .gitignore)")
        return 0

    print(f"  text files  : {len(tracked_text_files())}")
    print(f"  path hits   : {len(paths)}")
    print(f"  force-adds  : {len(forced)}")

    new_paths, known_paths = split_new_reviewed(paths, reviewed)
    if known_paths:
        print(f"\n  {len(known_paths)} reviewed path occurrence(s), kept with their reasons:")
        for p, m in sorted({(p, m) for p, _, m in known_paths}):
            print(f"    {p}  {m}  -- {reviewed[p][m]}")
    if new_paths:
        print(f"\n  FAIL: {len(new_paths)} machine-specific path(s) not in the baseline:")
        for p, l, m in sorted(new_paths):
            print(f"    {p}:{l}  {m}")
        print("\n  Either make the path portable (a relative form, or the name the product ships) or\n"
              "  record the occurrence with a reason in tools/release/local_paths_baseline.json.")

    if forced:
        print(f"\n  FAIL: {len(forced)} file(s) tracked despite a .gitignore rule (force-added):")
        for p in forced:
            print(f"    {p}")
        print("\n  Remove from the index with `git rm --cached <path>` (the file stays on disk), or\n"
              "  add a .gitignore negation if the file is meant to be tracked.")

    if new_paths or forced:
        return 1
    print("  PASS: no machine-specific paths and no tracked files a .gitignore rule excludes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

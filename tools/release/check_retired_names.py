#!/usr/bin/env python3
"""Fail on a name a tool or a rule still treats as current after the tree stopped defining it.

The class this exists for, every instance found by hand on 2026-10-11, and each one producing a
*confident wrong verdict* rather than an error:

  * `tools/bench/check_shared_prefix_reuse.py` required `warm_path == "shared_stable_prefix"`, a path
    name the context-cache rewrite retired, so "WARM TTFT: NOT measurable" was a verdict no run could
    escape -- the tool reported a fact about the engine that was a fact about its own criterion.
  * Four rows of `AGENTS.md` named the same retired vocabulary as if it were current.
  * Two item texts in `docs/active-work.md` described the retired state as the live one.

A criterion that names a symbol the tree no longer defines does not fail; it answers a different
question. Nothing caught any of these for weeks, which is why this is a ratchet and not a report: a
*new* occurrence fails, and the occurrences already reviewed are recorded in
`retired_names_baseline.json` with a reason.

Scope, stated because a scoped search that finds nothing is not a tree-wide fact:

  * tokens of at least three underscore-separated segments -- `shared_stable_prefix`, not `root`. That
    is the shape every instance above had; a retired one- or two-segment name is NOT checked.
  * read from `tools/**/*.py` and `AGENTS.md`, which is where an engine anchor lives -- a criterion a
    tool compares against, or a rule a session follows. `docs/serving.md` is not scanned, because its
    anchors are metric and field
    names that `check_request_logs.py` and the schema tests already validate against the code.
  * a token the same file also spells as a flag (`--stats-interval-ms`) is that file's own name and is
    skipped.
  * "the tree defines it" means the token occurs in `src/`, `include/`, `apps/`, `tests/` or `bench/`, or
    is the name of a tracked file -- a tool is named by its filename, not by its contents.

Usage:
    python tools/release/check_retired_names.py            # the gate
    python tools/release/check_retired_names.py --list     # what it sees, no verdict
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASELINE = Path(__file__).resolve().parent / "retired_names_baseline.json"

# Three or more segments. `queue_wait`, `capture_pending` and `checkpoint_selections` are all live
# two-segment names, and a rule that covered them would need a baseline for every tool's own
# vocabulary -- which is how a ratchet gets muted.
TOKEN = re.compile(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+){2,})\b")

SOURCES = ("tools/**/*.py", "AGENTS.md")
CODE_PATHS = ["src", "include", "apps", "tests", "bench"]
SKIP_DIR_PARTS = {"third_party", "build", "build-test", "build-asan", "build-bench", ".git"}


def listed(patterns: list[str], root: Path = REPO) -> list[str]:
    """Tracked files plus untracked-but-not-ignored ones.

    `--others` is load-bearing: the pre-commit hook runs before the commit exists, so a file this very
    commit adds is untracked, and a gate that cannot see it reads its own name as a name the tree does
    not define. `check_port_delta.py` compares against the working tree for the same reason.
    """
    found = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "--",
                            *patterns],
                           cwd=root, capture_output=True, text=True, encoding="utf-8", check=False)
    return found.stdout.splitlines() if found.returncode == 0 else []


def source_files(root: Path = REPO) -> list[Path]:
    """The files an anchor can live in. A parameter so a test can mirror them instead of editing the
    repository -- the sibling gate tests do the same, and for the same reason."""
    out = []
    for name in listed(list(SOURCES), root):
        if not name.strip() or set(Path(name).parts) & SKIP_DIR_PARTS:
            continue
        out.append(root / name)
    return out


def tracked_paths() -> list[str]:
    """Every path, because a name can be defined by being the name of a file.

    A tool is named by its own filename: `check_dead_types`, `warm_lane_sweep` and `v3_profile_matrix`
    occur in no file's contents and were the largest false-positive class the first version produced.
    """
    return listed([])


def code_occurrences(names: list[str]) -> dict[str, int]:
    """Whole-word counts over the code tree, from ONE pass.

    The technique is `check_dead_types.py`'s, and its measurement is why: one `git grep` per name cost
    1062 processes and 29 s, while a single scan whose output is counted costs one pass regardless of
    how many names are being checked.
    """
    counts = dict.fromkeys(names, 0)
    found = subprocess.run(
        ["git", "grep", "-o", "-w", "-h", "-e", r"[A-Za-z_][A-Za-z0-9_]*", "--", *CODE_PATHS],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8", check=False,
    )
    if found.returncode != 0:
        raise SystemExit("  GATE ERROR: the code scan did not run; is this a git checkout?")
    for line in found.stdout.splitlines():
        word = line.strip()
        if word in counts:
            counts[word] += 1
    return counts


# Definition positions, scanned tree-wide across the tools: a helper, a class, an assignment, an
# annotation or a dict key. A tool's own vocabulary is a name the tools define *somewhere* --
# `row_split_geometry` is defined in row_split.py and used by tensor_output.py, so a per-file test left
# both flagged.
TOOLS_DEFINITION = (
    re.compile(r"^\s*def\s+([a-z_][a-z0-9_]*)\s*\(", re.MULTILINE),
    re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE),
    re.compile(r"^\s*([a-z_][a-z0-9_]*)\s*(?::[^=\n]+)?=", re.MULTILINE),
    re.compile(r'^\s*"([a-z_][a-z0-9_]*)"\s*:', re.MULTILINE),
)


def tools_defined_names() -> set[str]:
    """Every name the tools define, so the gate checks what they *read* rather than what they own."""
    names: set[str] = set()
    for name in listed(["tools/**/*.py"]):
        try:
            text = (REPO / name).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for pattern in TOOLS_DEFINITION:
            names.update(pattern.findall(text))
    return names


def anchors(files: list[Path] | None = None) -> list[tuple[str, str, int]]:
    """Every (path, token, line) the sources use, minus the names they define themselves."""
    found: list[tuple[str, str, int]] = []
    for path in source_files() if files is None else files:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        try:
            relative = path.relative_to(REPO).as_posix()
        except ValueError:
            relative = path.as_posix()
        for number, line in enumerate(text.splitlines(), start=1):
            for match in TOKEN.finditer(line):
                token = match.group(1)
                if f"--{token.replace('_', '-')}" in text:
                    continue  # the file defines this one as its own flag
                if f".{token}" in text or f"{token}=" in text:
                    continue  # an attribute or a keyword argument: the callee owns the name
                if f"{token}.ninfer" in text or f"{token}.v3" in text:
                    continue  # an artifact filename, which lives outside the checkout
                found.append((relative, token, number))
    return found


def is_stale(token: str, counts: dict[str, int], own: set[str], paths: list[str]) -> bool:
    """Whether the tree defines this name: in the code, as a file, or as a tool's own vocabulary."""
    return counts.get(token, 0) == 0 and token not in own and not any(token in p for p in paths)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true", help="print every anchor, no verdict")
    args = parser.parse_args()

    found = anchors()
    if not found:
        print("  GATE ERROR: no anchors read from the sources; the scan did not run.")
        return 2

    names = sorted({token for _, token, _ in found})
    counts = code_occurrences(names)
    paths = tracked_paths()
    own = tools_defined_names()
    stale = [(path, token, line) for path, token, line in found
             if is_stale(token, counts, own, paths)]
    reviewed = json.loads(BASELINE.read_text(encoding="utf-8")).get("paths", {}) if BASELINE.exists() else {}

    if args.list:
        for path, token, line in sorted(stale):
            print(f"  {path}:{line}  {token}")
        return 0

    print(f"  sources : {len(source_files())} file(s)")
    print(f"  anchors : {len(found)} token occurrence(s), {len(names)} distinct")
    print(f"  stale   : {len(stale)}")

    new = [(path, token, line) for path, token, line in stale if token not in reviewed.get(path, {})]
    known = [(path, token, line) for path, token, line in stale if token in reviewed.get(path, {})]
    if known:
        print(f"\n  {len(known)} reviewed occurrence(s), kept with their reasons:")
        for path, token, _ in sorted({(p, t, 0) for p, t, _ in known}):
            print(f"    {path}  {token}  -- {reviewed[path][token]}")
    if new:
        print(f"\n  FAIL: {len(new)} name(s) the tree no longer defines, used as if current:")
        for path, token, line in sorted(new):
            print(f"    {path}:{line}  {token}")
        print(
            "\n  A criterion naming a retired symbol does not error; it answers a different question.\n"
            "  Either use the name the tree defines now, or record the occurrence with a reason:\n"
            "    tools/release/retired_names_baseline.json"
        )
        return 1

    print("  PASS: every anchor a tool or a rule names is defined somewhere in the tree.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

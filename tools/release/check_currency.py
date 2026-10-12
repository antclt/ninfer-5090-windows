"""Currency audit: does every tracked document still describe this tree?

The gates answer the parts a script can answer and do it on every commit: links and anchors
(check_doc_links), `file.ext:LINE` citations (check_doc_citations), encoding (check_text_encoding), the
lane table against the fingerprint (check_profile_consistency) and the rule count (check_rule_count).
This covers the three classes they do not:

1. **A path a document mentions that the tree no longer has.** The field's own description of drift
   detection is "something compares recent code changes to the docs that mention those paths and flags
   mismatches", and after a merge that moved 1,500 files this is the class most likely to be live. Only
   tokens that look like repository paths are considered, and a token is accepted if any tracked file
   ends with it, so `engine-architecture.md` written without its directory is not a finding.
2. **A state-shaped count that contradicts the tree** - the rules skill's own example, a document
   claiming a test count the baseline disagrees with. Counts are read from the files that own them
   (test_baseline.json, profiles.py, the hook) rather than from a second copy.
3. **A document that says it is superseded while sitting in a live location.** The field's advice is to
   archive rather than update; this names the files that should move.

    python tools/release/check_currency.py            # report
    python tools/release/check_currency.py --strict   # exit 1 on any finding
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Directories whose contents are records rather than current claims: a research note or an ADR states
# what was true when it was written, and rewriting those is how a record stops being one. Upstream-facing
# reports and model cards are dated records of another tree; a vendored skill's references are its own
# (`.opencode/skills/desloppify` naming `docs/CLAUDE.md` is that project's layout, not this one's).
RECORD_DIRS = ("docs/research/", "docs/adr/", "docs/maintainer/", "docs/upstream-reports/",
               "model-cards/", ".opencode/skills/", "docs/features/")

SUPERSEDED = re.compile(
    r"\b(superseded|withdrawn|obsolete|no longer (?:exists|used|applies)|has been removed|"
    r"was removed|retired)\b", re.I)
LIVE_MARKER = re.compile(r"^\s*(?:#+\s*)?\*{0,2}(SUPERSEDED|WITHDRAWN|OBSOLETE|ARCHIVED)\b", re.I)

# a token that looks like a repository path: at least one slash, a known extension or directory shape
PATH_TOKEN = re.compile(r"`([A-Za-z0-9_][\w./-]*/[\w./-]+)`")
KNOWN_EXT = (".py", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".md", ".json", ".cmake", ".cmd", ".bat",
             ".txt", ".csv", ".yaml", ".yml", ".tsv", ".ps1", ".sh", ".cuh")

# Counts are only read from suite-shaped phrases. "20 tests" in a feature note is a statement about
# that feature, and a suite's own history of counts ("size moved from 135 to 174") is a record of
# transitions rather than a claim about today - the first version of this check flagged both.
COUNT_PATTERNS = [
    (re.compile(r"\bthe suite (?:is|holds|has)\s+(\d{2,4})\b", re.I), "tests"),
    (re.compile(r"\b(\d{2,4})\s+ctest\s+tests?\b", re.I), "tests"),
    (re.compile(r"\bsuite_size[\"':\s]+(\d{2,4})", re.I), "tests"),
    (re.compile(r"\bthe hook runs\s+(\d{1,3})\s+gates?\b", re.I), "gates"),
    (re.compile(r"\b(\d{1,2})\s+shipped lanes?\b", re.I), "lanes"),
]


def tracked(pattern: str = "*.md") -> list[Path]:
    out = subprocess.run(["git", "-C", str(REPO), "ls-files", pattern], capture_output=True,
                         text=True, encoding="utf-8", errors="replace", check=False)
    return [REPO / line for line in out.stdout.splitlines() if line.strip()]


# The repository's own top-level entries, filled in main() from git ls-files. A path token whose first
# segment is not one of these belongs to something else - an artifact's internal layout, another
# project's tree, or a skill's own directory - and is not this tree's to resolve.
TOP_LEVEL: set[str] = set()


def read(path: Path) -> str:
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8-sig", errors="replace")


def tree_truth() -> dict[str, int]:
    """The counts, read from the files that own them."""
    truth: dict[str, int] = {}
    baseline = json.loads(read(REPO / "tools" / "release" / "test_baseline.json") or "{}")
    truth["tests"] = baseline.get("suite_size", 0)
    hook = read(REPO / ".githooks" / "pre-commit")
    truth["gates"] = len(re.findall(r'^\s*gate\s+"', hook, re.M))
    profiles = read(REPO / "tools" / "release" / "profiles.py")
    truth["lanes"] = len(re.findall(r"^    dict\(file=", profiles, re.M))
    return truth


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--strict", action="store_true", help="exit 1 on any finding")
    args = parser.parse_args()

    truth = tree_truth()
    all_files = [line for line in subprocess.run(
        ["git", "-C", str(REPO), "ls-files"], capture_output=True, text=True, encoding="utf-8",
        errors="replace", check=False).stdout.splitlines() if line.strip()]
    TOP_LEVEL.update(f.split("/", 1)[0] for f in all_files)
    basenames = {f.rsplit("/", 1)[-1] for f in all_files}
    docs = tracked()

    missing_paths: list[tuple[str, str]] = []
    stale_counts: list[tuple[str, str, int, int]] = []
    superseded_live: list[tuple[str, str]] = []
    stale_headers: list[tuple[str, str]] = []

    for doc in docs:
        rel = doc.relative_to(REPO).as_posix()
        text = read(doc)
        # A record states what was true when it was written, and a path in one is correct for its date -
        # rewriting those stops them being records. docs/research also cites other projects by
        # construction, and their layouts are not this tree's to resolve. Both classes are excluded here
        # for the same reason the count check excludes them.
        if rel.startswith(RECORD_DIRS):
            continue
        for token in sorted(set(PATH_TOKEN.findall(text))):
            if token.startswith(("http", "www.")) or "\\" in token:
                continue
            if not token.endswith(KNOWN_EXT):
                continue
            if (REPO / token).exists():
                continue
            # resolved against the document's own directory: a skill naming knowledge/x.md means
            # .opencode/skills/<that skill>/knowledge/x.md, and calling that missing is noise
            if (doc.parent / token).exists():
                continue
            # the first segment must be a real top-level entry of this repository, or the token is
            # artifact-internal (models/..., text/tokenizer.json) or another project's layout
            first = token.split("/", 1)[0]
            if first not in TOP_LEVEL:
                continue
            if token.rsplit("/", 1)[-1] in basenames:
                continue  # named without its directory: resolvable, not a finding
            missing_paths.append((rel, token))
        if rel not in RECORD_DIRS and not rel.startswith(RECORD_DIRS):
            for pattern, kind in COUNT_PATTERNS:
                for match in pattern.finditer(text):
                    claimed = int(match.group(1))
                    actual = truth.get(kind, 0)
                    if actual and claimed != actual and abs(claimed - actual) > 0:
                        stale_counts.append((rel, match.group(0), claimed, actual))
        marker = LIVE_MARKER.search(text)
        if marker:
            superseded_live.append((rel, marker.group(1).upper()))
        header = re.search(r"current as of (\d{4}-\d{2}-\d{2})", text, re.I)
        if header:
            stale_headers.append((rel, header.group(1)))

    print(f"documents scanned: {len(docs)}")
    print("  tree truth: " + ", ".join(f"{k}={v}" for k, v in truth.items()))

    print(f"\n1. paths a document names that the tree does not have: {len(missing_paths)}")
    for rel, token in missing_paths[:40]:
        print(f"   {rel}  ->  {token}")
    if len(missing_paths) > 40:
        print(f"   ... and {len(missing_paths) - 40} more")

    print(f"\n2. counts that contradict the tree: {len(stale_counts)}")
    for rel, text, claimed, actual in stale_counts[:30]:
        print(f"   {rel}: says {claimed!r}, tree has {actual}")

    print(f"\n3. documents marked superseded in a live location: {len(superseded_live)}")
    for rel, mark in superseded_live[:30]:
        print(f"   {rel}  ({mark})")

    print(f"\n4. 'current as of' headers (a reminder, not a verdict): {len(stale_headers)}")
    for rel, date in sorted(stale_headers, key=lambda pair: pair[1])[:20]:
        print(f"   {date}  {rel}")

    findings = len(missing_paths) + len(stale_counts) + len(superseded_live)
    print(f"\n  findings (classes 1-3): {findings}")
    print("  Record directories are excluded from the count check by design: docs/research, docs/adr and"
          "\n  docs/maintainer state what was true when written, and rewriting those stops them being"
          "\n  records. A finding in class 1 there is still a path a reader would follow.")
    return 1 if (args.strict and findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())

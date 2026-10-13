"""Fold the doc tables that quote the lane figures, after fold_lane_figures.py has moved the table.

README.md and RELEASE_NOTES.md quote each lane's tok/s and acceptance, and check_profile_consistency
compares them against profiles.PROFILES. Once the table has been folded, the docs are stale; this prints
what each doc currently says next to what the table says, and (with --apply) replaces the numbers in the
figure rows - markdown table rows that quote the pair. Prose that names a launcher is left alone, because
a margin ("+7.0% against 0.8%") is not an acceptance figure and the first percentage on such a line is
not the one to rewrite.

The numbers come from profiles.PROFILES -- not from the gate's messages, which do not say which side of
the comparison they are printing. Dry by default; idempotent by construction, since a line whose figures
already match is rewritten with the same text. Run it after fold_lane_figures.py, then run both gates:
they are the verification, and check_profile_consistency is the one that reads these two documents.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "release"))
import profiles  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    table = {p["file"]: p for p in profiles.PROFILES if p.get("file")}
    changed_total = 0

    for doc in ("README.md", "RELEASE_NOTES.md"):
        path = REPO / doc
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        out, hits, skipped = [], 0, 0
        for line in lines:
            lane = next((f for f in table if f in line), None)
            if lane is None:
                out.append(line)
                continue
            # Only a figure row is rewritten - a markdown table row quoting the pair. Prose that names
            # a launcher can quote margins ("+7.0% against 0.8%"), and the first `\d+\.\d+%` on such a
            # line is not an acceptance figure. The dry run caught this tool about to rewrite one into
            # the other on 2026-10-13, before it had touched a doc. A skipped line is left alone, and
            # the gate then reports any figure the fold did not reach.
            if not (line.lstrip().startswith("|") and re.search(r"\d+\s*tok/s", line)
                    and re.search(r"\d+\.\d+%", line)):
                skipped += 1
                out.append(line)
                continue
            entry = table[lane]
            tok_new = f"{float(entry['tok']):.0f}"
            acc_new = f"{float(entry['acc'].rstrip('%')):.1f}%"
            tok_hits = re.findall(r"(\d+)\s*tok/s", line)
            acc_hits = re.findall(r"(\d+\.\d+)%", line)
            print(f"  {doc} [{lane}]")
            print(f"      doc:   tok={tok_hits} acc={acc_hits}")
            print(f"      table: tok={tok_new} acc={acc_new}")
            new = line
            if tok_hits:
                new = re.sub(r"\d+(?=\s*tok/s)", tok_new, new, count=1)
            if acc_hits:
                new = re.sub(r"\d+\.\d+(?=%)", acc_new.rstrip("%"), new, count=1)
            if new != line:
                hits += 1
            out.append(new)
        if skipped:
            print(f"  {doc}: {skipped} line(s) name a lane but are not figure rows - left alone")
        if args.apply and hits:
            path.write_text("".join(out), encoding="utf-8", newline="")
            print(f"  {doc}: {hits} line(s) updated")
        else:
            print(f"  {doc}: {hits} line(s) would change")
        changed_total += hits

    print(f"\ntotal lines to change: {changed_total}" + ("" if args.apply else "  (dry run)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

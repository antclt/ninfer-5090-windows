---
name: c-review
description: C/C++ security review whose coverage is checkable: enumerates every reviewable unit with tree-sitter, requires a ledger entry per unit, and reports the sites no phase ruled out. Use when auditing native C/C++ for memory corruption, integer overflow, races or platform-specific defects, or when a review's coverage itself has to be auditable rather than asserted.
---

# C/C++ security review, with the coverage gate

Vendored from Trail of Bits' `c-review` plugin (`github.com/trailofbits/skills`,
`plugins/c-review`, v2.0.3), CC-BY-SA 4.0 — the licence travels in `LICENSE`, and
`UPSTREAM-README.md` is theirs. Upstream drives its phases through Claude Code's `Workflow` tool,
which opencode does not have. This port keeps the substance — the unit enumeration, the ledger, the
phase order and the false-positive rule — and runs the orchestration with opencode's own subagents.
The scripts under `scripts/` are upstream's, unmodified; they are plain PEP 723 programs and need
`uv`, not the plugin runtime.

## The gate is the point

A review that reports "no findings" is indistinguishable from a review that did not look. The gate
makes the denominator explicit:

- `scripts/enumerate_units.py` parses the target with tree-sitter and writes every function, macro
  and type as a *unit*.
- A review is complete only when every unit has a ledger entry naming the phase that examined it and
  what that phase concluded.
- `scripts/check_ledger.py` diffs the two and names the units nothing ruled out.

Report that uncovered count *with* the findings. A run that cannot state its denominator has not
finished, and a finding without a `file:line` citation is not a finding.

## Running it here

1. **Enumerate.** `uv run --script scripts/enumerate_units.py --scope <dir> --out units.json`
2. **Review in phases**, one subagent per phase, in upstream's order:
   - **Detect** — per unit: memory safety, integer arithmetic, concurrency, resource lifetime.
   - **Sweep** — the class axis: one agent per bug class over every unit the earlier phase did not
     rule out. Skipped when nothing is left.
   - **Verify** — each candidate against the artifact, at the cited line. The `fp-check` skill
     installed beside this one is the false-positive pass; use it rather than re-deriving one.
   - **Report.** `uv run --script scripts/render_report.py` from the result document.
3. **Gate.** `uv run --script scripts/check_ledger.py --units units.json --ledger ledger.json`

## Where this fits in this repository

`cpp-cuda-review` is the project's own review skill and stays the entry point for a change: it knows
the ownership boundaries, the oracle contract and the traps this tree has already paid for. This skill
adds the one thing that one does not have — a checkable denominator — so it is for a *sweep* (a whole
subsystem, a release's divergence) rather than for a diff. The two are complements: run this to find
candidate sites and to prove the sweep covered them, then judge each candidate under
`cpp-cuda-review`'s contract checks.

## Rationalisations this rejects

- "The code is small, so I read it all." The gate counts units, not confidence.
- "No findings", without the denominator.
- A finding with no citation, or one that names a file but not a line.
- Reporting a suspected defect as real before the verify phase. Upstream's rule and this repo's are
  the same: attribute it, or say "unattributed" — and note that a device and an oracle disagreeing is
  at least as likely to be the harness as either side.

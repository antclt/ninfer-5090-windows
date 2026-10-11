# GPU and C/C++ review tooling: what exists, what is measured, what was adopted

**Surveyed 2026-10-10**, after a question that had no prior answer in this tree: is there a reviewer
worth adding, beyond `cpp-cuda-review`, for a C++/CUDA engine on one GPU?

Every claim below is labelled. `[measured]` is a measurement in a cited paper or in this session;
`[read in source]` is read from the tool's own code or documentation; `[reported]` is a third party's
statement that this session did not verify.

## 1. GPU kernel verifiers: the measured record is poor

The field's own evaluations do not support adopting a GPU verifier, and this is `[measured]` by the
authors of those tools:

- **GPUVerify** — proves race-freedom for OpenCL/CUDA by Hoare-logic reasoning. Faial's CAV'21
  evaluation reports **34.8 % correct, 2.6 % false alarms, 62.6 % unknown** over its benchmark set
  `[measured]` (Cogumbreiro et al., *Checking Data-Race Freedom of GPU Kernels, Compositionally*,
  CAV 2021). The same paper excludes ESBMC-GPU and Simulee because *"ESBMC-GPU did not find a single
  data-race in our benchmarks, while Simulee produced false alarms for every DRF-kernel given"*
  `[measured]`, and reports that **PUG** *"incorrectly identifies the example as DRF"* `[measured]`.
- **ESBMC-GPU** — context-bounded model checking; its own paper reports 85 % of chosen benchmarks
  verified `[measured]` (Monteiro et al., *JSCP* 2018), which does not reconcile with Faial's
  zero-detection result on a different set. Either way, neither number is a reason to add it here.

**What this means for this port:** a clean static-verifier report would not be evidence, and a
non-clean one would mostly be unknown. The cheap dynamic checks already in the tree — `racecheck`,
`memcheck`, `synccheck` via `tools/scripts/test_v3_compute_sanitizer.cmd` — are the right instruments
at this scope, and RESOLVE states their limit precisely `[measured]`: *"Compute Sanitizer runs on all
three and reports nothing, which is the answer we expect and also all it can offer: a clean race
report says neither that the kernel is deterministic nor that it computes the reference function."*

## 2. The one technique worth taking: delay and compare

**RESOLVE** (`arXiv 2610.05683`, 2026) validates GPU kernels by *delaying memory operations and
comparing the kernel's full output bitwise against an unperturbed run* — delaying a racing access
changes which of the two lands first, and a kernel whose output moves was not deterministic
`[read in source]`. **SuperCollider** (Stephenson et al., 2026) applies the same idea per access
through a compiler pass `[reported]`.

This port already has the cheap half: `docs/research/lane-coverage-2026-10-08.md` records the
cross-path determinism check (the same continuation served cold and through the shared-prefix path,
byte-identical) and the soak's counters. The half it does not have is at the *kernel* level — a
deliberate perturbation of scheduling or ordering followed by a bitwise comparison. That is a
candidate lever, not a decision: it needs a kernel where a race is plausible and a comparison that
can see it, and nothing in this release's findings pointed at one.

## 3. The agent-skill ecosystem: one adoption, and the ecosystem's own lesson

`[read in source]` Trail of Bits publishes `trailofbits/skills` (`c-review`, `fp-check`,
`modern-cpp`, `sharp-edges`, `differential-review`, `constant-time-analysis`, and others), with a
vetted subset in `trailofbits/skills-curated`. Its `c-review` is the only C/C++ reviewer found whose
*coverage* is itself checkable: `enumerate_units.py` parses the target with tree-sitter and
`check_ledger.py` diffs the unit list against the review's ledger, so a report carries the sites
nothing ruled out.

**What was adopted, and in what form:**

| skill | form | why |
|---|---|---|
| `fp-check` | as published | it fits opencode's tool set unchanged, and its purpose — a true/false-positive verdict with the evidence, before a finding is reported — is this repo's own rule |
| `c-review` | vendored, orchestration ported | upstream's body says *"make one `Workflow` call"*, and opencode has no such tool. The scripts are unmodified; the phases run as subagents. The coverage gate is the part worth having and it is the part that survives |

Both are CC-BY-SA 4.0, so the licence and the upstream README travel with them, and `.opencode/` is
not staged into the release archive — the copyleft stays out of the shipped product.

**The ecosystem's own lesson** `[reported]`: a reviewer who installed and used more than 40 skills
concluded that *"fewer than ten survive contact with actual work"* (Towards AI, 2026-09). This tree
already curates and dates rejections; the survey's conclusion is therefore two skills, not ten.

## Sources

- Cogumbreiro, C. et al. *Checking Data-Race Freedom of GPU Kernels, Compositionally.* CAV 2021 —
  the GPUVerify/ESBMC-GPU/PUG/Simulee comparison. `[measured]`
- Monteiro, F. R. et al. *ESBMC-GPU: a context-bounded model checking tool to verify CUDA programs.*
  JSCP 2018. `[measured]`
- *RESOLVE: Language-Agnostic Validation of GPU Kernels Through Testing, Reduction, and Proof.*
  arXiv 2610.05683, 2026 — delay-and-compare, and the Compute Sanitizer limit. `[read in source]`
- `github.com/trailofbits/skills` — plugin inventory, licences and the `c-review` scripts.
  `[read in source]`
- *I installed 40 Claude skills; three survived.* Towards AI, 2026-09. `[reported]`

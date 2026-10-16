# AGENTS.md

This is a Windows port of [Neroued/ninfer](https://github.com/Neroued/ninfer). The product contract
below is upstream's, kept verbatim. Everything after **Windows port** is this port's own layer.

## Product and architecture

NInfer is a C++/CUDA inference engine for maximum single-GPU performance, targeting `sm_120a`
on NVIDIA GeForce RTX 5090. Choose designs for functional and numerical correctness, clear
ownership, and the performance goal within the requested scope.

It implements `Qwen3_5ForCausalLM`, `Qwen3_5MoeForCausalLM`, and eager Text/Vision/MTP for
`Qwen4ExpForCausalLM`. Checkpoints and recipe combinations
of existing representations use the same architecture, binding and execution path; do not add
checkpoint-specific execution registration. Generation uses one GPU, one resident model, and
one to eight resident execution lanes fixed at startup.

V3 `.ninfer` is the only C++ product artifact. Generation, offline CausalScoring, CLI, serving, and
inference benchmarks use the public Engine. There is no Python model-inference route or
installed/exported C++ SDK.

This is a local, single-owner project with trusted local models, artifacts, and workflows.
Do not derive requirements from another deployment or trust model.

Model data is immutable; every Program owns its mutable state and device allocations. The loader
validates, uploads, and binds the stored representation; actual Parameters and Ops determine
execution support, without a whole-artifact capability registry.
Runtime owns execution and publication policy; product/serving own input acquisition and protocol
translation. Model code does not acquire media or own transport.

Keep architecture, binding, and execution explicit. Without a product requirement, do not introduce
generic model graphs, family base classes, plugin discovery, string-driven execution, hidden device
allocation, or runtime weight repacking. New mathematical architectures, execution platforms,
large-scale continuous batching, or priority/QoS require an explicit product change. When a task
changes product or architecture, update affected contracts and implementation together.

## Change policy

Project-owned contracts do not preserve backward compatibility. Replace behavior completely:
remove superseded aliases, fallbacks, transition branches, and their tests within the affected contract.
Advertised OpenAI and Anthropic protocols are external contracts; update affected schema tests
and serving documentation together.
Update stable requirements in their existing authoritative document; maintain one current authority.
Use Conventional Commit subjects with concise lowercase types for every commit.

## Verification and reporting

Before changing Ops, numerical or state semantics, read
[Op development](docs/maintainer/op-development.md). Qualify affected production routes directly
against the contract's independent mathematical oracle or specified exact reference. Kernel parity
and plausible model output do not establish mathematical correctness.

Measure performance at the claimed scope. An Op microbenchmark establishes an Op result,
not an end-to-end improvement. For comparisons, establish the baseline, workloads, metrics,
aggregation, and acceptance criteria before evaluating results.
Distinguish new capability, fallback replacement, and improvement to an optimized implementation.
Report the baseline, hardware/toolchain, workloads or commands, run conditions, metrics, coverage,
result distribution, worst changes, and exceptions. Include small and unexplained regressions;
do not dismiss slowdowns as noise without evidence.

### Before proposing anything

Do the research and re-check the work *first*, then propose.

- **A number about cost, time or memory is measured, or labelled an estimate.** Compute what a
  design implies, then measure it. Three separate cost claims were falsified by arithmetic
  standing in for the figure.
- **A claim about another project is read to its mechanism, not taken from a headline** — and a
  feature request's premise can be wrong. `llama.cpp #23184` asked for chaining that already existed
  in the tree at the commit before it was filed.
- **A claim about this tree is re-read, not recalled.** Name the file and the line.
- **Search before proposing an approach at all, and say which sources were used.** If nobody has
  done it, that absence is the finding and is worth stating as such.
- **A test result is only evidence if the setup could have passed for the right reason.** Confirm
  the case under test reached the branch it names. Twice in one session a harness omitted an argument
  — a model path, then a prompt — so every "refusal" was for the wrong reason.
- **Say which of the three it is**: measured here, read in the source, or reported by a third party.
- **A control that agrees for a structural reason is not a control.** Name what would have to be
  false for it to fail. At `top_k = 1` the target's support is one token, so `p >= q` holds for every
  draft — which is why the sparse accept cases agreeing there said nothing about the accept rule.
- **Attribute a disagreement before reporting it.** When a device and an oracle disagree, the third
  possibility is the harness. "Unattributed" is a reportable result; picking the easier side is not.
- **A measurement that decides a shipped setting has to outlive the session.** A decision resting on
  a sweep run from a scratch script is unverifiable the moment that script is gone.

### Reporting and completion

Selective reporting and evidence gaming are prohibited, even when every disclosed statement is
individually true. Put favourable and unfavourable findings in the final reply itself, including
regressions, costs, rejected approaches, failures subsequently fixed, unresolved issues, and
verification gaps. Small or unexplained adverse results must remain visible. Apply the same evidence
standard to gains and regressions, and never change a baseline or an acceptance criterion to
manufacture a favourable conclusion. Once the deliverable and its acceptance conditions are
satisfied, stop and report.

## Reference navigation

Read the authority relevant to the current decision; this is not a mandatory reading list.

| Decision | Entry point |
|---|---|
| Product capabilities and exact commands | `README.md`, executable `--help`; `docs/cli.md`, `docs/serving.md`, `docs/perplexity.md` |
| Execution, model/runtime ownership, scheduling, graphs | `docs/maintainer/engine-architecture.md`; `docs/maintainer/project-map.md` |
| Context resources, checkpoints, replicas; physical KV | `docs/maintainer/resource-scheduling-and-context-cache.md`; `docs/maintainer/paged-kv-cache.md` — both written in Chinese upstream, so read `docs/maintainer/README.md` first |
| Artifact, layout, codec, conversion, or model mathematics | model/artifact references linked from `docs/README.md` |
| A maintainer document written in Chinese upstream | `docs/maintainer/README.md` indexes all seven and says why they are not translated |
| Op contracts, implementation ownership, numerical/performance qualification | `docs/maintainer/op-development.md` |
| The shipped lanes: artifact, route, draft number, KV dtype, context | `tools/release/profiles.py` |
| Test/benchmark commands and published performance | `tests/README.md`, `bench/README.md`, `docs/performance.md` |
| In-tree C++ interface | `include/ninfer/engine.h`, `include/ninfer/types.h` |

## Windows port

### Local environment

MSVC 14.51, CUDA 13.3. Tooling Python is `C:\vllm-env\Scripts\python.exe`; use it explicitly.
Build with `cmake --build <build-dir> -j`, adjusting parallelism only on real resource pressure.

There are **four** build trees, all gitignored: `build/` for the apps and bench, `build-test/` for
the suite the release gate runs, `build-asan/` for the host-only sanitizer tests, and `build-bench/`
for the standalone bench binaries.

The shipped surface is governed by `tools/release/profiles.py`. There are eight lanes: four
artifacts × {MTP, DFlash2}, each with its own measured draft number, KV dtype and `--lm-head-draft`
setting, all at 262,144 context with vision. The generated launchers at the repository root render
from that module, so profile values come from there and nowhere else.

Select model artifacts by explicit path — never glob order, modification time, or unqualified
"latest". Source checkpoints and large artifacts are prerequisites; download or regenerate them only
when that work is in scope.

### Tooling

`.githooks/pre-commit` is the fast gate set — enable once with `git config core.hooksPath .githooks`.
Read the hook as the authority on what it runs; a count of its gates in prose goes stale.

| Situation | Tool |
|---|---|
| a C++ or upstream change reaching the suite | `tools/scripts/test_v3.cmd`, then `tools/release/check_test_baseline.py` **with `NINFER_TEST_ARTIFACT` set** — without it the real-model tests skip and the gate fails on missing coverage rather than on a regression |
| anything order- or state-dependent | the suite recipe passes `--schedule-random`; run it twice before believing a fixed order |
| a device-side memory, race or synchronisation question | `tools/scripts/test_v3_compute_sanitizer.cmd` — memcheck on a small subset, and `racecheck`/`initcheck`/`synccheck` are the wider set |
| a host-side lifetime question | `tools/scripts/test_v3_asan.cmd` — ASan cannot instrument device code, which is why the two recipes are separate |
| a check that passes here and fails in CI | `tools/scripts/verify_as_ci.cmd` **first**, before forming any hypothesis |
| a change to a workflow file | **dispatch it** — the runner's shell is part of the change, and a `run:` block on Windows is PowerShell, so cmd syntax needs `shell: cmd` |
| a declared type nothing can reach | `check_dead_types.py` — a `struct`/`class`/`enum` named exactly once, at its declaration. Found the copy-drafting withdrawal |
| a name a tool still treats as current after the tree stopped defining it | `check_retired_names.py` — a criterion naming a retired symbol does not error; it answers a different question |
| copy-paste, and whether it is *new* | `check_duplication.py` — jscpd against `.jscpd-baseline.json`; it fails only past 5 *new* clones, and exit 2 means jscpd analysed nothing, which is not a pass |
| a lane's phase split, and whether its prefix cache engaged | `tools/bench/report_serve_phases.py` — reads `request-log-jsonl`, never a wall clock. `prefix_cache_hit_tokens` is the field that says whether a cache effect exists |
| a request-latency regression in the serve path | `tools/bench/probe_prefill_floor.py` — prints phase fields **by name** from the request log's own schema. It is how the 15.6 ms quanta were found |
| speculative acceptance or decode between artifacts | `tools/bench/realtext_acceptance.py` — real text at a lane's own flags, greedy so the arms are comparable |
| Python tooling, before committing it | `ruff check tools tests` and `mypy tools/release tools/convert tests` — both enforced by the hook |
| a lingering suspicion of flakiness | `ctest --test-dir build-test --repeat until-fail:5` |
| a file you are about to remove | `tools/scripts/reference_audit.py <path>` — a filename search is not a reference search |
| what upstream already decided, or what is proposed and unmerged | the upstream tracker, **both halves**: `gh issue list --repo Neroued/ninfer --search <term>` **and** `gh pr list --repo Neroued/ninfer --state open` |
| a kernel's performance | `ncu` / Nsight, records under `profiles/ncu/` and `profiles/nsys/`. `profiles/` is gitignored, so the tree cannot show them |

`.codegraph/` is indexed: a code question ("where is X", "who calls X", "how does X work") goes to
`codegraph_explore` before `grep` or `Read`. Manual search is for what it does not index — docs,
configs, logs.

### Skills

Skills are **project-local and untracked**: `.opencode/` is in `.gitignore`, and upstream has no
`.opencode` at all. Nothing in this repository reads them, and no gate checks them. On a machine
that has them, `.opencode/skills/` holds 35 C++/CUDA and language-agnostic skills — `cpp-cuda-review`
is this repo's review skill, with `cuda-debugging`, `ncu-report`, `sanitizers`, `address-sanitizer`,
`c-review`, `fp-check` and `how` beside it. A clone of this repository has none of them and needs
none: they shape an agent's method, not the engine. Restore them with
`git checkout pre-trim-2026-10-13 -- .opencode`.

| Situation | Skill, where present |
|---|---|
| a C++/CUDA change on its way in | `cpp-cuda-review` |
| "how does X work", or an unfamiliar subsystem | `how` |
| a whole subsystem needing a review whose coverage is checkable | `c-review` |
| a suspected bug to confirm or discard **before** reporting | `fp-check` |
| a device-side memory, race or sync question | `cuda-debugging`; `sanitizers` and `address-sanitizer` for the host side |
| a kernel's performance | `ncu-report` — records under `profiles/ncu/` and `profiles/nsys/`, also gitignored |
| before claiming a fix works | `principle-prove-it-works` |
| `AGENTS.md` or an ADR may have gone stale | `rules-check-drift` |

### Rules earned by failures

Each is a rule, not a story. The parenthetical is the failure that produced it.

**Verification and harnesses**

1. Run a recipe *through* the recipe. `ctest --test-dir build-asan -R <broad regex>` pulls in device
   tests that ASan cannot instrument and that hang — one such run burned fifty minutes. The recipe's
   scope is part of the recipe. (It said "six", then "five", then "three", then three of four named.)
2. Validate the harness before believing its numbers. Four times a measurement returned plausible
   zeros — a loop variable colliding with the read-only `$Host`, a log path the launcher named
   differently from the loop tag, a run too short to flush, a server that never started. A zero that
   cannot be explained is not a result. The same applies to a failure: check the harness handed it
   what you think before diagnosing.
3. Redirect a long-running command to a file; a truncating filter kills it. `Select-Object -First N`
   closes the pipeline and terminates the command, and `| Out-Null` swallows the success you were
   checking.
4. Parse a tool's tabular output by header name, never by column position. A sweep read
   `workspace_peak_bytes` as the round count and printed a clean table of `rounds=107360256`.
5. Validate the harness on a *control* first. A field-by-field comparison needs its `x` against `x`
   printed first, so a broken instrument reads as broken rather than as a result.
6. Give every step of a long chain its own log file and print its start line before running it. A
   driver that buffers a step's output loses it when the driver dies, and a chain that reports `rc=0`
   per step must always have a file behind the claim. **Long runs launch detached**:
   `tools/scripts/run_detached.py --log <file> -- <command>` puts the child in no shell's tree, which
   is what survives the hourly service eviction that kills every shell this location hosts.

**Measurement**

7. A cost you can compute is not a cost you have measured. Three separate cost claims in one session
   were falsified by arithmetic standing in for the figure: a "roughly double" work claim that
   measured +5.7 %, a ~480 MiB graph allowance that measured 288 MiB and did not vary with width.
8. A projection is not a measurement, and the first live run will correct it. `prepared` and TTFT
   were projected twice from a render benchmark and were wrong both times, once by 2×.
9. A bench that does not reproduce the process can refute a true hypothesis. The same renderer measured
   5.99 ms in a fresh process and 48.3 ms in a long-lived server: a stand-in must reproduce the
   *conditions* of the thing it replaces, not only its input.
10. Compare alternatives by **interleaving** them. This card's clocks are not pinned, so decode drifts
    up to ~9 % between windows; measure A then B and you have measured the window.
11. This card's early runs are unreliable and the spread is measurable — 5.8 % and 7.6 % in the first
    half of a session against 0.0 % and 0.3 % in the second. Any effect below ~8 % needs more
    repetitions than a recipe's default. Say which half of a session a number came from.
12. One GPU job at a time, and never a measurement beside one. A number taken while something else holds
    the card is a different number, not a smaller one. Give automated GPU work a concurrency lock.
13. Say which process a number came from. Two lanes with identical request bodies produced different
    text for a reason in neither: they differed in cache capacity, which changes the plan's prefill.
14. A figure carries its configuration, and a series carries the axis it was measured along. A startup
    line reading "pinning host state | 1.46 GiB" was recorded as the 16-slot cost; it is the 8-slot
    figure. Before extrapolating, confirm the thing being extrapolated is present in the destination.
15. A numerics change re-states every recorded figure, and the tree will not tell you. Restoring the
    accurate `silu` moved corpus perplexity 1.606336 → 1.609905 and silently staled every ADR quoting
    a score. Sweep the recorded measurements yourself after changing an Op's arithmetic.

**Editing and tooling**

16. Never inline a script through PowerShell. Quotes inside quotes break the argument splitting; it
    happened five times in one session and a quoted path handed to `&` is still an inline script. Write
    a file first, and check the output's *size* — a command can succeed while writing an empty file.
17. Never edit a file from memory. Anchor on a unique substring you have read; five edits failed in one
    session on a reconstructed anchor. Scope a probe by reading the block it lives in.
18. A scripted edit needs an expected occurrence count, and must be idempotent. Assert the count, or
    treat "already applied" as success.
19. Never rewrite code with a regex that crosses a line boundary — the closing pattern stops at the
    first `)` and truncates every call site.
20. A scripted edit's verification must use the file's **real consumer**, not a convenient parser.
    PowerShell 5.1's `Set-Content -Encoding utf8` writes a BOM and `Get-Content` decodes with the
    console codepage, so a round trip re-encodes every byte ≥ 0x80 and 181 lines were lost while every
    other gate passed. Write through `[System.IO.File]::WriteAllText(..., UTF8Encoding($false))`.
21. The checkout must not decide the bytes a comparison sees. `.gitattributes` pins text to LF except
    `*.bat`/`*.cmd`; a generator that digests a file must write platform-independently. A test that
    hashes a file certifies the checkout, not the file — 18 of 26 TTFT digests described bytes no LF
    checkout produces, and the gate passed here and failed everywhere else.
22. Do not guard with string presence over prose. Assert the specific call site instead.
23. Instrument the branch before changing the candidate. An edit answers "did this change the outcome?",
    never "which branch ran?". The permanent form is a counter, because the context cache cannot log.
24. A helper that returns a view over its own temporary is a silent corruption generator —
    `std::string_view v = trim(render_content(...))` dangles and the text stays the right length.
25. Compare structures field by field, not by their largest member. Two renderers produced identical
    bytes with different cache, message and execution boundaries and passed every text-only check.
26. Batch what goes into `include/ninfer/types.h`, because the device tree reaches it — two fields
    landed separately and rebuilt every CUDA object twice, ~20 minutes each.
27. Batch edits, not the response. The cap is on the whole response; anchor `edit` on a short unique
    string and verify after every edit, because a truncated tool call fails *silently*.
28. Do not ration work against an assumed context limit. A session is not short, and compaction exists.

**Process and hygiene**

29. The build and the running product contend for the same files. Stop the server before rebuilding
    (`LNK1104` reads as a broken build rather than a held file), and verify with `Get-Process` **and**
    a port check in the same command as the thing that depends on them.
30. `git stash` on an in-progress merge drops `MERGE_HEAD`, losing the shape and breaking
    `check_port_delta.py`. Restore the state file, or commit the resolution.
31. Reference this tree relatively, or a path that exists will read as missing — absolute workspace
    paths stopped resolving mid-session while relative ones kept working. Derive paths from `__file__`.
32. Confirm a commit landed: `git commit -F <file>` for anything longer than a line, then read
    `git log -1`. A failed attempt leaves the index staged, so commit with explicit paths or reset the
    index between attempts.
33. One branch, one worktree, one checkout. Work lands on `dev`; `main` is only ever a squashed
    release cut. Three remotes exist and only `upstream` (`Neroued/ninfer`) is upstream; ask of each
    branch by name, because "nothing returned" is how both mistakes happened.
34. Integrate upstream by merging into `dev`, never by parking commits on a tracking branch.
35. Publish every version you build, in order, or do not build it — a gap in the release list reads as
    a withdrawn release. Tag a release while on the release branch: `git tag -f` tags HEAD.
36. Check before you package, not after. A package built before its review has to be re-cut and
    re-packaged.
37. Revert every diagnostic probe before committing, and never commit its rationale. A threshold
    changed to test a hypothesis was committed with a comment asserting it fixed something the
    measurement had already disproved. A claim in a comment is a claim.
38. Prove an extraction by byte-identity: regenerating output that must not change rules out any
    change at all.
39. An assertion that cannot fail is worse than no assertion. Confirm it would have failed without the
    thing it names.
40. A failing expectation is as much a defect as a failing tool. Write expectations from the format's
    definition, not from what the code happened to produce.
41. A superseded design's surface is removed in the same wave — archive, option fields, counters, stats,
    tests, flags and document together. Anything kept "for now" becomes advertised surface the product
    cannot honour.
42. Revert or land a measurement that decides a shipped setting, and keep its record. A sweep run from
    a scratch script is unverifiable the moment the script is gone.

**Python and the artifact layer**

43. The Python directory is not the C++ one; read the schema you are coding against. The C++ structs
    are a richer form of the same data, not the same data.
44. A binding is usually a slice of a fused parent, and NVFP4 cannot decode a slice — 409 of 1,513
    bindings cover part of a larger packed parent. Decode the parent once and take the slice.
45. An identifier that encodes a format hint is not interchangeable with one that does not. A call that
    *succeeds* is not evidence the hint was right: a store is accepted only once it can produce values.
46. Compute a tool's memory cost before running it, and refuse the run when it is over budget. A
    per-element cost times a count is a number, and the number comes before the command — one
    unbounded verifier exhausted a 47.8 GiB machine.
47. A memory guard that reports zero is a broken guard and must fail rather than pass. Read the child's
    own `PeakWorkingSetSize` through ctypes with explicit `argtypes`; polling from the parent misses a
    fast child.
48. A subprocess gets an explicit text encoding, and non-ASCII printed output goes through an ASCII
    replace helper — otherwise stdout aborts on the first em dash.

**Per-domain and serving**

49. A rejected request's reason is in the engine's log; the client surfaces only `HTTP 400` with no
    body. Start a detached lane with its output captured before diagnosing one.
50. A cache that serves an extension but not a repetition has a gap, not a design. When a cache has a
    hit counter, publish it; when it has a hit *condition*, ask which common cases fall outside it.
51. Phase fields mean only what their definition covers. `prepare_seconds` spans the whole frontend —
    render, tokenize and layout — so a projection that subtracts one term from it is a claim about the
    others, and it had never been checked against them.

**Remote and provenance**

52. Read the access level off the Hub API, not off an impression: `https://huggingface.co/api/models/<repo>`
    returns `gated` and `private` directly, and both read `false`.
53. Read primary sources before reasoning from this tree. Upstream's converter, loader and maintainer
    notes are authoritative; the port is not. Absence from `src/` is not licence to delete an upstream
    file — `tools/convert/qwen4_exp.py` has zero C++ references and is still upstream's, wired into
    `__main__.py` and named by `official_recipes.py`.
54. Verify a claim about a file before asserting it, and scope a negative claim's search. "This layer
    cannot log" came from a grep limited to `context_cache/`; a truncated `Select-Object -First 20` cut
    a match list short and an ADR was written on the wrong premise.
55. A document that records a file as kept is a reference to it. Retention language near a mention
    ("renamed rather than deleted", "moved to") is a live reference, and a name search will miss it.
56. A transient file lock can mark a measurement failed while its data is intact — read the score out of
    the backend's own file, and read nothing inside a directory while it is being written.

**Text and identity**

57. Re-verify a provenance claim against a baseline. A file byte-identical across two repositories is
    usually upstream's own, unchanged by either; a shared-line ratio means nothing until you measure
    the baseline for unrelated files in the same project.
58. A cache that never reports a counter cannot be debugged, and a driver that dies takes its buffer
    with it — see rule 6.

## References

Read [Engine architecture](docs/maintainer/engine-architecture.md) before changing execution or
ownership. [README](README.md) and executable `--help` define capabilities and exact commands.
The [documentation map](docs/README.md) routes to detailed contracts;
[Tests](tests/README.md) and [Benchmarks](bench/README.md) own their commands and execution details.

## Two rules for correcting a document

**A document's invariant that the code breaks is a suspected bug in the code.** Do not edit the
document to match the code; that buries the defect under a lie that now looks deliberate. Report it
and change neither side until it is settled.

**Never confirm a document claim without the evidence in hand, and say what you checked.** "Looks
right" and "I read it earlier" are not acks. Open the cited line, name what you found there, and if
the claim is behavioural rather than textual, run it.
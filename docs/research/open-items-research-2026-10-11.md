# The open items, researched — 2026-10-11

Written after v1.2.0 was cut with item 1 disclosed, on the directive that every open item should get a
fix or a solution before shipping, that upstream's issues and pull requests are not this port's to fix
but may contain something usable, and that the work should use the indexed tool, code mode, skills and
online research.

**Evidence classes are labelled per claim**: *measured here*, *read in source* (file:line), or
*reported by a third party* (URL). Nothing in this note changed a lane or an artifact; the two things
that were *measured* here are marked as such and are reproducible from the commands given.

---

## 1. The A16 SwiGLU decomposition above T=16 — the fix is an instantiation, not a kernel

**The fused route already exists, and it is capacity-limited by a template instance.** *Read in source*:
`src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_small_t.cu` applies `Nvfp4SwiGluEpilogue` — the same
fused epilogue the A4 routes use — through `Nvfp4SlicedInstance<8, 8, 2>` for T ≤ 8 and
`<16, 8, 2>` for T ≤ 16. Above that, `nvfp4_linear_swiglu_plan.cpp:40` routes every A16/AllowA8 call
through `LinearA16Post`, which materializes gate/up in BF16 and applies `silu_mul` to the buffer.

So extending the family above T=16 is a larger-capacity instance, a dispatch arm, and the oracle cases
`tests/ops/linear_swiglu/test_nvfp4.cpp:81` currently stops short of. The estimate this note replaces
was "a day of kernel work"; that estimate was wrong in the expensive direction.

**Resolved 2026-10-11, and the resolution is wider than the instance this section proposed.** The
landed fix adds `nvfp4_linear_swiglu_matrix.cu` — a sliced instance to T=32 and then the A16 MMA
schedules with `SwiGluRowMajorMmaEpilogue`, the pairing the FP8 and Q8 A16 families already use — and
**deletes** the materializing fallback and its workspace, so every extent is fused. The oracle runs 19
extents (`1,2,4,5,8,9,16,17,24,25,32,33,64,65,96,97,128,129,1024`) and passes at all of them, against
the same 6.3e-3 tail bound that failed at 8.1e-3 before the change; the suite is 154/156 with the two
baselined failures and the baseline gate passes. The `dflash2-nvfp4-cluster-merge-findings.md` note that
called this "the obvious follow-up ... not written because the alternative is new numerics with no
oracle of its own" is answered by the oracle this item's own reproduction produced.

**What the field does, read in its own source: it materializes the intermediate.** *Reported*: TensorRT-LLM — "The GEMM + SwiGLU
fusion in Gated-MLP combines two Matmul operations and one SwiGLU operation into a single kernel.
Currently this is only supported for FP8"
(<https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/fp8-quantization.html>);
epilogue fusion "eliminates intermediate global memory reads and writes by applying the activation
function directly within the GEMM epilogue" (fal.ai, via search); CUTLASS does it with an epilogue
visitor tree; and tile-level SwiGLU fusion is current work (arXiv 2607.02521) — so fusion exists in the
toolkits and in FP8 serving, and the question this item turns on is what the *shipped* engines do with
quantized weights. The reason the
nonlinearity is not run *in* the quantized precision is a hardware one — tensor cores for the GEMM,
higher precision for the activation.

**Three primaries, read rather than summarised.** vLLM applies
`SiluAndMul` as a **separate kernel** over the materialized gate/up tensor —
`vllm/model_executor/layers/activation.py`, `class SiluAndMul(CustomOp)` with
`self.op = torch.ops._C.silu_and_mul` and `forward_native` written as `F.silu(x[..., :d]) * x[..., d:]`,
i.e. a BF16/FP16 round trip between the projection and the activation, which is the same shape as this
route's. llama.cpp fuses only the **activation pair** — `ggml_glu`, `ggml_swiglu`, `ggml_swiglu_swapped`
and `GGML_GLU_OP_SWIGLU` in `ggml/include/ggml.h` — consuming an already-materialized gate/up tensor.
And TensorRT-LLM's GEMM + SwiGLU fusion is documented as FP8-only. **So no engine in this survey fuses
the activation into a quantized GEMM's epilogue, and this route's T ≤ 16 family is ahead of the field
rather than behind it.** The reason to extend it is internal consistency with this port's own A16 tail
bound — the reason the item's own text gives for not widening the criterion.

**A claim in this note's first version was wrong, and the check that refutes it is the useful part.** It
said the fallback calls `silu_mul` on non-contiguous slices and therefore takes
`silu_and_mul_strided_input_kernel`, a per-element loop with two divisions each. It does not.
`nvfp4_linear_swiglu_plan.cpp:96-104` passes `projected.slice(0, 0, 17408)` and
`projected.slice(0, 17408, 17408)` — **dim-0 slices** of a `[34816, T]` BF16 buffer allocated at 256-byte
alignment — and `can_use_dim0_split_fast_path` (`src/ops/launcher/silu_and_mul.cu:17-33`) accepts
exactly that shape: rank-2, even `ne[0]`, `nb[0] == sizeof(bf16)` (the row direction tight within the
slice) and `nb[1] % sizeof(bf16x2) == 0`. The launcher therefore takes
`silu_and_mul_dim0_split_kernel` — the `bf162`-vectorized path, with no per-element index arithmetic. The
decomposition's real cost is the BF16 round trip through that buffer, which the route's own comment
already states: "It materializes a [34816,T] BF16 buffer, so it trades that traffic for serving the
extent at all." So there is **one** defect here — the rounding against the profile's own tail bound — and
not two.

**A neighbouring measurement worth reading before adopting anything.** *Reported*: upstream #364/#365 —
the NVFP4 W4A4 `linear_add` at `[5120,17408]` launches `M32N64` at every A4 width up to 64 tokens,
"80 CTAs" on a 170-SM card, so "a speculative verify therefore streams the weights from fewer than half
the SMs". The fix is a halved tile with a deeper pipeline, **bit-identical**, −8..−13 % at T=8..32 —
and a commenter reports it *slower* end-to-end without a thorough test. That is this repository's own
red flag about op microbenchmarks: the op result is not the end-to-end result, and the end-to-end claim
here would have to be measured on a shipped lane.

## 2. `ninfer_topk_logprobs_test` order-dependence — a named producer, demonstrated; the message was never captured

**The op is deterministic by construction.** *Read in source*: `src/ops/kernel/topk_logprobs.cuh:5-8`
— "one independent CTA per column", "no device allocation", all scratch static; and `:98-110` documents
a total order ("Descending by value, ascending by row on equal values ... this is a total order and the
sort is deterministic"). The launcher is one kernel launch and a `cudaGetLastError`
(`src/ops/launcher/topk_logprobs.cu:34-42`).

**There is a deliberate red path for this test, and the suite recipe runs it.** *Read in source*: the
tie-break carries a mutation seam (`topk_logprobs.cuh:102-108`) whose comment says it is "the seam a
perturbation test flips"; the launcher reads `NINFER_OP_MUTATION` once per launch in the test tree only
(`:22-30`); `tools/release/check_test_mutation.py:212-214` pops the variable from a copied environment
and sets it per invocation; `tests/release/test_test_mutation.py:13` asserts that setting it "really
does turn the suite red"; and `tools/scripts/test_v3.cmd:71` runs that gate as part of the suite recipe.

**Measured here, both directions** — the rule that an instrument never seen to fail is not evidence:

```
build-test\tests\ninfer_topk_logprobs_test.exe                          -> OK topk_logprobs        rc=0
$env:NINFER_OP_MUTATION='1'; ...\infer_topk_logprobs_test.exe           -> FAIL topk_logprobs      rc=1
    topk_logprobs one distinct value: slot 7 index 32, expected 7
    topk_logprobs few distinct values: slot 0 index 179, expected 2
```

**Measured here, the suite's own concurrency shape**:
`ctest --test-dir build-test -R topk --repeat until-fail:25 -j 4` → **100 % passed, 60 s** (both top-k
op tests, random order, four-way parallel). No recurrence.

**The failing run's message is not recorded anywhere.** `git grep -i topk -- docs` returns the item
itself (`docs/active-work.md:3025`) and unrelated matches; the retained
`build-test/Testing/Temporary/gate-ctest.log` is a later green run (`33/156 Test #104:
ninfer_topk_logprobs_test ... Passed 1.41 sec`).

**Disposition**: the observed signature — one failure inside the suite, unreproducible alone — has a
*deliberate* producer that runs inside the same recipe and fails this exact test with index-ordering
messages. That is the leading candidate, not a proof. **The falsifier is the message**: if a recurrence
fails on anything other than the tie-break (a numeric mismatch, a crash, a timeout), the seam is
excluded and the remaining candidate is device-level nondeterminism, which the construction argues
against but does not exclude. A recurrence *is* captured now: the recipe runs ctest with
`--output-on-failure` and the gate keeps its log.

## 3. The fork's published calibration profile — same formula, same corpus, different realization

**The formula is identical.** *Read in source*: `tools/convert/calibration.py:23` —
`d_x = binary32(2688 / amax_site)`, with `FULL_RANGE = 2688.0` and the comment `# 2688 = 6 * 448, the
A4 block-scale orientation`; `tools/convert/quantization/nvfp4.py:30` — `FULL_RANGE = 2688.0  # 6 *
448`. The fork's script, `out/reference_calibrate_nvfp4full.py:40,453-456` (UTF-16, read with a decoder
rather than `Get-Content`): `FULL_RANGE = 2688.0`, `divisor_for` returning
`float(torch.tensor(FULL_RANGE / amax).item())`, and a docstring line "maximum absolute represented BF16
activation over the complete fixed corpus". Identical.

**The corpus is identical.** *Read*: `tools/convert/calibration_corpus.json` holds ten documents as
strings, and its first document — "The history of numerical weather prediction begins with Lewis Fry
Richardson's 1922 attempt..." — is byte-for-byte the fork's first `_CALIBRATION_DOCUMENTS` entry
(`:51-61`). So the corpus is not the explanation.

**What differs is the realization.** *Read in source*: the fork's script carries its own assignment —
`FULL_ATTENTION_LAYERS = tuple(range(3, 64, 4))`, `EARLY_ATTENTION_INPUT_LAYERS = (3, 7, 11, 15, 19,
23)`, `BF16_ATTENTION_OUTPUT_LAYERS = (3, 7)`, `BF16_GDN_OUTPUT_LAYERS = (4,)`,
`NVFP4_MLP_LAYERS_FROM_SOURCE = tuple(range(56))` (`:41-45`) — and it measures amax through *that*
model with forward pre-hooks on each site (`:371-390`, `:433-437`). A different assignment changes the
activations downstream of it, so the maxima differ site by site.

**Our own tree already measured that this matters.** *Read in source*:
`tools/convert/official_recipes.py` (the recipe docstring) — "A divisor is `2688 / max|activation|`,
and a maximum does not transfer between weight realizations: borrowing another quantization's stored
scales measured 2.2 % / 0.45 % worse on the fixed corpus, and the sites whose input is a derived
intermediate (MLP down) are the ones that move most."

**Residual**: the fork's published divisors were produced by *its* build. The one experiment that would
separate corpus from realization is re-running `reference_calibrate_nvfp4full.py` against this
checkpoint, which needs its torch/transformers environment. Not run.

## 4. The splice fix's missing counterexample — a construction, not a literature question

*Read in source*: spans below the window are compared, which declines a splice whose cached ids were
computed under different spans; no input was constructed that reaches the old path.

The reachable class is BPE's non-prefix-stability: appending a suffix can change the identity of the
*last* cached token (a merge across the boundary), so a cached id can be one the new tokenization would
not produce.

*Reported, and it names both the problem and a better fix*: **Incremental BPE Tokenization** (arXiv
2605.30813; OpenReview `ZbWgrDzCQo`) "incrementally maintains BPE tokenization results for every prefix
of the input text, implementing the standard BPE merge procedure", and "provides native support for
full-prefix tasks (e.g., FIM, token healing) by efficiently providing tokenizations of all prefixes". So
the conservative span comparison is a *correct* answer to a documented problem, and the published
alternative — maintaining the merge state per prefix — is the design that would *recover* the splices
this fix declines. That is an implementation to aim at rather than a defect to repair, and the
counterexample is still worth constructing, because the span check has never been *seen* to decline a
splice.

## 5. The cache-hitting race — upstream is replacing the subsystem

*Reported*: upstream PR #335 "replace the checkpoint catalog with a hybrid prefix cache" (121 files),
whose body states the constraint our own docs describe — Gated DeltaNet layers "carry a recurrent state
that cannot be rolled back, so a prompt prefix is reusable only where the complete state at that exact
frontier exists". Its proposal issue #334 carries a user confirmation: "full prefills are just gone".
This is a design reference for the same ground, not a patch for this tree.

## 6–7. The cancellation windows

*Reported*: upstream #371 names a *different* cancellation defect with its mechanism: "A same-session
follow-up materializes in `ConsumeToActive`", and a cancelled follow-up (or an edited last message)
re-prefills the whole conversation — 6.8 s instead of ~0.1 s on a 34k-token conversation. That is a
user-visible TTFT cliff in the same subsystem our item 6 lives in, and it is reproducible in shape on a
shipped lane.

Item 7's arm was VOID twice because the subject *finished* before the socket closed. The harness fix is
to keep the subject generating while the client abandons: a long output cap, and abort after the first
streamed token rather than after a fixed delay.

## 8. Warm TTFT — the instrument exists

`tools/bench/first_request_lane.py` is the instrument ("whether a first request is slower than later
identical ones"), and its own header warns that `--seed` must be pinned. One cold-process run closes
the item; nothing external is needed.

## 9. The precision-assignment search — the literature agrees with our own measurement

*Read in source, two primaries*:

- **ov-freeze** (arXiv 2403.18159, read in the paper's HTML body): "low-bit quantization non-uniformly
  impacts different parts of the attention modules. Specifically, **o- and v-projection layers are more
  susceptible to quantization errors than other weight layers** in the multi-head self-attention
  module", with ov-freeze — freezing those two during QAT — reaching "near floating point precision
  performance, i.e., less than 0.7 % loss of accuracy" for LLaMAv2-7B at 4-bit weights. So the
  isolation arm this item asks for should contrast **o/v against q/k**, which is the opposite of the
  naive guess.
- **QServe** (arXiv 2405.04532v2, read in its own text): "SmoothAttention effectively smooths the
  outliers in **Keys**. **Values doesn't suffer from outliers.**" That is about the *cache*, not the
  weights: the key side carries the outliers and the value side does not.

*Reported, snippet only — the full text is paywalled, an ACM DOI returned 403*: a sensitivity study
distinguishing W8A8, where "activation quantization is the primary bottleneck", from W4A8, where weight
quantization is. That is this port's regime (NVFP4 weights, A8/A16 activations), and it points the same
way as ov-freeze: at 4-bit weights, *weight* precision is the lever. Upstream PR #284 (admitting a Q6
gate_up shape) is that lever approached from the artifact side.

**And the activation-side arm is cheap.** *Read in source*: `src/ops/linear_add/nvfp4/
nvfp4_linear_add_a16.cu` and `src/ops/linear_add/fp8/fp8_linear_add_a16.cu` already exist, so the MLP
down projection has an A16 route in both quantized families. Moving its activation from A8 to A16 is a
dispatch or policy question, not a kernel gap.

**The convergence worth acting on**: our own calibration note says the sites whose input is a *derived
intermediate* (the MLP down projection) are the ones that move most, and that is where the literature
locates activation sensitivity. So the arm this supports is **MLP activations A8 → A16 at the down
projection**, reachable through the existing A16 routes, plus the q/k-versus-v/o isolation the lane
coverage note asks for. Upstream PR #284 (admitting a Q6 gate_up shape and evaluating it unfused) is
the same question approached from the weight side.

## 10. PromptLookup's tree integration — the design is published, so what is missing is ours

*Reported*: **SuffixDecoding** (arXiv 2411.04975, NeurIPS 2025) "leverages suffix trees to perform fast
pattern matching and find possible continuations" and "constructs a speculation tree"; **SAM Decoding**
(ACL 2025, <https://aclanthology.org/2025.acl-long.595.pdf>) does the same with a suffix automaton and
multiple retrieval sources; and SGLang's ngram roadmap (#21052) describes it as "trie paths for almost
every suffix" verified by the target "in one pass".

So the consumer design is not the unknown: a model-free proposer's *multiple* continuations per suffix
are exactly what a verify tree wants. The missing piece is this tree's own consumer for a non-neural
proposer.

## 11. Upstream #208, #333, #229 — and #320, which is ours

**#320 (open) is the one that concerns this port directly.** *Reported*: Xid 79 "GPU has fallen off the
bus" during prefill, **8 occurrences in 4.2 days**, on a single RTX 5090 running
`qwen3_8_27b_nvfp4.ninfer` — this port's artifact class on this port's card. The mechanism named is
`activate()` publishing the KV block table on the legacy stream ("`stream = nullptr` ... unordered
against the `cudaStreamNonBlocking` engine streams"), reported present on `master` (`bace20dc`) and
`dev` (`8eaed538`).

**Checked here.** *Measured*: `git merge-base --is-ancestor 8eaed538 HEAD` → true, so the report was
about a commit this tree contains. *Read in source*, at HEAD the rewrite closed it:

- `src/models/qwen3_5/program/storage/kv_address_space.h:227-230` —
  `void activate(KVAddressSpaceHandle, std::uint32_t, std::int32_t, cudaStream_t stream)` with **no
  default**, passing the stream to `commit_activation(std::move(reservation), stream)`;
- `commit_activation` ends in `tables_->publish(activation.row_->handle(), 0, publish_scratch_,
  stream)` (`:306`) — the stream reaches the publish;
- both call sites pass the compute stream: `transactions/binding.cpp:580,585` — `device.stream`;
- and a scoped search finds **no caller of `.activate(` or `->activate(`** in `src` or `apps`, so the
  only route in is `create_active` (`:196-206`), which requires a stream.

Upstream PR #211 (closed, **not merged**) proposed exactly this fix for the old layout
(`src/targets/qwen3_6/...`, which no longer exists: both `dev` and `master` return 404 for those
paths): "thread `cudaStream_t` through `activate()` and pass `device.stream` at both call sites". The
rewrite reached the same conclusion independently.

**Scope of this check**: the *activation publish* path was audited. Other `= nullptr` defaults exist
(`src/core/paged_kv_cache.h:254-268`, `state_store.h`, `state_image.h`, `linear_attention_state.h`) and
were not audited for call sites that omit a stream; every caller inspected passes an explicit one
(`graphs.cpp:201,473`; `binding.cpp:492`; `context_transaction.cpp:82,90`; `state_store.h:451`).

**#208.** The commenter's fix is in a fork's *own* long-context layout: `causal_cache/small_t.cu`,
`Geometry::SmallTSplitScale`, `physical_pages_s[64]`, `SmallTMaximumSplits` — names that occur nowhere
in this tree's `src`, while upstream's `causal_cache` layout is identical to ours
(`bf16, causal_softmax_attention.cpp, fp8, int8, k8v4, nvfp4` + the same per-dtype file list), so the
fork's file is its own work and the fix is not directly usable. Its *failure class* — a split heuristic
under-provisioning against a fixed shared buffer — is checked absent here: `common/causal_partition.h`
sets `kMaxSplits = 256` and `key_shift = 6` (≥64 keys per split), `bf16/template_launch.cuh:48`
**throws** if `live(visible).splits > capacity`, and shared memory is sized by tile constants
(`static_assert(Bytes <= 99 * 1024)`), never by keys-per-split.

**#333.** No fix upstream. The reported shape — a prefill immediately after a ~98 %-cache-replay prefill
— is testable here on a DFlash2 lane.

**#229.** A fixed 5 ms pressure-search budget whose failure falls back to evicting everything, reported
independently on an RTX 5090 with this artifact class. Reachable only with several large sessions
resident, and every shipped lane runs `--max-concurrency 1`.

## Usable from upstream, mechanisms read rather than headlines

| item | what it gives this port |
|---|---|
| **#365 / #364** | the verify-width under-fill, measured: 80 CTAs on a 170-SM card; the fix is bit-identical and −8..−13 % at the op, unconfirmed end-to-end by its own commenter |
| **#316** | the tool-call parser is all-or-nothing: generation stopping inside a malformed `<tool_call>` returns the whole region as assistant text. Partial recovery is the fix |
| **#335 / #334** | the hybrid prefix cache and the GDN rollback constraint; a user confirmation on the proposal |
| **#292** | a shared Q5 K-split MMA, +11 % MTP3 decode — and the artifacts carry Q5 sites; needs an end-to-end measurement here |
| **#221** | `mtp_graph_profiles` carries no topology class, so "any draft window past five fails at startup" — relevant only if the MTP draft cap is raised |
| **#268** | fold the sigmoid gate into the causal reduce epilogue: one graph node instead of two, measured on the node count |
| **#284** | admitting a Q6 fused gate_up shape and evaluating it unfused — the same precision question from the weight side |
| **#233 / #84** | upstream is developing a native Windows MSVC build — the ground this port already covers |
| **#340** | *not* a conflict: its `SO_REUSEADDR` sits inside `#if defined(__linux__)`, for Linux TIME_WAIT, and never touches the Windows path this port changed |
| **#341** | Linux-only, confirmed by the same patch |

## What was not done, and one correction

- **A claim in this note's first version was wrong and is corrected in item 1**: the A16 fallback does
  not take the strided `silu_mul` kernel, it takes the vectorized dim0-split path. It was found by
  checking the claim against `can_use_dim0_split_fast_path` rather than by re-reading the sentence that
  made it.
- The ACM sensitivity study returned 403, so its W8A8/W4A8 distinction stays a search snippet rather
  than a read source; QServe's and ov-freeze's statements were read in full.
- The fork's calibration script was **read, not re-run** — separating corpus from realization needs its
  torch environment.
- Item 4's counterexample, item 8's cold-start run, #371's reproduction and #333's shape are **plans**,
  not measurements.
- The #320 audit covers the activation publish path only, as scoped above.
- No lane, artifact or flag was changed by this research. The two measurements taken here — the mutation
  seam both ways, and 25 concurrent repeats of the top-k op tests — are reproducible from the commands
  quoted in item 2.

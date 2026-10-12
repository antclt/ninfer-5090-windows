# Upstream and forks: what this port can harvest

Date: 2026-10-12. Scope: engine-level improvements available to **this** port
(`C:\AI\ninfer-v3-windows`, origin `headpiece747/ninfer-5090-windows`) in **prefill throughput,
decode token speed and accuracy**, from three sources: upstream `dev` (40 commits ahead), upstream's
open issues and pull requests, and the fork population (316 forks).

**Rule this report follows — upstream is not ours to fix.** Upstream's issues and PRs are treated
purely as a catalogue of mechanisms. Nothing here proposes an upstream PR, an upstream comment, a
patch to upstream's tree, or an upstream plan. Each item is judged only by *can this port use the
mechanism to make a better implementation?* A rejection with a reason is a result; several items
below are rejected in one line.

**Number provenance.** Every number is one of:

- **[third party]** — measured by the issue/PR author on their hardware, quoted from the issue or PR.
- **[read in source]** — read in this port's or upstream's tree; not measured.
- **[measured here]** — measured in this session. No performance figure below is; this session
  measured nothing on the GPU. Where a figure is an author's estimate, it is labelled as such.

Method: `git log --oneline HEAD..upstream/dev` + `--shortstat` + commit bodies; `gh issue list/view`
and `gh pr view/diff`; `gh api repos/Neroued/ninfer/forks` (through a Python script, per the repo's
quoting rule) and the cross-repo compare API; targeted reads of this port's files. The port's own
research under `docs/research/` (e.g. `fp8-nvfp4-prefill-evidence.md`, `kv-dtype-evidence.md`) is
assumed current and is not re-derived.

---

## 1. Upstream `dev` is 40 commits ahead — none is a prefill/decode/accuracy change for this port's model

`git log --oneline HEAD..upstream/dev` returns 40 commits (merge-base `81c8ce09`). Classification,
by subject, body, file paths, and — for the six commits that looked behavioural — a read of the diff
or body:

| sha | subject | class |
|---|---|---|
| `90d25edd` | refactor(ops): establish family organization and independent linear split | (a) reorg + API reshape (`linear_split.h` added) |
| `7ee63999` | refactor(ops): consolidate shared implementation primitives | (a) reorg |
| `20d3a13f` | refactor(ops): organize basic operator families | (a) reorg |
| `1844d861` | refactor(ops): organize fused operator families | (a) reorg |
| `4eb2c042` | refactor(ops): unify segmented linear projections | (a) reorg |
| `d53eb7ae` | refactor(ops): organize remaining linear variants | (a) reorg |
| `6733be72` | refactor(ops): organize rope and kv cache families | (a) reorg |
| `c012155c` | refactor(ops): organize softmax attention family | (a) reorg |
| `4366c7b6` | refactor(ops): organize linear attention family | (a) reorg |
| `8cc45088` | refactor(ops): organize sampling and sequence operations | (a) reorg |
| `e6585d15` | refactor(ops): organize sparse moe execution | (a) reorg |
| `ae54c1a9` | refactor(ops): finish family migration cleanup | (a) reorg |
| `6e772ee3` | docs(ops): consolidate refactoring guidance | (a) docs |
| `3124dffe` | refactor: streamline execution and op tooling | (a) (−5283 lines) |
| `6a42de35` | refactor(core): separate memory ownership and cuda execution | (a) reorg (478 files) |
| `f1ace85e` | docs: consolidate model and maintenance references | (a) docs (−5704 lines) |
| `74671a4e` | feat(core): add unified cpu execution foundations | (b) new CPU executor/task pool replaces `host_worker_pool`; no number in body |
| `642d588f` | fix(tests): synchronize hc mix fixture initialization | (b) test-only, 4 lines |
| `5989682a` | feat: add optimized hc block and final mix | (b) Flash-Next op; validation inventory, no throughput figure |
| `70a11f12` | feat: add optimized hc residual ops | (b) Flash-Next op; no throughput figure |
| `c6f47b52` | docs: sync rmsnorm and gdn control documentation | (a) docs |
| `f7d3090b` | feat: add 96x2560 gdn control and reorganize norm ops | (b) carries an unexplained **44.96 %** per-op difference; "No end-to-end performance claim is made" |
| `6a4104fd` | feat: add sigmoid gated rmsnorm and explicitly name silu gate | (b) new op variant; no number |
| `fbcd8504` | feat(ops): support quantized kv in qsa sparse gqa | (b) Flash-Next |
| `47cdc08a` | feat(ops): add bf16 qsa sparse gqa | (b) Flash-Next |
| `e70fc17f` | feat(ops): add qsa indexer | (b) Flash-Next |
| `fc4e3331` | feat(ops): support quantized kv in qsa prepare | (b) Flash-Next |
| `d02703e2` | docs(qsa): remove intermediate bf16 key rounding | (a) docs |
| `30b489f6` | feat(ops): add bf16 qsa prepare | (b) Flash-Next |
| `1a9d676e` | feat(ops): add qsa tail extraction | (b) Flash-Next |
| `d6a8de0e` | feat(ops): add qsa split input projections | (b) Flash-Next |
| `be12563a` | feat(ops): implement ple convolution and state extraction | (b) Flash-Next |
| `fbd285b5` | docs: define expert weight cache and offload design | (a) docs (807 lines) |
| `f2620157` | feat(ops): add qwen4exp gdn shapes and reorganize gdn operators | (b) + (a) Flash-Next |
| `a0d27b22` | feat: support mixed row spans in paged kv | (b) engine change; no number |
| `be9ad8a3` | feat: define ple shared op contracts | (b) contracts only (1 file) |
| `30e3a92f` | docs: plan main workspace qwen4exp tasks | (a) docs |
| `dc8b58fb` | docs: define independent qwen4exp op tasks | (a) docs |
| `d8c0fa6f` | feat: add qwen4exp weight conversion and paged nf4 | (b) converter/artifact |
| `84d1cf02` | docs: record qwen4exp support and linear plans | (a) docs |

Findings:

1. **The rest are not more of the same reorg.** The first 15 are a *second-stage* `src/ops/` regroup
   into `elementwise`, `embedding`, `indexing`, `normalization`, `reduction`, `rope`, `sampling`,
   `sequence` (`git ls-tree upstream/dev src/ops/`), while this port carries its own partition
   (`attn_input_proj`, `candidate_selector`, `launcher`, `linear_add`, `linear_pair`,
   `linear_swiglu`, `linear_topk`, `rmsnorm_rope`, `wrapper`, ...). Both trees hold ~720 op files
   (716 here, 731 upstream). **This is a different partition of the same code, not work the port is
   missing**; merging it would be a large mechanical conflict for no measured behaviour change.
2. **16 of the remaining 25 feature commits build Qwen3.8-Flash-Next / qwen4exp ops** — HC block and
   residual, PLE convolution, QSA indexer/prepare/gqa, 96x2560 GDN control, sigmoid-gated RMSNorm,
   paged-NF4 conversion. Their own docs name the target: `docs/maintainer/qwen3_8-flash-next-hc.md`,
   `-ple.md`, `-model.md`, `qwen4_exp-execution.md` (`git show --stat 5989682a`, `be12563a`,
   `90d25edd`). This port implements `Qwen3_5ForCausalLM`/`Qwen3_5MoeForCausalLM` and ships five
   Qwen3.8-27B images (`README.md:49`); it does not load Flash-Next. **New-model capability, not a
   prefill/decode/accuracy gap for the served models.**
3. **The only measured number any of the 40 carries is a caution.** `f7d3090b`'s body states:
   "Prior comparative performance differences remain unexplained, up to 44.96 percent. Eight D128
   kernel instruction sequences match the pre-refactor baseline. No end-to-end performance claim is
   made." [read in source] That is upstream's own norm/GDN-control refactor reporting an
   unexplained per-op difference of up to 44.96 % against its baseline — a reason to treat the
   refactor wave as *not* a source of gains.
4. **One engine change is worth a look later:** `a0d27b22` adds `tokens_per_row` to
   `KVPlaneGeometry` (spans 1 and 4; span 4 requires PageMajor) — the prerequisite for a
   multi-token-row KV format such as #173's compressed KV. This port has `src/core/paged_kv_cache.h`
   but no such format; its shipped KV dtypes (`k8v4`, `fp8` in `tools/release/profiles.py`) are
   one-token rows. Nothing in the port needs it today.

---

## 2. Open issues and PRs: the catalogue to harvest

Upstream has 85 open issues and 56 open PRs. Below, each item is mechanism → number → whether this
port can use it, with the file in *this* repo it would touch. Rejections are one line.

### 2.1 Prefill

**(a) #283 + PR series #286–#290 — an NVFP4 expert profile for `ops::sparse_moe`.** [third party, RTX 5090]
Mechanism: `prepare_sparse_moe_weights` accepts expert banks only as Q4+Q5/Q6 or Q8+Q8; the series
adds an NVFP4 expert profile (decode, small-token and prefill routes) plus a container `divisors`
member so a stacked plane can carry several per-matrix FP32 divisors, or a lossy fold that rescales
block scales. Measured (`tools/bench/run_serve_corpus.py`, same binary, five seeds): prefill
18,237 → 23,685 tok/s at 7,680 tokens (**+29.9 %**), +17.5 % at 64,512, +11.7 % at 130,048, +7.3 %
at 260,096; decode unchanged (MTP3 median −0.6 %, inside the instrument's 0.31–1.20 % drift);
perplexity +0.19 % stated by the author as a floor; weights −475.9 MiB; workspace identical. The
fold alternative measured at 2.6–2.7 % relative L2 on block scales and NLL 1.490148 → 1.493334.
Applicability: **MoE only, and this port ships dense images** — but it implements
`Qwen3_5MoeForCausalLM` and carries `src/ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu`, so
this is the strongest measured prefill item if a MoE artifact is ever built. Where it would land:
`src/ops/weight_input.cpp` (the admission gate), `src/ops/sparse_moe/`, the artifact schema, and a
`tools/convert` recipe. Needs the GPU to build and qualify.

**(b) #356 + PR #355 — NVFP4 tiled prefill under-occupies the device.** [third party]
Mechanism: nvfp4 tiled attention launches 1 CTA per (query tile, head) over the full KV range —
~2.26 waves/SM vs ~7.91 for fp8/k8v4. The fix partitions the KV range over `blockIdx.z`, writes
FP32 partials, and reuses `causal_tiled_merge`; acceptance is "mma >= 15 % faster, decode unchanged"
(no end-to-end number is posted).
Applicability: **conditional on `--kv-dtype nvfp4`.** This port's nvfp4 tiled route has no split —
`src/ops/softmax_attention/dense/causal_cache/nvfp4/tiled_launch.cu` contains no split/merge/
`blockIdx.z` — while its fp8 and k8v4 routes already do (`fp8/tiled_launch.cu:5,17` and
`k8v4/tiled_launch.cu:5,17` use `causal_tiled_merge`) [read in source]. The port's eight lanes run
`k8v4`/`fp8` KV (`tools/release/profiles.py`), so nothing shipped gains today; the PR becomes
relevant if an nvfp4-KV lane is enabled. The port's own `docs/research/kv-dtype-evidence.md` owns
the KV-dtype question.

**(c) #303 — bounded multi-chunk prefill execution unit.** [third party, design only]
Mechanism: keep up to N=4 prefill chunks on the compute stream before the synchronization boundary;
budget stays 1 when decode is runnable, so decode is not starved. The author states explicitly "I am
not making a performance claim yet". Where: `src/models/qwen3_5/execution/text.cpp` (the per-chunk
`ctx_.synchronize()`), `src/models/qwen3_5/program/prefill.cpp`. **Rejected for now — no
measurement to weigh**; the prototype is on an older layout.

**(d) #277 — raise default `prefill_chunk` 1024 → 2048. Rejected: already exceeded.**
The port's default is **8192** (`include/ninfer/types.h:165`, `src/serve/serve_options.h:36`,
`apps/cli/options.h:30`) [read in source], four times the proposal. The issue's own measurement
(2332-token prompt: 782 ms at 1024 → 773 ms at 2048, math-neutral) is consistent with the port
having already banked a larger version of this.

**(e) #167 + #324 — FP8 A8 GEMM TMA staging. Rejected: already present.**
The port stages FP8 A8 through TMA at all five call sites (`launch_fp8_a8_tma` callers: `linear`
24, `attn_input_proj` 1, `gdn_input_proj` 1, `linear_add` 1, `linear_swiglu` 1) and introduced its
own route in port commit `909fb087` "perf(linear): optimize fp8 34816x5120 with tma split-k"
[read in source]. Upstream's PR reports +2.40 %/+4.95 %/+4.38 % at chunks 1024/4096/8192 on
qwen3.8-27b with decode off this path [third party]; the port's equivalent exists, and its
FP8-vs-NVFP4 prefill study is `docs/research/fp8-nvfp4-prefill-evidence.md`.

**(f) #68 — MoE tail warms L2 for the next consumer.** [third party, RTX 5090]
Mechanism: `prefetch.global.L2` hints issued in the MoE D4 epilogue (next layer's projection codes)
and the D1 tail (shared-expert down codes), behind a small cross-layer `WeightPrefetchSpan`
contract. Measured interleaved A/B/A/B on Qwen3.6-35B-A3B T=1: **363.7 → 377.2 tok/s (+3.7 %)**;
the projection GEMV drops 580.6 → 482.2 µs, the hint-issuing grid grows 40.4 µs, the step
3391 → 3306 µs. Applicability: MoE only; where it would land:
`src/ops/sparse_moe/decode/sparse_moe_decode_kernels.cu` plus an ops-surface contract. Needs GPU.

### 2.2 Decode

**(a) #364 + PR #365 — NVFP4 W4A4 `linear_add` [5120,17408] at verify widths. The strongest item
on this list.** [third party, RTX 5090]
Mechanism: `M32N64` gives only 80 CTAs on a 170-SM 5090 (5120/64), so a speculative verify streams
the 50 MB down-projection from under half the SMs. The fix selects `M32N32` with a four-stage
pipeline — `Nvfp4A4MmaSchedule<32, 32, 256, 2, 4, 4, 2>` — when `kInputRows == 17408 && tokens <= 32`;
each output row keeps the same K order and MMA sequence, so outputs are bit-identical. Measured:
T=8 46.59 → 40.40 µs (−13.3 %), T=9..32 −8.1..−9.2 %, T≥33 unchanged within ±1 %; end-to-end
DFlash2 rounds 16.75–16.81 → 16.40–16.44 ms at K=7 and 17.72–17.79 → 17.44–17.52 ms at K=15
(−1.6..−2.5 %) with identical acceptance counts; perplexity unchanged and nine greedy completions
byte-identical. The implementation is 11 added lines + 1 deletion.
Applicability: **the port's code is the exact pre-image.** `src/ops/linear_add/nvfp4/nvfp4_linear_add_a4.cu:15`
defines `M32N64 = Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>` and `:37` dispatches every
`tokens <= 64` to it; `Nvfp4N5120K17408 = Nvfp4Geometry<5120, 17408>` in `nvfp4_geometry.h`, so the
PR's `kInputRows == 17408` predicate is the port's own `launch_problem<Geometry>` template. All
five shipped images are NVFP4-MLP Qwen3.8-27B and every lane's verify width is ≤32 (DFlash2 draft
7/9 → T≈8/10; MTP draft 4/5 → T=5/6, `tools/release/profiles.py`). [read in source]
**GPU to re-verify** (the port's own `ninfer_nvfp4_linear_add_bench` and a DFlash2 round), because
the measurement is the author's on their 5090, not this machine's.

**(b) #279 — `ChunkedSmallT` re-streams the full KV once per query chunk.** [third party, op-level]
Mechanism: above the prompt route limit the query axis is partitioned while every chunk still
sweeps the full visible key range, so total KV sweeps = `ceil(W / chunk)`: W=9..10 takes 2 sweeps,
W=13..16 takes 2 (3 on q16). Measured on `ninfer_causal_softmax_attention_bench`, q24, batch 1,
cold: 16K bf16 W=8 77 µs @868 GB/s vs W=16 135 µs @497 GB/s (1.75×); 64K 201 vs 393 µs (1.96×);
16K nvfp4 47 vs 88 µs. At 64K that is ~190 µs wasted per full-attention layer per verify step
(16 such layers on qwen3_6_27b).
Applicability: **this port publishes the configuration the issue says triggers it.** Three lanes
run DFlash2 with `draft=9` → W=10, inside the 9..10 band (`tools/release/profiles.py`: QUASAR,
NVFP4-full and NVIDIA DFlash2 lanes), while the issue notes its own published configs (K=7 → W=8)
never trigger it. Where: `src/ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp`
(`launch_chunked_small_t`); the issue says a single-sweep fix needs new TokenTile-12/16 kernels
(`causal_attention_split_capacity` throws for tokens > 8 on q24), and that acceptance-vs-K at
32–64K is unmeasured because the author had no DFlash-capable artifact — **this port has them**.
GPU needed: kernel work plus `tools/bench/realtext_acceptance.py`.

**(c) #327 — reciprocal-multiply in the NVFP4 activation quantize.** [third party, RTX 5090]
Mechanism: per-route quotient policy on the shared W4A4 quantizer; only the plain-linear MMA route
opts into one reciprocal + multiplies instead of sixteen fp32 divisions (other routes keep
divisions and bit-identical behaviour). Measured, sustained graph-mode interleaved A/B, both run
orders: down-proj T=8 30.79 → 29.64 µs/op (−3.7 %), o-proj T=8 13.02 → 11.90 (−8.6 %), T=4096 tied.
The ≈0.4 %-of-a-round figure is **an author estimate, not a measurement** (the issue says so). A GPU
probe found 0.15 % of e2m1 codes flip; oracle suites pass. An unscoped variant regressed prefill
T=4096 by ≈1 %, cause unresolved — hence the scoping. Where: `src/ops/linear/nvfp4/nvfp4_a4.cu:59`
(`launch_nvfp4_a4_quantize`) and its operand path. GPU to A/B on this port's route.

**(d) #330 — NVFP4 A16 sliced-K schedule knobs at T=3..16.** [third party, RTX 5070 Ti **only**]
Mechanism: two template knobs on `Nvfp4A16SlicedKMmaSchedule` — `StageTokens` (activation rows held
in shared memory) and `RowTiles` (16-row MMA tiles sharing one activation stage) — with defaults
equal to current behaviour, so outputs stay bit-identical. Measured on a 70-SM 5070 Ti:
`linear_swiglu` A16 [34816,5120] T=3 165.1 → 149.1 µs (−9.7 %), T=4 −9.9 %, T=16 −5.7 %;
`linear` [5120,17408] T=3/4 −9 %; tp2 half shape 85.0 → 78.2 µs at T=4; end-to-end MTP3 round
−3.3 %/−2.9 % at tp 2. The author states the tp 1 end-to-end number is an estimate and that the
**5090 may already be fine** with the current schedule.
Applicability: the port has the template (`src/ops/linear/nvfp4/nvfp4_a16_sliced_k_mma.cuh`,
`nvfp4_schedule.cuh:221`) and the MTP lanes it would affect, but the tuning is per-GPU. **The 5090
verdict is open and cheap to take**: `bench/ops/nvfp4_linear_swiglu_bench` at T=1..16. GPU needed.

**(e) #292 + #293 — Q5 shared K-split MMA. Rejected: no Q5 artifact.**
The PR's headline is +11.4 % MTP3 decode (197.26 → 219.68 tok/s) and +1.4 % ordinary decode, measured
on `qwen3.8-27b (q5_g64_fp16)` with fp8 KV on a 5090 [third party]; the op-level win (T=2..16
−26..−33 %) does not surface end to end because the model's decode path does not spend enough time
in those four shapes (the author says so). This port ships five NVFP4 images (`README.md:49`) and no
Q5 artifact, so there is no lane to gain from it.

**(f) #311 — pipeline Q4/Q5 attention groups at T=32. Rejected: no Q4/Q5 artifact** (same reason;
and the PR is a draft whose own C8 A/B sign flips between blocks: AG vs A −4.8 % then +3.5 %).

**(g) #199 — sparse_moe: two Q4 group quads in flight in the routed gate/up dot.** MoE only.
Mechanism: feed depth 2 in the routed gate/up dot product. Numbers: fixture −7.40 % on `trace-like`
T=4; inside a real round −2.75 % over 5240 launches, kernel total −3.11/−3.13 %; the routed-down
kernel −0.15 % (nothing); a W8 draft instantiation of some kernels +6.1..6.9 %. [third party]
Applicability: MoE only; `src/ops/sparse_moe/decode/sparse_moe_decode_kernels.cu`.

**(h) #385 — DFlash2 CUDA-graph reservation grows ~1.16 GiB/batch after `08902f73`. This port has
the exact code.** [third party; code confirmed read in this port]
Mechanism: `08902f73` ("feat: add gbnf constrained decoding across generation backends",
2026-10-06 — **in this port's history**) doubled the long-profile per-class graph allowance
96 → 192 MiB and changed the aggregation from `per_batch × max_concurrency` to
`(per_batch + 8 MiB) × max_concurrency`. DFlash2 gives each profile its own topology class, so the
long classes cost 192×3 = 576 MiB/batch instead of 288; at C=4 that is 1,920 → 3,104 MiB
(**+1,184 MiB**). The reported startup shortfall 377,262,593 B matches byte-for-byte
(1,240,924,160 − 863,661,567). The port's code:
`src/models/qwen3_5/program/planning/startup.cpp:893` reads
`(final_visible <= 4096 ? 64ULL : 192ULL) * kMiB` and `:899` multiplies
`per_batch_allowance + 8ULL * kMiB` by `max_concurrency` [read in source]. The affected shape is
"DFlash2 + 32 GB card + large `--max-context` + `k8v4`" — three of this port's eight lanes are
exactly that (DFlash2, 262,144 context, 5090 32 GB). Fix shape (port-side, in the port's own tree):
reserve the grammar-mask graph storage only when constrained decoding is enabled, and/or decouple
the additive term from the topology-class count. GPU needed to verify startup and memory.

**(i) #320 — Xid 79 during prefill: the KV block table published on the legacy stream. Rejected:
the port already threads the stream.**
The issue's defect is that `activate()` has no stream parameter and the publish lands on legacy
stream 0 while prefill is in flight. In this port, `activate(...)` and `commit_activation(...)`
require an explicit `cudaStream_t` — `src/models/qwen3_5/program/storage/kv_address_space.h:228` and
`:272`, with no `= nullptr` on that path [read in source]. The issue remains open upstream
(8 crashes in 4.2 days, one with a core dump); nothing for this port to take.

**(j) #381 — `O_DIRECT` hardcoded for payload reads. Same flag here; low impact on local storage.**
The port's `src/artifact/file_io.cpp:162` opens with
`FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN` (Windows) and `:178` with `O_DIRECT` (POSIX);
the fill loop the issue asks for exists at `:126` [read in source]. The issue's measured effect —
22 m 58 s → 5.3 s for an 18.6 GiB load — is on cloud-backed storage [third party], and the author
states local disks are near-identical. This port loads from local disks, so the item is relevant
only if artifacts move to a network share; note the Windows branch is the one the issue says already
had the fill loop.

**(k) #375 — KV precision tail. Rejected for this port's dtypes.**
Mechanism: keep the newest N tokens exact in a device-only shadow ring and merge two online-softmax
partials (body quantized, tail exact) in FP32. Measured [third party, RTX 5070 Ti]: 64 MiB per
sequence at N=1024/C=1 (+4 MiB rollback reserve, byte-exact to the model); decode median −0.18 %,
0/64 cells above +2 %; KLD improves in 24/24 cells; quality gain monotone in body coarseness.
**Decisive for this port: "On `fp8`, `nvfp4` and `k8v4` the ring is allocated but inert"** — those
are the port's shipped KV dtypes. It would pay off only for `int8`/`bf16` KV lanes, which this port
does not publish.

### 2.3 Accuracy

**(a) #374 — fused residual rounding depends on tile occupancy. Likely present here; accuracy and
determinism item.** [third party]
Mechanism: the `FullTokens` instance lets nvcc contract `accumulator * alpha + residual` into one
FMA, while tail tiles and the 1024-token TMA chunks round twice — so one token's prefill bits depend
on where it falls in the chunk. Measured by the issue: before the fix, bits differ at 9 NVFP4 widths
(the tile-aligned ones) and 14–21 FP8 widths per shape; after, 55–67 widths per shape are
bit-identical; the SASS of the A16 decode kernels is unchanged. The fork's per-format choice —
NVFP4 explicit MUL then ADD, FP8 FMA — removes the width dependence with perplexity identical to
every digit and GSM8K unchanged (1,319 problems, McNemar p = 0.50). Origin commits:
`MirkoCovizzi/ninfer-rtx5090-mobile` `87f06898` (nvfp4) and `37a1ed49` (fp8); the width-sweep test is
`ValerioDolci/ninfer-tp2` `f08572f3`; the per-format decision is `c3383145`.
Applicability: the port has both affected prefill routes — NVFP4 A4 and FP8 A8 `linear_add`
(`src/ops/linear_add/nvfp4/nvfp4_linear_add_a4.cu`, `src/ops/linear_add/fp8/fp8_linear_add_a8.cu`)
and the shared epilogue `src/ops/linear/common/epilogue.cuh` (`LinearResidualAddEpilogue`,
`value + residual.load(...)`) [read in source]. Whether the port's tile-aligned instance contracts
to FMA is not settled by reading; **the width sweep settles it**, and the fork provides both the
test and the fix. GPU needed.

**(b) #119 — NVFP4 MTP self-consistency on code. Exactly this port's artifact type.** [third party, RTX 5090]
Numbers: groupwise-int MTP3 171.7 tok/s at 84.2 % acceptance (93/85/75 % by draft position) vs
NVFP4 MTP3 152.2 tok/s at 71.6 % (84/72/58 %) on `scenario_code_python`; on structured JSON and
prose NVFP4 is faster with comparable acceptance. The author's narrower claim is that the NVFP4
target and its own embedded MTP draft are less self-consistent on code, not that NVFP4 is worse
generally. This port's QUASAR and NVIDIA lanes are NVFP4 + MTP4 + `--lm-head-draft` — the same
combination. Candidate action: run the port's own per-position acceptance split (the port has
`tools/bench/realtext_acceptance.py`, which reads per-arm acceptance from a lane's own flags) and
compare against the groupwise alternative, e.g. the port's `nvfp4full_noex` lane which already
re-encodes BF16 exceptions. GPU needed.

**(c) #232 — `gdn_gating_proj` numeric tolerance under MSVC. Free to check; same criterion here.**
The port carries the same gate at `tests/ops/test_gdn_gating_proj.cpp:42`
(`kGdnNormControlFp32{/*relative_l2=*/8.0e-4, ...}`) [read in source]. The issue reports a
deterministic failure under MSVC 19.44/CUDA 12.8 on a V100: rel_l2 = 0.000896 vs limit 0.0008
(ratio 1.12), with the healthy sibling case already at ratio 0.795 — the BF16 staging noise floor
is ~0.9e-3, so 8.0e-4 has no headroom outside the author's environment. Suggested resolution:
restore 1.0e-3 (the pre-`ad54e179` value) or document MSVC as a known limitation.
Applicability: run `ninfer_gdn_gating_proj_test` in `build-test` on MSVC 14.51/CUDA 13.3; if it
fails, the port's own test gate is red and the fix is one constant. Cheap; GPU needed only because
it is an op test.

**(d) #298 — finetune conversion notes.** [third party, WSL2/5090]
Two findings worth having on record: (1) `validate_tokenizer_config` reads `add_bos_token` with a
default of `true`, so a finetune whose `tokenizer_config.json` omits the key (newer `transformers`
exports do) is refused at startup with "tokenizer_config.json does not match Qwen3.5 tokenizer
prefix semantics" — the workaround is `--resource tokenizer_config.json=<stock>`; (2) the DFlash2
companion does not transfer to finetunes: `--spec dflash2 --draft-tokens 7` accepted 3.3–5.1 % of
drafts and fell to 67–75 tok/s, versus MTP at 175–206 tok/s with 67–85 % acceptance. Applicability:
this port converts its own artifacts from stock checkpoints, so (1) bites only if the port converts
finetunes; (2) is worth a line in the port's DFlash2 documentation if it ever ships a finetune lane.
The same report also gives this port's operating point an independent check: WSL2 at published
speeds (code 161.4 tok/s at 60 % MTP acceptance, a 147k prompt at 3.47k tok/s prefill).

**(e) #171 — NIAH `judge_strategy: rule` scores wording, not retrieval.** [third party]
The in-tree needle harness's rule judge can pass on answer phrasing; a measurement-tooling defect
for long-context accuracy claims. Applicability: if the port uses NIAH results to claim 262k-context
accuracy, check its judge (`tools/` harness) rather than trusting the score. No GPU needed to read.

**(f) #165 — YaRN context extension.** Proposal only; this port already runs 262,144 context and
its lanes are measured there. Gzenz's fork carries a working YaRN-for-DFlash2 variant (see §3).
Not needed for the three areas unless context beyond 262k is wanted.

### 2.4 Checked and rejected in one line

- **#264 (NVFP4 fused SwiGLU ragged widths)** — not applicable as written: upstream's fused route
  had a private kernel that missed the partial-M-tile change, while this port's fused route
  delegates to the *shared* `launch_nvfp4_a4_tma_mma` (`src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_a4_tma.cu:9`)
  and the plan sends every width ≥256 to it (`nvfp4_linear_swiglu_plan.cpp:45`) [read in source].
  Residual check, one read: that the shared kernel takes a partial last M tile. The PR's numbers
  (faster at every sampled ragged width 513–3000, 5.93 % worst / 9.03 % median; op workspace
  566.43 → 35.34 MiB with the engine arena unchanged) are for upstream's structure.
- **#221 / tmark00 `6e8d0554` (MTP graph profiles topology class)** — already present:
  `src/models/qwen3_5/program/planning/graph_profiles.cpp:70` assigns
  `profile.max > 96U ? 1U : 0U` [read in source].
- **#235 (CUDA 12.9 floor), #233/#84/#59 (Windows ports)** — this port is already a native
  MSVC/CUDA 13.3 build.
- **#345/#344/#343/#362 (retrieval KV windows, nonresident pages), #335/#334/#350 (prefix-cache
  redesigns)** — the port has its own context-cache rewrite; these are upstream-shaped alternatives
  without measured wins on this port's lanes.
- **#294/#33 (grammar/structured output)** — the port already carries gbnf (`08902f73` is in its
  history).
- **#173 (rk2v4-e8 compressed KV)** — a new KV format with no 5090 numbers beyond design; its
  paged-KV prerequisite is upstream `a0d27b22` (§1.4). Watch, do not take.
- **#174 (full-vocab Q4G64 MTP proposal head), #172 (resumable output-limit generations),
  #376 (configurable KV headroom), #390 (DGX Spark), #313 (two-GPU TP)** — out of the three areas
  or out of this port's single-5090 scope. #376 is the mitigation named by #385 and is subsumed by
  fixing #385 properly.

---

## 3. Forks

Enumerated with `gh api repos/Neroued/ninfer/forks` — **316 forks**. Thirty candidates were compared
against `upstream/dev` with the cross-repo compare API, then their recent commits listed. What the
port has already harvested is recorded in `NOTICE` (the 1.0.x Windows layer derives from
`Don-Chad/ninfer-3090`, 501 of 524 lines identical on one file).

- **`headpiece747/ninfer-5090-windows`** — this port's own origin. The compare API 404s for it
  (`gh api repos/Neroued/ninfer/compare/Neroued:dev...headpiece747:dev`), and the cause is
  attributed: the repo is **not a GitHub fork** (`gh api repos/headpiece747/ninfer-5090-windows`
  returns `"fork": false, "parent": null`), and GitHub's cross-repo compare requires a fork
  relationship. Not a defect; use local `git log` for this one.
- **`Don-Chad/ninfer-3090`** (444★, ahead 130 / behind 263). Perf-relevant commits:
  `48f9d256` "prevent KV loan reservation starvation and handle worker crash"; `ff24c4f0` enables
  integer-activation prefill by default on **sm_86** with `NINFER_W4A8_PREFILL` opt-out (not the
  5090); `09f483a8` adds timings/throughput metrics to responses. The KV-starvation fix targets a
  runtime layout 263 commits behind; the port's context cache was rewritten since, so read the diff
  before taking it. [read in source: commit subjects]
- **`gzenz/ninfer`** (23★, ahead 208, ~198 commits since 2026-09-10). A production fork of the
  context cache/planner with heavy diagnostics. Perf/accuracy-relevant: `576e72ea` "Support
  concurrent staged-prefill lanes and apply YARN scaling to DFlash target RoPE positions";
  `3a014fd6` "YaRN for dflash2"; `104dda52` value-aware demote-to-host; `4e124b81` host-pool
  placement policy; `300ddb9f` blocking CUDA synchronization by default. The staged-prefill-lane
  work is prefill-relevant (concurrency > 1) and the YaRN-on-DFlash-target work is an
  accuracy/context-extension item; both are worth one diff read each. [read in source: subjects]
- **`natpate/ninfer-windows`** (178★, ahead 106). MSVC port fixes (`2618c8ed` `__uint128_t` →
  `_umul128`; `bb0046d7` C2326 lambda capture; `b6123c08` conforming preprocessor for CUDA 13.3
  CCCL; `2a2676d2` `localtime_s`; `/utf-8`), plus `322a9b10` "support weight-only NVFP4 DFlash2
  execution" and a merge of upstream's "nvfp4 W4A4 TMA tile-contiguous scales + partial last M
  tile". The port compiles and ships today, so the MSVC fixes are history; the NVFP4-DFlash2 item
  overlaps the port's own draft encoding. [read in source: subjects]
- **`MirkoCovizzi/ninfer-rtx5090-mobile`** (19★, ahead 30). `87f06898` / `37a1ed49` — the
  residual-rounding fixes that are #374's origin; `ad1f578d` vectorized lossless NVFP4 scale
  expansion; `b6a252ad` NVFP4 scale compression with pooled residency; `dff96dca` adaptive draft
  widths. The rounding fixes are the ones with a measured bit-identity effect.
- **`ValerioDolci/ninfer-tp2`** (25★, ahead 225). Mostly two-GPU (not this port), but single-GPU
  items: `91582814` (the #330 knobs), `070e8476`/`a9d8ed8c`/`16032b99` (residual rounding + the
  width-sweep test), `12286e95` "require explicit publication streams" (the #320-class fix — this
  port already has it), `795f590b` artifact staging-slot alignment, `f30acbb3` `--vram-headroom-mib`,
  `169514ea`/`21ad2f4a`/`4b9c098e` INT8 causal-attention split budgeting from the device's own SM
  count.
- **`tmark00/ninfer`** (ahead 81). `853d268c` two-stage chunked GDN kernels (upstream `0784e76f` +
  `e9c7046f`); `ebfdb1bf` `NINFER_CUDA_SYNC` schedule selection; `6e8d0554` the MTP topology-class
  fix that is already in this port. The GDN chunked kernels are the one item to read for this port's
  GDN layers.
- **`knoopx/ninfer`** (ahead 9). `5f7532a9` "fast INT8 prompt kernel with key splits and
  wave-aligned chunks" — prefill attention for INT8 KV; relevant only if an int8-KV lane exists
  (the port publishes k8v4/fp8).
- **`lkarlslund/ninfer6000`** (7★, ahead 58). Flash-Next perf: QSA group-order sorting, FP8 MTP
  drafter, INT8 group-64 KV, adaptive MTP draft count, MoE entry fusion. New-model work; not this
  port's model.
- **`star1872/ninfer-offload`** (ahead 67). MoE routed-expert offload to host with device replicas —
  a small-VRAM capability, not a 5090 win.
- **`Oceania2018/ninfer-5090-laptop`** (ahead 13). `0e9066cf` runs NVFP4 `linear_swiglu` A16 above
  T=16 in 16-token chunks; `971ce513` host state dtype; checkpoint pruning. A 5090-adjacent fork;
  the swiglu chunking item is worth one read.
- **`toddballinger/ninfer-5080`** (35★, ahead 255). Admission/KV-denial forensics and decision-API
  work, `00e8e47f`/`4b62aca3` Q4/Q5 routing ports. Scheduler diagnostics rather than prefill/decode
  wins.
- **`ByronLeeeee/ninfer-extended`** (ahead 30). BF16 ASR/OCR/embedding extensions, "reuse bf16
  weights across wider prefill tiles" — different model families.
- **Platform/other forks with nothing to harvest for the three areas**: `p4block/ninfer-thor`
  (Jetson), `JCraigWasTaken/ninfer-gfx906`, `teo-mateo/ninfer-sm120`, `devan-carlin/ninfer`,
  `SV8ARJ/ninfer-windows-5090`, `2dameneko/ninfer-xx90-win`, `chillaheal/ninfer-win`,
  `YurunTao/ninfer-4080S-32G`, `ruwwww/ninfer-5060ti`, `Doelfke/ninfer-yarn` (43★ — YaRN for
  older layouts), `geoffwatts/ninfer-v100`, `natpate`'s MSVC peers, `cometkim/ninfer` (10 squashed
  branches — `feat/kernel-perf`, `feat/mtp7`, `feat/1m-context`, `feat/hyperquant`; squashed
  commits hide the mechanism, and its former role as an NVFP4 artifact supplier is retired since
  every port artifact is now built locally).

---

## 4. What is worth doing, in order

Each item names whether the GPU is needed to evaluate it. "GPU" means the item cannot be settled by
reading; a number taken without it is an estimate.

1. **#364/#365 — NVFP4 down-projection tile at verify widths.** 11 lines in
   `src/ops/linear_add/nvfp4/nvfp4_linear_add_a4.cu`, bit-identical output, −8..−13 % op and
   −1.6..−2.5 % DFlash2 round on a 5090 [third party], applicable to all five shipped images.
   **GPU**: re-run `ninfer_nvfp4_linear_add_bench` and one DFlash2 round on this port's binary.
2. **#385 — DFlash2 graph reservation.** The port has the exact code at
   `startup.cpp:893,899` and three lanes in the affected shape. Adopt a conditional/decoupled form
   in the port's own tree. **GPU**: startup plus `MemorySummary` on a DFlash2 lane.
3. **#374 — residual rounding width dependence.** Accuracy and determinism: one token's bits should
   not depend on its position in a chunk. Port both fork fixes (`87f06898`, `37a1ed49`,
   `c3383145`) with the width-sweep test (`f08572f3`). **GPU**: the sweep.
4. **#327 — reciprocal quantize on the plain-linear MMA route.** Small decode win, scoped route,
   measured on a 5090 [third party]. **GPU**: A/B on this port's route.
5. **#232 — MSVC `gdn_gating_proj` tolerance.** Run the test in `build-test`; if it fails, restore
   1.0e-3 or document. One constant, accuracy gate. **GPU**: only because it is an op test.
6. **#279 — `ChunkedSmallT` double KV sweep.** This port publishes draft=9 DFlash2 lanes (W=10) at
   long context, which is the band the issue measured at 1.75–1.96× attention cost [third party].
   Decide adaptive K-cap vs TokenTile-12/16 kernels, backed by an acceptance-vs-K measurement the
   port can now take. **GPU**.
7. **#283 — NVFP4 MoE expert profile.** +29.9 % prefill at 8k [third party] if a MoE artifact is in
   scope; needs the `divisors` container decision. **GPU**.
8. **#68 — MoE L2 prefetch.** +3.7 % decode on 35B MoE [third party]; MoE only; needs a cross-layer
   contract. **GPU**.
9. **#330 — A16 sliced-K knobs.** Cheap 5090 verdict: `bench/ops/nvfp4_linear_swiglu_bench` T=1..16.
   **GPU**.
10. **#355 — nvfp4 tiled prefill split.** Only if an nvfp4-KV lane is wanted; the port's fp8/k8v4
    routes show the pattern. **GPU**.
11. **#303 — multi-chunk prefill execution unit.** Design only; take a measurement first (the
    port's own prefill phase split via `tools/bench/report_serve_phases.py`). **GPU**.
12. **Forks, read-only first:** gzenz's staged-prefill lanes and YaRN-for-DFlash2 (`576e72ea`,
    `3a014fd6`), tmark00's two-stage GDN kernels (`853d268c`), Oceania2018's swiglu A16 chunking
    (`0e9066cf`). Each is one diff read before any GPU time.

Rejected without GPU time: upstream's 40-commit ops reorg (a different partition of the same code),
#292/#293 and #311 (no Q5/Q4 artifact), #277 (the port's default is already 8192), #167/#324
(already present), #320 (already fixed here), #264 (upstream's structure is not this port's), #221
(already present), #375 (inert on k8v4/fp8), #381 (local storage), #235/#233/#84/#59 (already
Windows), #294/#33 (already have gbnf).

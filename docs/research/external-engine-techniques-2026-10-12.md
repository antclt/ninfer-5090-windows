# External engine techniques this port might be missing — prefill, decode, accuracy

Research note, 2026-10-12. Written for the owner's question: which techniques do vLLM, SGLang,
TensorRT-LLM, llama.cpp, FlashInfer, FastDeploy, DeepGEMM, CUTLASS and FlashAttention ship that this
engine does not have, in three areas — **prefill throughput, decode token speed, and accuracy**.
Scope is the shipped product: Qwen3.8-27B dense/MoE, NVFP4/FP8 `.ninfer` artifacts, one RTX 5090
(`sm_120a`, 170 SMs), 1-8 resident lanes, 262,144-token context, paged KV, DFlash2/MTP speculative
decoding, decode CUDA graphs.

**Labels.** Every claim carries one of these, and every number says which one it is:

| label | meaning |
|---|---|
| **[measured there]** | the project ran it and published the figure; the workload travels with the number |
| **[read in source]** | read in the owning project's code, docs, PR or release notes; path and line given where it is a file |
| **[reported]** | asserted by a party that is not the project, and not reproduced here |
| **[in-tree]** | this repository's own prior measurement, cited to the note that holds it |

**Nothing here was measured on this machine's GPU.** No lane, artifact or flag was changed. The
applicability paragraphs are read from this tree's source; where a verdict depends on a runtime figure
it says so and the item is flagged **needs the GPU** in the final list.

**What this port already has**, so "missing" means something: chunked prefill with
`prefill_chunk = 8192` and 128-token alignment (`include/ninfer/types.h:165`,
`apps/cli/options.cpp:229`); five KV codecs selected **globally** by `--kv-dtype`
(`apps/cli/options.cpp:150`); split partitions for causal attention with `kMaxSplits = 256` and a
64-key floor (`src/ops/softmax_attention/common/causal_partition.h`); CUDA graphs for decode, the
draft model and MTP (`src/models/qwen3_5/program/decode.cpp:73`,
`src/models/qwen3_5/execution/draft.cpp:714`,
`src/models/qwen3_5/program/speculative/mtp.cpp:212`) and **none for prefill**; and a round that runs
the decode batch and then **one** prefill step for a single lane
(`src/runtime/engine/engine_core.h:2299-2352`) — prefill and decode tokens are not mixed into one
batch.

These are other projects' work to harvest, not defects to report to them and not code to contribute
back. Nothing below asks those projects to change.

---

## 1. Prefill

### P1. Prefill/context CUDA graphs (piecewise, breakable, bucketed)

**Mechanism.** Capture the transformer body (everything except the attention op, whose metadata
varies) as a CUDA graph; run attention eager; pad the iteration's token count to the next captured
bucket and replay. This removes host launch overhead from the context phase without needing a graph
per exact token count.

**Measured there.** vLLM PR **#10058** "[V1] Integrate Piecewise CUDA graphs", merged 2024-11-06
(<https://github.com/vllm-project/vllm/pull/10058>), reports `benchmark_latency.py`:
OPT-125M fp16 **0.41 s → 0.21 s**; Llama-3.1-8B bf16 **1.26 s → 1.04 s** (V0 with decode-only CUDA
graphs: 0.99 s). The PR also states the cost and the limits at the time: "The compilation + capturing
time takes at most ~17 secs"; "piecewise CUDA graphs are not compatible with custom ops … FP8 or
other quantizations are not supported". **[measured there]**

**Read in source.** TensorRT-LLM, `docs/source/features/torch_compile_and_piecewise_cuda_graph.md`:
"runs cudagraph-unsupported components (primarily attention) in eager mode while capturing and
replaying the supported parts with CUDA Graph to reduce context-phase launch overhead"; "padding is
mandatory for Piecewise CUDA Graph because context-phase token counts vary widely"; default capture
list `[2**i for i in range(8)] + [i for i in range(256, 3073, 256)]`; and the honest caveat: "Even
with Piecewise CUDA Graph enabled, you may still observe bubbles in the context (prefill) phase,
primarily due to the attention operator's substantial host-side overhead." SGLang now ships a
prefill runner with three backends (`python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`):
"breakable" (default on CUDA) "Captures the transformer body with one request slot, then replays it
with live batch metadata; multi-request prefill is supported by running attention metadata and the
LM-head/logits tail outside the captured body"; "full" captures one graph per token bucket and
"pads unused request slots with zero-length sentinels". **[read in source]**

**Applicability.** NInfer has no prefill graph — the only capture sites are decode, draft and MTP.
With 8192-token chunks the per-chunk launch overhead is amortized over a large batch, so the vLLM
1.26→1.04 s figure (a latency benchmark with small batches) does not transfer directly; the win here
would come from many small prefills or from smaller chunks, and it is measurable only on this
machine. Also note the vLLM limitation that mattered for a quantized engine: the first piecewise
implementation excluded custom ops and therefore FP8. **Needs the GPU** to decide whether it is worth
the capture complexity.

### P2. Chunked prefill as a token budget, with a starvation guard and optional mixing

**Mechanism.** One scheduler for both phases: each request carries `num_computed_tokens` and the
scheduler hands out a fixed token budget per step. Chunking is then just "how many of this request's
remaining tokens fit", and a long prefill can be capped so it does not monopolize the budget while
other requests wait.

**Read in source.** vLLM `vllm/config/scheduler.py`: `DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048`,
`enable_chunked_prefill: bool = True`, and two guard knobs —
`long_prefill_token_threshold` (default 0; "The cap is not applied when the request is the only one
in the batch, since there is no other request for it to starve") and
`long_prefill_token_threshold_adaptive` ("Floor the effective long prefill token threshold at a fair
share of the token budget: max_num_batched_tokens divided by the number of queued and running
requests"). SGLang exposes `prefill_decode_interval` ("The number of decode rounds to run after a
prefill batch before scheduling the next prefill") and `enable_mixed_chunk` ("Enabling mixing prefill
and decode in a batch when using chunked prefill", default `False`) in
`python/sglang/srt/arg_groups/fields/schedule.py`. **[read in source]**

**Measured there.** The vLLM V1 blog (2025-01-27,
<https://blog.vllm.ai/2025/01/27/v1-alpha-release.html>): the scheduler "removes the traditional
distinction between 'prefill' and 'decode' phases … Scheduling decisions are represented as a simple
dictionary, e.g. `{request_id: num_tokens}`"; V1 delivers "up to **1.7x higher throughput** compared
to V0 (without multi-step scheduling)" on Llama-3.1-8B/3.3-70B ShareGPT, and prefix caching costs
"less than 1% decrease in throughput even when the cache hit rate is 0%". **[measured there]**

**Applicability.** NInfer's chunk (8192) is four times vLLM's default; its scheduler already has a
prefill cursor and per-round budgets (`src/runtime/engine/scheduler.h:129-143`), and shipped lanes
run `--max-concurrency 1`, so the starvation guard has little to bite on. The genuinely different
mechanism is **mixing prefill and decode tokens into one forward batch**, which NInfer does not do:
its round runs the decode batch and then one prefill step (`engine_core.h:2299-2352`). vLLM does it
by default, SGLang offers it as `enable_mixed_chunk`, and FlashInfer ships a fused kernel family for
it (POD-Attention, "Fused prefill+decode for mixed batching", FlashInfer README). Whether it helps
here depends on whether a lane's decode and another lane's prefill currently serialize on the GPU —
**needs the GPU** to measure, and it is a scheduler change, not a kernel.

### P3. Cascade attention for shared prefixes

**Mechanism.** Split attention for a batch that shares a prefix: a multi-query kernel computes the
attention state against the shared prefix (KV resident in SMEM, loaded once), a single-query decode
kernel handles each request's unique suffix, and the two partial states are merged with a
renormalizing merge operator (the FlashAttention/Flash-Decoding log-sum-exp merge).

**Measured there.** FlashInfer blog "Cascade Inference", 2024-02-02
(<https://flashinfer.ai/2024/02/02/cascade-inference.html>): "up to **31x speedup** compared to the
baseline vLLM PageAttention implementation and **26x speedup** compared to FlashInfer batch decoding
operator without cascading on a H100 SXM 80GB", measured at 32768-token shared prefix with batch
size ≥128 and unique suffix ≤256; the blog states the benefit "increases as shared prefix length and
batch size grows and decreases as we increase unique suffix length". **[measured there]**

**Applicability.** NInfer's 1-8 lanes are two orders of magnitude below the batch size where the 31x
was measured, and the effect is explicitly batch-size-driven, so the payoff here is likely small.
NInfer's context cache already reuses prefixes across requests through the checkpoint path
(`PrefixReusePath::Checkpoint`, `include/ninfer/types.h:902`); cascade attention is a different
lever (KV read traffic within a decode batch). Lowest-priority prefill item; **needs the GPU** if
pursued.

### P4. The sm_120 attention reality: FlashAttention-4 is an SM80-class kernel on this arch

**Read in source.** `flash_attn/cute/flash_fwd_sm120.py` (header, lines 1-6): "SM120 (Blackwell
GeForce / DGX Spark) forward pass. SM120 uses the same SM80-era MMA instructions
(`mma.sync.aligned.m16n8k16`) but has a smaller shared memory capacity (99 KB vs 163 KB on SM80).
This module subclasses `FlashAttentionForwardSm80` and overrides the SMEM capacity check
accordingly." `flash_attn/cute/interface.py` imports `FlashAttentionForwardSm120` /
`FlashAttentionBackwardSm120` next to the SM100 classes, so the sm_120 path is real but is the
Ampere-style pipeline; the FA3 README still requires "H100 / H800 GPU", and the FA4 README describes
the CuTeDSL implementation as "optimized for Hopper and Blackwell GPUs (e.g. H100, B200)".
**[read in source]**

**Why this is a finding, not a candidate.** The techniques usually listed for prefill attention on
Blackwell — warp-specialized producer/consumer, TMA multicast, tcgen05 — are the sm_100/sm_103 path
and do not port to sm_120. On this card the attention frontier is the FA2-style mma pipeline plus the
99 KB SMEM budget, which this tree's tiled/grouped schedules already target. Two consequences: any
plan that assumes "adopt the Blackwell attention kernel" is wrong here, and the comparison target for
NInfer's prefill attention is FA4's sm_120 path (SM80 code with a 99 KB capacity check), not FA4's
headline numbers. No number to harvest; the fact is the deliverable.

### P5. MoE dispatch at prefill: which grouped-GEMM layout is even available on sm_120

**Mechanism.** Two published layouts for expert GEMMs. *Contiguous*: tokens are sorted by expert into
one tensor and each expert segment is M-block aligned, so one grouped GEMM walks the segments.
*Masked*: a mask says which portions are valid, which lets the kernel run under a CUDA graph whose
shape was captured before the CPU knew the per-expert token counts.

**Read in source.** DeepGEMM README: requirements are "NVIDIA SM90 or SM100 architecture GPU" — so
**DeepGEMM does not support sm_120**, and the masked variant exists precisely for graph-captured
decode: "During the inference decoding phase, when CUDA graph is enabled and the CPU is unaware of
the number of tokens each expert receives, we support masked grouped GEMMs. By providing a mask
tensor, the kernel computes only the valid portions." Its own headline (2025-04-18 news) is "up to
**1550 TFLOPS** on H800" — H800, not this card. **[read in source]**

CUTLASS, by contrast, has sm_120 grouped kernels: `CHANGELOG.md` records "Add ptr-array TMA
collective for tensor/token-scaled FP8 grouped GEMM Blackwell SM120/SM121 kernels … enabling
ptr-array grouped GEMM (MoE expert dispatch) with tensor- and token-level FP8 scaling"; "Support
Blackwell SM120 mixed input blockscaled grouped GEMM"; "exposes NVFP4/MX Grouped GEMM in the CUTLASS
Profiler"; and one figure — "Add support for 128x32xK and 128x64xK tile sizes for SM120 blockscaled
MMA collective builders, yielding up to **30% performance improvement on Blackwell SM121** related
kernels". **[read in source; the 30% is NVIDIA's own reported figure, SM121]**

**Applicability.** NInfer's `sparse_moe_prefill` already has a routed/tiled design — route jobs per
expert column tile, `expert_offsets`, packed index maps
(`src/ops/sparse_moe/prefill/sparse_moe_prefill.h:27-68`) — so the item is not "add grouped GEMM"
but "compare against CUTLASS's SM120 ptr-array collective and consider the masked layout for
graph-captured decode". DeepGEMM is not a candidate on this arch. **Needs the GPU** to compare.

### P6. Hybrid-model prefix caching: Marconi's admission policy and junction points

**Mechanism.** In hybrid models (attention + recurrent state, e.g. GDN), the recurrent state cannot
be rolled back, so partial-overlap cache hits are impossible and every cached entry is exact-match.
Marconi's contribution is *which* entries to keep: admission and eviction scored by forecast reuse
likelihood across a taxonomy of hit scenarios and by compute saved per byte of memory, rather than
by recency alone.

**Measured there.** Marconi (arXiv 2411.19379v3, abstract): "their use of in-place state updates for
recurrent layers precludes rolling back cache entries for partial sequence overlaps, and instead
mandates only exact-match cache hits; the effect is a deluge of (large) cache entries per sequence,
most of which yield minimal reuse opportunities"; the system "achieves up to **34.4x higher token hit
rates (71.1% or 617 ms lower TTFT)** compared to state-of-the-art prefix caching systems".
**[measured there]**

**Read in source.** vLLM's v1 scheduler carries the idea into code: a chunk is clipped "where Mamba
state must be cached" with the comment "Marconi shared-prefix junction: cache its state so sibling
requests sharing the prefix can reuse it"
(`vllm/v1/core/sched/scheduler.py`, `_mamba_block_aligned_split` at line 443, and the imported
`get_mamba_prefill_checkpoint_position` from `vllm/v1/kv_cache_interface.py`). **[read in source]**

**Applicability.** NInfer already caches GDN state at checkpoints, and its own tracker has the same
constraint documented upstream (`docs/research/open-items-research-2026-10-11.md` §5, "upstream PR
#335 replace the checkpoint catalog with a hybrid prefix cache"). The harvestable part is the
**policy**: which checkpoints to admit and evict, and stopping a chunk at a shared-prefix junction so
a sibling can resume. **Needs the GPU** to evaluate against the current checkpoint catalog.

---

## 2. Decode

### D1. Adaptive verification: an online acceptance estimator decides how much to verify

**Mechanism.** Score every draft position with its survival probability (the running product of
per-position acceptance estimates), then admit the globally best slots until a token budget — derived
from a step-cost table profiled at startup — is spent. Slots compete across requests, so a confident
request keeps its whole block while a doubtful one is trimmed after a token or two. vLLM's shipped
estimator is a per-position logistic regression on `logit(q_max)` (the draft's most likely token's
log-odds), refit online from the target's accept/reject outcomes; the feature must use `q_max` rather
than the sampled token's probability to stay lossless.

**Measured there.** vLLM PR **#52228** "[Model Runner V2][Spec Decode] Adaptive Verification without
a confidence head", merged 2026-09-14
(<https://github.com/vllm-project/vllm/pull/52228>). Workloads: concurrency 64, ≤2K in / 2K out,
temp 1.0 / top-p 0.95, natural EOS. Mean throughput with AV on: **+19.0%** MiMo-V2.5-Pro-FP4 +
DFlash, **+17.6%** Inkling-NVFP4 + MTP8, **+44.5%** Kimi-K2.5 + DFlash, and **+14.7%**
Muse-Glimmer-30B + **DFlash2** (TPOT −13.4%, accept len 4.43 → 4.14 on GSM8K). Accuracy: "GSM8K
accuracy is unchanged with AV on (pooled **−0.08pp over 5,276 paired questions**, McNemar p = 0.77)".
The estimator matches a trained confidence head ("tok/s +2.1%, median TPOT −3.9%, accepted/drafted
+4.70pp"). **[measured there]**

**Read in source.** vLLM `docs/features/speculative_decoding/adaptive_verification.md`: "The budget
itself comes from a cost model profiled at startup… picks the token count that maximizes expected
accepted tokens per second"; requirements — "The attention backend must tolerate device-decided query
lengths, since the CPU lengths only bound them from above" and "Full cudagraphs are required: step
costs are profiled from captured graphs, so `--enforce-eager` is rejected at startup". The same PR
records why raw `q_max` is not enough: it is overconfident (mean prediction 0.784 against observed
0.723; ECE 7× the fitted estimator's), and the fit provides calibration. **[read in source]**

**Applicability.** This is the strongest decode item: NInfer's DFlash2 is a **parallel** drafter, the
same class as the Muse-Glimmer-30B + DFlash2 row, and the mechanism's benefit grows exactly where
this port lives — variable acceptance and a fixed draft window of 4-9. It does require the verify
step to accept a device-decided query length, which is the same plumbing D3 is about. **Needs the
GPU.**

### D2. Draft-width control: batch-size schedule, early exit, post-draft pruning, and trees

**Mechanism, four variants.**
1. **By batch size (shipped):** pick `K` from a configured `[start_bs, end_bs, K]` schedule before
   drafting; `K = 0` turns speculation off entirely at high load.
2. **By confidence during drafting (DSL):** an autoregressive drafter stops early when confidence
   drops.
3. **After parallel drafting (D-Cut):** a parallel drafter emits all K drafts in one pass, then keeps
   a top-K subset by draft score before verification — either a fixed ratio or `auto`, a keep-length
   chosen from a warmup-profiled cost table. The bonus token is always retained.
4. **By tree (EAGLE-2):** build a context-aware draft *tree* rather than a chain, ranked by the draft
   model's own confidence, and verify the whole tree in one pass under a tree attention mask.

**Read in source.** vLLM `docs/features/speculative_decoding/dynamic_speculative_decoding.md` gives
the shipped schema — `"num_speculative_tokens_per_batch_size": [[1,64,3],[65,128,1],[129,512,0]]` —
and the limitation "Full Cudagraph only works with Model Runner V2. MRv1 only supports piece-wise
cuda graph with this feature". **[read in source]**

**Measured there (open PR — not merged).** vLLM PR **#47131** "Add D-cut: Adaptive Verification
Depth Pruning for Batched Speculative Decoding" (`mergedAt: null`,
<https://github.com/vllm-project/vllm/pull/47131>): 5 models × 5 datasets, greedy, 8×H20,
concurrency swept 4-64, reported at 32; speedup vs autoregressive decoding — Llama-3.1-8B **1.12x**
(unpruned DFlash-16 0.67x), Qwen3-4B **1.35x**, Qwen3-8B **1.21x**, **Qwen3.5-27B 1.25x** (unpruned
DFlash-16 0.94x; MTP baseline 1.31x), Qwen3.5-35B-A3B MoE **2.50x**; "D-Cut keeps **92% of DFlash's
τ** on average (85-97%)". Under temperature-1 sampling (Qwen3-8B, H20) D-Cut sustains **1.15x** over
AR where unpruned DFlash falls below AR (**1.68x** relative to DFlash). The PR also notes the
compatibility that matters for a graph-based engine: async scheduling "assumes a fixed draft width",
and full CUDA graph mode falls back to piecewise. **[measured by the submitter, open PR]**

**Measured there (tree drafting).** EAGLE-2 (arXiv 2406.16858v2, abstract): "we found that the
acceptance rate of draft tokens is also context-dependent … the draft model of EAGLE is well
calibrated: the confidence scores from the draft model approximate acceptance rates with small
errors", measured "speedup ratios **3.05x-4.26x**, which is **20%-40% faster than EAGLE-1**". EAGLE-3
(arXiv 2503.01840v3, abstract): "speedup ratio **up to 6.5x**, with about **1.4x improvement over
EAGLE-2**. In the SGLang framework, EAGLE-3 achieves a **1.38x throughput improvement at a batch
size of 64**." **[measured there]**

**Applicability.** NInfer has both drafter classes: DFlash2 is parallel (D-Cut's target) and MTP is
autoregressive (variant 2's target). NInfer's draft window is a fixed per-profile value (4-9), so a
per-step keep length is a real change. The Qwen3.5-27B row is the same target family as this
product's artifact, which makes it the most directly comparable figure in this note. **Needs the
GPU.**

### D3. The verify-width problem meets CUDA graphs

**Mechanism.** Dynamic verification and full CUDA graphs are in tension: a graph is captured for a
fixed shape, while adaptive verification wants a device-decided query length per step. The published
resolutions: capture graphs over padded buckets and let the kernel consume a device-side length
(vLLM's adaptive verification, which rejects `--enforce-eager` and profiles step costs *from the
captured graphs*); fall back to piecewise graphs when the dynamic path is on (vLLM's DSD doc and the
D-Cut PR); or keep a fixed length and lose the adaptation (TensorRT-LLM, below).

**Read in source.** TensorRT-LLM `docs/source/features/speculative-decoding.md`: "For all speculation
algorithms, when speculation is enabled, a single sequence of draft tokens with length `max_draft_len`
is created for every request. **There is currently no way to dynamically disable speculation**, thus
speed ups are only observable at low batch sizes." The same doc has the *tree* alternative to a wider
linear window: EAGLE-3 "dynamic tree mode … the drafter expands multiple candidate tokens at each
layer instead of a single token. This can improve acceptance rates compared to linear drafting at the
cost of additional compute per generation step", configured by `use_dynamic_tree`,
`dynamic_tree_max_topK`, `max_total_draft_tokens` (bounded by `topK × max_draft_len`). **[read in
source]**

**Applicability.** NInfer captures graphs for decode, draft and MTP, so the question to answer
locally is: what is captured for the *verify* step, and what does a variable accepted length or a
variable K invalidate? The vLLM pattern — pad to buckets, keep the length on the device, profile the
cost from the graphs — is the design to copy if the verify graph is to survive dynamic K. **Needs
local reading first, then the GPU.**

### D4. Model-free drafting is shipped on the GPU, and shipped *combined* with neural drafting

**Mechanism.** Suffix/n-gram proposers build their proposals by pattern-matching the prompt and prior
output; suffix decoding adds frequency counts and an adaptive number of tokens per request per step.
The combination with a neural drafter is shipped in two different ways: **selection** (the model-free
proposal overrides the neural draft when the match is long enough) and **concatenation** (the model
fills part of the draft budget, the pattern matcher the rest).

**Read in source.**
- vLLM ships both proposers and a GPU n-gram implementation: `vllm/v1/spec_decode/ngram_proposer_gpu.py`,
  `vllm/v1/spec_decode/suffix_decoding.py`; the docs say Suffix Decoding "can pattern-match against
  both the prompt and previous generations", "uses frequency counts", and "speculates an adaptive
  number of tokens for each request at each iteration"; `num_speculative_tokens` is only a maximum.
- TensorRT-LLM ships **selection**: the Suffix Automaton "is a model-free, GPU-based pattern-matching
  draft enhancer… SA is very accurate when it matches (exact pattern repetition), while neural methods
  are better for novel content — combining them gives the best of both worlds", combinable with MTP,
  EAGLE 3 and PARD via `use_sa_spec`, `sa_spec_threshold` (default 4, "minimum suffix match length
  required to override the neural draft").
- FastDeploy ships **concatenation**: PR **#3610** "Support new speculative decoding method 'hybrid
  mtp with ngram'", merged 2025-08-26 (<https://github.com/PaddlePaddle/FastDeploy/pull/3610>),
  "Support multi-mtp token added by some ngram token", configured as
  `{"method": "mtp", "num_model_steps": 2, "mtp_strategy": "with_ngram", "num_speculative_tokens": 5}`
  — MTP-2 plus ngram-3; kernels `ngram_match.cu`, `ngram_match_mixed.cu`.

**[read in source]** No measured number travels with these docs; vLLM's table calls n-gram "low to
medium gain".

**Applicability.** This repository's own research already concluded that the missing piece here is
the **consumer**, not the proposer: "the proposer side needs no new interface; the consumer does not
exist yet" (`docs/research/open-items-research-2026-10-11.md` §10, which also records that neither
vLLM's `NgramProposer` nor llama.cpp's lookup returns a tree). The external engines shipping the
combination confirms the design space is real. **Needs the GPU** to measure on this port's prompts.

### D5. DFlash2 acceptance: a measured reference on this exact artifact class

**Measured there.** vLLM's CI asserts acceptance and accuracy for the same target and drafter this
port uses — `nvidia/Qwen3.8-27B-NVFP4` + `z-lab/Qwen3.8-27B-DFlash2`, `num_speculative_tokens=7`,
GSM8K with 1319 questions, `max_num_seqs=32`: `expected_acceptance_len = 6.14 × 0.95` (≈5.83) and
`expected_accuracy = 0.90`
(`tests/v1/e2e/spec_decode/acceptance_rates/dflash/test_dflash.py`). The same file pins Qwen3-8B +
`z-lab/Qwen3-8B-DFlash-b16` at expected acceptance lengths from the DFlash paper's Table 1:
**mt-bench 4.24, humaneval 6.50, gsm8k 6.54 × 0.975**. **[read in source; measured in vLLM's CI]**

**Applicability.** This repo's own baseline note measures its DFlash2 acceptance "3.2x below" the
published numbers at the recommended width (`docs/research/dflash2-acceptance-baseline.md`), and the
DFlash paper's own abstract claims "over 6x lossless acceleration… up to 2.5x higher speedup than the
state-of-the-art speculative decoding method EAGLE-3" **[measured there]**. The vLLM CI figure is a
useful second reference point on the *same* quantized artifact, independent of the model card the
in-tree note already uses. **Needs the GPU** to close the gap.

### D6. Small-batch GEMM on sm_120: the tile is selected by M, and batch-invariance forbids split-K

**Mechanism.** At T=1..16 a linear layer reads the whole weight matrix per step, so the kernel is
weight-bandwidth-bound and the design question is how many CTAs can stream weights concurrently, not
tensor-core throughput. The field's answer on sm_120 is a family of small-M tile configs selected by
M, all TMA and all with cluster 1×1×1.

**Read in source.** vLLM `csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_sm120_fp8_dispatch.cuh`:
`if (M <= 16)` → TileShape 16×64×128 with a custom EpilogueTile M=16; `if (M <= 32)` → 32×64×128
with EpilogueTile M=32; `if (M <= 256)` → 64×64×128 Pingpong, carrying CUTLASS's own comment "SM120
Cooperative kernel requires Tile M >= 128. For M=64 tile, we use Pingpong schedule which is more
flexible with small tiles", plus "CUTLASS 3.x on SM120 currently restricts programmatic multicast
(Cluster > 1) … Reverting to 1x1x1 to ensure compilation". The NVFP4 SM120 kernel
(`csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_sm120_kernels.cu`) dispatches on
`mp2 = max(16, next_pow_2(m))`: ≤256 → a 256×128×128 tile, above → 128×128×128, both cluster 1×1×1;
its batch-invariant mode asserts a persistent tile scheduler with the comment "stream-K or split-K
would break numerical invariance". **[read in source]**

**Measured there.** No benchmark travels with these files. The figures this port has for the same
regime are its own: upstream #364/#365 — the NVFP4 W4A4 `linear_add` at `[5120,17408]` launches
`M32N64` at every A4 width up to 64 tokens ("80 CTAs" on a 170-SM card), and a halved tile with a
deeper pipeline was bit-identical and −8..−13% at T=8..32 at the op, with a commenter reporting it
slower end-to-end (`docs/research/open-items-research-2026-10-11.md` §1). **[in-tree]**

**Applicability.** NInfer's A16/A8 families carry their own schedules; the external precedent is the
*shape* of the fix — an M-indexed family of tile configs including an M≤16 arm with its own epilogue
tile, rather than one schedule for all widths — plus the batch-invariance constraint if
reproducible eval outputs are ever wanted. **Needs the GPU.**

---

## 3. Accuracy

### A1. Per-layer KV precision policy

**Mechanism.** Keep selected attention layer types (or explicit layer indices) at the model's native
KV dtype while the rest of the network uses the quantized cache dtype.

**Read in source.** vLLM `docs/features/quantization/quantized_kvcache.md`: "Some attention layer
types (e.g. sliding-window) are more sensitive to KV-cache quantization. The
`--kv-cache-dtype-skip-layers` flag leaves the specified layers at the model's native dtype while
keeping the rest of the layers under the chosen quantized dtype" — accepting either layer indices or
type names. The same doc documents the two FP8 schemes (per-tensor, and per-attention-head requiring
FA + llm-compressor calibration) and one further mechanism: with the FA3 backend and FP8 KV,
"attention operations are also performed in the quantized (FP8) domain… queries are quantized to
FP8". llama.cpp's granularity is per *model component* rather than per layer: `common/arg.cpp`
defines `-ctk/--cache-type-k` and `-ctv/--cache-type-v` for the target plus
`-ctkd/--cache-type-k-draft` and `-ctvd/--cache-type-v-draft` for the speculative draft model ("KV
cache data type for K for the draft model") — a second axis where the draft model's cache can be
quantized differently from the target's. **[read in source]**

**Applicability.** NInfer's `--kv-dtype` is one global choice
(`apps/cli/options.cpp:150`), so a per-layer policy is absent. The hybrid stack (GDN + full attention)
is exactly the case where a blanket KV dtype is a compromise, and vLLM's own example names the
sensitive class rather than proving it on this model. **Needs the GPU** to decide which layers, and
that experiment needs the accuracy protocol in A5 to be resolvable.

### A2. RoPE-carrying K dimensions at higher precision

**Mechanism.** Part of each K vector carries positional information. Quantizing that part with the
rest of the vector is where KV quantization error concentrates; two published answers are to quantize
K *before* RoPE is applied (KVQuant), or to store the RoPE dimensions in a higher-precision format
inside an otherwise low-bit row (TensorRT-LLM's `skip_rope_quantization`, with a 2FP4 option for the
residual dimensions on DeepSeek-V4).

**Measured there.** KVQuant (arXiv 2401.18079v6, abstract): "Pre-RoPE Key Quantization, where we
quantize Key activations before the rotary positional embedding to mitigate its impact on
quantization"; measured result "< 0.1 perplexity degradation with 3-bit quantization on both
Wikitext-2 and C4". KIVI, a separate paper, measured that the key cache should be quantized
per-channel at all (A3). **[measured there]**

**Read in source.** TensorRT-LLM `docs/source/features/kv-cache-compression.md`:
`skip_rope_quantization` — "true: copy the supported target RoPE part unchanged at the hot-cache
precision, such as FP8 or BF16. The remaining values use NVFP4"; DeepSeek-V4 defaults to
`nvfp4_residual_dim = 64` using "2FP4 for its 64 RoPE values"; and the availability line names this
port's model family: "Skipping RoPE is available for DeepSeek-V4, GLM-5 (`glm_moe_dsa`), and the
**Qwen3.5 series**." The same page's cold-page NVFP4 path requires "compute capability SM100, SM103,
or SM107" — so the storage-tier implementation is **not available on sm_120**, only the idea is.
**[read in source]**

**Applicability.** NInfer's K codecs quantize whole rows (fp8 row256 with a Hadamard preparation,
int8 group-64, NVFP4 group-16 — `src/ops/kv_cache/*`), so a RoPE-dimension split is a layout and
kernel change. It is the most model-aware accuracy lever found in this survey, and it is already
sanctioned for this model family by a vendor. **Needs the GPU.**

### A3. KIVI: the K quantization axis is the token axis, and the recent tokens stay in FP16

**Mechanism.** KIVI's measured element distribution is asymmetric: keys carry channel-wise outliers,
values token-wise ones. So K is quantized with a scale per channel computed over a group of
consecutive **tokens**, V with a scale per token computed over groups along the **head dimension**,
and the most recent `residual_length` tokens are kept in FP16 as a high-precision tail.

**Measured there.** KIVI (arXiv 2402.02750v2, abstract): "Our findings indicate that the key cache
should be quantized per-channel, i.e., group elements along the channel dimension and quantize them
together. In contrast, the value cache should be quantized per-token"; result — Llama/Falcon/Mistral
"maintain almost the same quality while using **2.6× less peak memory**". **[measured there]**

**Read in source.** The mechanism is explicit in their code: `models/utils_quant.py`'s
`process_input_by_channel` flattens the KV cache to `[head_dim × heads, tokens]` and groups along the
**token** axis (`input_flatten = input.view(-1, num_features).transpose(0, 1)`), while
`quantize_and_pack` (the V path) groups along the last dimension within a token;
`models/llama_kivi.py` keeps `residual_length` recent tokens in FP16
(`assert self.residual_length % self.group_size == 0`), and the README calls it "the number of recent
fp16 tokens". **[read in source]**

**Applicability.** NInfer's `int8_g64_codec.cuh` is a "Signed int8, **per-token** G64 KV-cache codec"
with a Hadamard rotation as the outlier mitigation (`src/ops/kv_cache/hadamard_d256.cuh`) — the
value-style axis applied to K as well. KIVI's K axis (per channel, across a token group) and its FP16
tail are both different mechanisms; the token-group axis also means a group cannot be finalized until
it is full, which a paged cache could absorb by aligning groups to pages while NInfer writes per
token. Whether it beats a Hadamard-rotated per-token G64 on this model is an open, measurable
question, and it is a kernel/layout change. **Needs the GPU.**

### A4. What did *not* pay: a 4-bit V cache bought capacity and lost decode throughput

**Measured there.** vLLM PR **#38479** (merged 2026-04-15), `turboquant_k8v4` on Qwen3-4B:
GSM8K **0.860 vs 0.900** unquantized baseline (200 questions, NIAH 100%), and decode throughput
**79-100% of baseline** (short-decode 8977 → 7113 tok/s; TPOT 11.9 → 15.0 ms). **[measured there;
compiled in `docs/research/kv-dtype-evidence.md`, which also records that the −4.0 pp is 8/200
questions ≈1.9σ, that no FP8-KV control arm was run, and that this tree's k8v4 is a different codec
(FP8 K + NVFP4 V) than vLLM's.]**

**Why it is here.** It is the cleanest external counter-example to "fewer KV bytes is a decode win":
the memory saving was real (2.6x vs BF16, 1.31x vs FP8) and the decode path was slower, in the
project's own numbers. The same page records that vLLM's user-facing docs document **FP8 only**, and
that its own accuracy tracker still lists "Publish recommended config table" as unchecked.
**[read in source]**

**Applicability.** NInfer ships `k8v4` as a lane option; before it becomes a default for
capacity reasons, this is the prior to beat. **Needs the GPU** (decode throughput and quality on
this artifact).

### A5. How the field measures quality for a precision or spec-decode change

**Mechanism.** Gate the change on a task accuracy with a paired design and a fixed seed, not on
perplexity alone; report the acceptance/quality trade in the same run.

**Measured there.**
- vLLM's DFlash CI gates the quantized target on **GSM8K accuracy ≥ 0.90 over 1319 questions**
  alongside acceptance length, for Qwen3.8-27B-NVFP4 + DFlash2
  (`tests/v1/e2e/spec_decode/acceptance_rates/dflash/test_dflash.py`). **[read in source]**
- vLLM PR #52228 uses a **paired McNemar** design for the same question: "same 1319 GSM8K questions
  and the same per-request seed in both arms … Pairing lets McNemar's exact test run on the
  discordant answers, which is far more sensitive at n=1319 than comparing two independent accuracy
  rates", pooled **−0.08pp over 5,276 paired questions, p = 0.769** for the AV change. **[measured
  there]**

**Applicability.** This repo's own `docs/research/task-accuracy-protocol.md` derives the same
design (paired McNemar; 258 paired items for ~4-5.5 pt at 80% power) and notes that perplexity hides
task damage (the QHR W4A4KV4 32B model loses 3.9 pt on AIME-120 while GSM8K moves 0.0). The external
practice adds two details worth copying: **pin the same seed in both arms**, and **run the accuracy
gate in the same CI job as the acceptance gate**. No GPU needed to adopt the design; the GPU is
needed to run it.

---

## 4. Cross-cutting applicability facts (negative results included)

- **DeepGEMM does not support sm_120** ("Requirements: NVIDIA SM90 or SM100 architecture GPU",
  README). Its MoE layouts are still the clearest description of the contiguous/masked split.
- **TensorRT-LLM's KV compression paths require SM100/SM103/SM107**; on sm_120 only the *idea* of
  skipping RoPE quantization ports.
- **FlashAttention-4's sm_120 path is an SM80-class kernel** (99 KB SMEM), not the tcgen05/TMA path
  its Blackwell headline describes.
- **vLLM's first piecewise CUDA graph excluded custom ops, and therefore FP8** (PR #10058);
  whether that still holds in current `main` was not verified here.
- **Dynamic verification and full CUDA graphs conflict** in vLLM today (DSD needs Model Runner V2 or
  piecewise; D-Cut forces the piecewise fallback and is incompatible with async scheduling).
- **TensorRT-LLM cannot dynamically disable speculation** ("There is currently no way to dynamically
  disable speculation"), which is the capability vLLM's DSD/AV provide.
- **A 4-bit V cache is not a free win**: vLLM's own merged measurement has it slower in decode and
  −4.0 pp GSM8K on 200 questions (A4).

## 5. Provenance

| source | what was read |
|---|---|
| vLLM PR #10058 (merged 2024-11-06) | piecewise CUDA graph integration; latency table |
| vLLM v1 blog 2025-01-27 | scheduler model; 1.7x throughput; prefix-cache overhead |
| vLLM `vllm/config/scheduler.py`, `vllm/v1/core/sched/scheduler.py` | chunk budget defaults; Marconi junction; mamba chunk clipping |
| vLLM `docs/features/quantization/quantized_kvcache.md` | FP8 KV schemes; skip-layers; FP8-domain attention |
| vLLM `docs/features/speculative_decoding/{dynamic_speculative_decoding,adaptive_verification,README,n_gram,suffix}.md` | DSD schema; AV requirements; method table; suffix proposer |
| vLLM PR #52228 (merged 2026-09-14) | online acceptance estimator; +14.7…+44.5%; McNemar −0.08pp |
| vLLM PR #47131 (open) | D-Cut; Qwen3.5-27B 1.25x vs AR; τ retention 92% |
| vLLM `tests/v1/e2e/spec_decode/acceptance_rates/dflash/test_dflash.py` | Qwen3.8-27B-NVFP4 + DFlash2 acceptance 6.14×0.95, accuracy 0.90 |
| vLLM `vllm/v1/spec_decode/{dflash.py,ngram_proposer_gpu.py,suffix_decoding.py}` | parallel DFlash drafting; GPU proposers |
| vLLM `csrc/.../scaled_mm_sm120_fp8_dispatch.cuh`, `nvfp4_scaled_mm_sm120_kernels.cu` | M-indexed small-batch tiles; Pingpong vs Cooperative; batch-invariance vs split-K |
| vLLM PR #38479 (merged 2026-04-15) | k8v4 GSM8K 0.860 vs 0.900; decode 79-100% of baseline |
| SGLang `runner/prefill_cuda_graph_runner.py`, `arg_groups/fields/schedule.py` | prefill graph backends; chunk/mixing knobs |
| TensorRT-LLM `docs/source/features/{speculative-decoding,kv-cache-compression,torch_compile_and_piecewise_cuda_graph,overlap-scheduler}.md` | dynamic tree; SA; cold-page NVFP4 + skip-RoPE; piecewise graph |
| llama.cpp `common/arg.cpp` | `-ctk/-ctv` and draft-model `-ctkd/-ctvd` KV types |
| FlashInfer blog 2024-02-02 | cascade attention; 31x at batch ≥128 |
| FlashInfer README | sm_120 support; POD-Attention; FP4 GEMM |
| DeepGEMM README | SM90/SM100 only; contiguous/masked MoE layouts |
| CUTLASS `CHANGELOG.md` | SM120 grouped FP8/NVFP4 kernels; SM121 30% tile figure |
| FlashAttention `flash_attn/cute/{interface.py,flash_fwd_sm120.py}` | sm_120 = SM80-class path, 99 KB SMEM |
| KIVI `models/{utils_quant.py,llama_kivi.py}` | per-channel K grouped along the token axis; FP16 residual tail |
| FastDeploy PR #3610 (merged 2025-08-26) | hybrid MTP + ngram concatenation |
| arXiv abstracts: 2602.06036 (DFlash), 2503.01840 (EAGLE-3), 2406.16858 (EAGLE-2), 2402.02750 (KIVI), 2401.18079 (KVQuant), 2411.19379 (Marconi) | mechanism and measured headline for each |

## 6. What is worth doing, in order

Ordered by evidence strength × fit with this engine. "GPU" marks items whose verdict requires a
measurement on this machine's card; everything else is reading or implementation work.

| # | item | why it ranks here | GPU? |
|---|---|---|---|
| 1 | **Adaptive verification / draft trimming** (D1, D2) | The strongest external evidence in this survey, measured on the same drafter class: +14.7% on DFlash2, up to +44.5% on DFlash, accuracy unchanged (5,276 paired questions). NInfer's fixed window 4-9 is exactly the regime it targets. | yes |
| 2 | **Verify-width ↔ CUDA-graph audit** (D3) | Prerequisite for item 1: find what the verify graph captures today and what a variable accepted length or K invalidates. Local reading first. | yes |
| 3 | **DFlash2 acceptance comparison** (D5) | This repo already measures 3.2x below the published baseline; vLLM's CI gives a second, same-artifact reference (≈5.83 at K=7). Cheap to check against the existing acceptance instrument. | yes |
| 4 | **Per-layer KV precision policy** (A1) | Absent here (one global `--kv-dtype`), and the hybrid stack is the case it exists for. Implementation is a policy change; the evidence for *which* layers needs the A5 protocol. | yes |
| 5 | **RoPE-dimension K precision** (A2) | Vendor-sanctioned for the Qwen3.5 series, and backed by KVQuant's measured pre-RoPE finding; the largest model-aware accuracy lever found. Layout + kernel change. | yes |
| 6 | **Prefill CUDA graphs** (P1) | Three engines ship it and vLLM measured 1.26→1.04 s on a small-batch latency benchmark; the win here is unproven at 8192-token chunks, so measure first. | yes |
| 7 | **Mixed prefill+decode batching / budget guard** (P2) | The field's default (vLLM) or opt-in (SGLang); NInfer runs them in separate steps. Scheduler change with a clear measurement. | yes |
| 8 | **Model-free proposer with a tree consumer** (D4) | Already an open item in this repo; the external engines confirm the two combination designs (selection, concatenation). Consumer work, then measure. | yes |
| 9 | **KIVI per-channel K and FP16-tail experiment** (A3) | A measured paper result that contradicts this tree's per-token axis, plus a high-precision tail mechanism; kernel work before it can be tried. | yes |
| 10 | **Cascade attention** (P3) | Measured 31x at batch ≥128; NInfer's 1-8 lanes are far outside that regime. Lowest priority. | yes |
| 11 | **CUTLASS SM120 grouped-GEMM reference for MoE** (P5) | DeepGEMM is not available on sm_120; CUTLASS's ptr-array SM120 collective is the porting reference, with one NVIDIA-reported 30% figure on SM121. | yes |

Items that need no GPU: none of the *verdicts* — every one of the eleven above ends in a measurement
on this card, which is the honest summary of a survey of other projects' speed and accuracy claims.
The reading work (D3 audit, A5 protocol design, D4 consumer design) can start without it.

Two sections are applicability facts rather than actions and are therefore not in the list: **D6**
(small-batch tiles are an M-indexed family, and batch-invariance forbids split-K) and **P4** (the
sm_120 attention frontier is the SM80-style pipeline, not the Blackwell one). Both bound what a
kernel port can be, and both are reflected in the items above.


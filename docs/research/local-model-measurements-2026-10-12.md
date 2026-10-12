# What to measure on a local model, beyond tokens/s and accuracy

Written 2026-10-12 for this product's deployment shape: one user, one GPU, one resident model, 1-8
lanes. The question was what the *rest* of the measurements are, given that token speed and accuracy
are known. Sources are cited inline; each section says what this port already measures and what it
does not, because a metric list is only useful here if it names the instrument.

## 1. Latency is three numbers, not one, and the tail is a fourth

The field's framing is settled and identical across the primary sources: **TTFT** (time to first
token, i.e. prefill + queueing), **TPOT** or **ITL** (time per output token / inter-token latency,
i.e. steady-state decode), and **end-to-end latency** = TTFT + TPOT x output tokens
([Databricks, LLM Inference Performance Engineering](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices);
[NVIDIA NIM benchmarking metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html)).

Three things about that framing are easy to get wrong, and each is documented in a primary source:

* **Tools disagree on the definition.** NVIDIA's own note: "tools differ on whether TTFT is included
  in the average. AIPerf excludes TTFT" while LLMPerf includes it, and LLMPerf was later patched to
  exclude it "to keep it consistent with how other industry standard performance benchmarks are
  done" ([NVIDIA NIM](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html);
  [AWS Neuron benchmarking guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/llm-inference-benchmarking-guide.html)).
  A TPOT figure without its definition is not comparable.
* **Per-request and token-weighted means differ.** vLLM reports average TPOT as the mean of each
  request's TPOT (requests weighted equally) and average ITL as token-weighted
  ([Anyscale](https://docs.anyscale.com/llm/serving/benchmarking/metrics)).
* **The input/output length distribution is part of the number.** NVIDIA's methodology "emphasizes
  input and output sequence-length distributions because prompt length changes prefill work, memory
  requirements, and latency characteristics... If one provider benchmarks 128-token prompts and
  another uses 16K-token prompts, their TTFT numbers are not measuring equivalent jobs"
  ([Hivenet](https://www.hivenet.com/post/llm-inference-metrics-ttft-tps)).

**The tail is the fourth number.** MLPerf Inference's server scenario scores on **99th-percentile**
TTFT and TPOT, and for long-context work it set 99th-percentile TTFT of 6 s and 99th-percentile TPOT
of 175 ms ([MLCommons, MLPerf Inference v5.0](https://mlcommons.org/2025/04/llm-inference-v5)). A
mean hides the stalls a user actually feels.

**Here**: `tools/bench/report_serve_phases.py` reads the phase split by name from the request log
(`prepare`, `prefill`, `ttft`, `total`, `queue_wait`), `tools/bench/probe_prefill_floor.py` isolates
the prefill floor, and `tools/bench/first_request_lane.py` measures the cold first request. Percentiles
are **not** reported anywhere - the log carries per-request values, so the p95/p99 is computable but
nobody computes it.

## 2. Throughput is a curve against concurrency, not a point

Aggregate tokens/second across requests, and requests/second, both measured as a function of
concurrent requests - the standard framing again
([Databricks](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)).
For a local single-user deployment the curve matters at its *left* end (does batching hurt latency at
1-2 concurrent?) rather than at its peak.

**Here**: the request log's `queue_wait` is the local signal, and `tools/release/v3_profile_matrix.py`
runs lanes at fixed concurrency. The curve is not published.

## 3. Prefill rate is a different number from decode rate, and it is the one that scales

Prompt-processing tokens/second is measured separately, because prefill work grows with prompt length
while decode does not. This port has the sharpest reason to care: its own captures show the prefill
cost decaying with length (`profiles/nsys/prefill-decay-2026-10-08/`, filed VOID by its own `VOID.md`),
and the phase split puts a prefill floor at ~46 ms on this card.

**Here**: `ninfer_bench`'s prefill columns, the nsys captures under `profiles/nsys/`, and
`report_serve_phases.py`'s `prefill` field.

## 4. Memory: bytes per token is the single source of truth, and the context that fits is the product

The KV cache costs `2 x layers x kv_heads x head_dim x bytes` per token, and for a 70B-class model at
BF16 that is ~0.33 MB/token, so a 128K context needs ~42 GB for one session
([GMI Cloud](https://www.gmicloud.ai/en/blog/kv-cache-optimization-for-llm-inference-how-cache-aware-serving-reduces-cost-and-latency);
[DigitalApplied](https://www.digitalapplied.com/blog/how-much-vram-run-llm-quantization-kv-cache-context-2026)).
Two consequences a local deployment lives with:

* **The effective context is set by VRAM, not by the model's advertised window**: "A model that
  'supports' 128K context but runs on a 12 GB GPU may be limited to 14K tokens in practice"
  ([SitePoint](https://www.sitepoint.com/kv-cache-survival-guide-local-llms)).
* **Spilling to host memory is a cliff, not a slope**: "holding a 30B Q4 model fully resident instead
  of spilling to RAM is the difference between roughly 45 t/s and single digits"
  ([Pandit](https://bhavishyapandit9.substack.com/p/best-local-llm-for-every-hardware-tier)).
  Quantizing the cache buys context at a quality cost, which is a *trade* and needs both numbers.

**Here**: this port's per-token figures are already recorded per KV format - `fp8` 516 B, `k8v4` 402 B,
`nvfp4` 288 B, `bf16` 1024 B at D256 (`docs/research/kv-dtype-evidence.md`) - and
`tools/release/check_host_kv.py` compares the reservation against what a lane can serve. The
VRAM-vs-context curve is not published as a curve.

## 5. Quality has to be measured at the shipped quantization, per domain

Perplexity is a cheap corpus-level proxy and task accuracy is the real measure; both must be taken at
the quantization and KV format that ships, because that is where the quality is actually spent.
This port already treats them as a pair: a KV-format change carries "+0.144% corpus perplexity overall
and +0.202% on the worst domain" (`tools/release/profiles.py`), and the quality axis is weighted by
the same code-weighted rule as performance (`CONTEXT.md`).

**Here**: `ninfer-perplexity` and the capability suite, with the corpus figures quoted per lane.

## 6. Long-context quality is its own axis, and retrieval understates it

Single-needle retrieval is a weak instrument: "Despite achieving nearly perfect accuracy in the vanilla
Needle-in-a-Haystack (NIAH) test, all models exhibit large performance drops as the context length
increases", and every model in that study failed to hold its 4K baseline at its claimed length
([RULER](https://medium.com/@techsachin/ruler-benchmark-to-evaluate-long-context-modeling-capabilities-of-language-models-7eb13a269e36)).
The practical gap is measured: "every model's RULER score is 10-25 points below its NIAH-2
single-needle score at the same context length", and multi-needle tasks drop further - "GPT-5.5 drops
22 points, Opus 4.7 drops 33 points" at 1M
([DigitalApplied](https://www.digitalapplied.com/blog/long-context-retrieval-needle-in-haystack-2026)).
The rule to take from it: weight multi-needle scores over single-needle for any workload that
integrates information across a long context.

**Here**: long-context *perplexity* is measured (the corpus at 64K in the KV-format A/Bs), but there
is **no NIAH or RULER-style accuracy instrument** in this tree. That is a gap, not an oversight
someone already covered.

## 7. Speculative decoding's own numbers: acceptance rate and tokens per round

For any deployment that drafts, acceptance is a first-class metric: it is what converts a drafter into
speed, and it moves independently of the kernels. This port measures it everywhere -
`tools/bench/realtext_acceptance.py` (real text, greedy, the lane's own flags),
`tools/bench/lane_fingerprint.py`'s `accept` column, and the request log's
`dflash2 accepted N/M` line - and treats a collapse in it as a defect to attribute (this session's
0-of-1566 case was a real one, from a staging-swizzle substitution).

## 8. Cache effectiveness: hit tokens and the path that served them

Prefix caching is a local-serving lever with a directly measurable effect on TTFT. This port records
`prefix_cache_hit_tokens` and the reuse *path* per request, and refuses to conclude from hit counts
alone: `tools/bench/check_shared_prefix_reuse.py` requires an exact repeat that hits and a shuffled
conversation that does not.

## 9. Stability: spread, determinism, and long-run behaviour

Three separate things, all measurable here:

* **Run-to-run spread** - this card reads 10.3% session-to-session on one lane, against within-run
  spreads of 0.3-5.1% (`tools/bench/lane_fingerprint.py`'s docstring), which is why any effect below
  ~8% needs interleaving and repetition to establish.
* **Determinism across paths** - `tools/bench/probe_cross_path_determinism.py` compares the cached and
  cold paths on the same continuation.
* **Soak and abort behaviour** - a 220-cycle soak with hourly evictions is recorded in
  `docs/research/silent-shell-death-2026-10-10.md`, and `tools/bench/probe_cancellation.py` checks
  that an abandoned request frees its slot.

## 10. Energy: tokens per joule, which is power divided by throughput

Inference dominates the energy lifecycle (60-90% of production draw), so energy per token is the
efficiency metric that matters, and it is reported as **joules per token** or **tokens per joule**
([John Snow Labs](https://www.johnsnowlabs.com/tokens-per-joule-how-to-quantify-and-reduce-the-energy-footprint-of-clinical-llm-inference)).
Published figures for scale: ~0.385-0.684 J/token for four code models on HumanEval
([SustainableSE](https://luiscruz.github.io/course_sustainableSE/2025/p1_measuring_software/g23_llm_inference.html)),
~3-4 J/token for a 65B, and 0.12-0.20 J/token for Llama-3.1-8B at batch 64
([ML.ENERGY](https://ml.energy/blog/measurement/energy/llm-inference-energy-a-longitudinal-analysis)).
The relation to hold onto: energy per token is power / throughput, so a throughput win is an energy win
only if power grows more slowly - in that study, 3-5x throughput bought 15-41% lower energy per token.

**Here**: there is **no energy instrument**. `nvidia-smi --query-gpu=power.draw` (NVML) is the local
route, and it is not wired to anything.

## 11. Agentic reliability, for a local model driving tools

When the model runs an agent, the failure modes change shape: tool-call validity, over-refusal, and
state drift across a long session
([Towards a Science of AI Agent Reliability](https://arxiv.org/html/2602.16666v3)). The measurable
local version is narrow and testable: does the tool-call parser accept and correctly interpret what the
model emits, and does a long session's state stay consistent.

**Here**: `src/models/qwen3_5/frontend/tool_call_parser.cpp` exists with schema and oracle tests, and
the capability suite exercises real tasks; state drift over a long session is not measured.

## The short version for a local, single-user deployment

If only four things can be measured, they are: **TTFT and TPOT with their percentiles** (section 1),
**bytes per token and the context that fits** (section 4), **quality at the shipped quantization per
domain** (section 5), and **run-to-run spread** (section 9) - because on one card the binding
constraints are latency and VRAM, not aggregate throughput, and the spread is what tells you whether
any of the other numbers mean anything. Everything else here is a refinement of those four, and the
two this port cannot currently measure are **long-context accuracy beyond perplexity** and **energy**.

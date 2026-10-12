# Where this port's time actually goes: a measured baseline, 2026-10-12

Taken before deciding what to adopt from upstream, the forks or other engines, so that a recommendation
has something to be judged against. Every number here was **measured on this machine** with the shipped
QUASAR dflash2 artifact, unless a row says otherwise.

## What was measured, and how

    ninfer_bench --weights C:\AI\models\qwen3_8_27b_nvfp4qat.v3.ninfer --kv-dtype k8v4 \
        --spec dflash2 --draft-tokens <depth> --lm-head-draft --prefill-chunk <chunk> ...

`ninfer_bench` drives the public `ninfer::Engine` directly, so this is the product route and not an Op
microbenchmark. CSVs are under `profiles/bench/width-sweep-2026-10-12/` on this machine (gitignored);
the commands above regenerate them. Columns are read by header name, never by position: the report has
83 columns and one empty field.

The artifact is 17.65 GiB. The lane's own memory at pp8192/k8v4 reads: weights 16.94 GiB, sequence
0.87 GiB, workspace 0.78 GiB, CUDA-graph allowance 0.57 GiB, KV payload 0.20 GiB.

## Prefill

| prompt | prefill tok/s | sd | prefill s |
|---|---|---|---|
| 512 | 10,959 | 74 | 0.047 |
| 2,048 | 13,462 | 44 | 0.152 |
| **8,192** | **14,525** | 5 | 0.564 |
| 32,768 | 11,352 | 17 | 2.887 |
| 65,536 | 8,962 | 58 | 7.313 |

Prefill peaks at exactly the shipped chunk size and then falls: **−22% at 32k and −38% at 64k** relative
to the peak. The fall is not the chunk. Swept at 32k/64k, with the shipped 8192 as the control:

| chunk | pp32768 tok/s | pp65536 tok/s |
|---|---|---|
| **8,192 (shipped)** | **11,352** | **8,962** |
| 2,048 | 9,569 | 7,388 |
| 4,096 | 11,171 | 8,641 |
| 16,384 | 11,393 | 8,871 |

So the shipped chunk is the best of four and the long-prompt fall is something else — most likely
attention's growing share of a chunk's work, which is physics rather than a setting. **What this does not
establish** is where the time goes inside a long-prompt chunk: that needs a kernel-level profile
(`--profile-measured` brackets a repetition with the CUDA profiler API; `profiles/nsys/` does not exist
on this machine), and it is the profile that would say whether upstream's tiled-prefill and split-KV
prefill work applies here.

## Decode, and what speculative decoding is worth

| shape | with dflash2 depth 9 | without spec | ratio | acceptance |
|---|---|---|---|---|
| pp512 + tg128 | 137.3 tok/s | 82.1 | 1.67× | 13.3% |
| pp8192 + tg128 | 287.4 tok/s | 81.1 | **3.54×** | 40.2% |

The pp512 row's 13.3% acceptance is the bench corpus, not the lane: this repository already recorded
that the tiled corpus reads 0.085–0.129 on a one-token seed, and real domains sit at 0.44–0.55. The
pp8192 row is the one to quote.

## The draft window: a corpus result that real text reverses, and a setting already decided

This section first read "+14.3%, ready to take". It is kept with the correction because the way it was
wrong is the point, and because this repository's rule is to preserve an adverse finding rather than
quietly drop it.

**What the bench corpus said.** `ninfer_bench -pg 8192,128`, dflash2, k8v4, interleaved, three paired
rounds: depth 9 280.6/288.4/289.0 tok/s against depth 11 320.2/330.3/330.3, +14.1 to +14.5%, with each
arm's own spread at 2.1-3.1%. Mechanism from the same runs: 0.91 to 1.07 tokens per round over 140 and
120 rounds.

**What real text says.** `tools/bench/realtext_acceptance.py --lane quasar_v3_dflash2_vision`, real text,
greedy, three prompts of 12,000 characters, two interleaved rounds per depth:

| depth | round 1 | round 2 | accepted | drafted | acceptance |
|---|---|---|---|---|---|
| **9 (shipped)** | **241.2 tok/s** | **240.5** | 1,139 | 3,515 | **32.40%** |
| 11 | 232.0 | 229.6 | 1,128 | 4,394 | 25.67% |

Depth 11 is **−4.5% on decode and −6.7 points of acceptance**, and the accepted count *falls* while the
drafted count rises 25%: the extra positions are overhead. The instrument is tight — acceptance identical
to four decimals across rounds, decode within 1% — so the reversal is about five times its own spread.

**Why the two disagree, in the instrument's own words.** `realtext_acceptance.py` exists because "a bench
corpus cannot answer this question from either end of its range... the same corpus behind a real prompt
saturates it (0.991 in four consecutive runs), because the corpus is synthetic and tiled. Real domains sit
at 0.44-0.55". The corpus arm read 40.2% at pp8192 — near that band but not in it — and this real-text arm
reads 32.4%. The synthetic corpus is easier to predict than the text it stands in for, and at these widths
that difference is larger than the effect being measured.

**And the setting is already decided by a better measurement than either.** All four DFlash2 lanes' depths
come from multi-domain sweeps at served length, decided by **maximin over domains** — the width that
maximises the *worst* domain — which is the rule a single-domain measurement cannot satisfy:

* **Swift (7)**: "Width 7 was measured against every window 1-15 on the code domain and re-measured on four
  others: 13 is 33% faster on code and slower on chinese, prose and dialogue, so the width that maximises
  the worst domain is the one already shipped."
* **NVIDIA (9)**: "depth 9 leads by 13.2% on 36,000 characters of real code (258.5 against 228.3) and
  loses 8.9% on 36,000 characters of Chinese (295.9 against 324.8). Maximin over the two served domains
  takes depth 9."
* **ninfer (9)**: swept against depth 7 on all five domains, then re-measured through the matrix's widths
  protocol with three interleaved rounds inside one lane each — 9 leads on all three treatments, and the
  greedy readings that disagreed were single-run comparisons that "cannot resolve anything this size".
* **QUASAR (9)**: moved 7 to 9 on the served regime, where "prose is the known cost at −4.2%".

So the corpus result here is not a discovery: it reproduces the documented *code-domain* effect, which
those lanes already trade against their worst domain. **No change is made**, and the honest residue is one
gap: widths 11-13 have not been measured per domain for QUASAR, ninfer or NVIDIA, so a future sweep could
still find something — but it would have to be a maximin sweep, not a corpus one.

## Where the time goes: a kernel profile, and the one that was void

`profiles/nsys/prefill-decay-2026-10-08/VOID.md` records that a previous session asked this same question
and filed its traces **VOID**: the harness killed `nsys` instead of the serve, and both arms used one port
so the second arm's request was answered by the first arm's process. Its own recipe for a re-run is "one
port per arm, stop the serve and leave `nsys` to finalise, and compare every kernel total against the
request's measured wall time before drawing any conclusion".

The captures below are immune to both faults by construction — `ninfer_bench` drives `ninfer::Engine`
in-process, so there is no serve process to kill and no port to share — and the first one passes the
recipe's own check.

**Prefill at 65,536 tokens** (`profiles/nsys/prefill-65536-2026-10-12.sqlite`): **7.227 s of GPU kernel
time against 7.313 s measured**, so the profile accounts for the whole phase.

| share | launches | time | kernel |
|---|---|---|---|
| **46.6%** | 128 | 3,368 ms | `mxfp8_kv_tiled_mma_kernel` (the causal attention route) |
| 18.3% | 512 | 1,323 ms | `nvfp4_a4_tma_kernel` |
| 9.8% | 512 | 705 ms | `nvfp4_a4_tma_kernel` |
| 6.5% | 384 | 471 ms | `nvfp4_a4_tma_kernel` |
| 3.9% | 384 | 285 ms | `gated_delta_net::chunk_recurren...` |
| 3.9% | 512 | 278 ms | `nvfp4_a4_tma_kernel` |
| 2.2% | 1,113 | 162 ms | `rmsnorm_cta_bf16x2_kernel` + `rmsnorm_warp_bf16x2_kernel` |
| 1.7% | 512 | 123 ms | `nvfp4_a4_quantize_kernel` |
| 1.6% | 384 | 119 ms | `causal_conv1d_prefill_pairs_kernel` |

So the long-prompt fall is attention: at 8k the GEMMs dominate a chunk's work, and at 64k the causal
attention route is **46.6%** of it. That is physics rather than a setting, which is why the chunk sweep
found nothing to fix — and it is also the exact kernel the unmerged attention work targets.

**Decode at pp8192+tg128, depth 11** (`decode-8192-2026-10-12.sqlite`): **not validated, and therefore not
quoted.** It reports 0.600 s of kernel time where the two phases measured 0.564 s of prefill plus 0.445 s
of decode, so roughly 40% of the work is missing from the capture and every share in it is a share of a
subset. What it shows — NVFP4 GEMM kernels holding most of the time, attention at ~10% — is consistent
with decode being GEMM-bound, but by `VOID.md`'s standard that is a direction to check, not a result.

## Harvested: the NVFP4 down-projection tile at verify widths, verified here

Upstream issue #364 / PR #365 (not ours to fix; read for the mechanism) name a route this port runs
unchanged: the NVFP4 W4A4 `linear_add` for `[5120, 17408]`. At 64 output rows per CTA it launches 80
CTAs on this card's 170 SMs, so a speculative verify streams the down projection's 50 MB of weights from
under half the device. The PR's fix is to select `M32N32` with a deeper pipeline when the geometry is
`17408` and the width is at most 32 — 12 lines, and this port's `src/ops/linear_add/nvfp4/
nvfp4_linear_add_a4.cu` was the pre-image line for line.

**Applied and measured here**, `ninfer_nvfp4_linear_add_bench --n 5120 --k 17408 --policy a4`, median of
20, two interleaved rounds of the two binaries (`op-tile-new.csv`, `op-tile-new-r2.csv`, `op-tile-old.csv`):

| T | M32N64 (before) | M32N32 (after), round 1 | round 2 | change |
|---|---|---|---|---|
| 1 | 36.224 | 36.224 | 36.192 | 0.0% |
| 2 | 36.192 | 36.192 | 36.224 | 0.0% |
| 4 | 38.240 | 38.272 | 38.272 | 0.0% |
| **8** | 42.368 | 38.272 | 38.272 | **−9.7%** |
| 12 | 42.368 | 39.712 | 39.712 | −6.3% |
| 16 | 44.384 | 40.320 | 39.712 | −9.2 to −10.5% |
| 24 | 44.384 | 39.744 | 40.320 | −9.2 to −10.5% |
| 32 | 44.384 | 40.320 | 40.288 | −9.1 to −9.2% |
| 33 | 44.416 | 44.384 | 43.840 | 0.0% |
| 48 | 44.416 | 44.384 | 44.416 | 0.0% |
| 64 | 44.384 | 44.384 | 44.384 | 0.0% |

The two rounds of the new arm agree to within 0.6 µs (1.5%), so the effect is about six times the arm's
own reproducibility. The seam behaves: T=33 returns to `M32N64` and reads the same as before, and T≤4 is
untouched.

**Bit-identity is established by this repository's own oracle**, not by the PR's claim:
`ninfer_linear_add_nvfp4_test` compares the route against its independent reference, and a `T=33` case was
added at the 32/33 seam the change creates — the width list had 32 but not 33, so the boundary was
uncovered until now. It passes.

**What is not established.** The end-to-end effect. The PR reports −1.6..−2.5% on a DFlash2 round; a single
op kernel is a fraction of a round's GPU time, and this card's round-to-round spread is 2-3%, so an
end-to-end A/B here would need more repetitions than one session can interleave — the op-level number is
the one this session can stand behind. The route's five shipped images all take it (every lane's verify
width is ≤32: DFlash2 draft 7/9 gives T≤8/10, MTP draft 4/5 gives T=5/6), and the lane fingerprint was
re-run after the rebuild to confirm no lane figure moved.

## Cross-check: the strongest external item, verified at its source

`docs/research/external-engine-techniques-2026-10-12.md` ranks vLLM's **adaptive verification** first. Read
at the source rather than through the summary: PR #52228, "[Model Runner V2][Spec Decode] Adaptive
Verification without a confidence head", state **MERGED**, 900 additions / 28 deletions. Its mechanism is
an **online acceptance estimator** that predicts a draft token's acceptance from the draft's own logits --
the feature is `logit(q_max)`, the log-odds of the draft distribution's most likely token -- so the verify
length becomes per-round and per-request instead of a fixed window. Its results summary reads **+18.5%**
MiMo-V2.5-Pro-FP4 + DFlash, **+18.1%** Inkling-NVFP4 + MTP8, **+14.6%** Muse-Glimmer-30B + DFlash2. (The
report transcribed the last as 14.7%; the PR says 14.6. Small, but it is the kind of difference that is
worth checking rather than repeating.)

Two things follow for this port, and the second corrects the first:

* The static form of the same lever was measured here as **+14.3%** by widening the draft window from 9
  to 11 — on the bench corpus. **Real text reverses it**: the same lane, real text, greedy, two
  interleaved rounds per depth, reads depth 9 at 241.2/240.5 tok/s and 32.40% acceptance against depth 11
  at 232.0/229.6 and 25.67%. So the corpus arm was measuring the synthetic corpus, not the served regime,
  and no lane's window is changed. The mechanism the PR names still stands, and it is *why* the shipped
  windows are what they are: wider windows pay on the domains where drafts survive and cost on the others,
  which is a maximin decision over domains rather than a single-domain one.
* The adaptive form is the same lever taken further, and its measured gains are largest exactly where
  drafts overrun acceptance — which is a per-request decision, not a per-lane one, and so is not something
  a fixed window can capture either way. Its prerequisite is the part to cost before wanting it: the
  attention backend must accept a **variable** verify length, and this port captures its decode round into
  a CUDA graph per width. The acceptance estimator itself is cheap here -- the draft already runs this
  tree's LM head, so `logit(q_max)` is already computed.

## One thing the instrument itself showed

At depth 11 the per-position counters read **exactly 26.1% at positions 5 through 10**, with `proposed`
identical at 115. Five positions reporting one value is more likely a counter that stops advancing than
five coincidences, so the per-position tail beyond the drafter's real length should not be read as data
until that is checked.

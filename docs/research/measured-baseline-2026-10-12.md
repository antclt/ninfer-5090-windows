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

## The draft window is set too short: depth 9 → 11 is +14.3%

Swept first, grouped (`decode_output_tok_s_mean`, pp8192+tg128):

| depth | tok/s | rounds | acceptance |
|---|---|---|---|
| 3 | 218.7 | 195 | 77.4% |
| 5 | 239.2 | 160 | 61.5% |
| 7 | 280.8 | 140 | 50.5% |
| 9 (shipped) | 287.4 | 140 | 40.2% |
| **11** | **321.4** | 120 | 40.9% |
| 15 | 319.1 | 115 | 30.6% |

Then **interleaved**, because this card drifts up to ~9% between windows and measuring A then B measures
the window. Three paired rounds, two shapes:

| round | pp8192: 9 → 11 | pp32768: 9 → 11 |
|---|---|---|
| 1 | 280.6 → 320.2 (**+14.1%**) | 571.8 → 609.6 (+6.6%) |
| 2 | 288.4 → 330.3 (**+14.5%**) | 588.8 → 622.5 (+5.7%) |
| 3 | 289.0 → 330.3 (**+14.3%**) | 586.7 → 619.3 (+5.6%) |

Per-round spread of each arm: **2.1–3.1%**, so the effect is 5–7× the instrument's own noise, and the
sign is the same in all six pairs.

**The mechanism**, from the same runs at pp8192:

| depth | rounds for 128 tokens | acceptance | tokens per round |
|---|---|---|---|
| 9 | 140 | 40.2% | 0.91 |
| 11 | 120 | 40.9% | 1.07 |

The wider window does not raise the *rate* much; it raises the *yield per round* by 17.6%, and the round
costs less than that. Per draft position, path acceptance at pp8192 is 83.3% at pos0, 34.8% at pos4 and
**still 26.1% at pos5–pos10** — the tail keeps paying, which is why 9 was too short. Depth 11 to 15 is a
plateau (321.4 against 319.1, inside the spread), so 11 is the first point on it.

**What this does not establish.** One artifact (QUASAR's DFlash2 drafter); the pp32768 arm ran at 94–99%
acceptance, which is the tiled corpus saturating the drafter rather than real traffic, so only its *sign*
is reported and its magnitude is not; and the shipped lanes' own figures come from the fingerprint's real
prompts at depth 9, which is a different workload again. Applying the change means regenerating the
launchers, re-running the lane fingerprint, and updating the `Spec` column the profile-consistency gate
checks — and the Swift lane's drafter is a different artifact, so its own optimum has to be measured
rather than assumed.

## One thing the instrument itself showed

At depth 11 the per-position counters read **exactly 26.1% at positions 5 through 10**, with `proposed`
identical at 115. Five positions reporting one value is more likely a counter that stops advancing than
five coincidences, so the per-position tail beyond the drafter's real length should not be read as data
until that is checked.

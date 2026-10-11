# The global activation scale: does the divisor change what the model says?

**Measured 2026-10-10.** `docs/research/activation-scale-method-evidence.md` records four first-party
policies for the NVFP4 global activation scale and states in its verdict table that **no perplexity or
benchmark-accuracy comparison of them exists anywhere** — NVIDIA ships `max` as the default and
`nvfp4_act_headroom` as the alternative, TensorRT-LLM ships a runtime-recomputed one, and nobody has
published what the choice costs in quality. This measures it on a shipped artifact.

## What is actually being changed

The port's calibration stores `input_scale_divisor = 2688 / amax` per site
(`tools/convert/calibration.py:173`, `FULL_RANGE = 2688.0`), and the recipes attach it to every
re-encoded projection's *input* use as `activation_input_divisor`
(`tools/convert/official_recipes.py:487-495`). Scaling that divisor by k moves the per-tensor activation
max by 1/k and leaves every weight alone.

| arm | transform | what it stands for |
|---|---|---|
| shipped | ×1 | `max`: the global scale is the calibrated per-tensor maximum |
| x2, x4 | divisor × 2, × 4 | the per-tensor max divided by 2 and 4 |
| d2, d4 | divisor ÷ 2, ÷ 4 | the per-tensor max doubled and quadrupled — the direction `nvfp4_act_headroom` moves it |
| const | divisor = 1.0 | `input_scale1` / `constant_amax`: amax pinned to 2688 for every site |

This is a controlled stand-in for those policies, not a reimplementation: one number changes, on the
same sources, through the same recipe, and the arms are labelled by the transform rather than by the
policy they approximate.

## How the arms were built, and the two checks that they mean anything

The divisor is read *while the official recipe runs*, so the seam is the module attribute the loader
reads at call time (`official_recipes.py:411`): a per-arm recipe file patches
`official_recipes._CALIBRATION_PATH` for the duration of the call. No converter change, no tree change.

Two falsifiers, because a flat result from a transform that never took would look identical to a real
flat result:

1. **The seam, in-process** (`tools/bench/probe_divisor_seam.py`): `load_calibration()` returns 53.4925 unpatched, 213.9701
   under the x4 patch, 26.7463 under d2 and 1.0000 under const, and the original value after the patch
   is restored. `python -m tools.convert` imports the recipes relatively (`from .official_recipes import
   RECIPES`), so the patched module object is the one the recipe reads.
2. **The stored artifact** (`tools/bench/read_artifact_divisors.py`, reading the same 4-byte FP32 auxiliary on the use that
   `verify_artifact.py`'s `check_divisors` reads):

| artifact | layer-0 GDN q/k/v/z | gdn/output | mlp/gate |
|---|---|---|---|
| shipped | 53.4925 | 1075.2 | 836.0 |
| x4 | 213.9702 | 1075.2 | 836.0 |
| x2 | 106.9851 | 1075.2 | 836.0 |
| d2 | 26.7463 | 1075.2 | 836.0 |

The second check also establishes the transform's *scope*: only the calibrated hidden-state-input sites
move. `gdn/output` and `mlp/gate` are identical in every arm, because those sites are imported from the
source checkpoint with their own divisors rather than re-encoded through the calibration.

## The measurement

Each arm is scored over the full 1M-token corpus with `ninfer-perplexity --topk-record` (1,044,557
positions, k=60), and compared against the **shipped artifact scored the same way** with
`tools/release/per_domain_kl.py`. The reference is the shipped image rather than a BF16 model:
`per_domain_kl.py` explains why a BF16 reference cannot exist on this card (51.75 GiB of weights against
31.85 GiB of VRAM), and for this question the shipped image is the right reference anyway — same weights,
same recipe, one number apart. Both sides of every comparison report the same prefill signature
(`cbe34f62…`), so the two runs took the same path.

**Control.** The shipped image against itself reads 0.000000 nats in every domain with 0 floored entries.
The `floored` column is a monotone sanity check across the arms below (0 → 8.7M → 16.6M → 36.9M).

| arm | transform | chinese | english ref | english long | code | **OVERALL** |
|---|---|---|---|---|---|---|
| x2 | divisor ×2 | 0.586790 | 0.540744 | 0.587312 | 0.223816 | **0.485091** |
| x4 | divisor ×4 | 4.179822 | 4.740187 | 4.909916 | 3.141363 | **4.244283** |
| d2 | divisor ÷2 | 0.174663 | 0.140292 | 0.130718 | 0.051320 | **0.124380** |
| d4 | divisor ÷4 | 0.174556 | 0.142268 | 0.130985 | 0.051129 | **0.124867** |
| const | divisor = 1.0 | 0.180148 | 0.143136 | 0.133545 | 0.052611 | **0.127496** |

## What the numbers say

**The global activation scale is a first-order lever, not a second-order knob.** The yardstick is
already in this project's own record: the 2026-10-08 re-encode comparison (`.audit/models-rebuild.tsv`,
`fp8attn` against `nvidia` — a different quantization recipe of the same base model, i.e. a change this
port accepted and shipped) reads 0.139802 nats overall with the same domain ordering. Halving or
quartering the divisor moves the distribution by **0.1244**, slightly *less* than that accepted recipe
swap. So the answer to the question the evidence document could not answer is: the choice is real, and
its headroom direction costs about what a recipe change costs.

**The clipping direction is destructive, which is what `max` is for.** ×2 reads 0.485 and ×4 reads 4.24 —
3.5× and 30× the accepted-recipe difference. That is the mechanism the `max` policy's own documentation
gives: anchoring the scale to the calibrated per-tensor maximum means any larger activation at inference
saturates, and shrinking the assumed maximum deliberately is exactly that saturation.

**Headroom saturates.** ÷2, ÷4 and the constant policy agree within 2.5 % (0.124380, 0.124867,
0.127496) across an eight-fold further range of the divisor, so beyond one halving the curve is flat and
stays flat all the way to the documented `input_scale1` extreme. NVIDIA's `nvfp4_act_headroom` uses a
large `rho`, but its effect on the artifact is bounded: past the first halving there is nothing left to
move.

**Direction of the polarity, measured rather than assumed.** Larger divisor ⇒ the assumed per-tensor
maximum is *smaller* ⇒ clipping; smaller divisor ⇒ headroom. Before this sweep the sign of that mapping
was reasoned about twice and could have been wrong either way; the x2/x4 arms settle it.

**Which direction is *better* is answered, and the shipped setting stands.** Perplexity over the same
corpus, same tool, same defaults:

| artifact | overall PPL | against shipped |
|---|---|---|
| shipped (`max`) | 4.724219 | — |
| **x1-rebuilt (control: shipped divisors, today's code)** | **4.724219** | **exactly equal** |
| d2 (divisor ÷2) | 4.730689 | +0.14 % |
| const (`input_scale1`) | 4.729511 | +0.11 % |

The control is what makes the difference attributable: the arms were rebuilt today while the shipped
image was built earlier, so a rebuild of the *same* line with the *shipped* divisors was scored the same
way — and it reproduces the shipped figure to the last digit. There is no rebuild drift, so the 0.11-0.14 %
is the divisor and nothing else.

Headroom is neutral-to-marginally-worse: three of four domains read slightly worse, Chinese slightly
better. **No change ships.** The value of the sweep is that the shipped `max` policy is now the best
*measured* point rather than an unexamined default — the evidence document's question 4 had no answer
anywhere in the field, and this is one.

**A calibration worth keeping.** 0.124 nats of per-domain KL corresponds here to a 0.14 % perplexity
change, and the accepted recipe swap's 0.140 nats to a change of a similar size in this port's own
record. So for future decisions on this artifact, a KL in the ~0.1-nat range is a real distributional
move with a *small* quality cost, while 0.5 nats and above (the ×2 arm) is a quality loss on a different
scale. Perplexity alone cannot see the first kind, which is why both instruments exist.

## Instrument notes

- Sources for a rebuild live under `C:\AI\models\hf-src\`; `build_model.py --dry-run` prints the
  *archive* layout (`<repo>/hf-src`), which does not exist in this checkout. The first sweep failed on
  exactly that, with the error lost because python's stdout does not reach a shell redirect under a
  detached launcher.
- A rebuild refuses a name that already exists, so the sweep's x4 build reported `rc=1` while the
  artifact from the foreground run was already in place and identical by construction.
- The `.audit/models-rebuild.tsv` rows for this sweep carry the falsifier's numbers.

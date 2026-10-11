# The proposal head, per lane, on real text

**Measured 2026-10-10.** ADR-0005 requires the proposal head to be a per-profile decision "from a
record. Never from convention, and never uniformly across profiles". It listed QUASAR, the NVFP4-full
image and the retired image. All eight shipped lanes passed `--lm-head-draft`, and the two artifacts
added afterwards — swift and nvidia — had no record anywhere in the tree. This closes that: one
harness, all eight lanes, two rounds per arm.

## The question, and why the bench could not answer it

`--lm-head-draft` selects the artifact's indexed proposal head: a Q4 copy of the head over a
**frequency** shortlist (`tools/convert/proposal.py:14-52`, 131,072 of 248,320 rows by corpus counts,
`format="q4_g64_fp16"` by default). Without it the drafter proposes through the artifact's own head at
whatever precision that artifact binds. So the flag trades a cheaper draft step for a restricted one.

The bench cannot decide it, and this repo says so in its own tooling table: `ninfer_bench -n` generates
after a one-token seed, and the same corpus behind a real prompt saturates acceptance. Both ends were
measured here before the real instrument was used:

| arm | prompt | spec acc | decode out t/s |
|---|---|---|---|
| `--lm-head-draft` | 2,048 | 0.7943 | 522.5 |
| full head | 2,048 | 0.9829 | 635.7 |
| `--lm-head-draft` | 9,216 | 0.9914 | 629.0 |
| full head | 9,216 | 1.0000 | 628.8 |

The 9,216 arm saturates at exactly the 0.9914 the tooling table warns about, and the 2,048 arm sits
above the 0.44-0.55 band real domains occupy. The bench therefore reported a mechanism, not a verdict.

## The measurement

`tools/bench/realtext_acceptance.py`, greedy, real text (this repository's own C++ source under the
perplexity corpus), 3 prompts x 512 tokens, two rounds per arm, each arm a fresh serve. The lane's own
flags are composed through `profiles.launcher_args`, so only the proposal head differs between arms.

| lane | shipped (`--lm-head-draft`) | full head | decode for the shipped setting |
|---|---|---|---|
| quasar dflash2 | 32.4 % / **249.5** | 30.5 % / 230.8 | **+8.1 %** |
| quasar mtp4 | 48.9 % / **177.9** | 53.6 % / 169.3 | **+5.1 %** |
| ninfer dflash2 | 31.8 % / **243.8** | 32.0 % / 233.5 | **+4.4 %** |
| ninfer mtp4 | 51.1 % / **180.0** | 52.3 % / 163.7 | **+9.9 %** |
| swift dflash2 | 39.6 % / **245.5** | 39.1 % / 232.4 | **+5.6 %** |
| swift mtp4 | 45.9 % / **177.2** | 47.0 % / 163.3 | **+8.5 %** |
| nvidia dflash2 | 28.8 % / 226.5 | 38.3 % / **271.6** | **−16.6 %** |
| nvidia mtp4 | 51.9 % / **185.2** | 54.7 % / 171.7 | **+7.9 %** |

Decode is tok/s and acceptance is accepted/drafted pooled over the arm. Every arm's two rounds agreed
exactly on acceptance and within 0.4 % on decode.

**The criterion is decode rate**, per ADR-0005's own consequence: "Acceptance rate is not the selection
criterion; measured decode rate is." Five of the seven confirming lanes are the reason that sentence
exists — quasar mtp4, ninfer mtp4, swift mtp4, nvidia mtp4 and ninfer dflash2 all *accept more* with the
full head and still decode *slower*. The flag buys more accepted tokens per unit of draft work than it
loses in acceptance, except on one lane.

## The one lane that differs, and two explanations that do not survive checking

nvidia dflash2 loses 9.5 acceptance points and 16.6 % of decode with the shipped setting. Two candidate
mechanisms were checked against the tree and both are wrong:

- **"Its proposal is derived from a quantized head while the others' are not."** False: every recipe
  calls `add_proposal` with a BF16 source, and the nvidia recipe is no exception
  (`tools/convert/official_recipes.py:827`, `source=model.source("text/output_head", base)`).
- **"Its shortlist is ranked from the wrong head."** False: the ranking is a corpus frequency count
  (`proposal.py:14-17`, `freq_corpus/fixtures/ranking/ranking.train.counts.i64`), identical for every
  artifact, and `select_shortlist` takes the top rows by that count alone.

What *does* differ is the precision of the head each artifact binds — the unsloth line assigns its
output head Q8, and NVIDIA's own stock quantizes `lm_head` to NVFP4, which the nvidia recipe imports
verbatim. The mechanism connecting that to a 9.5-point acceptance loss is **not established here** and
is not asserted.

## The head-side alternative, and why the engine closes it

`docs/research/target-precision-and-draft-acceptance.md` item 6 names the head-side lever: the indexed
head "is the one component that both consumes the drafter's hidden state and is itself quantized (Q4 in
the shipped recipe)". It cannot be raised on this route. `src/ops/linear_topk/linear_topk.cpp:35-49`
resolves three head profiles and no others — full rows at Q8, full rows at FP8, and the *shortlist* path
at Q4 only — so a proposal head at any other precision is refused. Measured: an NVIDIA image rebuilt with
a Q8 proposal head failed at startup with `FATAL server failed during startup | linear_topk: unsupported
head profile`, while the same lane's shipped arm served normally in the same run.

## The mechanism, as a hypothesis with a test

The shortlist is ranked by corpus frequency and is identical for every artifact (`proposal.py:14-17`),
and the engine fixes its precision at Q4. What differs between the lanes is the *draft's* encoding: three
lines ship NVFP4 draft projections and the NVIDIA line ships Q8, measured on that target as the better of
the two (60.9 % against 57.7 % on the DFlash2 lane, `official_recipes.py:716-719`). A Q8 draft whose
proposals are less frequency-concentrated would lose more to a frequency-ranked truncation — which is the
shape of the effect measured here: nvidia dflash2 is the only lane that loses, and it is the only lane
whose draft is Q8.

**That hypothesis is falsified, and it was the third to be.** An NVIDIA image rebuilt with the NVFP4 draft
the other three lines use reads **28.8 % and 220.6/222.8 tok/s** with the flag on — the same acceptance as
the shipped Q8 draft to the last digit (1103 accepted of 3828 drafted, both arms), and 2.5 % slower,
against 38.3 % and 269.9/270.2 with the flag off. So the draft's encoding is not the mechanism, and the
measured remedy stands on its own: clear the flag. The three falsified explanations — proposal provenance,
shortlist ranking and draft encoding — are recorded rather than a plausible-sounding cause being left in
their place; what makes this lane different is still not established.

## The same lane on the served protocol, interleaved — and the greedy arm does not decide

The greedy measurement above says the flag loses on this lane. That is not the protocol the lanes serve,
and not the protocol the depth decision was taken with. Re-measured on the `code` cell of
`lane_fingerprint.py` — the served sampling, one cell, `--no-write`, the flag flipped between runs so the
arms alternate:

| run | `--lm-head-draft` | code cell tok/s | acceptance |
|---|---|---|---|
| 1 | **on** (shipped) | **329.5** (spread 1.1) | **45.9 %** |
| 2 | off | 257.2 (spread 1.3) | 35.9 % |
| 3 | **on** | **323.6** (spread 5.1) | **45.9 %** |
| 4 | off | 257.9 (spread 0.9) | 35.9 % |

**The sign reverses, and the shipped setting wins by 25-28 %.** Interleaving is what makes that a
measurement rather than a window: the first comparison here was between two full fingerprints two hours
apart, which is the failure `AGENTS.md` records as "measure A and then B and you have measured the
window" — and that comparison reported a 20 % gap with the opposite sign.

**The disagreement is a property of the accept rule, not noise.** At temperature 0 the test is whether the
draft's argmax equals the target's, so a candidate set restricted to a *frequency*-ranked shortlist loses
whenever the target's argmax is a token the ranking under-weights — and on this corpus, code, it does so
often. At the served sampling the target's token is drawn from a wide distribution and the test is a
ratio, so the same restriction costs far less. Both numbers are real; only one of them is the regime the
product runs in.

## What follows

- **No setting changes.** The flag stays on all eight lanes, and the record ADR-0005 asks for now exists
  for the two artifacts that had none.
- The greedy arm is kept in this note rather than deleted, because the disagreement between the two
  protocols is the finding: a decision taken on greedy alone would have shipped a 25 % regression.
- The mechanism behind the *greedy* penalty is the shortlist's frequency ranking, which the falsified
  draft-encoding arm corroborates (an NVFP4 draft reads the same 1103 of 3828 at greedy). The served
  regime does not need it explained to be decided.

## Instrument notes

- `--no-lm-head-draft` was added to `realtext_acceptance.py` for this comparison; the lane's own flag
  set is *removed* rather than substituted, because a duplicate flag is refused.
- The same edit replaced the tool's hand-written `LANE_FLAGS` with `profiles.launcher_args`, and passes
  `launcher_environment`. Its hardcoded list had drifted to a draft depth and KV dtype that no lane runs,
  and it omitted the lane's `NINFER_CUDA_SYNC` entirely — so its earlier records described a
  configuration that does not ship.
- A cached arm now answers only for the flags it ran under: the tool's `result.json` carries the flag
  list and a stale directory is ignored rather than returned.
- **Under a background or detached launcher, python's stdout does not reach a shell redirect.** The
  sweep's per-invocation logs were empty while the tool ran correctly. Measured both ways in
  `profiles/bench/redirect-*.log`: the shell's own `echo` reaches the file, python's output does not.
  A run that must leave a record has to write it itself.

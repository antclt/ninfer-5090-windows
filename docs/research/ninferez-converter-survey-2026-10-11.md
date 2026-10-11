# NInferEZ-Converter, surveyed against this tree (2026-10-11)

`https://github.com/BenGamliel/NInferEZ-Converter` was raised as a possible source. It is a Windows
desktop + JSON-backend front end for a converter, Apache-2.0, created the day it was asked about, and its
`vendor/tools/` carries module names identical to this tree's (`artifact/codecs/{fp8_row,nvfp4,row_split}.py`,
`artifact/{reader,schema,writer}.py`, `convert/*`). This records what it is, what its local changes are,
and what each one means here - so the question does not have to be re-answered.

**Sources read**: the repository's `README.md`, `vendor/UPSTREAM.md`, `docs/VENDOR-CHANGES.md`,
`vendor/LICENSING.md`, `vendor/THIRD-PARTY-NOTICES.txt`, `docs/SUPPORTED-INPUTS.md`, its file tree, and
`docs/upstream-weight-conversion.md`'s headings; against this tree's `tools/artifact/writer.py`,
`tools/upgrade_nvfp4_v2_to_v3.py`, `tools/convert/` and `tools/artifact/` listings, and a tree-wide grep
for `__builtin_ctz`.

## What it is, and why it is behind this tree

A fork of **`iamwavecut/ninfer-all`** at base commit `91576f32` (**2026-09-27**), which
`vendor/LICENSING.md` describes as consolidating `ashalliants/ninfer-3090`, `Don-Chad/ninfer-3090`,
`Neroued/ninfer` and other credited forks. This port tracks `Neroued/ninfer` directly and merges it
through **2026-10-07**, so every upstream-derived module it vendors is older than the one here. Nothing
in that half is worth taking: we are ahead of it, not behind.

Its local changes are the only candidates, and each is checked below.

## Its local changes, against this tree

**1. `writer.py`: hard link replaced by a Windows atomic no-overwrite rename**, "so Portable output
publication works on NTFS and FAT/exFAT targets". This one targets a mechanism this tree uses:
`tools/artifact/writer.py:270` publishes with `os.link`. **No change is needed here**, for three
checkable reasons:

- the temporary is created *beside its destination* - `tempfile.mkstemp(prefix=f".{target.name}.",
  suffix=".tmp", dir=target.parent)` (`tools/artifact/writer.py:163`) - so the link is always within one
  directory and therefore one volume. The cross-volume failure a link can have cannot arise;
- every destination is checked for existence before any file is opened
  (`tools/artifact/writer.py:149-151`, raising `FileExistsError`), so the no-overwrite property does not
  depend on the link's own semantics;
- a failure aborts loudly and cleanly: `abort()` unlinks the temporaries *and* the already-published
  targets and re-raises (`tools/artifact/writer.py:279-289`). There is no partial artifact to mistake for
  a complete one.

The residual difference is a **capability, not a defect**: this tree cannot *write* an artifact onto a
filesystem without hard links (exFAT/FAT, some network shares) - it fails with a clear error instead. No
product requirement covers that: artifacts are built on NTFS here, and what ships to users is a zip.

**2. `upgrade_ninfer_v2_to_v3.py`: the same publication question, plus progress/cancellation hooks.**
This tree's copy links into the destination directory too (`tools/upgrade_ninfer_v2_to_v3.py:880`), so the
first half is the same answer. The progress hooks are a UX difference, not a defect.

**3. GGUF sources and ternary quantization - capabilities this tree does not have.** Its
`convert/sources/gguf.py` (20 KB), `gguf_source.py` (32 KB), `gguf_blocks.py` (20 KB) and `ternary.py`
(18 KB) have no counterpart here. Neither is a product requirement: this port's artifacts are built from
Hugging Face source checkpoints. **If GGUF ever becomes one, upstream's own open pull request is the
place to take it from** - `Neroued/ninfer` #282, "convert: read GGUF (any ggml quant level) as a
conversion source" - not a third-party fork's implementation of the same idea.

**4. BOM handling in `resources.py` and `safetensors.py`.** This tree already gates encodings
(`tools/release/check_text_encoding.py`) and reads Hugging Face checkpoints, whose metadata carries no
BOM. No action.

**5. Engine-side MSVC workarounds, which do not apply.** Its `vendor/UPSTREAM.md` lists replacing a
GCC-only `__builtin_ctz` with `std::countr_zero`, generating device-profile JSON in bounded chunks to
avoid MSVC C2026, and splitting a serving option parser to avoid C1061. This tree builds with MSVC today
and uses no `__builtin_ctz` anywhere under `src/` (checked tree-wide), so there is nothing to port.

**6. A host-only `.ninfer` inspector and JSON identity commands.** This tree already has
`tools/artifact/inspect.py` *and* `inspect_diff.py`; theirs is not more.

## The conversion method, compared (the question that matters)

Asked afterwards, and it deserves its own section, because the first survey looked at provenance and
peripheral changes and not at the quantizer.

**Their core quantizer is the same code.** Three-way sizes, and the equality is exact rather than close:

| module | upstream | this tree | theirs |
|---|---|---|---|
| `quantization/groupwise.py` | 4,298 B | **4,298 B** | 11,354 B |
| `methods.py` | 11,522 | **11,522** | 14,253 |
| `proposal.py` | 3,483 | **3,483** | 4,847 |
| `quantization/fp8_row.py` | 4,704 | 4,704 | 4,704 |

So `groupwise.py`, `methods.py` and `proposal.py` are upstream's, unchanged, in this tree; the only
modules that differ are the ones this port added (`quantization/nvfp4.py`, `calibration.py`,
`verify_artifact.py`, `sources/quant_scheme.py`) and the ones that fork added.

**Their one addition to the method set is an MSE scale search** - `grouped_mse` in their `methods.py`,
`_mse_scale_candidates` and `quantize_matrix_mse` in their `groupwise.py`. It is not in their own
documented method table, which lists `nvfp4` from floating-point input as "Supply a custom quantizer".
**This tree built that method and measured it as a regression** (`docs/active-work.md` item 2, the
instrument recoverable from `900a0f78`, reverted `b891e1e9`): wired into the 128 object groups the NVIDIA
artifact re-encodes locally and measured paired against the shipping artifact in one window on the full
corpus, **max-abs 4.915181334 against searched 4.925917194 - +0.218 %, the wrong way**. Weight
reconstruction error fell 40-66 % while perplexity rose, and the mechanism is the finding: per-weight MSE
and output perplexity are decoupled, because the blocks a search improves are the ones it clips, and the
clipped value carries signal. Scope, stated: that covers the sites this port re-encodes (the
`self_attn`/`linear_attn` groups the producer quantised as FP8); a GGUF or ternary source has a different
input distribution and is untested here, and this tree has no such source. What survives is "max-abs is
the right default for weights with no producer evidence behind them".

**Their QUASAR driver reaches the same conclusion as this tree's QAT recipe, by a slightly worse route.**
`quasar_nvfp4.py` says the public mixed recipe does not match the QUASAR checkpoint and imports every
quantised projection verbatim through `import_encoded`, and that the GDN `a`/`b` projections cannot be
imported as NVFP4 because the layout needs 128-row tiles - which is this tree's recipe's own analysis,
word for word in substance. The difference is one step: theirs **dequantises those two from the NVFP4
source** ("an upcast, not a re-quantisation"), where this tree's recipe takes them from the **BF16 base
checkpoint**, so ours carries the original values and theirs carries the source's quantization error.

**The activation axis, where this tree is ahead on a measured result.** Their converter has no
calibration at all: no `calibration.py`, no corpus, and no `amax` or `2688` anywhere in its recipes. This
tree computes a per-site divisor from a fixed corpus and has measured the choice - `max` is the best
measured point (shipped 4.724219 against ÷2 4.730689 and a constant 4.729511), which is item 10 of the
consolidated list.

**Verdict: not better.** Same core quantizer, older base, one extra method that this tree measured as a
regression, one import step that loses information this tree keeps, and no calibration on the axis this
tree has measured. What it has that this tree does not is GGUF and ternary *sources* - capabilities, not
methods, and neither a requirement here.

## What is worth knowing

The lineage, not the code: `iamwavecut/ninfer-all` consolidates `Neroued/ninfer` with the 3090 forks,
including `Don-Chad/ninfer-3090` - which this port's own `NOTICE` credits for the 1.0.x Windows layer. So
NInferEZ is a **sibling** project working the same upstream, not a source to take from, and its converter
front end (a .NET wizard plus a JSON backend) is the part this product has no requirement for.

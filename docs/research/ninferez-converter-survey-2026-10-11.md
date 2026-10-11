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

## What is worth knowing

The lineage, not the code: `iamwavecut/ninfer-all` consolidates `Neroued/ninfer` with the 3090 forks,
including `Don-Chad/ninfer-3090` - which this port's own `NOTICE` credits for the 1.0.x Windows layer. So
NInferEZ is a **sibling** project working the same upstream, not a source to take from, and its converter
front end (a .NET wizard plus a JSON backend) is the part this product has no requirement for.

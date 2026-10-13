# NInfer documentation

Start with the [project README](../README.md) to build NInfer, download a published artifact, and
run the CLI or HTTP server.

## User guides

| Document | Purpose |
|---|---|
| [CLI](cli.md) | text, chat-history, image/video input, output streams, sampling, MTP, and common runtime options |
| [HTTP serving](serving.md) | OpenAI Responses/Chat Completions, Anthropic Messages, state, streaming, token counting, authentication, and tool calls |
| [Performance](performance.md) | RTX 5090 measurement coverage, per-model serving results, methodology, and publication rules |
| [Weight conversion](weight-conversion.md) | official recipes, custom formats and sources, conversion methods, optional components and artifact output |
| [Perplexity](perplexity.md) | fixed-corpus and custom-text causal perplexity, comparison rules, progress, and reports |
| [opencode settings](opencode-settings.md) | the four shipped model entries, compaction, and the concurrency answer for OpenCode Desktop |
| [v2 to v3 flag diff](v2-v3-flag-diff.md) | what changed between the retired v2 launchers and the shipped v3 ones, flag by flag |
| [Active work](active-work.md) | temporary: work in progress and the questions already closed, removed when the list empties |
| [CLI examples](../examples/cli/) | committed text, multimodal, thinking, long-decode, and long-context inputs |

The executable `--help` output is the exact source for command-line option spelling and defaults.

## Model artifacts

This port ships four weight lines, as **five images** — the unsloth line ships twice, because the
DFlash2 lane runs the no-exception build and its MTP lane the BF16-exception one. All five are built by
this port from the sources below, fetched at pinned revisions:

| Model | Weights | Built from | Versioned model card source |
|---|---|---|---|
| Qwen3.8-27B | `nvfp4qat` (QUASAR) | [QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4) | [model card](../model-cards/Qwen3.8-27B-nvfp4qat-NInfer/README.md) |
| Qwen3.8-27B | `nvfp4full` | [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) | — |
| Qwen3.8-27B | `nvfp4full_noex` | the same unsloth line, BF16 exceptions re-encoded; DFlash2 lane | — |
| Qwen3.8-27B | `nvfp4swift15` (Swift 1.5) | [ukisai/Swift-1.5-Qwen3.8-27b-NVFP4](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b-NVFP4) on its [BF16 finetune](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b) | — |
| Qwen3.8-27B | `nvfp4nvidia` (NVIDIA ModelOpt) | [nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) | — |

All five shipped artifacts are built locally by this port's converter from these sources; none is
fetched prebuilt. `download_model.py` fetches sources.

Upstream publishes artifacts for other checkpoints, each with its own model card on its own
repository. This port ships none of them and keeps no copy of their pages:
[upstream's README](https://github.com/Neroued/ninfer#readme) is their authority.

The storage contract the shipped artifacts implement is the
[Qwen3.8-27B artifact reference](maintainer/qwen3.8-27b-artifact.md): identity, object inventory,
formats, layouts, aliases and source transforms, with a section per line in Sections 14 to 19.

## Repository-local guides

- [Benchmarks](../bench/README.md)
- [Tests](../tests/README.md)
- [Tools](../tools/README.md)
- [Capability evaluation](../eval/README.md)

## Maintainer references

The active references under [`maintainer/`](maintainer/) record current architecture, model,
artifact, and maintenance contracts. These files are not additional user workflows or installed
API documentation.

**Seven of them are written in Chinese by upstream.**
[Maintainer documents: English index](maintainer/README.md) summarises in English what each one
decides and links to the original, which is the source of truth. The originals are not translated or
modified: a translation is a permanent divergence, and every future upstream merge would then conflict
in every translated file.

[`adr/`](adr/) holds the decisions behind the shipped profiles and artifact rules, each with the
measurement that produced it: why a v3 container is required, why speculation is not bit-identical,
the draft-binding contract, why the profiles are vision-only, and why every profile flag is
measured rather than chosen.

[Engine architecture](maintainer/engine-architecture.md) is the single top-level reference. The
other references own narrower contracts:

| Document | Responsibility |
|---|---|
| [Engine architecture](maintainer/engine-architecture.md) | model/config/weight ownership, loading-to-execution flow, requests, scheduling, transactions and graphs |
| [Build system](maintainer/build-system.md) | CMake targets, explicit source ownership, CUDA compilation boundaries, presets and developer configuration |
| [Artifact container](maintainer/artifact-container.md) | v3 directory, objects, logical bindings, Uses, resources and file framing/sharding |
| [Artifact conventions](maintainer/artifact-conventions.md) | what a new conversion must match to join the shipped lanes: all-NVFP4 text, the weight envelope, the lane shape and the measurement gates |
| [Numeric formats](maintainer/tensor-formats.md) | represented values, codes/scales, conversion arithmetic and numerical interpretation |
| [Storage layouts](maintainer/storage-layouts.md) | packing, plane offsets, padding, encoded sizes and view addressing |
| [Qwen3.5 model](maintainer/qwen3_5-model.md) | Dense/MoE mathematics, instance config, logical parameters, MTP, Vision and state semantics |
| [DFlash and DFlash2](maintainer/dflash.md) | conditioning, masked draft computation, proposal distributions and backend state |
| [Constrained decoding](maintainer/constrained-decoding.md) | GBNF/JSON/choice/regex, tool policies, masks, speculative transactions and observations |
| [Resource scheduling and context cache](maintainer/resource-scheduling-and-context-cache.md) | core design for continuation, retention, incremental resources, preemption and recovery |
| [Paged KV context store](maintainer/paged-kv-cache.md) | typed pools, pages, replicas, address spaces, reservations and consumer views |
| [ReplaySSM GDN](maintainer/replayssm-gdn.md) | raw transition records and faithful commitment of the verified state prefix |
| [Op development](maintainer/op-development.md) | semantic boundaries, source ownership, numerical qualification and performance evidence |
| [Operational logging](maintainer/logging.md) | log ownership, presentation, severity and data policy |
| [Linear tuning and measurement](maintainer/linear-tuning.md) | logical metrics, timing, tuning ranges, dispatch tradeoffs and performance reports |

Model cards contain official artifact facts and source provenance. The
[conversion guide](weight-conversion.md) is the entry point for making an artifact. Exact config
fields, parameter expansion and native supported domains are maintained by the code linked from
these references.

## Dated records

Three directories hold the working record behind the decisions above. They are not user guides and not
contracts; they are what a later reader needs in order to check a claim instead of taking it on trust.

| Directory | What it holds |
|---|---|
| [`research/`](research/) | dated investigations: measurements, comparisons against other engines, and the evidence for a setting that shipped, each written while its question was open |
| [`features/`](features/) | design notes for runtime work this port carries, including the weight-only NVFP4 DFlash2 runtime and the MSVC test-portability change |
| [`upstream-reports/`](upstream-reports/) | what this port reported or considered reporting upstream, and the fork-divergence record |

## Planned work

Qwen3.8-Flash-Next uses the Qwen4Exp architecture. Its converter and many supporting Ops are
implemented; Engine integration remains pending. The execution design owns the current support
status and remaining requirements. Concrete Op contracts live in `include/ninfer/ops/`.

| Design | Responsibility |
|---|---|
| [Qwen4Exp model and weights](maintainer/qwen4_exp-model.md) | architecture, quantization, merged matrices, source binding and residency budgets |
| [Qwen4Exp execution and state](maintainer/qwen4_exp-execution.md) | layer flow, cross-Op handoffs, transactions, state lifetime and remaining integration |
| [Qwen4Exp PLE I/O](maintainer/qwen4_exp-ple-io.md) | exact Host lookup, cold backing, bounded async reads, H2D and cancellation |
| [Expert weight cache](maintainer/expert-weight-cache.md) | affine expert sources, explicit placement, GPU replica cache and pending mixed execution |
| [Linear test maintenance](maintainer/linear-test-reorganization-plan.md) | pending consolidation of test runners and explicit case contracts |

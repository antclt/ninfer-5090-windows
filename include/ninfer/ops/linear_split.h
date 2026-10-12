#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/projection_weights.h"

#include "core/device.h"
#include <span>

namespace ninfer::ops {

/**
 * Y_s[r,t] = sum_k decode(W)[begin_s+r,k] * X[k,t], with begin_s equal to the
 * sum of preceding output row counts. For paired weights, W is the logical row
 * concatenation [first; second]; an output segment may cross that parent boundary.
 * No physical weight concatenation or packed parent output is materialized.
 *
 * X is contiguous BF16 [K,T]; outputs are ordered, contiguous BF16 [N_s,T], with positive row
 * counts summing to the weight row count. T>0; activation/output bases are 16-byte
 * aligned. Outputs are mutually disjoint and overlap neither X nor any weight plane.
 * Inputs and weights are preserved. Logical and padded weight dimensions agree.
 *
 * Registered forms include BF16, Q8-G32/FP16, FP8 row-scale, NVFP4 block-scale and
 * paired Q4-G64/FP16 + Q5-G64/FP16. Single-parent activation permissions follow
 * LinearPolicy; paired forms use A16.
 *
 * This port additionally registers one NVFP4 drafter geometry upstream does not carry: the
 * BlockScaleK16M128x4 `[6144,5120]` parent writing the three ordered segments q `[4096,T]`,
 * k `[1024,T]` and v `[1024,T]`, weight-only A16. Our QUASAR artifacts' draft modules are
 * NVFP4-encoded and upstream's NVFP4 split registry carries no drafter geometry, so no
 * upstream route serves them. It writes the three allocations directly with no transient
 * workspace and serves extents above its fused small-T family in 32-token chunks.
 * Workspace is caller-owned, call-scoped and disjoint from all operands. The Op
 * allocates no hidden device memory, owns no persistent state and supports Graph replay.
 */
void linear_split(const Tensor& x, const ProjectionWeights& weights,
                  std::span<const Tensor> outputs, MemoryArena& workspace,
                  CudaExecutionView execution);

/** Single-parent A16 computation with no transient workspace. */
void linear_split(const Tensor& x, const Weight& weight, std::span<const Tensor> outputs,
                  CudaExecutionView execution);

/**
 * Minimum call-local capacity covering inclusive [min_columns,max_columns], both positive.
 * Output row counts select the same registered segmentation as execution. Inspects only
 * host-visible format, shape and activation permission; no device backing is required.
 * Capacity starts at a 256-byte-aligned arena cursor and excludes trailing padding.
 */
[[nodiscard]] std::size_t
linear_split_workspace_capacity_bytes(const ProjectionWeights& weights,
                                      std::span<const std::int32_t> output_rows,
                                      std::int32_t min_columns, std::int32_t max_columns);

} // namespace ninfer::ops

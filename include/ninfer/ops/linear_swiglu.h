#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include "core/device.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

// Capacity for every extent in the inclusive interval; invalid profiles or intervals throw.
[[nodiscard]] std::size_t linear_swiglu_workspace_capacity_bytes(QType qtype,
                                                                 std::int32_t gate_up_rows,
                                                                 std::int32_t input_rows,
                                                                 std::int32_t min_tokens,
                                                                 std::int32_t max_tokens);

/**
 * Policy-bearing capacity query. Q4/Q8 use A16 under every policy. NVFP4 uses A16 under
 * A16Only/AllowA8 at every positive T — fused through T=16, then the port's wide-T A16 MMA
 * route — and AllowA4 accepts every positive T. Row-scaled FP8 accepts all policies, with
 * A8 permitted by AllowA8/AllowA4.
 * A permissive policy covers whichever qualified route the private resolver selects across the
 * requested interval.
 */
[[nodiscard]] std::size_t
linear_swiglu_workspace_capacity_bytes(QType qtype, std::int32_t gate_up_rows,
                                       std::int32_t input_rows, LinearPolicy policy,
                                       std::int32_t min_tokens, std::int32_t max_tokens);

/**
 * out[i,t] = SiLU((W_gate * x)[i,t]) * (W_up * x)[i,t].
 *
 * x [K,T] and out [M,T] are contiguous, 16-byte-aligned BF16 matrices.
 * gate_up_weight [2*M,K] stores all M gate rows followed by their matching M up rows.
 * Registered Q4_G64_FP16, Q8_G32_FP16, FP8_E4M3FN_ROW_BF16 and NVFP4 geometries
 * use their native stored representation. Inputs and weight storage must not overlap out.
 * Writes the entire output and has no persistent state.
 *
 * Q4/Q8 use A16 under every policy. FP8 permits A8 with AllowA8/AllowA4.
 * NVFP4 permits A4 with AllowA4; A16Only/AllowA8 support only T <= 16 for NVFP4.
 * Otherwise every positive T is admitted. The overload without a policy uses A16Only.
 * Caller workspace covers the complete operation and is scoped to the call.
 */
void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, LinearPolicy policy,
                   MemoryArena& ws, CudaExecutionView execution);

/**
 * A16-only convenience form. Q4/Q8 and row-scaled FP8 retain their complete positive-T domain.
 * NVFP4 keeps the complete positive-T domain: the fused small-T family through T=16, then the
 * port's wide-T fused A16 MMA route (no workspace).
 */
void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, MemoryArena& ws,
                   CudaExecutionView execution);

} // namespace ninfer::ops

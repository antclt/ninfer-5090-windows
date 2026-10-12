// NVFP4-encoded DFlash2 drafter attention projection: the fused [6144,5120] parent writes q
// [4096,T], k [1024,T] and v [1024,T] directly. Port-only route -- upstream registers the
// NVFP4 split geometries n14336_k5120 and n16384_k5120 and carries no drafter geometry, so
// neither serves an NVFP4-encoded drafter such as the one this port's QUASAR artifacts contain.
// See include/ninfer/ops/linear_split.h for the contract this implements.
//
// The kernels, segmented output and epilogue are upstream's. This file selects among them per
// token count: a GEMV at one token, an exact-token SIMT family from 2 to 32, and a chunk walk
// above that.

#include "ops/linear/split/nvfp4/shapes/n6144_k5120/launch.h"

#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/common/output.cuh"
#include "ops/linear/kernels/nvfp4/nvfp4_dflash2_geometry.h"
#include "ops/linear/kernels/nvfp4/nvfp4_operands.h"
#include "ops/linear/kernels/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/kernels/nvfp4/nvfp4_template_launch.cuh"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace ninfer::ops::detail::linear_split::nvfp4_n6144_k5120 {
namespace {

using Geometry = Nvfp4DFlash2QkvGeometry;
using Output   = LinearBf16SegmentedOutput<4096, 1024, 1024>;

// Measured low-T warp mapping for the drafter's exact-token family. Upstream's
// Nvfp4A16SimtSchedule takes these as parameters, so the schedule itself is upstream's; only the
// per-token-count choice is this port's measurement.
template <int ActiveTokens>
struct Nvfp4DFlash2AttentionSmallTProductionSchedule {
    static_assert(ActiveTokens >= kNvfp4FirstSmallT);
    static_assert(ActiveTokens <= kNvfp4LastSmallT);
    static constexpr int kWarpsPerCta       = ActiveTokens >= 17 ? 4 : (ActiveTokens >= 8 ? 16 : 8);
    static constexpr int kValuesPerLane     = ActiveTokens >= 17 && ActiveTokens <= 20 ? 8 : 16;
    static constexpr auto kActivationAccess = ActiveTokens <= 4
                                                  ? Nvfp4SimtActivationAccess::SharedPhase
                                                  : Nvfp4SimtActivationAccess::TokenPacked;
    using Type =
        Nvfp4A16SimtSchedule<kWarpsPerCta, 1, 2, kValuesPerLane, ActiveTokens, 1, kActivationAccess,
                             Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                             Nvfp4SimtBlockOrder::RowsContiguous, 1>;
};

Output segmented(std::span<const Tensor> outputs) {
    return Output{static_cast<__nv_bfloat16*>(outputs[0].data),
                  static_cast<__nv_bfloat16*>(outputs[1].data),
                  static_cast<__nv_bfloat16*>(outputs[2].data)};
}

void launch_decode(const Tensor& x, const Weight& weight, std::span<const Tensor> outputs,
                   cudaStream_t stream) {
    using Schedule =
        Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>;
    launch_nvfp4_a16_gemv<Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>>(
        nvfp4_a16_operands(x, weight), segmented(outputs), LinearIdentityEpilogue{}, stream);
}

template <int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, std::span<const Tensor> outputs,
                  cudaStream_t stream) {
    using Schedule = typename Nvfp4DFlash2AttentionSmallTProductionSchedule<ActiveTokens>::Type;
    launch_nvfp4_a16_simt<
        Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows, ActiveTokens, true>>(
        nvfp4_a16_operands(x, weight), segmented(outputs), LinearIdentityEpilogue{}, stream);
}

template <std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    using Launch = void (*)(const Tensor&, const Weight&, std::span<const Tensor>, cudaStream_t);
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_exact<kNvfp4FirstSmallT + static_cast<int>(Offsets)>...};
}

constexpr auto kLaunchers =
    make_launchers(std::make_index_sequence<kNvfp4LastSmallT - kNvfp4FirstSmallT + 1>{});

} // namespace

void launch_a16(const Tensor& x, const Weight& weight, std::span<const Tensor> outputs,
                cudaStream_t stream) {
    constexpr std::int32_t kChunk = kNvfp4LastSmallT;
    for (std::int32_t token_begin = 0; token_begin < x.ne[1]; token_begin += kChunk) {
        const std::int32_t active = std::min(kChunk, x.ne[1] - token_begin);
        const Tensor x_slice      = x.slice(1, token_begin, active);
        const Tensor q_slice      = outputs[0].slice(1, token_begin, active);
        const Tensor k_slice      = outputs[1].slice(1, token_begin, active);
        const Tensor v_slice      = outputs[2].slice(1, token_begin, active);
        const std::array<const Tensor, 3> slices{q_slice, k_slice, v_slice};
        if (active == 1) {
            launch_decode(x_slice, weight, slices, stream);
        } else {
            kLaunchers[active - kNvfp4FirstSmallT](x_slice, weight, slices, stream);
        }
    }
}

} // namespace ninfer::ops::detail::linear_split::nvfp4_n6144_k5120

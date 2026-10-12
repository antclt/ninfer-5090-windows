#include "core/weight.h"
#include "ops/linear/swiglu/nvfp4/dispatch.h"

#include "ops/linear/kernels/nvfp4/nvfp4_instances.cuh"
#include "ops/linear/kernels/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/kernels/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear/swiglu/paired_output.cuh"
#include "ops/linear/swiglu/row_major_mma_epilogue.cuh"

#include <cuda_bf16.h>

#include <cstdint>

// The fused A16 routes above the small-T family. Both arms apply the nonlinearity to FP32
// accumulators, which is what holds the profile's A16 tail bound at every extent; routing these
// extents through a materialized BF16 projection does not (8.1e-3 against 6.3e-3 at 17/32/64/129).
// The A16 MMA schedules are the ones the registered [34816,5120] Linear selects at the same
// extents, and the row-major epilogue is the pairing every A16 MMA family here uses.
namespace ninfer::ops::detail {
namespace {

using Geometry              = Nvfp4N34816K5120;
constexpr int kIntermediate = Geometry::kOutputRows / 2;

template <class Schedule>
void launch_sliced(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using S = Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>;
    launch_nvfp4_a16_sliced_k_mma<S>(
        nvfp4_a16_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), kIntermediate},
        SwiGluPairEpilogue{}, stream, SwiGluBranchRows<8>{});
}

template <class Schedule>
void launch_mma(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using S = Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>;
    launch_nvfp4_a16_mma<S>(nvfp4_a16_operands(x, weight),
                            LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), kIntermediate},
                            SwiGluRowMajorMmaEpilogue{}, stream, SwiGluRowMajorMmaRows<S>{});
}

} // namespace

void launch_nvfp4_swiglu_matrix(const Tensor& x, const Weight& weight, Tensor& out,
                                       cudaStream_t stream) {
    if (x.ne[1] <= 32) {
        return launch_sliced<Nvfp4SlicedInstance<32, 8, 2>>(x, weight, out, stream);
    }
    if (x.ne[1] <= 64) {
        return launch_mma<Nvfp4A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>(x, weight, out, stream);
    }
    if (x.ne[1] <= 96) {
        return launch_mma<Nvfp4A16MmaSchedule<64, 96, 128, 64, 16, 1, 2>>(x, weight, out, stream);
    }
    launch_mma<Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>(x, weight, out, stream);
}

} // namespace ninfer::ops::detail

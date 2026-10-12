#pragma once
#include "ops/linear/kernels/fp8/fp8_launch.h"
#include "ops/linear/kernels/fp8/fp8_template_launch.cuh"
#include "ops/linear/kernels/fp8/fp8_instances.cuh"
#include <algorithm>

namespace ninfer::ops::detail {
template <class Geometry, class Schedule>
void fp8_linear_a16_gemv(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_fp8_a16_gemv<Fp8ScheduleInstance<Schedule, Geometry::kInputRows>>(
        fp8_a16_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
        LinearIdentityEpilogue{}, stream);
}

template <class Geometry, int Capacity, class Schedule, bool Exact = false>
void fp8_linear_a16_simt(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_fp8_a16_simt<Fp8ScheduleInstance<Schedule, Geometry::kInputRows, Capacity, Exact>>(
        fp8_a16_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
        LinearIdentityEpilogue{}, stream);
}

template <class Geometry, class Schedule>
void fp8_linear_a16_mma(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_fp8_a16_mma<Fp8ScheduleInstance<Schedule, Geometry::kInputRows>>(
        fp8_a16_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
        LinearIdentityEpilogue{}, stream);
}

template <class Geometry, class Schedule>
void fp8_linear_a16_sliced_k(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Schedule, Geometry::kInputRows>>(
        fp8_a16_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
        LinearIdentityEpilogue{}, stream);
}

template <class Geometry, class Schedule>
void launch_fp8_a8(const Tensor& x, const Weight& w, Tensor& out, Fp8A8Workspace scratch,
                   cudaStream_t stream) {
    launch_fp8_a8_quantize(x, w, scratch, stream);
    // scratch.partials is forwarded for the same reason launch_fp8_a8_tma forwards it: the moment a
    // schedule in this ladder selects a split-K tile, the kernel reads it. In this tree that cannot
    // happen -- every schedule with kSplitWaveCtas > 0 is a TMA one and takes the other launcher, and
    // no shape file here carries a non-Windows arm -- so the block is unreachable on this platform
    // while remaining part of upstream's shared source. Kept rather than removed: deleting it would
    // grow this port's divergence from upstream in order to delete code that is inert here, and a
    // shape file ungated later fails at runtime with "FP8 split-K requires aligned caller partials"
    // rather than at the call site. Reported as dead surface by a review on 2026-10-11, which is what
    // prompted this paragraph.
    launch_fp8_a8_mma<Fp8ScheduleInstance<Schedule, Geometry::kInputRows>>(
        fp8_a8_operands(w, scratch, x.ne[1]),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n}, LinearIdentityEpilogue{},
        stream, Fp8IdentityRows{}, scratch.partials);
}

template <class Geometry, class Schedule>
void launch_fp8_a8_tma(const Tensor& x, const Weight& w, Tensor& out, Fp8A8Workspace scratch,
                       cudaStream_t stream) {
    launch_fp8_a8_quantize(x, w, scratch, stream);
    launch_fp8_a8_tma_mma<Fp8ScheduleInstance<Schedule, Geometry::kInputRows>>(
        fp8_a8_operands(w, scratch, x.ne[1]),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n}, LinearIdentityEpilogue{},
        stream, scratch.partials);
}
} // namespace ninfer::ops::detail

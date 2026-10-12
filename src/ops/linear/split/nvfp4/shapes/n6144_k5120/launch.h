#pragma once
#include "ops/linear/split/dispatch.h"

namespace ninfer::ops::detail::linear_split::nvfp4_n6144_k5120 {

// The DFlash2 drafter's weight-only A16 NVFP4 route: one fused [6144,5120] parent writing the
// three ordered segments <4096,1024,1024>. Port-only -- see the shape's own comment.
void launch_a16(const Tensor&, const Weight&, std::span<const Tensor>, cudaStream_t);

} // namespace ninfer::ops::detail::linear_split::nvfp4_n6144_k5120

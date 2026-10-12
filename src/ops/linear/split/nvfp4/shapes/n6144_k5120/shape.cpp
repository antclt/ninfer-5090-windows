#include "ops/linear/split/nvfp4/shapes/n6144_k5120/launch.h"

namespace ninfer::ops::detail::linear_split::nvfp4_n6144_k5120 {
namespace {

// Weight-only A16: the drafter projection carries no activation quantization, so this route has
// no A4 branch and needs no caller workspace.
void launch(const Tensor& x, Weights weights, std::span<const Tensor> outputs, MemoryArena*,
            cudaStream_t stream) {
    launch_a16(x, weights.first, outputs, stream);
}

} // namespace
} // namespace ninfer::ops::detail::linear_split::nvfp4_n6144_k5120

namespace ninfer::ops::detail::linear_split {
const SplitShape kNvfp4SplitN6144K5120{6144, 5120, kSplitRows<4096, 1024, 1024>,
                                       nvfp4_n6144_k5120::launch};
} // namespace ninfer::ops::detail::linear_split

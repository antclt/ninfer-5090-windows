#pragma once

#include <cstdint>

namespace ninfer::ops::detail {

// Whether the query tiles already fill the SMs, which is the term the per-family wave-balance search
// is missing. That search compares only wave counts, and more splits always balance better, so it
// walks to its cap whenever the improvement is a single unit -- measured on this product's shapes,
// all eight steps from 1 to 8 qualify at a prefill width of 8192. Each split costs a full FP32
// accumulator of kCausalHeadDim depth per head per width (4*256+8 = 1032 bytes here), so at 24 heads
// that cap costs 1.62 GiB of the 2.246 GiB workspace to hold partials for work that was never short
// of occupancy.
//
// FlashAttention's num_splits_heuristic carries the missing term and states the reason: it returns
// 1 split once the query tiles already fill 80% of the SMs, "however, we also don't want too many
// splits as that would incur more HBM reads/writes". Here ctas is 1536 against a threshold of 136,
// so the guard is not marginal -- it is the difference between splitting a saturated launch and not.
// Below the threshold the wave-balance search still runs, because there the splits are what fill
// the machine.
inline bool tiled_partition_saturated(std::int64_t ctas, int multiprocessor_count) {
    return multiprocessor_count > 0 && ctas * 5 >= static_cast<std::int64_t>(multiprocessor_count) * 4;
}

} // namespace ninfer::ops::detail

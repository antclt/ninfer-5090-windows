#include "ops/softmax_attention/dense/causal_cache/k8v4/plan.h"
#include "ops/softmax_attention/dense/causal_cache/k8v4/operands.h"
#include "ops/softmax_attention/common/mxfp8_tiled_plan.h"
#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
constexpr int kGroupedPrefillMaxWidth = 80;
} // namespace

K8V4KvCausalPlan make_k8v4_kv_causal_plan(int heads, int width, int batch,
                                          CausalAttentionExecutionEnvelope envelope,
                                          int multiprocessor_count) {
    if (multiprocessor_count <= 0 || (heads != 24 && heads != 16) || width < 1 || batch < 1 ||
        batch > 8 || (batch > 1 && width > 16) || envelope.min_visible_keys == 0 ||
        envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > kCausalAttentionMaximumVisibleKeys)
        throw std::invalid_argument("K8V4 attention: invalid plan inputs");
    constexpr int grouped_limit = K8V4KvCausalPlan::kTokenTile;
    const auto family           = width <= grouped_limit             ? K8V4KvFamily::Grouped
                                  : width <= kGroupedPrefillMaxWidth ? K8V4KvFamily::ParallelGrouped
                                                                     : K8V4KvFamily::Tiled;
    // The parallel path partitions the QUERY axis, and every chunk sweeps the whole visible key range,
    // so the number of chunks is the number of times the KV is read: at W=10 the old `(width+1)/2`
    // split it into two chunks of five and read the KV twice. The kernel's own bound is
    // `Tokens * GroupSize <= 64`, and GroupSize is QHeads/KVHeads -- 6 here, 8 on the 16-head geometry
    // -- so a 24-head model can take a 10-wide query tile in one chunk, which is exactly what a
    // DFlash2 lane at draft 9 runs. Above that the halving stays, so no other width changes shape.
    constexpr int kHeads24QueryTileMax = 10;
    const int max_query_tile = heads == 24 ? kHeads24QueryTileMax : grouped_limit;
    if (family == K8V4KvFamily::Tiled)
        return {
            family,
            heads,
            width,
            batch,
            0,
            envelope,
            mxfp8_tiled_partition(heads, width, envelope.max_visible_keys, multiprocessor_count)};
    const int tiles = family == K8V4KvFamily::ParallelGrouped && width > max_query_tile
                          ? (width + grouped_limit - 1) / grouped_limit
                          : 1;
    // Three bands, and the guard on the middle one is load-bearing: below the wider tile the whole
    // width is one chunk; up to 16 it is halved; above that it is capped at the token tile, because
    // the launch enumerates query tiles 5..10 and a halved width of 26 is not one of them. Dropping
    // that guard is what made a 51-column warmup prefill fail with "K8V4 parallel plan exceeds its
    // query tiles" -- the lane refused to serve rather than computing anything wrong.
    const int query_tile =
        family == K8V4KvFamily::ParallelGrouped
            ? (width <= max_query_tile ? width
                                       : (width <= 16 ? (width + 1) / 2 : grouped_limit))
            : std::min(width, grouped_limit);
    const int independent_tiles = batch * (heads == 24 ? 4 : 2) * tiles;
    // Decode budgets two CTAs per SM; wider queries start with one and double the
    // budget if complete query-tile groups would leave over 10% of the SMs idle.
    const std::int64_t sms = multiprocessor_count;
    const auto budget =
        width == 1 || causal_query_tiles_underfill_sms(independent_tiles, multiprocessor_count)
            ? 2 * sms
            : sms;
    CausalKvPartition partition{1, causal_partition_target(budget, independent_tiles)};
    // Bound partial traffic by keeping enough KV work in each split.
    partition.key_shift = (width == 1 ? 7 : 8) - (heads == 16 ? 1 : 0);
    partition.capacity  = partition.active(envelope.max_visible_keys);
    return {family, heads, width, batch, query_tile, envelope, partition};
}

std::size_t k8v4_kv_workspace_bytes(int heads, int batch, int min_width, int max_width,
                                    CausalAttentionExecutionEnvelope envelope,
                                    int multiprocessor_count) {
    std::size_t maximum = 0;
    for (int width = min_width; width <= std::min(max_width, kGroupedPrefillMaxWidth); ++width) {
        const auto plan =
            make_k8v4_kv_causal_plan(heads, width, batch, envelope, multiprocessor_count);
        if (plan.family == K8V4KvFamily::Tiled) continue;
        const int splits = plan.partition.capacity;
        WorkspaceLayoutBuilder layout;
        (void)allocate_causal_partials(layout, heads, width, splits, batch);
        maximum = std::max(maximum, layout.peak_bytes(1));
    }
    return std::max(maximum, mxfp8_tiled_workspace_bytes(
                                 heads, std::max(min_width, kGroupedPrefillMaxWidth + 1), max_width,
                                 envelope.max_visible_keys, multiprocessor_count));
}

} // namespace ninfer::ops::detail

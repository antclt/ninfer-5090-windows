#pragma once

#include "core/layout.h"
#include "core/paged_kv_storage.h"
#include "core/paged_kv_geometry.h"
#include "core/tensor.h"


#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer {

/** Non-owning, single-sequence view consumed by growing-cache Ops. */
struct PagedKVLayerView {
    Tensor k_pages;
    Tensor v_pages;
    Tensor k_scale_pages;
    Tensor v_scale_pages;
    Tensor block_table;
    std::int32_t head_dim     = 0;
    std::int32_t num_kv_heads = 0;
    KvCacheStorage storage    = KvCacheStorage::BFloat16;
};

/** Non-owning multi-sequence view consumed by batched growing-cache Ops. */
struct PagedKVBatchLayerView {
    Tensor k_pages;
    Tensor v_pages;
    Tensor k_scale_pages;
    Tensor v_scale_pages;
    Tensor block_tables;
    std::int32_t head_dim     = 0;
    std::int32_t num_kv_heads = 0;
    KvCacheStorage storage    = KvCacheStorage::BFloat16;

    [[nodiscard]] Device device() const;
};

/** Rebinds one checked single-sequence table row as a one-row batched view. */
[[nodiscard]] PagedKVBatchLayerView single_row_paged_kv_batch_view(const PagedKVLayerView& cache);

/** One layer's device planes for a resolved storage layout, in device order.
 *
 * The order is the contract the Op readers index by -- kv_cache_append and
 * causal_softmax_attention walk these planes positionally -- and which planes exist is a property
 * of the storage layout, not of any one model. Both builders in qwen3_5 wrote this list out by
 * hand -- four pushes in decoder_state, two in startup because that storage is BFloat16 and has no
 * scale planes -- and one of them spelled the alignment as a literal that already defaulted to 256. A layer's meaning is still assigned by repetition at the call site, which is what
 * KVPlaneGeometry's "a plane is storage-only" note asks for.
 */
[[nodiscard]] inline std::vector<KVPlaneGeometry> storage_planes(const PagedKVStorageLayout& layout,
                                                                std::int32_t kv_heads) {
    std::vector<KVPlaneGeometry> planes;
    planes.reserve(layout.planes_per_layer());
    planes.push_back({layout.key.data_dtype, layout.key.data_leading_extent, kv_heads});
    planes.push_back({layout.value.data_dtype, layout.value.data_leading_extent, kv_heads});
    if (layout.key.has_scale()) {
        planes.push_back({layout.key.scale_dtype, layout.key.scale_leading_extent, kv_heads});
    }
    if (layout.value.has_scale()) {
        planes.push_back({layout.value.scale_dtype, layout.value.scale_leading_extent, kv_heads});
    }
    return planes;
}
struct DeviceKVPagePoolSpec {
    std::uint32_t page_group_count = 0;
    KVPageGeometry geometry;
};

struct KVExecutionTableSpec {
    std::uint32_t logical_page_capacity = 0;
    std::int32_t table_rows             = 0;
};

struct DeviceKVPlaneLayout {
    KVPlaneGeometry geometry;
    TensorRegion storage;
};

struct DeviceKVPagePoolLayout {
    DeviceKVPagePoolSpec spec;
    std::vector<DeviceKVPlaneLayout> planes;

    [[nodiscard]] std::size_t payload_bytes() const noexcept;
};

struct KVExecutionTableLayout {
    KVExecutionTableSpec spec;
    TensorRegion block_tables;
};

[[nodiscard]] DeviceKVPagePoolLayout plan_device_kv_page_pool(LayoutBuilder& builder,
                                                              const DeviceKVPagePoolSpec& spec);

[[nodiscard]] KVExecutionTableLayout plan_kv_execution_tables(LayoutBuilder& builder,
                                                              const KVExecutionTableSpec& spec);

} // namespace ninfer

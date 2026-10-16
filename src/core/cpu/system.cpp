#include "core/cpu/system.h"
// The platform-independent half (affinity validation, worker ordering, feature masking, worker
// startup) lives here so the port's Windows implementation of this file can share it. Port change:
// a Windows topology source cannot reuse an hwloc-backed .cpp, and duplicating that half would put
// two copies of it in the tree.
#include "core/cpu/system_portable.inc"

#include <hwloc.h>

#include <algorithm>
#include <cerrno>
#include <condition_variable>
#include <exception>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <system_error>
#include <tuple>
#include <utility>

namespace ninfer {
namespace {

struct Bitmap {
    Bitmap() : value(hwloc_bitmap_alloc()) {
        if (!value) throw std::bad_alloc();
    }

    ~Bitmap() { hwloc_bitmap_free(value); }

    Bitmap(const Bitmap&)            = delete;
    Bitmap& operator=(const Bitmap&) = delete;
    hwloc_bitmap_t value;
};

void check(int result, const char* operation) {
    if (result < 0) throw std::system_error(errno, std::generic_category(), operation);
}

CpuAffinity cpu_list(hwloc_const_bitmap_t bitmap) {
    CpuAffinity result;
    int cpu;
    hwloc_bitmap_foreach_begin(cpu, bitmap) { result.push_back(cpu); }
    hwloc_bitmap_foreach_end();
    return result;
}

bool data_cache(hwloc_obj_type_t type) {
    return type == HWLOC_OBJ_L1CACHE || type == HWLOC_OBJ_L2CACHE || type == HWLOC_OBJ_L3CACHE ||
           type == HWLOC_OBJ_L4CACHE || type == HWLOC_OBJ_L5CACHE;
}

} // namespace

struct CpuTopology::Impl {
    hwloc_topology_t topology = nullptr;
    CpuTopologyInfo info;

    ~Impl() {
        if (topology) hwloc_topology_destroy(topology);
    }
};

CpuTopology::CpuTopology() : impl_(std::make_unique<Impl>()) {
    auto& state = *impl_;
    check(hwloc_topology_init(&state.topology), "initialize CPU topology");
    check(hwloc_topology_set_flags(state.topology, HWLOC_TOPOLOGY_FLAG_DONT_CHANGE_BINDING),
          "configure CPU topology");
    check(hwloc_topology_set_io_types_filter(state.topology, HWLOC_TYPE_FILTER_KEEP_NONE),
          "disable CPU topology I/O discovery");
    check(hwloc_topology_load(state.topology), "discover CPU topology");
    Bitmap allowed;
    check(hwloc_get_cpubind(state.topology, allowed.value, HWLOC_CPUBIND_THREAD),
          "read CPU affinity");
    hwloc_bitmap_and(allowed.value, allowed.value,
                     hwloc_topology_get_allowed_cpuset(state.topology));
    std::map<hwloc_obj_t, std::uint32_t> caches;
    hwloc_obj_t pu = nullptr;
    while ((pu = hwloc_get_next_obj_by_type(state.topology, HWLOC_OBJ_PU, pu))) {
        if (!hwloc_bitmap_isset(allowed.value, pu->os_index)) continue;
        CpuProcessor processor{.cpu = pu->os_index};
        hwloc_obj_t cache = nullptr;
        for (auto parent = pu->parent; parent; parent = parent->parent) {
            if (parent->type == HWLOC_OBJ_CORE) processor.core = parent->logical_index;
            if (data_cache(parent->type)) cache = parent;
        }
        if (cache) {
            const auto [it, inserted] = caches.emplace(cache, state.info.caches.size());
            processor.cache           = it->second;
            if (inserted) {
                state.info.caches.push_back({it->second, cache->attr->cache.depth,
                                             cache->attr->cache.size, cache->attr->cache.linesize});
            }
        }
        state.info.processors.push_back(processor);
    }
    if (state.info.processors.empty()) throw std::runtime_error("no available CPUs");
}

CpuTopology::~CpuTopology() = default;

const CpuTopologyInfo& CpuTopology::info() const noexcept { return impl_->info; }

CpuAffinity current_thread_affinity(const CpuTopology& topology) {
    Bitmap bitmap;
    check(hwloc_get_cpubind(topology.impl_->topology, bitmap.value, HWLOC_CPUBIND_THREAD),
          "read CPU affinity");
    return cpu_list(bitmap.value);
}

// validate_cpu_affinity, cpu_worker_order and plan_cpu_team are platform-independent and live in
// system_portable.inc.

void bind_current_thread(const CpuTopology& topology, std::span<const std::uint32_t> cpus) {
    validate_cpu_affinity(topology.info(), cpus);
    Bitmap requested;
    for (const auto cpu : cpus) hwloc_bitmap_set(requested.value, cpu);
    check(hwloc_set_cpubind(topology.impl_->topology, requested.value,
                            HWLOC_CPUBIND_THREAD | HWLOC_CPUBIND_STRICT),
          "bind CPU thread");
    Bitmap actual;
    check(hwloc_get_cpubind(topology.impl_->topology, actual.value, HWLOC_CPUBIND_THREAD),
          "verify CPU affinity");
    if (!hwloc_bitmap_isequal(actual.value, requested.value))
        throw std::runtime_error("CPU affinity did not match the requested set");
}

// cpu_worker_order and plan_cpu_team are platform-independent and live in system_portable.inc.

// cpu_detail::usable_features is platform-independent (it masks raw register values) and lives in
// system_portable.inc.

// current_cpu_features and current_cpu_identity read the CPUID leaves directly, so each platform
// keeps its own implementation: cpu/x86.cpp upstream (GCC <cpuid.h>), the port's
// cpu/system_windows.cpp (MSVC intrinsics). Neither belongs in this hwloc-backed Linux file.

// cpu_detail::start_workers and cpu_detail::join_workers are platform-independent and live
// in system_portable.inc.

} // namespace ninfer

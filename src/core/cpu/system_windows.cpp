// Native Windows implementation of the CPU topology, affinity and feature query.
//
// Upstream's cpu/system.cpp answers these through hwloc and <cpuid.h>, and
// src/core/CMakeLists.txt gates it to Linux x86-64. This file is the port's own implementation of
// the same core/cpu/system.h interface for MSVC, selected by CMake on Windows; the
// platform-independent half (affinity validation, worker ordering, feature masking, worker startup)
// is shared with upstream's file through core/cpu/system_portable.inc.
//
// Two Windows properties shape the code below and are worth stating where they are relied on:
//
//  * A logical processor is addressed as (processor group, bit within that group), and a group
//    holds at most 64 of them. CpuAffinity flattens the pair to `group * 64 + bit` so it stays a
//    single ordered sequence, which is also the numbering a NUMA-aware caller expects.
//  * A thread belongs to exactly one processor group. GetThreadGroupAffinity reports the thread's
//    affinity over its primary group, and SetThreadGroupAffinity restricts the thread to one group.
//    That is not a limitation for this layer's actual use: plan_cpu_team assigns one fixed CPU per
//    worker, so every binding it produces is single-group. bind_current_thread refuses a set that
//    spans groups rather than silently binding the subset it can express.

#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0601  // Get/SetThreadGroupAffinity and GetLogicalProcessorInformationEx
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include "core/cpu/system.h"
#include "core/cpu/system_portable.inc"

#include <windows.h>

#include <intrin.h>
#include <immintrin.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <system_error>
#include <vector>

namespace ninfer {
namespace {

// A processor group holds at most 64 logical processors; the bit index is the processor's index
// within its group. KAFFINITY is ULONG_PTR, so one bit covers every index a group can hold.
constexpr std::uint32_t kProcessorsPerGroup = 64;

std::uint32_t flatten(WORD group, std::uint32_t bit) {
    return static_cast<std::uint32_t>(group) * kProcessorsPerGroup + bit;
}

WORD group_of(std::uint32_t cpu) {
    return static_cast<WORD>(cpu / kProcessorsPerGroup);
}

KAFFINITY bit_of(std::uint32_t cpu) {
    return KAFFINITY{1} << (cpu % kProcessorsPerGroup);
}

[[noreturn]] void fail(const char* operation) {
    throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(), operation);
}

// The documented two-call contract: a null buffer reports the required length through
// ERROR_INSUFFICIENT_BUFFER, and the second call fills it.
std::vector<std::byte> processor_information(LOGICAL_PROCESSOR_RELATIONSHIP relationship) {
    DWORD length = 0;
    ::GetLogicalProcessorInformationEx(relationship, nullptr, &length);
    if (::GetLastError() != ERROR_INSUFFICIENT_BUFFER || length == 0) {
        fail("size CPU topology");
    }
    std::vector<std::byte> buffer(length);
    if (::GetLogicalProcessorInformationEx(
            relationship, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()),
            &length) == FALSE) {
        fail("discover CPU topology");
    }
    buffer.resize(length);
    return buffer;
}

} // namespace

struct CpuTopology::Impl {
    CpuTopologyInfo info;
};

CpuTopology::CpuTopology() : impl_(std::make_unique<Impl>()) {
    auto& state = *impl_;

    // One RelationProcessorCore record per core, carrying the mask of every logical processor that
    // shares it. A core never spans groups, but iterate GroupCount rather than assuming it.
    const auto cores = processor_information(RelationProcessorCore);
    std::uint32_t core_index = 0;
    for (const auto* entry = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(
             cores.data());
         reinterpret_cast<const std::byte*>(entry) < cores.data() + cores.size();
         entry = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(
             reinterpret_cast<const std::byte*>(entry) + entry->Size)) {
        if (entry->Relationship != RelationProcessorCore) continue;
        const auto& core = entry->Processor;
        for (WORD index = 0; index < core.GroupCount; ++index) {
            const auto& mask = core.GroupMask[index];
            for (std::uint32_t bit = 0; bit < kProcessorsPerGroup; ++bit) {
                if ((mask.Mask & (KAFFINITY{1} << bit)) == 0) continue;
                state.info.processors.push_back(
                    CpuProcessor{flatten(mask.Group, bit), core_index, std::nullopt});
            }
        }
        ++core_index;
    }

    // Limited to the discovering thread's allowed affinity: a thread that inherited a restricted
    // set must not see processors it cannot use. GetThreadGroupAffinity reports the thread's
    // affinity over its primary group, which is exactly the set bind_current_thread can express, so
    // the snapshot and the binding agree. On a system with more than 64 logical processors this
    // makes the snapshot one group wide -- the widest a single thread's affinity can be.
    GROUP_AFFINITY allowed{};
    if (::GetThreadGroupAffinity(::GetCurrentThread(), &allowed) == FALSE) {
        fail("read CPU affinity");
    }
    state.info.processors.erase(
        std::remove_if(state.info.processors.begin(), state.info.processors.end(),
                       [&](const CpuProcessor& processor) {
                           return group_of(processor.cpu) != allowed.Group ||
                                  (allowed.Mask & bit_of(processor.cpu)) == 0;
                       }),
        state.info.processors.end());
    if (state.info.processors.empty()) throw std::runtime_error("no available CPUs");

    // Data and unified caches only, assigned in ascending level so that each processor keeps the
    // outermost one containing it -- the same choice the hwloc walk makes by overwriting on the way
    // up the tree, and the property cpu_worker_order sorts by.
    struct Found {
        WORD group;
        KAFFINITY mask;
        std::uint32_t level;
        std::uint32_t line_bytes;
        std::uint64_t bytes;
    };
    std::vector<Found> found;
    const auto caches = processor_information(RelationCache);
    for (const auto* entry = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(
             caches.data());
         reinterpret_cast<const std::byte*>(entry) < caches.data() + caches.size();
         entry = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(
             reinterpret_cast<const std::byte*>(entry) + entry->Size)) {
        if (entry->Relationship != RelationCache) continue;
        const auto& cache = entry->Cache;
        if (cache.Type != CacheData && cache.Type != CacheUnified) continue;
        // A cache record carries one group affinity, not an array: its GroupCount is always 1.
        found.push_back(Found{cache.GroupMask.Group, cache.GroupMask.Mask, cache.Level,
                              cache.LineSize, cache.CacheSize});
    }
    std::stable_sort(found.begin(), found.end(),
                     [](const Found& a, const Found& b) { return a.level < b.level; });
    for (const auto& cache : found) {
        const auto id = static_cast<std::uint32_t>(state.info.caches.size());
        state.info.caches.push_back({id, cache.level, cache.bytes, cache.line_bytes});
        for (auto& processor : state.info.processors) {
            if (group_of(processor.cpu) != cache.group) continue;
            if ((cache.mask & bit_of(processor.cpu)) != 0) processor.cache = id;
        }
    }
}

CpuTopology::~CpuTopology() = default;

const CpuTopologyInfo& CpuTopology::info() const noexcept { return impl_->info; }

CpuAffinity current_thread_affinity(const CpuTopology& /*topology*/) {
    GROUP_AFFINITY affinity{};
    if (::GetThreadGroupAffinity(::GetCurrentThread(), &affinity) == FALSE) {
        fail("read CPU affinity");
    }
    CpuAffinity result;
    for (std::uint32_t bit = 0; bit < kProcessorsPerGroup; ++bit) {
        if ((affinity.Mask & (KAFFINITY{1} << bit)) != 0) {
            result.push_back(flatten(affinity.Group, bit));
        }
    }
    return result;
}

// validate_cpu_affinity, cpu_worker_order and plan_cpu_team are platform-independent and live in
// system_portable.inc.

void bind_current_thread(const CpuTopology& topology, std::span<const std::uint32_t> cpus) {
    validate_cpu_affinity(topology.info(), cpus);
    const WORD group = group_of(cpus.front());
    for (const auto cpu : cpus) {
        if (group_of(cpu) != group) {
            throw std::invalid_argument(
                "CPU affinity spans Windows processor groups; a thread belongs to one group");
        }
    }
    GROUP_AFFINITY requested{};
    requested.Group = group;
    for (const auto cpu : cpus) requested.Mask |= bit_of(cpu);
    if (::SetThreadGroupAffinity(::GetCurrentThread(), &requested, nullptr) == FALSE) {
        fail("bind CPU thread");
    }
    GROUP_AFFINITY actual{};
    if (::GetThreadGroupAffinity(::GetCurrentThread(), &actual) == FALSE) {
        fail("verify CPU affinity");
    }
    if (actual.Group != requested.Group || actual.Mask != requested.Mask) {
        throw std::runtime_error("CPU affinity did not match the requested set");
    }
}

// cpu_detail::usable_features is platform-independent (it masks raw register values) and lives in
// system_portable.inc.

CpuFeatures current_cpu_features() noexcept {
    cpu_detail::FeatureRegisters state;
    int registers[4] = {};
    __cpuid(registers, 1);
    state.leaf1_ecx = static_cast<std::uint32_t>(registers[2]);
    // OSXSAVE (leaf 1 ECX bit 27) is what makes XGETBV legal, so it gates the read rather than the
    // reverse: reading XCR0 where the OS never enabled it faults.
    if ((state.leaf1_ecx & (1U << 27)) != 0) {
        state.xcr0 = static_cast<std::uint64_t>(_xgetbv(0));
    }
    __cpuid(registers, 0);
    if (registers[0] >= 7) {
        __cpuidex(registers, 7, 0);
        const int max_subleaf = registers[0];
        state.leaf7_0_ebx    = static_cast<std::uint32_t>(registers[1]);
        state.leaf7_0_ecx    = static_cast<std::uint32_t>(registers[2]);
        if (max_subleaf >= 1) {
            __cpuidex(registers, 7, 1);
            state.leaf7_1_eax = static_cast<std::uint32_t>(registers[0]);
        }
    }
    return cpu_detail::usable_features(state);
}

CpuIdentity current_cpu_identity() noexcept {
    CpuIdentity result;
    int registers[4] = {};
    __cpuid(registers, 0);
    // The vendor string is EBX, EDX, ECX in that order.
    std::memcpy(result.vendor.data(), &registers[1], 4);
    std::memcpy(result.vendor.data() + 4, &registers[3], 4);
    std::memcpy(result.vendor.data() + 8, &registers[2], 4);
    __cpuid(registers, 1);
    const auto eax    = static_cast<std::uint32_t>(registers[0]);
    const auto ecx    = static_cast<std::uint32_t>(registers[2]);
    const auto family = (eax >> 8) & 15U;
    result.family     = family + (family == 15 ? (eax >> 20) & 255U : 0);
    result.model    = ((eax >> 4) & 15U) | ((family == 6 || family == 15) ? (eax >> 12) & 240U : 0);
    result.stepping = eax & 15U;
    result.virtualized = (ecx & (1U << 31)) != 0;
    return result;
}

// cpu_detail::start_workers and cpu_detail::join_workers are platform-independent and live in
// system_portable.inc.

} // namespace ninfer

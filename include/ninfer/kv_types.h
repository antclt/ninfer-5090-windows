#pragma once

#include <cstdint>

namespace ninfer {

// The KV storage formats, alone in their own header.
//
// This enum lived in `ninfer/types.h`, which is 45 KB carrying the whole product's public surface, and
// the device tree reached it through one include: `src/core/paged_kv_storage.h` names this enum and
// nothing else from that file. Measured 2026-10-11 in build-test: touching `ninfer/types.h` rebuilt 314
// objects, 298 host and 16 CUDA, so adding a single field to a stats struct recompiled the ops.
//
// A narrow header for the one name the device side needs is the fix. `types.h` includes this, so the
// public surface is unchanged; `src/core/paged_kv_storage.h` and `ninfer/kv_storage.h` include it
// instead, so the ops and the KV codecs no longer see the stats at all.
enum class KvCacheStorage : std::uint8_t {
    BFloat16,
    Int8Group64,
    Fp8E4M3Row256,
    Nvfp4Group16,
    Fp8KeyNvfp4Value,
};

} // namespace ninfer

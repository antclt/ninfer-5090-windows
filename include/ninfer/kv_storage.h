#pragma once

#include "ninfer/types.h"

#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer {

// One name table for one format. `KvCacheStorage` is the authority for what a KV storage *is*; these
// two functions are the things every consumer was writing out by hand -- the name it prints and the
// flag it parses -- and there were four copies of the parse and five of the name.
//
// The spelling is the engine's own: "fp8-e4m3-r256" and "int8-g64", which is what the top-k record's
// 32-byte field carries (`apps/perplexity/topk_record.h` and `tools/release/per_domain_kl.py` both
// cite it) and what `per_domain_kl.py` *compares* between two records, refusing the pair when they
// differ. A second spelling in a log or a report is therefore not a cosmetic difference: it is a
// value that cannot be compared with the first. The flag names stay short -- bf16, int8, fp8, nvfp4,
// k8v4 -- because those are what `--kv-dtype` accepts and what the usage strings list.
inline const char* kv_storage_name(KvCacheStorage storage) noexcept {
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return "bf16";
    case KvCacheStorage::Int8Group64:
        return "int8-g64";
    case KvCacheStorage::Fp8E4M3Row256:
        return "fp8-e4m3-r256";
    case KvCacheStorage::Nvfp4Group16:
        return "nvfp4";
    case KvCacheStorage::Fp8KeyNvfp4Value:
        return "k8v4";
    }
    return "unknown";
}

inline KvCacheStorage parse_kv_storage(std::string_view text) {
    if (text == "bf16") { return KvCacheStorage::BFloat16; }
    if (text == "int8") { return KvCacheStorage::Int8Group64; }
    if (text == "fp8") { return KvCacheStorage::Fp8E4M3Row256; }
    if (text == "nvfp4") { return KvCacheStorage::Nvfp4Group16; }
    if (text == "k8v4") { return KvCacheStorage::Fp8KeyNvfp4Value; }
    throw std::invalid_argument("invalid kv-dtype: " + std::string(text));
}

} // namespace ninfer

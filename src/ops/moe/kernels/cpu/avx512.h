#pragma once

#include "ops/common/cpu/avx512.h"

namespace ninfer::ops::detail::moe_cpu {
struct Avx512 {
    using Float                = __m512;
    static constexpr int lanes = 16, rows = 4, max_columns = 4;
    static constexpr bool bf16_gate = false;

    static Float zero() noexcept { return _mm512_setzero_ps(); }

    static Float load(const float* p) noexcept { return _mm512_loadu_ps(p); }

    static void store(float* p, Float v) noexcept { _mm512_storeu_ps(p, v); }

    static Float bf16(const std::uint16_t* p) noexcept {
        return ::ninfer::ops::common::cpu::avx512::load_bf16(p);
    }

    static Float fma(Float a, Float b, Float c) noexcept { return _mm512_fmadd_ps(a, b, c); }

    // _cvtsh_ss is AVX512-FP16. MSVC 14.51 does not declare it and /arch has no switch for it
    // (measured: /arch:AVX512FP16 is rejected as an unknown option), so this header cannot name it.
    // F16C's vcvtph2ps is the same IEEE binary16 -> binary32 conversion, and an AVX-512 CPU has F16C.
    static Float scale(std::uint16_t bits) noexcept {
        return _mm512_set1_ps(_mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(bits))));
    }

    static float sum(Float v) noexcept { return ::ninfer::ops::common::cpu::avx512::reduce_sum(v); }

    template <int Bits>
    static Float decode(const std::uint8_t* code, const std::uint8_t* high) noexcept {
        return common::cpu::avx512::decode_signed_q<Bits>(code, high);
    }
};
} // namespace ninfer::ops::detail::moe_cpu

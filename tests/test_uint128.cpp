// The 128-bit integer the cost-model arithmetic is written against.
//
// The same assertions run against both branches of this header: on GCC and Clang `Uint128` is the
// builtin `unsigned __int128`, on MSVC it is the fallback struct, so a case that passes on one and
// fails on the other is exactly the divergence the fallback exists to prevent. Nothing here touches
// `.lo`, `.hi` or `widening_multiply_high`, because those exist only in the fallback branch.
//
// Expectations are derived rather than recorded, and two of them are derivations the header's own
// comments supply: (2^64-1)^2 is 2^128 - 2^65 + 1, so its truncation to 128 bits is 1 - 2^65; and
// 2^64 is congruent to 1 modulo 3, so 2^64 % 3 is 1 and (2^64 - 1) % 3 is 0. The divisor case is the
// bug that comment records as fixed: "comparing rem[2] against divisor.hi instead is off by 64 bits,
// which made every dividend below 2^128 look larger than a 10^20 divisor and returned a zero
// quotient".
#include "core/uint128.h"

#include <cstdint>
#include <iostream>

namespace {

using ninfer::Uint128;

constexpr Uint128 kOne = Uint128{1};

// The header's claim that every operation stays a constant expression, asserted rather than trusted.
static_assert((kOne << 64) >> 64 == kOne);
static_assert(Uint128{0} + Uint128{0} == Uint128{0});
static_assert((kOne << 64) / 2 == kOne << 63);
static_assert(((kOne << 64) - kOne) % 3 == Uint128{0});

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;

    // Shifts at each boundary the header special-cases: zero, below 64, exactly 64, above it, 128, and
    // the bit that falls off the top.
    failures += check((kOne << 0) == kOne, "a shift by zero changed the value");
    failures += check((kOne << 63) == (kOne << 64) >> 1, "a 63-bit shift disagreed with a 64/1 pair");
    failures += check((kOne << 64) >> 64 == kOne, "a 64-bit round trip lost the high word");
    failures += check((kOne << 127) >> 127 == kOne, "a 127-bit round trip lost the top bit");
    failures += check((kOne << 128) == Uint128{0}, "a shift by 128 did not clear the value");
    failures += check(((kOne << 127) << 1) == Uint128{0}, "shifting the top bit out did not wrap");

    // Carries and borrows across the word boundary, and the low word a borrow leaves behind.
    failures += check((kOne << 64) + (kOne << 64) == (kOne << 65), "the carry was lost");
    failures += check(((kOne << 64) - kOne) + kOne == (kOne << 64), "the borrow was lost");
    failures += check(static_cast<std::uint64_t>((kOne << 64) - kOne) == 0xFFFFFFFFFFFFFFFFULL,
                      "the low word after a borrow is not all ones");

    // (2^64 - 1)^2 truncated to 128 bits is 1 - 2^65, which is what makes this a test of the widening
    // multiply's high word rather than of its low one.
    const Uint128 all_ones = (kOne << 64) - kOne;
    failures += check(all_ones * all_ones + (kOne << 65) == kOne,
                      "the widening multiply's high word is wrong");

    // Division and remainder by a 64-bit divisor.
    failures += check((kOne << 64) / 2 == kOne << 63, "an exact 64-bit division is wrong");
    failures += check((kOne << 64) % 3 == kOne, "2^64 is not 1 modulo 3");
    failures += check(((kOne << 64) - kOne) % 3 == Uint128{0}, "2^64 - 1 is not 0 modulo 3");
    failures += check((kOne << 64) / 3 * 3 + (kOne << 64) % 3 == (kOne << 64),
                      "the quotient and remainder do not reconstruct the dividend");

    // Division by a 128-bit divisor, which is the path the header records a bug in. 10^20 is built as
    // a 128-bit value because it does not fit in 64 bits.
    const Uint128 divisor = Uint128{10000000000000000000ULL} * 10;
    failures += check(divisor / divisor == kOne, "a value divided by itself is not one");
    failures += check(divisor * 3 / divisor == Uint128{3},
                      "a dividend above the divisor returned a zero quotient");
    failures += check((divisor - kOne) / divisor == Uint128{0},
                      "a dividend below the divisor returned a non-zero quotient");
    failures += check(divisor * 3 % divisor == Uint128{0}, "an exact 128-bit remainder is not zero");
    failures += check((divisor * 3 + kOne) % divisor == kOne, "a 128-bit remainder is wrong");

    // The comparisons, which is what the sentinel arithmetic reads.
    failures += check(divisor > divisor - kOne, "a larger value did not compare greater");
    failures += check(divisor - kOne < divisor, "a smaller value did not compare less");
    failures += check(divisor >= divisor && divisor <= divisor, "a value did not compare equal to itself");

    std::cout << (failures == 0 ? "OK" : "FAIL") << " uint128\n";
    return failures == 0 ? 0 : 1;
}

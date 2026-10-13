#pragma once

// `__builtin_unreachable` is a GCC/Clang builtin; MSVC spells the same promise `__assume`.
// Both exist for one shape of code: a switch over an enum that covered every enumerator, where the
// fall-off-the-end path is a promise to the optimizer rather than a runtime case. Measured on this
// port: MSVC 14.51 does not declare `__builtin_unreachable` (C3861), so the call sites cannot name it.
//
// The [[noreturn]] attribute carries the author's promise and is what lets the optimizer drop the
// path after the call. It is a promise rather than a fact here in the same way the builtin is: the
// enumerators are covered, so control really cannot arrive.
namespace ninfer::ops::common {

#if defined(_MSC_VER)
[[noreturn]] inline void unreachable() noexcept { __assume(false); }
#else
[[noreturn]] inline void unreachable() noexcept { __builtin_unreachable(); }
#endif

} // namespace ninfer::ops::common

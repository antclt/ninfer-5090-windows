#include "ops/linear_add/linear_add_test_common.h"
#include "ops/linear_common.h"

#include <array>
#include <cstdint>

namespace {

using namespace ninfer;
using namespace ninfer::test::linear_add;

struct Profile {
    std::int32_t n;
    std::int32_t k;
    std::int32_t first_a4;
    std::uint32_t seed;
};

int run_profile(const Profile& profile) {
    const std::array invocations{
        Invocation{.tokens = 1, .policy = ops::LinearPolicy::A16Only},
        Invocation{.tokens = 4, .policy = ops::LinearPolicy::A16Only},
        Invocation{.tokens = 5, .policy = ops::LinearPolicy::A16Only},
        Invocation{.tokens = 8, .policy = ops::LinearPolicy::A16Only},
        Invocation{.tokens = 16, .policy = ops::LinearPolicy::A16Only},
        Invocation{.tokens = 17, .policy = ops::LinearPolicy::A16Only},
        Invocation{.tokens = 24, .policy = ops::LinearPolicy::A16Only},
        Invocation{.tokens = 25, .policy = ops::LinearPolicy::A16Only},
        Invocation{.tokens = 32, .policy = ops::LinearPolicy::A16Only},
        Invocation{.tokens = 33, .policy = ops::LinearPolicy::A16Only},
        Invocation{.tokens = 48, .policy = ops::LinearPolicy::A16Only},
        Invocation{.tokens = 49, .policy = ops::LinearPolicy::A16Only},
        Invocation{.tokens = 64, .policy = ops::LinearPolicy::A16Only},
        Invocation{.tokens = 65, .policy = ops::LinearPolicy::A16Only},
        Invocation{.tokens = 128,
                   .policy = ops::LinearPolicy::A16Only,
                   .graph  = GraphReplay::OriginalTwice},
        Invocation{.tokens = 129, .policy = ops::LinearPolicy::A16Only},
        Invocation{.tokens = 1024, .policy = ops::LinearPolicy::A16Only},
        Invocation{.tokens     = profile.first_a4,
                   .policy     = ops::LinearPolicy::AllowA4,
                   .arithmetic = Arithmetic::A4},
        Invocation{
            .tokens = 17, .policy = ops::LinearPolicy::AllowA4, .arithmetic = Arithmetic::A4},
        Invocation{.tokens = 8, .policy = ops::LinearPolicy::AllowA4, .arithmetic = Arithmetic::A4},
        Invocation{
            .tokens = 16, .policy = ops::LinearPolicy::AllowA4, .arithmetic = Arithmetic::A4},
        Invocation{
            .tokens = 32, .policy = ops::LinearPolicy::AllowA4, .arithmetic = Arithmetic::A4},
        // 32 and 33 straddle the narrow-tile seam on the [5120, 17408] down projection: the last
        // width its M32N32 schedule takes, and the first that returns to M32N64.
        Invocation{
            .tokens = 33, .policy = ops::LinearPolicy::AllowA4, .arithmetic = Arithmetic::A4},
        Invocation{
            .tokens = 64, .policy = ops::LinearPolicy::AllowA4, .arithmetic = Arithmetic::A4},
        Invocation{
            .tokens = 96, .policy = ops::LinearPolicy::AllowA4, .arithmetic = Arithmetic::A4},
        Invocation{.tokens     = 128,
                   .policy     = ops::LinearPolicy::AllowA4,
                   .arithmetic = Arithmetic::A4,
                   .graph      = GraphReplay::OriginalTwice},
        Invocation{
            .tokens = 129, .policy = ops::LinearPolicy::AllowA4, .arithmetic = Arithmetic::A4},
        // 1023, 1024 and 1025 straddle this route's floor. 1024 was the narrowest width it
        // took before; 1025 is the first ragged one it takes now, and its last M tile holds a
        // single real token, which is the emptiest grid this route ever runs.
        Invocation{
            .tokens = 1023, .policy = ops::LinearPolicy::AllowA4, .arithmetic = Arithmetic::A4},
        Invocation{
            .tokens = 1024, .policy = ops::LinearPolicy::AllowA4, .arithmetic = Arithmetic::A4},
        Invocation{
            .tokens = 1025, .policy = ops::LinearPolicy::AllowA4, .arithmetic = Arithmetic::A4},
    };
    return run_cases("NVFP4 linear_add", WeightFormat::NVFP4,
                     InvocationCase{
                         .n               = profile.n,
                         .k               = profile.k,
                         .seed            = profile.seed,
                         .invocations     = invocations,
                         .fixture         = FixturePattern::Hashed,
                         .rows            = Samples::BoundarySeven,
                         .columns         = Samples::BoundarySeven,
                         .scan_all_finite = false,
                     });
}

// Issue #374: a token's bits must not depend on the batch width. The fused residual update used to
// round differently in launches whose width filled every token tile, because the multiply and the add
// were separate statements and only those launches let the compiler contract them into one FMA -- so
// the same token prefilled to different bits depending on where it fell in the chunk.
//
// Two things make this sensitive enough to fail. The activation is uniform -- one nonzero column,
// identical for every token -- so the per-tile activation scale cannot itself depend on how many tokens
// share a tile. And the residual cancels the product: with a generic residual the difference in the
// update disappears in the output's bf16 rounding and the check would pass with or without the fix.
// The product is measured once, at one width, with a zero residual, so every width below is given
// byte-identical inputs and any difference in the result is the update's rounding.
} // namespace

int main() {
    return ninfer::test::linear_common::run_test_main("NVFP4 linear_add", [] {
        return run_profile(Profile{.n = 5120, .k = 6144, .first_a4 = 17, .seed = 811U}) +
               run_profile(Profile{.n = 5120, .k = 17408, .first_a4 = 8, .seed = 821U});
    });
}

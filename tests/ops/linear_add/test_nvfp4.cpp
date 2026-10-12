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
int check_width_invariance(std::int32_t n, std::int32_t k, std::uint32_t seed) {
    constexpr std::int32_t kMaximumTokens = 1025;
    quantized_weight::PatternedWeightOptions options;
    options.weight_scale_divisor = 0.125F;
    options.input_scale_divisor  = 3.5F;
    quantized_weight::PackedWeight host_weight =
        quantized_weight::make_patterned_weight(QType::NVFP4, n, k, seed, options);
    GuardedDeviceBuffer device_weight(host_weight.payload.size());
    device_weight.copy_from_host(host_weight.payload.data(), host_weight.payload.size());
    const Weight weight = host_weight.device_weight(device_weight.data());

    std::vector<std::uint16_t> activation(static_cast<std::size_t>(k) * kMaximumTokens, 0);
    for (std::int32_t token = 0; token < kMaximumTokens; ++token) {
        activation[static_cast<std::size_t>(token) * k] = f32_to_bf16(6.0F);
    }
    GuardedDeviceBuffer device_activation(activation.size() * sizeof(std::uint16_t));
    device_activation.copy_from_host(activation.data(), device_activation.bytes());

    const auto run = [&](std::int32_t tokens, const std::vector<std::uint16_t>& residual_words,
                         std::vector<std::uint16_t>& out) {
        GuardedDeviceBuffer output(static_cast<std::size_t>(n) * tokens * sizeof(std::uint16_t));
        output.copy_from_host(residual_words.data(), output.bytes());
        Tensor x(device_activation.data(), DType::BF16, {k, tokens});
        Tensor residual(output.data(), DType::BF16, {n, tokens});
        const std::size_t capacity = ops::linear_add_workspace_capacity_bytes(
            QType::NVFP4, n, k, ops::LinearPolicy::AllowA4, tokens, tokens);
        MemoryArena workspace(std::max<std::size_t>(capacity, 256));
        ops::linear_add(x, weight, residual, ops::LinearPolicy::AllowA4, workspace, nullptr);
        cuda_check(cudaDeviceSynchronize(), "synchronize width-invariance LinearAdd");
        out.assign(static_cast<std::size_t>(n) * tokens, 0);
        output.copy_to_host(out.data(), out.size() * sizeof(std::uint16_t));
    };

    std::vector<std::uint16_t> product;
    run(kMaximumTokens, std::vector<std::uint16_t>(static_cast<std::size_t>(n) * kMaximumTokens, 0),
        product);

    // Token 0's column of the product, negated in bf16 and replicated across tokens: every token then
    // cancels its own product.
    std::vector<std::uint16_t> cancelling(static_cast<std::size_t>(n) * kMaximumTokens, 0);
    for (std::int32_t row = 0; row < n; ++row) {
        const std::uint16_t negated = static_cast<std::uint16_t>(product[row] ^ 0x8000U);
        for (std::int32_t token = 0; token < kMaximumTokens; ++token) {
            cancelling[static_cast<std::size_t>(token) * n + row] = negated;
        }
    }

    // The sweep must stay inside one route. `AllowA4` permits the A4 route but the dispatch selects
    // A16 below the frontier, so a sweep across it compares two different kernels and reports their
    // legitimate difference as a defect -- measured: with the frontier at 17, T<=16 agreed with T=1 and
    // every T>=17 differed in exactly the same 4,480 of 5,120 words, which is the signature of a route
    // boundary rather than of rounding.
    const std::int32_t first_a4 = k == 6144 ? 17 : 8;
    std::vector<std::int32_t> widths{first_a4, first_a4 + 1, 32, 33, 48, 49, 64, 65, 96, 128, 129};
    for (const std::int32_t extra : {192, 256, 384, 512, 768, 1023, 1024, 1025}) {
        widths.push_back(extra);
    }
    int failures = 0;
    std::vector<std::uint16_t> reference;
    for (const std::int32_t tokens : widths) {
        std::vector<std::uint16_t> actual;
        run(tokens, cancelling, actual);
        const std::string label = "NVFP4 width invariance [" + std::to_string(n) + "," +
                                  std::to_string(k) + "] T=" + std::to_string(tokens);
        if (reference.empty()) {
            reference.assign(actual.begin(), actual.begin() + n);
            continue;
        }
        if (!std::equal(reference.begin(), reference.end(), actual.begin())) {
            std::size_t differing = 0;
            for (std::size_t index = 0; index < reference.size(); ++index) {
                if (reference[index] != actual[index]) { ++differing; }
            }
            std::cerr << label << ": token 0's bits depend on the batch width -- " << differing
                      << " of " << n << " bf16 word(s) differ from T=" << widths.front() << "\n";
            ++failures;
        }
    }
    return failures;
}

} // namespace

int main() {
    return ninfer::test::linear_common::run_test_main("NVFP4 linear_add", [] {
        return run_profile(Profile{.n = 5120, .k = 6144, .first_a4 = 17, .seed = 811U}) +
               run_profile(Profile{.n = 5120, .k = 17408, .first_a4 = 8, .seed = 821U});
    });
}

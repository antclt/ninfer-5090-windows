#include "core/weight.h"
#include "ninfer/ops/linear_add.h"
#include "core/device.h"

#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;

constexpr double kBf16UnitRoundoff = 1.0 / 256.0;
constexpr ReductionCriterion kA16Tolerance{
    kBf16UnitRoundoff,
    kBf16UnitRoundoff,
    2.0 * kBf16UnitRoundoff,
};
constexpr ReductionCriterion kA4Tolerance{0.16, kBf16UnitRoundoff, 0.16};

struct Invocation {
    std::int32_t tokens;
    ops::LinearPolicy policy;
};

std::vector<std::int32_t> sampled_indices(std::int32_t extent) {
    std::vector<std::int32_t> result;
    for (const std::int32_t index :
         {0, 1, extent / 4, extent / 2, (3 * extent) / 4, extent - 2, extent - 1}) {
        if (index >= 0 && index < extent &&
            std::find(result.begin(), result.end(), index) == result.end()) {
            result.push_back(index);
        }
    }
    return result;
}

std::vector<std::uint16_t> make_activation(std::int32_t rows, std::int32_t tokens,
                                           std::uint32_t seed) {
    std::vector<std::uint16_t> result(static_cast<std::size_t>(rows) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t row = 0; row < rows; ++row) {
            std::uint32_t value = seed ^ (static_cast<std::uint32_t>(row) * 0x9e3779b9U) ^
                                  (static_cast<std::uint32_t>(token) * 0x85ebca6bU);
            value ^= value >> 16;
            value *= 0x7feb352dU;
            value ^= value >> 15;
            const float represented =
                static_cast<float>(static_cast<int>(value & 0xffU) - 128) * (1.0F / 256.0F);
            result[static_cast<std::size_t>(token) * rows + row] = f32_to_bf16(represented);
        }
    }
    return result;
}

std::vector<std::uint16_t> make_residual(std::int32_t rows, std::int32_t tokens,
                                         std::uint32_t seed) {
    std::vector<std::uint16_t> result(static_cast<std::size_t>(rows) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t row = 0; row < rows; ++row) {
            const std::uint32_t coordinate = static_cast<std::uint32_t>(row) * 23U +
                                             static_cast<std::uint32_t>(token) * 41U + seed * 7U;
            const float represented =
                static_cast<float>(static_cast<int>(coordinate & 0xffU) - 128) * (1.0F / 128.0F);
            result[static_cast<std::size_t>(token) * rows + row] = f32_to_bf16(represented);
        }
    }
    return result;
}

int verify_preserved(const GuardedDeviceBuffer& device, std::span<const std::uint8_t> expected,
                     std::string_view label) {
    std::vector<std::uint8_t> actual(expected.size());
    device.copy_to_host(actual.data(), actual.size());
    if (std::equal(actual.begin(), actual.end(), expected.begin(), expected.end())) { return 0; }
    std::cerr << label << ": payload was modified\n";
    return 1;
}

int run_shape(std::int32_t n, std::int32_t k, std::uint32_t seed) {
    const std::int32_t first_a4 = k == 6144 ? 17 : 8;
    const std::array invocations{
        Invocation{1, ops::LinearPolicy::A16Only},
        Invocation{4, ops::LinearPolicy::A16Only},
        Invocation{5, ops::LinearPolicy::A16Only},
        Invocation{8, ops::LinearPolicy::A16Only},
        Invocation{16, ops::LinearPolicy::A16Only},
        Invocation{17, ops::LinearPolicy::A16Only},
        Invocation{24, ops::LinearPolicy::A16Only},
        Invocation{25, ops::LinearPolicy::A16Only},
        Invocation{32, ops::LinearPolicy::A16Only},
        Invocation{33, ops::LinearPolicy::A16Only},
        Invocation{48, ops::LinearPolicy::A16Only},
        Invocation{49, ops::LinearPolicy::A16Only},
        Invocation{64, ops::LinearPolicy::A16Only},
        Invocation{65, ops::LinearPolicy::A16Only},
        Invocation{128, ops::LinearPolicy::A16Only},
        Invocation{129, ops::LinearPolicy::A16Only},
        Invocation{1024, ops::LinearPolicy::A16Only},
        Invocation{first_a4, ops::LinearPolicy::AllowA4},
        Invocation{17, ops::LinearPolicy::AllowA4},
        Invocation{8, ops::LinearPolicy::AllowA4},
        Invocation{16, ops::LinearPolicy::AllowA4},
        Invocation{32, ops::LinearPolicy::AllowA4},
        // 32 and 33 straddle the narrow-tile seam on the [5120, 17408] down projection: the last
        // width its M32N32 schedule takes, and the first that returns to M32N64.
        Invocation{33, ops::LinearPolicy::AllowA4},
        Invocation{64, ops::LinearPolicy::AllowA4},
        Invocation{96, ops::LinearPolicy::AllowA4},
        Invocation{128, ops::LinearPolicy::AllowA4},
        Invocation{129, ops::LinearPolicy::AllowA4},
        // 1023, 1024 and 1025 straddle this route's floor. 1024 was the narrowest width it
        // took before; 1025 is the first ragged one it takes now, and its last M tile holds a
        // single real token, which is the emptiest grid this route ever runs.
        Invocation{1023, ops::LinearPolicy::AllowA4},
        Invocation{1024, ops::LinearPolicy::AllowA4},
        Invocation{1025, ops::LinearPolicy::AllowA4},
    };
    // The invocation list is what drives the host buffers, so take the bound from it rather than
    // from a literal that silently caps it.
    const std::int32_t kMaximumTokens =
        std::max_element(
            invocations.begin(), invocations.end(),
            [](const Invocation& a, const Invocation& b) { return a.tokens < b.tokens; })
            ->tokens;
    quantized_weight::PatternedWeightOptions options;
    options.weight_scale_divisor = 0.125F;
    options.input_scale_divisor  = 3.5F;
    quantized_weight::PackedWeight host_weight =
        quantized_weight::make_patterned_weight(QType::NVFP4, n, k, seed, options);
    const std::vector<std::int32_t> rows = sampled_indices(n);
    const std::vector<float> materialized_weight =
        quantized_weight::materialize_rows_fp32(host_weight, rows);
    const std::vector<std::uint16_t> activation = make_activation(k, kMaximumTokens, seed + 1U);
    const std::vector<std::uint16_t> initial_residual = make_residual(n, kMaximumTokens, seed + 2U);

    GuardedDeviceBuffer device_activation(activation.size() * sizeof(std::uint16_t));
    device_activation.copy_from_host(activation.data(), device_activation.bytes());
    GuardedDeviceBuffer device_weight(host_weight.payload.size());
    device_weight.copy_from_host(host_weight.payload.data(), host_weight.payload.size());
    const Weight weight = host_weight.device_weight(device_weight.data());

    int failures = 0;
    for (const Invocation invocation : invocations) {
        const std::size_t output_words = static_cast<std::size_t>(n) * invocation.tokens;
        GuardedDeviceBuffer output(output_words * sizeof(std::uint16_t));
        output.copy_from_host(initial_residual.data(), output.bytes());
        Tensor x(device_activation.data(), DType::BF16, {k, invocation.tokens});
        Tensor residual(output.data(), DType::BF16, {n, invocation.tokens});
        const std::size_t capacity = ops::linear_add_workspace_capacity_bytes(
            QType::NVFP4, n, k, invocation.policy, invocation.tokens, invocation.tokens);
        WorkspaceArena workspace(std::max<std::size_t>(capacity, 256));
        ops::linear_add(x, weight, residual, invocation.policy, workspace, nullptr);
        cuda_check(cudaDeviceSynchronize(), "synchronize NVFP4 linear_add");

        if (invocation.tokens == 128) {
            cudaStream_t stream;
            cudaGraph_t graph;
            cudaGraphExec_t executable;
            CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
            ops::linear_add(x, weight, residual, invocation.policy, workspace, stream);
            CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
            CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
            for (int replay = 0; replay < 2; ++replay) {
                CUDA_CHECK(cudaMemcpyAsync(output.data(), initial_residual.data(), output.bytes(),
                                           cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaGraphLaunch(executable, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }
            CUDA_CHECK(cudaGraphExecDestroy(executable));
            CUDA_CHECK(cudaGraphDestroy(graph));
            CUDA_CHECK(cudaStreamDestroy(stream));
        }

        const bool a4           = invocation.policy == ops::LinearPolicy::AllowA4;
        const std::string label = "NVFP4 linear_add [" + std::to_string(n) + "," +
                                  std::to_string(k) + "] " + (a4 ? "A4" : "A16") +
                                  " T=" + std::to_string(invocation.tokens);
        if (workspace.peak_used() != capacity) {
            std::cerr << label << ": workspace query/execution high-water mismatch\n";
            ++failures;
        }
        failures += output.verify_guards(label);

        std::vector<std::uint16_t> actual_bits(output_words);
        output.copy_to_host(actual_bits.data(), output.bytes());
        const std::vector<std::int32_t> tokens = sampled_indices(invocation.tokens);
        std::vector<double> actual;
        std::vector<double> expected;
        actual.reserve(rows.size() * tokens.size());
        expected.reserve(rows.size() * tokens.size());
        for (std::size_t sampled_row = 0; sampled_row < rows.size(); ++sampled_row) {
            const std::int32_t row = rows[sampled_row];
            const float* weight_row =
                materialized_weight.data() + sampled_row * static_cast<std::size_t>(k);
            for (const std::int32_t token : tokens) {
                double sum = 0.0;
                const std::uint16_t* activation_row =
                    activation.data() + static_cast<std::size_t>(token) * k;
                for (std::int32_t column = 0; column < k; ++column) {
                    sum += static_cast<double>(weight_row[column]) *
                           static_cast<double>(bf16_to_f32(activation_row[column]));
                }
                const std::size_t index = static_cast<std::size_t>(token) * n + row;
                actual.push_back(static_cast<double>(bf16_to_f32(actual_bits[index])));
                expected.push_back(sum + static_cast<double>(bf16_to_f32(initial_residual[index])));
            }
        }
        failures += verify_reduction(label, actual, expected, a4 ? kA4Tolerance : kA16Tolerance);
    }

    failures += device_activation.verify_guards("NVFP4 linear_add activation");
    failures += device_weight.verify_guards("NVFP4 linear_add weight");
    failures += verify_preserved(
        device_activation,
        std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(activation.data()),
                                      activation.size() * sizeof(std::uint16_t)),
        "NVFP4 linear_add activation");
    failures += verify_preserved(device_weight, host_weight.payload, "NVFP4 linear_add weight");
    return failures;
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
        WorkspaceArena workspace(std::max<std::size_t>(capacity, 256));
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
    if (ninfer::test::cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    failures += run_shape(5120, 6144, 811U);
    failures += run_shape(5120, 17408, 821U);
    failures += check_width_invariance(5120, 6144, 833U);
    failures += check_width_invariance(5120, 17408, 839U);
    std::cout << (failures == 0 ? "OK" : "FAIL") << " NVFP4 linear_add\n";
    return failures == 0 ? 0 : 1;
}

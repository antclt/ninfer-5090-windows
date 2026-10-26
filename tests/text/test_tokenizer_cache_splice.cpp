#include "text/tokenizer.h"
#include "text/unicode.h"

#include <nlohmann/json.hpp>

#include <cassert>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int check(bool condition, const char* message) {
    if (condition) return 0;
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

std::string byte_level_symbol(std::uint8_t target) {
    std::uint32_t next = 256;
    for (int value = 0; value <= 255; ++value) {
        const bool visible = (value >= 33 && value <= 126) || (value >= 161 && value <= 172) ||
                             (value >= 174 && value <= 255);
        const std::uint32_t codepoint = visible ? static_cast<std::uint32_t>(value) : next++;
        if (value == target) {
            return ninfer::text::unicode_internal::codepoint_to_utf8(
                static_cast<std::int32_t>(codepoint));
        }
    }
    throw std::logic_error("byte-level test symbol is outside one byte");
}

nlohmann::json make_added_token(int id, std::string content, bool special) {
    return {
        {"id", id},
        {"content", std::move(content)},
        {"single_word", false},
        {"lstrip", false},
        {"rstrip", false},
        {"normalized", false},
        {"special", special}
    };
}

std::string build_tokenizer_json() {
    nlohmann::json vocab = nlohmann::json::object();
    for (int value = 0; value <= 255; ++value) {
        const auto byte = static_cast<std::uint8_t>(value);
        vocab[byte_level_symbol(byte)] = value;
    }
    // Added control tokens for conversation turns
    nlohmann::json added_tokens = nlohmann::json::array({
        make_added_token(256, "<|im_start|>", true),
        make_added_token(257, "<|im_end|>", true),
        make_added_token(258, "\n", false),
        make_added_token(259, "user", false),
        make_added_token(260, "assistant", false),
    });

    nlohmann::json doc = {
        {"model",
         {{"type", "BPE"},
          {"vocab", std::move(vocab)},
          {"merges", nlohmann::json::array()}}},
        {"added_tokens", added_tokens},
        {"normalizer", {{"type", "NFC"}}},
        {"pre_tokenizer",
         nlohmann::json::parse(
             R"json({"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"use_regex":false}]})json")},
        {"decoder", {{"type", "ByteLevel"}}}
    };
    return doc.dump();
}

std::string build_tokenizer_config_json() {
    nlohmann::json decoder = nlohmann::json::object();
    for (const auto& token : {
        make_added_token(256, "<|im_start|>", true),
        make_added_token(257, "<|im_end|>", true),
        make_added_token(258, "\n", false),
        make_added_token(259, "user", false),
        make_added_token(260, "assistant", false),
    }) {
        auto val = token;
        int id = val["id"];
        val.erase("id");
        decoder[std::to_string(id)] = std::move(val);
    }
    nlohmann::json doc = {{"added_tokens_decoder", decoder}};
    return doc.dump();
}

int test_exact_repeat_caching() {
    const std::string tok_json = build_tokenizer_json();
    const std::string cfg_json = build_tokenizer_config_json();
    const ninfer::text::Tokenizer tokenizer({tok_json, cfg_json});

    int failures = 0;
    failures += check(tokenizer.encode_cache_splices() == 0, "initial splices == 0");
    failures += check(tokenizer.encode_cache_fallbacks() == 0, "initial fallbacks == 0");

    const std::string prompt = "<|im_start|>user\nHello world<|im_end|>\n";
    const std::vector<std::size_t> boundaries = {0, prompt.size()};

    // First call: populates cache, not a splice
    const auto res1 = tokenizer.encode_with_boundaries(prompt, boundaries);
    failures += check(!res1.input_ids.empty(), "first encode returned tokens");
    failures += check(tokenizer.encode_cache_splices() == 0, "first encode splices == 0");
    failures += check(tokenizer.encode_cache_fallbacks() == 0, "first encode fallbacks == 0");

    // Second call: exact match should hit cache repeat
    const auto res2 = tokenizer.encode_with_boundaries(prompt, boundaries);
    failures += check(res2.input_ids == res1.input_ids, "repeat token ids match");
    failures += check(res2.boundaries.size() == res1.boundaries.size(), "repeat boundaries count match");
    failures += check(tokenizer.encode_cache_splices() == 1, "repeat incremented splices to 1");
    failures += check(tokenizer.encode_cache_fallbacks() == 0, "repeat fallbacks remains 0");

    // Third call: another exact repeat
    const auto res3 = tokenizer.encode_with_boundaries(prompt, boundaries);
    failures += check(res3.input_ids == res1.input_ids, "3rd repeat token ids match");
    failures += check(tokenizer.encode_cache_splices() == 2, "3rd repeat incremented splices to 2");
    failures += check(tokenizer.encode_cache_fallbacks() == 0, "3rd repeat fallbacks remains 0");

    return failures;
}

int test_incremental_turn_splicing() {
    const std::string tok_json = build_tokenizer_json();
    const std::string cfg_json = build_tokenizer_config_json();
    const ninfer::text::Tokenizer tokenizer({tok_json, cfg_json});

    int failures = 0;

    // Turn 1
    const std::string turn1 = "<|im_start|>user\nHello<|im_end|>\n";
    // We place boundaries: at start, at end of user message, at end of turn
    const std::size_t b0 = 0;
    const std::size_t b1 = 16; // before <|im_end|>
    const std::size_t b2 = turn1.size();
    const std::vector<std::size_t> bounds1 = {b0, b1, b2};

    const auto res1 = tokenizer.encode_with_boundaries(turn1, bounds1);
    failures += check(!res1.input_ids.empty(), "turn1 encoded successfully");
    failures += check(tokenizer.encode_cache_splices() == 0, "turn1 splices == 0");

    // Turn 2: Extends Turn 1 strictly
    const std::string turn2 = turn1 + "<|im_start|>assistant\nHi<|im_end|>\n";
    const std::size_t b3 = turn1.size() + 21;
    const std::size_t b4 = turn2.size();
    const std::vector<std::size_t> bounds2 = {b0, b1, b2, b3, b4};

    const auto res2 = tokenizer.encode_with_boundaries(turn2, bounds2);
    failures += check(!res2.input_ids.empty(), "turn2 encoded successfully");
    // Should have spliced from cache at seam b2
    failures += check(tokenizer.encode_cache_splices() >= 1, "turn2 triggered cache splice");
    failures += check(tokenizer.encode_cache_fallbacks() == 0, "turn2 zero fallbacks");

    // Verify against cold oracle
    const ninfer::text::Tokenizer cold_oracle({tok_json, cfg_json});
    const auto oracle_res = cold_oracle.encode_with_boundaries(turn2, bounds2);
    failures += check(res2.input_ids == oracle_res.input_ids, "spliced ids match cold oracle exactly");
    failures += check(res2.boundaries.size() == oracle_res.boundaries.size(), "spliced boundaries count matches oracle");
    for (std::size_t i = 0; i < res2.boundaries.size(); ++i) {
        failures += check(res2.boundaries[i].exact_frontier == oracle_res.boundaries[i].exact_frontier,
                          "boundary exact frontier matches oracle");
        failures += check(res2.boundaries[i].stable_frontier == oracle_res.boundaries[i].stable_frontier,
                          "boundary stable frontier matches oracle");
    }

    // Turn 3: Extends Turn 2 strictly (tests chaining on previously spliced cache entry)
    const std::string turn3 = turn2 + "<|im_start|>user\nBye<|im_end|>\n";
    const std::size_t b5 = turn2.size() + 14;
    const std::size_t b6 = turn3.size();
    const std::vector<std::size_t> bounds3 = {b0, b1, b2, b3, b4, b5, b6};

    const auto res3 = tokenizer.encode_with_boundaries(turn3, bounds3);
    failures += check(!res3.input_ids.empty(), "turn3 encoded successfully");
    failures += check(tokenizer.encode_cache_splices() >= 2, "turn3 triggered chained cache splice");
    failures += check(tokenizer.encode_cache_fallbacks() == 0, "turn3 zero fallbacks");

    const auto oracle_res3 = cold_oracle.encode_with_boundaries(turn3, bounds3);
    failures += check(res3.input_ids == oracle_res3.input_ids, "turn3 spliced ids match cold oracle exactly");
    failures += check(res3.boundaries.size() == oracle_res3.boundaries.size(), "turn3 boundaries count matches oracle");
    for (std::size_t i = 0; i < res3.boundaries.size(); ++i) {
        failures += check(res3.boundaries[i].exact_frontier == oracle_res3.boundaries[i].exact_frontier,
                          "turn3 boundary exact frontier matches oracle");
        failures += check(res3.boundaries[i].stable_frontier == oracle_res3.boundaries[i].stable_frontier,
                          "turn3 boundary stable frontier matches oracle");
    }

    return failures;
}

} // namespace

int main() {
    try {
        int total_failures = 0;
        std::cout << "Running test_exact_repeat_caching...\n";
        total_failures += test_exact_repeat_caching();

        std::cout << "Running test_incremental_turn_splicing...\n";
        total_failures += test_incremental_turn_splicing();

        if (total_failures == 0) {
            std::cout << "PASS: All tokenizer cache splice tests passed.\n";
            return 0;
        }
        std::cerr << "FAIL: " << total_failures << " test(s) failed.\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "FATAL EXCEPTION: " << e.what() << '\n';
        return 1;
    } catch (...) {
        std::cerr << "FATAL UNKNOWN EXCEPTION\n";
        return 1;
    }
}

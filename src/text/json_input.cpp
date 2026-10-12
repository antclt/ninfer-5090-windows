#include "text/json_input.h"
#include "json_number.h"

#include <algorithm>
#include <utility>

namespace ninfer::text {
namespace {
using Json = nlohmann::ordered_json;

std::string escape(std::string_view key) {
    std::string result;
    for (char c : key) result += c == '~' ? "~0" : c == '/' ? "~1" : std::string(1, c);
    return result;
}

class NumberSax {
public:
    explicit NumberSax(ParsedJsonNumbers& result) : result_(result), dom_(result.value) {}

    bool null() {
        scalar();
        return dom_.null();
    }

    bool boolean(bool value) {
        scalar();
        return dom_.boolean(value);
    }

    bool number_integer(Json::number_integer_t value) {
        scalar();
        return dom_.number_integer(value);
    }

    bool number_unsigned(Json::number_unsigned_t value) {
        scalar();
        return dom_.number_unsigned(value);
    }

    bool number_float(Json::number_float_t value, const Json::string_t& source) {
        clear_value();
        if (!xgrammar::NumberSpellingPreserved(source, value))
            result_.inexact_numbers.push_back(pointer());
        advance();
        return dom_.number_float(value, source);
    }

    bool string(Json::string_t& value) {
        scalar();
        return dom_.string(value);
    }

    bool binary(Json::binary_t& value) {
        scalar();
        return dom_.binary(value);
    }

    bool start_object(std::size_t size) {
        if (!push_frame({})) { return false; }
        return dom_.start_object(size);
    }

    bool key(Json::string_t& value) {
        stack_.back().key = value;
        return dom_.key(value);
    }

    bool end_object() {
        stack_.pop_back();
        advance();
        return dom_.end_object();
    }

    bool start_array(std::size_t size) {
        if (!push_frame({.array = true})) { return false; }
        return dom_.start_array(size);
    }

    bool end_array() {
        stack_.pop_back();
        advance();
        return dom_.end_array();
    }

    bool parse_error(std::size_t position, const std::string& token,
                     const nlohmann::detail::exception& error) {
        return dom_.parse_error(position, token, error);
    }

private:
    struct Frame {
        bool array        = false;
        std::size_t index = 0;
        std::string key;
    };

    std::string pointer() const {
        std::string path;
        for (const auto& frame : stack_)
            path += '/' + (frame.array ? std::to_string(frame.index) : escape(frame.key));
        return path;
    }

    // A request body's nesting is bounded here, at the one place the document is built, because a
    // deeply nested body reaches three recursive consumers: this parser's own schema walk
    // (`find`, below), nlohmann's recursive destructor on the finished document, and the frontend's
    // schema walkers. A client controls the nesting and the body limit is 384 MiB, so an unbounded
    // depth is a remote stack exhaustion -- an uncatchable Windows stack overflow -- from one
    // well-formed request. 64 levels is far past any real tool schema or template argument.
    // Returning false makes it a parse error, which the caller maps to a rejected request.
    static constexpr std::size_t kMaximumDepth = 64;

    bool push_frame(Frame frame) {
        if (stack_.size() >= kMaximumDepth) { return false; }
        clear_value();
        stack_.push_back(std::move(frame));
        return true;
    }

    void advance() {
        if (!stack_.empty() && stack_.back().array) ++stack_.back().index;
    }

    void clear_value() {
        if (result_.inexact_numbers.empty()) return;
        const auto path = pointer();
        std::erase_if(result_.inexact_numbers,
                      [&](const auto& old) { return old == path || old.starts_with(path + '/'); });
    }

    void scalar() {
        clear_value();
        advance();
    }

    ParsedJsonNumbers& result_;
    using InputAdapter =
        decltype(nlohmann::detail::input_adapter(std::declval<std::string_view>()));
    nlohmann::detail::json_sax_dom_parser<Json, InputAdapter> dom_;
    std::vector<Frame> stack_;
};

std::optional<std::string> find(const Json& node, const std::string& path,
                                const std::vector<std::string>& inexact, bool literal = false) {
    if (node.is_number()) {
        if (std::find(inexact.begin(), inexact.end(), path) != inexact.end()) return path;
        return {};
    }
    if (literal) {
        if (node.is_structured())
            for (const auto& [key, value] : node.items())
                if (auto found = find(value, path + '/' + escape(key), inexact, true)) return found;
        return {};
    }
    if (!node.is_object()) return {};
    for (const char* key :
         {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "const", "enum"})
        if (node.contains(key))
            if (auto found = find(node[key], path + '/' + key, inexact, true)) return found;
    for (const char* key :
         {"properties", "$defs", "definitions", "anyOf", "oneOf", "allOf", "prefixItems"}) {
        if (!node.contains(key) || !node[key].is_structured()) continue;
        for (const auto& [index, child] : node[key].items())
            if (auto found = find(child, path + '/' + key + '/' + escape(index), inexact))
                return found;
    }
    for (const char* key : {"items", "additionalItems", "additionalProperties"}) {
        if (!node.contains(key)) continue;
        if (node[key].is_array()) {
            for (std::size_t i = 0; i < node[key].size(); ++i)
                if (auto found =
                        find(node[key][i], path + '/' + key + '/' + std::to_string(i), inexact))
                    return found;
        } else if (auto found = find(node[key], path + '/' + key, inexact))
            return found;
    }
    return {};
}
} // namespace

ParsedJsonNumbers parse_json_numbers(std::string_view source) {
    ParsedJsonNumbers parsed;
    NumberSax sax(parsed);
    // The SAX parser returns false rather than throwing when a callback refuses a value, and ignoring
    // that leaves a TRUNCATED document: the depth bound above then surfaced as "schema must be an
    // object or boolean" deep inside the walk, on a body that had simply been cut off. Throw the same
    // exception type the parser's own error path does, so every caller's existing handling applies --
    // `prepare_json_schema` maps it to InvalidJsonSchema and the serve layer to a rejected request.
    if (!Json::sax_parse(source, &sax)) {
        throw Json::parse_error::create(101, 0, "JSON nests deeper than the 64 levels this engine "
                                                "accepts, or is otherwise malformed",
                                        nullptr);
    }
    return parsed;
}

std::optional<std::string> inexact_schema_number(const ParsedJsonNumbers& parsed,
                                                 const std::string& schema_pointer) {
    if (parsed.inexact_numbers.empty()) return {};
    const Json::json_pointer pointer(schema_pointer);
    if (!parsed.value.contains(pointer)) return {};
    return find(parsed.value.at(pointer), schema_pointer, parsed.inexact_numbers);
}
} // namespace ninfer::text

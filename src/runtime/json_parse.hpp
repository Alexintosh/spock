#pragma once

/// Minimal JSON parser for artifact manifests.
/// Supports only the subset needed: objects, arrays, strings, numbers, booleans, null.
/// No floating-point numbers needed in the manifest - all values we parse are
/// integers, strings, or arrays thereof.

#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace spock::runtime {

struct JsonValue;
using JsonObject = std::map<std::string, JsonValue>;
using JsonArray = std::vector<JsonValue>;

struct JsonValue {
  std::variant<std::nullptr_t, bool, std::int64_t, double, std::string, JsonArray, JsonObject> data;

  bool is_null() const { return std::holds_alternative<std::nullptr_t>(data); }
  bool is_bool() const { return std::holds_alternative<bool>(data); }
  bool is_int() const { return std::holds_alternative<std::int64_t>(data); }
  bool is_number() const { return std::holds_alternative<double>(data) || std::holds_alternative<std::int64_t>(data); }
  bool is_string() const { return std::holds_alternative<std::string>(data); }
  bool is_array() const { return std::holds_alternative<JsonArray>(data); }
  bool is_object() const { return std::holds_alternative<JsonObject>(data); }

  bool as_bool() const { return std::get<bool>(data); }
  std::int64_t as_int() const;
  double as_double() const;
  const std::string& as_string() const { return std::get<std::string>(data); }
  const JsonArray& as_array() const { return std::get<JsonArray>(data); }
  const JsonObject& as_object() const { return std::get<JsonObject>(data); }

  /// Object field access. Returns nullptr for missing keys.
  const JsonValue* get(std::string_view key) const;

  /// Convenience: get string field or empty.
  std::string get_string(std::string_view key, const std::string& default_val = "") const;

  /// Convenience: get int field or default.
  std::int64_t get_int(std::string_view key, std::int64_t default_val = 0) const;
};

/// Parse JSON from a string. Throws on invalid input.
JsonValue parse_json(std::string_view text);

/// Read a file and parse it as JSON.
JsonValue read_json_file(const std::string& path);

}  // namespace spock::runtime

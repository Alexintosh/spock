#include "json_parse.hpp"

#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>

namespace spock::runtime {

namespace {

struct JsonParser {
  std::string_view text;
  std::size_t pos = 0;

  [[noreturn]] void error(const char* msg) {
    throw std::runtime_error(
        std::string("JSON parse error at offset ") + std::to_string(pos) + ": " + msg);
  }

  char peek() {
    if (pos >= text.size()) error("unexpected end of input");
    return text[pos];
  }

  char advance() {
    if (pos >= text.size()) error("unexpected end of input");
    return text[pos++];
  }

  void expect(char c) {
    char got = advance();
    if (got != c) {
      char buf[2] = {got, '\0'};
      error((std::string("expected '") + c + "', got '" + buf + "'").c_str());
    }
  }

  void skip_whitespace() {
    while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\n' ||
                                  text[pos] == '\r' || text[pos] == '\t')) {
      ++pos;
    }
  }

  JsonValue parse_value() {
    skip_whitespace();
    if (pos >= text.size()) error("unexpected end of input");
    char c = peek();
    if (c == '{') return parse_object();
    if (c == '[') return parse_array();
    if (c == '"') return parse_string();
    if (c == 't' || c == 'f') return parse_bool();
    if (c == 'n') return parse_null();
    if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
    error("unexpected character");
  }

  JsonValue parse_object() {
    expect('{');
    JsonObject obj;
    skip_whitespace();
    if (peek() == '}') {
      advance();
      return JsonValue{obj};
    }
    for (;;) {
      skip_whitespace();
      auto key = parse_string().as_string();
      skip_whitespace();
      expect(':');
      skip_whitespace();
      auto val = parse_value();
      obj.emplace(std::move(key), std::move(val));
      skip_whitespace();
      char c = peek();
      if (c == '}') {
        advance();
        break;
      }
      expect(',');
    }
    return JsonValue{obj};
  }

  JsonValue parse_array() {
    expect('[');
    JsonArray arr;
    skip_whitespace();
    if (peek() == ']') {
      advance();
      return JsonValue{arr};
    }
    for (;;) {
      skip_whitespace();
      arr.push_back(parse_value());
      skip_whitespace();
      char c = peek();
      if (c == ']') {
        advance();
        break;
      }
      expect(',');
    }
    return JsonValue{arr};
  }

  JsonValue parse_string() {
    expect('"');
    std::string result;
    for (;;) {
      char c = advance();
      if (c == '"') break;
      if (c == '\\') {
        c = advance();
        switch (c) {
          case '"':  result += '"'; break;
          case '\\': result += '\\'; break;
          case '/':  result += '/'; break;
          case 'n':  result += '\n'; break;
          case 't':  result += '\t'; break;
          case 'r':  result += '\r'; break;
          case 'b':  result += '\b'; break;
          case 'f':  result += '\f'; break;
          default:
            error("unsupported escape sequence");
        }
      } else {
        result += c;
      }
    }
    return JsonValue{std::move(result)};
  }

  JsonValue parse_number() {
    std::size_t start = pos;
    bool is_float = false;

    if (peek() == '-') advance();

    // integer part
    if (peek() == '0') {
      advance();
    } else if (peek() >= '1' && peek() <= '9') {
      advance();
      while (pos < text.size() && peek() >= '0' && peek() <= '9') advance();
    } else {
      error("invalid number");
    }

    // fractional part
    if (pos < text.size() && peek() == '.') {
      is_float = true;
      advance();
      while (pos < text.size() && peek() >= '0' && peek() <= '9') advance();
    }

    // exponent
    if (pos < text.size() && (peek() == 'e' || peek() == 'E')) {
      is_float = true;
      advance();
      if (pos < text.size() && (peek() == '+' || peek() == '-')) advance();
      while (pos < text.size() && peek() >= '0' && peek() <= '9') advance();
    }

    auto num_str = std::string(text.substr(start, pos - start));
    if (is_float) {
      double val = std::stod(num_str);
      return JsonValue{val};
    } else {
      std::int64_t val = std::stoll(num_str);
      return JsonValue{val};
    }
  }

  JsonValue parse_bool() {
    if (text.substr(pos, 4) == "true") {
      pos += 4;
      return JsonValue{true};
    }
    if (text.substr(pos, 5) == "false") {
      pos += 5;
      return JsonValue{false};
    }
    error("invalid boolean");
  }

  JsonValue parse_null() {
    if (text.substr(pos, 4) == "null") {
      pos += 4;
      return JsonValue{nullptr};
    }
    error("invalid null");
  }
};

}  // anonymous namespace

JsonValue parse_json(std::string_view text) {
  JsonParser parser{text};
  auto result = parser.parse_value();
  parser.skip_whitespace();
  if (parser.pos != text.size()) {
    parser.error("trailing data after JSON value");
  }
  return result;
}

JsonValue read_json_file(const std::string& path) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) {
    throw std::runtime_error("cannot open file: " + path);
  }
  std::ostringstream ss;
  ss << ifs.rdbuf();
  return parse_json(ss.str());
}

std::int64_t JsonValue::as_int() const {
  if (std::holds_alternative<std::int64_t>(data)) return std::get<std::int64_t>(data);
  if (std::holds_alternative<double>(data)) return static_cast<std::int64_t>(std::get<double>(data));
  throw std::runtime_error("JsonValue is not a number");
}

double JsonValue::as_double() const {
  if (std::holds_alternative<double>(data)) return std::get<double>(data);
  if (std::holds_alternative<std::int64_t>(data))
    return static_cast<double>(std::get<std::int64_t>(data));
  throw std::runtime_error("JsonValue is not a number");
}

const JsonValue* JsonValue::get(std::string_view key) const {
  if (!is_object()) return nullptr;
  auto& obj = as_object();
  auto it = obj.find(std::string(key));
  if (it == obj.end()) return nullptr;
  return &it->second;
}

std::string JsonValue::get_string(std::string_view key, const std::string& default_val) const {
  auto* v = get(key);
  if (!v || !v->is_string()) return default_val;
  return v->as_string();
}

std::int64_t JsonValue::get_int(std::string_view key, std::int64_t default_val) const {
  auto* v = get(key);
  if (!v || !v->is_number()) return default_val;
  return v->as_int();
}

}  // namespace spock::runtime

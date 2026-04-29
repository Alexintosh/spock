#include "runtime/deltanet_chunk.hpp"
#include "runtime/json_parse.hpp"

#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::vector<float> parse_float_array(const spock::runtime::JsonValue* value,
                                     const char* field_name) {
  if (value == nullptr || !value->is_array()) {
    throw std::runtime_error(std::string("missing array field: ") + field_name);
  }
  std::vector<float> result;
  result.reserve(value->as_array().size());
  for (const auto& entry : value->as_array()) {
    if (!entry.is_number()) {
      throw std::runtime_error(std::string("non-numeric value in field: ") + field_name);
    }
    result.push_back(static_cast<float>(entry.as_double()));
  }
  return result;
}

void print_array(const std::vector<float>& values) {
  std::cout << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) std::cout << ", ";
    std::cout << values[i];
  }
  std::cout << "]";
}

}  // namespace

int main(int argc, char** argv) {
  std::string input_path;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--input" && i + 1 < argc) {
      input_path = argv[++i];
    } else if (arg == "--help") {
      std::cout << "usage: spock-deltanet-chunk --input FILE\n";
      return 0;
    }
  }

  if (input_path.empty()) {
    std::cerr << "spock-deltanet-chunk: --input is required\n";
    return 2;
  }

  try {
    const auto root = spock::runtime::read_json_file(input_path);
    if (!root.is_object()) {
      throw std::runtime_error("root JSON value must be an object");
    }

    spock::runtime::DeltaNetChunkConfig config;
    config.num_heads = static_cast<std::size_t>(root.get_int("num_heads"));
    config.sequence_length = static_cast<std::size_t>(root.get_int("sequence_length"));
    config.key_dim = static_cast<std::size_t>(root.get_int("key_dim"));
    config.value_dim = static_cast<std::size_t>(root.get_int("value_dim"));
    config.chunk_size = static_cast<std::size_t>(root.get_int("chunk_size", 64));
    if (const auto* use_norm = root.get("use_qk_l2norm")) {
      if (!use_norm->is_bool()) {
        throw std::runtime_error("use_qk_l2norm must be a boolean");
      }
      config.use_qk_l2norm = use_norm->as_bool();
    }

    spock::runtime::DeltaNetChunkInputs inputs;
    inputs.query = parse_float_array(root.get("query"), "query");
    inputs.key = parse_float_array(root.get("key"), "key");
    inputs.value = parse_float_array(root.get("value"), "value");
    inputs.g = parse_float_array(root.get("g"), "g");
    inputs.beta = parse_float_array(root.get("beta"), "beta");
    if (const auto* initial_state = root.get("initial_state")) {
      if (!initial_state->is_null()) {
        inputs.initial_state = parse_float_array(initial_state, "initial_state");
      }
    }

    const auto outputs = spock::runtime::run_deltanet_chunk_rule(config, inputs);

    std::cout << std::setprecision(9);
    std::cout << "{\n";
    std::cout << "  \"core_attn_out\": ";
    print_array(outputs.core_attn_out);
    std::cout << ",\n";
    std::cout << "  \"final_state\": ";
    print_array(outputs.final_state);
    std::cout << "\n}\n";
  } catch (const std::exception& e) {
    std::cerr << "spock-deltanet-chunk: " << e.what() << "\n";
    return 1;
  }

  return 0;
}

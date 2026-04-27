#include "model/qwen35_config.hpp"

namespace spock::model {

const std::array<LayerKind, Qwen35Config::layer_count>& Qwen35Config::layer_schedule() {
  static constexpr std::array<LayerKind, Qwen35Config::layer_count> schedule = [] {
    std::array<LayerKind, Qwen35Config::layer_count> result{};
    for (std::size_t i = 0; i < result.size(); ++i) {
      result[i] = (i % 4 == 3) ? LayerKind::FullAttention : LayerKind::DeltaNet;
    }
    return result;
  }();
  return schedule;
}

std::string_view to_string(const LayerKind kind) {
  switch (kind) {
    case LayerKind::DeltaNet:
      return "deltanet";
    case LayerKind::FullAttention:
      return "attention";
  }
  return "unknown";
}

}  // namespace spock::model

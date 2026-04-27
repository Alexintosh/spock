#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace spock::model {

enum class LayerKind : std::uint8_t {
  DeltaNet = 0,
  FullAttention = 1,
};

struct Qwen35Config {
  static constexpr std::string_view model_id = "Qwen/Qwen3.5-0.8B";
  static constexpr std::size_t layer_count = 24;
  static constexpr std::size_t hidden_size = 1024;
  static constexpr std::size_t intermediate_size = 3584;
  static constexpr std::size_t attention_q_heads = 8;
  static constexpr std::size_t attention_kv_heads = 2;
  static constexpr std::size_t attention_head_dim = 256;
  static constexpr std::size_t deltanet_heads = 16;
  static constexpr std::size_t deltanet_key_dim = 128;
  static constexpr std::size_t deltanet_value_dim = 128;
  static constexpr std::size_t deltanet_conv_kernel = 4;
  static constexpr std::size_t max_sequence_length_v1 = 2048;

  static const std::array<LayerKind, layer_count>& layer_schedule();
};

std::string_view to_string(LayerKind kind);

}  // namespace spock::model

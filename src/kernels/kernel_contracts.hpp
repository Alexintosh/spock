#pragma once

#include <cstdint>

namespace spock::kernels {

struct DecodeDescriptorBindings {
  static constexpr std::uint32_t weights = 0;
  static constexpr std::uint32_t activations = 1;
  static constexpr std::uint32_t delta_state = 2;
  static constexpr std::uint32_t kv_cache = 3;
  static constexpr std::uint32_t scratch = 4;
};

struct BarrierProbeBindings {
  static constexpr std::uint32_t control = 0;
  static constexpr std::uint32_t trace = 1;
};

}  // namespace spock::kernels

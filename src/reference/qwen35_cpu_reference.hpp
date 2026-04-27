#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace spock::reference {

struct DecodeSettings {
  std::uint32_t max_new_tokens = 16;
  bool greedy = true;
  std::uint32_t bos_token_id = 0;
  std::uint32_t eos_token_id = 0;
};

class Qwen35CpuReference {
 public:
  std::vector<std::uint32_t> decode(std::span<const std::uint32_t> prompt,
                                    const DecodeSettings& settings) const;
};

float rms_norm_scale(std::span<const float> values, float epsilon);

}  // namespace spock::reference

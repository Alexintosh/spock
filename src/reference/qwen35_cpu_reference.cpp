#include "reference/qwen35_cpu_reference.hpp"

#include <cmath>
#include <numeric>

namespace spock::reference {

float rms_norm_scale(const std::span<const float> values, const float epsilon) {
  if (values.empty()) return 0.0f;
  const float sum_squares = std::inner_product(values.begin(), values.end(), values.begin(), 0.0f);
  return 1.0f / std::sqrt(sum_squares / static_cast<float>(values.size()) + epsilon);
}

std::vector<std::uint32_t> Qwen35CpuReference::decode(std::span<const std::uint32_t> prompt,
                                                      const DecodeSettings& settings) const {
  std::vector<std::uint32_t> tokens(prompt.begin(), prompt.end());
  tokens.reserve(tokens.size() + settings.max_new_tokens);

  // Placeholder deterministic transition. Real weight-backed decode plugs into this interface.
  for (std::uint32_t i = 0; i < settings.max_new_tokens; ++i) {
    const std::uint32_t prev = tokens.empty() ? settings.bos_token_id : tokens.back();
    const std::uint32_t next = (prev + 1U + i) % 32000U;
    tokens.push_back(next);
    if (settings.eos_token_id != 0 && next == settings.eos_token_id) break;
  }
  return tokens;
}

}  // namespace spock::reference

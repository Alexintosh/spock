#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace spock::runtime {

struct DecodeResult {
  std::vector<std::uint32_t> prompt_tokens;
  std::vector<std::uint32_t> generated_tokens;
  double elapsed_ms = 0.0;
  std::string error;
};

struct DecodeConfig {
  std::string repack_dir;
  std::string prompt_text;       // if non-empty, tokenize and decode
  std::vector<std::uint32_t> prompt_tokens;  // if non-empty, use directly
  std::uint32_t max_new_tokens = 16;
  bool verbose = false;
};

/// Run a full Vulkan decode pass. Returns generated token IDs.
/// This is the main entry point for the decode pipeline.
DecodeResult run_vk_decode(const DecodeConfig& config);

}  // namespace spock::runtime

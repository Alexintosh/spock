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
  bool debug_dump = false;  // dump hidden state after each layer
  bool diagnose_handoff = false;  // after prefill, dump chunk vs recurrent state comparison
  bool diagnose_decode_drift = false;  // compare free-run vs rebuilt state at target decode step
  int dump_step_hiddens = -1;  // if >= 0, dump per-layer hiddens at this decode step (stderr JSON)
  int dump_step_components = -1;  // if >= 0, dump component-level intermediates (input, post-mixer, post-mlp, final-norm) at this decode step (stderr JSON)
  bool experiment_attn_o_proj_f32_residual = false;  // diagnostic: keep attention o_proj fp32 until residual add
  bool experiment_mlp_down_f32_residual = false;  // diagnostic: keep MLP down projection fp32 until residual add
};

/// Run a full Vulkan decode pass. Returns generated token IDs.
/// This is the main entry point for the decode pipeline.
DecodeResult run_vk_decode(const DecodeConfig& config);

}  // namespace spock::runtime

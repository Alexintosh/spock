#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace spock::runtime {

struct DecodeResult {
  std::vector<std::uint32_t> prompt_tokens;
  std::vector<std::uint32_t> generated_tokens;
  double elapsed_ms = 0.0;
  double prefill_ms = 0.0;       // wall-clock prefill time
  double decode_ms = 0.0;        // wall-clock decode-only time (sum of per-token)
  double gpu_decode_us = 0.0;    // GPU-side decode time (sum of per-token, from timestamp queries)
  std::uint32_t decode_submit_count = 0;  // main decode-loop final/chunk submits
  std::uint32_t chunked_decode_submit_count = 0;  // subset submitted as multi-step chunks
  std::vector<double> per_token_ms;   // host wall-clock per decode token
  std::vector<double> per_token_gpu_us; // GPU-side per decode token
  std::map<std::string, double> gpu_region_us; // GPU-side per-region summed microseconds (block timestamps)
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

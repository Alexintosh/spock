#include "runtime/vk_decode.hpp"
#include "runtime/vk_session.hpp"

namespace spock::runtime {

DecodeResult run_vk_decode(const DecodeConfig& config) {
#if !SPOCK_HAS_VULKAN || defined(SPOCK_VULKAN_STUB)
  DecodeResult result;
  result.error = "Vulkan not available (built with stub)";
  return result;
#else
  DecodeSession session(config.repack_dir, config.verbose);
  return session.decode(
      config.prompt_tokens,
      config.max_new_tokens,
      config.verbose,
      config.debug_dump,
      config.diagnose_handoff,
      config.diagnose_decode_drift,
      config.dump_step_hiddens,
      config.dump_step_components,
      config.experiment_attn_o_proj_f32_residual,
      config.experiment_mlp_down_f32_residual,
      config.dump_dn_recurrent_state_pre_layer,
      config.dump_dn_recurrent_state_pre_file);
#endif
}

}  // namespace spock::runtime

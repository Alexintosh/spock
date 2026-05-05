#pragma once
#include "runtime/vk_decode.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#if SPOCK_HAS_VULKAN && !defined(SPOCK_VULKAN_STUB)
#include "runtime/vk_device.hpp"
#include "runtime/weight_loader.hpp"
#include "runtime/deltanet_chunk.hpp"
#include "model/qwen35_config.hpp"
#endif

#if SPOCK_HAS_VULKAN && !defined(SPOCK_VULKAN_STUB)

namespace spock::runtime {

/// Persistent decode session. Owns device, pipelines, buffers, and weights
/// across multiple decode calls. Construct once, call decode() repeatedly.
class DecodeSession {
 public:
  /// Initialize a session with the given weight repack directory.
  explicit DecodeSession(const std::string& repack_dir, bool verbose = false);

  DecodeSession(const DecodeSession&) = delete;
  DecodeSession& operator=(const DecodeSession&) = delete;
  ~DecodeSession();

  /// Run a full decode pass: prefill prompt_tokens, generate max_new_tokens.
  DecodeResult decode(const std::vector<uint32_t>& prompt_tokens,
                      uint32_t max_new_tokens,
                      bool verbose = false,
                      bool debug_dump = false,
                      bool diagnose_handoff = false,
                      bool diagnose_decode_drift = false,
                      int dump_step_hiddens = -1,
                      int dump_step_components = -1,
                      bool experiment_attn_o_proj_f32_residual = false,
                      bool experiment_mlp_down_f32_residual = false);

  /// Clear all recurrent state (KV cache, DeltaNet state) for a fresh prompt.
  void reset();

  const VulkanCapabilities& device_capabilities() const { return dev_.capabilities(); }

 private:
  struct Pipelines {
    VkPipelineLayout pipeline_layout_3;
    VkPipelineLayout pipeline_layout_2;
    VkPipelineLayout pipeline_layout_32;
    VkPipelineLayout pipeline_layout_4;
    VkPipelineLayout pipeline_layout_cp;
    VkPipelineLayout pipeline_layout_6_32;
    VkPipelineLayout pipeline_layout_8_32;
    VkDescriptorSetLayout ds_layout_3;
    VkDescriptorSetLayout ds_layout_2;
    VkDescriptorSetLayout ds_layout_4;
    VkDescriptorSetLayout ds_layout_7;
    VkDescriptorSetLayout ds_layout_6;
    VkDescriptorSetLayout ds_layout_8;

    VkPipeline embedding;
    VkPipeline embedding_from_buffer;
    VkPipeline rmsnorm;
    VkPipeline matvec;
    VkPipeline matvec_f32_out;
    VkPipeline argmax;
    VkPipeline silu_gate;
    VkPipeline residual_add;
    VkPipeline residual_add_mixed;
    VkPipeline rope_apply;
    VkPipeline attention_decode;
    VkPipeline kv_cache_store;
    VkPipeline sigmoid_gate;
    VkPipeline rms_norm_per_head;
    VkPipeline split_q_gate;
    VkPipeline deltanet_recurrent;
    VkPipeline conv1d_step;
    VkPipeline deltanet_norm_gate;
    VkPipeline l2_norm_per_head;
    VkPipeline deltanet_compute_g_beta;
    VkPipeline deltanet_conv_l2_qk;
    VkPipeline deltanet_recurrent_gbeta;
    VkPipeline deltanet_recurrent_gbeta_norm_gate;


    VkPipeline deltanet_chunk_prefill;
    VkPipeline deltanet_chunk_prefill_tiled;
    VkPipeline deltanet_prefill_collect;
    VkPipeline deltanet_chunk_last_to_fp16;


    VkShaderModule embedding_module;
    VkShaderModule embedding_from_buffer_module;
    VkShaderModule rmsnorm_module;
    VkShaderModule matvec_module;
    VkShaderModule matvec_f32_out_module;
    VkShaderModule argmax_module;
    VkShaderModule silu_gate_module;
    VkShaderModule residual_add_module;
    VkShaderModule residual_add_mixed_module;
    VkShaderModule rope_apply_module;
    VkShaderModule attention_decode_module;
    VkShaderModule kv_cache_store_module;
    VkShaderModule sigmoid_gate_module;
    VkShaderModule rms_norm_per_head_module;
    VkShaderModule split_q_gate_module;
    VkShaderModule deltanet_recurrent_module;
    VkShaderModule conv1d_step_module;
    VkShaderModule deltanet_norm_gate_module;
    VkShaderModule l2_norm_per_head_module;
    VkShaderModule deltanet_compute_g_beta_module;
    VkShaderModule deltanet_chunk_prefill_module;
    VkShaderModule deltanet_chunk_prefill_tiled_module;
    VkShaderModule deltanet_prefill_collect_module;
    VkShaderModule deltanet_chunk_last_to_fp16_module;
    VkShaderModule deltanet_conv_l2_qk_module;
    VkShaderModule deltanet_recurrent_gbeta_module;
    VkShaderModule deltanet_recurrent_gbeta_norm_gate_module;

  };

  struct Buffers {
    VulkanDevice::Buffer act_a;
    VulkanDevice::Buffer act_b;
    VulkanDevice::Buffer act_c;
    VulkanDevice::Buffer logits;
    VulkanDevice::Buffer argmax_result;
    VulkanDevice::Buffer mlp_gate;
    VulkanDevice::Buffer mlp_up;
    VulkanDevice::Buffer mlp_silu;

    VulkanDevice::Buffer q_proj;
    VulkanDevice::Buffer q;
    VulkanDevice::Buffer gate;
    VulkanDevice::Buffer k;
    VulkanDevice::Buffer v;
    VulkanDevice::Buffer attn_out;
    VulkanDevice::Buffer gated_attn;
    VulkanDevice::Buffer attn_proj_f32;
    VulkanDevice::Buffer kv_cache;
    VulkanDevice::Buffer rope_freq;

    VulkanDevice::Buffer dn_qkv;
    VulkanDevice::Buffer dn_z;
    VulkanDevice::Buffer dn_a;
    VulkanDevice::Buffer dn_b;
    VulkanDevice::Buffer dn_q;
    VulkanDevice::Buffer dn_kv_out;
    VulkanDevice::Buffer dn_state;
    VulkanDevice::Buffer dn_conv_state;
    VulkanDevice::Buffer dn_a_log_bias;

    VulkanDevice::Buffer weights;
    VulkanDevice::Buffer final_norm;

    // Diagnostic prefill collection buffers (allocated per decode call when
    // SPOCK_GPU_COLLECT_PREFILL_COMPARE=1)
    VulkanDevice::Buffer dn_collect_q;
    VulkanDevice::Buffer dn_collect_k;
    VulkanDevice::Buffer dn_collect_v;
    VulkanDevice::Buffer dn_collect_g;
    VulkanDevice::Buffer dn_collect_beta;
    bool collect_bufs_allocated_ = false;

    // Persistent GPU prefill collection buffers for GPU→GPU chunk prefill
    // (allocated when SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1)
    // Layout: [NUM_DN_LAYERS][DN_HEADS][prompt_len][dim] fp32 per tensor
    VulkanDevice::Buffer dn_persist_q;
    VulkanDevice::Buffer dn_persist_k;
    VulkanDevice::Buffer dn_persist_v;
    VulkanDevice::Buffer dn_persist_g;
    VulkanDevice::Buffer dn_persist_beta;
    bool persist_bufs_allocated_ = false;

    size_t act_bytes;
    size_t act_c_bytes;
    size_t kv_cache_layer_bytes;
    size_t dn_state_per_layer;
    size_t dn_conv_per_layer;

  };

  struct DescriptorSets {
    VkDescriptorSet embedding;
    VkDescriptorSet embedding_from_buffer;

    VkDescriptorSet input_norm;
    VkDescriptorSet residual1;
    VkDescriptorSet post_norm;
    VkDescriptorSet gate;
    VkDescriptorSet up;
    VkDescriptorSet silu_gate;
    VkDescriptorSet down;
    VkDescriptorSet down_f32;
    VkDescriptorSet residual2;
    VkDescriptorSet mlp_residual_mixed;
    VkDescriptorSet final_norm;
    VkDescriptorSet lm_head;
    VkDescriptorSet argmax;

    VkDescriptorSet q_proj;
    VkDescriptorSet k_proj;
    VkDescriptorSet v_proj;
    VkDescriptorSet split_q_gate;
    VkDescriptorSet q_norm;
    VkDescriptorSet k_norm;
    VkDescriptorSet rope;
    VkDescriptorSet rope_k;
    VkDescriptorSet kv_store;
    VkDescriptorSet attn;
    VkDescriptorSet sigmoid_gate;
    VkDescriptorSet o_proj;
    VkDescriptorSet o_proj_f32;
    VkDescriptorSet attn_residual_mixed;

    VkDescriptorSet dn_qkv_proj;
    VkDescriptorSet dn_z_proj;
    VkDescriptorSet dn_a_proj;
    VkDescriptorSet dn_b_proj;
    VkDescriptorSet dn_conv;
    VkDescriptorSet dn_split_q;
    VkDescriptorSet dn_split_kv;
    VkDescriptorSet dn_l2_q;
    VkDescriptorSet dn_l2_k;
    VkDescriptorSet dn_recurrent;
    VkDescriptorSet dn_norm_gate;
    VkDescriptorSet dn_out_proj;
    VkDescriptorSet dn_compute_g_beta;
    VkDescriptorSet dn_recurrent_gbeta;
    VkDescriptorSet dn_recurrent_gbeta_norm_gate;
    VkDescriptorSet dn_chunk_prefill;
    VkDescriptorSet dn_prefill_collect;

    VkDescriptorSet dn_chunk_last_to_fp16;

  };

  /// Per-layer stable descriptor sets allocated when SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS
  /// env var is set.  Eliminates per-layer descriptor mutation in decode(), enabling
  /// future single-command-buffer recording per token.
  /// dn_l2_q/dn_l2_k, dn_compute_g_beta, dn_recurrent, dn_norm_gate, and dn_out_proj
  /// are covered here.
  struct PerLayerDescriptorSets {
    // Common MLP/norm — all model layers
    std::vector<VkDescriptorSet> input_norm;          // ds_layout_3
    std::vector<VkDescriptorSet> residual1;           // ds_layout_3
    std::vector<VkDescriptorSet> post_norm;           // ds_layout_3
    std::vector<VkDescriptorSet> gate;                // ds_layout_3
    std::vector<VkDescriptorSet> up;                  // ds_layout_3
    std::vector<VkDescriptorSet> down;                // ds_layout_3
    std::vector<VkDescriptorSet> down_f32;            // ds_layout_3
    std::vector<VkDescriptorSet> residual2;           // ds_layout_3
    std::vector<VkDescriptorSet> mlp_residual_mixed;  // ds_layout_3

    // Attention-specific — allocated for all layers; only bound for attention layers
    std::vector<VkDescriptorSet> q_proj;              // ds_layout_3
    std::vector<VkDescriptorSet> k_proj;              // ds_layout_3
    std::vector<VkDescriptorSet> v_proj;              // ds_layout_3
    std::vector<VkDescriptorSet> q_norm;              // ds_layout_3
    std::vector<VkDescriptorSet> k_norm;              // ds_layout_3
    std::vector<VkDescriptorSet> kv_store;            // ds_layout_3
    std::vector<VkDescriptorSet> attn;                // ds_layout_3
    std::vector<VkDescriptorSet> o_proj;              // ds_layout_3
    std::vector<VkDescriptorSet> o_proj_f32;          // ds_layout_3
    std::vector<VkDescriptorSet> attn_residual_mixed; // ds_layout_3

    // DeltaNet-specific — allocated for all layers; only bound for DeltaNet layers
    std::vector<VkDescriptorSet> dn_qkv_proj;         // ds_layout_3
    std::vector<VkDescriptorSet> dn_z_proj;           // ds_layout_3
    std::vector<VkDescriptorSet> dn_a_proj;           // ds_layout_3
    std::vector<VkDescriptorSet> dn_b_proj;           // ds_layout_3
    std::vector<VkDescriptorSet> dn_conv;             // ds_layout_3
    std::vector<VkDescriptorSet> dn_l2_q;             // ds_layout_3
    std::vector<VkDescriptorSet> dn_l2_k;             // ds_layout_3
    std::vector<VkDescriptorSet> dn_recurrent;        // ds_layout_3
    std::vector<VkDescriptorSet> dn_norm_gate;        // ds_layout_3
    std::vector<VkDescriptorSet> dn_out_proj;         // ds_layout_3
    std::vector<VkDescriptorSet> dn_compute_g_beta;  // ds_layout_4
    std::vector<VkDescriptorSet> dn_recurrent_gbeta;  // ds_layout_6
    std::vector<VkDescriptorSet> dn_recurrent_gbeta_norm_gate;  // ds_layout_8
  };
  VulkanDevice dev_;
  bool verbose_;

  std::unique_ptr<Pipelines> pipes_;
  std::unique_ptr<Buffers> bufs_;
  std::unique_ptr<DescriptorSets> dsets_;
  std::unique_ptr<PerLayerDescriptorSets> per_layer_sets_;
  bool per_layer_sets_enabled_ = false;

  WeightArtifact artifact_;

  std::vector<std::vector<float>> cached_a_log_;
  std::vector<std::vector<float>> cached_dt_bias_;

  // --- Prefill chunk collection state ---
  // Per-DeltaNet-layer vectors that accumulate Q/K/V/g/beta across prompt tokens.
  // Filled during prefill, consumed by run_chunk_prefill().
  struct PrefillChunkState {
    // [num_heads * seq_len * k_dim]
    std::vector<float> query;
    // [num_heads * seq_len * k_dim]
    std::vector<float> key;
    // [num_heads * seq_len * v_dim]
    std::vector<float> value;
    // [num_heads * seq_len]
    std::vector<float> g;
    // [num_heads * seq_len]
    std::vector<float> beta;
    // Per-token gate output (from norm+gate): [seq_len * v_dim]
    std::vector<float> gated_output;
    // Per-token z values: [seq_len * v_dim]
    std::vector<float> z;
    // Per-token out_proj output: [seq_len * hidden_size]
    std::vector<float> out_proj_output;
    // Per-token intermediate (post-input-norm hidden): [seq_len * hidden_size]
    std::vector<float> post_input_norm;
  };
  std::vector<PrefillChunkState> prefill_chunks_;  // one per DeltaNet layer
  uint32_t prefill_token_count_ = 0;

  // --- Chunk correction state ---
  // Per-DeltaNet-layer: chunk rule core_attn_out for the last prefill token
  // Shape: [DN_HEADS * DN_V_DIM] fp32
  std::vector<std::vector<float>> chunk_core_attn_out_last_;
  // GPU buffer: per-layer pre-norm hidden state snapshots for last token
  // Shape: [LAYERS * HIDDEN] fp16
  VulkanDevice::Buffer prefill_snapshots_;
  // GPU buffer: per-layer fp16 last-token core_attn_out for GPU handoff path.
  // Shape: [NUM_DN_LAYERS * DN_VAL_TOTAL] fp16.
  // Populated by gpu_chunk_prefill_from_gpu_collect in no-compare tiled fast path;
  // consumed by correct_last_token_hidden() to skip CPU conversion/upload.
  VulkanDevice::Buffer dn_chunk_attn_out_;

  /// After prefill loop, run chunk rule per DeltaNet layer and upload final state.
  void run_chunk_prefill();

  /// GPU path for run_chunk_prefill: dispatches deltanet_chunk_prefill.comp
  /// with serial per-head dispatches, or deltanet_chunk_prefill_tiled.comp
  /// with one tiled dispatch when requested.
  /// Q/K/V/g/beta are still CPU-collected; this gate moves only the chunk-rule
  /// computation to GPU, not full prefill collection/offload.
  void gpu_chunk_prefill(uint32_t dn_idx, PrefillChunkState& chunk, uint32_t seq_len, bool tiled);

  /// GPU path for run_chunk_prefill using GPU-collected (device-local persistent)
  /// Q/K/V/g/beta buffers. Binds the per-layer segment directly to the selected
  /// chunk-prefill shader without CPU upload.
  void gpu_chunk_prefill_from_gpu_collect(uint32_t dn_idx, PrefillChunkState& chunk, uint32_t seq_len, bool tiled);

  /// After run_chunk_prefill, reprocess all layers for the last token using chunk-corrected
  /// DeltaNet outputs to produce the correct hidden state for the first decode step.
  /// If out_layer_hidden is non-null, fills it with [LAYERS * HIDDEN] fp16 per-layer hiddens.
  void correct_last_token_hidden(const std::vector<uint32_t>& tokens, uint32_t prompt_len,
                                  std::vector<uint16_t>* out_layer_hidden = nullptr);

  /// Diagnostic: after prefill+chunk_prefill, dump handoff state for analysis.
  void diagnose_handoff(const std::vector<uint32_t>& tokens, uint32_t prompt_len);


  /// Diagnostic: compare free-run decode state at a target step against a freshly
  /// reprefilled state to localize recurrent decode drift.
  void diagnose_decode_drift(const std::vector<uint32_t>& full_prefix_tokens,
                              uint32_t target_decode_step,
                              const std::vector<uint16_t>& free_hidden,
                              const std::vector<float>& free_logits,
                              const std::vector<std::vector<float>>& free_dn_state,
                              const std::vector<std::vector<uint16_t>>& free_kv_cache = {},
                              const std::vector<uint16_t>& free_layer_hidden = {});
  /// Layer-major prefill: process all tokens through each layer sequentially.
  /// For DeltaNet layers, uses chunk rule instead of recurrent update.
  void layer_major_prefill(const std::vector<uint32_t>& tokens,
                           uint32_t prompt_len, bool verbose);

  static constexpr uint32_t HIDDEN = model::Qwen35Config::hidden_size;
  static constexpr uint32_t INTER = model::Qwen35Config::intermediate_size;
  static constexpr uint32_t VOCAB = 248320;
  static constexpr uint32_t LAYERS = model::Qwen35Config::layer_count;
  static constexpr uint32_t Q_HEADS = model::Qwen35Config::attention_q_heads;
  static constexpr uint32_t KV_HEADS = model::Qwen35Config::attention_kv_heads;
  static constexpr uint32_t HEAD_DIM = model::Qwen35Config::attention_head_dim;
  static constexpr uint32_t MAX_SEQ = model::Qwen35Config::max_sequence_length_v1;
  static constexpr uint32_t ROTARY_DIM = 64;
  static constexpr uint32_t KV_GROUP = Q_HEADS / KV_HEADS;
  static constexpr uint32_t NUM_ATTN_LAYERS = 6;
  static constexpr uint32_t DN_HEADS = model::Qwen35Config::deltanet_heads;
  static constexpr uint32_t DN_K_DIM = model::Qwen35Config::deltanet_key_dim;
  static constexpr uint32_t DN_V_DIM = model::Qwen35Config::deltanet_value_dim;
  static constexpr uint32_t DN_CONV_KS = model::Qwen35Config::deltanet_conv_kernel;
  static constexpr uint32_t DN_KEY_TOTAL = DN_HEADS * DN_K_DIM;
  static constexpr uint32_t DN_VAL_TOTAL = DN_HEADS * DN_V_DIM;
  static constexpr uint32_t DN_CONV_DIM = DN_KEY_TOTAL * 2 + DN_VAL_TOTAL;
  static constexpr uint32_t NUM_DN_LAYERS = 18;
  // Per-layer flag: gpu_chunk_prefill_from_gpu_collect produced a GPU-resident
  // fp16 last-token core_attn_out in dn_chunk_attn_out_ for this dn_idx.
  // Set in the no-compare triple-gated fast path; consumed by
  // correct_last_token_hidden() to skip CPU conversion/upload.
  std::vector<bool> gpu_chunk_handoff_ready_;
};

}  // namespace spock::runtime

#else  // Vulkan stub

namespace spock::runtime {

class DecodeSession {
 public:
  explicit DecodeSession(const std::string&, bool = false) {}
  DecodeResult decode(const std::vector<uint32_t>&, uint32_t, bool = false, bool = false, bool = false, bool = false, int = -1, int = -1) {
    DecodeResult r;
    r.error = "Vulkan not available (built with stub)";
    return r;
  }
  void reset() {}
};

}  // namespace spock::runtime

#endif  // SPOCK_HAS_VULKAN

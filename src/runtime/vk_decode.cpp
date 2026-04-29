#include "runtime/vk_decode.hpp"
#include "runtime/vk_device.hpp"
#include "runtime/weight_loader.hpp"
#include "model/qwen35_config.hpp"

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <algorithm>

#if SPOCK_HAS_VULKAN && !defined(SPOCK_VULKAN_STUB)
#include <vulkan/vulkan.h>
#endif

namespace spock::runtime {

namespace {

/// Read SPIR-V shader bytes from a file.
std::vector<uint32_t> read_spirv(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open shader: " + path);
  auto size = f.tellg();
  f.seekg(0);
  std::vector<uint32_t> words(size / 4);
  f.read(reinterpret_cast<char*>(words.data()), size);
  return words;
}

float half_to_float(uint16_t h) {
  uint32_t sign = (h >> 15) & 1;
  uint32_t exponent = (h >> 10) & 0x1f;
  uint32_t mantissa = h & 0x3ff;

  uint32_t f;
  if (exponent == 0) {
    if (mantissa == 0) {
      f = sign << 31;
    } else {
      exponent = 127 - 15 + 1;
      while ((mantissa & 0x400) == 0) {
        mantissa <<= 1;
        exponent--;
      }
      mantissa &= 0x3ff;
      f = (sign << 31) | (exponent << 23) | (mantissa << 13);
    }
  } else if (exponent == 31) {
    f = (sign << 31) | (0xff << 23) | (mantissa << 13);
  } else {
    f = (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13);
  }

  float result;
  memcpy(&result, &f, 4);
  return result;
}

void upload_raw(VulkanDevice& dev, const VulkanDevice::Buffer& buf,
                const void* data, size_t size) {
  dev.upload_to_device(buf, data, static_cast<VkDeviceSize>(size));
}

uint32_t float_to_bits(float f) {
  uint32_t bits;
  memcpy(&bits, &f, 4);
  return bits;
}

}  // namespace

DecodeResult run_vk_decode(const DecodeConfig& config) {
  DecodeResult result;

#if !SPOCK_HAS_VULKAN || defined(SPOCK_VULKAN_STUB)
  result.error = "Vulkan not available (built with stub)";
  return result;
#else

  auto barrier = [&](VkCommandBuffer cmd_buf, VkBuffer buf, VkDeviceSize size) {
    VkBufferMemoryBarrier bmb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    bmb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bmb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.buffer = buf;
    bmb.offset = 0;
    bmb.size = size;
    vkCmdPipelineBarrier(cmd_buf,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 1, &bmb, 0, nullptr);
  };

  // --- 1. Initialize Vulkan device ---
  VulkanDevice dev;
  try {
    dev.initialize();
  } catch (const std::exception& e) {
    result.error = std::string("Vulkan init failed: ") + e.what();
    return result;
  }

  const auto& caps = dev.capabilities();
  if (config.verbose) {
    std::cerr << "Vulkan device: " << caps.device_name << "\n";
    std::cerr << "Subgroup size: " << caps.subgroup_size << "\n";
    std::cerr << "Max shared memory: " << caps.max_shared_memory_bytes << "\n";
  }

  // --- 2. Load weight artifact ---
  WeightArtifact artifact;
  try {
    artifact = WeightArtifact::load(config.repack_dir);
  } catch (const std::exception& e) {
    result.error = std::string("Weight load failed: ") + e.what();
    return result;
  }

  if (config.verbose) {
    std::cerr << "Loaded " << artifact.tensor_count() << " tensors ("
              << (artifact.total_bytes() / (1024*1024)) << " MiB)\n";
  }

  // --- 3. Load shaders ---
  auto shader_dir = std::string(SHADER_DIR);
  auto embedding_shader = read_spirv(shader_dir + "/embedding_lookup.comp.spv");
  auto rmsnorm_shader = read_spirv(shader_dir + "/rms_norm.comp.spv");
  auto matvec_shader = read_spirv(shader_dir + "/matvec.comp.spv");
  auto argmax_shader = read_spirv(shader_dir + "/argmax.comp.spv");
  auto silu_gate_shader = read_spirv(shader_dir + "/silu_gate.comp.spv");
  auto rope_apply_shader = read_spirv(shader_dir + "/rope_apply.comp.spv");
  auto attention_decode_shader = read_spirv(shader_dir + "/attention_decode.comp.spv");
  auto kv_cache_store_shader = read_spirv(shader_dir + "/kv_cache_store.comp.spv");
  auto sigmoid_gate_shader = read_spirv(shader_dir + "/sigmoid_gate.comp.spv");
  auto rms_norm_per_head_shader = read_spirv(shader_dir + "/rms_norm_per_head.comp.spv");
  auto split_q_gate_shader = read_spirv(shader_dir + "/split_q_gate.comp.spv");
  auto residual_add_shader = read_spirv(shader_dir + "/residual_add.comp.spv");
  auto deltanet_recurrent_shader = read_spirv(shader_dir + "/deltanet_recurrent.comp.spv");
  auto conv1d_step_shader = read_spirv(shader_dir + "/conv1d_step.comp.spv");
  auto deltanet_norm_gate_shader = read_spirv(shader_dir + "/deltanet_norm_gate.comp.spv");
  auto l2_norm_per_head_shader = read_spirv(shader_dir + "/l2_norm_per_head.comp.spv");

  // --- 4. Pipeline infrastructure ---
  std::vector<VkDescriptorSetLayoutBinding> bindings_3 = {
      {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
  };
  std::vector<VkDescriptorSetLayoutBinding> bindings_2 = {
      {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
  };

  auto ds_layout_3 = dev.create_descriptor_set_layout(bindings_3);
  auto ds_layout_2 = dev.create_descriptor_set_layout(bindings_2);
  auto pipeline_layout_3 = dev.create_pipeline_layout(ds_layout_3, 8);
  auto pipeline_layout_2 = dev.create_pipeline_layout(ds_layout_2, 8);
  auto pipeline_layout_32 = dev.create_pipeline_layout(ds_layout_3, 32);

  auto embedding_module = dev.create_shader_module(embedding_shader);
  auto rmsnorm_module = dev.create_shader_module(rmsnorm_shader);
  auto matvec_module = dev.create_shader_module(matvec_shader);
  auto argmax_module = dev.create_shader_module(argmax_shader);
  auto silu_gate_module = dev.create_shader_module(silu_gate_shader);
  auto residual_add_module = dev.create_shader_module(residual_add_shader);
  auto rope_apply_module = dev.create_shader_module(rope_apply_shader);
  auto attention_decode_module = dev.create_shader_module(attention_decode_shader);
  auto kv_cache_store_module = dev.create_shader_module(kv_cache_store_shader);
  auto sigmoid_gate_module_new = dev.create_shader_module(sigmoid_gate_shader);
  auto rms_norm_per_head_module = dev.create_shader_module(rms_norm_per_head_shader);
  auto split_q_gate_module = dev.create_shader_module(split_q_gate_shader);
  auto deltanet_recurrent_module = dev.create_shader_module(deltanet_recurrent_shader);
  auto conv1d_step_module = dev.create_shader_module(conv1d_step_shader);
  auto deltanet_norm_gate_module = dev.create_shader_module(deltanet_norm_gate_shader);
  auto l2_norm_per_head_module = dev.create_shader_module(l2_norm_per_head_shader);

  auto embedding_pipeline = dev.create_compute_pipeline(embedding_module, pipeline_layout_2);
  auto rmsnorm_pipeline = dev.create_compute_pipeline(rmsnorm_module, pipeline_layout_3);
  auto matvec_pipeline = dev.create_compute_pipeline(matvec_module, pipeline_layout_3);
  auto argmax_pipeline = dev.create_compute_pipeline(argmax_module, pipeline_layout_2);
  auto silu_gate_pipeline = dev.create_compute_pipeline(silu_gate_module, pipeline_layout_3);
  auto residual_add_pipeline = dev.create_compute_pipeline(residual_add_module, pipeline_layout_3);

  auto rope_apply_pipeline = dev.create_compute_pipeline(rope_apply_module, pipeline_layout_32);
  auto attention_decode_pipeline = dev.create_compute_pipeline(attention_decode_module, pipeline_layout_32);
  auto kv_cache_store_pipeline = dev.create_compute_pipeline(kv_cache_store_module, pipeline_layout_32);
  auto sigmoid_gate_pipeline_new = dev.create_compute_pipeline(sigmoid_gate_module_new, pipeline_layout_3);
  auto rms_norm_per_head_pipeline = dev.create_compute_pipeline(rms_norm_per_head_module, pipeline_layout_32);
  auto split_q_gate_pipeline = dev.create_compute_pipeline(split_q_gate_module, pipeline_layout_3);
  auto deltanet_recurrent_pipeline = dev.create_compute_pipeline(deltanet_recurrent_module, pipeline_layout_32);
  auto conv1d_step_pipeline = dev.create_compute_pipeline(conv1d_step_module, pipeline_layout_3);
  auto deltanet_norm_gate_pipeline = dev.create_compute_pipeline(deltanet_norm_gate_module, pipeline_layout_32);
  auto l2_norm_per_head_pipeline = dev.create_compute_pipeline(l2_norm_per_head_module, pipeline_layout_3);

  // --- 5. Allocate GPU buffers ---
  constexpr uint32_t HIDDEN = model::Qwen35Config::hidden_size;
  constexpr uint32_t INTER = model::Qwen35Config::intermediate_size;
  constexpr uint32_t VOCAB = 248320;
  constexpr uint32_t LAYERS = model::Qwen35Config::layer_count;
  constexpr uint32_t Q_HEADS = model::Qwen35Config::attention_q_heads;
  constexpr uint32_t KV_HEADS = model::Qwen35Config::attention_kv_heads;
  constexpr uint32_t HEAD_DIM = model::Qwen35Config::attention_head_dim;
  constexpr uint32_t MAX_SEQ = model::Qwen35Config::max_sequence_length_v1;
  constexpr uint32_t ROTARY_DIM = 64;  // HEAD_DIM * partial_rotary_factor(0.25)
  constexpr uint32_t KV_GROUP = Q_HEADS / KV_HEADS;  // 4
  constexpr uint32_t NUM_ATTN_LAYERS = 6;
  constexpr uint32_t DN_HEADS = model::Qwen35Config::deltanet_heads;       // 16
  constexpr uint32_t DN_K_DIM = model::Qwen35Config::deltanet_key_dim;    // 128
  constexpr uint32_t DN_V_DIM = model::Qwen35Config::deltanet_value_dim;  // 128
  constexpr uint32_t DN_CONV_KS = model::Qwen35Config::deltanet_conv_kernel;  // 4
  constexpr uint32_t DN_KEY_TOTAL = DN_HEADS * DN_K_DIM;                  // 2048
  constexpr uint32_t DN_VAL_TOTAL = DN_HEADS * DN_V_DIM;                  // 2048
  constexpr uint32_t DN_CONV_DIM = DN_KEY_TOTAL * 2 + DN_VAL_TOTAL;       // 6144
  constexpr uint32_t NUM_DN_LAYERS = 18;
  const float RMS_EPS = 1e-6f;
  const float ATTN_SCALE = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
  static const float DN_Q_SCALE = 1.0f / std::sqrt(static_cast<float>(DN_K_DIM));
  size_t act_bytes = HIDDEN * 2;
  auto act_a = dev.create_device_local_buffer(act_bytes);
  auto act_b = dev.create_device_local_buffer(act_bytes);
  auto act_c = dev.create_device_local_buffer(act_bytes);

  auto logits = dev.create_device_local_buffer(VOCAB * 2);
  auto argmax_result = dev.create_device_local_buffer(4);

  size_t inter_bytes = INTER * 2;
  auto mlp_gate_buf = dev.create_device_local_buffer(inter_bytes);
  auto mlp_up_buf = dev.create_device_local_buffer(inter_bytes);
  auto mlp_silu_buf = dev.create_device_local_buffer(inter_bytes);

  // Attention buffers
  size_t q_dim = Q_HEADS * HEAD_DIM;           // 2048
  size_t kv_dim = KV_HEADS * HEAD_DIM;          // 512
  size_t q_proj_out = q_dim * 2;                // 4096 (query + gate interleaved)
  size_t attn_bytes = q_proj_out * 2;            // q_proj output
  auto q_proj_buf = dev.create_device_local_buffer(q_proj_out * 2);   // [4096] fp16
  auto q_buf = dev.create_device_local_buffer(q_dim * 2);             // [2048] fp16
  auto gate_buf = dev.create_device_local_buffer(q_dim * 2);          // [2048] fp16
  auto k_buf = dev.create_device_local_buffer(kv_dim * 2);            // [512] fp16
  auto v_buf = dev.create_device_local_buffer(kv_dim * 2);            // [512] fp16
  auto attn_out_buf = dev.create_device_local_buffer(q_dim * 2);      // [2048] fp16
  auto gated_attn_buf = dev.create_device_local_buffer(q_dim * 2);    // [2048] fp16

  // KV cache: per attention layer [max_seq * 2 * kv_heads * head_dim] fp16
  size_t kv_cache_layer_bytes = MAX_SEQ * 2 * KV_HEADS * HEAD_DIM * 2;  // 4 MiB
  auto kv_cache_buf = dev.create_device_local_buffer(kv_cache_layer_bytes * NUM_ATTN_LAYERS);
  // Zero-init KV cache (device-local is undefined)
  {
    std::vector<uint8_t> zeros(kv_cache_layer_bytes * NUM_ATTN_LAYERS, 0);
    upload_raw(dev, kv_cache_buf, zeros.data(), zeros.size());
  }

  // Precompute RoPE frequencies (cos/sin for single position, updated per step)
  size_t rope_freq_bytes = ROTARY_DIM * 2;  // 32 cos + 32 sin packed as [cos0,sin0,cos1,sin1,...]
  auto rope_freq_buf = dev.create_device_local_buffer(ROTARY_DIM * 4);  // float32

  // DeltaNet buffers
  size_t dn_kv_bytes = DN_CONV_DIM * 2;  // QKV+conv buf [6144] fp16 (key*2 + val)
  auto dn_qkv_buf = dev.create_device_local_buffer(dn_kv_bytes);
  auto dn_z_buf = dev.create_device_local_buffer(DN_VAL_TOTAL * 2);    // Z gate [2048] fp16
  auto dn_a_buf = dev.create_device_local_buffer(DN_HEADS * 2);        // a [16] fp16
  auto dn_b_buf = dev.create_device_local_buffer(DN_HEADS * 2);        // b [16] fp16
  auto dn_q_buf = dev.create_device_local_buffer(DN_KEY_TOTAL * 2);    // Q [2048] fp16 (after split, L2-normed)
  auto dn_kv_out_buf = dev.create_device_local_buffer(dn_kv_bytes);    // K+V+output [4096] fp16

  // DeltaNet recurrent state: [num_heads * k_dim * v_dim] fp32 per layer + g/beta
  // State: 16 * 128 * 128 * 4 = 1 MiB per layer
  // g + beta: 16 + 16 = 32 floats = 128 bytes
  size_t dn_state_per_layer = DN_HEADS * DN_K_DIM * DN_V_DIM * 4 + DN_HEADS * 2 * 4;  // ~1 MiB + 128 B
  auto dn_state_buf = dev.create_device_local_buffer(dn_state_per_layer * NUM_DN_LAYERS);
  // Zero-init recurrent state (device-local is undefined)
  {
    std::vector<uint8_t> zeros(dn_state_per_layer * NUM_DN_LAYERS, 0);
    upload_raw(dev, dn_state_buf, zeros.data(), zeros.size());
  }

  // DeltaNet conv state: [conv_dim * kernel_size] fp16 per layer = 6144 * 4 * 2 = 49 KiB
  size_t dn_conv_per_layer = DN_CONV_DIM * DN_CONV_KS * 2;  // 49152 bytes
  auto dn_conv_state_buf = dev.create_device_local_buffer(dn_conv_per_layer * NUM_DN_LAYERS);
  // Zero-init conv state (device-local is undefined)
  {
    std::vector<uint8_t> zeros(dn_conv_per_layer * NUM_DN_LAYERS, 0);
    upload_raw(dev, dn_conv_state_buf, zeros.data(), zeros.size());
  }
  // --- 6. Upload weights ---
  auto total_size = artifact.total_bytes();
  auto weights_buf = dev.create_device_local_buffer(total_size);
  {
    std::ifstream wf(artifact.weights_file_path(), std::ios::binary);
    if (!wf) {
      result.error = "cannot open weights file: " + artifact.weights_file_path();
      return result;
    }
    std::vector<char> weights_data(total_size);
    wf.read(weights_data.data(), total_size);
    upload_raw(dev, weights_buf, weights_data.data(), total_size);
  }

  auto& final_norm_info = artifact.final_norm();
  auto final_norm_raw = read_tensor_bytes(artifact, final_norm_info);
  auto final_norm_buf = dev.create_device_local_buffer(final_norm_info.nbytes);
  upload_raw(dev, final_norm_buf, final_norm_raw.data(), final_norm_info.nbytes);

  // --- 7. Allocate descriptor sets ---
  // One descriptor set per dispatch to avoid aliasing.
  // Vulkan descriptor sets are read by the GPU at execution time,
  // not recording time. Reusing a descriptor set within one submit
  // causes all dispatches to see the last-written state.
  auto embedding_ds = dev.allocate_descriptor_set(ds_layout_2);

  auto input_norm_ds = dev.allocate_descriptor_set(ds_layout_3);
  auto residual1_ds = dev.allocate_descriptor_set(ds_layout_3);
  auto post_norm_ds = dev.allocate_descriptor_set(ds_layout_3);
  auto gate_ds = dev.allocate_descriptor_set(ds_layout_3);
  auto up_ds = dev.allocate_descriptor_set(ds_layout_3);
  auto silu_gate_ds = dev.allocate_descriptor_set(ds_layout_3);
  auto down_ds = dev.allocate_descriptor_set(ds_layout_3);
  auto residual2_ds = dev.allocate_descriptor_set(ds_layout_3);

  auto final_norm_ds = dev.allocate_descriptor_set(ds_layout_3);
  auto lm_head_ds = dev.allocate_descriptor_set(ds_layout_3);
  auto argmax_ds = dev.allocate_descriptor_set(ds_layout_2);

  // Attention descriptor sets (one per dispatch)
  auto q_proj_ds = dev.allocate_descriptor_set(ds_layout_3);
  auto k_proj_ds = dev.allocate_descriptor_set(ds_layout_3);
  auto v_proj_ds = dev.allocate_descriptor_set(ds_layout_3);
  auto split_q_gate_ds = dev.allocate_descriptor_set(ds_layout_3);
  auto q_norm_ds = dev.allocate_descriptor_set(ds_layout_3);
  auto k_norm_ds = dev.allocate_descriptor_set(ds_layout_3);
  auto rope_ds = dev.allocate_descriptor_set(ds_layout_3);
  auto rope_k_ds = dev.allocate_descriptor_set(ds_layout_3);
  auto kv_store_ds = dev.allocate_descriptor_set(ds_layout_3);
  auto attn_ds = dev.allocate_descriptor_set(ds_layout_3);
  auto sigmoid_gate_ds = dev.allocate_descriptor_set(ds_layout_3);
  auto o_proj_ds = dev.allocate_descriptor_set(ds_layout_3);

  // DeltaNet descriptor sets
  auto dn_qkv_proj_ds = dev.allocate_descriptor_set(ds_layout_3);    // matvec: weights × act_b → dn_qkv_buf
  auto dn_z_proj_ds = dev.allocate_descriptor_set(ds_layout_3);     // matvec: weights × act_b → dn_z_buf
  auto dn_a_proj_ds = dev.allocate_descriptor_set(ds_layout_3);     // matvec: weights × act_b → dn_a_buf
  auto dn_b_proj_ds = dev.allocate_descriptor_set(ds_layout_3);     // matvec: weights × act_b → dn_b_buf
  auto dn_conv_ds = dev.allocate_descriptor_set(ds_layout_3);       // conv1d: qkv + state + weights
  auto dn_split_q_ds = dev.allocate_descriptor_set(ds_layout_3);    // copy Q from qkv
  auto dn_split_kv_ds = dev.allocate_descriptor_set(ds_layout_3);   // copy K,V from qkv
  auto dn_l2_q_ds = dev.allocate_descriptor_set(ds_layout_3);      // L2 norm Q
  auto dn_l2_k_ds = dev.allocate_descriptor_set(ds_layout_3);      // L2 norm K
  auto dn_recurrent_ds = dev.allocate_descriptor_set(ds_layout_3); // recurrent update
  auto dn_norm_gate_ds = dev.allocate_descriptor_set(ds_layout_3);  // RMSNorm+SiLU gate
  auto dn_out_proj_ds = dev.allocate_descriptor_set(ds_layout_3);   // matvec: weights × output → act_b

  // Pre-configure descriptor sets that don't change between layers.
  dev.update_descriptor_set(embedding_ds, 0, weights_buf,
      artifact.token_embedding().offset, artifact.token_embedding().nbytes);
  dev.update_descriptor_set(embedding_ds, 1, act_a);

  dev.update_descriptor_set(silu_gate_ds, 0, mlp_gate_buf);
  dev.update_descriptor_set(silu_gate_ds, 1, mlp_up_buf);
  dev.update_descriptor_set(silu_gate_ds, 2, mlp_silu_buf);

  dev.update_descriptor_set(final_norm_ds, 0, act_a);
  dev.update_descriptor_set(final_norm_ds, 1, final_norm_buf);
  dev.update_descriptor_set(final_norm_ds, 2, act_b);

  dev.update_descriptor_set(lm_head_ds, 0, weights_buf,
      artifact.token_embedding().offset, artifact.token_embedding().nbytes);
  dev.update_descriptor_set(lm_head_ds, 1, act_b);
  dev.update_descriptor_set(lm_head_ds, 2, logits);

  dev.update_descriptor_set(argmax_ds, 0, logits);
  dev.update_descriptor_set(argmax_ds, 1, argmax_result);

  // Pre-configure static attention descriptor sets
  dev.update_descriptor_set(split_q_gate_ds, 0, q_proj_buf);
  dev.update_descriptor_set(split_q_gate_ds, 1, q_buf);
  dev.update_descriptor_set(split_q_gate_ds, 2, gate_buf);

  dev.update_descriptor_set(sigmoid_gate_ds, 0, attn_out_buf);
  dev.update_descriptor_set(sigmoid_gate_ds, 1, gate_buf);
  dev.update_descriptor_set(sigmoid_gate_ds, 2, gated_attn_buf);
  // --- 8. Decode loop ---
  auto tokens = config.prompt_tokens.empty()
      ? std::vector<uint32_t>{1, 2, 3}
      : config.prompt_tokens;
  result.prompt_tokens = tokens;

  auto t0 = std::chrono::high_resolution_clock::now();

  // Cache a_log and dt_bias for all DeltaNet layers (constant across steps)
  const auto& layer_sched = model::Qwen35Config::layer_schedule();
  std::vector<std::vector<float>> cached_a_log(NUM_DN_LAYERS);
  std::vector<std::vector<float>> cached_dt_bias(NUM_DN_LAYERS);
  {
    uint32_t dn_ci = 0;
    for (uint32_t i = 0; i < LAYERS; ++i) {
      if (layer_sched[i] == model::LayerKind::FullAttention) continue;
      auto ai = artifact.find_by_role("layer." + std::to_string(i) + ".delta_a_log");
      auto di = artifact.find_by_role("layer." + std::to_string(i) + ".delta_dt_bias");
      if (!ai || !di) { ++dn_ci; continue; }
      auto a_raw = read_tensor_bytes(artifact, *ai);
      auto d_raw = read_tensor_bytes(artifact, *di);
      cached_a_log[dn_ci].resize(DN_HEADS);
      cached_dt_bias[dn_ci].resize(DN_HEADS);
      memcpy(cached_a_log[dn_ci].data(), a_raw.data(), DN_HEADS * 4);
      const uint16_t* dt_f16 = reinterpret_cast<const uint16_t*>(d_raw.data());
      for (uint32_t h = 0; h < DN_HEADS; ++h) cached_dt_bias[dn_ci][h] = half_to_float(dt_f16[h]);
      ++dn_ci;
    }
  }

  // Total steps: (prompt_len - 1) prefill + max_new_tokens decode
  uint32_t prompt_len = static_cast<uint32_t>(tokens.size());
  uint32_t total_steps = (prompt_len > 1 ? prompt_len - 1 : 0) + config.max_new_tokens;

  for (uint32_t step = 0; step < total_steps; ++step) {
    bool is_prefill = (step + 1 < prompt_len);
    uint32_t current_token = is_prefill ? tokens[step] : tokens.back();

    // --- Embedding lookup ---
    {
      auto cmd = dev.allocate_command_buffer();
      dev.begin_command_buffer(cmd);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, embedding_pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          pipeline_layout_2, 0, 1, &embedding_ds, 0, nullptr);
      uint32_t push_token = current_token;
      vkCmdPushConstants(cmd, pipeline_layout_2, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &push_token);
      vkCmdDispatch(cmd, 1, 1, 1);
      dev.end_command_buffer(cmd);
      dev.submit_and_wait(cmd);
    }

    // --- Per-layer processing ---
    const auto& schedule = model::Qwen35Config::layer_schedule();
    uint32_t seq_pos = step;  // absolute position through prefill then decode

    // Compute RoPE frequencies for this position (CPU side)
    {
      constexpr float theta = 10000000.0f;
      constexpr uint32_t rotary_pairs = ROTARY_DIM / 2;  // 32
      std::vector<float> rope_freq(ROTARY_DIM);  // cos/sin interleaved
      for (uint32_t i = 0; i < rotary_pairs; ++i) {
        float inv_freq = 1.0f / std::pow(theta, 2.0f * static_cast<float>(i) / static_cast<float>(ROTARY_DIM));
        float angle = static_cast<float>(seq_pos) * inv_freq;
        rope_freq[i * 2]     = std::cos(angle);
        rope_freq[i * 2 + 1] = std::sin(angle);
      }
      auto rope_upload_buf = dev.create_host_visible_buffer(ROTARY_DIM * 4);
      dev.upload_to_host_visible(rope_upload_buf, rope_freq.data(), ROTARY_DIM * 4);
      {
        auto cmd = dev.allocate_command_buffer();
        dev.begin_command_buffer(cmd);
        VkBufferCopy copy_region{0, 0, ROTARY_DIM * 4};
        vkCmdCopyBuffer(cmd, rope_upload_buf.buffer, rope_freq_buf.buffer, 1, &copy_region);
        dev.end_command_buffer(cmd);
        dev.submit_and_wait(cmd);
      }
      dev.destroy_buffer(rope_upload_buf);
    }

    // Map attention layer index: layer 3→0, 7→1, 11→2, 15→3, 19→4, 23→5
    auto attn_layer_idx = [](uint32_t layer) -> uint32_t {
      return (layer + 1) / 4 - 1;  // works for layers 3,7,11,15,19,23
    };

    for (uint32_t layer = 0; layer < LAYERS; ++layer) {
      auto input_norm_w = artifact.find_by_role(
          "layer." + std::to_string(layer) + ".input_norm");
      auto post_norm_w = artifact.find_by_role(
          "layer." + std::to_string(layer) + ".post_norm");
      auto gate_w = artifact.find_by_role(
          "layer." + std::to_string(layer) + ".mlp_gate");
      auto up_w = artifact.find_by_role(
          "layer." + std::to_string(layer) + ".mlp_up");
      auto down_w = artifact.find_by_role(
          "layer." + std::to_string(layer) + ".mlp_down");

      bool is_attn = (schedule[layer] == model::LayerKind::FullAttention);
      uint32_t attn_idx = is_attn ? attn_layer_idx(layer) : 0;

      // Attention weight lookups
      decltype(artifact.find_by_role("")) attn_q_w, attn_k_w, attn_v_w, attn_o_w;
      decltype(artifact.find_by_role("")) attn_q_norm_w, attn_k_norm_w;
      if (is_attn) {
        attn_q_w = artifact.find_by_role("layer." + std::to_string(layer) + ".attn_q");
        attn_k_w = artifact.find_by_role("layer." + std::to_string(layer) + ".attn_k");
        attn_v_w = artifact.find_by_role("layer." + std::to_string(layer) + ".attn_v");
        attn_o_w = artifact.find_by_role("layer." + std::to_string(layer) + ".attn_o");
        attn_q_norm_w = artifact.find_by_role("layer." + std::to_string(layer) + ".attn_q_norm");
        attn_k_norm_w = artifact.find_by_role("layer." + std::to_string(layer) + ".attn_k_norm");
      }

      // DeltaNet weight lookups
      decltype(artifact.find_by_role("")) dn_qkv_w, dn_z_w, dn_a_w, dn_b_w;
      decltype(artifact.find_by_role("")) dn_out_w, dn_conv_w, dt_bias_w, a_log_w, dn_norm_w;
      if (!is_attn) {
        dn_qkv_w = artifact.find_by_role("layer." + std::to_string(layer) + ".delta_in_proj_qkv");
        dn_z_w = artifact.find_by_role("layer." + std::to_string(layer) + ".delta_in_proj_z");
        dn_a_w = artifact.find_by_role("layer." + std::to_string(layer) + ".delta_in_proj_a");
        dn_b_w = artifact.find_by_role("layer." + std::to_string(layer) + ".delta_in_proj_b");
        dn_out_w = artifact.find_by_role("layer." + std::to_string(layer) + ".delta_out_proj");
        dn_conv_w = artifact.find_by_role("layer." + std::to_string(layer) + ".delta_conv");
        dt_bias_w = artifact.find_by_role("layer." + std::to_string(layer) + ".delta_dt_bias");
        a_log_w = artifact.find_by_role("layer." + std::to_string(layer) + ".delta_a_log");
        dn_norm_w = artifact.find_by_role("layer." + std::to_string(layer) + ".delta_norm");
      }

      // Update per-layer descriptor sets with layer-specific weight offsets.
      dev.update_descriptor_set(input_norm_ds, 0, act_a);
      dev.update_descriptor_set(input_norm_ds, 1, weights_buf,
          input_norm_w->offset, input_norm_w->nbytes);
      dev.update_descriptor_set(input_norm_ds, 2, act_b);

      dev.update_descriptor_set(residual1_ds, 0, act_a);
      dev.update_descriptor_set(residual1_ds, 1, act_b);
      dev.update_descriptor_set(residual1_ds, 2, act_c);

      dev.update_descriptor_set(post_norm_ds, 0, act_c);
      dev.update_descriptor_set(post_norm_ds, 1, weights_buf,
          post_norm_w->offset, post_norm_w->nbytes);
      dev.update_descriptor_set(post_norm_ds, 2, act_a);

      dev.update_descriptor_set(gate_ds, 0, weights_buf,
          gate_w->offset, gate_w->nbytes);
      dev.update_descriptor_set(gate_ds, 1, act_a);
      dev.update_descriptor_set(gate_ds, 2, mlp_gate_buf);

      dev.update_descriptor_set(up_ds, 0, weights_buf,
          up_w->offset, up_w->nbytes);
      dev.update_descriptor_set(up_ds, 1, act_a);
      dev.update_descriptor_set(up_ds, 2, mlp_up_buf);

      dev.update_descriptor_set(down_ds, 0, weights_buf,
          down_w->offset, down_w->nbytes);
      dev.update_descriptor_set(down_ds, 1, mlp_silu_buf);
      dev.update_descriptor_set(down_ds, 2, act_b);

      dev.update_descriptor_set(residual2_ds, 0, act_c);
      dev.update_descriptor_set(residual2_ds, 1, act_b);
      dev.update_descriptor_set(residual2_ds, 2, act_a);

      // Attention descriptor set updates (layer-specific)
      if (is_attn) {
        // q_proj: weights × act_b → q_proj_buf
        dev.update_descriptor_set(q_proj_ds, 0, weights_buf, attn_q_w->offset, attn_q_w->nbytes);
        dev.update_descriptor_set(q_proj_ds, 1, act_b);
        dev.update_descriptor_set(q_proj_ds, 2, q_proj_buf);

        // k_proj: weights × act_b → k_buf
        dev.update_descriptor_set(k_proj_ds, 0, weights_buf, attn_k_w->offset, attn_k_w->nbytes);
        dev.update_descriptor_set(k_proj_ds, 1, act_b);
        dev.update_descriptor_set(k_proj_ds, 2, k_buf);

        // v_proj: weights × act_b → v_buf
        dev.update_descriptor_set(v_proj_ds, 0, weights_buf, attn_v_w->offset, attn_v_w->nbytes);
        dev.update_descriptor_set(v_proj_ds, 1, act_b);
        dev.update_descriptor_set(v_proj_ds, 2, v_buf);

        // q_norm: per-head RMSNorm q_buf with q_norm weights → q_buf
        dev.update_descriptor_set(q_norm_ds, 0, q_buf);
        dev.update_descriptor_set(q_norm_ds, 1, weights_buf, attn_q_norm_w->offset, attn_q_norm_w->nbytes);
        dev.update_descriptor_set(q_norm_ds, 2, q_buf);  // in-place

        // k_norm: per-head RMSNorm k_buf with k_norm weights → k_buf
        dev.update_descriptor_set(k_norm_ds, 0, k_buf);
        dev.update_descriptor_set(k_norm_ds, 1, weights_buf, attn_k_norm_w->offset, attn_k_norm_w->nbytes);
        dev.update_descriptor_set(k_norm_ds, 2, k_buf);  // in-place

        // RoPE: q_buf + k_buf + freq → q_buf (needs separate input/output to avoid race)
        // RoPE: apply to Q and K separately
        dev.update_descriptor_set(rope_ds, 0, q_buf);
        dev.update_descriptor_set(rope_ds, 1, rope_freq_buf);
        dev.update_descriptor_set(rope_ds, 2, q_buf);

        dev.update_descriptor_set(rope_k_ds, 0, k_buf);
        dev.update_descriptor_set(rope_k_ds, 1, rope_freq_buf);
        dev.update_descriptor_set(rope_k_ds, 2, k_buf);

        // KV cache store: k_buf + v_buf → kv_cache at current position
        uint32_t kv_layer_offset = attn_idx * MAX_SEQ * 2 * KV_HEADS * HEAD_DIM * 2;
        dev.update_descriptor_set(kv_store_ds, 0, k_buf);
        dev.update_descriptor_set(kv_store_ds, 1, v_buf);
        dev.update_descriptor_set(kv_store_ds, 2, kv_cache_buf, kv_layer_offset);

        // Attention: q_buf + kv_cache → attn_out_buf
        dev.update_descriptor_set(attn_ds, 0, q_buf);
        dev.update_descriptor_set(attn_ds, 1, kv_cache_buf, kv_layer_offset);
        dev.update_descriptor_set(attn_ds, 2, attn_out_buf);

        // Output projection: attn weights × gated_attn → act_b
        dev.update_descriptor_set(o_proj_ds, 0, weights_buf, attn_o_w->offset, attn_o_w->nbytes);
        dev.update_descriptor_set(o_proj_ds, 1, gated_attn_buf);
        dev.update_descriptor_set(o_proj_ds, 2, act_b);
      }

      // DeltaNet descriptor set updates (layer-specific)
      if (!is_attn) {
        // Compute DeltaNet layer index (0-based among DeltaNet layers)
        // Count how many DeltaNet layers come before this layer
        uint32_t dn_idx = 0;
        for (uint32_t i = 0; i < layer; ++i) {
          if (schedule[i] != model::LayerKind::FullAttention) ++dn_idx;
        }

        // Projections
        dev.update_descriptor_set(dn_qkv_proj_ds, 0, weights_buf, dn_qkv_w->offset, dn_qkv_w->nbytes);
        dev.update_descriptor_set(dn_qkv_proj_ds, 1, act_b);
        dev.update_descriptor_set(dn_qkv_proj_ds, 2, dn_qkv_buf);

        dev.update_descriptor_set(dn_z_proj_ds, 0, weights_buf, dn_z_w->offset, dn_z_w->nbytes);
        dev.update_descriptor_set(dn_z_proj_ds, 1, act_b);
        dev.update_descriptor_set(dn_z_proj_ds, 2, dn_z_buf);

        dev.update_descriptor_set(dn_a_proj_ds, 0, weights_buf, dn_a_w->offset, dn_a_w->nbytes);
        dev.update_descriptor_set(dn_a_proj_ds, 1, act_b);
        dev.update_descriptor_set(dn_a_proj_ds, 2, dn_a_buf);

        dev.update_descriptor_set(dn_b_proj_ds, 0, weights_buf, dn_b_w->offset, dn_b_w->nbytes);
        dev.update_descriptor_set(dn_b_proj_ds, 1, act_b);
        dev.update_descriptor_set(dn_b_proj_ds, 2, dn_b_buf);

        // Conv1d: qkv + conv_state + conv_weights
        dev.update_descriptor_set(dn_conv_ds, 0, dn_qkv_buf);
        uint32_t conv_state_offset = dn_idx * DN_CONV_DIM * DN_CONV_KS * 2;
        dev.update_descriptor_set(dn_conv_ds, 1, dn_conv_state_buf, conv_state_offset);
        dev.update_descriptor_set(dn_conv_ds, 2, weights_buf, dn_conv_w->offset, dn_conv_w->nbytes);
        if (config.debug_dump && layer == 0) {
          std::cerr << "      DEBUG conv weight offset=" << dn_conv_w->offset << " nbytes=" << dn_conv_w->nbytes << std::endl;
        }

        // Recurrent: q + kv_out + state
        dev.update_descriptor_set(dn_recurrent_ds, 0, dn_q_buf);
        dev.update_descriptor_set(dn_recurrent_ds, 1, dn_kv_out_buf);
        uint32_t state_offset_bytes = dn_idx * dn_state_per_layer;
        dev.update_descriptor_set(dn_recurrent_ds, 2, dn_state_buf, state_offset_bytes);

        // Norm+gate: output + z + norm_weight
        dev.update_descriptor_set(dn_norm_gate_ds, 0, dn_kv_out_buf);  // output in V section
        dev.update_descriptor_set(dn_norm_gate_ds, 1, dn_z_buf);
        dev.update_descriptor_set(dn_norm_gate_ds, 2, weights_buf, dn_norm_w->offset, dn_norm_w->nbytes);

        // Output projection
        dev.update_descriptor_set(dn_out_proj_ds, 0, weights_buf, dn_out_w->offset, dn_out_w->nbytes);
        dev.update_descriptor_set(dn_out_proj_ds, 1, dn_kv_out_buf);  // output from recurrent
        dev.update_descriptor_set(dn_out_proj_ds, 2, act_b);
      }
      // Record layer command buffer with 9 dispatches.
      auto cmd = dev.allocate_command_buffer();
      dev.begin_command_buffer(cmd);

      struct { uint32_t N; uint32_t eps_bits; } rms_push = { HIDDEN, float_to_bits(RMS_EPS) };
      struct { uint32_t N; uint32_t pad; } res_push = { HIDDEN, 0 };
      struct { uint32_t out_dim; uint32_t in_dim; } mv_push;
      struct { uint32_t N; uint32_t pad; } silu_push = { INTER, 0 };

      // 1. input_norm(act_a) → act_b
      // For attention: in main cmd. For DeltaNet: done in Submit 1.
      if (is_attn) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rmsnorm_pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout_3, 0, 1, &input_norm_ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &rms_push);
        vkCmdDispatch(cmd, 1, 1, 1);
        barrier(cmd, act_b.buffer, act_bytes);
      }
      // 2. Token mixer
      if (is_attn) {
        size_t q_proj_bytes = Q_HEADS * HEAD_DIM * 2 * 2;  // 4096 * 2 bytes
        size_t q_bytes = Q_HEADS * HEAD_DIM * 2;          // 2048 * 2 bytes
        size_t kv_bytes = KV_HEADS * HEAD_DIM * 2;        // 512 * 2 bytes
        size_t attn_out_bytes = q_bytes;
        uint32_t kv_cache_layer_bytes = MAX_SEQ * 2 * KV_HEADS * HEAD_DIM * 2;

        // 2a. q_proj(act_b) → q_proj_buf [4096]
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, matvec_pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout_3, 0, 1, &q_proj_ds, 0, nullptr);
        struct { uint32_t out_dim; uint32_t in_dim; } q_mv = { Q_HEADS * HEAD_DIM * 2, HIDDEN };
        vkCmdPushConstants(cmd, pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &q_mv);
        vkCmdDispatch(cmd, (Q_HEADS * HEAD_DIM * 2 + 63) / 64, 1, 1);
        barrier(cmd, q_proj_buf.buffer, q_proj_bytes);

        // 2b. k_proj(act_b) → k_buf [512]
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, matvec_pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout_3, 0, 1, &k_proj_ds, 0, nullptr);
        struct { uint32_t out_dim; uint32_t in_dim; } k_mv = { KV_HEADS * HEAD_DIM, HIDDEN };
        vkCmdPushConstants(cmd, pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &k_mv);
        vkCmdDispatch(cmd, (KV_HEADS * HEAD_DIM + 63) / 64, 1, 1);
        barrier(cmd, k_buf.buffer, kv_bytes);

        // 2c. v_proj(act_b) → v_buf [512]
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, matvec_pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout_3, 0, 1, &v_proj_ds, 0, nullptr);
        struct { uint32_t out_dim; uint32_t in_dim; } v_mv = { KV_HEADS * HEAD_DIM, HIDDEN };
        vkCmdPushConstants(cmd, pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &v_mv);
        vkCmdDispatch(cmd, (KV_HEADS * HEAD_DIM + 63) / 64, 1, 1);
        barrier(cmd, v_buf.buffer, kv_bytes);

        // 2d. Split q_proj_buf → q_buf + gate_buf
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, split_q_gate_pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout_3, 0, 1, &split_q_gate_ds, 0, nullptr);
        struct { uint32_t num_heads; uint32_t head_dim; uint32_t total_input; } split_push = { Q_HEADS, HEAD_DIM, Q_HEADS * HEAD_DIM * 2 };
        vkCmdPushConstants(cmd, pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, &split_push);
        vkCmdDispatch(cmd, 1, 1, 1);
        barrier(cmd, q_buf.buffer, q_bytes);
        barrier(cmd, gate_buf.buffer, q_bytes);

        // 2e. q_norm: per-head RMSNorm q_buf [2048] with q_norm [256]
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rms_norm_per_head_pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout_32, 0, 1, &q_norm_ds, 0, nullptr);
        struct { uint32_t num_heads; uint32_t head_dim; uint32_t eps_bits; } qnorm_push = { Q_HEADS, HEAD_DIM, float_to_bits(RMS_EPS) };
        vkCmdPushConstants(cmd, pipeline_layout_32, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, &qnorm_push);
        vkCmdDispatch(cmd, Q_HEADS, 1, 1);
        barrier(cmd, q_buf.buffer, q_bytes);

        // 2f. k_norm: per-head RMSNorm k_buf [512] with k_norm [256]
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rms_norm_per_head_pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout_32, 0, 1, &k_norm_ds, 0, nullptr);
        struct { uint32_t num_heads; uint32_t head_dim; uint32_t eps_bits; } knorm_push = { KV_HEADS, HEAD_DIM, float_to_bits(RMS_EPS) };
        vkCmdPushConstants(cmd, pipeline_layout_32, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, &knorm_push);
        vkCmdDispatch(cmd, KV_HEADS, 1, 1);
        barrier(cmd, k_buf.buffer, kv_bytes);

        // 2g. Apply RoPE to Q (in-place)
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rope_apply_pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout_32, 0, 1, &rope_ds, 0, nullptr);
        struct { uint32_t num_heads; uint32_t head_dim; uint32_t rotary_dim; } rope_q_push = { Q_HEADS, HEAD_DIM, ROTARY_DIM };
        vkCmdPushConstants(cmd, pipeline_layout_32, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, &rope_q_push);
        vkCmdDispatch(cmd, 1, 1, 1);
        barrier(cmd, q_buf.buffer, q_bytes);

        // 2g2. Apply RoPE to K (in-place)
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rope_apply_pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout_32, 0, 1, &rope_k_ds, 0, nullptr);
        struct { uint32_t num_heads; uint32_t head_dim; uint32_t rotary_dim; } rope_k_push = { KV_HEADS, HEAD_DIM, ROTARY_DIM };
        vkCmdPushConstants(cmd, pipeline_layout_32, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, &rope_k_push);
        vkCmdDispatch(cmd, 1, 1, 1);
        barrier(cmd, k_buf.buffer, kv_bytes);

        // 2h. Store K,V into KV cache at position step
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kv_cache_store_pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout_32, 0, 1, &kv_store_ds, 0, nullptr);
        struct { uint32_t kv_heads; uint32_t head_dim; uint32_t position; uint32_t max_seq_len; } kvs_push = { KV_HEADS, HEAD_DIM, step, MAX_SEQ };
        vkCmdPushConstants(cmd, pipeline_layout_32, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, &kvs_push);
        vkCmdDispatch(cmd, 1, 1, 1);
        VkDeviceSize kv_barrier_size = kv_cache_layer_bytes;
        barrier(cmd, kv_cache_buf.buffer, kv_barrier_size);

        // 2i. Attention: Q @ K_cache^T, softmax, weighted V → attn_out_buf
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, attention_decode_pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout_32, 0, 1, &attn_ds, 0, nullptr);
        struct { uint32_t q_heads; uint32_t kv_heads; uint32_t head_dim; uint32_t kv_group_size; uint32_t seq_len; uint32_t max_seq_len; float scale; } attn_push;
        attn_push.q_heads = Q_HEADS;
        attn_push.kv_heads = KV_HEADS;
        attn_push.head_dim = HEAD_DIM;
        attn_push.kv_group_size = KV_GROUP;
        attn_push.seq_len = step + 1;
        attn_push.max_seq_len = MAX_SEQ;
        attn_push.scale = ATTN_SCALE;
        vkCmdPushConstants(cmd, pipeline_layout_32, VK_SHADER_STAGE_COMPUTE_BIT, 0, 28, &attn_push);
        vkCmdDispatch(cmd, Q_HEADS, 1, 1);
        barrier(cmd, attn_out_buf.buffer, attn_out_bytes);

        // 2j. Sigmoid gate: attn_out * sigmoid(gate) → gated_attn_buf
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sigmoid_gate_pipeline_new);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout_3, 0, 1, &sigmoid_gate_ds, 0, nullptr);
        struct { uint32_t N; uint32_t pad; } sg_push = { Q_HEADS * HEAD_DIM, 0 };
        vkCmdPushConstants(cmd, pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &sg_push);
        vkCmdDispatch(cmd, 1, 1, 1);
        barrier(cmd, gated_attn_buf.buffer, q_bytes);
        // 2k. Output projection: o_proj(gated_attn_buf) → act_b
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, matvec_pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout_3, 0, 1, &o_proj_ds, 0, nullptr);
        struct { uint32_t out_dim; uint32_t in_dim; } o_mv = { HIDDEN, Q_HEADS * HEAD_DIM };
        vkCmdPushConstants(cmd, pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &o_mv);
        vkCmdDispatch(cmd, (HIDDEN + 63) / 64, 1, 1);
        barrier(cmd, act_b.buffer, act_bytes);
      } else {
        // DeltaNet layers: full recurrent decode path
        // Split into two submits: (1) projections + conv1d + download a/b
        //                         (2) upload g/beta + recurrent + norm + output proj
        size_t dn_q_bytes = DN_KEY_TOTAL * 2;
        size_t dn_v_bytes = DN_VAL_TOTAL * 2;
        size_t dn_kv_bytes = DN_CONV_DIM * 2;

        // --- Submit 1: Projections and conv1d ---
        {
          auto cmd1 = dev.allocate_command_buffer();
          dev.begin_command_buffer(cmd1);

          // input_norm(act_a) → act_b (must run before projections)
          vkCmdBindPipeline(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE, rmsnorm_pipeline);
          vkCmdBindDescriptorSets(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE,
              pipeline_layout_3, 0, 1, &input_norm_ds, 0, nullptr);
          struct { uint32_t N; uint32_t eps_bits; } rms_dn = { HIDDEN, float_to_bits(RMS_EPS) };
          vkCmdPushConstants(cmd1, pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &rms_dn);
          vkCmdDispatch(cmd1, 1, 1, 1);
          barrier(cmd1, act_b.buffer, act_bytes);

          // QKV projection
          vkCmdBindPipeline(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE, matvec_pipeline);
          vkCmdBindDescriptorSets(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE,
              pipeline_layout_3, 0, 1, &dn_qkv_proj_ds, 0, nullptr);
          struct { uint32_t out_dim; uint32_t in_dim; } dn_qkv_mv = { DN_CONV_DIM, HIDDEN };
          vkCmdPushConstants(cmd1, pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &dn_qkv_mv);
          vkCmdDispatch(cmd1, (DN_CONV_DIM + 63) / 64, 1, 1);
          barrier(cmd1, dn_qkv_buf.buffer, dn_kv_bytes);

          // Debug: dump QKV proj output for layer 0, step 0
          if (layer == 0 && step == 0 && config.debug_dump) {
            dev.end_command_buffer(cmd1);
            dev.submit_and_wait(cmd1);
            {
              std::vector<uint16_t> dump(DN_CONV_DIM);
              dev.download_from_device(dn_qkv_buf, dump.data(), DN_CONV_DIM * 2);
              std::cerr << "      QKV proj [0..4]:";
              for (int i = 0; i < 5; ++i) std::cerr << " " << half_to_float(dump[i]);
              std::cerr << "\n      QKV proj [2048..2052]:";
              for (int i = 0; i < 5; ++i) std::cerr << " " << half_to_float(dump[DN_KEY_TOTAL + i]);
              std::cerr << "\n      QKV proj [4096..4100]:";
              for (int i = 0; i < 5; ++i) std::cerr << " " << half_to_float(dump[DN_KEY_TOTAL*2 + i]);
              std::cerr << "\n";
            }
            cmd1 = dev.allocate_command_buffer();
            dev.begin_command_buffer(cmd1);
            // Dump conv state (should be all zeros for first step)
            {
              size_t dump_bytes = std::min(size_t(20), dn_conv_per_layer);
              std::vector<uint16_t> conv_dump(dump_bytes);
              dev.download_from_device(dn_conv_state_buf, conv_dump.data(), dump_bytes * 2);
              std::cerr << "      conv_state[0..9]:";
              for (size_t i = 0; i < std::min(size_t(10), conv_dump.size()); ++i) std::cerr << " " << half_to_float(conv_dump[i]);
              std::cerr << "\n";
            }
          }

          // Z gate projection
          vkCmdBindPipeline(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE, matvec_pipeline);
          vkCmdBindDescriptorSets(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE,
              pipeline_layout_3, 0, 1, &dn_z_proj_ds, 0, nullptr);
          struct { uint32_t out_dim; uint32_t in_dim; } dn_z_mv = { DN_VAL_TOTAL, HIDDEN };
          vkCmdPushConstants(cmd1, pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &dn_z_mv);
          vkCmdDispatch(cmd1, (DN_VAL_TOTAL + 63) / 64, 1, 1);
          barrier(cmd1, dn_z_buf.buffer, DN_VAL_TOTAL * 2);

          // A projection
          vkCmdBindPipeline(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE, matvec_pipeline);
          vkCmdBindDescriptorSets(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE,
              pipeline_layout_3, 0, 1, &dn_a_proj_ds, 0, nullptr);
          struct { uint32_t out_dim; uint32_t in_dim; } dn_a_mv = { DN_HEADS, HIDDEN };
          vkCmdPushConstants(cmd1, pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &dn_a_mv);
          vkCmdDispatch(cmd1, 1, 1, 1);
          barrier(cmd1, dn_a_buf.buffer, DN_HEADS * 2);

          // B projection
          vkCmdBindPipeline(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE, matvec_pipeline);
          vkCmdBindDescriptorSets(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE,
              pipeline_layout_3, 0, 1, &dn_b_proj_ds, 0, nullptr);
          struct { uint32_t out_dim; uint32_t in_dim; } dn_b_mv = { DN_HEADS, HIDDEN };
          vkCmdPushConstants(cmd1, pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &dn_b_mv);
          vkCmdDispatch(cmd1, 1, 1, 1);
          barrier(cmd1, dn_b_buf.buffer, DN_HEADS * 2);

          // Conv1d step
          // Dump QKV before conv1d to check for corruption
          if (config.debug_dump) {
            dev.end_command_buffer(cmd1);
            dev.submit_and_wait(cmd1);
            {
              std::vector<uint16_t> qkv_check(10);
              dev.download_from_device(dn_qkv_buf, qkv_check.data(), 10 * 2);
              std::cerr << "      QKV before conv1d [0..4]:";
              for (int i = 0; i < 5; ++i) std::cerr << " " << half_to_float(qkv_check[i]);
              std::cerr << "\n";
            }
            cmd1 = dev.allocate_command_buffer();
            dev.begin_command_buffer(cmd1);
          }
          vkCmdBindPipeline(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE, conv1d_step_pipeline);
          vkCmdBindDescriptorSets(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE,
              pipeline_layout_3, 0, 1, &dn_conv_ds, 0, nullptr);
          struct { uint32_t conv_dim; uint32_t kernel_size; } conv_push = { DN_CONV_DIM, DN_CONV_KS };
          vkCmdPushConstants(cmd1, pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &conv_push);
          vkCmdDispatch(cmd1, 1, 1, 1);
          barrier(cmd1, dn_qkv_buf.buffer, dn_kv_bytes);



          // L2-norm Q (in-place on dn_qkv_buf[0..2047])
          dev.update_descriptor_set(dn_l2_q_ds, 0, dn_qkv_buf, 0, DN_KEY_TOTAL * 2);
          dev.update_descriptor_set(dn_l2_q_ds, 1, dn_qkv_buf, 0, DN_KEY_TOTAL * 2);
          dev.update_descriptor_set(dn_l2_q_ds, 2, dn_qkv_buf, 0, DN_KEY_TOTAL * 2);
          vkCmdBindPipeline(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE, l2_norm_per_head_pipeline);
          vkCmdBindDescriptorSets(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE,
              pipeline_layout_3, 0, 1, &dn_l2_q_ds, 0, nullptr);
          struct { uint32_t num_heads; uint32_t head_dim; } l2q_push = { DN_HEADS, DN_K_DIM };
          vkCmdPushConstants(cmd1, pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &l2q_push);
          vkCmdDispatch(cmd1, DN_HEADS, 1, 1);
          barrier(cmd1, dn_qkv_buf.buffer, dn_kv_bytes);

          // L2-norm K (in-place on dn_qkv_buf[2048..4095])
          dev.update_descriptor_set(dn_l2_k_ds, 0, dn_qkv_buf, DN_KEY_TOTAL * 2, DN_KEY_TOTAL * 2);
          dev.update_descriptor_set(dn_l2_k_ds, 1, dn_qkv_buf, DN_KEY_TOTAL * 2, DN_KEY_TOTAL * 2);
          dev.update_descriptor_set(dn_l2_k_ds, 2, dn_qkv_buf, DN_KEY_TOTAL * 2, DN_KEY_TOTAL * 2);
          vkCmdBindPipeline(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE, l2_norm_per_head_pipeline);
          vkCmdBindDescriptorSets(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE,
              pipeline_layout_3, 0, 1, &dn_l2_k_ds, 0, nullptr);
          struct { uint32_t num_heads; uint32_t head_dim; } l2k_push = { DN_HEADS, DN_K_DIM };
          vkCmdPushConstants(cmd1, pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &l2k_push);
          vkCmdDispatch(cmd1, DN_HEADS, 1, 1);
          barrier(cmd1, dn_qkv_buf.buffer, dn_kv_bytes);
          dev.end_command_buffer(cmd1);
          dev.submit_and_wait(cmd1);
        }
        // Debug: dump intermediate buffers for layer 0, step 0
        if (config.debug_dump && layer == 0 && step == 0) {
          {
            std::vector<uint16_t> dump(HIDDEN);
            dev.download_from_device(act_b, dump.data(), act_bytes);
            std::cerr << "      after_input_norm act_b first 5:";
            for (int i = 0; i < 5; ++i) std::cerr << " " << half_to_float(dump[i]);
            std::cerr << "\n";
          }
          {
            size_t dn_total = DN_CONV_DIM;
            std::vector<uint16_t> dump(dn_total);
            dev.download_from_device(dn_qkv_buf, dump.data(), dn_total * 2);
            std::cerr << "      after_conv1d Q[0..4]:";
            for (int i = 0; i < 5; ++i) std::cerr << " " << half_to_float(dump[i]);
            std::cerr << "\n      after_conv1d K[0..4]:";
            for (int i = 0; i < 5; ++i) std::cerr << " " << half_to_float(dump[DN_KEY_TOTAL + i]);
            std::cerr << "\n      after_conv1d V[0..4]:";
            for (int i = 0; i < 5; ++i) std::cerr << " " << half_to_float(dump[DN_KEY_TOTAL*2 + i]);
            std::cerr << "\n";
          }
          {
            std::vector<uint16_t> dump(DN_VAL_TOTAL);
            dev.download_from_device(dn_z_buf, dump.data(), DN_VAL_TOTAL * 2);
            std::cerr << "      dn_z[0..4]:";
            for (int i = 0; i < 5; ++i) std::cerr << " " << half_to_float(dump[i]);
            std::cerr << "\n";
          }
          {
            std::vector<uint16_t> dump(DN_HEADS);
            dev.download_from_device(dn_a_buf, dump.data(), DN_HEADS * 2);
            std::cerr << "      dn_a[0..4]:";
            for (int i = 0; i < 5; ++i) std::cerr << " " << half_to_float(dump[i]);
            std::cerr << "\n";
          }
          {
            std::vector<uint16_t> dump(DN_HEADS);
            dev.download_from_device(dn_b_buf, dump.data(), DN_HEADS * 2);
            std::cerr << "      dn_b[0..4]:";
            for (int i = 0; i < 5; ++i) std::cerr << " " << half_to_float(dump[i]);
            std::cerr << "\n";
          }
        }

        // --- CPU: Compute g, beta from a_log, dt_bias, a, b. Upload to state buffer. ---
        {
          std::vector<uint16_t> a_fp16(DN_HEADS), b_fp16(DN_HEADS);
          dev.download_from_device(dn_a_buf, a_fp16.data(), DN_HEADS * 2);
          dev.download_from_device(dn_b_buf, b_fp16.data(), DN_HEADS * 2);
          uint32_t dn_idx = 0;
          for (uint32_t i = 0; i < layer; ++i) {
            if (schedule[i] != model::LayerKind::FullAttention) ++dn_idx;
          }
          const auto& a_log_cached = cached_a_log[dn_idx];
          const auto& dt_bias_cached = cached_dt_bias[dn_idx];

          std::vector<float> g_beta(2 * DN_HEADS);
          for (uint32_t h = 0; h < DN_HEADS; ++h) {
            float a_val = half_to_float(a_fp16[h]);
            float b_val = half_to_float(b_fp16[h]);
            float a_log_val = a_log_cached[h];
            float dt_bias_val = dt_bias_cached[h];
            float sp = std::log(1.0f + std::exp(a_val + dt_bias_val));
            g_beta[h] = -std::exp(a_log_val) * sp;
            g_beta[DN_HEADS + h] = 1.0f / (1.0f + std::exp(-b_val));
          }

          VkDeviceSize g_beta_offset = dn_idx * dn_state_per_layer + DN_HEADS * DN_K_DIM * DN_V_DIM * 4;
          auto g_beta_upload = dev.create_host_visible_buffer(g_beta.size() * 4);
          dev.upload_to_host_visible(g_beta_upload, g_beta.data(), g_beta.size() * 4);
          {
            auto upload_cmd = dev.allocate_command_buffer();
            dev.begin_command_buffer(upload_cmd);
            VkBufferCopy copy{0, g_beta_offset, g_beta.size() * 4};
            vkCmdCopyBuffer(upload_cmd, g_beta_upload.buffer, dn_state_buf.buffer, 1, &copy);
            dev.end_command_buffer(upload_cmd);
            dev.submit_and_wait(upload_cmd);
          }
          dev.destroy_buffer(g_beta_upload);
        }

        // --- Record into main layer command buffer ---
        dev.update_descriptor_set(dn_recurrent_ds, 0, dn_qkv_buf, 0, DN_KEY_TOTAL * 2);
        dev.update_descriptor_set(dn_recurrent_ds, 1, dn_qkv_buf, DN_KEY_TOTAL * 2, (DN_KEY_TOTAL + DN_VAL_TOTAL) * 2);

        // Recurrent update (FP32 state)
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, deltanet_recurrent_pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout_32, 0, 1, &dn_recurrent_ds, 0, nullptr);
        uint32_t state_float_total = DN_HEADS * DN_K_DIM * DN_V_DIM;
        struct { uint32_t num_heads; uint32_t k_dim; uint32_t v_dim; uint32_t state_total; uint32_t q_scale_bits; } dn_rec_push = { DN_HEADS, DN_K_DIM, DN_V_DIM, state_float_total, float_to_bits(DN_Q_SCALE) };
        vkCmdPushConstants(cmd, pipeline_layout_32, VK_SHADER_STAGE_COMPUTE_BIT, 0, 20, &dn_rec_push);
        vkCmdDispatch(cmd, DN_HEADS, 1, 1);
        barrier(cmd, dn_qkv_buf.buffer, dn_kv_bytes);

        // Norm+gate
        dev.update_descriptor_set(dn_norm_gate_ds, 0, dn_qkv_buf, (DN_KEY_TOTAL + DN_KEY_TOTAL) * 2, DN_VAL_TOTAL * 2);
        dev.update_descriptor_set(dn_norm_gate_ds, 1, dn_z_buf);
        dev.update_descriptor_set(dn_norm_gate_ds, 2, weights_buf, dn_norm_w->offset, dn_norm_w->nbytes);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, deltanet_norm_gate_pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout_32, 0, 1, &dn_norm_gate_ds, 0, nullptr);
        struct { uint32_t num_heads; uint32_t head_dim; uint32_t eps_bits; uint32_t output_offset; } dn_ng_push = { DN_HEADS, DN_V_DIM, float_to_bits(RMS_EPS), 0 };
        vkCmdPushConstants(cmd, pipeline_layout_32, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, &dn_ng_push);
        vkCmdDispatch(cmd, DN_HEADS, 1, 1);
        barrier(cmd, dn_qkv_buf.buffer, dn_kv_bytes);

        // Output projection
        dev.update_descriptor_set(dn_out_proj_ds, 0, weights_buf, dn_out_w->offset, dn_out_w->nbytes);
        dev.update_descriptor_set(dn_out_proj_ds, 1, dn_qkv_buf, (DN_KEY_TOTAL + DN_KEY_TOTAL) * 2, DN_VAL_TOTAL * 2);
        dev.update_descriptor_set(dn_out_proj_ds, 2, act_b);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, matvec_pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout_3, 0, 1, &dn_out_proj_ds, 0, nullptr);
        struct { uint32_t out_dim; uint32_t in_dim; } dn_out_mv = { HIDDEN, DN_VAL_TOTAL };
        vkCmdPushConstants(cmd, pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &dn_out_mv);
        vkCmdDispatch(cmd, (HIDDEN + 63) / 64, 1, 1);
        barrier(cmd, act_b.buffer, act_bytes);
      }
      // 3. residual_add(act_a, act_b) → act_c
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, residual_add_pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          pipeline_layout_3, 0, 1, &residual1_ds, 0, nullptr);
      vkCmdPushConstants(cmd, pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &res_push);
      vkCmdDispatch(cmd, 1, 1, 1);
      barrier(cmd, act_c.buffer, act_bytes);

      // 4. post_norm(act_c) → act_a
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rmsnorm_pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          pipeline_layout_3, 0, 1, &post_norm_ds, 0, nullptr);
      vkCmdPushConstants(cmd, pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &rms_push);
      vkCmdDispatch(cmd, 1, 1, 1);
      barrier(cmd, act_a.buffer, act_bytes);

      // 5. gate_matvec(act_a) → mlp_gate_buf
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, matvec_pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          pipeline_layout_3, 0, 1, &gate_ds, 0, nullptr);
      mv_push = { INTER, HIDDEN };
      vkCmdPushConstants(cmd, pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &mv_push);
      vkCmdDispatch(cmd, (INTER + 63) / 64, 1, 1);
      barrier(cmd, mlp_gate_buf.buffer, inter_bytes);

      // 6. up_matvec(act_a) → mlp_up_buf
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, matvec_pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          pipeline_layout_3, 0, 1, &up_ds, 0, nullptr);
      mv_push = { INTER, HIDDEN };
      vkCmdPushConstants(cmd, pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &mv_push);
      vkCmdDispatch(cmd, (INTER + 63) / 64, 1, 1);
      barrier(cmd, mlp_up_buf.buffer, inter_bytes);

      // 7. silu_gate(mlp_gate_buf, mlp_up_buf) → mlp_silu_buf
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, silu_gate_pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          pipeline_layout_3, 0, 1, &silu_gate_ds, 0, nullptr);
      vkCmdPushConstants(cmd, pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &silu_push);
      vkCmdDispatch(cmd, 1, 1, 1);
      barrier(cmd, mlp_silu_buf.buffer, inter_bytes);

      // 8. down_matvec(mlp_silu_buf) → act_b
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, matvec_pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          pipeline_layout_3, 0, 1, &down_ds, 0, nullptr);
      mv_push = { HIDDEN, INTER };
      vkCmdPushConstants(cmd, pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &mv_push);
      vkCmdDispatch(cmd, (HIDDEN + 63) / 64, 1, 1);
      barrier(cmd, act_b.buffer, act_bytes);

      // 9. residual_add(act_c, act_b) → act_a
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, residual_add_pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          pipeline_layout_3, 0, 1, &residual2_ds, 0, nullptr);
      vkCmdPushConstants(cmd, pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &res_push);
      vkCmdDispatch(cmd, 1, 1, 1);
      barrier(cmd, act_a.buffer, act_bytes);

      dev.end_command_buffer(cmd);
      dev.submit_and_wait(cmd);

      // Debug: dump hidden state after this layer
      if (config.debug_dump && (step == 0 || step == prompt_len - 1)) {
        std::vector<uint16_t> dump(HIDDEN);
        dev.download_from_device(act_a, dump.data(), act_bytes);
        std::cerr << "    layer " << layer << " (" << (is_attn ? "attn" : "dn") << ") act_a first 5:";
        for (int i = 0; i < 5; ++i) {
          std::cerr << " " << half_to_float(dump[i]);
        }
        std::cerr << "\n";
      }
    }


    // Skip LM head + argmax for prefill steps
    if (is_prefill) {
      if (config.verbose) {
        std::cerr << "  prefill " << step << ": token " << current_token << "\n";
      }
      continue;
    }
    // --- Final RMSNorm + LM head + Argmax ---
    {
      auto cmd = dev.allocate_command_buffer();
      dev.begin_command_buffer(cmd);

      struct { uint32_t N; uint32_t eps_bits; } fn_push = { HIDDEN, float_to_bits(RMS_EPS) };
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rmsnorm_pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          pipeline_layout_3, 0, 1, &final_norm_ds, 0, nullptr);
      vkCmdPushConstants(cmd, pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &fn_push);
      vkCmdDispatch(cmd, 1, 1, 1);
      barrier(cmd, act_b.buffer, act_bytes);

      struct { uint32_t out_dim; uint32_t in_dim; } lm_push = { VOCAB, HIDDEN };
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, matvec_pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          pipeline_layout_3, 0, 1, &lm_head_ds, 0, nullptr);
      vkCmdPushConstants(cmd, pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &lm_push);
      vkCmdDispatch(cmd, (VOCAB + 63) / 64, 1, 1);
      barrier(cmd, logits.buffer, VOCAB * 2);

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, argmax_pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          pipeline_layout_2, 0, 1, &argmax_ds, 0, nullptr);
      uint32_t argmax_push = VOCAB;
      vkCmdPushConstants(cmd, pipeline_layout_2, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &argmax_push);
      vkCmdDispatch(cmd, 1, 1, 1);

      dev.end_command_buffer(cmd);
      dev.submit_and_wait(cmd);
    }

    uint32_t next_token = 0;
    dev.download_from_device(argmax_result, &next_token, 4);
    if (config.debug_dump) {
      std::vector<uint16_t> logit_dump(VOCAB);
      dev.download_from_device(logits, logit_dump.data(), VOCAB * 2);
      std::vector<std::pair<float, uint32_t>> top;
      top.reserve(5);
      for (uint32_t i = 0; i < VOCAB; ++i) {
        float value = half_to_float(logit_dump[i]);
        if (top.size() < 5) {
          top.emplace_back(value, i);
          if (top.size() == 5) {
            std::sort(top.begin(), top.end(), std::greater<>());
          }
        } else if (value > top.back().first) {
          top.back() = {value, i};
          std::sort(top.begin(), top.end(), std::greater<>());
        }
      }
      uint32_t decode_step = step - (prompt_len > 1 ? prompt_len - 1 : 0);
      std::cerr << "  decode " << decode_step << " top5:";
      for (const auto& [value, token] : top) {
        std::cerr << " (" << token << "," << value << ")";
      }
      std::cerr << "\n";
    }
    tokens.push_back(next_token);
    result.generated_tokens.push_back(next_token);

    if (config.verbose) {
      uint32_t decode_step = step - (prompt_len > 1 ? prompt_len - 1 : 0);
      std::cerr << "  decode " << decode_step << ": token " << next_token << "\n";
    }
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  result.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  // Cleanup
  dev.destroy_buffer(act_a);
  dev.destroy_buffer(act_b);
  dev.destroy_buffer(act_c);
  dev.destroy_buffer(logits);
  dev.destroy_buffer(argmax_result);
  dev.destroy_buffer(mlp_gate_buf);
  dev.destroy_buffer(mlp_up_buf);
  dev.destroy_buffer(mlp_silu_buf);
  dev.destroy_buffer(q_proj_buf);
  dev.destroy_buffer(q_buf);
  dev.destroy_buffer(gate_buf);
  dev.destroy_buffer(k_buf);
  dev.destroy_buffer(v_buf);
  dev.destroy_buffer(attn_out_buf);
  dev.destroy_buffer(gated_attn_buf);
  dev.destroy_buffer(kv_cache_buf);
  dev.destroy_buffer(rope_freq_buf);
  dev.destroy_buffer(dn_qkv_buf);
  dev.destroy_buffer(dn_z_buf);
  dev.destroy_buffer(dn_a_buf);
  dev.destroy_buffer(dn_b_buf);
  dev.destroy_buffer(dn_q_buf);
  dev.destroy_buffer(dn_kv_out_buf);
  dev.destroy_buffer(dn_state_buf);
  dev.destroy_buffer(dn_conv_state_buf);
  dev.destroy_buffer(weights_buf);
  dev.destroy_buffer(final_norm_buf);
  dev.destroy_pipeline(embedding_pipeline);
  dev.destroy_pipeline(rmsnorm_pipeline);
  dev.destroy_pipeline(matvec_pipeline);
  dev.destroy_pipeline(argmax_pipeline);
  dev.destroy_pipeline(silu_gate_pipeline);
  dev.destroy_pipeline(residual_add_pipeline);
  dev.destroy_pipeline(rope_apply_pipeline);
  dev.destroy_pipeline(attention_decode_pipeline);
  dev.destroy_pipeline(kv_cache_store_pipeline);
  dev.destroy_pipeline(sigmoid_gate_pipeline_new);
  dev.destroy_pipeline(rms_norm_per_head_pipeline);
  dev.destroy_pipeline(split_q_gate_pipeline);
  dev.destroy_pipeline(deltanet_recurrent_pipeline);
  dev.destroy_pipeline(conv1d_step_pipeline);
  dev.destroy_pipeline(deltanet_norm_gate_pipeline);
  dev.destroy_pipeline(l2_norm_per_head_pipeline);
  dev.destroy_shader_module(embedding_module);
  dev.destroy_shader_module(rmsnorm_module);
  dev.destroy_shader_module(matvec_module);
  dev.destroy_shader_module(argmax_module);
  dev.destroy_shader_module(silu_gate_module);
  dev.destroy_shader_module(residual_add_module);
  dev.destroy_shader_module(rope_apply_module);
  dev.destroy_shader_module(attention_decode_module);
  dev.destroy_shader_module(kv_cache_store_module);
  dev.destroy_shader_module(sigmoid_gate_module_new);
  dev.destroy_shader_module(rms_norm_per_head_module);
  dev.destroy_shader_module(split_q_gate_module);
  dev.destroy_shader_module(deltanet_recurrent_module);
  dev.destroy_shader_module(conv1d_step_module);
  dev.destroy_shader_module(deltanet_norm_gate_module);
  dev.destroy_shader_module(l2_norm_per_head_module);
  dev.destroy_pipeline_layout(pipeline_layout_3);
  dev.destroy_pipeline_layout(pipeline_layout_2);
  dev.destroy_pipeline_layout(pipeline_layout_32);
  dev.destroy_descriptor_set_layout(ds_layout_3);
  dev.destroy_descriptor_set_layout(ds_layout_2);
  dev.destroy();

  return result;
#endif
}

}  // namespace spock::runtime

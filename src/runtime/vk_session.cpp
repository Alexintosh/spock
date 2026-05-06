#include "runtime/vk_session.hpp"

#if SPOCK_HAS_VULKAN && !defined(SPOCK_VULKAN_STUB)

#include "model/qwen35_config.hpp"

#include "runtime/deltanet_chunk.hpp"

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <algorithm>
#include <utility>

#include <vulkan/vulkan.h>

namespace spock::runtime {

namespace {

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

uint32_t float_to_bits(float f) {
  uint32_t bits;
  memcpy(&bits, &f, 4);
  return bits;
}

VkDeviceSize align_storage_offset(VkDeviceSize value) {
  constexpr VkDeviceSize kConservativeStorageBufferAlignment = 256;
  return (value + kConservativeStorageBufferAlignment - 1) &
         ~(kConservativeStorageBufferAlignment - 1);
}

uint16_t float_to_half(float f) {
  uint32_t bits;
  memcpy(&bits, &f, 4);
  uint32_t sign = (bits >> 31) & 1;
  uint32_t exponent = (bits >> 23) & 0xff;
  uint32_t mantissa = bits & 0x7fffff;
  uint16_t h_sign = sign << 15;
  if (exponent == 0) {
    return h_sign;
  } else if (exponent == 0xff) {
    return h_sign | 0x7c00 | (mantissa ? 0x200 : 0);
  }
  int32_t new_exp = exponent - 127 + 15;
  if (new_exp <= 0) {
    return h_sign;  // flush to zero
  } else if (new_exp >= 31) {
    return h_sign | 0x7c00;  // infinity
  }
  return h_sign | (new_exp << 10) | (mantissa >> 13);
}


void upload_raw(VulkanDevice& dev, const VulkanDevice::Buffer& buf,
                const void* data, size_t size) {
  dev.upload_to_device(buf, data, static_cast<VkDeviceSize>(size));
}

std::vector<uint32_t> read_spirv(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open shader: " + path);
  auto size = f.tellg();
  f.seekg(0);
  std::vector<uint32_t> words(size / 4);
  f.read(reinterpret_cast<char*>(words.data()), size);
  return words;
}

}  // namespace


// ---------------------------------------------------------------------------
// Constructor / destructor
// ---------------------------------------------------------------------------

DecodeSession::DecodeSession(const std::string& repack_dir, bool verbose)
    : verbose_(verbose) {
  dev_.initialize();

  if (verbose_) {
    const auto& caps = dev_.capabilities();
    std::cerr << "Vulkan device: " << caps.device_name << "\n";
    std::cerr << "Subgroup size: " << caps.subgroup_size << "\n";
    std::cerr << "Max shared memory: " << caps.max_shared_memory_bytes << "\n";
  }

  // --- Load weights ---
  artifact_ = WeightArtifact::load(repack_dir);
  if (verbose_) {
    std::cerr << "Loaded " << artifact_.tensor_count() << " tensors ("
              << (artifact_.total_bytes() / (1024 * 1024)) << " MiB)\n";
  }

  // --- Load shaders ---
  auto shader_dir = std::string(SHADER_DIR);
  auto emb_spv = read_spirv(shader_dir + "/embedding_lookup.comp.spv");
  auto emb_buf_spv = read_spirv(shader_dir + "/embedding_lookup_from_buffer.comp.spv");
  auto rms_spv = read_spirv(shader_dir + "/rms_norm.comp.spv");
  auto mv_tiled_spv = read_spirv(shader_dir + "/matvec_tiled.comp.spv");
  auto mv_spv = read_spirv(shader_dir + "/matvec.comp.spv");
  auto mv_f32out_spv = read_spirv(shader_dir + "/matvec_f32_out.comp.spv");
  auto am_spv = read_spirv(shader_dir + "/argmax.comp.spv");
  auto sg_spv = read_spirv(shader_dir + "/silu_gate.comp.spv");
  auto ra_spv = read_spirv(shader_dir + "/residual_add.comp.spv");
  auto ra_mixed_spv = read_spirv(shader_dir + "/residual_add_mixed.comp.spv");
  auto rope_spv = read_spirv(shader_dir + "/rope_apply.comp.spv");
  auto attn_spv = read_spirv(shader_dir + "/attention_decode.comp.spv");
  auto kv_spv = read_spirv(shader_dir + "/kv_cache_store.comp.spv");
  auto sig_spv = read_spirv(shader_dir + "/sigmoid_gate.comp.spv");
  auto rnph_spv = read_spirv(shader_dir + "/rms_norm_per_head.comp.spv");
  auto sqg_spv = read_spirv(shader_dir + "/split_q_gate.comp.spv");
  auto dnr_spv = read_spirv(shader_dir + "/deltanet_recurrent.comp.spv");
  auto c1d_spv = read_spirv(shader_dir + "/conv1d_step.comp.spv");
  auto dnng_spv = read_spirv(shader_dir + "/deltanet_norm_gate.comp.spv");
  auto l2n_spv = read_spirv(shader_dir + "/l2_norm_per_head.comp.spv");
  auto dncgb_spv = read_spirv(shader_dir + "/deltanet_compute_g_beta.comp.spv");
  auto dncp_spv = read_spirv(shader_dir + "/deltanet_chunk_prefill.comp.spv");
  auto dncp_tiled_spv = read_spirv(shader_dir + "/deltanet_chunk_prefill_tiled.comp.spv");
  auto dncoll_spv = read_spirv(shader_dir + "/deltanet_prefill_collect.comp.spv");
  auto dnclfp16_spv = read_spirv(shader_dir + "/deltanet_chunk_last_to_fp16.comp.spv");
  auto dn_conv_l2_spv = read_spirv(shader_dir + "/deltanet_conv_l2_qk.comp.spv");
  auto dn_rec_gbeta_spv = read_spirv(shader_dir + "/deltanet_recurrent_gbeta.comp.spv");
  auto lmht_spv = read_spirv(shader_dir + "/lm_head_tiled.comp.spv");
  auto dn_rec_gbeta_ng_spv = read_spirv(shader_dir + "/deltanet_recurrent_gbeta_norm_gate.comp.spv");
  // --- Pipeline infrastructure ---
  pipes_ = std::make_unique<Pipelines>();

  std::vector<VkDescriptorSetLayoutBinding> bindings_3 = {
      {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
  };
  std::vector<VkDescriptorSetLayoutBinding> bindings_2 = {
      {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
  };

  std::vector<VkDescriptorSetLayoutBinding> bindings_4 = {
      {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
  };

  std::vector<VkDescriptorSetLayoutBinding> bindings_7 = {
      {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
  };

  std::vector<VkDescriptorSetLayoutBinding> bindings_6 = {
      {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
  };

  std::vector<VkDescriptorSetLayoutBinding> bindings_8 = {
      {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
  };
  pipes_->ds_layout_3 = dev_.create_descriptor_set_layout(bindings_3);
  pipes_->ds_layout_2 = dev_.create_descriptor_set_layout(bindings_2);
  pipes_->ds_layout_4 = dev_.create_descriptor_set_layout(bindings_4);
  pipes_->ds_layout_6 = dev_.create_descriptor_set_layout(bindings_6);
  pipes_->pipeline_layout_3 = dev_.create_pipeline_layout(pipes_->ds_layout_3, 8);
  pipes_->pipeline_layout_2 = dev_.create_pipeline_layout(pipes_->ds_layout_2, 8);
  pipes_->pipeline_layout_32 = dev_.create_pipeline_layout(pipes_->ds_layout_3, 32);
  pipes_->pipeline_layout_4 = dev_.create_pipeline_layout(pipes_->ds_layout_4, 8);
  pipes_->pipeline_layout_6_32 = dev_.create_pipeline_layout(pipes_->ds_layout_6, 32);
  pipes_->ds_layout_8 = dev_.create_descriptor_set_layout(bindings_8);
  pipes_->pipeline_layout_8_32 = dev_.create_pipeline_layout(pipes_->ds_layout_8, 32);
  pipes_->ds_layout_7 = dev_.create_descriptor_set_layout(bindings_7);
  pipes_->pipeline_layout_cp = dev_.create_pipeline_layout(pipes_->ds_layout_7, 40);

  auto make_module = [&](auto& spv) { return dev_.create_shader_module(spv); };
  pipes_->embedding_module = make_module(emb_spv);
  pipes_->embedding_from_buffer_module = make_module(emb_buf_spv);
  pipes_->rmsnorm_module = make_module(rms_spv);
  pipes_->matvec_module = make_module(mv_spv);
  pipes_->matvec_tiled_module = make_module(mv_tiled_spv);
  pipes_->matvec_f32_out_module = make_module(mv_f32out_spv);
  pipes_->argmax_module = make_module(am_spv);
  pipes_->silu_gate_module = make_module(sg_spv);
  pipes_->residual_add_module = make_module(ra_spv);
  pipes_->residual_add_mixed_module = make_module(ra_mixed_spv);
  pipes_->rope_apply_module = make_module(rope_spv);
  pipes_->attention_decode_module = make_module(attn_spv);
  pipes_->kv_cache_store_module = make_module(kv_spv);
  pipes_->sigmoid_gate_module = make_module(sig_spv);
  pipes_->rms_norm_per_head_module = make_module(rnph_spv);
  pipes_->split_q_gate_module = make_module(sqg_spv);
  pipes_->deltanet_recurrent_module = make_module(dnr_spv);
  pipes_->conv1d_step_module = make_module(c1d_spv);
  pipes_->deltanet_norm_gate_module = make_module(dnng_spv);
  pipes_->l2_norm_per_head_module = make_module(l2n_spv);
  pipes_->deltanet_compute_g_beta_module = make_module(dncgb_spv);
  pipes_->deltanet_chunk_prefill_module = make_module(dncp_spv);
  pipes_->deltanet_chunk_prefill_tiled_module = make_module(dncp_tiled_spv);
  pipes_->deltanet_prefill_collect_module = make_module(dncoll_spv);
  pipes_->deltanet_chunk_last_to_fp16_module = make_module(dnclfp16_spv);
  pipes_->deltanet_conv_l2_qk_module = make_module(dn_conv_l2_spv);
  pipes_->deltanet_recurrent_gbeta_module = make_module(dn_rec_gbeta_spv);
  pipes_->deltanet_recurrent_gbeta_norm_gate_module = make_module(dn_rec_gbeta_ng_spv);
  pipes_->lm_head_tiled_module = make_module(lmht_spv);

  auto make_pipe = [&](VkShaderModule m, VkPipelineLayout l) {
    return dev_.create_compute_pipeline(m, l);
  };
  pipes_->embedding = make_pipe(pipes_->embedding_module, pipes_->pipeline_layout_2);
  pipes_->embedding_from_buffer = make_pipe(pipes_->embedding_from_buffer_module, pipes_->pipeline_layout_3);
  pipes_->rmsnorm = make_pipe(pipes_->rmsnorm_module, pipes_->pipeline_layout_3);
  pipes_->matvec = make_pipe(pipes_->matvec_module, pipes_->pipeline_layout_3);
  pipes_->matvec_tiled = make_pipe(pipes_->matvec_tiled_module, pipes_->pipeline_layout_3);
  pipes_->matvec_f32_out = make_pipe(pipes_->matvec_f32_out_module, pipes_->pipeline_layout_3);
  pipes_->argmax = make_pipe(pipes_->argmax_module, pipes_->pipeline_layout_2);
  pipes_->silu_gate = make_pipe(pipes_->silu_gate_module, pipes_->pipeline_layout_3);
  pipes_->residual_add = make_pipe(pipes_->residual_add_module, pipes_->pipeline_layout_3);
  pipes_->residual_add_mixed = make_pipe(pipes_->residual_add_mixed_module, pipes_->pipeline_layout_3);
  pipes_->rope_apply = make_pipe(pipes_->rope_apply_module, pipes_->pipeline_layout_32);
  pipes_->attention_decode = make_pipe(pipes_->attention_decode_module, pipes_->pipeline_layout_32);
  pipes_->kv_cache_store = make_pipe(pipes_->kv_cache_store_module, pipes_->pipeline_layout_32);
  pipes_->sigmoid_gate = make_pipe(pipes_->sigmoid_gate_module, pipes_->pipeline_layout_3);
  pipes_->rms_norm_per_head = make_pipe(pipes_->rms_norm_per_head_module, pipes_->pipeline_layout_32);
  pipes_->split_q_gate = make_pipe(pipes_->split_q_gate_module, pipes_->pipeline_layout_3);
  pipes_->deltanet_recurrent = make_pipe(pipes_->deltanet_recurrent_module, pipes_->pipeline_layout_32);
  pipes_->conv1d_step = make_pipe(pipes_->conv1d_step_module, pipes_->pipeline_layout_3);
  pipes_->deltanet_norm_gate = make_pipe(pipes_->deltanet_norm_gate_module, pipes_->pipeline_layout_32);
  pipes_->l2_norm_per_head = make_pipe(pipes_->l2_norm_per_head_module, pipes_->pipeline_layout_3);
  pipes_->deltanet_compute_g_beta = make_pipe(pipes_->deltanet_compute_g_beta_module, pipes_->pipeline_layout_4);
  pipes_->deltanet_chunk_prefill = make_pipe(pipes_->deltanet_chunk_prefill_module, pipes_->pipeline_layout_cp);
  pipes_->deltanet_chunk_prefill_tiled = make_pipe(pipes_->deltanet_chunk_prefill_tiled_module, pipes_->pipeline_layout_cp);
  pipes_->deltanet_prefill_collect = make_pipe(pipes_->deltanet_prefill_collect_module, pipes_->pipeline_layout_cp);
  pipes_->deltanet_chunk_last_to_fp16 = make_pipe(pipes_->deltanet_chunk_last_to_fp16_module, pipes_->pipeline_layout_2);
  pipes_->deltanet_conv_l2_qk = make_pipe(pipes_->deltanet_conv_l2_qk_module, pipes_->pipeline_layout_32);
  pipes_->deltanet_recurrent_gbeta = make_pipe(pipes_->deltanet_recurrent_gbeta_module, pipes_->pipeline_layout_6_32);
  pipes_->deltanet_recurrent_gbeta_norm_gate = make_pipe(pipes_->deltanet_recurrent_gbeta_norm_gate_module, pipes_->pipeline_layout_8_32);
  pipes_->lm_head_tiled = make_pipe(pipes_->lm_head_tiled_module, pipes_->pipeline_layout_3);
  // --- Allocate buffers ---
  bufs_ = std::make_unique<Buffers>();

  constexpr float RMS_EPS_F = 1e-6f;
  (void)RMS_EPS_F;
  constexpr float ATTN_SCALE_F = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
  (void)ATTN_SCALE_F;
  constexpr float DN_Q_SCALE_F = 1.0f / std::sqrt(static_cast<float>(DN_K_DIM));
  (void)DN_Q_SCALE_F;

  bufs_->act_bytes = HIDDEN * 2;
  bufs_->act_c_bytes = HIDDEN * 2;
  bufs_->act_a = dev_.create_device_local_buffer(bufs_->act_bytes);
  bufs_->act_b = dev_.create_device_local_buffer(bufs_->act_bytes);
  bufs_->act_c = dev_.create_device_local_buffer(bufs_->act_c_bytes);
  bufs_->logits = dev_.create_device_local_buffer(VOCAB * 2);
  bufs_->argmax_result = dev_.create_device_local_buffer(4);

  size_t inter_bytes = INTER * 2;
  bufs_->mlp_gate = dev_.create_device_local_buffer(inter_bytes);
  bufs_->mlp_up = dev_.create_device_local_buffer(inter_bytes);
  bufs_->mlp_silu = dev_.create_device_local_buffer(inter_bytes);

  // Attention buffers
  size_t q_proj_out = Q_HEADS * HEAD_DIM * 2;  // 4096 (query + gate interleaved)
  bufs_->q_proj = dev_.create_device_local_buffer(q_proj_out * 2);
  bufs_->q = dev_.create_device_local_buffer(Q_HEADS * HEAD_DIM * 2);
  bufs_->gate = dev_.create_device_local_buffer(Q_HEADS * HEAD_DIM * 2);
  bufs_->k = dev_.create_device_local_buffer(KV_HEADS * HEAD_DIM * 2);
  bufs_->v = dev_.create_device_local_buffer(KV_HEADS * HEAD_DIM * 2);
  bufs_->attn_out = dev_.create_device_local_buffer(Q_HEADS * HEAD_DIM * 4);
  bufs_->gated_attn = dev_.create_device_local_buffer(Q_HEADS * HEAD_DIM * 4);
  bufs_->attn_proj_f32 = dev_.create_device_local_buffer(HIDDEN * 4);

  bufs_->kv_cache_layer_bytes = MAX_SEQ * 2 * KV_HEADS * HEAD_DIM * 2;
  bufs_->kv_cache = dev_.create_device_local_buffer(
      bufs_->kv_cache_layer_bytes * NUM_ATTN_LAYERS);
  // Zero-init KV cache
  {
    std::vector<uint8_t> zeros(bufs_->kv_cache_layer_bytes * NUM_ATTN_LAYERS, 0);
    upload_raw(dev_, bufs_->kv_cache, zeros.data(), zeros.size());
  }

  // RoPE frequency table: precomputed cos/sin for all MAX_SEQ positions
  // Each position stores ROTARY_DIM floats (cos0,sin0,cos1,sin1,...)
  bufs_->rope_freq = dev_.create_device_local_buffer(MAX_SEQ * ROTARY_DIM * 4);
  {
    constexpr float theta = 10000000.0f;
    constexpr uint32_t rotary_pairs = ROTARY_DIM / 2;
    std::vector<float> rope_table(MAX_SEQ * ROTARY_DIM);
    for (uint32_t pos = 0; pos < MAX_SEQ; ++pos) {
      for (uint32_t i = 0; i < rotary_pairs; ++i) {
        float inv_freq = 1.0f / std::pow(theta, 2.0f * static_cast<float>(i) / static_cast<float>(ROTARY_DIM));
        float angle = static_cast<float>(pos) * inv_freq;
        rope_table[pos * ROTARY_DIM + i * 2]     = std::cos(angle);
        rope_table[pos * ROTARY_DIM + i * 2 + 1] = std::sin(angle);
      }
    }
    upload_raw(dev_, bufs_->rope_freq, rope_table.data(), rope_table.size() * 4);
  }

  // DeltaNet buffers
  bufs_->dn_qkv = dev_.create_device_local_buffer(DN_CONV_DIM * 2);
  bufs_->dn_z = dev_.create_device_local_buffer(DN_VAL_TOTAL * 2);
  bufs_->dn_a = dev_.create_device_local_buffer(DN_HEADS * 2);
  bufs_->dn_b = dev_.create_device_local_buffer(DN_HEADS * 2);
  bufs_->dn_q = dev_.create_device_local_buffer(DN_KEY_TOTAL * 2);
  bufs_->dn_kv_out = dev_.create_device_local_buffer(DN_CONV_DIM * 2);

  bufs_->dn_state_per_layer = DN_HEADS * DN_K_DIM * DN_V_DIM * 4 + DN_HEADS * 2 * 4;
  bufs_->dn_state = dev_.create_device_local_buffer(
      bufs_->dn_state_per_layer * NUM_DN_LAYERS);
  {
    std::vector<uint8_t> zeros(bufs_->dn_state_per_layer * NUM_DN_LAYERS, 0);
    upload_raw(dev_, bufs_->dn_state, zeros.data(), zeros.size());
  }

  bufs_->dn_conv_per_layer = DN_CONV_DIM * DN_CONV_KS * 2;
  bufs_->dn_conv_state = dev_.create_device_local_buffer(
      bufs_->dn_conv_per_layer * NUM_DN_LAYERS);
  {
    std::vector<uint8_t> zeros(bufs_->dn_conv_per_layer * NUM_DN_LAYERS, 0);
    upload_raw(dev_, bufs_->dn_conv_state, zeros.data(), zeros.size());
  }

  // --- Chunk-correction buffers ---
  // Snapshots: per-layer pre-norm hidden state for the last prefill token
  prefill_snapshots_ = dev_.create_device_local_buffer(LAYERS * HIDDEN * 2);
  // Per-layer fp16 last-token core_attn_out: GPU handoff buffer.
  // Each layer gets DN_VAL_TOTAL * 2 bytes (fp16).
  dn_chunk_attn_out_ = dev_.create_device_local_buffer(NUM_DN_LAYERS * DN_VAL_TOTAL * 2);
  chunk_core_attn_out_last_.resize(NUM_DN_LAYERS);

  // Upload weights
  auto total_size = artifact_.total_bytes();
  bufs_->weights = dev_.create_device_local_buffer(total_size);
  {
    std::ifstream wf(artifact_.weights_file_path(), std::ios::binary);
    if (!wf) throw std::runtime_error("cannot open weights file: " + artifact_.weights_file_path());
    std::vector<char> weights_data(total_size);
    wf.read(weights_data.data(), total_size);
    upload_raw(dev_, bufs_->weights, weights_data.data(), total_size);
  }

  auto& final_norm_info = artifact_.final_norm();
  auto final_norm_raw = read_tensor_bytes(artifact_, final_norm_info);
  bufs_->final_norm = dev_.create_device_local_buffer(final_norm_info.nbytes);
  upload_raw(dev_, bufs_->final_norm, final_norm_raw.data(), final_norm_info.nbytes);

  // --- Allocate descriptor sets ---
  dsets_ = std::make_unique<DescriptorSets>();
  auto alloc3 = [&]() { return dev_.allocate_descriptor_set(pipes_->ds_layout_3); };
  auto alloc2 = [&]() { return dev_.allocate_descriptor_set(pipes_->ds_layout_2); };

  dsets_->embedding = alloc2();
  dsets_->embedding_from_buffer = alloc3();
  dsets_->input_norm = alloc3();
  dsets_->residual1 = alloc3();
  dsets_->post_norm = alloc3();
  dsets_->gate = alloc3();
  dsets_->up = alloc3();
  dsets_->silu_gate = alloc3();
  dsets_->down = alloc3();
  dsets_->down_f32 = alloc3();
  dsets_->residual2 = alloc3();
  dsets_->mlp_residual_mixed = alloc3();
  dsets_->final_norm = alloc3();
  dsets_->lm_head = alloc3();
  dsets_->argmax = alloc2();

  dsets_->q_proj = alloc3();
  dsets_->k_proj = alloc3();
  dsets_->v_proj = alloc3();
  dsets_->split_q_gate = alloc3();
  dsets_->q_norm = alloc3();
  dsets_->k_norm = alloc3();
  dsets_->rope = alloc3();
  dsets_->rope_k = alloc3();
  dsets_->kv_store = alloc3();
  dsets_->attn = alloc3();
  dsets_->sigmoid_gate = alloc3();
  dsets_->o_proj = alloc3();
  dsets_->o_proj_f32 = alloc3();
  dsets_->attn_residual_mixed = alloc3();

  dsets_->dn_qkv_proj = alloc3();
  dsets_->dn_z_proj = alloc3();
  dsets_->dn_a_proj = alloc3();
  dsets_->dn_b_proj = alloc3();
  dsets_->dn_conv = alloc3();
  dsets_->dn_split_q = alloc3();
  dsets_->dn_split_kv = alloc3();
  dsets_->dn_l2_q = alloc3();
  dsets_->dn_l2_k = alloc3();
  dsets_->dn_recurrent = alloc3();
  dsets_->dn_norm_gate = alloc3();
  dsets_->dn_out_proj = alloc3();
  dsets_->dn_compute_g_beta = dev_.allocate_descriptor_set(pipes_->ds_layout_4);
  dsets_->dn_recurrent_gbeta = dev_.allocate_descriptor_set(pipes_->ds_layout_6);
  dsets_->dn_recurrent_gbeta_norm_gate = dev_.allocate_descriptor_set(pipes_->ds_layout_8);
  dsets_->dn_chunk_prefill = dev_.allocate_descriptor_set(pipes_->ds_layout_7);
  dsets_->dn_prefill_collect = dev_.allocate_descriptor_set(pipes_->ds_layout_7);
  dsets_->dn_chunk_last_to_fp16 = alloc2();

  // Pre-bind RoPE descriptors: full rope_freq table, Q/K buffers
  dev_.update_descriptor_set(dsets_->rope, 0, bufs_->q);
  dev_.update_descriptor_set(dsets_->rope, 1, bufs_->rope_freq, 0,
                             MAX_SEQ * ROTARY_DIM * 4);
  dev_.update_descriptor_set(dsets_->rope, 2, bufs_->q);
  dev_.update_descriptor_set(dsets_->rope_k, 0, bufs_->k);
  dev_.update_descriptor_set(dsets_->rope_k, 1, bufs_->rope_freq, 0,
                             MAX_SEQ * ROTARY_DIM * 4);
  dev_.update_descriptor_set(dsets_->rope_k, 2, bufs_->k);

  // Cache DeltaNet a_log and dt_bias
  const auto& layer_sched = model::Qwen35Config::layer_schedule();
  cached_a_log_.resize(NUM_DN_LAYERS);
  cached_dt_bias_.resize(NUM_DN_LAYERS);
  {
    uint32_t dn_ci = 0;
    for (uint32_t i = 0; i < LAYERS; ++i) {
      if (layer_sched[i] == model::LayerKind::FullAttention) continue;
      auto ai = artifact_.find_by_role("layer." + std::to_string(i) + ".delta_a_log");
      auto di = artifact_.find_by_role("layer." + std::to_string(i) + ".delta_dt_bias");
      if (!ai || !di) { ++dn_ci; continue; }
      auto a_raw = read_tensor_bytes(artifact_, *ai);
      auto d_raw = read_tensor_bytes(artifact_, *di);
      cached_a_log_[dn_ci].resize(DN_HEADS);
      cached_dt_bias_[dn_ci].resize(DN_HEADS);
      memcpy(cached_a_log_[dn_ci].data(), a_raw.data(), DN_HEADS * 4);
      const uint16_t* dt_f16 = reinterpret_cast<const uint16_t*>(d_raw.data());
      for (uint32_t h = 0; h < DN_HEADS; ++h)
        cached_dt_bias_[dn_ci][h] = half_to_float(dt_f16[h]);
      ++dn_ci;
    }
  }

  // Upload a_log and dt_bias buffer for GPU-side g/beta computation
  {
    std::vector<float> ab_data(NUM_DN_LAYERS * DN_HEADS * 2);
    for (uint32_t l = 0; l < NUM_DN_LAYERS; ++l) {
      for (uint32_t h = 0; h < DN_HEADS; ++h) {
        ab_data[(l * DN_HEADS + h) * 2 + 0] = cached_a_log_[l][h];
        ab_data[(l * DN_HEADS + h) * 2 + 1] = cached_dt_bias_[l][h];
      }
    }
    bufs_->dn_a_log_bias = dev_.create_device_local_buffer(ab_data.size() * 4);
    upload_raw(dev_, bufs_->dn_a_log_bias, ab_data.data(), ab_data.size() * 4);
  }

  // --- Per-layer stable descriptor sets (SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS) ---
  // Eliminates per-layer descriptor mutation in decode() by pre-binding
  // weight offsets and per-layer buffer offsets at construction time.
  // RoPE descriptors (D.rope, D.rope_k) are pre-bound once at construction time;
  // L2-norm q/k (dn_l2_q, dn_l2_k), dn_recurrent, dn_norm_gate, and dn_out_proj are covered here;
  // All DeltaNet dispatch-target sub-step descriptors are covered under the per-layer descriptor gate;
  // dn_split_q and dn_split_kv remain internal decomposition descriptors, not part of this per-dispatch-target list.
  per_layer_sets_enabled_ = []() {
    const char* e = std::getenv("SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS");
    return e && e[0] == '1' && e[1] == '\0';
  }();
  if (per_layer_sets_enabled_) {
    per_layer_sets_ = std::make_unique<PerLayerDescriptorSets>();
    const auto& schedule = model::Qwen35Config::layer_schedule();
    auto alloc3 = [&]() { return dev_.allocate_descriptor_set(pipes_->ds_layout_3); };
    // Allocate LAYERS copies of each mutable descriptor set
    per_layer_sets_->input_norm.resize(LAYERS);
    per_layer_sets_->residual1.resize(LAYERS);
    per_layer_sets_->post_norm.resize(LAYERS);
    per_layer_sets_->gate.resize(LAYERS);
    per_layer_sets_->up.resize(LAYERS);
    per_layer_sets_->down.resize(LAYERS);
    per_layer_sets_->down_f32.resize(LAYERS);
    per_layer_sets_->residual2.resize(LAYERS);
    per_layer_sets_->mlp_residual_mixed.resize(LAYERS);
    per_layer_sets_->q_proj.resize(LAYERS);
    per_layer_sets_->k_proj.resize(LAYERS);
    per_layer_sets_->v_proj.resize(LAYERS);
    per_layer_sets_->q_norm.resize(LAYERS);
    per_layer_sets_->k_norm.resize(LAYERS);
    per_layer_sets_->kv_store.resize(LAYERS);
    per_layer_sets_->attn.resize(LAYERS);
    per_layer_sets_->o_proj.resize(LAYERS);
    per_layer_sets_->o_proj_f32.resize(LAYERS);
    per_layer_sets_->attn_residual_mixed.resize(LAYERS);
    per_layer_sets_->dn_qkv_proj.resize(LAYERS);
    per_layer_sets_->dn_z_proj.resize(LAYERS);
    per_layer_sets_->dn_a_proj.resize(LAYERS);
    per_layer_sets_->dn_b_proj.resize(LAYERS);
    per_layer_sets_->dn_conv.resize(LAYERS);
    per_layer_sets_->dn_l2_q.resize(LAYERS);
    per_layer_sets_->dn_l2_k.resize(LAYERS);
    per_layer_sets_->dn_recurrent.resize(LAYERS);
    per_layer_sets_->dn_norm_gate.resize(LAYERS);
    per_layer_sets_->dn_out_proj.resize(LAYERS);
    per_layer_sets_->dn_compute_g_beta.resize(LAYERS);
    per_layer_sets_->dn_recurrent_gbeta.resize(LAYERS);
    per_layer_sets_->dn_recurrent_gbeta_norm_gate.resize(LAYERS);
    for (uint32_t i = 0; i < LAYERS; ++i) {
      per_layer_sets_->input_norm[i] = alloc3();
      per_layer_sets_->residual1[i] = alloc3();
      per_layer_sets_->post_norm[i] = alloc3();
      per_layer_sets_->gate[i] = alloc3();
      per_layer_sets_->up[i] = alloc3();
      per_layer_sets_->down[i] = alloc3();
      per_layer_sets_->down_f32[i] = alloc3();
      per_layer_sets_->residual2[i] = alloc3();
      per_layer_sets_->mlp_residual_mixed[i] = alloc3();
      per_layer_sets_->q_proj[i] = alloc3();
      per_layer_sets_->k_proj[i] = alloc3();
      per_layer_sets_->v_proj[i] = alloc3();
      per_layer_sets_->q_norm[i] = alloc3();
      per_layer_sets_->k_norm[i] = alloc3();
      per_layer_sets_->kv_store[i] = alloc3();
      per_layer_sets_->attn[i] = alloc3();
      per_layer_sets_->o_proj[i] = alloc3();
      per_layer_sets_->o_proj_f32[i] = alloc3();
      per_layer_sets_->attn_residual_mixed[i] = alloc3();
      per_layer_sets_->dn_qkv_proj[i] = alloc3();
      per_layer_sets_->dn_z_proj[i] = alloc3();
      per_layer_sets_->dn_a_proj[i] = alloc3();
      per_layer_sets_->dn_b_proj[i] = alloc3();
      per_layer_sets_->dn_conv[i] = alloc3();
      per_layer_sets_->dn_l2_q[i] = alloc3();
      per_layer_sets_->dn_l2_k[i] = alloc3();
      per_layer_sets_->dn_recurrent[i] = alloc3();
      per_layer_sets_->dn_norm_gate[i] = alloc3();
      per_layer_sets_->dn_out_proj[i] = alloc3();
      per_layer_sets_->dn_compute_g_beta[i] = dev_.allocate_descriptor_set(pipes_->ds_layout_4);
      per_layer_sets_->dn_recurrent_gbeta[i] = dev_.allocate_descriptor_set(pipes_->ds_layout_6);
      per_layer_sets_->dn_recurrent_gbeta_norm_gate[i] = dev_.allocate_descriptor_set(pipes_->ds_layout_8);
    }
    // Pre-bind per-layer weight offsets and static buffer references
    auto attn_layer_idx = [](uint32_t layer) -> uint32_t {
      return (layer + 1) / 4 - 1;
    };
    for (uint32_t layer = 0; layer < LAYERS; ++layer) {
      bool is_attn = (schedule[layer] == model::LayerKind::FullAttention);
      uint32_t attn_idx = is_attn ? attn_layer_idx(layer) : 0;
      // Common MLP/norm weight lookups (all layers)
      auto input_norm_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".input_norm");
      auto post_norm_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".post_norm");
      auto gate_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".mlp_gate");
      auto up_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".mlp_up");
      auto down_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".mlp_down");
      assert(input_norm_w && post_norm_w && gate_w && up_w && down_w);
      auto& B = *bufs_;
      // input_norm: binding 0=act_a, 1=weight, 2=act_b
      dev_.update_descriptor_set(per_layer_sets_->input_norm[layer], 0, B.act_a);
      dev_.update_descriptor_set(per_layer_sets_->input_norm[layer], 1, B.weights, input_norm_w->offset, input_norm_w->nbytes);
      dev_.update_descriptor_set(per_layer_sets_->input_norm[layer], 2, B.act_b);
      // residual1: 0=act_a, 1=act_b, 2=act_c
      dev_.update_descriptor_set(per_layer_sets_->residual1[layer], 0, B.act_a);
      dev_.update_descriptor_set(per_layer_sets_->residual1[layer], 1, B.act_b);
      dev_.update_descriptor_set(per_layer_sets_->residual1[layer], 2, B.act_c);
      // post_norm: 0=act_c, 1=weight, 2=act_a
      dev_.update_descriptor_set(per_layer_sets_->post_norm[layer], 0, B.act_c);
      dev_.update_descriptor_set(per_layer_sets_->post_norm[layer], 1, B.weights, post_norm_w->offset, post_norm_w->nbytes);
      dev_.update_descriptor_set(per_layer_sets_->post_norm[layer], 2, B.act_a);
      // gate: 0=weight, 1=act_a, 2=mlp_gate
      dev_.update_descriptor_set(per_layer_sets_->gate[layer], 0, B.weights, gate_w->offset, gate_w->nbytes);
      dev_.update_descriptor_set(per_layer_sets_->gate[layer], 1, B.act_a);
      dev_.update_descriptor_set(per_layer_sets_->gate[layer], 2, B.mlp_gate);
      // up: 0=weight, 1=act_a, 2=mlp_up
      dev_.update_descriptor_set(per_layer_sets_->up[layer], 0, B.weights, up_w->offset, up_w->nbytes);
      dev_.update_descriptor_set(per_layer_sets_->up[layer], 1, B.act_a);
      dev_.update_descriptor_set(per_layer_sets_->up[layer], 2, B.mlp_up);
      // down: 0=weight, 1=mlp_silu, 2=act_b
      dev_.update_descriptor_set(per_layer_sets_->down[layer], 0, B.weights, down_w->offset, down_w->nbytes);
      dev_.update_descriptor_set(per_layer_sets_->down[layer], 1, B.mlp_silu);
      dev_.update_descriptor_set(per_layer_sets_->down[layer], 2, B.act_b);
      // down_f32: 0=weight, 1=mlp_silu, 2=attn_proj_f32
      dev_.update_descriptor_set(per_layer_sets_->down_f32[layer], 0, B.weights, down_w->offset, down_w->nbytes);
      dev_.update_descriptor_set(per_layer_sets_->down_f32[layer], 1, B.mlp_silu);
      dev_.update_descriptor_set(per_layer_sets_->down_f32[layer], 2, B.attn_proj_f32);
      // residual2: 0=act_c, 1=act_b, 2=act_a
      dev_.update_descriptor_set(per_layer_sets_->residual2[layer], 0, B.act_c);
      dev_.update_descriptor_set(per_layer_sets_->residual2[layer], 1, B.act_b);
      dev_.update_descriptor_set(per_layer_sets_->residual2[layer], 2, B.act_a);
      // mlp_residual_mixed: 0=attn_proj_f32, 1=act_c, 2=act_a
      dev_.update_descriptor_set(per_layer_sets_->mlp_residual_mixed[layer], 0, B.attn_proj_f32);
      dev_.update_descriptor_set(per_layer_sets_->mlp_residual_mixed[layer], 1, B.act_c);
      dev_.update_descriptor_set(per_layer_sets_->mlp_residual_mixed[layer], 2, B.act_a);
      // Attention-specific — pre-bind for all layers but only bound when is_attn
      if (is_attn) {
        auto attn_q_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".attn_q");
        auto attn_k_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".attn_k");
        auto attn_v_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".attn_v");
        auto attn_o_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".attn_o");
        auto attn_q_norm_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".attn_q_norm");
        auto attn_k_norm_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".attn_k_norm");
        assert(attn_q_w && attn_k_w && attn_v_w && attn_o_w && attn_q_norm_w && attn_k_norm_w);
        uint32_t kv_layer_offset = attn_idx * MAX_SEQ * 2 * KV_HEADS * HEAD_DIM * 2;
        // q_proj
        dev_.update_descriptor_set(per_layer_sets_->q_proj[layer], 0, B.weights, attn_q_w->offset, attn_q_w->nbytes);
        dev_.update_descriptor_set(per_layer_sets_->q_proj[layer], 1, B.act_b);
        dev_.update_descriptor_set(per_layer_sets_->q_proj[layer], 2, B.q_proj);
        // k_proj
        dev_.update_descriptor_set(per_layer_sets_->k_proj[layer], 0, B.weights, attn_k_w->offset, attn_k_w->nbytes);
        dev_.update_descriptor_set(per_layer_sets_->k_proj[layer], 1, B.act_b);
        dev_.update_descriptor_set(per_layer_sets_->k_proj[layer], 2, B.k);
        // v_proj
        dev_.update_descriptor_set(per_layer_sets_->v_proj[layer], 0, B.weights, attn_v_w->offset, attn_v_w->nbytes);
        dev_.update_descriptor_set(per_layer_sets_->v_proj[layer], 1, B.act_b);
        dev_.update_descriptor_set(per_layer_sets_->v_proj[layer], 2, B.v);
        // q_norm
        dev_.update_descriptor_set(per_layer_sets_->q_norm[layer], 0, B.q);
        dev_.update_descriptor_set(per_layer_sets_->q_norm[layer], 1, B.weights, attn_q_norm_w->offset, attn_q_norm_w->nbytes);
        dev_.update_descriptor_set(per_layer_sets_->q_norm[layer], 2, B.q);
        // k_norm
        dev_.update_descriptor_set(per_layer_sets_->k_norm[layer], 0, B.k);
        dev_.update_descriptor_set(per_layer_sets_->k_norm[layer], 1, B.weights, attn_k_norm_w->offset, attn_k_norm_w->nbytes);
        dev_.update_descriptor_set(per_layer_sets_->k_norm[layer], 2, B.k);
        // kv_store: 0=k, 1=v, 2=kv_cache at layer offset
        dev_.update_descriptor_set(per_layer_sets_->kv_store[layer], 0, B.k);
        dev_.update_descriptor_set(per_layer_sets_->kv_store[layer], 1, B.v);
        dev_.update_descriptor_set(per_layer_sets_->kv_store[layer], 2, B.kv_cache, kv_layer_offset);
        // attn: 0=q, 1=kv_cache, 2=attn_out
        dev_.update_descriptor_set(per_layer_sets_->attn[layer], 0, B.q);
        dev_.update_descriptor_set(per_layer_sets_->attn[layer], 1, B.kv_cache, kv_layer_offset);
        dev_.update_descriptor_set(per_layer_sets_->attn[layer], 2, B.attn_out);
        // o_proj
        dev_.update_descriptor_set(per_layer_sets_->o_proj[layer], 0, B.weights, attn_o_w->offset, attn_o_w->nbytes);
        dev_.update_descriptor_set(per_layer_sets_->o_proj[layer], 1, B.gated_attn);
        dev_.update_descriptor_set(per_layer_sets_->o_proj[layer], 2, B.act_b);
        // o_proj_f32
        dev_.update_descriptor_set(per_layer_sets_->o_proj_f32[layer], 0, B.weights, attn_o_w->offset, attn_o_w->nbytes);
        dev_.update_descriptor_set(per_layer_sets_->o_proj_f32[layer], 1, B.gated_attn);
        dev_.update_descriptor_set(per_layer_sets_->o_proj_f32[layer], 2, B.attn_proj_f32);
        // attn_residual_mixed
        dev_.update_descriptor_set(per_layer_sets_->attn_residual_mixed[layer], 0, B.attn_proj_f32);
        dev_.update_descriptor_set(per_layer_sets_->attn_residual_mixed[layer], 1, B.act_a);
        dev_.update_descriptor_set(per_layer_sets_->attn_residual_mixed[layer], 2, B.act_c);
      }
      // DeltaNet-specific — pre-bind for all layers but only bound when !is_attn
      if (!is_attn) {
        uint32_t dn_idx = 0;
        for (uint32_t j = 0; j < layer; ++j) {
          if (schedule[j] != model::LayerKind::FullAttention) ++dn_idx;
        }
        auto dn_qkv_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".delta_in_proj_qkv");
        auto dn_z_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".delta_in_proj_z");
        auto dn_a_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".delta_in_proj_a");
        auto dn_b_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".delta_in_proj_b");
        auto dn_conv_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".delta_conv");
        auto dn_norm_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".delta_norm");
        auto dn_out_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".delta_out_proj");
        assert(dn_qkv_w && dn_z_w && dn_a_w && dn_b_w && dn_conv_w && dn_norm_w && dn_out_w);
        uint32_t conv_state_offset = dn_idx * DN_CONV_DIM * DN_CONV_KS * 2;
        // dn_qkv_proj
        dev_.update_descriptor_set(per_layer_sets_->dn_qkv_proj[layer], 0, B.weights, dn_qkv_w->offset, dn_qkv_w->nbytes);
        dev_.update_descriptor_set(per_layer_sets_->dn_qkv_proj[layer], 1, B.act_b);
        dev_.update_descriptor_set(per_layer_sets_->dn_qkv_proj[layer], 2, B.dn_qkv);
        // dn_z_proj
        dev_.update_descriptor_set(per_layer_sets_->dn_z_proj[layer], 0, B.weights, dn_z_w->offset, dn_z_w->nbytes);
        dev_.update_descriptor_set(per_layer_sets_->dn_z_proj[layer], 1, B.act_b);
        dev_.update_descriptor_set(per_layer_sets_->dn_z_proj[layer], 2, B.dn_z);
        // dn_a_proj
        dev_.update_descriptor_set(per_layer_sets_->dn_a_proj[layer], 0, B.weights, dn_a_w->offset, dn_a_w->nbytes);
        dev_.update_descriptor_set(per_layer_sets_->dn_a_proj[layer], 1, B.act_b);
        dev_.update_descriptor_set(per_layer_sets_->dn_a_proj[layer], 2, B.dn_a);
        // dn_b_proj
        dev_.update_descriptor_set(per_layer_sets_->dn_b_proj[layer], 0, B.weights, dn_b_w->offset, dn_b_w->nbytes);
        dev_.update_descriptor_set(per_layer_sets_->dn_b_proj[layer], 1, B.act_b);
        dev_.update_descriptor_set(per_layer_sets_->dn_b_proj[layer], 2, B.dn_b);
        // dn_conv: 0=dn_qkv, 1=dn_conv_state, 2=weight
        dev_.update_descriptor_set(per_layer_sets_->dn_conv[layer], 0, B.dn_qkv);
        dev_.update_descriptor_set(per_layer_sets_->dn_conv[layer], 1, B.dn_conv_state, conv_state_offset);
        dev_.update_descriptor_set(per_layer_sets_->dn_conv[layer], 2, B.weights, dn_conv_w->offset, dn_conv_w->nbytes);
        // dn_l2_q: Q region of dn_qkv (all 3 bindings at offset 0, size DN_KEY_TOTAL*2)
        dev_.update_descriptor_set(per_layer_sets_->dn_l2_q[layer], 0, B.dn_qkv, 0, DN_KEY_TOTAL * 2);
        dev_.update_descriptor_set(per_layer_sets_->dn_l2_q[layer], 1, B.dn_qkv, 0, DN_KEY_TOTAL * 2);
        dev_.update_descriptor_set(per_layer_sets_->dn_l2_q[layer], 2, B.dn_qkv, 0, DN_KEY_TOTAL * 2);
        // dn_l2_k: K region of dn_qkv (all 3 bindings at offset DN_KEY_TOTAL*2, size DN_KEY_TOTAL*2)
        dev_.update_descriptor_set(per_layer_sets_->dn_l2_k[layer], 0, B.dn_qkv, DN_KEY_TOTAL * 2, DN_KEY_TOTAL * 2);
        dev_.update_descriptor_set(per_layer_sets_->dn_l2_k[layer], 1, B.dn_qkv, DN_KEY_TOTAL * 2, DN_KEY_TOTAL * 2);
        dev_.update_descriptor_set(per_layer_sets_->dn_l2_k[layer], 2, B.dn_qkv, DN_KEY_TOTAL * 2, DN_KEY_TOTAL * 2);
        // dn_recurrent: bindings 0=Q(dn_qkv), 1=KV(dn_qkv), 2=dn_state[layer]
        VkDeviceSize rec_state_off = dn_idx * B.dn_state_per_layer;
        dev_.update_descriptor_set(per_layer_sets_->dn_recurrent[layer], 0, B.dn_qkv, 0, DN_KEY_TOTAL * 2);
        dev_.update_descriptor_set(per_layer_sets_->dn_recurrent[layer], 1, B.dn_qkv, DN_KEY_TOTAL * 2,
            (DN_KEY_TOTAL + DN_VAL_TOTAL) * 2);
        dev_.update_descriptor_set(per_layer_sets_->dn_recurrent[layer], 2, B.dn_state, rec_state_off);
        // dn_norm_gate: bindings 0=dn_qkv(V section) io, 1=dn_z gate, 2=delta_norm weight
        dev_.update_descriptor_set(per_layer_sets_->dn_norm_gate[layer], 0, B.dn_qkv, (DN_KEY_TOTAL + DN_KEY_TOTAL) * 2, DN_VAL_TOTAL * 2);
        dev_.update_descriptor_set(per_layer_sets_->dn_norm_gate[layer], 1, B.dn_z);
        dev_.update_descriptor_set(per_layer_sets_->dn_norm_gate[layer], 2, B.weights, dn_norm_w->offset, dn_norm_w->nbytes);
        // dn_out_proj: bindings 0=weight(delta_out_proj), 1=dn_qkv(V section), 2=act_b
        dev_.update_descriptor_set(per_layer_sets_->dn_out_proj[layer], 0, B.weights, dn_out_w->offset, dn_out_w->nbytes);
        dev_.update_descriptor_set(per_layer_sets_->dn_out_proj[layer], 1, B.dn_qkv, (DN_KEY_TOTAL + DN_KEY_TOTAL) * 2, DN_VAL_TOTAL * 2);
        dev_.update_descriptor_set(per_layer_sets_->dn_out_proj[layer], 2, B.act_b);
        // dn_compute_g_beta: bindings 0=dn_a, 1=dn_b, 2=dn_a_log_bias, 3=dn_state[layer]
        VkDeviceSize g_beta_state_off = dn_idx * B.dn_state_per_layer + DN_HEADS * DN_K_DIM * DN_V_DIM * 4;
        dev_.update_descriptor_set(per_layer_sets_->dn_compute_g_beta[layer], 0, B.dn_a, 0, DN_HEADS * 2);
        dev_.update_descriptor_set(per_layer_sets_->dn_compute_g_beta[layer], 1, B.dn_b, 0, DN_HEADS * 2);
        dev_.update_descriptor_set(per_layer_sets_->dn_compute_g_beta[layer], 2, B.dn_a_log_bias, 0,
            NUM_DN_LAYERS * DN_HEADS * 2 * 4);
        dev_.update_descriptor_set(per_layer_sets_->dn_compute_g_beta[layer], 3, B.dn_state, g_beta_state_off, DN_HEADS * 2 * 4);
        // dn_recurrent_gbeta (fused): bindings 0=dn_a, 1=dn_b, 2=dn_a_log_bias, 3=Q(dn_qkv), 4=KV(dn_qkv), 5=dn_state[layer]
        dev_.update_descriptor_set(per_layer_sets_->dn_recurrent_gbeta[layer], 0, B.dn_a, 0, DN_HEADS * 2);
        dev_.update_descriptor_set(per_layer_sets_->dn_recurrent_gbeta[layer], 1, B.dn_b, 0, DN_HEADS * 2);
        dev_.update_descriptor_set(per_layer_sets_->dn_recurrent_gbeta[layer], 2, B.dn_a_log_bias, 0,
            NUM_DN_LAYERS * DN_HEADS * 2 * 4);
        dev_.update_descriptor_set(per_layer_sets_->dn_recurrent_gbeta[layer], 3, B.dn_qkv, 0, DN_KEY_TOTAL * 2);
        dev_.update_descriptor_set(per_layer_sets_->dn_recurrent_gbeta[layer], 4, B.dn_qkv, DN_KEY_TOTAL * 2,
            (DN_KEY_TOTAL + DN_VAL_TOTAL) * 2);
        dev_.update_descriptor_set(per_layer_sets_->dn_recurrent_gbeta[layer], 5, B.dn_state, rec_state_off);
        // dn_recurrent_gbeta_norm_gate (fused): recurrent_gbeta + norm_gate
        dev_.update_descriptor_set(per_layer_sets_->dn_recurrent_gbeta_norm_gate[layer], 0, B.dn_a, 0, DN_HEADS * 2);
        dev_.update_descriptor_set(per_layer_sets_->dn_recurrent_gbeta_norm_gate[layer], 1, B.dn_b, 0, DN_HEADS * 2);
        dev_.update_descriptor_set(per_layer_sets_->dn_recurrent_gbeta_norm_gate[layer], 2, B.dn_a_log_bias, 0,
            NUM_DN_LAYERS * DN_HEADS * 2 * 4);
        dev_.update_descriptor_set(per_layer_sets_->dn_recurrent_gbeta_norm_gate[layer], 3, B.dn_qkv, 0, DN_KEY_TOTAL * 2);
        dev_.update_descriptor_set(per_layer_sets_->dn_recurrent_gbeta_norm_gate[layer], 4, B.dn_qkv, DN_KEY_TOTAL * 2,
            (DN_KEY_TOTAL + DN_VAL_TOTAL) * 2);
        dev_.update_descriptor_set(per_layer_sets_->dn_recurrent_gbeta_norm_gate[layer], 5, B.dn_state, rec_state_off);
        dev_.update_descriptor_set(per_layer_sets_->dn_recurrent_gbeta_norm_gate[layer], 6, B.dn_z, 0, DN_VAL_TOTAL * 2);
        dev_.update_descriptor_set(per_layer_sets_->dn_recurrent_gbeta_norm_gate[layer], 7, B.weights, dn_norm_w->offset, dn_norm_w->nbytes);
      }
    }
    if (verbose_) {
      std::cerr << "  per-layer descriptor sets: " << LAYERS << " x 32 sets pre-bound\n";
    }
  }

  // Pre-configure static descriptor sets
  auto& w = bufs_->weights;
  auto& a = bufs_->act_a;
  auto& b = bufs_->act_b;

  dev_.update_descriptor_set(dsets_->embedding, 0, w,
      artifact_.token_embedding().offset, artifact_.token_embedding().nbytes);
  dev_.update_descriptor_set(dsets_->embedding, 1, a);

  dev_.update_descriptor_set(dsets_->embedding_from_buffer, 0, bufs_->argmax_result);
  dev_.update_descriptor_set(dsets_->embedding_from_buffer, 1, w,
      artifact_.token_embedding().offset, artifact_.token_embedding().nbytes);
  dev_.update_descriptor_set(dsets_->embedding_from_buffer, 2, a);

  dev_.update_descriptor_set(dsets_->silu_gate, 0, bufs_->mlp_gate);
  dev_.update_descriptor_set(dsets_->silu_gate, 1, bufs_->mlp_up);
  dev_.update_descriptor_set(dsets_->silu_gate, 2, bufs_->mlp_silu);

  dev_.update_descriptor_set(dsets_->final_norm, 0, a);
  dev_.update_descriptor_set(dsets_->final_norm, 1, bufs_->final_norm);
  dev_.update_descriptor_set(dsets_->final_norm, 2, b);

  dev_.update_descriptor_set(dsets_->lm_head, 0, w,
      artifact_.token_embedding().offset, artifact_.token_embedding().nbytes);
  dev_.update_descriptor_set(dsets_->lm_head, 1, b);
  dev_.update_descriptor_set(dsets_->lm_head, 2, bufs_->logits);

  dev_.update_descriptor_set(dsets_->argmax, 0, bufs_->logits);
  dev_.update_descriptor_set(dsets_->argmax, 1, bufs_->argmax_result);

  dev_.update_descriptor_set(dsets_->split_q_gate, 0, bufs_->q_proj);
  dev_.update_descriptor_set(dsets_->split_q_gate, 1, bufs_->q);
  dev_.update_descriptor_set(dsets_->split_q_gate, 2, bufs_->gate);

  dev_.update_descriptor_set(dsets_->sigmoid_gate, 0, bufs_->attn_out);
  dev_.update_descriptor_set(dsets_->sigmoid_gate, 1, bufs_->gate);
  dev_.update_descriptor_set(dsets_->sigmoid_gate, 2, bufs_->gated_attn);

  // Pre-configure g/beta compute descriptor set (bindings 0,1,2 static)
  dev_.update_descriptor_set(dsets_->dn_compute_g_beta, 0, bufs_->dn_a, 0, DN_HEADS * 2);
  dev_.update_descriptor_set(dsets_->dn_compute_g_beta, 1, bufs_->dn_b, 0, DN_HEADS * 2);
  dev_.update_descriptor_set(dsets_->dn_compute_g_beta, 2, bufs_->dn_a_log_bias, 0,
      NUM_DN_LAYERS * DN_HEADS * 2 * 4);

  // Pre-configure fused g/beta+recurrent descriptor set (bindings 0,1,2,3,4,5 all static
  // except state offset changes per-layer — same pattern as dn_recurrent).
  // Bindings 0-2: same as dn_compute_g_beta (dn_a, dn_b, dn_a_log_bias).
  // Bindings 3-4: Q and KV slices of dn_qkv (same offsets as dn_recurrent).
  // Binding 5: dn_state — updated per-layer like dn_recurrent.
  dev_.update_descriptor_set(dsets_->dn_recurrent_gbeta, 0, bufs_->dn_a, 0, DN_HEADS * 2);
  dev_.update_descriptor_set(dsets_->dn_recurrent_gbeta, 1, bufs_->dn_b, 0, DN_HEADS * 2);
  dev_.update_descriptor_set(dsets_->dn_recurrent_gbeta, 2, bufs_->dn_a_log_bias, 0,
      NUM_DN_LAYERS * DN_HEADS * 2 * 4);
  dev_.update_descriptor_set(dsets_->dn_recurrent_gbeta, 3, bufs_->dn_qkv, 0, DN_KEY_TOTAL * 2);
  dev_.update_descriptor_set(dsets_->dn_recurrent_gbeta, 4, bufs_->dn_qkv, DN_KEY_TOTAL * 2,
      (DN_KEY_TOTAL + DN_VAL_TOTAL) * 2);
  // Binding 5 (dn_state) is updated per-layer in decode() for non-per-layer mode.
  dev_.update_descriptor_set(dsets_->dn_recurrent_gbeta_norm_gate, 0, bufs_->dn_a, 0, DN_HEADS * 2);
  dev_.update_descriptor_set(dsets_->dn_recurrent_gbeta_norm_gate, 1, bufs_->dn_b, 0, DN_HEADS * 2);
  dev_.update_descriptor_set(dsets_->dn_recurrent_gbeta_norm_gate, 2, bufs_->dn_a_log_bias, 0,
      NUM_DN_LAYERS * DN_HEADS * 2 * 4);
  dev_.update_descriptor_set(dsets_->dn_recurrent_gbeta_norm_gate, 3, bufs_->dn_qkv, 0, DN_KEY_TOTAL * 2);
  dev_.update_descriptor_set(dsets_->dn_recurrent_gbeta_norm_gate, 4, bufs_->dn_qkv, DN_KEY_TOTAL * 2,
      (DN_KEY_TOTAL + DN_VAL_TOTAL) * 2);
  dev_.update_descriptor_set(dsets_->dn_recurrent_gbeta_norm_gate, 6, bufs_->dn_z, 0, DN_VAL_TOTAL * 2);
  // Bindings 5 (dn_state) and 7 (delta_norm weight) are updated per-layer
  // in decode() for non-per-layer mode.
  gpu_chunk_handoff_ready_.resize(NUM_DN_LAYERS, false);

}
DecodeSession::~DecodeSession() {
  auto& p = *pipes_;
  auto& b = *bufs_;
  auto& d = *dsets_;
  (void)d;  // descriptor sets freed implicitly when pool is destroyed

  // Buffers
  dev_.destroy_buffer(b.act_a);
  dev_.destroy_buffer(b.act_b);
  dev_.destroy_buffer(b.act_c);
  dev_.destroy_buffer(b.logits);
  dev_.destroy_buffer(b.argmax_result);
  dev_.destroy_buffer(b.mlp_gate);
  dev_.destroy_buffer(b.mlp_up);
  dev_.destroy_buffer(b.mlp_silu);
  dev_.destroy_buffer(b.q_proj);
  dev_.destroy_buffer(b.q);
  dev_.destroy_buffer(b.gate);
  dev_.destroy_buffer(b.k);
  dev_.destroy_buffer(b.v);
  dev_.destroy_buffer(b.attn_out);
  dev_.destroy_buffer(b.gated_attn);
  dev_.destroy_buffer(b.attn_proj_f32);
  dev_.destroy_buffer(b.kv_cache);
  dev_.destroy_buffer(b.rope_freq);
  dev_.destroy_buffer(b.dn_qkv);
  dev_.destroy_buffer(b.dn_z);
  dev_.destroy_buffer(b.dn_a);
  dev_.destroy_buffer(b.dn_b);
  dev_.destroy_buffer(b.dn_q);
  dev_.destroy_buffer(b.dn_kv_out);
  dev_.destroy_buffer(b.dn_state);
  dev_.destroy_buffer(b.dn_conv_state);
  dev_.destroy_buffer(b.dn_a_log_bias);
  dev_.destroy_buffer(b.weights);
  dev_.destroy_buffer(b.final_norm);
  dev_.destroy_buffer(prefill_snapshots_);
  dev_.destroy_buffer(dn_chunk_attn_out_);

  // Diagnostic prefill collection buffers
  if (b.collect_bufs_allocated_) {
    dev_.destroy_buffer(b.dn_collect_q);
    dev_.destroy_buffer(b.dn_collect_k);
    dev_.destroy_buffer(b.dn_collect_v);
    dev_.destroy_buffer(b.dn_collect_g);
    dev_.destroy_buffer(b.dn_collect_beta);
  }

  // Persistent GPU prefill collection buffers
  if (b.persist_bufs_allocated_) {
    dev_.destroy_buffer(b.dn_persist_q);
    dev_.destroy_buffer(b.dn_persist_k);
    dev_.destroy_buffer(b.dn_persist_v);
    dev_.destroy_buffer(b.dn_persist_g);
    dev_.destroy_buffer(b.dn_persist_beta);
  }

  // Pipelines
  dev_.destroy_pipeline(p.embedding);
  dev_.destroy_pipeline(p.embedding_from_buffer);
  dev_.destroy_pipeline(p.rmsnorm);
  dev_.destroy_pipeline(p.matvec);
  dev_.destroy_pipeline(p.matvec_tiled);
  dev_.destroy_pipeline(p.matvec_f32_out);
  dev_.destroy_pipeline(p.argmax);
  dev_.destroy_pipeline(p.silu_gate);
  dev_.destroy_pipeline(p.residual_add);
  dev_.destroy_pipeline(p.residual_add_mixed);
  dev_.destroy_pipeline(p.rope_apply);
  dev_.destroy_pipeline(p.attention_decode);
  dev_.destroy_pipeline(p.kv_cache_store);
  dev_.destroy_pipeline(p.sigmoid_gate);
  dev_.destroy_pipeline(p.rms_norm_per_head);
  dev_.destroy_pipeline(p.split_q_gate);
  dev_.destroy_pipeline(p.deltanet_recurrent);
  dev_.destroy_pipeline(p.conv1d_step);
  dev_.destroy_pipeline(p.deltanet_norm_gate);
  dev_.destroy_pipeline(p.l2_norm_per_head);
  dev_.destroy_pipeline(p.deltanet_compute_g_beta);
  dev_.destroy_pipeline(p.deltanet_chunk_prefill);
  dev_.destroy_pipeline(p.deltanet_chunk_prefill_tiled);
  dev_.destroy_pipeline(p.deltanet_prefill_collect);
  dev_.destroy_pipeline(p.deltanet_chunk_last_to_fp16);
  dev_.destroy_pipeline(p.deltanet_conv_l2_qk);
  dev_.destroy_pipeline(p.deltanet_recurrent_gbeta);
  dev_.destroy_pipeline(p.deltanet_recurrent_gbeta_norm_gate);
  dev_.destroy_pipeline(p.lm_head_tiled);

  // Shader modules
  dev_.destroy_shader_module(p.embedding_module);
  dev_.destroy_shader_module(p.embedding_from_buffer_module);
  dev_.destroy_shader_module(p.rmsnorm_module);
  dev_.destroy_shader_module(p.matvec_module);
  dev_.destroy_shader_module(p.matvec_tiled_module);
  dev_.destroy_shader_module(p.matvec_f32_out_module);
  dev_.destroy_shader_module(p.argmax_module);
  dev_.destroy_shader_module(p.silu_gate_module);
  dev_.destroy_shader_module(p.residual_add_module);
  dev_.destroy_shader_module(p.residual_add_mixed_module);
  dev_.destroy_shader_module(p.rope_apply_module);
  dev_.destroy_shader_module(p.attention_decode_module);
  dev_.destroy_shader_module(p.kv_cache_store_module);
  dev_.destroy_shader_module(p.sigmoid_gate_module);
  dev_.destroy_shader_module(p.rms_norm_per_head_module);
  dev_.destroy_shader_module(p.split_q_gate_module);
  dev_.destroy_shader_module(p.deltanet_recurrent_module);
  dev_.destroy_shader_module(p.conv1d_step_module);
  dev_.destroy_shader_module(p.deltanet_norm_gate_module);
  dev_.destroy_shader_module(p.l2_norm_per_head_module);
  dev_.destroy_shader_module(p.deltanet_compute_g_beta_module);
  dev_.destroy_shader_module(p.deltanet_chunk_prefill_module);
  dev_.destroy_shader_module(p.deltanet_chunk_prefill_tiled_module);
  dev_.destroy_shader_module(p.deltanet_prefill_collect_module);
  dev_.destroy_shader_module(p.deltanet_chunk_last_to_fp16_module);
  dev_.destroy_shader_module(p.deltanet_conv_l2_qk_module);
  dev_.destroy_shader_module(p.deltanet_recurrent_gbeta_module);
  dev_.destroy_shader_module(p.deltanet_recurrent_gbeta_norm_gate_module);
  dev_.destroy_shader_module(p.lm_head_tiled_module);

  // Pipeline layouts and descriptor set layouts
  dev_.destroy_pipeline_layout(p.pipeline_layout_3);
  dev_.destroy_pipeline_layout(p.pipeline_layout_2);
  dev_.destroy_pipeline_layout(p.pipeline_layout_32);
  dev_.destroy_pipeline_layout(p.pipeline_layout_4);
  dev_.destroy_pipeline_layout(p.pipeline_layout_6_32);
  dev_.destroy_pipeline_layout(p.pipeline_layout_8_32);
  dev_.destroy_pipeline_layout(p.pipeline_layout_cp);
  dev_.destroy_descriptor_set_layout(p.ds_layout_3);
  dev_.destroy_descriptor_set_layout(p.ds_layout_2);
  dev_.destroy_descriptor_set_layout(p.ds_layout_4);
  dev_.destroy_descriptor_set_layout(p.ds_layout_6);
  dev_.destroy_descriptor_set_layout(p.ds_layout_8);
  dev_.destroy_descriptor_set_layout(p.ds_layout_7);

  dev_.destroy();
}

void DecodeSession::reset() {
  // Zero KV cache
  {
    std::vector<uint8_t> zeros(bufs_->kv_cache_layer_bytes * NUM_ATTN_LAYERS, 0);
    upload_raw(dev_, bufs_->kv_cache, zeros.data(), zeros.size());
  }
  // Zero DeltaNet state
  {
    std::vector<uint8_t> zeros(bufs_->dn_state_per_layer * NUM_DN_LAYERS, 0);
    upload_raw(dev_, bufs_->dn_state, zeros.data(), zeros.size());
  }
  // Zero conv state
  {
    std::vector<uint8_t> zeros(bufs_->dn_conv_per_layer * NUM_DN_LAYERS, 0);
    upload_raw(dev_, bufs_->dn_conv_state, zeros.data(), zeros.size());
  }
}


// ---------------------------------------------------------------------------
// layer_major_prefill()
// ---------------------------------------------------------------------------

void DecodeSession::layer_major_prefill(
    const std::vector<uint32_t>& tokens, uint32_t prompt_len, bool verbose) {
  const auto& P = *pipes_;
  const auto& B = *bufs_;
  const auto& D = *dsets_;
  const auto& schedule = model::Qwen35Config::layer_schedule();

  constexpr float RMS_EPS = 1e-6f;
  constexpr float ATTN_SCALE = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
  static const float DN_Q_SCALE = 1.0f / std::sqrt(static_cast<float>(DN_K_DIM));

  auto barrier = [&](VkCommandBuffer cmd_buf, VkBuffer buf, VkDeviceSize size,
                     VkDeviceSize offset = 0) {
    VkBufferMemoryBarrier bmb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    bmb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bmb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.buffer = buf;
    bmb.offset = offset;
    bmb.size = size;
    vkCmdPipelineBarrier(cmd_buf,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 1, &bmb, 0, nullptr);
  };

  auto attn_layer_idx = [](uint32_t layer) -> uint32_t {
    return (layer + 1) / 4 - 1;
  };

  // Initialize prefill chunk collection
  prefill_chunks_.resize(NUM_DN_LAYERS);
  prefill_token_count_ = prompt_len;
  for (auto& chunk : prefill_chunks_) {
    chunk.query.clear();
    chunk.key.clear();
    chunk.value.clear();
    chunk.g.clear();
    chunk.beta.clear();
  }

  // Diagnostic: GPU prefill collection for comparison
  const char* collect_env = std::getenv("SPOCK_GPU_COLLECT_PREFILL_COMPARE");
  const bool collect_compare =
      collect_env && collect_env[0] == '1' && collect_env[1] == '\0';
  if (collect_compare && prompt_len > 0) {
    size_t qk_bytes = static_cast<size_t>(DN_HEADS) * prompt_len * DN_K_DIM * 4;
    size_t v_bytes  = static_cast<size_t>(DN_HEADS) * prompt_len * DN_V_DIM * 4;
    size_t gb_bytes = static_cast<size_t>(DN_HEADS) * prompt_len * 4;
    if (bufs_->collect_bufs_allocated_) {
      dev_.destroy_buffer(bufs_->dn_collect_q);
      dev_.destroy_buffer(bufs_->dn_collect_k);
      dev_.destroy_buffer(bufs_->dn_collect_v);
      dev_.destroy_buffer(bufs_->dn_collect_g);
      dev_.destroy_buffer(bufs_->dn_collect_beta);
    }
    bufs_->dn_collect_q    = dev_.create_device_local_buffer(qk_bytes);
    bufs_->dn_collect_k    = dev_.create_device_local_buffer(qk_bytes);
    bufs_->dn_collect_v    = dev_.create_device_local_buffer(v_bytes);
    bufs_->dn_collect_g    = dev_.create_device_local_buffer(gb_bytes);
    bufs_->dn_collect_beta = dev_.create_device_local_buffer(gb_bytes);
    bufs_->collect_bufs_allocated_ = true;
  }

  // Persistent GPU prefill collection for GPU→GPU chunk prefill
  // (SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1)
  // Allocates per-tensor buffers with NUM_DN_LAYERS segments
  const char* gpu_chunk_env = std::getenv("SPOCK_GPU_CHUNK_PREFILL");
  const char* persist_env = std::getenv("SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT");
  const bool gpu_chunk_enabled =
      gpu_chunk_env && gpu_chunk_env[0] == '1' && gpu_chunk_env[1] == '\0';
  const bool gpu_collect_persist =
      gpu_chunk_enabled &&
      persist_env && persist_env[0] == '1' && persist_env[1] == '\0';
  const bool gpu_chunk_prefill_compare = []() {
    const char* e = std::getenv("SPOCK_GPU_CHUNK_PREFILL_COMPARE");
    return e && e[0] == '1' && e[1] == '\0';
  }();
  const bool skip_cpu_collection =
      gpu_collect_persist && !collect_compare && !gpu_chunk_prefill_compare;
  if (gpu_collect_persist && prompt_len > 0) {
    VkDeviceSize qk_persist_stride = align_storage_offset(
        static_cast<VkDeviceSize>(DN_HEADS) * prompt_len * DN_K_DIM * 4);
    VkDeviceSize v_persist_stride = align_storage_offset(
        static_cast<VkDeviceSize>(DN_HEADS) * prompt_len * DN_V_DIM * 4);
    VkDeviceSize gb_persist_stride = align_storage_offset(
        static_cast<VkDeviceSize>(DN_HEADS) * prompt_len * 4);
    if (bufs_->persist_bufs_allocated_) {
      dev_.destroy_buffer(bufs_->dn_persist_q);
      dev_.destroy_buffer(bufs_->dn_persist_k);
      dev_.destroy_buffer(bufs_->dn_persist_v);
      dev_.destroy_buffer(bufs_->dn_persist_g);
      dev_.destroy_buffer(bufs_->dn_persist_beta);
    }
    bufs_->dn_persist_q    = dev_.create_device_local_buffer(qk_persist_stride * NUM_DN_LAYERS);
    bufs_->dn_persist_k    = dev_.create_device_local_buffer(qk_persist_stride * NUM_DN_LAYERS);
    bufs_->dn_persist_v    = dev_.create_device_local_buffer(v_persist_stride * NUM_DN_LAYERS);
    bufs_->dn_persist_g    = dev_.create_device_local_buffer(gb_persist_stride * NUM_DN_LAYERS);
    bufs_->dn_persist_beta = dev_.create_device_local_buffer(gb_persist_stride * NUM_DN_LAYERS);
    bufs_->persist_bufs_allocated_ = true;
  }


  // Allocate per-token hidden state buffer (fp16, prompt_len * HIDDEN)
  size_t hidden_bytes = prompt_len * B.act_bytes;
  auto hidden_buf = dev_.create_device_local_buffer(hidden_bytes);

  // Helper: copy between hidden[t] and act_a
  auto copy_to_hidden = [&](uint32_t t) {
    auto cmd = dev_.allocate_command_buffer();
    dev_.begin_command_buffer(cmd);
    VkBufferCopy copy{0, t * B.act_bytes, B.act_bytes};
    vkCmdCopyBuffer(cmd, B.act_a.buffer, hidden_buf.buffer, 1, &copy);
    dev_.end_command_buffer(cmd);
    dev_.submit_and_wait(cmd);
  };

  auto copy_from_hidden = [&](uint32_t t) {
    auto cmd = dev_.allocate_command_buffer();
    dev_.begin_command_buffer(cmd);
    VkBufferCopy copy{t * B.act_bytes, 0, B.act_bytes};
    vkCmdCopyBuffer(cmd, hidden_buf.buffer, B.act_a.buffer, 1, &copy);
    dev_.end_command_buffer(cmd);
    dev_.submit_and_wait(cmd);
  };

  // Phase 1: Embed all tokens → hidden buffer
  for (uint32_t t = 0; t < prompt_len; ++t) {
    {
      auto cmd = dev_.allocate_command_buffer();
      dev_.begin_command_buffer(cmd);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.embedding);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          P.pipeline_layout_2, 0, 1, &D.embedding, 0, nullptr);
      uint32_t push_token = tokens[t];
      vkCmdPushConstants(cmd, P.pipeline_layout_2, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &push_token);
      vkCmdDispatch(cmd, 1, 1, 1);
      dev_.end_command_buffer(cmd);
      dev_.submit_and_wait(cmd);
    }
    copy_to_hidden(t);
  }

  if (verbose) {
    std::cerr << "  prefill: embedded " << prompt_len << " tokens\n";
  }

  // Phase 2: Layer-major processing
  for (uint32_t layer = 0; layer < LAYERS; ++layer) {
    bool is_attn = (schedule[layer] == model::LayerKind::FullAttention);

    // --- Common weight lookups ---
    auto input_norm_w = artifact_.find_by_role(
        "layer." + std::to_string(layer) + ".input_norm");
    auto post_norm_w = artifact_.find_by_role(
        "layer." + std::to_string(layer) + ".post_norm");
    auto gate_w = artifact_.find_by_role(
        "layer." + std::to_string(layer) + ".mlp_gate");
    auto up_w = artifact_.find_by_role(
        "layer." + std::to_string(layer) + ".mlp_up");
    auto down_w = artifact_.find_by_role(
        "layer." + std::to_string(layer) + ".mlp_down");

    if (is_attn) {
      // --- Attention layer: process all tokens sequentially ---
      uint32_t attn_idx = attn_layer_idx(layer);
      auto attn_q_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".attn_q");
      auto attn_k_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".attn_k");
      auto attn_v_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".attn_v");
      auto attn_o_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".attn_o");
      auto attn_q_norm_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".attn_q_norm");
      auto attn_k_norm_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".attn_k_norm");

      for (uint32_t t = 0; t < prompt_len; ++t) {
        copy_from_hidden(t);
        // Save pre-norm hidden state snapshot for chunk correction (last token)
        if (t == prompt_len - 1) {
          auto snap_cmd = dev_.allocate_command_buffer();
          dev_.begin_command_buffer(snap_cmd);
          VkBufferCopy snap_copy{0, layer * HIDDEN * 2, HIDDEN * 2};
          vkCmdCopyBuffer(snap_cmd, B.act_a.buffer, prefill_snapshots_.buffer, 1, &snap_copy);
          dev_.end_command_buffer(snap_cmd);
          dev_.submit_and_wait(snap_cmd);
        }

        // Update descriptor sets for this layer
        dev_.update_descriptor_set(D.input_norm, 0, B.act_a);
        dev_.update_descriptor_set(D.input_norm, 1, B.weights, input_norm_w->offset, input_norm_w->nbytes);
        dev_.update_descriptor_set(D.input_norm, 2, B.act_b);
        dev_.update_descriptor_set(D.residual1, 0, B.act_a);
        dev_.update_descriptor_set(D.residual1, 1, B.act_b);
        dev_.update_descriptor_set(D.residual1, 2, B.act_c);
        dev_.update_descriptor_set(D.post_norm, 0, B.act_c);
        dev_.update_descriptor_set(D.post_norm, 1, B.weights, post_norm_w->offset, post_norm_w->nbytes);
        dev_.update_descriptor_set(D.post_norm, 2, B.act_a);
        dev_.update_descriptor_set(D.gate, 0, B.weights, gate_w->offset, gate_w->nbytes);
        dev_.update_descriptor_set(D.gate, 1, B.act_a);
        dev_.update_descriptor_set(D.gate, 2, B.mlp_gate);
        dev_.update_descriptor_set(D.up, 0, B.weights, up_w->offset, up_w->nbytes);
        dev_.update_descriptor_set(D.up, 1, B.act_a);
        dev_.update_descriptor_set(D.up, 2, B.mlp_up);
        dev_.update_descriptor_set(D.down, 0, B.weights, down_w->offset, down_w->nbytes);
        dev_.update_descriptor_set(D.down, 1, B.mlp_silu);
        dev_.update_descriptor_set(D.down, 2, B.act_b);
        dev_.update_descriptor_set(D.residual2, 0, B.act_c);
        dev_.update_descriptor_set(D.residual2, 1, B.act_b);
        dev_.update_descriptor_set(D.residual2, 2, B.act_a);

        dev_.update_descriptor_set(D.q_proj, 0, B.weights, attn_q_w->offset, attn_q_w->nbytes);
        dev_.update_descriptor_set(D.q_proj, 1, B.act_b);
        dev_.update_descriptor_set(D.q_proj, 2, B.q_proj);
        dev_.update_descriptor_set(D.k_proj, 0, B.weights, attn_k_w->offset, attn_k_w->nbytes);
        dev_.update_descriptor_set(D.k_proj, 1, B.act_b);
        dev_.update_descriptor_set(D.k_proj, 2, B.k);
        dev_.update_descriptor_set(D.v_proj, 0, B.weights, attn_v_w->offset, attn_v_w->nbytes);
        dev_.update_descriptor_set(D.v_proj, 1, B.act_b);
        dev_.update_descriptor_set(D.v_proj, 2, B.v);
        dev_.update_descriptor_set(D.q_norm, 0, B.q);
        dev_.update_descriptor_set(D.q_norm, 1, B.weights, attn_q_norm_w->offset, attn_q_norm_w->nbytes);
        dev_.update_descriptor_set(D.q_norm, 2, B.q);
        dev_.update_descriptor_set(D.k_norm, 0, B.k);
        dev_.update_descriptor_set(D.k_norm, 1, B.weights, attn_k_norm_w->offset, attn_k_norm_w->nbytes);
        dev_.update_descriptor_set(D.k_norm, 2, B.k);

        uint32_t kv_layer_offset = attn_idx * MAX_SEQ * 2 * KV_HEADS * HEAD_DIM * 2;
        dev_.update_descriptor_set(D.kv_store, 0, B.k);
        dev_.update_descriptor_set(D.kv_store, 1, B.v);
        dev_.update_descriptor_set(D.kv_store, 2, B.kv_cache, kv_layer_offset);
        dev_.update_descriptor_set(D.attn, 0, B.q);
        dev_.update_descriptor_set(D.attn, 1, B.kv_cache, kv_layer_offset);
        dev_.update_descriptor_set(D.attn, 2, B.attn_out);
        dev_.update_descriptor_set(D.o_proj, 0, B.weights, attn_o_w->offset, attn_o_w->nbytes);
        dev_.update_descriptor_set(D.o_proj, 1, B.gated_attn);
        dev_.update_descriptor_set(D.o_proj, 2, B.act_b);

        // Record and submit command buffer for this token
        auto cmd = dev_.allocate_command_buffer();
        dev_.begin_command_buffer(cmd);

        struct { uint32_t N; uint32_t eps_bits; } rms_push = { HIDDEN, float_to_bits(RMS_EPS) };
        struct { uint32_t N; uint32_t pad; } res_push = { HIDDEN, 0 };
        struct { uint32_t out_dim; uint32_t in_dim; } mv_push;
        struct { uint32_t N; uint32_t pad; } silu_push = { INTER, 0 };

        // 1. input_norm
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.rmsnorm);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            P.pipeline_layout_3, 0, 1, &D.input_norm, 0, nullptr);
        vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &rms_push);
        vkCmdDispatch(cmd, 1, 1, 1);
        barrier(cmd, B.act_b.buffer, B.act_bytes);

        // 2. Attention token mixer
        {
          size_t q_proj_bytes = Q_HEADS * HEAD_DIM * 2 * 2;
          size_t q_bytes = Q_HEADS * HEAD_DIM * 2;
          size_t kv_bytes = KV_HEADS * HEAD_DIM * 2;
          size_t attn_out_bytes = q_bytes * 2;
          uint32_t kv_cache_layer_bytes = MAX_SEQ * 2 * KV_HEADS * HEAD_DIM * 2;

          // q_proj
          vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.matvec);
          vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_3, 0, 1, &D.q_proj, 0, nullptr);
          struct { uint32_t out_dim; uint32_t in_dim; } q_mv = { Q_HEADS * HEAD_DIM * 2, HIDDEN };
          vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &q_mv);
          vkCmdDispatch(cmd, (Q_HEADS * HEAD_DIM * 2 + 63) / 64, 1, 1);
          barrier(cmd, B.q_proj.buffer, q_proj_bytes);

          // k_proj
          vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.matvec);
          vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_3, 0, 1, &D.k_proj, 0, nullptr);
          struct { uint32_t out_dim; uint32_t in_dim; } k_mv = { KV_HEADS * HEAD_DIM, HIDDEN };
          vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &k_mv);
          vkCmdDispatch(cmd, (KV_HEADS * HEAD_DIM + 63) / 64, 1, 1);
          barrier(cmd, B.k.buffer, kv_bytes);

          // v_proj
          vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.matvec);
          vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_3, 0, 1, &D.v_proj, 0, nullptr);
          struct { uint32_t out_dim; uint32_t in_dim; } v_mv = { KV_HEADS * HEAD_DIM, HIDDEN };
          vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &v_mv);
          vkCmdDispatch(cmd, (KV_HEADS * HEAD_DIM + 63) / 64, 1, 1);
          barrier(cmd, B.v.buffer, kv_bytes);

          // split q+gate
          vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.split_q_gate);
          vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_3, 0, 1, &D.split_q_gate, 0, nullptr);
          struct { uint32_t num_heads; uint32_t head_dim; uint32_t total_input; } split_push = { Q_HEADS, HEAD_DIM, Q_HEADS * HEAD_DIM * 2 };
          vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, &split_push);
          vkCmdDispatch(cmd, 1, 1, 1);
          barrier(cmd, B.q.buffer, q_bytes);
          barrier(cmd, B.gate.buffer, q_bytes);

          // q_norm
          vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.rms_norm_per_head);
          vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_32, 0, 1, &D.q_norm, 0, nullptr);
          struct { uint32_t num_heads; uint32_t head_dim; uint32_t eps_bits; } qnorm_push = { Q_HEADS, HEAD_DIM, float_to_bits(RMS_EPS) };
          vkCmdPushConstants(cmd, P.pipeline_layout_32, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, &qnorm_push);
          vkCmdDispatch(cmd, Q_HEADS, 1, 1);
          barrier(cmd, B.q.buffer, q_bytes);

          // k_norm
          vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.rms_norm_per_head);
          vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_32, 0, 1, &D.k_norm, 0, nullptr);
          struct { uint32_t num_heads; uint32_t head_dim; uint32_t eps_bits; } knorm_push = { KV_HEADS, HEAD_DIM, float_to_bits(RMS_EPS) };
          vkCmdPushConstants(cmd, P.pipeline_layout_32, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, &knorm_push);
          vkCmdDispatch(cmd, KV_HEADS, 1, 1);
          barrier(cmd, B.k.buffer, kv_bytes);

          // RoPE Q
          vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.rope_apply);
          vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_32, 0, 1, &D.rope, 0, nullptr);
          struct { uint32_t num_heads; uint32_t head_dim; uint32_t rotary_dim; uint32_t freq_offset; } rope_q_push = { Q_HEADS, HEAD_DIM, ROTARY_DIM, t * ROTARY_DIM };
          vkCmdPushConstants(cmd, P.pipeline_layout_32, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, &rope_q_push);
          vkCmdDispatch(cmd, 1, 1, 1);
          barrier(cmd, B.q.buffer, q_bytes);

          // RoPE K
          vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.rope_apply);
          vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_32, 0, 1, &D.rope_k, 0, nullptr);
          struct { uint32_t num_heads; uint32_t head_dim; uint32_t rotary_dim; uint32_t freq_offset; } rope_k_push = { KV_HEADS, HEAD_DIM, ROTARY_DIM, t * ROTARY_DIM };
          vkCmdPushConstants(cmd, P.pipeline_layout_32, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, &rope_k_push);
          vkCmdDispatch(cmd, 1, 1, 1);
          barrier(cmd, B.k.buffer, kv_bytes);

          // KV cache store at position t
          vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.kv_cache_store);
          vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_32, 0, 1, &D.kv_store, 0, nullptr);
          struct { uint32_t kv_heads; uint32_t head_dim; uint32_t position; uint32_t max_seq_len; } kvs_push = { KV_HEADS, HEAD_DIM, t, MAX_SEQ };
          vkCmdPushConstants(cmd, P.pipeline_layout_32, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, &kvs_push);
          vkCmdDispatch(cmd, 1, 1, 1);
          barrier(cmd, B.kv_cache.buffer, kv_cache_layer_bytes, kv_layer_offset);

          // Attention (sees all K/V up to position t)
          vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.attention_decode);
          vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_32, 0, 1, &D.attn, 0, nullptr);
          struct { uint32_t q_heads; uint32_t kv_heads; uint32_t head_dim; uint32_t kv_group_size; uint32_t seq_len; uint32_t max_seq_len; float scale; } attn_push;
          attn_push.q_heads = Q_HEADS;
          attn_push.kv_heads = KV_HEADS;
          attn_push.head_dim = HEAD_DIM;
          attn_push.kv_group_size = KV_GROUP;
          attn_push.seq_len = t + 1;
          attn_push.max_seq_len = MAX_SEQ;
          attn_push.scale = ATTN_SCALE;
          vkCmdPushConstants(cmd, P.pipeline_layout_32, VK_SHADER_STAGE_COMPUTE_BIT, 0, 28, &attn_push);
          vkCmdDispatch(cmd, Q_HEADS, 1, 1);
          barrier(cmd, B.attn_out.buffer, attn_out_bytes);

          // Sigmoid gate
          vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.sigmoid_gate);
          vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_3, 0, 1, &D.sigmoid_gate, 0, nullptr);
          struct { uint32_t N; uint32_t pad; } sg_push = { Q_HEADS * HEAD_DIM, 0 };
          vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &sg_push);
          vkCmdDispatch(cmd, 1, 1, 1);
          barrier(cmd, B.gated_attn.buffer, q_bytes * 2);

          // Output projection
          vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.matvec);
          vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_3, 0, 1, &D.o_proj, 0, nullptr);
          struct { uint32_t out_dim; uint32_t in_dim; } o_mv = { HIDDEN, Q_HEADS * HEAD_DIM };
          vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &o_mv);
          vkCmdDispatch(cmd, (HIDDEN + 63) / 64, 1, 1);
          barrier(cmd, B.act_b.buffer, B.act_bytes);
        }

        // 3. residual + MLP (common path)
        // residual_add(act_a, act_b) → act_c
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.residual_add);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            P.pipeline_layout_3, 0, 1, &D.residual1, 0, nullptr);
        vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &res_push);
        vkCmdDispatch(cmd, 1, 1, 1);
        barrier(cmd, B.act_c.buffer, B.act_c_bytes);

        // post_norm(act_c) → act_a
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.rmsnorm);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            P.pipeline_layout_3, 0, 1, &D.post_norm, 0, nullptr);
        vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &rms_push);
        vkCmdDispatch(cmd, 1, 1, 1);
        barrier(cmd, B.act_a.buffer, B.act_bytes);

        // gate_matvec(act_a) → mlp_gate_buf
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.matvec);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            P.pipeline_layout_3, 0, 1, &D.gate, 0, nullptr);
        mv_push = { INTER, HIDDEN };
        vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &mv_push);
        vkCmdDispatch(cmd, (INTER + 63) / 64, 1, 1);
        barrier(cmd, B.mlp_gate.buffer, INTER * 2);

        // up_matvec(act_a) → mlp_up_buf
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.matvec);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            P.pipeline_layout_3, 0, 1, &D.up, 0, nullptr);
        mv_push = { INTER, HIDDEN };
        vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &mv_push);
        vkCmdDispatch(cmd, (INTER + 63) / 64, 1, 1);
        barrier(cmd, B.mlp_up.buffer, INTER * 2);

        // silu_gate
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.silu_gate);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            P.pipeline_layout_3, 0, 1, &D.silu_gate, 0, nullptr);
        vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &silu_push);
        vkCmdDispatch(cmd, 1, 1, 1);
        barrier(cmd, B.mlp_silu.buffer, INTER * 2);

        // down_matvec(mlp_silu_buf) → act_b
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.matvec);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            P.pipeline_layout_3, 0, 1, &D.down, 0, nullptr);
        mv_push = { HIDDEN, INTER };
        vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &mv_push);
        vkCmdDispatch(cmd, (HIDDEN + 63) / 64, 1, 1);
        barrier(cmd, B.act_b.buffer, B.act_bytes);

        // residual_add(act_c, act_b) → act_a
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.residual_add);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            P.pipeline_layout_3, 0, 1, &D.residual2, 0, nullptr);
        vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &res_push);
        vkCmdDispatch(cmd, 1, 1, 1);
        barrier(cmd, B.act_a.buffer, B.act_bytes);

        dev_.end_command_buffer(cmd);
        dev_.submit_and_wait(cmd);

        copy_to_hidden(t);
      }  // end for each token (attention)

    } else {
      // --- DeltaNet layer: recurrent per-token path, collect Q/K/V/g/beta for chunk ---
      uint32_t dn_idx = 0;
      for (uint32_t i = 0; i < layer; ++i) {
        if (schedule[i] != model::LayerKind::FullAttention) ++dn_idx;
      }

      auto dn_qkv_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".delta_in_proj_qkv");
      auto dn_z_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".delta_in_proj_z");
      auto dn_a_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".delta_in_proj_a");
      auto dn_b_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".delta_in_proj_b");
      auto dn_out_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".delta_out_proj");
      auto dn_conv_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".delta_conv");
      auto dn_norm_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".delta_norm");

      // --- Phase 1: Collect Q/K/V/g/beta for all prompt tokens ---
      // Save per-token layer-input hidden states (fp16) for residual in Phase 3.
      // We already have them in hidden_buf and will re-read in Phase 3.
      for (uint32_t t = 0; t < prompt_len; ++t) {
        copy_from_hidden(t);

        // Save pre-norm hidden state snapshot for chunk correction (last token)
        if (t == prompt_len - 1) {
          auto snap_cmd = dev_.allocate_command_buffer();
          dev_.begin_command_buffer(snap_cmd);
          VkBufferCopy snap_copy{0, layer * HIDDEN * 2, HIDDEN * 2};
          vkCmdCopyBuffer(snap_cmd, B.act_a.buffer, prefill_snapshots_.buffer, 1, &snap_copy);
          dev_.end_command_buffer(snap_cmd);
          dev_.submit_and_wait(snap_cmd);
        }

        // Update descriptor sets for this layer (per-token bindings)
        dev_.update_descriptor_set(D.input_norm, 0, B.act_a);
        dev_.update_descriptor_set(D.input_norm, 1, B.weights, input_norm_w->offset, input_norm_w->nbytes);
        dev_.update_descriptor_set(D.input_norm, 2, B.act_b);
        dev_.update_descriptor_set(D.dn_qkv_proj, 0, B.weights, dn_qkv_w->offset, dn_qkv_w->nbytes);
        dev_.update_descriptor_set(D.dn_qkv_proj, 1, B.act_b);
        dev_.update_descriptor_set(D.dn_qkv_proj, 2, B.dn_qkv);
        dev_.update_descriptor_set(D.dn_z_proj, 0, B.weights, dn_z_w->offset, dn_z_w->nbytes);
        dev_.update_descriptor_set(D.dn_z_proj, 1, B.act_b);
        dev_.update_descriptor_set(D.dn_z_proj, 2, B.dn_z);
        dev_.update_descriptor_set(D.dn_a_proj, 0, B.weights, dn_a_w->offset, dn_a_w->nbytes);
        dev_.update_descriptor_set(D.dn_a_proj, 1, B.act_b);
        dev_.update_descriptor_set(D.dn_a_proj, 2, B.dn_a);
        dev_.update_descriptor_set(D.dn_b_proj, 0, B.weights, dn_b_w->offset, dn_b_w->nbytes);
        dev_.update_descriptor_set(D.dn_b_proj, 1, B.act_b);
        dev_.update_descriptor_set(D.dn_b_proj, 2, B.dn_b);
        dev_.update_descriptor_set(D.dn_conv, 0, B.dn_qkv);
        uint32_t conv_state_offset = dn_idx * DN_CONV_DIM * DN_CONV_KS * 2;
        dev_.update_descriptor_set(D.dn_conv, 1, B.dn_conv_state, conv_state_offset);
        dev_.update_descriptor_set(D.dn_conv, 2, B.weights, dn_conv_w->offset, dn_conv_w->nbytes);

        // Submit 1: projections + conv1d + L2 norms
        {
          auto cmd1 = dev_.allocate_command_buffer();
          dev_.begin_command_buffer(cmd1);

          size_t dn_kv_bytes = DN_CONV_DIM * 2;

          // input_norm
          vkCmdBindPipeline(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE, P.rmsnorm);
          vkCmdBindDescriptorSets(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_3, 0, 1, &D.input_norm, 0, nullptr);
          struct { uint32_t N; uint32_t eps_bits; } rms_dn = { HIDDEN, float_to_bits(RMS_EPS) };
          vkCmdPushConstants(cmd1, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &rms_dn);
          vkCmdDispatch(cmd1, 1, 1, 1);
          barrier(cmd1, B.act_b.buffer, B.act_bytes);

          // QKV projection
          vkCmdBindPipeline(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE, P.matvec);
          vkCmdBindDescriptorSets(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_3, 0, 1, &D.dn_qkv_proj, 0, nullptr);
          struct { uint32_t out_dim; uint32_t in_dim; } dn_qkv_mv = { DN_CONV_DIM, HIDDEN };
          vkCmdPushConstants(cmd1, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &dn_qkv_mv);
          vkCmdDispatch(cmd1, (DN_CONV_DIM + 63) / 64, 1, 1);
          barrier(cmd1, B.dn_qkv.buffer, dn_kv_bytes);

          // Z gate projection
          vkCmdBindPipeline(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE, P.matvec);
          vkCmdBindDescriptorSets(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_3, 0, 1, &D.dn_z_proj, 0, nullptr);
          struct { uint32_t out_dim; uint32_t in_dim; } dn_z_mv = { DN_VAL_TOTAL, HIDDEN };
          vkCmdPushConstants(cmd1, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &dn_z_mv);
          vkCmdDispatch(cmd1, (DN_VAL_TOTAL + 63) / 64, 1, 1);
          barrier(cmd1, B.dn_z.buffer, DN_VAL_TOTAL * 2);

          // A projection
          vkCmdBindPipeline(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE, P.matvec);
          vkCmdBindDescriptorSets(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_3, 0, 1, &D.dn_a_proj, 0, nullptr);
          struct { uint32_t out_dim; uint32_t in_dim; } dn_a_mv = { DN_HEADS, HIDDEN };
          vkCmdPushConstants(cmd1, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &dn_a_mv);
          vkCmdDispatch(cmd1, 1, 1, 1);
          barrier(cmd1, B.dn_a.buffer, DN_HEADS * 2);

          // B projection
          vkCmdBindPipeline(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE, P.matvec);
          vkCmdBindDescriptorSets(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_3, 0, 1, &D.dn_b_proj, 0, nullptr);
          struct { uint32_t out_dim; uint32_t in_dim; } dn_b_mv = { DN_HEADS, HIDDEN };
          vkCmdPushConstants(cmd1, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &dn_b_mv);
          vkCmdDispatch(cmd1, 1, 1, 1);
          barrier(cmd1, B.dn_b.buffer, DN_HEADS * 2);

          // Conv1d step
          vkCmdBindPipeline(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE, P.conv1d_step);
          vkCmdBindDescriptorSets(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_3, 0, 1, &D.dn_conv, 0, nullptr);
          struct { uint32_t conv_dim; uint32_t kernel_size; } conv_push = { DN_CONV_DIM, DN_CONV_KS };
          vkCmdPushConstants(cmd1, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &conv_push);
          vkCmdDispatch(cmd1, 1, 1, 1);
          barrier(cmd1, B.dn_qkv.buffer, dn_kv_bytes);

          // L2-norm Q
          dev_.update_descriptor_set(D.dn_l2_q, 0, B.dn_qkv, 0, DN_KEY_TOTAL * 2);
          dev_.update_descriptor_set(D.dn_l2_q, 1, B.dn_qkv, 0, DN_KEY_TOTAL * 2);
          dev_.update_descriptor_set(D.dn_l2_q, 2, B.dn_qkv, 0, DN_KEY_TOTAL * 2);
          vkCmdBindPipeline(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE, P.l2_norm_per_head);
          vkCmdBindDescriptorSets(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_3, 0, 1, &D.dn_l2_q, 0, nullptr);
          struct { uint32_t num_heads; uint32_t head_dim; } l2q_push = { DN_HEADS, DN_K_DIM };
          vkCmdPushConstants(cmd1, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &l2q_push);
          vkCmdDispatch(cmd1, DN_HEADS, 1, 1);
          barrier(cmd1, B.dn_qkv.buffer, dn_kv_bytes);

          // L2-norm K
          dev_.update_descriptor_set(D.dn_l2_k, 0, B.dn_qkv, DN_KEY_TOTAL * 2, DN_KEY_TOTAL * 2);
          dev_.update_descriptor_set(D.dn_l2_k, 1, B.dn_qkv, DN_KEY_TOTAL * 2, DN_KEY_TOTAL * 2);
          dev_.update_descriptor_set(D.dn_l2_k, 2, B.dn_qkv, DN_KEY_TOTAL * 2, DN_KEY_TOTAL * 2);
          vkCmdBindPipeline(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE, P.l2_norm_per_head);
          vkCmdBindDescriptorSets(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_3, 0, 1, &D.dn_l2_k, 0, nullptr);
          struct { uint32_t num_heads; uint32_t head_dim; } l2k_push = { DN_HEADS, DN_K_DIM };
          vkCmdPushConstants(cmd1, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &l2k_push);
          vkCmdDispatch(cmd1, DN_HEADS, 1, 1);
          barrier(cmd1, B.dn_qkv.buffer, dn_kv_bytes);

          dev_.end_command_buffer(cmd1);
          dev_.submit_and_wait(cmd1);
        }

        // GPU: Compute g, beta from dn_a, dn_b + cached a_log, dt_bias; write to state tail
        {
          VkDeviceSize g_beta_offset = dn_idx * B.dn_state_per_layer + DN_HEADS * DN_K_DIM * DN_V_DIM * 4;
          dev_.update_descriptor_set(D.dn_compute_g_beta, 3, B.dn_state, g_beta_offset, DN_HEADS * 2 * 4);

          auto gb_cmd = dev_.allocate_command_buffer();
          dev_.begin_command_buffer(gb_cmd);

          vkCmdBindPipeline(gb_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.deltanet_compute_g_beta);
          vkCmdBindDescriptorSets(gb_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_4, 0, 1, &D.dn_compute_g_beta, 0, nullptr);
          struct { uint32_t num_heads; uint32_t layer_idx; } gb_pc = { DN_HEADS, dn_idx };
          vkCmdPushConstants(gb_cmd, P.pipeline_layout_4, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &gb_pc);
          vkCmdDispatch(gb_cmd, DN_HEADS, 1, 1);

          dev_.end_command_buffer(gb_cmd);
          dev_.submit_and_wait(gb_cmd);
        }


        // GPU: diagnostic prefill collection (SPOCK_GPU_COLLECT_PREFILL_COMPARE=1)
        if (collect_compare) {
          VkDeviceSize g_beta_off_coll = dn_idx * B.dn_state_per_layer + DN_HEADS * DN_K_DIM * DN_V_DIM * 4;
          dev_.update_descriptor_set(D.dn_prefill_collect, 0, B.dn_qkv, 0, DN_CONV_DIM * 2);
          dev_.update_descriptor_set(D.dn_prefill_collect, 1, B.dn_state, g_beta_off_coll, DN_HEADS * 2 * 4);
          dev_.update_descriptor_set(D.dn_prefill_collect, 2, bufs_->dn_collect_q);
          dev_.update_descriptor_set(D.dn_prefill_collect, 3, bufs_->dn_collect_k);
          dev_.update_descriptor_set(D.dn_prefill_collect, 4, bufs_->dn_collect_v);
          dev_.update_descriptor_set(D.dn_prefill_collect, 5, bufs_->dn_collect_g);
          dev_.update_descriptor_set(D.dn_prefill_collect, 6, bufs_->dn_collect_beta);

          struct { uint32_t num_heads; uint32_t seq_len; uint32_t token_idx; uint32_t k_dim; uint32_t v_dim; } coll_pc =
              { DN_HEADS, prompt_len, t, DN_K_DIM, DN_V_DIM };

          auto coll_cmd = dev_.allocate_command_buffer();
          dev_.begin_command_buffer(coll_cmd);
          vkCmdBindPipeline(coll_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.deltanet_prefill_collect);
          vkCmdBindDescriptorSets(coll_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_cp, 0, 1, &D.dn_prefill_collect, 0, nullptr);
          vkCmdPushConstants(coll_cmd, P.pipeline_layout_cp, VK_SHADER_STAGE_COMPUTE_BIT, 0, 20, &coll_pc);
          vkCmdDispatch(coll_cmd, DN_HEADS, 1, 1);
          dev_.end_command_buffer(coll_cmd);
          dev_.submit_and_wait(coll_cmd);
        }

        // GPU: persistent prefill collection for GPU→GPU chunk prefill
        if (gpu_collect_persist) {
          VkDeviceSize qk_stride = align_storage_offset(
              static_cast<VkDeviceSize>(DN_HEADS) * prompt_len * DN_K_DIM * 4);
          VkDeviceSize v_stride = align_storage_offset(
              static_cast<VkDeviceSize>(DN_HEADS) * prompt_len * DN_V_DIM * 4);
          VkDeviceSize gb_stride = align_storage_offset(
              static_cast<VkDeviceSize>(DN_HEADS) * prompt_len * 4);
          VkDeviceSize qk_off = static_cast<VkDeviceSize>(dn_idx) * qk_stride;
          VkDeviceSize v_off  = static_cast<VkDeviceSize>(dn_idx) * v_stride;
          VkDeviceSize gb_off = static_cast<VkDeviceSize>(dn_idx) * gb_stride;
          VkDeviceSize g_beta_off_persist = dn_idx * B.dn_state_per_layer + DN_HEADS * DN_K_DIM * DN_V_DIM * 4;

          dev_.update_descriptor_set(D.dn_prefill_collect, 0, B.dn_qkv, 0, DN_CONV_DIM * 2);
          dev_.update_descriptor_set(D.dn_prefill_collect, 1, B.dn_state, g_beta_off_persist, DN_HEADS * 2 * 4);
          dev_.update_descriptor_set(D.dn_prefill_collect, 2, bufs_->dn_persist_q, qk_off,
              static_cast<VkDeviceSize>(DN_HEADS) * prompt_len * DN_K_DIM * 4);
          dev_.update_descriptor_set(D.dn_prefill_collect, 3, bufs_->dn_persist_k, qk_off,
              static_cast<VkDeviceSize>(DN_HEADS) * prompt_len * DN_K_DIM * 4);
          dev_.update_descriptor_set(D.dn_prefill_collect, 4, bufs_->dn_persist_v, v_off,
              static_cast<VkDeviceSize>(DN_HEADS) * prompt_len * DN_V_DIM * 4);
          dev_.update_descriptor_set(D.dn_prefill_collect, 5, bufs_->dn_persist_g, gb_off,
              static_cast<VkDeviceSize>(DN_HEADS) * prompt_len * 4);
          dev_.update_descriptor_set(D.dn_prefill_collect, 6, bufs_->dn_persist_beta, gb_off,
              static_cast<VkDeviceSize>(DN_HEADS) * prompt_len * 4);

          struct { uint32_t num_heads; uint32_t seq_len; uint32_t token_idx; uint32_t k_dim; uint32_t v_dim; } persist_coll_pc =
              { DN_HEADS, prompt_len, t, DN_K_DIM, DN_V_DIM };

          auto persist_coll_cmd = dev_.allocate_command_buffer();
          dev_.begin_command_buffer(persist_coll_cmd);
          vkCmdBindPipeline(persist_coll_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.deltanet_prefill_collect);
          vkCmdBindDescriptorSets(persist_coll_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_cp, 0, 1, &D.dn_prefill_collect, 0, nullptr);
          vkCmdPushConstants(persist_coll_cmd, P.pipeline_layout_cp, VK_SHADER_STAGE_COMPUTE_BIT, 0, 20, &persist_coll_pc);
          vkCmdDispatch(persist_coll_cmd, DN_HEADS, 1, 1);
          dev_.end_command_buffer(persist_coll_cmd);
          dev_.submit_and_wait(persist_coll_cmd);
        }

        // --- Collect Q, K, V, g, beta for chunk prefill oracle ---
        // Skipped when GPU→GPU path is active and no diagnostic compare requested
        if (!skip_cpu_collection) {
          auto staging = dev_.create_host_visible_buffer(DN_CONV_DIM * 2);
          {
            auto cp_cmd = dev_.allocate_command_buffer();
            dev_.begin_command_buffer(cp_cmd);
            VkBufferCopy cp{0, 0, DN_CONV_DIM * 2};
            vkCmdCopyBuffer(cp_cmd, B.dn_qkv.buffer, staging.buffer, 1, &cp);
            dev_.end_command_buffer(cp_cmd);
            dev_.submit_and_wait(cp_cmd);
          }
          std::vector<uint16_t> raw_qkv(DN_CONV_DIM);
          dev_.download_from_device(staging, raw_qkv.data(), DN_CONV_DIM * 2);
          dev_.destroy_buffer(staging);

          VkDeviceSize g_beta_offset = static_cast<VkDeviceSize>(dn_idx) * B.dn_state_per_layer
              + static_cast<VkDeviceSize>(DN_HEADS) * DN_K_DIM * DN_V_DIM * 4;
          std::vector<float> g_beta(2 * DN_HEADS);
          {
            auto gb_staging = dev_.create_host_visible_buffer(2 * DN_HEADS * 4);
            auto gbcp_cmd = dev_.allocate_command_buffer();
            dev_.begin_command_buffer(gbcp_cmd);
            VkBufferCopy cp{ g_beta_offset, 0, 2 * DN_HEADS * 4 };
            vkCmdCopyBuffer(gbcp_cmd, B.dn_state.buffer, gb_staging.buffer, 1, &cp);
            dev_.end_command_buffer(gbcp_cmd);
            dev_.submit_and_wait(gbcp_cmd);
            dev_.download_from_device(gb_staging, g_beta.data(), 2 * DN_HEADS * 4);
            dev_.destroy_buffer(gb_staging);
          }

          auto& chunk = prefill_chunks_[dn_idx];
          for (uint32_t h = 0; h < DN_HEADS; ++h) {
            for (uint32_t d = 0; d < DN_K_DIM; ++d) {
              chunk.query.push_back(half_to_float(raw_qkv[h * DN_K_DIM + d]));
              chunk.key.push_back(half_to_float(raw_qkv[DN_KEY_TOTAL + h * DN_K_DIM + d]));
            }
            for (uint32_t d = 0; d < DN_V_DIM; ++d) {
              chunk.value.push_back(half_to_float(raw_qkv[DN_KEY_TOTAL + DN_KEY_TOTAL + h * DN_V_DIM + d]));
            }
            chunk.g.push_back(g_beta[h]);
            chunk.beta.push_back(g_beta[DN_HEADS + h]);
          }
        }

        // --- Recurrent step + norm_gate + out_proj ---
        {
          VkDescriptorSet ds_dn_rec = per_layer_sets_enabled_ ? per_layer_sets_->dn_recurrent[layer] : D.dn_recurrent;
          VkDescriptorSet ds_dn_norm_gate = per_layer_sets_enabled_ ? per_layer_sets_->dn_norm_gate[layer] : D.dn_norm_gate;
          VkDescriptorSet ds_dn_out_proj = per_layer_sets_enabled_ ? per_layer_sets_->dn_out_proj[layer] : D.dn_out_proj;
          if (!per_layer_sets_enabled_) {
            dev_.update_descriptor_set(D.dn_recurrent, 0, B.dn_qkv, 0, DN_KEY_TOTAL * 2);
            dev_.update_descriptor_set(D.dn_recurrent, 1, B.dn_qkv, DN_KEY_TOTAL * 2,
                (DN_KEY_TOTAL + DN_VAL_TOTAL) * 2);
            VkDeviceSize state_offset_bytes = static_cast<VkDeviceSize>(dn_idx) * B.dn_state_per_layer;
            dev_.update_descriptor_set(D.dn_recurrent, 2, B.dn_state, state_offset_bytes);
          }
          uint32_t state_float_total = DN_HEADS * DN_K_DIM * DN_V_DIM;

          auto rec_cmd = dev_.allocate_command_buffer();
          dev_.begin_command_buffer(rec_cmd);

          // deltanet_recurrent
          vkCmdBindPipeline(rec_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.deltanet_recurrent);
          vkCmdBindDescriptorSets(rec_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_32, 0, 1, &ds_dn_rec, 0, nullptr);
          struct { uint32_t num_heads; uint32_t k_dim; uint32_t v_dim; uint32_t state_total; uint32_t q_scale_bits; } dn_rec_push = { DN_HEADS, DN_K_DIM, DN_V_DIM, state_float_total, float_to_bits(DN_Q_SCALE) };
          vkCmdPushConstants(rec_cmd, P.pipeline_layout_32, VK_SHADER_STAGE_COMPUTE_BIT, 0, 20, &dn_rec_push);
          vkCmdDispatch(rec_cmd, DN_HEADS, 1, 1);
          barrier(rec_cmd, B.dn_qkv.buffer, DN_CONV_DIM * 2);

          // Norm+gate
          if (!per_layer_sets_enabled_) {
            dev_.update_descriptor_set(D.dn_norm_gate, 0, B.dn_qkv, (DN_KEY_TOTAL + DN_KEY_TOTAL) * 2, DN_VAL_TOTAL * 2);
            dev_.update_descriptor_set(D.dn_norm_gate, 1, B.dn_z);
            dev_.update_descriptor_set(D.dn_norm_gate, 2, B.weights, dn_norm_w->offset, dn_norm_w->nbytes);
          }
          vkCmdBindPipeline(rec_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.deltanet_norm_gate);
          vkCmdBindDescriptorSets(rec_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_32, 0, 1, &ds_dn_norm_gate, 0, nullptr);
          struct { uint32_t num_heads; uint32_t head_dim; uint32_t eps_bits; uint32_t output_offset; } dn_ng_push = { DN_HEADS, DN_V_DIM, float_to_bits(RMS_EPS), 0 };
          vkCmdPushConstants(rec_cmd, P.pipeline_layout_32, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, &dn_ng_push);
          vkCmdDispatch(rec_cmd, DN_HEADS, 1, 1);
          barrier(rec_cmd, B.dn_qkv.buffer, DN_CONV_DIM * 2);

          // Output projection
          if (!per_layer_sets_enabled_) {
            dev_.update_descriptor_set(D.dn_out_proj, 0, B.weights, dn_out_w->offset, dn_out_w->nbytes);
            dev_.update_descriptor_set(D.dn_out_proj, 1, B.dn_qkv, (DN_KEY_TOTAL + DN_KEY_TOTAL) * 2, DN_VAL_TOTAL * 2);
            dev_.update_descriptor_set(D.dn_out_proj, 2, B.act_b);
          }
          vkCmdBindPipeline(rec_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.matvec);
          vkCmdBindDescriptorSets(rec_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_3, 0, 1, &ds_dn_out_proj, 0, nullptr);
          struct { uint32_t out_dim; uint32_t in_dim; } dn_out_mv = { HIDDEN, DN_VAL_TOTAL };
          vkCmdPushConstants(rec_cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &dn_out_mv);
          vkCmdDispatch(rec_cmd, (HIDDEN + 63) / 64, 1, 1);
          barrier(rec_cmd, B.act_b.buffer, B.act_bytes);

          dev_.end_command_buffer(rec_cmd);
          dev_.submit_and_wait(rec_cmd);
        }

        // --- MLP tail (residual + post_norm + gate + up + silu + down + residual) ---
        {
          dev_.update_descriptor_set(D.residual1, 0, B.act_a);
          dev_.update_descriptor_set(D.residual1, 1, B.act_b);
          dev_.update_descriptor_set(D.residual1, 2, B.act_c);
          dev_.update_descriptor_set(D.post_norm, 0, B.act_c);
          dev_.update_descriptor_set(D.post_norm, 1, B.weights, post_norm_w->offset, post_norm_w->nbytes);
          dev_.update_descriptor_set(D.post_norm, 2, B.act_a);
          dev_.update_descriptor_set(D.gate, 0, B.weights, gate_w->offset, gate_w->nbytes);
          dev_.update_descriptor_set(D.gate, 1, B.act_a);
          dev_.update_descriptor_set(D.gate, 2, B.mlp_gate);
          dev_.update_descriptor_set(D.up, 0, B.weights, up_w->offset, up_w->nbytes);
          dev_.update_descriptor_set(D.up, 1, B.act_a);
          dev_.update_descriptor_set(D.up, 2, B.mlp_up);
          dev_.update_descriptor_set(D.down, 0, B.weights, down_w->offset, down_w->nbytes);
          dev_.update_descriptor_set(D.down, 1, B.mlp_silu);
          dev_.update_descriptor_set(D.down, 2, B.act_b);
          dev_.update_descriptor_set(D.residual2, 0, B.act_c);
          dev_.update_descriptor_set(D.residual2, 1, B.act_b);
          dev_.update_descriptor_set(D.residual2, 2, B.act_a);

          auto cmd = dev_.allocate_command_buffer();
          dev_.begin_command_buffer(cmd);

          struct { uint32_t N; uint32_t eps_bits; } rms_dn_tail = { HIDDEN, float_to_bits(RMS_EPS) };
          struct { uint32_t N; uint32_t pad; } res_push = { HIDDEN, 0 };
          struct { uint32_t out_dim; uint32_t in_dim; } mv_push;
          struct { uint32_t N; uint32_t pad; } silu_push = { INTER, 0 };

          // residual_add(act_a, act_b) → act_c
          vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.residual_add);
          vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_3, 0, 1, &D.residual1, 0, nullptr);
          vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &res_push);
          vkCmdDispatch(cmd, 1, 1, 1);
          barrier(cmd, B.act_c.buffer, B.act_c_bytes);

          // post_norm(act_c) → act_a
          vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.rmsnorm);
          vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_3, 0, 1, &D.post_norm, 0, nullptr);
          vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &rms_dn_tail);
          vkCmdDispatch(cmd, 1, 1, 1);
          barrier(cmd, B.act_a.buffer, B.act_bytes);

          // gate_matvec(act_a) → mlp_gate_buf
          vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.matvec);
          vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_3, 0, 1, &D.gate, 0, nullptr);
          mv_push = { INTER, HIDDEN };
          vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &mv_push);
          vkCmdDispatch(cmd, (INTER + 63) / 64, 1, 1);
          barrier(cmd, B.mlp_gate.buffer, INTER * 2);

          // up_matvec(act_a) → mlp_up_buf
          vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.matvec);
          vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_3, 0, 1, &D.up, 0, nullptr);
          mv_push = { INTER, HIDDEN };
          vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &mv_push);
          vkCmdDispatch(cmd, (INTER + 63) / 64, 1, 1);
          barrier(cmd, B.mlp_up.buffer, INTER * 2);

          // silu_gate
          vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.silu_gate);
          vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_3, 0, 1, &D.silu_gate, 0, nullptr);
          vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &silu_push);
          vkCmdDispatch(cmd, 1, 1, 1);
          barrier(cmd, B.mlp_silu.buffer, INTER * 2);

          // down_matvec(mlp_silu_buf) → act_b
          vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.matvec);
          vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_3, 0, 1, &D.down, 0, nullptr);
          mv_push = { HIDDEN, INTER };
          vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &mv_push);
          vkCmdDispatch(cmd, (HIDDEN + 63) / 64, 1, 1);
          barrier(cmd, B.act_b.buffer, B.act_bytes);

          // residual_add(act_c, act_b) → act_a
          vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.residual_add);
          vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_3, 0, 1, &D.residual2, 0, nullptr);
          vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &res_push);
          vkCmdDispatch(cmd, 1, 1, 1);
          barrier(cmd, B.act_a.buffer, B.act_bytes);

          dev_.end_command_buffer(cmd);
          dev_.submit_and_wait(cmd);
        }

        copy_to_hidden(t);
      }  // end Phase 1: collect Q/K/V/g/beta for all tokens (recurrent)


      // Diagnostic: compare GPU-collected vs CPU-collected DeltaNet prefill buffers
      if (collect_compare) {
        size_t qk_count = static_cast<size_t>(DN_HEADS) * prompt_len * DN_K_DIM;
        size_t v_count  = static_cast<size_t>(DN_HEADS) * prompt_len * DN_V_DIM;
        size_t gb_count = static_cast<size_t>(DN_HEADS) * prompt_len;

        auto download_buf = [&](const VulkanDevice::Buffer& src, size_t count) {
          auto staging = dev_.create_host_visible_buffer(count * 4);
          auto cp_cmd = dev_.allocate_command_buffer();
          dev_.begin_command_buffer(cp_cmd);
          VkBufferCopy cp{0, 0, static_cast<VkDeviceSize>(count * 4)};
          vkCmdCopyBuffer(cp_cmd, src.buffer, staging.buffer, 1, &cp);
          dev_.end_command_buffer(cp_cmd);
          dev_.submit_and_wait(cp_cmd);
          std::vector<float> out(count);
          std::memcpy(out.data(), staging.mapped, count * 4);
          dev_.destroy_buffer(staging);
          return out;
        };

        auto gpu_q    = download_buf(bufs_->dn_collect_q,    qk_count);
        auto gpu_k    = download_buf(bufs_->dn_collect_k,    qk_count);
        auto gpu_v    = download_buf(bufs_->dn_collect_v,    v_count);
        auto gpu_g    = download_buf(bufs_->dn_collect_g,    gb_count);
        auto gpu_beta = download_buf(bufs_->dn_collect_beta, gb_count);

        // Compare GPU head-major [head][seq][dim] against CPU token-major [token][head][dim]
        auto& chunk = prefill_chunks_[dn_idx];
        float max_abs_q = 0, max_rel_q = 0;
        float max_abs_k = 0, max_rel_k = 0;
        float max_abs_v = 0, max_rel_v = 0;
        float max_abs_g = 0, max_rel_g = 0;
        float max_abs_b = 0, max_rel_b = 0;
        uint32_t nan_count = 0;

        static constexpr float K_EPS = 1.0f;

        for (uint32_t h = 0; h < DN_HEADS; ++h) {
          for (uint32_t tok = 0; tok < prompt_len; ++tok) {
            // Q: [head][seq][k] vs [token][head][k]
            for (uint32_t d = 0; d < DN_K_DIM; ++d) {
              size_t gpu_idx = (static_cast<size_t>(h) * prompt_len + tok) * DN_K_DIM + d;
              size_t cpu_idx = (static_cast<size_t>(tok) * DN_HEADS + h) * DN_K_DIM + d;
              {
                float gv = gpu_q[gpu_idx];
                float cv = chunk.query[cpu_idx];
                if (std::isnan(gv)) { ++nan_count; continue; }
                float ab = std::abs(gv - cv);
                if (ab > max_abs_q) max_abs_q = ab;
                float rel = ab / std::max(K_EPS, std::abs(cv));
                if (rel > max_rel_q) max_rel_q = rel;
              }
              // K
              {
                float gv = gpu_k[gpu_idx];
                float cv = chunk.key[cpu_idx];
                if (std::isnan(gv)) { ++nan_count; continue; }
                float ab = std::abs(gv - cv);
                if (ab > max_abs_k) max_abs_k = ab;
                float rel = ab / std::max(K_EPS, std::abs(cv));
                if (rel > max_rel_k) max_rel_k = rel;
              }
            }
            // V: [head][seq][v] vs [token][head][v]
            for (uint32_t d = 0; d < DN_V_DIM; ++d) {
              size_t gpu_idx = (static_cast<size_t>(h) * prompt_len + tok) * DN_V_DIM + d;
              size_t cpu_idx = (static_cast<size_t>(tok) * DN_HEADS + h) * DN_V_DIM + d;
              float gv = gpu_v[gpu_idx];
              float cv = chunk.value[cpu_idx];
              if (std::isnan(gv)) { ++nan_count; continue; }
              float ab = std::abs(gv - cv);
              if (ab > max_abs_v) max_abs_v = ab;
              float rel = ab / std::max(K_EPS, std::abs(cv));
              if (rel > max_rel_v) max_rel_v = rel;
            }
            // G: [head][seq] vs [token][head]
            {
              size_t gpu_idx = static_cast<size_t>(h) * prompt_len + tok;
              size_t cpu_idx = static_cast<size_t>(tok) * DN_HEADS + h;
              float gv = gpu_g[gpu_idx];
              float cv = chunk.g[cpu_idx];
              if (std::isnan(gv)) { ++nan_count; continue; }
              float ab = std::abs(gv - cv);
              if (ab > max_abs_g) max_abs_g = ab;
              float rel = ab / std::max(K_EPS, std::abs(cv));
              if (rel > max_rel_g) max_rel_g = rel;
            }
            // Beta: [head][seq] vs [token][head]
            {
              size_t gpu_idx = static_cast<size_t>(h) * prompt_len + tok;
              size_t cpu_idx = static_cast<size_t>(tok) * DN_HEADS + h;
              float gv = gpu_beta[gpu_idx];
              float cv = chunk.beta[cpu_idx];
              if (std::isnan(gv)) { ++nan_count; continue; }
              float ab = std::abs(gv - cv);
              if (ab > max_abs_b) max_abs_b = ab;
              float rel = ab / std::max(K_EPS, std::abs(cv));
              if (rel > max_rel_b) max_rel_b = rel;
            }
          }
        }

        std::cerr << "SPOCK_GPU_COLLECT_PREFILL_COMPARE layer=" << dn_idx
                  << " seq_len=" << prompt_len
                  << " max_rel_q=" << max_rel_q
                  << " max_rel_k=" << max_rel_k
                  << " max_rel_v=" << max_rel_v
                  << " max_rel_g=" << max_rel_g
                  << " max_rel_beta=" << max_rel_b
                  << " max_abs_q=" << max_abs_q
                  << " max_abs_k=" << max_abs_k
                  << " max_abs_v=" << max_abs_v
                  << " max_abs_g=" << max_abs_g
                  << " max_abs_beta=" << max_abs_b
                  << " nan_count=" << nan_count << "\n";

        float max_rel = std::max({max_rel_q, max_rel_k, max_rel_v, max_rel_g, max_rel_b});
        if (max_rel > 1e-5f || nan_count > 0) {
          throw std::runtime_error(
              "SPOCK_GPU_COLLECT_PREFILL_COMPARE: mismatch in DeltaNet layer " +
              std::to_string(dn_idx));
        }
      }

      if (verbose) {
        std::cerr << "  prefill: layer " << layer << " (DeltaNet " << dn_idx << ") done (recurrent)\n";
      }
    }  // end DeltaNet layer
  }  // end for each layer

  // Copy last token's hidden state → act_a for decode
  copy_from_hidden(prompt_len - 1);

  dev_.destroy_buffer(hidden_buf);
  if (collect_compare && bufs_->collect_bufs_allocated_) {
    dev_.destroy_buffer(bufs_->dn_collect_q);
    dev_.destroy_buffer(bufs_->dn_collect_k);
    dev_.destroy_buffer(bufs_->dn_collect_v);
    dev_.destroy_buffer(bufs_->dn_collect_g);
    dev_.destroy_buffer(bufs_->dn_collect_beta);
    bufs_->collect_bufs_allocated_ = false;
  }

  if (verbose) {
    std::cerr << "  prefill: layer-major prefill complete\n";
  }
}

// ---------------------------------------------------------------------------
// decode()
// ---------------------------------------------------------------------------

DecodeResult DecodeSession::decode(
    const std::vector<uint32_t>& prompt_tokens,
    uint32_t max_new_tokens,
    bool verbose,
    bool debug_dump,
    bool diagnose_handoff,
    bool diagnose_decode_drift,
    int dump_step_hiddens,
    int dump_step_components,
    bool experiment_attn_o_proj_f32_residual,
    bool experiment_mlp_down_f32_residual,
    int dump_dn_recurrent_state_pre_layer,
    const std::string& dump_dn_recurrent_state_pre_file) {
  DecodeResult result;
  auto barrier = [&](VkCommandBuffer cmd_buf, VkBuffer buf, VkDeviceSize size,
                     VkDeviceSize offset = 0) {
    VkBufferMemoryBarrier bmb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    bmb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bmb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.buffer = buf;
    bmb.offset = offset;
    bmb.size = size;
    vkCmdPipelineBarrier(cmd_buf,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 1, &bmb, 0, nullptr);
  };

  const auto& P = *pipes_;
  const auto& B = *bufs_;
  const auto& D = *dsets_;

  constexpr float RMS_EPS = 1e-6f;
  constexpr float ATTN_SCALE = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
  static const float DN_Q_SCALE = 1.0f / std::sqrt(static_cast<float>(DN_K_DIM));

  auto tokens = prompt_tokens.empty()
      ? std::vector<uint32_t>{1, 2, 3}
      : prompt_tokens;
  result.prompt_tokens = tokens;
  bool dump_dn_recurrent_state_pre_written = false;

  auto t0 = std::chrono::high_resolution_clock::now();

  // Total steps: (prompt_len - 1) prefill + max_new_tokens decode
  uint32_t prompt_len = static_cast<uint32_t>(tokens.size());

  // --- Layer-major prefill for multi-token prompts ---
  // Recurrent per-token DeltaNet (no inline chunk correction).
  // run_chunk_prefill + correct_last_token_hidden fix up state & last token.
  if (prompt_len > 1) {
    layer_major_prefill(tokens, prompt_len, verbose);

    // Chunk-correct DeltaNet state before diagnostic or first decode step.
    run_chunk_prefill();

    if (diagnose_handoff) {
      this->diagnose_handoff(tokens, prompt_len);
      result.generated_tokens = {};
      return result;
    }

    correct_last_token_hidden(tokens, prompt_len);
  }

  // Decode loop starts at the first decode step (prompt_len - 1)
  uint32_t total_steps = (prompt_len > 1 ? prompt_len - 1 : 0) + max_new_tokens;
  uint32_t decode_start = (prompt_len > 1 ? prompt_len - 1 : 0);

  bool skip_layers = (prompt_len > 1);  // layer-major prefill already processed

  // Device-resident token gate: when enabled, argmax_result holds the
  // next embedding token_id on GPU, removing one CPU round-trip per step.
  const bool device_resident_token = []() {
    const char* e = std::getenv("SPOCK_GPU_DEVICE_RESIDENT_TOKEN");
    return e && e[0] == '1' && e[1] == '\0';
  }();
  if (device_resident_token && !tokens.empty()) {
    // Seed argmax_result with the last prompt token for the first decode step.
    upload_raw(dev_, B.argmax_result, &tokens.back(), 4);
  }
  // Deferred token download gate: when enabled alongside device-resident token,
  // avoids per-step CPU download of argmax_result by accumulating tokens in a
  // device-local buffer and downloading once after the loop.
  const bool defer_token_download = device_resident_token &&
      !verbose && !debug_dump && !diagnose_decode_drift &&
      []() {
        const char* e = std::getenv("SPOCK_GPU_DEFER_TOKEN_DOWNLOAD");
        return e && e[0] == '1' && e[1] == '\0';
      }();
  VulkanDevice::Buffer gen_tokens{};
  if (defer_token_download && max_new_tokens > 0) {
    gen_tokens = dev_.create_device_local_buffer(max_new_tokens * 4);
  }

  // Decode drift diagnostic: storage for free-run state snapshot
  // Merged DeltaNet command buffers: record projections+L2-norm and g/beta
  // dispatches directly into the per-layer cmd instead of separate cmd1/gb_cmd
  // command buffers with submit_and_wait. Eliminates 2 extra submits per
  // DeltaNet layer (36 per token on the fast path). Requires no diagnostics
  // that need intermediate GPU state (dump_step_components, dump_step_hiddens).
  const bool merge_deltanet_cmds = []() {
    const char* e = std::getenv("SPOCK_GPU_MERGED_DELTANET");
    return e && e[0] == '1' && e[1] == '\0';
  }();
  const bool can_merge_deltanet = merge_deltanet_cmds &&
      dump_step_components < 0 && dump_step_hiddens < 0;

  // Fused conv1d+L2-norm: replaces conv1d_step + L2 Q + L2 K with one pipeline.
  // Active only in the merged DeltaNet decode path.
  const bool fused_dn_conv_l2 = can_merge_deltanet && []() {
    const char* e = std::getenv("SPOCK_GPU_FUSED_DN_CONV_L2");
    return e && e[0] == '1' && e[1] == '\0';
  }();

  // Fused g/beta + recurrent: replaces deltanet_compute_g_beta + deltanet_recurrent
  // with one fused pipeline. Active only when merged DeltaNet is on and no diagnostics.
  // Eliminates the g/beta intermediate write/read to state tail.
  const bool fused_dn_gbeta_recurrent = can_merge_deltanet && []() {
    const char* e = std::getenv("SPOCK_GPU_FUSED_DN_GBETA_RECURRENT");
    return e && e[0] == '1' && e[1] == '\0';
  }();

  // Fused recurrent + norm_gate: computes g/beta inline, runs recurrent
  // update, and applies DeltaNet RMSNorm+SiLU gate in one dispatch.
  const bool fused_dn_rec_norm_gate = can_merge_deltanet && []() {
    const char* e = std::getenv("SPOCK_GPU_FUSED_DN_REC_NORM_GATE");
    return e && e[0] == '1' && e[1] == '\0';
  }();

  // Single-submit decode: record all dispatches for a decode step (embedding +
  // all layers + LM head + argmax) into one command buffer and submit once per token.
  // Requires per-layer descriptor sets and merged DeltaNet command buffers.
  // Disabled for prefill steps, skip_layers steps, and any diagnostic/dump modes.
  const bool single_submit_decode = []() {
    const char* e = std::getenv("SPOCK_GPU_SINGLE_SUBMIT");
    return e && e[0] == '1' && e[1] == '\0';
  }();
  const bool can_single_submit_base = single_submit_decode &&
      per_layer_sets_enabled_ && merge_deltanet_cmds &&
      dump_step_components < 0 && dump_step_hiddens < 0 &&
      !verbose && !debug_dump && !diagnose_decode_drift &&
      !experiment_attn_o_proj_f32_residual && !experiment_mlp_down_f32_residual;

  // --- Chunked decode gate ---
  // Gate: SPOCK_GPU_CHUNKED_DECODE=1
  // Chunk size: SPOCK_GPU_DECODE_CHUNK_SIZE (tokens per chunk, default 1)
  // First active implementation supports only the fully gated fast path and
  // disables GPU timestamp bookkeeping.  The bounded chunk must stay far below
  // the RADV long-dispatch boundary found by vk_barrier_probe.
  const bool chunked_decode_requested = []() {
    const char* e = std::getenv("SPOCK_GPU_CHUNKED_DECODE");
    return e && e[0] == '1' && e[1] == '\0';
  }();
  const uint32_t decode_chunk_size = []() {
    const char* e = std::getenv("SPOCK_GPU_DECODE_CHUNK_SIZE");
    if (e) {
      unsigned long v = std::strtoul(e, nullptr, 10);
      if (v >= 1 && v <= 1024) return static_cast<uint32_t>(v);
    }
    return 1u;
  }();
  // Variables parsed above; chunked_decode_enabled computed after gpu_block_timestamps.

  // GPU timestamp instrumentation gate: when enabled, records GPU timestamps
  // around decode-step command buffers and exposes per-token GPU execution time.
  // Requires timestamp_valid from device capabilities.
  const bool gpu_timestamps = []() {
    const char* e = std::getenv("SPOCK_GPU_TIMESTAMPS");
    return e && e[0] == '1' && e[1] == '\0';
  }() && dev_.capabilities().timestamp_valid;

  // Timestamp query pool: 2 queries per decode token (start + end).
  // Only allocated when gpu_timestamps is active.
  VkQueryPool ts_pool = VK_NULL_HANDLE;
  std::vector<uint32_t> ts_decode_steps;  // which decode_step indices have GPU timestamps
  if (gpu_timestamps && max_new_tokens > 0) {
    ts_pool = dev_.create_timestamp_query_pool(max_new_tokens * 2);
  }

  // Block-level GPU timestamp instrumentation: when enabled, records per-region
  // timestamps inside the single-submit decode command buffer. Active only when
  // gpu_timestamps is also active. Regions: embedding, layer_0..layer_23, final_norm,
  // lm_head, argmax. Requires can_single_submit_base (per-layer descriptors + merged
  // DeltaNet, no diagnostics/verbose).
  const bool gpu_block_timestamps = gpu_timestamps && can_single_submit_base && []() {
    const char* e = std::getenv("SPOCK_GPU_BLOCK_TIMESTAMPS");
    return e && e[0] == '1' && e[1] == '\0';
  }();
  // Block timestamp regions per single-submit step:
  //   embedding(2) + 24 layers x 2 + final_norm(2) + lm_head(2) + argmax(2) = 56
  static constexpr uint32_t TS_BLOCK_QUERIES_PER_STEP = 56;
  VkQueryPool ts_block_pool = VK_NULL_HANDLE;
  std::vector<uint32_t> ts_block_steps;  // which decode_step indices have block timestamps
  if (gpu_block_timestamps && max_new_tokens > 0) {
    ts_block_pool = dev_.create_timestamp_query_pool(max_new_tokens * TS_BLOCK_QUERIES_PER_STEP);
  }
  // Chunked decode keeps one command buffer open across a bounded number of
  // eligible decode tokens.  It requires device-resident token handoff and
  // deferred token download so no CPU readback is needed between chunked steps.
  const bool chunked_decode_enabled =
      chunked_decode_requested && can_single_submit_base &&
      device_resident_token && defer_token_download &&
      !gpu_block_timestamps;


  // Tiled LM-head optimization: replaces the per-invocation row dot product
  // with a shared-memory tiled reduction.  Default off; enable with
  // SPOCK_GPU_LM_HEAD_TILED=1.  Only affects the final LM-head dispatch
  // in decode, not general matvec or diagnostic paths.
  const bool lm_head_tiled = []() {
    const char* e = std::getenv("SPOCK_GPU_LM_HEAD_TILED");
    return e && e[0] == '1' && e[1] == '\0';
  }();
  // General tiled-matvec override: replaces the per-invocation row dot product
  // with a shared-memory tiled reduction for *every* matvec dispatch in decode.
  // Default off; enable with SPOCK_GPU_MATVEC_TILED=1.
  const bool matvec_tiled = []() {
    const char* e = std::getenv("SPOCK_GPU_MATVEC_TILED");
    return e && e[0] == '1' && e[1] == '\0';
  }();

  // Bind / dispatch helpers that select the tiled or vanilla pipeline.
  auto bind_matvec = [&](VkCommandBuffer cmd) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                      matvec_tiled ? P.matvec_tiled : P.matvec);
  };
  auto dispatch_matvec = [&](VkCommandBuffer cmd, uint32_t out_dim) {
    if (matvec_tiled)
      vkCmdDispatch(cmd, (out_dim + 7) / 8, 1, 1);
    else
      vkCmdDispatch(cmd, (out_dim + 63) / 64, 1, 1);
  };

  std::vector<uint16_t> drift_free_hidden;
  std::vector<float> drift_free_logits;
  std::vector<std::vector<float>> drift_free_dn_state;
  std::vector<std::vector<uint16_t>> drift_free_kv_cache;
  std::vector<uint32_t> drift_prefix_tokens;
  std::vector<uint16_t> drift_free_layer_hidden;
  // Per-layer hidden dump for external HF comparison
  std::vector<uint16_t> dump_layer_hidden;
  // Per-layer component-level intermediates for dump-step-components
  std::vector<uint16_t> dump_input_hidden;     // act_a BEFORE each layer
  std::vector<uint16_t> dump_mixer_output;     // act_b token mixer output before first residual add
  std::vector<uint16_t> dump_mixer_residual;   // act_c AFTER each layer (input + token_mixer_output)
  std::vector<uint16_t> dump_mlp_normed;       // act_a AFTER post_norm, before MLP gate/up
  std::vector<uint16_t> dump_post_mlp;         // act_a AFTER each layer
  std::vector<uint16_t> dump_final_norm;       // act_b after final RMSNorm, before LM head
  std::vector<uint16_t> dump_mlp_gate;         // mlp_gate after gate_matvec, before silu_gate
  std::vector<uint16_t> dump_mlp_up;           // mlp_up after up_matvec, before silu_gate
  std::vector<uint16_t> dump_mlp_product;      // mlp_silu after silu_gate, before down_matvec
  std::vector<uint16_t> dump_down_output;       // act_b after down_matvec, before residual_add (down projection output)
  std::vector<uint16_t> dump_dn_input_norm;     // DeltaNet input RMSNorm output (act_b)
  std::vector<uint16_t> dump_dn_qkv_raw;        // DeltaNet raw q/k/v projection output before conv and L2
  std::vector<uint16_t> dump_dn_conv_state_pre; // DeltaNet conv rolling state slice before conv1d mutates it
  std::vector<uint16_t> dump_dn_q;              // DeltaNet L2-normalized query
  std::vector<uint16_t> dump_dn_k;              // DeltaNet L2-normalized key
  std::vector<uint16_t> dump_dn_v;              // DeltaNet conv/Silu value
  std::vector<uint16_t> dump_dn_z;              // DeltaNet z gate projection
  std::vector<uint16_t> dump_dn_a;              // DeltaNet raw a projection
  std::vector<uint16_t> dump_dn_b;              // DeltaNet raw b projection
  std::vector<float> dump_dn_g_beta;            // DeltaNet g then beta per layer
  std::vector<uint16_t> dump_dn_core;           // DeltaNet recurrent output before norm_gate
  std::vector<uint16_t> dump_dn_gated;          // DeltaNet output after norm_gate, before out_proj
  std::vector<uint16_t> dump_dn_out;            // DeltaNet out_proj output before residual
  std::vector<uint16_t> dump_attn_q_norm;       // Attention query after q_norm, before RoPE
  std::vector<uint16_t> dump_attn_k_norm;       // Attention key after k_norm, before RoPE
  std::vector<uint16_t> dump_attn_gate;         // Attention sigmoid gate vector
  std::vector<uint16_t> dump_attn_v;            // Attention value projection
  std::vector<uint16_t> dump_attn_gated;        // Attention output after sigmoid gate, before o_proj
  std::vector<uint16_t> dump_attn_out;          // Attention o_proj output before residual
  VkCommandBuffer chunk_cmd = VK_NULL_HANDLE;
  uint32_t chunk_recorded_steps = 0;
  for (uint32_t step = decode_start; step < total_steps; ++step) {
    bool is_prefill = (step + 1 < prompt_len);
    uint32_t current_token = is_prefill ? tokens[step] : tokens.back();

    // Host-side per-decode-token timing
    auto t_step_start = std::chrono::high_resolution_clock::now();

    // Decode drift diagnostic target check (must be before skip_layers guard)
    constexpr uint32_t kTargetDecodeStep = 5;
    uint32_t decode_step = step - (prompt_len > 1 ? prompt_len - 1 : 0);
    bool is_target_diag = diagnose_decode_drift && decode_step == kTargetDecodeStep;
    bool is_dump_step = dump_step_hiddens >= 0 && static_cast<int>(decode_step) == dump_step_hiddens;
    bool is_dump_components = dump_step_components >= 0 && static_cast<int>(decode_step) == dump_step_components;
    if (is_target_diag && drift_free_layer_hidden.empty()) {
      drift_free_layer_hidden.resize(LAYERS * HIDDEN, 0);
    }
    if (is_dump_step && dump_layer_hidden.empty()) {
      dump_layer_hidden.resize(LAYERS * HIDDEN, 0);
    }
    if (is_dump_components && dump_input_hidden.empty()) {
      dump_input_hidden.resize(LAYERS * HIDDEN, 0);
      dump_mixer_output.resize(LAYERS * HIDDEN, 0);
      dump_mixer_residual.resize(LAYERS * HIDDEN, 0);
      dump_mlp_normed.resize(LAYERS * HIDDEN, 0);
      dump_post_mlp.resize(LAYERS * HIDDEN, 0);
      dump_final_norm.resize(HIDDEN, 0);
      dump_mlp_gate.resize(LAYERS * INTER, 0);
      dump_mlp_up.resize(LAYERS * INTER, 0);
      dump_mlp_product.resize(LAYERS * INTER, 0);
      dump_down_output.resize(LAYERS * HIDDEN, 0);
      dump_dn_input_norm.resize(LAYERS * HIDDEN, 0);
      dump_dn_qkv_raw.resize(LAYERS * DN_CONV_DIM, 0);
      dump_dn_conv_state_pre.resize(LAYERS * DN_CONV_DIM * DN_CONV_KS, 0);
      dump_dn_q.resize(LAYERS * DN_KEY_TOTAL, 0);
      dump_dn_k.resize(LAYERS * DN_KEY_TOTAL, 0);
      dump_dn_v.resize(LAYERS * DN_VAL_TOTAL, 0);
      dump_dn_z.resize(LAYERS * DN_VAL_TOTAL, 0);
      dump_dn_a.resize(LAYERS * DN_HEADS, 0);
      dump_dn_b.resize(LAYERS * DN_HEADS, 0);
      dump_dn_g_beta.resize(LAYERS * DN_HEADS * 2, 0.0f);
      dump_dn_core.resize(LAYERS * DN_VAL_TOTAL, 0);
      dump_dn_gated.resize(LAYERS * DN_VAL_TOTAL, 0);
      dump_dn_out.resize(LAYERS * HIDDEN, 0);
      dump_attn_q_norm.resize(LAYERS * Q_HEADS * HEAD_DIM, 0);
      dump_attn_k_norm.resize(LAYERS * KV_HEADS * HEAD_DIM, 0);
      dump_attn_gate.resize(LAYERS * Q_HEADS * HEAD_DIM, 0);
      dump_attn_v.resize(LAYERS * KV_HEADS * HEAD_DIM, 0);
      dump_attn_gated.resize(LAYERS * Q_HEADS * HEAD_DIM, 0);
      dump_attn_out.resize(LAYERS * HIDDEN, 0);
    }

    // Single-submit decode: compute per-step eligibility
    const bool was_skip_layers = skip_layers;
    const bool can_single_submit = can_single_submit_base && !is_prefill && !skip_layers;
    const bool use_chunked_cmd = chunked_decode_enabled && can_single_submit;
    const bool use_chunked_skip_cmd =
        chunked_decode_enabled && !is_prefill && was_skip_layers;
    VkCommandBuffer ss_cmd = VK_NULL_HANDLE;

    if (!skip_layers) {
    // --- Embedding lookup ---
    if (can_single_submit) {
      if (use_chunked_cmd) {
        if (chunk_cmd == VK_NULL_HANDLE) {
          chunk_cmd = dev_.allocate_command_buffer();
          dev_.begin_command_buffer(chunk_cmd);
          chunk_recorded_steps = 0;
        }
        ss_cmd = chunk_cmd;
        ++chunk_recorded_steps;
      } else {
        ss_cmd = dev_.allocate_command_buffer();
        dev_.begin_command_buffer(ss_cmd);
      }
      // GPU timestamp: reset queries and write start timestamp
      if (gpu_timestamps && ts_pool != VK_NULL_HANDLE) {
        uint32_t q_base = static_cast<uint32_t>(ts_decode_steps.size()) * 2;
        dev_.reset_query_pool(ts_pool, q_base, 2);
        vkCmdWriteTimestamp(ss_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            ts_pool, q_base);
        ts_decode_steps.push_back(decode_step);
      }
      // Block timestamps: reset all queries for this step and record start of embedding
      if (can_single_submit && ts_block_pool != VK_NULL_HANDLE) {
        uint32_t blk_base = static_cast<uint32_t>(ts_block_steps.size()) * TS_BLOCK_QUERIES_PER_STEP;
        dev_.reset_query_pool(ts_block_pool, blk_base, TS_BLOCK_QUERIES_PER_STEP);
        // Query pair 0: embedding
        vkCmdWriteTimestamp(ss_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            ts_block_pool, blk_base);
        ts_block_steps.push_back(decode_step);
      }
    }
    {
      VkCommandBuffer cmd = can_single_submit ? ss_cmd : dev_.allocate_command_buffer();
      if (!can_single_submit) dev_.begin_command_buffer(cmd);
      if (device_resident_token) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.embedding_from_buffer);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            P.pipeline_layout_3, 0, 1, &D.embedding_from_buffer, 0, nullptr);
      } else {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.embedding);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            P.pipeline_layout_2, 0, 1, &D.embedding, 0, nullptr);
        uint32_t push_token = current_token;
        vkCmdPushConstants(cmd, P.pipeline_layout_2, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &push_token);
      }
      vkCmdDispatch(cmd, 1, 1, 1);
      if (!can_single_submit) {
        dev_.end_command_buffer(cmd);
        dev_.submit_and_wait(cmd);
      } else {
        // Execution barrier: embedding writes act_a; layers read act_a.
        barrier(cmd, B.act_a.buffer, B.act_bytes);
        // Block timestamps: end of embedding region
        if (can_single_submit && ts_block_pool != VK_NULL_HANDLE &&
            !ts_block_steps.empty() && ts_block_steps.back() == decode_step) {
          uint32_t blk_base = static_cast<uint32_t>(ts_block_steps.size() - 1) * TS_BLOCK_QUERIES_PER_STEP;
          vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              ts_block_pool, blk_base + 1);
        }
      }
    }

    // --- Per-layer processing ---
    const auto& schedule = model::Qwen35Config::layer_schedule();
    uint32_t seq_pos = step;

    // RoPE frequencies are precomputed in the session-resident table;
    // descriptor binding below uses offset = seq_pos * ROTARY_DIM * 4

    auto attn_layer_idx = [](uint32_t layer) -> uint32_t {
      return (layer + 1) / 4 - 1;
    };

    for (uint32_t layer = 0; layer < LAYERS; ++layer) {
      auto input_norm_w = artifact_.find_by_role(
          "layer." + std::to_string(layer) + ".input_norm");
      auto post_norm_w = artifact_.find_by_role(
          "layer." + std::to_string(layer) + ".post_norm");
      auto gate_w = artifact_.find_by_role(
          "layer." + std::to_string(layer) + ".mlp_gate");
      auto up_w = artifact_.find_by_role(
          "layer." + std::to_string(layer) + ".mlp_up");
      auto down_w = artifact_.find_by_role(
          "layer." + std::to_string(layer) + ".mlp_down");

      bool is_attn = (schedule[layer] == model::LayerKind::FullAttention);
      uint32_t attn_idx = is_attn ? attn_layer_idx(layer) : 0;

      decltype(artifact_.find_by_role("")) attn_q_w, attn_k_w, attn_v_w, attn_o_w;
      decltype(artifact_.find_by_role("")) attn_q_norm_w, attn_k_norm_w;
      if (is_attn) {
        attn_q_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".attn_q");
        attn_k_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".attn_k");
        attn_v_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".attn_v");
        attn_o_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".attn_o");
        attn_q_norm_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".attn_q_norm");
        attn_k_norm_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".attn_k_norm");
      }

      decltype(artifact_.find_by_role("")) dn_qkv_w, dn_z_w, dn_a_w, dn_b_w;
      decltype(artifact_.find_by_role("")) dn_out_w, dn_conv_w, dt_bias_w, a_log_w, dn_norm_w;
      if (!is_attn) {
        dn_qkv_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".delta_in_proj_qkv");
        dn_z_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".delta_in_proj_z");
        dn_a_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".delta_in_proj_a");
        dn_b_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".delta_in_proj_b");
        dn_out_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".delta_out_proj");
        dn_conv_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".delta_conv");
        dt_bias_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".delta_dt_bias");
        a_log_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".delta_a_log");
        dn_norm_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".delta_norm");
      }

      // Update per-layer descriptor sets
      if (!per_layer_sets_enabled_) {
      dev_.update_descriptor_set(D.input_norm, 0, B.act_a);
      dev_.update_descriptor_set(D.input_norm, 1, B.weights,
          input_norm_w->offset, input_norm_w->nbytes);
      dev_.update_descriptor_set(D.input_norm, 2, B.act_b);

      dev_.update_descriptor_set(D.residual1, 0, B.act_a);
      dev_.update_descriptor_set(D.residual1, 1, B.act_b);
      dev_.update_descriptor_set(D.residual1, 2, B.act_c);

      dev_.update_descriptor_set(D.post_norm, 0, B.act_c);
      dev_.update_descriptor_set(D.post_norm, 1, B.weights,
          post_norm_w->offset, post_norm_w->nbytes);
      dev_.update_descriptor_set(D.post_norm, 2, B.act_a);

      dev_.update_descriptor_set(D.gate, 0, B.weights,
          gate_w->offset, gate_w->nbytes);
      dev_.update_descriptor_set(D.gate, 1, B.act_a);
      dev_.update_descriptor_set(D.gate, 2, B.mlp_gate);

      dev_.update_descriptor_set(D.up, 0, B.weights,
          up_w->offset, up_w->nbytes);
      dev_.update_descriptor_set(D.up, 1, B.act_a);
      dev_.update_descriptor_set(D.up, 2, B.mlp_up);

      dev_.update_descriptor_set(D.down, 0, B.weights,
          down_w->offset, down_w->nbytes);
      dev_.update_descriptor_set(D.down, 1, B.mlp_silu);
      dev_.update_descriptor_set(D.down, 2, B.act_b);
      dev_.update_descriptor_set(D.down_f32, 0, B.weights,
          down_w->offset, down_w->nbytes);
      dev_.update_descriptor_set(D.down_f32, 1, B.mlp_silu);
      dev_.update_descriptor_set(D.down_f32, 2, B.attn_proj_f32);

      dev_.update_descriptor_set(D.residual2, 0, B.act_c);
      dev_.update_descriptor_set(D.residual2, 1, B.act_b);
      dev_.update_descriptor_set(D.residual2, 2, B.act_a);
      dev_.update_descriptor_set(D.mlp_residual_mixed, 0, B.attn_proj_f32);
      dev_.update_descriptor_set(D.mlp_residual_mixed, 1, B.act_c);
      dev_.update_descriptor_set(D.mlp_residual_mixed, 2, B.act_a);

      if (is_attn) {
        dev_.update_descriptor_set(D.q_proj, 0, B.weights, attn_q_w->offset, attn_q_w->nbytes);
        dev_.update_descriptor_set(D.q_proj, 1, B.act_b);
        dev_.update_descriptor_set(D.q_proj, 2, B.q_proj);

        dev_.update_descriptor_set(D.k_proj, 0, B.weights, attn_k_w->offset, attn_k_w->nbytes);
        dev_.update_descriptor_set(D.k_proj, 1, B.act_b);
        dev_.update_descriptor_set(D.k_proj, 2, B.k);

        dev_.update_descriptor_set(D.v_proj, 0, B.weights, attn_v_w->offset, attn_v_w->nbytes);
        dev_.update_descriptor_set(D.v_proj, 1, B.act_b);
        dev_.update_descriptor_set(D.v_proj, 2, B.v);

        dev_.update_descriptor_set(D.q_norm, 0, B.q);
        dev_.update_descriptor_set(D.q_norm, 1, B.weights, attn_q_norm_w->offset, attn_q_norm_w->nbytes);
        dev_.update_descriptor_set(D.q_norm, 2, B.q);

        dev_.update_descriptor_set(D.k_norm, 0, B.k);
        dev_.update_descriptor_set(D.k_norm, 1, B.weights, attn_k_norm_w->offset, attn_k_norm_w->nbytes);
        dev_.update_descriptor_set(D.k_norm, 2, B.k);

        uint32_t kv_layer_offset = attn_idx * MAX_SEQ * 2 * KV_HEADS * HEAD_DIM * 2;
        dev_.update_descriptor_set(D.kv_store, 0, B.k);
        dev_.update_descriptor_set(D.kv_store, 1, B.v);
        dev_.update_descriptor_set(D.kv_store, 2, B.kv_cache, kv_layer_offset);

        dev_.update_descriptor_set(D.attn, 0, B.q);
        dev_.update_descriptor_set(D.attn, 1, B.kv_cache, kv_layer_offset);
        dev_.update_descriptor_set(D.attn, 2, B.attn_out);

        dev_.update_descriptor_set(D.o_proj, 0, B.weights, attn_o_w->offset, attn_o_w->nbytes);
        dev_.update_descriptor_set(D.o_proj, 1, B.gated_attn);
        dev_.update_descriptor_set(D.o_proj, 2, B.act_b);
        dev_.update_descriptor_set(D.o_proj_f32, 0, B.weights, attn_o_w->offset, attn_o_w->nbytes);
        dev_.update_descriptor_set(D.o_proj_f32, 1, B.gated_attn);
        dev_.update_descriptor_set(D.o_proj_f32, 2, B.attn_proj_f32);
        dev_.update_descriptor_set(D.attn_residual_mixed, 0, B.attn_proj_f32);
        dev_.update_descriptor_set(D.attn_residual_mixed, 1, B.act_a);
        dev_.update_descriptor_set(D.attn_residual_mixed, 2, B.act_c);
      }

      if (!is_attn) {
        uint32_t dn_idx = 0;
        for (uint32_t i = 0; i < layer; ++i) {
          if (schedule[i] != model::LayerKind::FullAttention) ++dn_idx;
        }

        dev_.update_descriptor_set(D.dn_qkv_proj, 0, B.weights, dn_qkv_w->offset, dn_qkv_w->nbytes);
        dev_.update_descriptor_set(D.dn_qkv_proj, 1, B.act_b);
        dev_.update_descriptor_set(D.dn_qkv_proj, 2, B.dn_qkv);

        dev_.update_descriptor_set(D.dn_z_proj, 0, B.weights, dn_z_w->offset, dn_z_w->nbytes);
        dev_.update_descriptor_set(D.dn_z_proj, 1, B.act_b);
        dev_.update_descriptor_set(D.dn_z_proj, 2, B.dn_z);

        dev_.update_descriptor_set(D.dn_a_proj, 0, B.weights, dn_a_w->offset, dn_a_w->nbytes);
        dev_.update_descriptor_set(D.dn_a_proj, 1, B.act_b);
        dev_.update_descriptor_set(D.dn_a_proj, 2, B.dn_a);

        dev_.update_descriptor_set(D.dn_b_proj, 0, B.weights, dn_b_w->offset, dn_b_w->nbytes);
        dev_.update_descriptor_set(D.dn_b_proj, 1, B.act_b);
        dev_.update_descriptor_set(D.dn_b_proj, 2, B.dn_b);

        dev_.update_descriptor_set(D.dn_conv, 0, B.dn_qkv);
        uint32_t conv_state_offset = dn_idx * DN_CONV_DIM * DN_CONV_KS * 2;
        dev_.update_descriptor_set(D.dn_conv, 1, B.dn_conv_state, conv_state_offset);
        dev_.update_descriptor_set(D.dn_conv, 2, B.weights, dn_conv_w->offset, dn_conv_w->nbytes);
      }
      }  // end if (!per_layer_sets_enabled_)

      // RoPE descriptors are pre-bound at construction time; position is now via push constant


      // Alias per-layer or shared descriptor sets for this layer
      VkDescriptorSet ds_input_norm = per_layer_sets_enabled_ ? per_layer_sets_->input_norm[layer] : D.input_norm;
      VkDescriptorSet ds_residual1 = per_layer_sets_enabled_ ? per_layer_sets_->residual1[layer] : D.residual1;
      VkDescriptorSet ds_post_norm = per_layer_sets_enabled_ ? per_layer_sets_->post_norm[layer] : D.post_norm;
      VkDescriptorSet ds_gate = per_layer_sets_enabled_ ? per_layer_sets_->gate[layer] : D.gate;
      VkDescriptorSet ds_up = per_layer_sets_enabled_ ? per_layer_sets_->up[layer] : D.up;
      VkDescriptorSet ds_down = per_layer_sets_enabled_ ? per_layer_sets_->down[layer] : D.down;
      VkDescriptorSet ds_down_f32 = per_layer_sets_enabled_ ? per_layer_sets_->down_f32[layer] : D.down_f32;
      VkDescriptorSet ds_residual2 = per_layer_sets_enabled_ ? per_layer_sets_->residual2[layer] : D.residual2;
      VkDescriptorSet ds_mlp_residual_mixed = per_layer_sets_enabled_ ? per_layer_sets_->mlp_residual_mixed[layer] : D.mlp_residual_mixed;
      VkDescriptorSet ds_q_proj = per_layer_sets_enabled_ ? per_layer_sets_->q_proj[layer] : D.q_proj;
      VkDescriptorSet ds_k_proj = per_layer_sets_enabled_ ? per_layer_sets_->k_proj[layer] : D.k_proj;
      VkDescriptorSet ds_v_proj = per_layer_sets_enabled_ ? per_layer_sets_->v_proj[layer] : D.v_proj;
      VkDescriptorSet ds_q_norm = per_layer_sets_enabled_ ? per_layer_sets_->q_norm[layer] : D.q_norm;
      VkDescriptorSet ds_k_norm = per_layer_sets_enabled_ ? per_layer_sets_->k_norm[layer] : D.k_norm;
      VkDescriptorSet ds_kv_store = per_layer_sets_enabled_ ? per_layer_sets_->kv_store[layer] : D.kv_store;
      VkDescriptorSet ds_attn = per_layer_sets_enabled_ ? per_layer_sets_->attn[layer] : D.attn;
      VkDescriptorSet ds_o_proj = per_layer_sets_enabled_ ? per_layer_sets_->o_proj[layer] : D.o_proj;
      VkDescriptorSet ds_o_proj_f32 = per_layer_sets_enabled_ ? per_layer_sets_->o_proj_f32[layer] : D.o_proj_f32;
      VkDescriptorSet ds_attn_residual_mixed = per_layer_sets_enabled_ ? per_layer_sets_->attn_residual_mixed[layer] : D.attn_residual_mixed;
      VkDescriptorSet ds_dn_qkv_proj = per_layer_sets_enabled_ ? per_layer_sets_->dn_qkv_proj[layer] : D.dn_qkv_proj;
      VkDescriptorSet ds_dn_z_proj = per_layer_sets_enabled_ ? per_layer_sets_->dn_z_proj[layer] : D.dn_z_proj;
      VkDescriptorSet ds_dn_a_proj = per_layer_sets_enabled_ ? per_layer_sets_->dn_a_proj[layer] : D.dn_a_proj;
      VkDescriptorSet ds_dn_b_proj = per_layer_sets_enabled_ ? per_layer_sets_->dn_b_proj[layer] : D.dn_b_proj;
      VkDescriptorSet ds_dn_conv = per_layer_sets_enabled_ ? per_layer_sets_->dn_conv[layer] : D.dn_conv;
      VkDescriptorSet ds_dn_l2_q = per_layer_sets_enabled_ ? per_layer_sets_->dn_l2_q[layer] : D.dn_l2_q;
      VkDescriptorSet ds_dn_l2_k = per_layer_sets_enabled_ ? per_layer_sets_->dn_l2_k[layer] : D.dn_l2_k;
      VkDescriptorSet ds_dn_compute_g_beta = per_layer_sets_enabled_ ? per_layer_sets_->dn_compute_g_beta[layer] : D.dn_compute_g_beta;
      VkDescriptorSet ds_dn_recurrent = per_layer_sets_enabled_ ? per_layer_sets_->dn_recurrent[layer] : D.dn_recurrent;
      VkDescriptorSet ds_dn_norm_gate = per_layer_sets_enabled_ ? per_layer_sets_->dn_norm_gate[layer] : D.dn_norm_gate;
      VkDescriptorSet ds_dn_out_proj = per_layer_sets_enabled_ ? per_layer_sets_->dn_out_proj[layer] : D.dn_out_proj;
      VkDescriptorSet ds_dn_recurrent_gbeta = per_layer_sets_enabled_ ? per_layer_sets_->dn_recurrent_gbeta[layer] : D.dn_recurrent_gbeta;
      VkDescriptorSet ds_dn_recurrent_gbeta_norm_gate = per_layer_sets_enabled_ ? per_layer_sets_->dn_recurrent_gbeta_norm_gate[layer] : D.dn_recurrent_gbeta_norm_gate;
      // Capture pre-layer input for dump-step-components
      if (dump_input_hidden.size() >= static_cast<size_t>(layer + 1) * HIDDEN) {
        size_t layer_off = static_cast<size_t>(layer) * HIDDEN;
        dev_.download_from_device(B.act_a, &dump_input_hidden[layer_off], HIDDEN * 2);
      }

      // Record layer command buffer
      VkCommandBuffer cmd = can_single_submit ? ss_cmd : dev_.allocate_command_buffer();
      if (!can_single_submit) dev_.begin_command_buffer(cmd);
      // Block timestamps: start of layer_N region (pair index 1 + layer, queries 2+2*layer and 2+2*layer+1)
      if (can_single_submit && ts_block_pool != VK_NULL_HANDLE &&
          !ts_block_steps.empty() && ts_block_steps.back() == decode_step) {
        uint32_t blk_base = static_cast<uint32_t>(ts_block_steps.size() - 1) * TS_BLOCK_QUERIES_PER_STEP;
        uint32_t layer_q = blk_base + 2 + layer * 2;
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            ts_block_pool, layer_q);
      }

      struct { uint32_t N; uint32_t eps_bits; } rms_push = { HIDDEN, float_to_bits(RMS_EPS) };
      struct { uint32_t N; uint32_t pad; } res_push = { HIDDEN, 0 };
      struct { uint32_t out_dim; uint32_t in_dim; } mv_push;
      struct { uint32_t N; uint32_t pad; } silu_push = { INTER, 0 };
      VulkanDevice::Buffer dn_core_staging{};
      VulkanDevice::Buffer dn_gated_staging{};
      VulkanDevice::Buffer dn_out_staging{};
      VulkanDevice::Buffer dn_qkv_raw_staging{};
      VulkanDevice::Buffer dn_conv_state_pre_staging{};
      VulkanDevice::Buffer mixer_output_staging{};
      VulkanDevice::Buffer mlp_normed_staging{};
      VulkanDevice::Buffer attn_q_norm_staging{};
      VulkanDevice::Buffer attn_k_norm_staging{};
      VulkanDevice::Buffer attn_out_staging{};
      bool capture_dn_stage = false;
      bool capture_attn_stage = false;
      bool use_attn_f32_residual = false;

      // 1. input_norm
      if (is_attn) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.rmsnorm);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            P.pipeline_layout_3, 0, 1, &ds_input_norm, 0, nullptr);
        vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &rms_push);
        vkCmdDispatch(cmd, 1, 1, 1);
        barrier(cmd, B.act_b.buffer, B.act_bytes);
      }

      // 2. Token mixer
      if (is_attn) {
        size_t q_proj_bytes = Q_HEADS * HEAD_DIM * 2 * 2;
        size_t q_bytes = Q_HEADS * HEAD_DIM * 2;
        size_t kv_bytes = KV_HEADS * HEAD_DIM * 2;
        size_t attn_out_bytes = q_bytes * 2;
        uint32_t kv_cache_layer_bytes = MAX_SEQ * 2 * KV_HEADS * HEAD_DIM * 2;
        uint32_t kv_layer_offset = attn_idx * kv_cache_layer_bytes;
        use_attn_f32_residual = experiment_attn_o_proj_f32_residual;
        capture_attn_stage = !use_attn_f32_residual &&
            dump_attn_q_norm.size() >= static_cast<size_t>(layer + 1) * Q_HEADS * HEAD_DIM;

        // q_proj
        bind_matvec(cmd);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            P.pipeline_layout_3, 0, 1, &ds_q_proj, 0, nullptr);
        struct { uint32_t out_dim; uint32_t in_dim; } q_mv = { Q_HEADS * HEAD_DIM * 2, HIDDEN };
        vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &q_mv);
        dispatch_matvec(cmd, Q_HEADS * HEAD_DIM * 2);
        barrier(cmd, B.q_proj.buffer, q_proj_bytes);

        // k_proj
        bind_matvec(cmd);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            P.pipeline_layout_3, 0, 1, &ds_k_proj, 0, nullptr);
        struct { uint32_t out_dim; uint32_t in_dim; } k_mv = { KV_HEADS * HEAD_DIM, HIDDEN };
        vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &k_mv);
        dispatch_matvec(cmd, KV_HEADS * HEAD_DIM);
        barrier(cmd, B.k.buffer, kv_bytes);

        // v_proj
        bind_matvec(cmd);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            P.pipeline_layout_3, 0, 1, &ds_v_proj, 0, nullptr);
        struct { uint32_t out_dim; uint32_t in_dim; } v_mv = { KV_HEADS * HEAD_DIM, HIDDEN };
        vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &v_mv);
        dispatch_matvec(cmd, KV_HEADS * HEAD_DIM);
        barrier(cmd, B.v.buffer, kv_bytes);

        // split q+gate
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.split_q_gate);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            P.pipeline_layout_3, 0, 1, &D.split_q_gate, 0, nullptr);
        struct { uint32_t num_heads; uint32_t head_dim; uint32_t total_input; } split_push = { Q_HEADS, HEAD_DIM, Q_HEADS * HEAD_DIM * 2 };
        vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, &split_push);
        vkCmdDispatch(cmd, 1, 1, 1);
        barrier(cmd, B.q.buffer, q_bytes);
        barrier(cmd, B.gate.buffer, q_bytes);

        // q_norm
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.rms_norm_per_head);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            P.pipeline_layout_32, 0, 1, &ds_q_norm, 0, nullptr);
        struct { uint32_t num_heads; uint32_t head_dim; uint32_t eps_bits; } qnorm_push = { Q_HEADS, HEAD_DIM, float_to_bits(RMS_EPS) };
        vkCmdPushConstants(cmd, P.pipeline_layout_32, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, &qnorm_push);
        vkCmdDispatch(cmd, Q_HEADS, 1, 1);
        barrier(cmd, B.q.buffer, q_bytes);
        if (capture_attn_stage) {
          attn_q_norm_staging = dev_.create_host_visible_buffer(q_bytes);
          VkBufferCopy cp{0, 0, q_bytes};
          vkCmdCopyBuffer(cmd, B.q.buffer, attn_q_norm_staging.buffer, 1, &cp);
        }

        // k_norm
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.rms_norm_per_head);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            P.pipeline_layout_32, 0, 1, &ds_k_norm, 0, nullptr);
        struct { uint32_t num_heads; uint32_t head_dim; uint32_t eps_bits; } knorm_push = { KV_HEADS, HEAD_DIM, float_to_bits(RMS_EPS) };
        vkCmdPushConstants(cmd, P.pipeline_layout_32, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, &knorm_push);
        vkCmdDispatch(cmd, KV_HEADS, 1, 1);
        barrier(cmd, B.k.buffer, kv_bytes);
        if (capture_attn_stage) {
          attn_k_norm_staging = dev_.create_host_visible_buffer(kv_bytes);
          VkBufferCopy cp{0, 0, kv_bytes};
          vkCmdCopyBuffer(cmd, B.k.buffer, attn_k_norm_staging.buffer, 1, &cp);
        }

        // RoPE Q
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.rope_apply);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            P.pipeline_layout_32, 0, 1, &D.rope, 0, nullptr);
        struct { uint32_t num_heads; uint32_t head_dim; uint32_t rotary_dim; uint32_t freq_offset; } rope_q_push = { Q_HEADS, HEAD_DIM, ROTARY_DIM, step * ROTARY_DIM };
        vkCmdPushConstants(cmd, P.pipeline_layout_32, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, &rope_q_push);
        vkCmdDispatch(cmd, 1, 1, 1);
        barrier(cmd, B.q.buffer, q_bytes);

        // RoPE K
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.rope_apply);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            P.pipeline_layout_32, 0, 1, &D.rope_k, 0, nullptr);
        struct { uint32_t num_heads; uint32_t head_dim; uint32_t rotary_dim; uint32_t freq_offset; } rope_k_push = { KV_HEADS, HEAD_DIM, ROTARY_DIM, step * ROTARY_DIM };
        vkCmdPushConstants(cmd, P.pipeline_layout_32, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, &rope_k_push);
        vkCmdDispatch(cmd, 1, 1, 1);
        barrier(cmd, B.k.buffer, kv_bytes);

        // KV cache store
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.kv_cache_store);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            P.pipeline_layout_32, 0, 1, &ds_kv_store, 0, nullptr);
        struct { uint32_t kv_heads; uint32_t head_dim; uint32_t position; uint32_t max_seq_len; } kvs_push = { KV_HEADS, HEAD_DIM, step, MAX_SEQ };
        vkCmdPushConstants(cmd, P.pipeline_layout_32, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, &kvs_push);
        vkCmdDispatch(cmd, 1, 1, 1);
        barrier(cmd, B.kv_cache.buffer, kv_cache_layer_bytes, kv_layer_offset);

        // Attention decode
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.attention_decode);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            P.pipeline_layout_32, 0, 1, &ds_attn, 0, nullptr);
        struct { uint32_t q_heads; uint32_t kv_heads; uint32_t head_dim; uint32_t kv_group_size; uint32_t seq_len; uint32_t max_seq_len; float scale; } attn_push;
        attn_push.q_heads = Q_HEADS;
        attn_push.kv_heads = KV_HEADS;
        attn_push.head_dim = HEAD_DIM;
        attn_push.kv_group_size = KV_GROUP;
        attn_push.seq_len = step + 1;
        attn_push.max_seq_len = MAX_SEQ;
        attn_push.scale = ATTN_SCALE;
        vkCmdPushConstants(cmd, P.pipeline_layout_32, VK_SHADER_STAGE_COMPUTE_BIT, 0, 28, &attn_push);
        vkCmdDispatch(cmd, Q_HEADS, 1, 1);
        barrier(cmd, B.attn_out.buffer, attn_out_bytes);

        // Sigmoid gate
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.sigmoid_gate);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            P.pipeline_layout_3, 0, 1, &D.sigmoid_gate, 0, nullptr);
        struct { uint32_t N; uint32_t pad; } sg_push = { Q_HEADS * HEAD_DIM, 0 };
        vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &sg_push);
        vkCmdDispatch(cmd, 1, 1, 1);
        barrier(cmd, B.gated_attn.buffer, q_bytes * 2);

        // Output projection
        struct { uint32_t out_dim; uint32_t in_dim; } o_mv = { HIDDEN, Q_HEADS * HEAD_DIM };
        if (use_attn_f32_residual) {
          vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.matvec_f32_out);
          vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_3, 0, 1, &ds_o_proj_f32, 0, nullptr);
          vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &o_mv);
          vkCmdDispatch(cmd, (HIDDEN + 63) / 64, 1, 1);
          barrier(cmd, B.attn_proj_f32.buffer, HIDDEN * 4);
        } else {
          bind_matvec(cmd);
          vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_3, 0, 1, &ds_o_proj, 0, nullptr);
          vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &o_mv);
          dispatch_matvec(cmd, HIDDEN);
          barrier(cmd, B.act_b.buffer, B.act_bytes);
          if (capture_attn_stage) {
            attn_out_staging = dev_.create_host_visible_buffer(HIDDEN * 2);
            VkBufferCopy cp{0, 0, HIDDEN * 2};
            vkCmdCopyBuffer(cmd, B.act_b.buffer, attn_out_staging.buffer, 1, &cp);
          }
        }
      } else {
        // DeltaNet recurrent decode path
        size_t dn_kv_bytes = DN_CONV_DIM * 2;

        uint32_t dn_idx = 0;
        for (uint32_t i = 0; i < layer; ++i) {
          if (schedule[i] != model::LayerKind::FullAttention) ++dn_idx;
        }

        // Phase 1: projections + conv1d + L2 norms
        if (can_merge_deltanet) {
          // Merged path: record directly into per-layer cmd.

          // input_norm
          vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.rmsnorm);
          vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_3, 0, 1, &ds_input_norm, 0, nullptr);
          struct { uint32_t N; uint32_t eps_bits; } rms_dn = { HIDDEN, float_to_bits(RMS_EPS) };
          vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &rms_dn);
          vkCmdDispatch(cmd, 1, 1, 1);
          barrier(cmd, B.act_b.buffer, B.act_bytes);

          // QKV projection
          bind_matvec(cmd);
          vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_3, 0, 1, &ds_dn_qkv_proj, 0, nullptr);
          struct { uint32_t out_dim; uint32_t in_dim; } dn_qkv_mv = { DN_CONV_DIM, HIDDEN };
          vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &dn_qkv_mv);
          dispatch_matvec(cmd, DN_CONV_DIM);
          barrier(cmd, B.dn_qkv.buffer, dn_kv_bytes);

          // Z gate projection
          bind_matvec(cmd);
          vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_3, 0, 1, &ds_dn_z_proj, 0, nullptr);
          struct { uint32_t out_dim; uint32_t in_dim; } dn_z_mv = { DN_VAL_TOTAL, HIDDEN };
          vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &dn_z_mv);
          dispatch_matvec(cmd, DN_VAL_TOTAL);
          barrier(cmd, B.dn_z.buffer, DN_VAL_TOTAL * 2);

          // A projection
          bind_matvec(cmd);
          vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_3, 0, 1, &ds_dn_a_proj, 0, nullptr);
          struct { uint32_t out_dim; uint32_t in_dim; } dn_a_mv = { DN_HEADS, HIDDEN };
          vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &dn_a_mv);
          dispatch_matvec(cmd, DN_HEADS);
          barrier(cmd, B.dn_a.buffer, DN_HEADS * 2);

          // B projection
          bind_matvec(cmd);
          vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_3, 0, 1, &ds_dn_b_proj, 0, nullptr);
          struct { uint32_t out_dim; uint32_t in_dim; } dn_b_mv = { DN_HEADS, HIDDEN };
          vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &dn_b_mv);
          dispatch_matvec(cmd, DN_HEADS);
          barrier(cmd, B.dn_b.buffer, DN_HEADS * 2);

          // Conv1d step + L2-norm Q + L2-norm K (fused or separate)
          if (fused_dn_conv_l2) {
            // Fused pipeline: conv1d + SiLU + L2-normalize Q/K in one dispatch
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.deltanet_conv_l2_qk);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                P.pipeline_layout_32, 0, 1, &ds_dn_conv, 0, nullptr);
            struct { uint32_t conv_dim; uint32_t kernel_size; uint32_t key_total; uint32_t num_heads; } fused_push = { DN_CONV_DIM, DN_CONV_KS, DN_KEY_TOTAL, DN_HEADS };
            vkCmdPushConstants(cmd, P.pipeline_layout_32, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, &fused_push);
            vkCmdDispatch(cmd, (DN_CONV_DIM + 127u) / 128u, 1, 1);
            barrier(cmd, B.dn_qkv.buffer, dn_kv_bytes);
          } else {
            // Conv1d step
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.conv1d_step);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                P.pipeline_layout_3, 0, 1, &ds_dn_conv, 0, nullptr);
            struct { uint32_t conv_dim; uint32_t kernel_size; } conv_push = { DN_CONV_DIM, DN_CONV_KS };
            vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &conv_push);
            vkCmdDispatch(cmd, 1, 1, 1);
            barrier(cmd, B.dn_qkv.buffer, dn_kv_bytes);

            // L2-norm Q
            if (!per_layer_sets_enabled_) {
              dev_.update_descriptor_set(D.dn_l2_q, 0, B.dn_qkv, 0, DN_KEY_TOTAL * 2);
              dev_.update_descriptor_set(D.dn_l2_q, 1, B.dn_qkv, 0, DN_KEY_TOTAL * 2);
              dev_.update_descriptor_set(D.dn_l2_q, 2, B.dn_qkv, 0, DN_KEY_TOTAL * 2);
            }
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.l2_norm_per_head);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                P.pipeline_layout_3, 0, 1, &ds_dn_l2_q, 0, nullptr);
            struct { uint32_t num_heads; uint32_t head_dim; } l2q_push = { DN_HEADS, DN_K_DIM };
            vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &l2q_push);
            vkCmdDispatch(cmd, DN_HEADS, 1, 1);
            barrier(cmd, B.dn_qkv.buffer, dn_kv_bytes);

            // L2-norm K
            if (!per_layer_sets_enabled_) {
              dev_.update_descriptor_set(D.dn_l2_k, 0, B.dn_qkv, DN_KEY_TOTAL * 2, DN_KEY_TOTAL * 2);
              dev_.update_descriptor_set(D.dn_l2_k, 1, B.dn_qkv, DN_KEY_TOTAL * 2, DN_KEY_TOTAL * 2);
              dev_.update_descriptor_set(D.dn_l2_k, 2, B.dn_qkv, DN_KEY_TOTAL * 2, DN_KEY_TOTAL * 2);
            }
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.l2_norm_per_head);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                P.pipeline_layout_3, 0, 1, &ds_dn_l2_k, 0, nullptr);
            struct { uint32_t num_heads; uint32_t head_dim; } l2k_push = { DN_HEADS, DN_K_DIM };
            vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &l2k_push);
            vkCmdDispatch(cmd, DN_HEADS, 1, 1);
            barrier(cmd, B.dn_qkv.buffer, dn_kv_bytes);
          }
        } else {
          // Fallback: separate command buffer (preserves intermediate submit
          // boundary for diagnostics).
          auto cmd1 = dev_.allocate_command_buffer();
          dev_.begin_command_buffer(cmd1);

          // input_norm
          vkCmdBindPipeline(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE, P.rmsnorm);
          vkCmdBindDescriptorSets(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_3, 0, 1, &ds_input_norm, 0, nullptr);
          struct { uint32_t N; uint32_t eps_bits; } rms_dn = { HIDDEN, float_to_bits(RMS_EPS) };
          vkCmdPushConstants(cmd1, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &rms_dn);
          vkCmdDispatch(cmd1, 1, 1, 1);
          barrier(cmd1, B.act_b.buffer, B.act_bytes);

          // QKV projection
          vkCmdBindPipeline(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE, P.matvec);
          vkCmdBindDescriptorSets(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_3, 0, 1, &ds_dn_qkv_proj, 0, nullptr);
          struct { uint32_t out_dim; uint32_t in_dim; } dn_qkv_mv = { DN_CONV_DIM, HIDDEN };
          vkCmdPushConstants(cmd1, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &dn_qkv_mv);
          vkCmdDispatch(cmd1, (DN_CONV_DIM + 63) / 64, 1, 1);
          barrier(cmd1, B.dn_qkv.buffer, dn_kv_bytes);
          if (dump_dn_qkv_raw.size() >= static_cast<size_t>(layer + 1) * DN_CONV_DIM) {
            dn_qkv_raw_staging = dev_.create_host_visible_buffer(DN_CONV_DIM * 2);
            VkBufferCopy cp{0, 0, DN_CONV_DIM * 2};
            vkCmdCopyBuffer(cmd1, B.dn_qkv.buffer, dn_qkv_raw_staging.buffer, 1, &cp);
          }

          // Z gate projection
          vkCmdBindPipeline(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE, P.matvec);
          vkCmdBindDescriptorSets(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_3, 0, 1, &ds_dn_z_proj, 0, nullptr);
          struct { uint32_t out_dim; uint32_t in_dim; } dn_z_mv = { DN_VAL_TOTAL, HIDDEN };
          vkCmdPushConstants(cmd1, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &dn_z_mv);
          vkCmdDispatch(cmd1, (DN_VAL_TOTAL + 63) / 64, 1, 1);
          barrier(cmd1, B.dn_z.buffer, DN_VAL_TOTAL * 2);

          // A projection
          vkCmdBindPipeline(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE, P.matvec);
          vkCmdBindDescriptorSets(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_3, 0, 1, &ds_dn_a_proj, 0, nullptr);
          struct { uint32_t out_dim; uint32_t in_dim; } dn_a_mv = { DN_HEADS, HIDDEN };
          vkCmdPushConstants(cmd1, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &dn_a_mv);
          vkCmdDispatch(cmd1, 1, 1, 1);
          barrier(cmd1, B.dn_a.buffer, DN_HEADS * 2);

          // B projection
          vkCmdBindPipeline(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE, P.matvec);
          vkCmdBindDescriptorSets(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_3, 0, 1, &ds_dn_b_proj, 0, nullptr);
          struct { uint32_t out_dim; uint32_t in_dim; } dn_b_mv = { DN_HEADS, HIDDEN };
          vkCmdPushConstants(cmd1, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &dn_b_mv);
          vkCmdDispatch(cmd1, 1, 1, 1);
          barrier(cmd1, B.dn_b.buffer, DN_HEADS * 2);

          // Capture conv state before conv1d_step mutates it
          if (dump_dn_conv_state_pre.size() >= static_cast<size_t>(layer + 1) * DN_CONV_DIM * DN_CONV_KS) {
            uint32_t conv_state_offset = dn_idx * DN_CONV_DIM * DN_CONV_KS * 2;
            dn_conv_state_pre_staging = dev_.create_host_visible_buffer(DN_CONV_DIM * DN_CONV_KS * 2);
            VkBufferCopy conv_cp{conv_state_offset, 0, DN_CONV_DIM * DN_CONV_KS * 2};
            VkBufferMemoryBarrier conv_state_read_barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            conv_state_read_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            conv_state_read_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            conv_state_read_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            conv_state_read_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            conv_state_read_barrier.buffer = B.dn_conv_state.buffer;
            conv_state_read_barrier.offset = conv_state_offset;
            conv_state_read_barrier.size = DN_CONV_DIM * DN_CONV_KS * 2;
            vkCmdPipelineBarrier(cmd1,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 0, nullptr, 1, &conv_state_read_barrier, 0, nullptr);
            vkCmdCopyBuffer(cmd1, B.dn_conv_state.buffer, dn_conv_state_pre_staging.buffer, 1, &conv_cp);
          }
          // Conv1d step
          vkCmdBindPipeline(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE, P.conv1d_step);
          vkCmdBindDescriptorSets(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_3, 0, 1, &ds_dn_conv, 0, nullptr);
          struct { uint32_t conv_dim; uint32_t kernel_size; } conv_push = { DN_CONV_DIM, DN_CONV_KS };
          vkCmdPushConstants(cmd1, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &conv_push);
          vkCmdDispatch(cmd1, 1, 1, 1);
          barrier(cmd1, B.dn_qkv.buffer, dn_kv_bytes);

          // L2-norm Q
          if (!per_layer_sets_enabled_) {
            dev_.update_descriptor_set(D.dn_l2_q, 0, B.dn_qkv, 0, DN_KEY_TOTAL * 2);
            dev_.update_descriptor_set(D.dn_l2_q, 1, B.dn_qkv, 0, DN_KEY_TOTAL * 2);
            dev_.update_descriptor_set(D.dn_l2_q, 2, B.dn_qkv, 0, DN_KEY_TOTAL * 2);
          }
          vkCmdBindPipeline(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE, P.l2_norm_per_head);
          vkCmdBindDescriptorSets(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_3, 0, 1, &ds_dn_l2_q, 0, nullptr);
          struct { uint32_t num_heads; uint32_t head_dim; } l2q_push = { DN_HEADS, DN_K_DIM };
          vkCmdPushConstants(cmd1, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &l2q_push);
          vkCmdDispatch(cmd1, DN_HEADS, 1, 1);
          barrier(cmd1, B.dn_qkv.buffer, dn_kv_bytes);

          // L2-norm K
          if (!per_layer_sets_enabled_) {
            dev_.update_descriptor_set(D.dn_l2_k, 0, B.dn_qkv, DN_KEY_TOTAL * 2, DN_KEY_TOTAL * 2);
            dev_.update_descriptor_set(D.dn_l2_k, 1, B.dn_qkv, DN_KEY_TOTAL * 2, DN_KEY_TOTAL * 2);
            dev_.update_descriptor_set(D.dn_l2_k, 2, B.dn_qkv, DN_KEY_TOTAL * 2, DN_KEY_TOTAL * 2);
          }
          vkCmdBindPipeline(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE, P.l2_norm_per_head);
          vkCmdBindDescriptorSets(cmd1, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_3, 0, 1, &ds_dn_l2_k, 0, nullptr);
          struct { uint32_t num_heads; uint32_t head_dim; } l2k_push = { DN_HEADS, DN_K_DIM };
          vkCmdPushConstants(cmd1, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &l2k_push);
          vkCmdDispatch(cmd1, DN_HEADS, 1, 1);
          barrier(cmd1, B.dn_qkv.buffer, dn_kv_bytes);

          dev_.end_command_buffer(cmd1);
          dev_.submit_and_wait(cmd1);
        }

        if (dn_qkv_raw_staging.buffer != VK_NULL_HANDLE) {
          size_t layer_base = static_cast<size_t>(layer) * DN_CONV_DIM;
          dev_.download_from_device(dn_qkv_raw_staging, &dump_dn_qkv_raw[layer_base], DN_CONV_DIM * 2);
          dev_.destroy_buffer(dn_qkv_raw_staging);
        }
        if (dn_conv_state_pre_staging.buffer != VK_NULL_HANDLE) {
          size_t conv_layer_base = static_cast<size_t>(layer) * DN_CONV_DIM * DN_CONV_KS;
          dev_.download_from_device(dn_conv_state_pre_staging, &dump_dn_conv_state_pre[conv_layer_base], DN_CONV_DIM * DN_CONV_KS * 2);
          dev_.destroy_buffer(dn_conv_state_pre_staging);
        }
        if (dump_dn_input_norm.size() >= static_cast<size_t>(layer + 1) * HIDDEN) {
          size_t layer_base = static_cast<size_t>(layer) * HIDDEN;
          dev_.download_from_device(B.act_b, &dump_dn_input_norm[layer_base], HIDDEN * 2);
        }
        if (dump_dn_q.size() >= static_cast<size_t>(layer + 1) * DN_KEY_TOTAL) {
          size_t layer_base = static_cast<size_t>(layer) * DN_KEY_TOTAL;
          dev_.download_from_device(B.dn_qkv, &dump_dn_q[layer_base], DN_KEY_TOTAL * 2);
        }
        if (dump_dn_k.size() >= static_cast<size_t>(layer + 1) * DN_KEY_TOTAL) {
          size_t layer_base = static_cast<size_t>(layer) * DN_KEY_TOTAL;
          auto staging = dev_.create_host_visible_buffer(DN_KEY_TOTAL * 2);
          auto cp_cmd = dev_.allocate_command_buffer();
          dev_.begin_command_buffer(cp_cmd);
          VkBufferCopy cp{static_cast<VkDeviceSize>(DN_KEY_TOTAL) * 2, 0, DN_KEY_TOTAL * 2};
          vkCmdCopyBuffer(cp_cmd, B.dn_qkv.buffer, staging.buffer, 1, &cp);
          dev_.end_command_buffer(cp_cmd);
          dev_.submit_and_wait(cp_cmd);
          dev_.download_from_device(staging, &dump_dn_k[layer_base], DN_KEY_TOTAL * 2);
          dev_.destroy_buffer(staging);
        }
        if (dump_dn_v.size() >= static_cast<size_t>(layer + 1) * DN_VAL_TOTAL) {
          size_t layer_base = static_cast<size_t>(layer) * DN_VAL_TOTAL;
          auto staging = dev_.create_host_visible_buffer(DN_VAL_TOTAL * 2);
          auto cp_cmd = dev_.allocate_command_buffer();
          dev_.begin_command_buffer(cp_cmd);
          VkBufferCopy cp{static_cast<VkDeviceSize>(DN_KEY_TOTAL + DN_KEY_TOTAL) * 2, 0, DN_VAL_TOTAL * 2};
          vkCmdCopyBuffer(cp_cmd, B.dn_qkv.buffer, staging.buffer, 1, &cp);
          dev_.end_command_buffer(cp_cmd);
          dev_.submit_and_wait(cp_cmd);
          dev_.download_from_device(staging, &dump_dn_v[layer_base], DN_VAL_TOTAL * 2);
          dev_.destroy_buffer(staging);
        }
        if (dump_dn_z.size() >= static_cast<size_t>(layer + 1) * DN_VAL_TOTAL) {
          size_t layer_base = static_cast<size_t>(layer) * DN_VAL_TOTAL;
          dev_.download_from_device(B.dn_z, &dump_dn_z[layer_base], DN_VAL_TOTAL * 2);
        }
        if (dump_dn_a.size() >= static_cast<size_t>(layer + 1) * DN_HEADS) {
          size_t layer_base = static_cast<size_t>(layer) * DN_HEADS;
          dev_.download_from_device(B.dn_a, &dump_dn_a[layer_base], DN_HEADS * 2);
        }
        if (dump_dn_b.size() >= static_cast<size_t>(layer + 1) * DN_HEADS) {
          size_t layer_base = static_cast<size_t>(layer) * DN_HEADS;
          dev_.download_from_device(B.dn_b, &dump_dn_b[layer_base], DN_HEADS * 2);
        }

        // GPU: Compute g, beta, write to state tail. The fused recurrent path
        // computes these scalars inline instead and skips this intermediate.
        if (!fused_dn_gbeta_recurrent && !fused_dn_rec_norm_gate) {
          VkDeviceSize g_beta_offset = dn_idx * B.dn_state_per_layer + DN_HEADS * DN_K_DIM * DN_V_DIM * 4;
          if (!per_layer_sets_enabled_) {
            dev_.update_descriptor_set(D.dn_compute_g_beta, 3, B.dn_state, g_beta_offset, DN_HEADS * 2 * 4);
          }

          if (can_merge_deltanet) {
            // Merged path: record directly into per-layer cmd.
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.deltanet_compute_g_beta);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                P.pipeline_layout_4, 0, 1, &ds_dn_compute_g_beta, 0, nullptr);
            struct { uint32_t num_heads; uint32_t layer_idx; } gb_pc = { DN_HEADS, dn_idx };
            vkCmdPushConstants(cmd, P.pipeline_layout_4, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &gb_pc);
            vkCmdDispatch(cmd, DN_HEADS, 1, 1);
            // Barrier: deltanet_compute_g_beta wrote to B.dn_state tail;
            // deltanet_recurrent reads the full B.dn_state (matrix + g/beta).
            VkDeviceSize state_off = static_cast<VkDeviceSize>(dn_idx) * B.dn_state_per_layer;
            barrier(cmd, B.dn_state.buffer, B.dn_state_per_layer, state_off);
          } else {
            // Fallback: separate command buffer.
            auto gb_cmd = dev_.allocate_command_buffer();
            dev_.begin_command_buffer(gb_cmd);

            vkCmdBindPipeline(gb_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.deltanet_compute_g_beta);
            vkCmdBindDescriptorSets(gb_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                P.pipeline_layout_4, 0, 1, &ds_dn_compute_g_beta, 0, nullptr);
            struct { uint32_t num_heads; uint32_t layer_idx; } gb_pc = { DN_HEADS, dn_idx };
            vkCmdPushConstants(gb_cmd, P.pipeline_layout_4, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &gb_pc);
            vkCmdDispatch(gb_cmd, DN_HEADS, 1, 1);

            dev_.end_command_buffer(gb_cmd);
            dev_.submit_and_wait(gb_cmd);
          }
        }

        if (dump_dn_g_beta.size() >= static_cast<size_t>(layer + 1) * DN_HEADS * 2) {
          size_t layer_base = static_cast<size_t>(layer) * DN_HEADS * 2;
          VkDeviceSize g_beta_offset = dn_idx * B.dn_state_per_layer + DN_HEADS * DN_K_DIM * DN_V_DIM * 4;
          auto staging = dev_.create_host_visible_buffer(DN_HEADS * 2 * 4);
          auto cp_cmd = dev_.allocate_command_buffer();
          dev_.begin_command_buffer(cp_cmd);
          VkBufferCopy cp{g_beta_offset, 0, DN_HEADS * 2 * 4};
          vkCmdCopyBuffer(cp_cmd, B.dn_state.buffer, staging.buffer, 1, &cp);
          dev_.end_command_buffer(cp_cmd);
          dev_.submit_and_wait(cp_cmd);
          dev_.download_from_device(staging, &dump_dn_g_beta[layer_base], DN_HEADS * 2 * 4);
          dev_.destroy_buffer(staging);
        }

        // Sidecar capture: raw dn_state (matrix + g/beta tail) pre-recurrent-update
        // for the requested layer. Only active when dump_dn_recurrent_state_pre_layer
        // matches the current layer and dump_step_components is active.
        // Byte layout per layer: DN_HEADS * DN_K_DIM * DN_V_DIM * 4 (matrix, fp32)
        //   + DN_HEADS * 2 * 4 (g/beta tail, fp32) = dn_state_per_layer bytes total.
        // Total fp32 count: 16*128*128 + 16*2 = 262144 + 32 = 262176 values.
        if (dump_dn_recurrent_state_pre_layer >= 0
            && dump_dn_recurrent_state_pre_layer == static_cast<int>(layer)
            && !dump_dn_recurrent_state_pre_file.empty()
            && is_dump_components) {
          VkDeviceSize cap_offset = static_cast<VkDeviceSize>(dn_idx) * B.dn_state_per_layer;
          auto cap_staging = dev_.create_host_visible_buffer(B.dn_state_per_layer);
          auto cap_cmd = dev_.allocate_command_buffer();
          dev_.begin_command_buffer(cap_cmd);
          // Barrier: g/beta compute wrote to state tail via compute shader;
          // the copy reads via transfer.
          VkBufferMemoryBarrier cap_barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
          cap_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
          cap_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
          cap_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
          cap_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
          cap_barrier.buffer = B.dn_state.buffer;
          cap_barrier.offset = cap_offset;
          cap_barrier.size = B.dn_state_per_layer;
          vkCmdPipelineBarrier(cap_cmd,
              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
              0, 0, nullptr, 1, &cap_barrier, 0, nullptr);
          VkBufferCopy cap_cp{cap_offset, 0, B.dn_state_per_layer};
          vkCmdCopyBuffer(cap_cmd, B.dn_state.buffer, cap_staging.buffer, 1, &cap_cp);
          dev_.end_command_buffer(cap_cmd);
          dev_.submit_and_wait(cap_cmd);
          // cap_staging is HOST_VISIBLE|HOST_COHERENT; data is available after submit_and_wait.
          std::vector<uint8_t> cap_bytes(B.dn_state_per_layer);
          std::memcpy(cap_bytes.data(), cap_staging.mapped, B.dn_state_per_layer);
          dev_.destroy_buffer(cap_staging);
          std::ofstream cap_f(dump_dn_recurrent_state_pre_file, std::ios::binary);
          if (!cap_f) {
            result.error = "cannot open DeltaNet recurrent state pre-update sidecar output: "
                + dump_dn_recurrent_state_pre_file;
            return result;
          }
          cap_f.write(reinterpret_cast<const char*>(cap_bytes.data()), cap_bytes.size());
          if (!cap_f) {
            result.error = "failed to write DeltaNet recurrent state pre-update sidecar output: "
                + dump_dn_recurrent_state_pre_file;
            return result;
          }
          dump_dn_recurrent_state_pre_written = true;
          if (verbose) {
            std::cerr << "  dn_recurrent_state_pre sidecar: " << cap_bytes.size()
                      << " bytes (" << (cap_bytes.size() / 4) << " fp32) -> "
                      << dump_dn_recurrent_state_pre_file << "\n";
          }
        }

        // deltanet_recurrent + norm_gate + out_proj
        if (!per_layer_sets_enabled_) {
          dev_.update_descriptor_set(D.dn_recurrent, 0, B.dn_qkv, 0, DN_KEY_TOTAL * 2);
          dev_.update_descriptor_set(D.dn_recurrent, 1, B.dn_qkv, DN_KEY_TOTAL * 2,
              (DN_KEY_TOTAL + DN_VAL_TOTAL) * 2);
          VkDeviceSize state_offset_bytes = static_cast<VkDeviceSize>(dn_idx) * B.dn_state_per_layer;
          dev_.update_descriptor_set(D.dn_recurrent, 2, B.dn_state, state_offset_bytes);
          dev_.update_descriptor_set(D.dn_recurrent_gbeta, 5, B.dn_state, state_offset_bytes);
          dev_.update_descriptor_set(D.dn_recurrent_gbeta_norm_gate, 5, B.dn_state, state_offset_bytes);
          dev_.update_descriptor_set(D.dn_recurrent_gbeta_norm_gate, 7, B.weights, dn_norm_w->offset, dn_norm_w->nbytes);
        }
        uint32_t state_float_total = DN_HEADS * DN_K_DIM * DN_V_DIM;

        if (fused_dn_rec_norm_gate) {
          vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.deltanet_recurrent_gbeta_norm_gate);
          vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_8_32, 0, 1, &ds_dn_recurrent_gbeta_norm_gate, 0, nullptr);
          struct { uint32_t num_heads; uint32_t k_dim; uint32_t v_dim; uint32_t state_total; uint32_t q_scale_bits; uint32_t layer_idx; uint32_t eps_bits; } dn_rec_ng_push = { DN_HEADS, DN_K_DIM, DN_V_DIM, state_float_total, float_to_bits(DN_Q_SCALE), dn_idx, float_to_bits(RMS_EPS) };
          vkCmdPushConstants(cmd, P.pipeline_layout_8_32, VK_SHADER_STAGE_COMPUTE_BIT, 0, 28, &dn_rec_ng_push);
          vkCmdDispatch(cmd, DN_HEADS, 1, 1);
        } else if (fused_dn_gbeta_recurrent) {
          vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.deltanet_recurrent_gbeta);
          vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_6_32, 0, 1, &ds_dn_recurrent_gbeta, 0, nullptr);
          struct { uint32_t num_heads; uint32_t k_dim; uint32_t v_dim; uint32_t state_total; uint32_t q_scale_bits; uint32_t layer_idx; } dn_rec_gb_push = { DN_HEADS, DN_K_DIM, DN_V_DIM, state_float_total, float_to_bits(DN_Q_SCALE), dn_idx };
          vkCmdPushConstants(cmd, P.pipeline_layout_6_32, VK_SHADER_STAGE_COMPUTE_BIT, 0, 24, &dn_rec_gb_push);
          vkCmdDispatch(cmd, DN_HEADS, 1, 1);
        } else {
          vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.deltanet_recurrent);
          vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_32, 0, 1, &ds_dn_recurrent, 0, nullptr);
          struct { uint32_t num_heads; uint32_t k_dim; uint32_t v_dim; uint32_t state_total; uint32_t q_scale_bits; } dn_rec_push = { DN_HEADS, DN_K_DIM, DN_V_DIM, state_float_total, float_to_bits(DN_Q_SCALE) };
          vkCmdPushConstants(cmd, P.pipeline_layout_32, VK_SHADER_STAGE_COMPUTE_BIT, 0, 20, &dn_rec_push);
          vkCmdDispatch(cmd, DN_HEADS, 1, 1);
        }
        barrier(cmd, B.dn_qkv.buffer, DN_CONV_DIM * 2);

        capture_dn_stage = dump_dn_core.size() >= static_cast<size_t>(layer + 1) * DN_VAL_TOTAL;
        if (capture_dn_stage) {
          dn_core_staging = dev_.create_host_visible_buffer(DN_VAL_TOTAL * 2);
          VkBufferCopy cp{static_cast<VkDeviceSize>(DN_KEY_TOTAL + DN_KEY_TOTAL) * 2, 0, DN_VAL_TOTAL * 2};
          vkCmdCopyBuffer(cmd, B.dn_qkv.buffer, dn_core_staging.buffer, 1, &cp);
        }

        // Norm+gate. The recurrent+norm fused path already wrote the gated
        // output to the V section consumed by out_proj.
        if (!fused_dn_rec_norm_gate) {
          if (!per_layer_sets_enabled_) {
            dev_.update_descriptor_set(D.dn_norm_gate, 0, B.dn_qkv, (DN_KEY_TOTAL + DN_KEY_TOTAL) * 2, DN_VAL_TOTAL * 2);
            dev_.update_descriptor_set(D.dn_norm_gate, 1, B.dn_z);
            dev_.update_descriptor_set(D.dn_norm_gate, 2, B.weights, dn_norm_w->offset, dn_norm_w->nbytes);
          }
          vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.deltanet_norm_gate);
          vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
              P.pipeline_layout_32, 0, 1, &ds_dn_norm_gate, 0, nullptr);
          struct { uint32_t num_heads; uint32_t head_dim; uint32_t eps_bits; uint32_t output_offset; } dn_ng_push = { DN_HEADS, DN_V_DIM, float_to_bits(RMS_EPS), 0 };
          vkCmdPushConstants(cmd, P.pipeline_layout_32, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, &dn_ng_push);
          vkCmdDispatch(cmd, DN_HEADS, 1, 1);
          barrier(cmd, B.dn_qkv.buffer, DN_CONV_DIM * 2);
        }
        if (capture_dn_stage) {
          dn_gated_staging = dev_.create_host_visible_buffer(DN_VAL_TOTAL * 2);
          VkBufferCopy cp{static_cast<VkDeviceSize>(DN_KEY_TOTAL + DN_KEY_TOTAL) * 2, 0, DN_VAL_TOTAL * 2};
          vkCmdCopyBuffer(cmd, B.dn_qkv.buffer, dn_gated_staging.buffer, 1, &cp);
        }

        // Output projection
        if (!per_layer_sets_enabled_) {
          dev_.update_descriptor_set(D.dn_out_proj, 0, B.weights, dn_out_w->offset, dn_out_w->nbytes);
          dev_.update_descriptor_set(D.dn_out_proj, 1, B.dn_qkv, (DN_KEY_TOTAL + DN_KEY_TOTAL) * 2, DN_VAL_TOTAL * 2);
          dev_.update_descriptor_set(D.dn_out_proj, 2, B.act_b);
        }
        bind_matvec(cmd);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            P.pipeline_layout_3, 0, 1, &ds_dn_out_proj, 0, nullptr);
        struct { uint32_t out_dim; uint32_t in_dim; } dn_out_mv = { HIDDEN, DN_VAL_TOTAL };
        vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &dn_out_mv);
        dispatch_matvec(cmd, HIDDEN);
        barrier(cmd, B.act_b.buffer, B.act_bytes);
        if (capture_dn_stage) {
          dn_out_staging = dev_.create_host_visible_buffer(HIDDEN * 2);
          VkBufferCopy cp{0, 0, HIDDEN * 2};
          vkCmdCopyBuffer(cmd, B.act_b.buffer, dn_out_staging.buffer, 1, &cp);
        }
      }

      // 3. residual_add(act_a, act_b) → act_c
      if (dump_mixer_output.size() >= static_cast<size_t>(layer + 1) * HIDDEN) {
        mixer_output_staging = dev_.create_host_visible_buffer(HIDDEN * 2);
        VkBufferCopy cp{0, 0, HIDDEN * 2};
        vkCmdCopyBuffer(cmd, B.act_b.buffer, mixer_output_staging.buffer, 1, &cp);
      }
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          use_attn_f32_residual ? P.residual_add_mixed : P.residual_add);
      const VkDescriptorSet residual1_set =
          use_attn_f32_residual ? ds_attn_residual_mixed : ds_residual1;
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          P.pipeline_layout_3, 0, 1, &residual1_set, 0, nullptr);
      vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &res_push);
      vkCmdDispatch(cmd, 1, 1, 1);
      barrier(cmd, B.act_c.buffer, B.act_c_bytes);

      // 4. post_norm(act_c) → act_a
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.rmsnorm);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          P.pipeline_layout_3, 0, 1, &ds_post_norm, 0, nullptr);
      vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &rms_push);
      vkCmdDispatch(cmd, 1, 1, 1);
      barrier(cmd, B.act_a.buffer, B.act_bytes);
      if (dump_mlp_normed.size() >= static_cast<size_t>(layer + 1) * HIDDEN) {
        mlp_normed_staging = dev_.create_host_visible_buffer(HIDDEN * 2);
        VkBufferCopy cp{0, 0, HIDDEN * 2};
        vkCmdCopyBuffer(cmd, B.act_a.buffer, mlp_normed_staging.buffer, 1, &cp);
      }

      // 5. gate_matvec(act_a) → mlp_gate_buf
      bind_matvec(cmd);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          P.pipeline_layout_3, 0, 1, &ds_gate, 0, nullptr);
      mv_push = { INTER, HIDDEN };
      vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &mv_push);
      dispatch_matvec(cmd, INTER);
      barrier(cmd, B.mlp_gate.buffer, INTER * 2);

      // 6. up_matvec(act_a) → mlp_up_buf
      bind_matvec(cmd);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          P.pipeline_layout_3, 0, 1, &ds_up, 0, nullptr);
      mv_push = { INTER, HIDDEN };
      vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &mv_push);
      dispatch_matvec(cmd, INTER);
      barrier(cmd, B.mlp_up.buffer, INTER * 2);

      // 7. silu_gate
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.silu_gate);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          P.pipeline_layout_3, 0, 1, &D.silu_gate, 0, nullptr);
      vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &silu_push);
      vkCmdDispatch(cmd, 1, 1, 1);
      barrier(cmd, B.mlp_silu.buffer, INTER * 2);

      // 8. down_matvec(mlp_silu_buf) → act_b, or fp32 scratch for precision experiment
      if (experiment_mlp_down_f32_residual)
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.matvec_f32_out);
      else
        bind_matvec(cmd);
      const VkDescriptorSet down_set =
          experiment_mlp_down_f32_residual ? ds_down_f32 : ds_down;
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          P.pipeline_layout_3, 0, 1, &down_set, 0, nullptr);
      mv_push = { HIDDEN, INTER };
      vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &mv_push);
      if (experiment_mlp_down_f32_residual)
        vkCmdDispatch(cmd, (HIDDEN + 63) / 64, 1, 1);
      else
        dispatch_matvec(cmd, HIDDEN);
      if (experiment_mlp_down_f32_residual) {
        barrier(cmd, B.attn_proj_f32.buffer, HIDDEN * 4);
      } else {
        barrier(cmd, B.act_b.buffer, B.act_bytes);
      }

      // 9. residual_add(act_c, act_b) → act_a
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          experiment_mlp_down_f32_residual ? P.residual_add_mixed : P.residual_add);
      const VkDescriptorSet residual2_set =
          experiment_mlp_down_f32_residual ? ds_mlp_residual_mixed : ds_residual2;
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          P.pipeline_layout_3, 0, 1, &residual2_set, 0, nullptr);
      vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &res_push);
      vkCmdDispatch(cmd, 1, 1, 1);
      barrier(cmd, B.act_a.buffer, B.act_bytes);
      // Block timestamps: end of layer_N region
      if (can_single_submit && ts_block_pool != VK_NULL_HANDLE &&
          !ts_block_steps.empty() && ts_block_steps.back() == decode_step) {
        uint32_t blk_base = static_cast<uint32_t>(ts_block_steps.size() - 1) * TS_BLOCK_QUERIES_PER_STEP;
        uint32_t layer_q = blk_base + 2 + layer * 2;
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            ts_block_pool, layer_q + 1);
      }

      if (!can_single_submit) {
        dev_.end_command_buffer(cmd);
        dev_.submit_and_wait(cmd);
      }

      if (!is_attn && dump_dn_core.size() >= static_cast<size_t>(layer + 1) * DN_VAL_TOTAL) {
        size_t val_base = static_cast<size_t>(layer) * DN_VAL_TOTAL;
        size_t hidden_base = static_cast<size_t>(layer) * HIDDEN;
        dev_.download_from_device(dn_core_staging, &dump_dn_core[val_base], DN_VAL_TOTAL * 2);
        dev_.download_from_device(dn_gated_staging, &dump_dn_gated[val_base], DN_VAL_TOTAL * 2);
        dev_.download_from_device(dn_out_staging, &dump_dn_out[hidden_base], HIDDEN * 2);
        dev_.destroy_buffer(dn_core_staging);
        dev_.destroy_buffer(dn_gated_staging);
        dev_.destroy_buffer(dn_out_staging);
      }
      if (mlp_normed_staging.buffer != VK_NULL_HANDLE) {
        size_t layer_off = static_cast<size_t>(layer) * HIDDEN;
        dev_.download_from_device(mlp_normed_staging, &dump_mlp_normed[layer_off], HIDDEN * 2);
        dev_.destroy_buffer(mlp_normed_staging);
      }
      if (mixer_output_staging.buffer != VK_NULL_HANDLE) {
        size_t layer_off = static_cast<size_t>(layer) * HIDDEN;
        dev_.download_from_device(mixer_output_staging, &dump_mixer_output[layer_off], HIDDEN * 2);
        dev_.destroy_buffer(mixer_output_staging);
      }
      if (is_attn && dump_attn_q_norm.size() >= static_cast<size_t>(layer + 1) * Q_HEADS * HEAD_DIM) {
        size_t q_base = static_cast<size_t>(layer) * Q_HEADS * HEAD_DIM;
        size_t kv_base = static_cast<size_t>(layer) * KV_HEADS * HEAD_DIM;
        size_t hidden_base = static_cast<size_t>(layer) * HIDDEN;
        dev_.download_from_device(attn_q_norm_staging, &dump_attn_q_norm[q_base], Q_HEADS * HEAD_DIM * 2);
        dev_.download_from_device(attn_k_norm_staging, &dump_attn_k_norm[kv_base], KV_HEADS * HEAD_DIM * 2);
        dev_.download_from_device(B.gate, &dump_attn_gate[q_base], Q_HEADS * HEAD_DIM * 2);
        dev_.download_from_device(B.v, &dump_attn_v[kv_base], KV_HEADS * HEAD_DIM * 2);
        dev_.download_from_device(B.gated_attn, &dump_attn_gated[q_base], Q_HEADS * HEAD_DIM * 2);
        dev_.download_from_device(attn_out_staging, &dump_attn_out[hidden_base], HIDDEN * 2);
        dev_.destroy_buffer(attn_q_norm_staging);
        dev_.destroy_buffer(attn_k_norm_staging);
        dev_.destroy_buffer(attn_out_staging);
      }

      // Per-layer hidden capture for decode drift diagnostic
      if (drift_free_layer_hidden.size() >= static_cast<size_t>(layer + 1) * HIDDEN) {
        size_t layer_off = static_cast<size_t>(layer) * HIDDEN;
        dev_.download_from_device(B.act_a, &drift_free_layer_hidden[layer_off], HIDDEN * 2);
      }
      // Per-layer hidden capture for dump-step-hiddens
      if (dump_layer_hidden.size() >= static_cast<size_t>(layer + 1) * HIDDEN) {
        size_t layer_off = static_cast<size_t>(layer) * HIDDEN;
        dev_.download_from_device(B.act_a, &dump_layer_hidden[layer_off], HIDDEN * 2);
      }
      // Per-layer mixer residual capture for dump-step-components (act_c = input + token_mixer_output)
      if (dump_mixer_residual.size() >= static_cast<size_t>(layer + 1) * HIDDEN) {
        size_t layer_off = static_cast<size_t>(layer) * HIDDEN;
        dev_.download_from_device(B.act_c, &dump_mixer_residual[layer_off], HIDDEN * 2);
      }
      // Per-layer post-MLP output capture for dump-step-components
      if (dump_post_mlp.size() >= static_cast<size_t>(layer + 1) * HIDDEN) {
        size_t layer_off = static_cast<size_t>(layer) * HIDDEN;
        dev_.download_from_device(B.act_a, &dump_post_mlp[layer_off], HIDDEN * 2);
      }
      // Per-layer MLP projection/product capture for dump-step-components.
      if (dump_mlp_gate.size() >= static_cast<size_t>(layer + 1) * INTER) {
        size_t layer_off = static_cast<size_t>(layer) * INTER;
        dev_.download_from_device(B.mlp_gate, &dump_mlp_gate[layer_off], INTER * 2);
      }
      if (dump_mlp_up.size() >= static_cast<size_t>(layer + 1) * INTER) {
        size_t layer_off = static_cast<size_t>(layer) * INTER;
        dev_.download_from_device(B.mlp_up, &dump_mlp_up[layer_off], INTER * 2);
      }
      if (dump_mlp_product.size() >= static_cast<size_t>(layer + 1) * INTER) {
        size_t layer_off = static_cast<size_t>(layer) * INTER;
        dev_.download_from_device(B.mlp_silu, &dump_mlp_product[layer_off], INTER * 2);
      }
      // Per-layer down_output capture for dump-step-components (act_b after down_matvec, before residual_add)
      // residual_add reads act_b and writes act_a, so act_b data survives after submit.
      if (dump_down_output.size() >= static_cast<size_t>(layer + 1) * HIDDEN) {
        size_t layer_off = static_cast<size_t>(layer) * HIDDEN;
        dev_.download_from_device(B.act_b, &dump_down_output[layer_off], HIDDEN * 2);
      }
    }
    }

    skip_layers = false;  // only skip for the first decode step

    // Skip LM head + argmax for prefill steps
    if (is_prefill) {
      continue;
    }
    // --- Final RMSNorm + LM head + Argmax ---
    {
      const bool use_chunked_final_cmd = use_chunked_cmd || use_chunked_skip_cmd;
      VkCommandBuffer cmd = VK_NULL_HANDLE;
      if (can_single_submit) {
        cmd = ss_cmd;
      } else if (use_chunked_skip_cmd) {
        if (chunk_cmd == VK_NULL_HANDLE) {
          chunk_cmd = dev_.allocate_command_buffer();
          dev_.begin_command_buffer(chunk_cmd);
          chunk_recorded_steps = 0;
        }
        cmd = chunk_cmd;
        ++chunk_recorded_steps;
        // GPU timestamp: start for chunked skip_layers first decode step.
        // The skip_layers first step opens chunk_cmd here since the embedding
        // section (which normally opens it) is skipped for skip_layers.
        if (gpu_timestamps && ts_pool != VK_NULL_HANDLE) {
          uint32_t q_base = static_cast<uint32_t>(ts_decode_steps.size()) * 2;
          dev_.reset_query_pool(ts_pool, q_base, 2);
          vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              ts_pool, q_base);
          ts_decode_steps.push_back(decode_step);
        }
      } else {
        cmd = dev_.allocate_command_buffer();
      }
      if (!can_single_submit && !use_chunked_skip_cmd) {
        dev_.begin_command_buffer(cmd);
        // GPU timestamp: start for non-single-submit decode (skip_layers first step).
        // Single-submit path writes its own start in the embedding section above.
        if (gpu_timestamps && ts_pool != VK_NULL_HANDLE) {
          uint32_t q_base = static_cast<uint32_t>(ts_decode_steps.size()) * 2;
          dev_.reset_query_pool(ts_pool, q_base, 2);
          vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              ts_pool, q_base);
          ts_decode_steps.push_back(decode_step);
        }
      }
      // Block timestamps: start of final_norm region (pair 25, queries 50/51)
      if (can_single_submit && ts_block_pool != VK_NULL_HANDLE &&
          !ts_block_steps.empty() && ts_block_steps.back() == decode_step) {
        uint32_t blk_base = static_cast<uint32_t>(ts_block_steps.size() - 1) * TS_BLOCK_QUERIES_PER_STEP;
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            ts_block_pool, blk_base + 50);
      }

      struct { uint32_t N; uint32_t eps_bits; } fn_push = { HIDDEN, float_to_bits(RMS_EPS) };
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.rmsnorm);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          P.pipeline_layout_3, 0, 1, &D.final_norm, 0, nullptr);
      vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &fn_push);
      vkCmdDispatch(cmd, 1, 1, 1);
      barrier(cmd, B.act_b.buffer, B.act_bytes);
      // Block timestamps: end of final_norm, start of lm_head (pair 26, queries 52/53)
      if (can_single_submit && ts_block_pool != VK_NULL_HANDLE &&
          !ts_block_steps.empty() && ts_block_steps.back() == decode_step) {
        uint32_t blk_base = static_cast<uint32_t>(ts_block_steps.size() - 1) * TS_BLOCK_QUERIES_PER_STEP;
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            ts_block_pool, blk_base + 51);  // final_norm end
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            ts_block_pool, blk_base + 52);  // lm_head start
      }

      struct { uint32_t out_dim; uint32_t in_dim; } lm_push = { VOCAB, HIDDEN };
      if (lm_head_tiled) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.lm_head_tiled);
      } else {
        bind_matvec(cmd);
      }
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          P.pipeline_layout_3, 0, 1, &D.lm_head, 0, nullptr);
      vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &lm_push);
      if (lm_head_tiled) {
        constexpr uint32_t kLmBlockRows = 8;
        vkCmdDispatch(cmd, (VOCAB + kLmBlockRows - 1) / kLmBlockRows, 1, 1);
      } else {
        dispatch_matvec(cmd, VOCAB);
      }
      barrier(cmd, B.logits.buffer, VOCAB * 2);
      // Block timestamps: end of lm_head, start of argmax (pair 27, queries 54/55)
      if (can_single_submit && ts_block_pool != VK_NULL_HANDLE &&
          !ts_block_steps.empty() && ts_block_steps.back() == decode_step) {
        uint32_t blk_base = static_cast<uint32_t>(ts_block_steps.size() - 1) * TS_BLOCK_QUERIES_PER_STEP;
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            ts_block_pool, blk_base + 53);  // lm_head end
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            ts_block_pool, blk_base + 54);  // argmax start
      }

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.argmax);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          P.pipeline_layout_2, 0, 1, &D.argmax, 0, nullptr);
      uint32_t argmax_push = VOCAB;
      vkCmdPushConstants(cmd, P.pipeline_layout_2, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &argmax_push);
      vkCmdDispatch(cmd, 1, 1, 1);
      // Deferred token download: copy argmax_result to gen_tokens on-device
      if (defer_token_download) {
        VkBufferMemoryBarrier argmax_copy_barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        argmax_copy_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        argmax_copy_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        argmax_copy_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        argmax_copy_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        argmax_copy_barrier.buffer = B.argmax_result.buffer;
        argmax_copy_barrier.offset = 0;
        argmax_copy_barrier.size = 4;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 1, &argmax_copy_barrier, 0, nullptr);
        VkBufferCopy token_copy{};
        token_copy.srcOffset = 0;
        token_copy.dstOffset = static_cast<VkDeviceSize>(decode_step) * 4;
        token_copy.size = 4;
        vkCmdCopyBuffer(cmd, B.argmax_result.buffer, gen_tokens.buffer, 1, &token_copy);
        if (use_chunked_final_cmd) {
          VkBufferMemoryBarrier next_token_barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
          next_token_barrier.srcAccessMask =
              VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
          next_token_barrier.dstAccessMask =
              VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
          next_token_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
          next_token_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
          next_token_barrier.buffer = B.argmax_result.buffer;
          next_token_barrier.offset = 0;
          next_token_barrier.size = 4;
          vkCmdPipelineBarrier(cmd,
              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
              0, 0, nullptr, 1, &next_token_barrier, 0, nullptr);
        }
      }
      // Block timestamps: end of argmax region
      if (can_single_submit && ts_block_pool != VK_NULL_HANDLE &&
          !ts_block_steps.empty() && ts_block_steps.back() == decode_step) {
        uint32_t blk_base = static_cast<uint32_t>(ts_block_steps.size() - 1) * TS_BLOCK_QUERIES_PER_STEP;
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            ts_block_pool, blk_base + 55);  // argmax end
      }

      // GPU timestamp: write end timestamp before closing.
      // Fires when a start was written either via single-submit (embedding section)
      // or via the non-single-submit path (skip_layers first decode step).
      if (gpu_timestamps && ts_pool != VK_NULL_HANDLE &&
          !ts_decode_steps.empty() && ts_decode_steps.back() == decode_step) {
        uint32_t q_base = static_cast<uint32_t>(ts_decode_steps.size() - 1) * 2;
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            ts_pool, q_base + 1);
      }

      if (use_chunked_final_cmd) {
        const bool flush_chunk =
            chunk_recorded_steps >= decode_chunk_size || step + 1 == total_steps;
        if (flush_chunk) {
          dev_.end_command_buffer(cmd);
          dev_.submit_and_wait(cmd);
          ++result.decode_submit_count;
          ++result.chunked_decode_submit_count;
          chunk_cmd = VK_NULL_HANDLE;
          chunk_recorded_steps = 0;
        }
      } else {
        dev_.end_command_buffer(cmd);
        dev_.submit_and_wait(cmd);  // single-submit: submits all layers + LM head
        ++result.decode_submit_count;
      }

      // Capture final RMSNorm output for dump-step-components
      if (!dump_final_norm.empty()) {
        dev_.download_from_device(B.act_b, dump_final_norm.data(), HIDDEN * 2);
      }
    }

    uint32_t next_token = 0;
    if (!defer_token_download) {
      dev_.download_from_device(B.argmax_result, &next_token, 4);
    }
    if (debug_dump) {
      std::vector<uint16_t> logit_dump(VOCAB);
      dev_.download_from_device(B.logits, logit_dump.data(), VOCAB * 2);
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

      // Dump hidden state norm
      std::vector<uint16_t> hidden_fp16(HIDDEN);
      dev_.download_from_device(B.act_a, hidden_fp16.data(), HIDDEN * 2);
      double hidden_norm = 0.0;
      for (uint32_t i = 0; i < HIDDEN; ++i) {
        float v = half_to_float(hidden_fp16[i]);
        hidden_norm += static_cast<double>(v) * v;
      }
      hidden_norm = std::sqrt(hidden_norm);
      std::cerr << "  decode " << decode_step << " hidden_norm: " << hidden_norm << "\n";

      // Dump DeltaNet state matrix norm for each DeltaNet layer
      std::cerr << "  decode " << decode_step << " dn_state_norms:";
      for (uint32_t dn = 0; dn < NUM_DN_LAYERS; ++dn) {
        VkDeviceSize state_off = dn * B.dn_state_per_layer;
        size_t matrix_floats = DN_HEADS * DN_K_DIM * DN_V_DIM;
        auto staging = dev_.create_host_visible_buffer(matrix_floats * 4);
        auto cp_cmd = dev_.allocate_command_buffer();
        dev_.begin_command_buffer(cp_cmd);
        VkBufferCopy cp{state_off, 0, matrix_floats * 4};
        vkCmdCopyBuffer(cp_cmd, B.dn_state.buffer, staging.buffer, 1, &cp);
        dev_.end_command_buffer(cp_cmd);
        dev_.submit_and_wait(cp_cmd);
        std::vector<float> state_data(matrix_floats);
        dev_.download_from_device(staging, state_data.data(), matrix_floats * 4);
        dev_.destroy_buffer(staging);
        double sn = 0.0;
        for (float v : state_data) sn += static_cast<double>(v) * v;
        sn = std::sqrt(sn);
        std::cerr << " dn" << dn << "=" << sn;
      }
      std::cerr << "\n";
    }

    // --- Decode drift diagnostic: snapshot free-run state at target step ---
    if (is_target_diag) {
      if (drift_free_hidden.empty()) {
        drift_free_hidden.resize(HIDDEN);
      }
        dev_.download_from_device(B.act_a, drift_free_hidden.data(), HIDDEN * 2);
        {
          std::vector<uint16_t> logits_fp16(VOCAB);
          dev_.download_from_device(B.logits, logits_fp16.data(), VOCAB * 2);
          drift_free_logits.resize(VOCAB);
          for (uint32_t i = 0; i < VOCAB; ++i) {
            drift_free_logits[i] = half_to_float(logits_fp16[i]);
          }
        }
        drift_free_dn_state.resize(NUM_DN_LAYERS);
        for (uint32_t dn = 0; dn < NUM_DN_LAYERS; ++dn) {
          VkDeviceSize state_off = dn * B.dn_state_per_layer;
          size_t matrix_floats = DN_HEADS * DN_K_DIM * DN_V_DIM;
          drift_free_dn_state[dn].resize(matrix_floats);
          auto staging = dev_.create_host_visible_buffer(matrix_floats * 4);
          auto cp_cmd = dev_.allocate_command_buffer();
          dev_.begin_command_buffer(cp_cmd);
          VkBufferCopy cp{state_off, 0, matrix_floats * 4};
          vkCmdCopyBuffer(cp_cmd, B.dn_state.buffer, staging.buffer, 1, &cp);
          dev_.end_command_buffer(cp_cmd);
          dev_.submit_and_wait(cp_cmd);
          dev_.download_from_device(staging, drift_free_dn_state[dn].data(), matrix_floats * 4);
          dev_.destroy_buffer(staging);
        }
        // KV cache capture: download first tokens.size() positions per attention layer
        {
          VkDeviceSize kv_bytes_per_layer = static_cast<VkDeviceSize>(tokens.size()) * 2 * KV_HEADS * HEAD_DIM * 2;
          drift_free_kv_cache.resize(NUM_ATTN_LAYERS);
          for (uint32_t kv_ai = 0; kv_ai < NUM_ATTN_LAYERS; ++kv_ai) {
            VkDeviceSize layer_off = static_cast<VkDeviceSize>(kv_ai) * MAX_SEQ * 2 * KV_HEADS * HEAD_DIM * 2;
            drift_free_kv_cache[kv_ai].resize(tokens.size() * 2 * KV_HEADS * HEAD_DIM);
            auto staging = dev_.create_host_visible_buffer(kv_bytes_per_layer);
            auto cp_cmd = dev_.allocate_command_buffer();
            dev_.begin_command_buffer(cp_cmd);
            VkBufferCopy cp{layer_off, 0, kv_bytes_per_layer};
            vkCmdCopyBuffer(cp_cmd, B.kv_cache.buffer, staging.buffer, 1, &cp);
            dev_.end_command_buffer(cp_cmd);
            dev_.submit_and_wait(cp_cmd);
            dev_.download_from_device(staging, drift_free_kv_cache[kv_ai].data(), kv_bytes_per_layer);
            dev_.destroy_buffer(staging);
          }
        }
        drift_prefix_tokens = tokens;  // snapshot before push_back
    }
    if (!defer_token_download) {
      tokens.push_back(next_token);
      result.generated_tokens.push_back(next_token);
    }

    if (!is_prefill) {
      auto t_step_end = std::chrono::high_resolution_clock::now();
      double step_ms = std::chrono::duration<double, std::milli>(t_step_end - t_step_start).count();
      result.per_token_ms.push_back(step_ms);
    }

    if (verbose) {
      uint32_t decode_step = step - (prompt_len > 1 ? prompt_len - 1 : 0);
      std::cerr << "  decode " << decode_step << ": token " << next_token << "\n";
    }
  }

  // --- Deferred token download: populate result from gen_tokens buffer ---
  if (defer_token_download) {
    uint32_t num_generated = total_steps - decode_start;
    // num_generated = actual decode steps (excluding prefill)
    if (num_generated > 0) {
      std::vector<uint32_t> gen_host(num_generated);
      dev_.download_from_device(gen_tokens, gen_host.data(), num_generated * 4);
      for (uint32_t t : gen_host) {
        tokens.push_back(t);
        result.generated_tokens.push_back(t);
      }
    }
    dev_.destroy_buffer(gen_tokens);
  }

  if (dump_dn_recurrent_state_pre_layer >= 0
      && !dump_dn_recurrent_state_pre_file.empty()
      && !dump_dn_recurrent_state_pre_written) {
    result.error = "requested DeltaNet recurrent state pre-update sidecar was not captured for layer "
        + std::to_string(dump_dn_recurrent_state_pre_layer);
    return result;
  }

  // --- Post-loop: dump per-layer hiddens if requested ---
  if (!dump_layer_hidden.empty()) {
    std::cerr << "{\n";
    std::cerr << "  \"diagnostic\": \"layer_hiddens\",\n";
    std::cerr << "  \"decode_step\": " << dump_step_hiddens << ",\n";
    std::cerr << "  \"layers\": [\n";
    for (uint32_t layer = 0; layer < LAYERS; ++layer) {
      if (layer > 0) std::cerr << ",\n";
      size_t base = static_cast<size_t>(layer) * HIDDEN;
      // Compute norm for quick comparison
      double norm = 0.0;
      for (uint32_t i = 0; i < HIDDEN; ++i) {
        float v = half_to_float(dump_layer_hidden[base + i]);
        norm += static_cast<double>(v) * v;
      }
      norm = std::sqrt(norm);
      std::cerr << "    {\"layer\": " << layer
                << ", \"norm\": " << norm
                << ", \"hidden_fp16\": [";
      for (uint32_t i = 0; i < HIDDEN; ++i) {
        if (i > 0) std::cerr << ", ";
        std::cerr << dump_layer_hidden[base + i];
      }
      std::cerr << "]}";
    }
    std::cerr << "\n  ]\n}\n";
  }

  // --- Post-loop: dump component-level intermediates if requested ---
  if (!dump_input_hidden.empty()) {
    std::cerr << "{\n";
    std::cerr << "  \"diagnostic\": \"layer_components\",\n";
    std::cerr << "  \"decode_step\": " << dump_step_components << ",\n";

    // Final RMSNorm output (before LM head)
    if (!dump_final_norm.empty()) {
      double fn_norm = 0.0;
      for (uint32_t i = 0; i < HIDDEN; ++i) {
        float v = half_to_float(dump_final_norm[i]);
        fn_norm += static_cast<double>(v) * v;
      }
      fn_norm = std::sqrt(fn_norm);
      std::cerr << "  \"final_norm\": {\"norm\": " << fn_norm << ", \"hidden_fp16\": [";
      for (uint32_t i = 0; i < HIDDEN; ++i) {
        if (i > 0) std::cerr << ", ";
        std::cerr << dump_final_norm[i];
      }
      std::cerr << "]},\n";
    }

    std::cerr << "  \"layers\": [\n";
    for (uint32_t layer = 0; layer < LAYERS; ++layer) {
      if (layer > 0) std::cerr << ",\n";
      size_t base = static_cast<size_t>(layer) * HIDDEN;

      // Input norm (pre-layer act_a)
      double input_norm = 0.0;
      for (uint32_t i = 0; i < HIDDEN; ++i) {
        float v = half_to_float(dump_input_hidden[base + i]);
        input_norm += static_cast<double>(v) * v;
      }
      input_norm = std::sqrt(input_norm);

      // Mixer output norm (act_b before first residual_add)
      double mixer_output_norm = 0.0;
      for (uint32_t i = 0; i < HIDDEN; ++i) {
        float v = half_to_float(dump_mixer_output[base + i]);
        mixer_output_norm += static_cast<double>(v) * v;
      }
      mixer_output_norm = std::sqrt(mixer_output_norm);

      // Mixer residual norm (act_c = input + token_mixer_output)
      double mixer_norm = 0.0;
      for (uint32_t i = 0; i < HIDDEN; ++i) {
        float v = half_to_float(dump_mixer_residual[base + i]);
        mixer_norm += static_cast<double>(v) * v;
      }
      mixer_norm = std::sqrt(mixer_norm);

      // MLP RMSNorm output norm (act_a after post_norm, before gate/up)
      double mlp_normed_norm = 0.0;
      for (uint32_t i = 0; i < HIDDEN; ++i) {
        float v = half_to_float(dump_mlp_normed[base + i]);
        mlp_normed_norm += static_cast<double>(v) * v;
      }
      mlp_normed_norm = std::sqrt(mlp_normed_norm);

      // Post-MLP norm (act_a after layer)
      double mlp_norm = 0.0;
      for (uint32_t i = 0; i < HIDDEN; ++i) {
        float v = half_to_float(dump_post_mlp[base + i]);
        mlp_norm += static_cast<double>(v) * v;
      }
      mlp_norm = std::sqrt(mlp_norm);

      // MLP product norm (mlp_silu = silu(gate)*up after silu_gate, before down_matvec)
      size_t mlp_base = static_cast<size_t>(layer) * INTER;
      double mlp_gate_norm = 0.0;
      for (uint32_t i = 0; i < INTER; ++i) {
        float v = half_to_float(dump_mlp_gate[mlp_base + i]);
        mlp_gate_norm += static_cast<double>(v) * v;
      }
      mlp_gate_norm = std::sqrt(mlp_gate_norm);

      double mlp_up_norm = 0.0;
      for (uint32_t i = 0; i < INTER; ++i) {
        float v = half_to_float(dump_mlp_up[mlp_base + i]);
        mlp_up_norm += static_cast<double>(v) * v;
      }
      mlp_up_norm = std::sqrt(mlp_up_norm);

      double mlp_product_norm = 0.0;
      for (uint32_t i = 0; i < INTER; ++i) {
        float v = half_to_float(dump_mlp_product[mlp_base + i]);
        mlp_product_norm += static_cast<double>(v) * v;
      }
      mlp_product_norm = std::sqrt(mlp_product_norm);

      // Down output norm (act_b after down_matvec, before residual_add)
      double down_output_norm = 0.0;
      for (uint32_t i = 0; i < HIDDEN; ++i) {
        float v = half_to_float(dump_down_output[base + i]);
        down_output_norm += static_cast<double>(v) * v;
      }
      down_output_norm = std::sqrt(down_output_norm);

      std::cerr << "    {\"layer\": " << layer
                << ", \"input_norm\": " << input_norm
                << ", \"mixer_output_norm\": " << mixer_output_norm
                << ", \"mixer_norm\": " << mixer_norm
                << ", \"mlp_normed_norm\": " << mlp_normed_norm
                << ", \"mlp_norm\": " << mlp_norm
                << ", \"mlp_gate_norm\": " << mlp_gate_norm
                << ", \"mlp_up_norm\": " << mlp_up_norm
                << ", \"mlp_product_norm\": " << mlp_product_norm
                << ", \"down_output_norm\": " << down_output_norm
                << ", \"input_hidden_fp16\": [";
      for (uint32_t i = 0; i < HIDDEN; ++i) {
        if (i > 0) std::cerr << ", ";
        std::cerr << dump_input_hidden[base + i];
      }
      std::cerr << "], \"mixer_output_fp16\": [";
      for (uint32_t i = 0; i < HIDDEN; ++i) {
        if (i > 0) std::cerr << ", ";
        std::cerr << dump_mixer_output[base + i];
      }
      std::cerr << "], \"mixer_residual_fp16\": [";
      for (uint32_t i = 0; i < HIDDEN; ++i) {
        if (i > 0) std::cerr << ", ";
        std::cerr << dump_mixer_residual[base + i];
      }
      std::cerr << "], \"mlp_normed_fp16\": [";
      for (uint32_t i = 0; i < HIDDEN; ++i) {
        if (i > 0) std::cerr << ", ";
        std::cerr << dump_mlp_normed[base + i];
      }
      std::cerr << "], \"post_mlp_fp16\": [";
      for (uint32_t i = 0; i < HIDDEN; ++i) {
        if (i > 0) std::cerr << ", ";
        std::cerr << dump_post_mlp[base + i];
      }
      std::cerr << "], \"mlp_gate_fp16\": [";
      for (uint32_t i = 0; i < INTER; ++i) {
        if (i > 0) std::cerr << ", ";
        std::cerr << dump_mlp_gate[mlp_base + i];
      }
      std::cerr << "], \"mlp_up_fp16\": [";
      for (uint32_t i = 0; i < INTER; ++i) {
        if (i > 0) std::cerr << ", ";
        std::cerr << dump_mlp_up[mlp_base + i];
      }
      std::cerr << "], \"mlp_product_fp16\": [";
      for (uint32_t i = 0; i < INTER; ++i) {
        if (i > 0) std::cerr << ", ";
        std::cerr << dump_mlp_product[mlp_base + i];
      }
      std::cerr << "], \"down_output_fp16\": [";
      for (uint32_t i = 0; i < HIDDEN; ++i) {
        if (i > 0) std::cerr << ", ";
        std::cerr << dump_down_output[base + i];
      }
      const auto& schedule = model::Qwen35Config::layer_schedule();
      if (schedule[layer] != model::LayerKind::FullAttention) {
        size_t dn_key_base = static_cast<size_t>(layer) * DN_KEY_TOTAL;
        size_t dn_val_base = static_cast<size_t>(layer) * DN_VAL_TOTAL;
        size_t dn_gb_base = static_cast<size_t>(layer) * DN_HEADS * 2;

        std::cerr << "], \"dn_input_norm_fp16\": [";
        for (uint32_t i = 0; i < HIDDEN; ++i) {
          if (i > 0) std::cerr << ", ";
          std::cerr << dump_dn_input_norm[base + i];
        }
        std::cerr << "], \"dn_qkv_raw_fp16\": [";
        for (uint32_t i = 0; i < DN_CONV_DIM; ++i) {
          if (i > 0) std::cerr << ", ";
          std::cerr << dump_dn_qkv_raw[static_cast<size_t>(layer) * DN_CONV_DIM + i];
        }
        std::cerr << "], \"dn_conv_state_pre_fp16\": [";
        {
          size_t conv_base = static_cast<size_t>(layer) * DN_CONV_DIM * DN_CONV_KS;
          for (uint32_t i = 0; i < DN_CONV_DIM * DN_CONV_KS; ++i) {
            if (i > 0) std::cerr << ", ";
            std::cerr << dump_dn_conv_state_pre[conv_base + i];
          }
        }
        std::cerr << "], \"dn_q_fp16\": [";
        for (uint32_t i = 0; i < DN_KEY_TOTAL; ++i) {
          if (i > 0) std::cerr << ", ";
          std::cerr << dump_dn_q[dn_key_base + i];
        }
        std::cerr << "], \"dn_k_fp16\": [";
        for (uint32_t i = 0; i < DN_KEY_TOTAL; ++i) {
          if (i > 0) std::cerr << ", ";
          std::cerr << dump_dn_k[dn_key_base + i];
        }
        std::cerr << "], \"dn_v_fp16\": [";
        for (uint32_t i = 0; i < DN_VAL_TOTAL; ++i) {
          if (i > 0) std::cerr << ", ";
          std::cerr << dump_dn_v[dn_val_base + i];
        }
        std::cerr << "], \"dn_z_fp16\": [";
        for (uint32_t i = 0; i < DN_VAL_TOTAL; ++i) {
          if (i > 0) std::cerr << ", ";
          std::cerr << dump_dn_z[dn_val_base + i];
        }
        std::cerr << "], \"dn_a_fp16\": [";
        for (uint32_t i = 0; i < DN_HEADS; ++i) {
          if (i > 0) std::cerr << ", ";
          std::cerr << dump_dn_a[static_cast<size_t>(layer) * DN_HEADS + i];
        }
        std::cerr << "], \"dn_b_fp16\": [";
        for (uint32_t i = 0; i < DN_HEADS; ++i) {
          if (i > 0) std::cerr << ", ";
          std::cerr << dump_dn_b[static_cast<size_t>(layer) * DN_HEADS + i];
        }
        std::cerr << "], \"dn_g\": [";
        for (uint32_t i = 0; i < DN_HEADS; ++i) {
          if (i > 0) std::cerr << ", ";
          std::cerr << dump_dn_g_beta[dn_gb_base + i];
        }
        std::cerr << "], \"dn_g_bits\": [";
        for (uint32_t i = 0; i < DN_HEADS; ++i) {
          if (i > 0) std::cerr << ", ";
          std::cerr << float_to_bits(dump_dn_g_beta[dn_gb_base + i]);
        }
        std::cerr << "], \"dn_beta\": [";
        for (uint32_t i = 0; i < DN_HEADS; ++i) {
          if (i > 0) std::cerr << ", ";
          std::cerr << dump_dn_g_beta[dn_gb_base + DN_HEADS + i];
        }
        std::cerr << "], \"dn_beta_bits\": [";
        for (uint32_t i = 0; i < DN_HEADS; ++i) {
          if (i > 0) std::cerr << ", ";
          std::cerr << float_to_bits(dump_dn_g_beta[dn_gb_base + DN_HEADS + i]);
        }
        std::cerr << "], \"dn_core_fp16\": [";
        for (uint32_t i = 0; i < DN_VAL_TOTAL; ++i) {
          if (i > 0) std::cerr << ", ";
          std::cerr << dump_dn_core[dn_val_base + i];
        }
        std::cerr << "], \"dn_gated_fp16\": [";
        for (uint32_t i = 0; i < DN_VAL_TOTAL; ++i) {
          if (i > 0) std::cerr << ", ";
          std::cerr << dump_dn_gated[dn_val_base + i];
        }
        std::cerr << "], \"dn_out_fp16\": [";
        for (uint32_t i = 0; i < HIDDEN; ++i) {
          if (i > 0) std::cerr << ", ";
          std::cerr << dump_dn_out[base + i];
        }
      } else {
        size_t attn_q_base = static_cast<size_t>(layer) * Q_HEADS * HEAD_DIM;
        size_t attn_kv_base = static_cast<size_t>(layer) * KV_HEADS * HEAD_DIM;

        std::cerr << "], \"attn_q_norm_fp16\": [";
        for (uint32_t i = 0; i < Q_HEADS * HEAD_DIM; ++i) {
          if (i > 0) std::cerr << ", ";
          std::cerr << dump_attn_q_norm[attn_q_base + i];
        }
        std::cerr << "], \"attn_k_norm_fp16\": [";
        for (uint32_t i = 0; i < KV_HEADS * HEAD_DIM; ++i) {
          if (i > 0) std::cerr << ", ";
          std::cerr << dump_attn_k_norm[attn_kv_base + i];
        }
        std::cerr << "], \"attn_gate_fp16\": [";
        for (uint32_t i = 0; i < Q_HEADS * HEAD_DIM; ++i) {
          if (i > 0) std::cerr << ", ";
          std::cerr << dump_attn_gate[attn_q_base + i];
        }
        std::cerr << "], \"attn_v_fp16\": [";
        for (uint32_t i = 0; i < KV_HEADS * HEAD_DIM; ++i) {
          if (i > 0) std::cerr << ", ";
          std::cerr << dump_attn_v[attn_kv_base + i];
        }
        std::cerr << "], \"attn_gated_fp16\": [";
        for (uint32_t i = 0; i < Q_HEADS * HEAD_DIM; ++i) {
          if (i > 0) std::cerr << ", ";
          std::cerr << dump_attn_gated[attn_q_base + i];
        }
        std::cerr << "], \"attn_out_fp16\": [";
        for (uint32_t i = 0; i < HIDDEN; ++i) {
          if (i > 0) std::cerr << ", ";
          std::cerr << dump_attn_out[base + i];
        }
      }
      std::cerr << "]}";
    }
    std::cerr << "\n  ]\n}\n";
  }

  // --- Post-loop: run decode drift diagnostic if enabled ---
  if (diagnose_decode_drift && !drift_free_hidden.empty()) {
    constexpr uint32_t kTargetDecodeStep = 5;
    this->diagnose_decode_drift(drift_prefix_tokens, kTargetDecodeStep,
                          drift_free_hidden, drift_free_logits,
                          drift_free_dn_state, drift_free_kv_cache,
                          drift_free_layer_hidden);
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  result.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  // Compute timing split: prefill vs decode
  // per_token_ms covers decode-only steps.
  double decode_sum = 0.0;
  for (double ms : result.per_token_ms) decode_sum += ms;
  result.decode_ms = decode_sum;
  result.prefill_ms = result.elapsed_ms - decode_sum;

  // Retrieve GPU timestamp results
  if (gpu_timestamps && ts_pool != VK_NULL_HANDLE) {
    uint32_t num_ts = static_cast<uint32_t>(ts_decode_steps.size());
    if (num_ts > 0) {
      auto ts_results = dev_.get_timestamp_results(ts_pool, 0, num_ts * 2);
      if (ts_results.size() == num_ts * 2) {
        float period = dev_.capabilities().timestamp_period;  // ns per tick
        for (uint32_t i = 0; i < num_ts; ++i) {
          uint64_t t_start = ts_results[i * 2];
          uint64_t t_end = ts_results[i * 2 + 1];
          double elapsed_us = static_cast<double>(t_end - t_start) * period / 1000.0;
          result.per_token_gpu_us.push_back(elapsed_us);
          result.gpu_decode_us += elapsed_us;
        }
      }
    }
    dev_.destroy_query_pool(ts_pool);
  }

  // Retrieve block-level GPU timestamp results
  if (gpu_block_timestamps && ts_block_pool != VK_NULL_HANDLE) {
    uint32_t num_blk = static_cast<uint32_t>(ts_block_steps.size());
    if (num_blk > 0) {
      auto blk_results = dev_.get_timestamp_results(ts_block_pool, 0, num_blk * TS_BLOCK_QUERIES_PER_STEP);
      if (blk_results.size() == num_blk * TS_BLOCK_QUERIES_PER_STEP) {
        float period = dev_.capabilities().timestamp_period;
        for (uint32_t s = 0; s < num_blk; ++s) {
          uint32_t base = s * TS_BLOCK_QUERIES_PER_STEP;
          // Region 0: embedding (queries 0,1)
          {
            double us = static_cast<double>(blk_results[base + 1] - blk_results[base + 0]) * period / 1000.0;
            result.gpu_region_us["embedding"] += us;
          }
          // Regions 1..24: layer_0..layer_23 (queries 2+2*layer .. 2+2*layer+1)
          for (uint32_t layer = 0; layer < LAYERS; ++layer) {
            uint32_t q0 = base + 2 + layer * 2;
            double us = static_cast<double>(blk_results[q0 + 1] - blk_results[q0]) * period / 1000.0;
            std::string name = "layer_" + std::to_string(layer);
            result.gpu_region_us[name] += us;
          }
          // Region 25: final_norm (queries 50,51)
          {
            double us = static_cast<double>(blk_results[base + 51] - blk_results[base + 50]) * period / 1000.0;
            result.gpu_region_us["final_norm"] += us;
          }
          // Region 26: lm_head (queries 52,53)
          {
            double us = static_cast<double>(blk_results[base + 53] - blk_results[base + 52]) * period / 1000.0;
            result.gpu_region_us["lm_head"] += us;
          }
          // Region 27: argmax (queries 54,55)
          {
            double us = static_cast<double>(blk_results[base + 55] - blk_results[base + 54]) * period / 1000.0;
            result.gpu_region_us["argmax"] += us;
          }
        }
      }
    }
    dev_.destroy_query_pool(ts_block_pool);
  }

  return result;
}

// ---------------------------------------------------------------------------
// run_chunk_prefill: Run DeltaNet chunk rule per layer and upload correct state
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// gpu_chunk_prefill: GPU path for one DeltaNet layer chunk-rule computation.
// Q/K/V/g/beta are still CPU-collected; this gate moves only the chunk-rule
// computation to GPU, not full prefill collection/offload.
// ---------------------------------------------------------------------------
void DecodeSession::gpu_chunk_prefill(uint32_t dn_idx, PrefillChunkState& chunk, uint32_t seq_len, bool tiled) {
  const auto& P = *pipes_;
  const uint32_t chunk_size = 64;
  uint32_t total_seq = seq_len + ((chunk_size - seq_len % chunk_size) % chunk_size);
  uint32_t chunk_count = total_seq / chunk_size;
  uint32_t num_heads = DN_HEADS;
  uint32_t k_dim = DN_K_DIM;
  uint32_t v_dim = DN_V_DIM;
  constexpr uint32_t TILE_V = 16;

  // Compute q_scale_bits as float_to_bits(1/sqrt(k_dim))
  float inv_sqrt_kdim = 1.0f / std::sqrt(static_cast<float>(k_dim));
  uint32_t q_scale_bits;
  memcpy(&q_scale_bits, &inv_sqrt_kdim, sizeof(q_scale_bits));

  // Sizes for host-visible buffers
  VkDeviceSize sz_q    = static_cast<VkDeviceSize>(num_heads * seq_len * k_dim) * sizeof(float);
  VkDeviceSize sz_k    = sz_q;
  VkDeviceSize sz_v    = static_cast<VkDeviceSize>(num_heads * seq_len * v_dim) * sizeof(float);
  VkDeviceSize sz_g    = static_cast<VkDeviceSize>(num_heads * seq_len) * sizeof(float);
  VkDeviceSize sz_beta = sz_g;
  VkDeviceSize sz_out  = (static_cast<VkDeviceSize>(num_heads * total_seq * v_dim) +
                           static_cast<VkDeviceSize>(num_heads * k_dim * v_dim)) * sizeof(float);
  VkDeviceSize sz_init = static_cast<VkDeviceSize>(num_heads * k_dim * v_dim) * sizeof(float);

  // Allocate host-visible buffers
  auto q_buf     = dev_.create_host_visible_buffer(sz_q);
  auto k_buf     = dev_.create_host_visible_buffer(sz_k);
  auto v_buf     = dev_.create_host_visible_buffer(sz_v);
  auto g_buf     = dev_.create_host_visible_buffer(sz_g);
  auto beta_buf  = dev_.create_host_visible_buffer(sz_beta);
  auto out_buf   = dev_.create_host_visible_buffer(sz_out);
  auto init_buf  = dev_.create_host_visible_buffer(sz_init);

  // Rearrange from collected [token][head][dim] to shader's [head][token][dim]
  // Query/key/value: collected as [seq_len * num_heads * dim] in token-major order
  auto rearrange = [](const std::vector<float>& token_major,
                      uint32_t num_heads, uint32_t seq, uint32_t dim) {
    std::vector<float> head_major(num_heads * seq * dim);
    for (uint32_t t = 0; t < seq; ++t) {
      for (uint32_t h = 0; h < num_heads; ++h) {
        for (uint32_t d = 0; d < dim; ++d) {
          head_major[(h * seq + t) * dim + d] = token_major[(t * num_heads + h) * dim + d];
        }
      }
    }
    return head_major;
  };
  // g and beta: collected as [seq_len * num_heads] in [token][head] order
  auto rearrange_scalar = [](const std::vector<float>& token_major,
                             uint32_t num_heads, uint32_t seq) {
    std::vector<float> head_major(num_heads * seq);
    for (uint32_t t = 0; t < seq; ++t) {
      for (uint32_t h = 0; h < num_heads; ++h) {
        head_major[h * seq + t] = token_major[t * num_heads + h];
      }
    }
    return head_major;
  };

  auto head_q = rearrange(chunk.query, num_heads, seq_len, k_dim);
  auto head_k = rearrange(chunk.key, num_heads, seq_len, k_dim);
  auto head_v = rearrange(chunk.value, num_heads, seq_len, v_dim);
  auto head_g = rearrange_scalar(chunk.g, num_heads, seq_len);
  auto head_beta = rearrange_scalar(chunk.beta, num_heads, seq_len);

  memcpy(q_buf.mapped,    head_q.data(),    static_cast<size_t>(sz_q));
  memcpy(k_buf.mapped,    head_k.data(),    static_cast<size_t>(sz_k));
  memcpy(v_buf.mapped,    head_v.data(),    static_cast<size_t>(sz_v));
  memcpy(g_buf.mapped,    head_g.data(),    static_cast<size_t>(sz_g));
  memcpy(beta_buf.mapped, head_beta.data(), static_cast<size_t>(sz_beta));
  // Zero-init initial state
  memset(init_buf.mapped, 0, static_cast<size_t>(sz_init));

  // Update descriptor set bindings 0..6
  dev_.update_descriptor_set(dsets_->dn_chunk_prefill, 0, q_buf);
  dev_.update_descriptor_set(dsets_->dn_chunk_prefill, 1, k_buf);
  dev_.update_descriptor_set(dsets_->dn_chunk_prefill, 2, v_buf);
  dev_.update_descriptor_set(dsets_->dn_chunk_prefill, 3, g_buf);
  dev_.update_descriptor_set(dsets_->dn_chunk_prefill, 4, beta_buf);
  dev_.update_descriptor_set(dsets_->dn_chunk_prefill, 5, out_buf);
  dev_.update_descriptor_set(dsets_->dn_chunk_prefill, 6, init_buf);

  // Push constants struct matching shader order
  struct PushConsts {
    uint32_t num_heads;
    uint32_t seq_len;
    uint32_t k_dim;
    uint32_t v_dim;
    uint32_t chunk_size;
    uint32_t q_scale_bits;
    uint32_t total_seq;
    uint32_t chunk_count;
    uint32_t use_qk_l2norm;
    uint32_t base_head;
  };
  PushConsts push_consts = {num_heads, seq_len, k_dim, v_dim, chunk_size,
                            q_scale_bits, total_seq, chunk_count, 0, 0};

  if (tiled) {
    VkCommandBuffer cmd = dev_.allocate_command_buffer();
    dev_.begin_command_buffer(cmd);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.deltanet_chunk_prefill_tiled);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            P.pipeline_layout_cp, 0, 1, &dsets_->dn_chunk_prefill, 0, nullptr);
    vkCmdPushConstants(cmd, P.pipeline_layout_cp, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(PushConsts), &push_consts);
    const uint32_t tile_count_v = (v_dim + TILE_V - 1) / TILE_V;
    vkCmdDispatch(cmd, num_heads, tile_count_v, 1);
    dev_.end_command_buffer(cmd);
    dev_.submit_and_wait(cmd);
  } else {
    // Conservative correctness workaround: submit each head as an independent
    // one-dispatch command buffer.  Probe runs showed that serialising per-head
    // dispatches within a single submission via pipeline barrier was not
    // sufficient for correct scratch / dispatch interaction on this device;
    // submitting each head separately forces the driver to complete the
    // scratch writes before the next head starts.
    for (uint32_t h = 0; h < num_heads; ++h) {
      VkCommandBuffer cmd = dev_.allocate_command_buffer();
      dev_.begin_command_buffer(cmd);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.deltanet_chunk_prefill);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              P.pipeline_layout_cp, 0, 1, &dsets_->dn_chunk_prefill, 0, nullptr);
      push_consts.base_head = h;
      vkCmdPushConstants(cmd, P.pipeline_layout_cp, VK_SHADER_STAGE_COMPUTE_BIT,
                         0, sizeof(PushConsts), &push_consts);
      vkCmdDispatch(cmd, 1, 1, 1);
      dev_.end_command_buffer(cmd);
      dev_.submit_and_wait(cmd);
    }
  }

  // Diagnostic: compare GPU output against CPU reference when env var is set
  if (const char* env_cmp = std::getenv("SPOCK_GPU_CHUNK_PREFILL_COMPARE");
      env_cmp && env_cmp[0] == '1' && env_cmp[1] == '\0') {
    DeltaNetChunkConfig ref_config;
    ref_config.num_heads = num_heads;
    ref_config.sequence_length = seq_len;
    ref_config.key_dim = k_dim;
    ref_config.value_dim = v_dim;
    ref_config.chunk_size = chunk_size;
    ref_config.use_qk_l2norm = false;

    DeltaNetChunkInputs ref_inputs;
    ref_inputs.query = head_q;
    ref_inputs.key = head_k;
    ref_inputs.value = head_v;
    ref_inputs.g = head_g;
    ref_inputs.beta = head_beta;

    auto ref_out = run_deltanet_chunk_rule(ref_config, ref_inputs);

    const float* gpu_data = static_cast<const float*>(out_buf.mapped);
    const std::size_t st_off = static_cast<std::size_t>(num_heads) * total_seq * v_dim;

    double max_rel_core = 0.0, max_rel_state = 0.0;
    double max_abs_core = 0.0, max_abs_state = 0.0;
    uint64_t nan_count = 0;

    // Compare core_attn_out: CPU [head][seq_len][v_dim] vs GPU [head][total_seq][v_dim]
    for (uint32_t h = 0; h < num_heads; ++h) {
      for (uint32_t t = 0; t < seq_len; ++t) {
        for (uint32_t vd = 0; vd < v_dim; ++vd) {
          float cpu_val = ref_out.core_attn_out[static_cast<std::size_t>(h * seq_len + t) * v_dim + vd];
          float gpu_val = gpu_data[(static_cast<std::size_t>(h) * total_seq + t) * v_dim + vd];
          if (std::isnan(cpu_val) || std::isinf(cpu_val) ||
              std::isnan(gpu_val) || std::isinf(gpu_val)) {
            ++nan_count;
            continue;
          }
          double abs_diff = std::abs(static_cast<double>(cpu_val) - static_cast<double>(gpu_val));
          double rel_diff = abs_diff / (std::abs(static_cast<double>(cpu_val)) + 1e-30);
          if (rel_diff > max_rel_core) max_rel_core = rel_diff;
          if (abs_diff > max_abs_core) max_abs_core = abs_diff;
        }
      }
    }

    // Compare final_state: both are [head][k_dim][v_dim]
    for (uint32_t h = 0; h < num_heads; ++h) {
      for (uint32_t kd = 0; kd < k_dim; ++kd) {
        for (uint32_t vd = 0; vd < v_dim; ++vd) {
          float cpu_val = ref_out.final_state[static_cast<std::size_t>((h * k_dim + kd) * v_dim + vd)];
          float gpu_val = gpu_data[st_off + (static_cast<std::size_t>(h) * k_dim + kd) * v_dim + vd];
          if (std::isnan(cpu_val) || std::isinf(cpu_val) ||
              std::isnan(gpu_val) || std::isinf(gpu_val)) {
            ++nan_count;
            continue;
          }
          double abs_diff = std::abs(static_cast<double>(cpu_val) - static_cast<double>(gpu_val));
          double rel_diff = abs_diff / (std::abs(static_cast<double>(cpu_val)) + 1e-30);
          if (rel_diff > max_rel_state) max_rel_state = rel_diff;
          if (abs_diff > max_abs_state) max_abs_state = abs_diff;
        }
      }
    }

    // Input range stats from rearranged vectors
    auto abs_max_of = [](const std::vector<float>& v) -> float {
      auto it = std::max_element(v.begin(), v.end(),
          [](float a, float b) { return std::abs(a) < std::abs(b); });
      return it != v.end() ? std::abs(*it) : 0.0f;
    };
    float q_abs_max = abs_max_of(head_q);
    float k_abs_max = abs_max_of(head_k);
    float v_abs_max = abs_max_of(head_v);

    auto [g_min_it, g_max_it] = std::minmax_element(head_g.begin(), head_g.end());
    auto [beta_min_it, beta_max_it] = std::minmax_element(head_beta.begin(), head_beta.end());
    float g_min    = g_min_it    != head_g.end()    ? *g_min_it    : 0.0f;
    float g_max    = g_max_it    != head_g.end()    ? *g_max_it    : 0.0f;
    float beta_min = beta_min_it != head_beta.end() ? *beta_min_it : 0.0f;
    float beta_max = beta_max_it != head_beta.end() ? *beta_max_it : 0.0f;

    std::fprintf(stderr,
        "SPOCK_GPU_CHUNK_PREFILL_COMPARE%s layer=%u seq_len=%u "
        "max_rel_core=%.6e max_rel_state=%.6e "
        "max_abs_core=%.6e max_abs_state=%.6e nan_count=%zu "
        "q_abs_max=%.6e k_abs_max=%.6e v_abs_max=%.6e "
        "g_min=%.6e g_max=%.6e beta_min=%.6e beta_max=%.6e\n",
        tiled ? " dispatch=tiled" : "",
        dn_idx, seq_len, max_rel_core, max_rel_state,
        max_abs_core, max_abs_state, static_cast<std::size_t>(nan_count),
        q_abs_max, k_abs_max, v_abs_max,
        g_min, g_max, beta_min, beta_max);
  }

  // Read back output
  const float* out_data = static_cast<const float*>(out_buf.mapped);
  const std::size_t state_offset = static_cast<std::size_t>(num_heads) * total_seq * v_dim;

  // Save core_attn_out for last token (token seq_len-1)
  auto& attn_last = chunk_core_attn_out_last_[dn_idx];
  attn_last.resize(num_heads * v_dim);
  for (uint32_t h = 0; h < num_heads; ++h) {
    for (uint32_t vd = 0; vd < v_dim; ++vd) {
      attn_last[h * v_dim + vd] =
          out_data[(h * total_seq + (seq_len - 1)) * v_dim + vd];
    }
  }

  // Upload final_state to GPU state buffer
  size_t state_matrix_bytes = static_cast<size_t>(num_heads * k_dim * v_dim) * sizeof(float);
  auto upload_buf = dev_.create_host_visible_buffer(state_matrix_bytes);
  memcpy(upload_buf.mapped, out_data + state_offset, state_matrix_bytes);

  VkDeviceSize state_offset_gpu = static_cast<VkDeviceSize>(dn_idx) * bufs_->dn_state_per_layer;
  {
    auto copy_cmd = dev_.allocate_command_buffer();
    dev_.begin_command_buffer(copy_cmd);
    VkBufferCopy copy{0, state_offset_gpu, static_cast<VkDeviceSize>(state_matrix_bytes)};
    vkCmdCopyBuffer(copy_cmd, upload_buf.buffer, bufs_->dn_state.buffer, 1, &copy);
    dev_.end_command_buffer(copy_cmd);
    dev_.submit_and_wait(copy_cmd);
  }
  dev_.destroy_buffer(upload_buf);

  // Cleanup host-visible buffers
  dev_.destroy_buffer(q_buf);
  dev_.destroy_buffer(k_buf);
  dev_.destroy_buffer(v_buf);
  dev_.destroy_buffer(g_buf);
  dev_.destroy_buffer(beta_buf);
  dev_.destroy_buffer(out_buf);
  dev_.destroy_buffer(init_buf);
}

// ---------------------------------------------------------------------------
// gpu_chunk_prefill_from_gpu_collect: GPU path for one DeltaNet layer chunk-rule
// computation using GPU-collected (device-local persistent) Q/K/V/g/beta buffers.
// Binds the per-layer segment directly to the selected chunk-prefill shader
// without CPU upload.
// ---------------------------------------------------------------------------
void DecodeSession::gpu_chunk_prefill_from_gpu_collect(uint32_t dn_idx, PrefillChunkState& chunk, uint32_t seq_len, bool tiled) {
  const auto& P = *pipes_;
  const auto& B = *bufs_;
  const uint32_t chunk_size = 64;
  uint32_t total_seq = seq_len + ((chunk_size - seq_len % chunk_size) % chunk_size);
  uint32_t chunk_count = total_seq / chunk_size;
  uint32_t num_heads = DN_HEADS;
  uint32_t k_dim = DN_K_DIM;
  uint32_t v_dim = DN_V_DIM;
  constexpr uint32_t TILE_V = 16;

  float inv_sqrt_kdim = 1.0f / std::sqrt(static_cast<float>(k_dim));
  uint32_t q_scale_bits;
  memcpy(&q_scale_bits, &inv_sqrt_kdim, sizeof(q_scale_bits));

  VkDeviceSize sz_q    = static_cast<VkDeviceSize>(num_heads * seq_len * k_dim) * sizeof(float);
  VkDeviceSize sz_k    = sz_q;
  VkDeviceSize sz_v    = static_cast<VkDeviceSize>(num_heads * seq_len * v_dim) * sizeof(float);
  VkDeviceSize sz_g    = static_cast<VkDeviceSize>(num_heads * seq_len) * sizeof(float);
  VkDeviceSize sz_beta = sz_g;
  VkDeviceSize sz_out  = (static_cast<VkDeviceSize>(num_heads * total_seq * v_dim) +
                           static_cast<VkDeviceSize>(num_heads * k_dim * v_dim)) * sizeof(float);
  VkDeviceSize sz_init = static_cast<VkDeviceSize>(num_heads * k_dim * v_dim) * sizeof(float);

  // Per-layer offsets into persistent device-local buffers
  VkDeviceSize q_offset = static_cast<VkDeviceSize>(dn_idx) * align_storage_offset(sz_q);
  VkDeviceSize v_offset = static_cast<VkDeviceSize>(dn_idx) * align_storage_offset(sz_v);
  VkDeviceSize gb_offset = static_cast<VkDeviceSize>(dn_idx) * align_storage_offset(sz_g);

  // Determine whether we can keep chunk-prefill output entirely on GPU.
  // Fast path: tiled dispatch, no CPU comparison active.
  bool chunk_compare_active = false;
  bool collect_compare_active = false;
  {
    const char* env_cmp = std::getenv("SPOCK_GPU_CHUNK_PREFILL_COMPARE");
    if (env_cmp && env_cmp[0] == '1' && env_cmp[1] == '\0') chunk_compare_active = true;
    const char* env_cmp2 = std::getenv("SPOCK_GPU_COLLECT_PREFILL_COMPARE");
    if (env_cmp2 && env_cmp2[0] == '1' && env_cmp2[1] == '\0') collect_compare_active = true;
  }
  bool use_gpu_handoff = tiled && !chunk_compare_active && !collect_compare_active;

  // init_buf: device-local + vkCmdFillBuffer for GPU-handoff fast path,
  // host-visible + CPU memset for compare/fallback/readback paths.
  VulkanDevice::Buffer init_buf;
  if (use_gpu_handoff) {
    init_buf = dev_.create_device_local_buffer(sz_init);
  } else {
    init_buf = dev_.create_host_visible_buffer(sz_init);
    memset(init_buf.mapped, 0, static_cast<size_t>(sz_init));
  }

  // out_buf: device-local for GPU handoff (no CPU access needed),
  // host-visible for compare/readback/CPU upload paths.
  VulkanDevice::Buffer out_buf;
  if (use_gpu_handoff) {
    out_buf = dev_.create_device_local_buffer(sz_out);
  } else {
    out_buf = dev_.create_host_visible_buffer(sz_out);
  }
  // Bind persistent GPU-collected buffers directly (no CPU upload)
  dev_.update_descriptor_set(dsets_->dn_chunk_prefill, 0, B.dn_persist_q, q_offset, sz_q);
  dev_.update_descriptor_set(dsets_->dn_chunk_prefill, 1, B.dn_persist_k, q_offset, sz_k);
  dev_.update_descriptor_set(dsets_->dn_chunk_prefill, 2, B.dn_persist_v, v_offset, sz_v);
  dev_.update_descriptor_set(dsets_->dn_chunk_prefill, 3, B.dn_persist_g, gb_offset, sz_g);
  dev_.update_descriptor_set(dsets_->dn_chunk_prefill, 4, B.dn_persist_beta, gb_offset, sz_beta);
  dev_.update_descriptor_set(dsets_->dn_chunk_prefill, 5, out_buf);
  dev_.update_descriptor_set(dsets_->dn_chunk_prefill, 6, init_buf);

  struct PushConsts {
    uint32_t num_heads;
    uint32_t seq_len;
    uint32_t k_dim;
    uint32_t v_dim;
    uint32_t chunk_size;
    uint32_t q_scale_bits;
    uint32_t total_seq;
    uint32_t chunk_count;
    uint32_t use_qk_l2norm;
    uint32_t base_head;
  };
  PushConsts push_consts = {num_heads, seq_len, k_dim, v_dim, chunk_size,
                            q_scale_bits, total_seq, chunk_count, 0, 0};

  if (tiled) {
    VkCommandBuffer cmd = dev_.allocate_command_buffer();
    dev_.begin_command_buffer(cmd);
    if (use_gpu_handoff) {
      vkCmdFillBuffer(cmd, init_buf.buffer, 0, sz_init, 0);
      VkBufferMemoryBarrier bmb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
      bmb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      bmb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      bmb.buffer = init_buf.buffer;
      bmb.offset = 0;
      bmb.size = VK_WHOLE_SIZE;
      vkCmdPipelineBarrier(cmd,
          VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          0, 0, nullptr, 1, &bmb, 0, nullptr);
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.deltanet_chunk_prefill_tiled);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            P.pipeline_layout_cp, 0, 1, &dsets_->dn_chunk_prefill, 0, nullptr);
    vkCmdPushConstants(cmd, P.pipeline_layout_cp, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(PushConsts), &push_consts);
    const uint32_t tile_count_v = (v_dim + TILE_V - 1) / TILE_V;
    vkCmdDispatch(cmd, num_heads, tile_count_v, 1);

    if (use_gpu_handoff) {
      // GPU handoff: final_state copy + fp32→fp16 extraction both on GPU.
      // No CPU readback, no float_to_half, no host-visible staging.

      // Barrier: tiled dispatch writes → transfer read for final_state copy
      {
        VkBufferMemoryBarrier bmb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        bmb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        bmb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bmb.buffer = out_buf.buffer;
        bmb.offset = 0;
        bmb.size = sz_out;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 1, &bmb, 0, nullptr);
      }

      // Copy final_state from out_buf to dn_state (GPU→GPU)
      VkDeviceSize state_offset_gpu = static_cast<VkDeviceSize>(dn_idx) * bufs_->dn_state_per_layer;
      VkDeviceSize state_matrix_bytes = static_cast<VkDeviceSize>(num_heads * k_dim * v_dim) * sizeof(float);
      VkDeviceSize src_state_off = static_cast<VkDeviceSize>(num_heads) * total_seq * v_dim * sizeof(float);
      VkBufferCopy state_copy{src_state_off, state_offset_gpu, state_matrix_bytes};
      vkCmdCopyBuffer(cmd, out_buf.buffer, bufs_->dn_state.buffer, 1, &state_copy);

      // Barrier: chunk shader write → shader read for last_to_fp16
      {
        VkBufferMemoryBarrier bmb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        bmb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        bmb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bmb.buffer = out_buf.buffer;
        bmb.offset = 0;
        bmb.size = sz_out;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 1, &bmb, 0, nullptr);
      }

      // Extract last-token core_attn_out via deltanet_chunk_last_to_fp16 shader
      VkDeviceSize attn_dst_offset = static_cast<VkDeviceSize>(dn_idx) * DN_VAL_TOTAL * 2;
      dev_.update_descriptor_set(dsets_->dn_chunk_last_to_fp16, 0, out_buf);
      dev_.update_descriptor_set(dsets_->dn_chunk_last_to_fp16, 1,
                                 dn_chunk_attn_out_, attn_dst_offset, DN_VAL_TOTAL * 2);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.deltanet_chunk_last_to_fp16);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              P.pipeline_layout_2, 0, 1, &dsets_->dn_chunk_last_to_fp16, 0, nullptr);
      {
        struct { uint32_t seq_len; uint32_t total_seq; } l2f_pc = {seq_len, total_seq};
        vkCmdPushConstants(cmd, P.pipeline_layout_2, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &l2f_pc);
      }
      vkCmdDispatch(cmd, (DN_VAL_TOTAL + 63) / 64, 1, 1);

      // Barrier: last_to_fp16 write → transfer read for correct_last_token_hidden
      {
        VkBufferMemoryBarrier bmb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        bmb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        bmb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bmb.buffer = dn_chunk_attn_out_.buffer;
        bmb.offset = attn_dst_offset;
        bmb.size = DN_VAL_TOTAL * 2;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 1, &bmb, 0, nullptr);
      }

      dev_.end_command_buffer(cmd);
      dev_.submit_and_wait(cmd);

      gpu_chunk_handoff_ready_[dn_idx] = true;
    } else {
      dev_.end_command_buffer(cmd);
      dev_.submit_and_wait(cmd);
    }
  } else {
    // Serial per-head dispatch (same conservative pattern as gpu_chunk_prefill)
    for (uint32_t h = 0; h < num_heads; ++h) {
      VkCommandBuffer cmd = dev_.allocate_command_buffer();
      dev_.begin_command_buffer(cmd);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.deltanet_chunk_prefill);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              P.pipeline_layout_cp, 0, 1, &dsets_->dn_chunk_prefill, 0, nullptr);
      push_consts.base_head = h;
      vkCmdPushConstants(cmd, P.pipeline_layout_cp, VK_SHADER_STAGE_COMPUTE_BIT,
                         0, sizeof(PushConsts), &push_consts);
      vkCmdDispatch(cmd, 1, 1, 1);
      dev_.end_command_buffer(cmd);
      dev_.submit_and_wait(cmd);
    }
  }

  // Diagnostic: compare GPU output against CPU reference when COMPARE=1
  if (chunk_compare_active) {
    if (chunk.query.empty()) {
      throw std::runtime_error(
          "SPOCK_GPU_CHUNK_PREFILL_COMPARE=1 but CPU chunk vectors are empty "
          "(GPU-collected path skips CPU collection when no compare is active)");
    }
    // Rearrange CPU-collected data to head-major for reference computation
    auto rearrange = [](const std::vector<float>& token_major,
                        uint32_t n_heads, uint32_t seq, uint32_t dim) {
      std::vector<float> head_major(n_heads * seq * dim);
      for (uint32_t t = 0; t < seq; ++t) {
        for (uint32_t h = 0; h < n_heads; ++h) {
          for (uint32_t d = 0; d < dim; ++d) {
            head_major[(h * seq + t) * dim + d] = token_major[(t * n_heads + h) * dim + d];
          }
        }
      }
      return head_major;
    };
    auto rearrange_scalar = [](const std::vector<float>& token_major,
                               uint32_t n_heads, uint32_t seq) {
      std::vector<float> head_major(n_heads * seq);
      for (uint32_t t = 0; t < seq; ++t) {
        for (uint32_t h = 0; h < n_heads; ++h) {
          head_major[h * seq + t] = token_major[t * n_heads + h];
        }
      }
      return head_major;
    };

    auto head_q = rearrange(chunk.query, num_heads, seq_len, k_dim);
    auto head_k = rearrange(chunk.key, num_heads, seq_len, k_dim);
    auto head_v = rearrange(chunk.value, num_heads, seq_len, v_dim);
    auto head_g = rearrange_scalar(chunk.g, num_heads, seq_len);
    auto head_beta = rearrange_scalar(chunk.beta, num_heads, seq_len);

    DeltaNetChunkConfig ref_config;
    ref_config.num_heads = num_heads;
    ref_config.sequence_length = seq_len;
    ref_config.key_dim = k_dim;
    ref_config.value_dim = v_dim;
    ref_config.chunk_size = chunk_size;
    ref_config.use_qk_l2norm = false;

    DeltaNetChunkInputs ref_inputs;
    ref_inputs.query = head_q;
    ref_inputs.key = head_k;
    ref_inputs.value = head_v;
    ref_inputs.g = head_g;
    ref_inputs.beta = head_beta;

    auto ref_out = run_deltanet_chunk_rule(ref_config, ref_inputs);

    const float* gpu_data = static_cast<const float*>(out_buf.mapped);
    const std::size_t st_off = static_cast<std::size_t>(num_heads) * total_seq * v_dim;

    double max_rel_core = 0.0, max_rel_state = 0.0;
    double max_abs_core = 0.0, max_abs_state = 0.0;
    uint64_t nan_count = 0;

    for (uint32_t h = 0; h < num_heads; ++h) {
      for (uint32_t t = 0; t < seq_len; ++t) {
        for (uint32_t vd = 0; vd < v_dim; ++vd) {
          float cpu_val = ref_out.core_attn_out[(static_cast<std::size_t>(h) * seq_len + t) * v_dim + vd];
          float gpu_val = gpu_data[(static_cast<std::size_t>(h) * total_seq + t) * v_dim + vd];
          if (std::isnan(cpu_val) || std::isinf(cpu_val) ||
              std::isnan(gpu_val) || std::isinf(gpu_val)) {
            ++nan_count;
            continue;
          }
          double abs_diff = std::abs(static_cast<double>(cpu_val) - static_cast<double>(gpu_val));
          double rel_diff = abs_diff / (std::abs(static_cast<double>(cpu_val)) + 1e-30);
          if (rel_diff > max_rel_core) max_rel_core = rel_diff;
          if (abs_diff > max_abs_core) max_abs_core = abs_diff;
        }
      }
    }

    for (uint32_t h = 0; h < num_heads; ++h) {
      for (uint32_t kd = 0; kd < k_dim; ++kd) {
        for (uint32_t vd = 0; vd < v_dim; ++vd) {
          float cpu_val = ref_out.final_state[(static_cast<std::size_t>(h) * k_dim + kd) * v_dim + vd];
          float gpu_val = gpu_data[st_off + (static_cast<std::size_t>(h) * k_dim + kd) * v_dim + vd];
          if (std::isnan(cpu_val) || std::isinf(cpu_val) ||
              std::isnan(gpu_val) || std::isinf(gpu_val)) {
            ++nan_count;
            continue;
          }
          double abs_diff = std::abs(static_cast<double>(cpu_val) - static_cast<double>(gpu_val));
          double rel_diff = abs_diff / (std::abs(static_cast<double>(cpu_val)) + 1e-30);
          if (rel_diff > max_rel_state) max_rel_state = rel_diff;
          if (abs_diff > max_abs_state) max_abs_state = abs_diff;
        }
      }
    }

    auto abs_max_of = [](const std::vector<float>& v) -> float {
      auto it = std::max_element(v.begin(), v.end(),
          [](float a, float b) { return std::abs(a) < std::abs(b); });
      return it != v.end() ? std::abs(*it) : 0.0f;
    };
    float q_abs_max = abs_max_of(head_q);
    float k_abs_max = abs_max_of(head_k);
    float v_abs_max = abs_max_of(head_v);

    auto [g_min_it, g_max_it] = std::minmax_element(head_g.begin(), head_g.end());
    auto [beta_min_it, beta_max_it] = std::minmax_element(head_beta.begin(), head_beta.end());
    float g_min    = g_min_it    != head_g.end()    ? *g_min_it    : 0.0f;
    float g_max    = g_max_it    != head_g.end()    ? *g_max_it    : 0.0f;
    float beta_min = beta_min_it != head_beta.end() ? *beta_min_it : 0.0f;
    float beta_max = beta_max_it != head_beta.end() ? *beta_max_it : 0.0f;

    std::fprintf(stderr,
        "SPOCK_GPU_CHUNK_PREFILL_COMPARE source=gpu_collect%s layer=%u seq_len=%u "
        "max_rel_core=%.6e max_rel_state=%.6e "
        "max_abs_core=%.6e max_abs_state=%.6e nan_count=%zu "
        "q_abs_max=%.6e k_abs_max=%.6e v_abs_max=%.6e "
        "g_min=%.6e g_max=%.6e beta_min=%.6e beta_max=%.6e\n",
        tiled ? " dispatch=tiled" : "",
        dn_idx, seq_len, max_rel_core, max_rel_state,
        max_abs_core, max_abs_state, static_cast<std::size_t>(nan_count),
        q_abs_max, k_abs_max, v_abs_max,
        g_min, g_max, beta_min, beta_max);
  }

  if (!use_gpu_handoff) {
    // Read back output: save core_attn_out_last and upload final_state (CPU path)
    const float* out_data = static_cast<const float*>(out_buf.mapped);
    const std::size_t state_offset = static_cast<std::size_t>(num_heads) * total_seq * v_dim;

    auto& attn_last = chunk_core_attn_out_last_[dn_idx];
    attn_last.resize(num_heads * v_dim);
    for (uint32_t h = 0; h < num_heads; ++h) {
      for (uint32_t vd = 0; vd < v_dim; ++vd) {
        attn_last[h * v_dim + vd] =
            out_data[(static_cast<std::size_t>(h) * total_seq + (seq_len - 1)) * v_dim + vd];
      }
    }

    size_t state_matrix_bytes = static_cast<size_t>(num_heads * k_dim * v_dim) * sizeof(float);
    auto upload_buf = dev_.create_host_visible_buffer(state_matrix_bytes);
    memcpy(upload_buf.mapped, out_data + state_offset, state_matrix_bytes);

    VkDeviceSize state_offset_gpu = static_cast<VkDeviceSize>(dn_idx) * bufs_->dn_state_per_layer;
    {
      auto copy_cmd = dev_.allocate_command_buffer();
      dev_.begin_command_buffer(copy_cmd);
      VkBufferCopy copy{0, state_offset_gpu, static_cast<VkDeviceSize>(state_matrix_bytes)};
      vkCmdCopyBuffer(copy_cmd, upload_buf.buffer, bufs_->dn_state.buffer, 1, &copy);
      dev_.end_command_buffer(copy_cmd);
      dev_.submit_and_wait(copy_cmd);
    }
    dev_.destroy_buffer(upload_buf);
  }

  dev_.destroy_buffer(out_buf);
  dev_.destroy_buffer(init_buf);
}

void DecodeSession::run_chunk_prefill() {
  const auto& schedule = model::Qwen35Config::layer_schedule();
  const uint32_t seq_len = prefill_token_count_;

  // Gate: SPOCK_GPU_CHUNK_PREFILL=1 uses GPU chunk-rule computation for each
  // DeltaNet layer. Default (unset or any other value) uses existing CPU path.
  // When SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1 is also set, Q/K/V/g/beta
  // are bound directly from GPU-collected persistent buffers instead of CPU upload.
  // When SPOCK_GPU_CHUNK_PREFILL_TILED=1 is also set, uses the tiled shader with
  // a single vkCmdDispatch(num_heads, ceil(v_dim/16), 1) per layer.
  if (const char* env = std::getenv("SPOCK_GPU_CHUNK_PREFILL"); env && env[0] == '1' && env[1] == '\0') {
    const char* from_gpu_env = std::getenv("SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT");
    const bool from_gpu = from_gpu_env && from_gpu_env[0] == '1' && from_gpu_env[1] == '\0';
    const char* tiled_env = std::getenv("SPOCK_GPU_CHUNK_PREFILL_TILED");
    const bool tiled = tiled_env && tiled_env[0] == '1' && tiled_env[1] == '\0';

    // Reset GPU handoff flags; gpu_chunk_prefill_from_gpu_collect sets
    // per-layer flags only when the no-compare tiled fast path is active.
    std::fill(gpu_chunk_handoff_ready_.begin(), gpu_chunk_handoff_ready_.end(), false);
    for (uint32_t dn_idx = 0; dn_idx < prefill_chunks_.size(); ++dn_idx) {
      auto& chunk = prefill_chunks_[dn_idx];
      if (from_gpu) {
        if (!bufs_->persist_bufs_allocated_) {
          throw std::runtime_error(
              "SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1 but GPU collection buffers are not allocated");
        }
        gpu_chunk_prefill_from_gpu_collect(dn_idx, chunk, seq_len, tiled);
      } else {
        if (chunk.query.empty()) continue;
        gpu_chunk_prefill(dn_idx, chunk, seq_len, tiled);
      }
    }

    // Destroy persistent GPU collection buffers after consumption
    if (from_gpu && bufs_->persist_bufs_allocated_) {
      dev_.destroy_buffer(bufs_->dn_persist_q);
      dev_.destroy_buffer(bufs_->dn_persist_k);
      dev_.destroy_buffer(bufs_->dn_persist_v);
      dev_.destroy_buffer(bufs_->dn_persist_g);
      dev_.destroy_buffer(bufs_->dn_persist_beta);
      bufs_->persist_bufs_allocated_ = false;
    }

    return;
  }

  for (uint32_t dn_idx = 0; dn_idx < prefill_chunks_.size(); ++dn_idx) {
    auto& chunk = prefill_chunks_[dn_idx];
    if (chunk.query.empty()) continue;

    // Rearrange from collected [token][head][dim] to chunk rule's [head][token][dim]
    // Query: collected as [seq_len * DN_HEADS * DN_K_DIM] in token-major order
    auto rearrange = [](const std::vector<float>& token_major,
                        uint32_t num_heads, uint32_t seq, uint32_t dim) {
      std::vector<float> head_major(num_heads * seq * dim);
      for (uint32_t t = 0; t < seq; ++t) {
        for (uint32_t h = 0; h < num_heads; ++h) {
          for (uint32_t d = 0; d < dim; ++d) {
            // src: [t][h][d], dst: [h][t][d]
            head_major[(h * seq + t) * dim + d] = token_major[(t * num_heads + h) * dim + d];
          }
        }
      }
      return head_major;
    };

    // g and beta: collected as [seq_len * DN_HEADS] in [token][head] order
    auto rearrange_scalar = [](const std::vector<float>& token_major,
                               uint32_t num_heads, uint32_t seq) {
      std::vector<float> head_major(num_heads * seq);
      for (uint32_t t = 0; t < seq; ++t) {
        for (uint32_t h = 0; h < num_heads; ++h) {
          head_major[h * seq + t] = token_major[t * num_heads + h];
        }
      }
      return head_major;
    };

    DeltaNetChunkConfig config;
    config.num_heads = DN_HEADS;
    config.sequence_length = seq_len;
    config.key_dim = DN_K_DIM;
    config.value_dim = DN_V_DIM;
    config.chunk_size = 64;  // HF default
    config.use_qk_l2norm = false;  // GPU prefill already L2-norms Q/K

    DeltaNetChunkInputs inputs;
    inputs.query = rearrange(chunk.query, DN_HEADS, seq_len, DN_K_DIM);
    inputs.key = rearrange(chunk.key, DN_HEADS, seq_len, DN_K_DIM);
    inputs.value = rearrange(chunk.value, DN_HEADS, seq_len, DN_V_DIM);
    inputs.g = rearrange_scalar(chunk.g, DN_HEADS, seq_len);
    inputs.beta = rearrange_scalar(chunk.beta, DN_HEADS, seq_len);
    // No initial state — chunk rule starts from zero

    auto outputs = run_deltanet_chunk_rule(config, inputs);
    // Save core_attn_out for last token (used by correct_last_token_hidden)
    auto& attn_last = chunk_core_attn_out_last_[dn_idx];
    attn_last.resize(DN_HEADS * DN_V_DIM);
    for (uint32_t h = 0; h < DN_HEADS; ++h) {
      for (uint32_t vd = 0; vd < DN_V_DIM; ++vd) {
        // core_attn_out is [head][token][vd]; extract token seq_len-1
        attn_last[h * DN_V_DIM + vd] =
            outputs.core_attn_out[(h * seq_len + (seq_len - 1)) * DN_V_DIM + vd];
      }
    }

    // Upload final_state to GPU, replacing the recurrent state for this layer
    size_t state_matrix_bytes = DN_HEADS * DN_K_DIM * DN_V_DIM * 4;
    VkDeviceSize state_offset = dn_idx * bufs_->dn_state_per_layer;

    auto upload_buf = dev_.create_host_visible_buffer(state_matrix_bytes);
    dev_.upload_to_host_visible(upload_buf, outputs.final_state.data(), state_matrix_bytes);
    {
      auto cmd = dev_.allocate_command_buffer();
      dev_.begin_command_buffer(cmd);
      VkBufferCopy copy{0, state_offset, state_matrix_bytes};
      vkCmdCopyBuffer(cmd, upload_buf.buffer, bufs_->dn_state.buffer, 1, &copy);
      dev_.end_command_buffer(cmd);
      dev_.submit_and_wait(cmd);
    }
    dev_.destroy_buffer(upload_buf);
  }
}


// ---------------------------------------------------------------------------
// correct_last_token_hidden: reprocess all layers for the final prefill token
// using chunk-corrected DeltaNet outputs, producing the correct hidden state
// for the first decode step. Runs entirely on GPU-resident buffers.
// ---------------------------------------------------------------------------
void DecodeSession::correct_last_token_hidden(
    const std::vector<uint32_t>& /*tokens*/, uint32_t prompt_len,
    std::vector<uint16_t>* out_layer_hidden) {
  const auto& schedule = model::Qwen35Config::layer_schedule();
  const auto& P = *pipes_;
  const auto& B = *bufs_;
  const auto& D = *dsets_;

  constexpr float RMS_EPS = 1e-6f;
  constexpr float ATTN_SCALE = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));

  auto barrier = [&](VkCommandBuffer cmd_buf, VkBuffer buf, VkDeviceSize size,
                     VkDeviceSize offset = 0) {
    VkBufferMemoryBarrier bmb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    bmb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bmb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.buffer = buf;
    bmb.offset = offset;
    bmb.size = size;
    vkCmdPipelineBarrier(cmd_buf,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 1, &bmb, 0, nullptr);
  };

  auto attn_layer_idx = [](uint32_t layer) -> uint32_t {
    return (layer + 1) / 4 - 1;
  };

  uint32_t seq_pos = prompt_len - 1;
  uint32_t dn_idx = 0;

  for (uint32_t layer = 0; layer < LAYERS; ++layer) {
    bool is_attn = (schedule[layer] == model::LayerKind::FullAttention);

    // --- Load pre-norm hidden state for this layer into act_a ---
    // For layer 0, use the snapshot taken during prefill.
    // For layers > 0, act_a already holds the corrected output of the previous layer.
    if (layer == 0) {
      auto snap_cmd = dev_.allocate_command_buffer();
      dev_.begin_command_buffer(snap_cmd);
      VkBufferCopy snap_copy{0, 0, HIDDEN * 2};
      vkCmdCopyBuffer(snap_cmd, prefill_snapshots_.buffer, B.act_a.buffer, 1, &snap_copy);
      dev_.end_command_buffer(snap_cmd);
      dev_.submit_and_wait(snap_cmd);
    }

    // --- Common weight lookups ---
    auto input_norm_w = artifact_.find_by_role(
        "layer." + std::to_string(layer) + ".input_norm");
    auto post_norm_w = artifact_.find_by_role(
        "layer." + std::to_string(layer) + ".post_norm");
    auto gate_w = artifact_.find_by_role(
        "layer." + std::to_string(layer) + ".mlp_gate");
    auto up_w = artifact_.find_by_role(
        "layer." + std::to_string(layer) + ".mlp_up");
    auto down_w = artifact_.find_by_role(
        "layer." + std::to_string(layer) + ".mlp_down");

    if (is_attn) {
      // --- FullAttention: run normal single-token recurrent path ---
      uint32_t attn_idx = attn_layer_idx(layer);
      auto attn_q_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".attn_q");
      auto attn_k_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".attn_k");
      auto attn_v_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".attn_v");
      auto attn_o_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".attn_o");
      auto attn_q_norm_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".attn_q_norm");
      auto attn_k_norm_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".attn_k_norm");

      // Update descriptor sets
      dev_.update_descriptor_set(D.input_norm, 0, B.act_a);
      dev_.update_descriptor_set(D.input_norm, 1, B.weights, input_norm_w->offset, input_norm_w->nbytes);
      dev_.update_descriptor_set(D.input_norm, 2, B.act_b);
      dev_.update_descriptor_set(D.residual1, 0, B.act_a);
      dev_.update_descriptor_set(D.residual1, 1, B.act_b);
      dev_.update_descriptor_set(D.residual1, 2, B.act_c);
      dev_.update_descriptor_set(D.post_norm, 0, B.act_c);
      dev_.update_descriptor_set(D.post_norm, 1, B.weights, post_norm_w->offset, post_norm_w->nbytes);
      dev_.update_descriptor_set(D.post_norm, 2, B.act_a);
      dev_.update_descriptor_set(D.gate, 0, B.weights, gate_w->offset, gate_w->nbytes);
      dev_.update_descriptor_set(D.gate, 1, B.act_a);
      dev_.update_descriptor_set(D.gate, 2, B.mlp_gate);
      dev_.update_descriptor_set(D.up, 0, B.weights, up_w->offset, up_w->nbytes);
      dev_.update_descriptor_set(D.up, 1, B.act_a);
      dev_.update_descriptor_set(D.up, 2, B.mlp_up);
      dev_.update_descriptor_set(D.down, 0, B.weights, down_w->offset, down_w->nbytes);
      dev_.update_descriptor_set(D.down, 1, B.mlp_silu);
      dev_.update_descriptor_set(D.down, 2, B.act_b);
      dev_.update_descriptor_set(D.residual2, 0, B.act_c);
      dev_.update_descriptor_set(D.residual2, 1, B.act_b);
      dev_.update_descriptor_set(D.residual2, 2, B.act_a);

      dev_.update_descriptor_set(D.q_proj, 0, B.weights, attn_q_w->offset, attn_q_w->nbytes);
      dev_.update_descriptor_set(D.q_proj, 1, B.act_b);
      dev_.update_descriptor_set(D.q_proj, 2, B.q_proj);
      dev_.update_descriptor_set(D.k_proj, 0, B.weights, attn_k_w->offset, attn_k_w->nbytes);
      dev_.update_descriptor_set(D.k_proj, 1, B.act_b);
      dev_.update_descriptor_set(D.k_proj, 2, B.k);
      dev_.update_descriptor_set(D.v_proj, 0, B.weights, attn_v_w->offset, attn_v_w->nbytes);
      dev_.update_descriptor_set(D.v_proj, 1, B.act_b);
      dev_.update_descriptor_set(D.v_proj, 2, B.v);
      dev_.update_descriptor_set(D.q_norm, 0, B.q);
      dev_.update_descriptor_set(D.q_norm, 1, B.weights, attn_q_norm_w->offset, attn_q_norm_w->nbytes);
      dev_.update_descriptor_set(D.q_norm, 2, B.q);
      dev_.update_descriptor_set(D.k_norm, 0, B.k);
      dev_.update_descriptor_set(D.k_norm, 1, B.weights, attn_k_norm_w->offset, attn_k_norm_w->nbytes);
      dev_.update_descriptor_set(D.k_norm, 2, B.k);
      uint32_t kv_layer_offset = attn_idx * MAX_SEQ * 2 * KV_HEADS * HEAD_DIM * 2;
      dev_.update_descriptor_set(D.kv_store, 0, B.k);
      dev_.update_descriptor_set(D.kv_store, 1, B.v);
      dev_.update_descriptor_set(D.kv_store, 2, B.kv_cache, kv_layer_offset);
      dev_.update_descriptor_set(D.attn, 0, B.q);
      dev_.update_descriptor_set(D.attn, 1, B.kv_cache, kv_layer_offset);
      dev_.update_descriptor_set(D.attn, 2, B.attn_out);
      dev_.update_descriptor_set(D.o_proj, 0, B.weights, attn_o_w->offset, attn_o_w->nbytes);
      dev_.update_descriptor_set(D.o_proj, 1, B.gated_attn);
      dev_.update_descriptor_set(D.o_proj, 2, B.act_b);

      // Record command buffer
      auto cmd = dev_.allocate_command_buffer();
      dev_.begin_command_buffer(cmd);

      struct { uint32_t N; uint32_t eps_bits; } rms_push = { HIDDEN, float_to_bits(RMS_EPS) };
      struct { uint32_t N; uint32_t pad; } res_push = { HIDDEN, 0 };
      struct { uint32_t out_dim; uint32_t in_dim; } mv_push;
      struct { uint32_t N; uint32_t pad; } silu_push = { INTER, 0 };

      size_t q_proj_bytes = Q_HEADS * HEAD_DIM * 2 * 2;
      size_t q_bytes = Q_HEADS * HEAD_DIM * 2;
      size_t kv_bytes = KV_HEADS * HEAD_DIM * 2;
      size_t attn_out_bytes = q_bytes * 2;
      uint32_t kv_cache_layer_bytes = MAX_SEQ * 2 * KV_HEADS * HEAD_DIM * 2;

      // 1. input_norm
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.rmsnorm);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          P.pipeline_layout_3, 0, 1, &D.input_norm, 0, nullptr);
      vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &rms_push);
      vkCmdDispatch(cmd, 1, 1, 1);
      barrier(cmd, B.act_b.buffer, B.act_bytes);

      // 2. Attention token mixer
      // q_proj
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.matvec);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          P.pipeline_layout_3, 0, 1, &D.q_proj, 0, nullptr);
      mv_push = { Q_HEADS * HEAD_DIM * 2, HIDDEN };
      vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &mv_push);
      vkCmdDispatch(cmd, (Q_HEADS * HEAD_DIM * 2 + 63) / 64, 1, 1);
      barrier(cmd, B.q_proj.buffer, q_proj_bytes);

      // k_proj
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.matvec);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          P.pipeline_layout_3, 0, 1, &D.k_proj, 0, nullptr);
      mv_push = { KV_HEADS * HEAD_DIM, HIDDEN };
      vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &mv_push);
      vkCmdDispatch(cmd, (KV_HEADS * HEAD_DIM + 63) / 64, 1, 1);
      barrier(cmd, B.k.buffer, kv_bytes);

      // v_proj
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.matvec);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          P.pipeline_layout_3, 0, 1, &D.v_proj, 0, nullptr);
      mv_push = { KV_HEADS * HEAD_DIM, HIDDEN };
      vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &mv_push);
      vkCmdDispatch(cmd, (KV_HEADS * HEAD_DIM + 63) / 64, 1, 1);
      barrier(cmd, B.v.buffer, kv_bytes);

      // split q+gate
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.split_q_gate);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          P.pipeline_layout_3, 0, 1, &D.split_q_gate, 0, nullptr);
      struct { uint32_t num_heads; uint32_t head_dim; uint32_t total_input; } split_push = { Q_HEADS, HEAD_DIM, Q_HEADS * HEAD_DIM * 2 };
      vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, &split_push);
      vkCmdDispatch(cmd, 1, 1, 1);
      barrier(cmd, B.q.buffer, q_bytes);
      barrier(cmd, B.gate.buffer, q_bytes);

      // q_norm
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.rms_norm_per_head);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          P.pipeline_layout_32, 0, 1, &D.q_norm, 0, nullptr);
      struct { uint32_t num_heads; uint32_t head_dim; uint32_t eps_bits; } qnorm_push = { Q_HEADS, HEAD_DIM, float_to_bits(RMS_EPS) };
      vkCmdPushConstants(cmd, P.pipeline_layout_32, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, &qnorm_push);
      vkCmdDispatch(cmd, Q_HEADS, 1, 1);
      barrier(cmd, B.q.buffer, q_bytes);

      // k_norm
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.rms_norm_per_head);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          P.pipeline_layout_32, 0, 1, &D.k_norm, 0, nullptr);
      struct { uint32_t num_heads; uint32_t head_dim; uint32_t eps_bits; } knorm_push = { KV_HEADS, HEAD_DIM, float_to_bits(RMS_EPS) };
      vkCmdPushConstants(cmd, P.pipeline_layout_32, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, &knorm_push);
      vkCmdDispatch(cmd, KV_HEADS, 1, 1);
      barrier(cmd, B.k.buffer, kv_bytes);

      // RoPE Q
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.rope_apply);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          P.pipeline_layout_32, 0, 1, &D.rope, 0, nullptr);
      struct { uint32_t num_heads; uint32_t head_dim; uint32_t rotary_dim; uint32_t freq_offset; } rope_q_push = { Q_HEADS, HEAD_DIM, ROTARY_DIM, seq_pos * ROTARY_DIM };
      vkCmdPushConstants(cmd, P.pipeline_layout_32, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, &rope_q_push);
      vkCmdDispatch(cmd, 1, 1, 1);
      barrier(cmd, B.q.buffer, q_bytes);

      // RoPE K
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.rope_apply);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          P.pipeline_layout_32, 0, 1, &D.rope_k, 0, nullptr);
      struct { uint32_t num_heads; uint32_t head_dim; uint32_t rotary_dim; uint32_t freq_offset; } rope_k_push = { KV_HEADS, HEAD_DIM, ROTARY_DIM, seq_pos * ROTARY_DIM };
      vkCmdPushConstants(cmd, P.pipeline_layout_32, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, &rope_k_push);
      vkCmdDispatch(cmd, 1, 1, 1);
      barrier(cmd, B.k.buffer, kv_bytes);

      // KV cache store at position seq_pos
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.kv_cache_store);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          P.pipeline_layout_32, 0, 1, &D.kv_store, 0, nullptr);
      struct { uint32_t kv_heads; uint32_t head_dim; uint32_t position; uint32_t max_seq_len; } kvs_push = { KV_HEADS, HEAD_DIM, seq_pos, MAX_SEQ };
      vkCmdPushConstants(cmd, P.pipeline_layout_32, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, &kvs_push);
      vkCmdDispatch(cmd, 1, 1, 1);
      barrier(cmd, B.kv_cache.buffer, kv_cache_layer_bytes, kv_layer_offset);

      // Attention (sees all K/V up to position seq_pos)
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.attention_decode);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          P.pipeline_layout_32, 0, 1, &D.attn, 0, nullptr);
      struct { uint32_t q_heads; uint32_t kv_heads; uint32_t head_dim; uint32_t kv_group_size; uint32_t seq_len; uint32_t max_seq_len; float scale; } attn_push;
      attn_push.q_heads = Q_HEADS;
      attn_push.kv_heads = KV_HEADS;
      attn_push.head_dim = HEAD_DIM;
      attn_push.kv_group_size = KV_GROUP;
      attn_push.seq_len = seq_pos + 1;
      attn_push.max_seq_len = MAX_SEQ;
      attn_push.scale = ATTN_SCALE;
      vkCmdPushConstants(cmd, P.pipeline_layout_32, VK_SHADER_STAGE_COMPUTE_BIT, 0, 28, &attn_push);
      vkCmdDispatch(cmd, Q_HEADS, 1, 1);
      barrier(cmd, B.attn_out.buffer, attn_out_bytes);

      // Sigmoid gate
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.sigmoid_gate);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          P.pipeline_layout_3, 0, 1, &D.sigmoid_gate, 0, nullptr);
      struct { uint32_t N; uint32_t pad; } sg_push = { Q_HEADS * HEAD_DIM, 0 };
      vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &sg_push);
      vkCmdDispatch(cmd, 1, 1, 1);
      barrier(cmd, B.gated_attn.buffer, q_bytes * 2);

      // Output projection
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.matvec);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          P.pipeline_layout_3, 0, 1, &D.o_proj, 0, nullptr);
      mv_push = { HIDDEN, Q_HEADS * HEAD_DIM };
      vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &mv_push);
      vkCmdDispatch(cmd, (HIDDEN + 63) / 64, 1, 1);
      barrier(cmd, B.act_b.buffer, B.act_bytes);

      // 3. residual_add(act_a, act_b) → act_c
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.residual_add);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          P.pipeline_layout_3, 0, 1, &D.residual1, 0, nullptr);
      vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &res_push);
      vkCmdDispatch(cmd, 1, 1, 1);
      barrier(cmd, B.act_c.buffer, B.act_c_bytes);

      // 4. post_norm(act_c) → act_a
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.rmsnorm);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          P.pipeline_layout_3, 0, 1, &D.post_norm, 0, nullptr);
      vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &rms_push);
      vkCmdDispatch(cmd, 1, 1, 1);
      barrier(cmd, B.act_a.buffer, B.act_bytes);

      // 5. gate_matvec(act_a) → mlp_gate_buf
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.matvec);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          P.pipeline_layout_3, 0, 1, &D.gate, 0, nullptr);
      mv_push = { INTER, HIDDEN };
      vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &mv_push);
      vkCmdDispatch(cmd, (INTER + 63) / 64, 1, 1);
      barrier(cmd, B.mlp_gate.buffer, INTER * 2);

      // 6. up_matvec(act_a) → mlp_up_buf
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.matvec);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          P.pipeline_layout_3, 0, 1, &D.up, 0, nullptr);
      mv_push = { INTER, HIDDEN };
      vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &mv_push);
      vkCmdDispatch(cmd, (INTER + 63) / 64, 1, 1);
      barrier(cmd, B.mlp_up.buffer, INTER * 2);

      // 7. silu_gate
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.silu_gate);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          P.pipeline_layout_3, 0, 1, &D.silu_gate, 0, nullptr);
      vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &silu_push);
      vkCmdDispatch(cmd, 1, 1, 1);
      barrier(cmd, B.mlp_silu.buffer, INTER * 2);

      // 8. down_matvec(mlp_silu_buf) → act_b
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.matvec);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          P.pipeline_layout_3, 0, 1, &D.down, 0, nullptr);
      mv_push = { HIDDEN, INTER };
      vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &mv_push);
      vkCmdDispatch(cmd, (HIDDEN + 63) / 64, 1, 1);
      barrier(cmd, B.act_b.buffer, B.act_bytes);

      // 9. residual_add(act_c, act_b) → act_a
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.residual_add);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          P.pipeline_layout_3, 0, 1, &D.residual2, 0, nullptr);
      vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &res_push);
      vkCmdDispatch(cmd, 1, 1, 1);
      barrier(cmd, B.act_a.buffer, B.act_bytes);

      dev_.end_command_buffer(cmd);
      dev_.submit_and_wait(cmd);

    } else {
      // --- DeltaNet: prime conv state + compute g/beta, then substitute chunk core_attn_out ---
      auto dn_qkv_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".delta_in_proj_qkv");
      auto dn_z_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".delta_in_proj_z");
      auto dn_a_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".delta_in_proj_a");
      auto dn_b_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".delta_in_proj_b");
      auto dn_out_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".delta_out_proj");
      auto dn_conv_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".delta_conv");
      auto dn_norm_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".delta_norm");

      // Update descriptor sets
      dev_.update_descriptor_set(D.input_norm, 0, B.act_a);
      dev_.update_descriptor_set(D.input_norm, 1, B.weights, input_norm_w->offset, input_norm_w->nbytes);
      dev_.update_descriptor_set(D.input_norm, 2, B.act_b);
      dev_.update_descriptor_set(D.dn_qkv_proj, 0, B.weights, dn_qkv_w->offset, dn_qkv_w->nbytes);
      dev_.update_descriptor_set(D.dn_qkv_proj, 1, B.act_b);
      dev_.update_descriptor_set(D.dn_qkv_proj, 2, B.dn_qkv);
      dev_.update_descriptor_set(D.dn_a_proj, 0, B.weights, dn_a_w->offset, dn_a_w->nbytes);
      dev_.update_descriptor_set(D.dn_a_proj, 1, B.act_b);
      dev_.update_descriptor_set(D.dn_a_proj, 2, B.dn_a);
      dev_.update_descriptor_set(D.dn_b_proj, 0, B.weights, dn_b_w->offset, dn_b_w->nbytes);
      dev_.update_descriptor_set(D.dn_b_proj, 1, B.act_b);
      dev_.update_descriptor_set(D.dn_b_proj, 2, B.dn_b);
      dev_.update_descriptor_set(D.dn_z_proj, 0, B.weights, dn_z_w->offset, dn_z_w->nbytes);
      dev_.update_descriptor_set(D.dn_z_proj, 1, B.act_b);
      dev_.update_descriptor_set(D.dn_z_proj, 2, B.dn_z);
      uint32_t conv_state_offset = dn_idx * DN_CONV_DIM * DN_CONV_KS * 2;
      dev_.update_descriptor_set(D.dn_conv, 0, B.dn_qkv);
      dev_.update_descriptor_set(D.dn_conv, 1, B.dn_conv_state, conv_state_offset);
      dev_.update_descriptor_set(D.dn_conv, 2, B.weights, dn_conv_w->offset, dn_conv_w->nbytes);
      dev_.update_descriptor_set(D.residual1, 0, B.act_a);
      dev_.update_descriptor_set(D.residual1, 1, B.act_b);
      dev_.update_descriptor_set(D.residual1, 2, B.act_c);
      dev_.update_descriptor_set(D.post_norm, 0, B.act_c);
      dev_.update_descriptor_set(D.post_norm, 1, B.weights, post_norm_w->offset, post_norm_w->nbytes);
      dev_.update_descriptor_set(D.post_norm, 2, B.act_a);
      dev_.update_descriptor_set(D.gate, 0, B.weights, gate_w->offset, gate_w->nbytes);
      dev_.update_descriptor_set(D.gate, 1, B.act_a);
      dev_.update_descriptor_set(D.gate, 2, B.mlp_gate);
      dev_.update_descriptor_set(D.up, 0, B.weights, up_w->offset, up_w->nbytes);
      dev_.update_descriptor_set(D.up, 1, B.act_a);
      dev_.update_descriptor_set(D.up, 2, B.mlp_up);
      dev_.update_descriptor_set(D.down, 0, B.weights, down_w->offset, down_w->nbytes);
      dev_.update_descriptor_set(D.down, 1, B.mlp_silu);
      dev_.update_descriptor_set(D.down, 2, B.act_b);
      dev_.update_descriptor_set(D.residual2, 0, B.act_c);
      dev_.update_descriptor_set(D.residual2, 1, B.act_b);
      dev_.update_descriptor_set(D.residual2, 2, B.act_a);

      // Submit 0: input_norm + QKV/A/B/Z projections
      {
        auto cmd0 = dev_.allocate_command_buffer();
        dev_.begin_command_buffer(cmd0);

        size_t dn_kv_bytes = DN_CONV_DIM * 2;

        // input_norm
        struct { uint32_t N; uint32_t eps_bits; } rms_dn = { HIDDEN, float_to_bits(RMS_EPS) };
        vkCmdBindPipeline(cmd0, VK_PIPELINE_BIND_POINT_COMPUTE, P.rmsnorm);
        vkCmdBindDescriptorSets(cmd0, VK_PIPELINE_BIND_POINT_COMPUTE,
            P.pipeline_layout_3, 0, 1, &D.input_norm, 0, nullptr);
        vkCmdPushConstants(cmd0, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &rms_dn);
        vkCmdDispatch(cmd0, 1, 1, 1);
        barrier(cmd0, B.act_b.buffer, B.act_bytes);

        // QKV projection
        struct { uint32_t out_dim; uint32_t in_dim; } dn_qkv_mv = { DN_CONV_DIM, HIDDEN };
        vkCmdBindPipeline(cmd0, VK_PIPELINE_BIND_POINT_COMPUTE, P.matvec);
        vkCmdBindDescriptorSets(cmd0, VK_PIPELINE_BIND_POINT_COMPUTE,
            P.pipeline_layout_3, 0, 1, &D.dn_qkv_proj, 0, nullptr);
        vkCmdPushConstants(cmd0, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &dn_qkv_mv);
        vkCmdDispatch(cmd0, (DN_CONV_DIM + 63) / 64, 1, 1);
        barrier(cmd0, B.dn_qkv.buffer, dn_kv_bytes);

        // A projection
        struct { uint32_t out_dim; uint32_t in_dim; } dn_a_mv = { DN_HEADS, HIDDEN };
        vkCmdBindPipeline(cmd0, VK_PIPELINE_BIND_POINT_COMPUTE, P.matvec);
        vkCmdBindDescriptorSets(cmd0, VK_PIPELINE_BIND_POINT_COMPUTE,
            P.pipeline_layout_3, 0, 1, &D.dn_a_proj, 0, nullptr);
        vkCmdPushConstants(cmd0, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &dn_a_mv);
        vkCmdDispatch(cmd0, 1, 1, 1);
        barrier(cmd0, B.dn_a.buffer, DN_HEADS * 2);

        // B projection
        struct { uint32_t out_dim; uint32_t in_dim; } dn_b_mv = { DN_HEADS, HIDDEN };
        vkCmdBindPipeline(cmd0, VK_PIPELINE_BIND_POINT_COMPUTE, P.matvec);
        vkCmdBindDescriptorSets(cmd0, VK_PIPELINE_BIND_POINT_COMPUTE,
            P.pipeline_layout_3, 0, 1, &D.dn_b_proj, 0, nullptr);
        vkCmdPushConstants(cmd0, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &dn_b_mv);
        vkCmdDispatch(cmd0, 1, 1, 1);
        barrier(cmd0, B.dn_b.buffer, DN_HEADS * 2);
        // Skip conv1d_step: layer_major_prefill already primed dn_conv_state
        // through the final prompt token. Do not advance conv state twice.

        // Z projection
        struct { uint32_t out_dim; uint32_t in_dim; } dn_z_mv = { DN_VAL_TOTAL, HIDDEN };
        vkCmdBindPipeline(cmd0, VK_PIPELINE_BIND_POINT_COMPUTE, P.matvec);
        vkCmdBindDescriptorSets(cmd0, VK_PIPELINE_BIND_POINT_COMPUTE,
            P.pipeline_layout_3, 0, 1, &D.dn_z_proj, 0, nullptr);
        vkCmdPushConstants(cmd0, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &dn_z_mv);
        vkCmdDispatch(cmd0, (DN_VAL_TOTAL + 63) / 64, 1, 1);
        barrier(cmd0, B.dn_z.buffer, DN_VAL_TOTAL * 2);

        dev_.end_command_buffer(cmd0);
        dev_.submit_and_wait(cmd0);
      }


      // GPU: compute g, beta from corrected A/B, overwriting stale recurrent-prefill values
      {
        VkDeviceSize g_beta_offset = dn_idx * B.dn_state_per_layer + DN_HEADS * DN_K_DIM * DN_V_DIM * 4;
        dev_.update_descriptor_set(D.dn_compute_g_beta, 3, B.dn_state, g_beta_offset, DN_HEADS * 2 * 4);

        auto gb_cmd = dev_.allocate_command_buffer();
        dev_.begin_command_buffer(gb_cmd);

        vkCmdBindPipeline(gb_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.deltanet_compute_g_beta);
        vkCmdBindDescriptorSets(gb_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            P.pipeline_layout_4, 0, 1, &D.dn_compute_g_beta, 0, nullptr);
        struct { uint32_t num_heads; uint32_t layer_idx; } gb_pc = { DN_HEADS, dn_idx };
        vkCmdPushConstants(gb_cmd, P.pipeline_layout_4, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &gb_pc);
        vkCmdDispatch(gb_cmd, DN_HEADS, 1, 1);

        dev_.end_command_buffer(gb_cmd);
        dev_.submit_and_wait(gb_cmd);
      }

      // Upload chunk core_attn_out → dn_qkv V region (fp16)
      // Must happen after Submit 0 so QKV proj doesn't overwrite it.
      if (gpu_chunk_handoff_ready_[dn_idx]) {
        // GPU handoff: direct GPU→GPU copy from dn_chunk_attn_out_ layer slice
        auto copy_cmd = dev_.allocate_command_buffer();
        dev_.begin_command_buffer(copy_cmd);
        VkDeviceSize src_offset = static_cast<VkDeviceSize>(dn_idx) * DN_VAL_TOTAL * 2;
        VkDeviceSize v_offset = static_cast<VkDeviceSize>(DN_KEY_TOTAL) * 4;
        VkBufferCopy copy{src_offset, v_offset, DN_VAL_TOTAL * 2};
        vkCmdCopyBuffer(copy_cmd, dn_chunk_attn_out_.buffer, B.dn_qkv.buffer, 1, &copy);
        dev_.end_command_buffer(copy_cmd);
        dev_.submit_and_wait(copy_cmd);
        gpu_chunk_handoff_ready_[dn_idx] = false;
      } else {
        const auto& attn_last = chunk_core_attn_out_last_[dn_idx];
        if (!attn_last.empty()) {
          std::vector<uint16_t> attn_fp16(DN_VAL_TOTAL);
          for (uint32_t i = 0; i < DN_VAL_TOTAL; ++i) {
            attn_fp16[i] = float_to_half(attn_last[i]);
          }
          auto staging = dev_.create_host_visible_buffer(DN_VAL_TOTAL * 2);
          dev_.upload_to_host_visible(staging, attn_fp16.data(), DN_VAL_TOTAL * 2);
          {
            auto copy_cmd = dev_.allocate_command_buffer();
            dev_.begin_command_buffer(copy_cmd);
            VkDeviceSize v_offset = static_cast<VkDeviceSize>(DN_KEY_TOTAL) * 4;
            VkBufferCopy copy{0, v_offset, DN_VAL_TOTAL * 2};
            vkCmdCopyBuffer(copy_cmd, staging.buffer, B.dn_qkv.buffer, 1, &copy);
            dev_.end_command_buffer(copy_cmd);
            dev_.submit_and_wait(copy_cmd);
          }
          dev_.destroy_buffer(staging);
        }
      }


      VkDescriptorSet ds_dn_norm_gate = per_layer_sets_enabled_ ? per_layer_sets_->dn_norm_gate[layer] : D.dn_norm_gate;
      VkDescriptorSet ds_dn_out_proj = per_layer_sets_enabled_ ? per_layer_sets_->dn_out_proj[layer] : D.dn_out_proj;

      // Submit 2: norm_gate + out_proj + residual + MLP
      {
        if (!per_layer_sets_enabled_) {
          dev_.update_descriptor_set(D.dn_norm_gate, 0, B.dn_qkv, (DN_KEY_TOTAL + DN_KEY_TOTAL) * 2, DN_VAL_TOTAL * 2);
          dev_.update_descriptor_set(D.dn_norm_gate, 1, B.dn_z);
          dev_.update_descriptor_set(D.dn_norm_gate, 2, B.weights, dn_norm_w->offset, dn_norm_w->nbytes);
        }

        if (!per_layer_sets_enabled_) {
          dev_.update_descriptor_set(D.dn_out_proj, 0, B.weights, dn_out_w->offset, dn_out_w->nbytes);
          dev_.update_descriptor_set(D.dn_out_proj, 1, B.dn_qkv, (DN_KEY_TOTAL + DN_KEY_TOTAL) * 2, DN_VAL_TOTAL * 2);
          dev_.update_descriptor_set(D.dn_out_proj, 2, B.act_b);
        }

        auto cmd = dev_.allocate_command_buffer();
        dev_.begin_command_buffer(cmd);

        struct { uint32_t N; uint32_t eps_bits; } rms_push = { HIDDEN, float_to_bits(RMS_EPS) };
        struct { uint32_t N; uint32_t pad; } res_push = { HIDDEN, 0 };
        struct { uint32_t out_dim; uint32_t in_dim; } mv_push;
        struct { uint32_t N; uint32_t pad; } silu_push = { INTER, 0 };

        // Norm+gate on chunk core_attn_out
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.deltanet_norm_gate);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            P.pipeline_layout_32, 0, 1, &ds_dn_norm_gate, 0, nullptr);
        struct { uint32_t num_heads; uint32_t head_dim; uint32_t eps_bits; uint32_t output_offset; } dn_ng_push = { DN_HEADS, DN_V_DIM, float_to_bits(RMS_EPS), 0 };
        vkCmdPushConstants(cmd, P.pipeline_layout_32, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, &dn_ng_push);
        vkCmdDispatch(cmd, DN_HEADS, 1, 1);
        barrier(cmd, B.dn_qkv.buffer, DN_CONV_DIM * 2);

        // Output projection
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.matvec);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            P.pipeline_layout_3, 0, 1, &ds_dn_out_proj, 0, nullptr);
        mv_push = { HIDDEN, DN_VAL_TOTAL };
        vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &mv_push);
        vkCmdDispatch(cmd, (HIDDEN + 63) / 64, 1, 1);
        barrier(cmd, B.act_b.buffer, B.act_bytes);

        // residual_add(act_a, act_b) → act_c
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.residual_add);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            P.pipeline_layout_3, 0, 1, &D.residual1, 0, nullptr);
        vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &res_push);
        vkCmdDispatch(cmd, 1, 1, 1);
        barrier(cmd, B.act_c.buffer, B.act_c_bytes);

        // post_norm(act_c) → act_a
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.rmsnorm);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            P.pipeline_layout_3, 0, 1, &D.post_norm, 0, nullptr);
        vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &rms_push);
        vkCmdDispatch(cmd, 1, 1, 1);
        barrier(cmd, B.act_a.buffer, B.act_bytes);

        // gate_matvec(act_a) → mlp_gate_buf
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.matvec);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            P.pipeline_layout_3, 0, 1, &D.gate, 0, nullptr);
        mv_push = { INTER, HIDDEN };
        vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &mv_push);
        vkCmdDispatch(cmd, (INTER + 63) / 64, 1, 1);
        barrier(cmd, B.mlp_gate.buffer, INTER * 2);

        // up_matvec(act_a) → mlp_up_buf
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.matvec);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            P.pipeline_layout_3, 0, 1, &D.up, 0, nullptr);
        mv_push = { INTER, HIDDEN };
        vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &mv_push);
        vkCmdDispatch(cmd, (INTER + 63) / 64, 1, 1);
        barrier(cmd, B.mlp_up.buffer, INTER * 2);

        // silu_gate
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.silu_gate);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            P.pipeline_layout_3, 0, 1, &D.silu_gate, 0, nullptr);
        vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &silu_push);
        vkCmdDispatch(cmd, 1, 1, 1);
        barrier(cmd, B.mlp_silu.buffer, INTER * 2);

        // down_matvec(mlp_silu_buf) → act_b
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.matvec);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            P.pipeline_layout_3, 0, 1, &D.down, 0, nullptr);
        mv_push = { HIDDEN, INTER };
        vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &mv_push);
        vkCmdDispatch(cmd, (HIDDEN + 63) / 64, 1, 1);
        barrier(cmd, B.act_b.buffer, B.act_bytes);

        // residual_add(act_c, act_b) → act_a
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.residual_add);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            P.pipeline_layout_3, 0, 1, &D.residual2, 0, nullptr);
        vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &res_push);
        vkCmdDispatch(cmd, 1, 1, 1);
        barrier(cmd, B.act_a.buffer, B.act_bytes);

        dev_.end_command_buffer(cmd);
        dev_.submit_and_wait(cmd);
      }

      ++dn_idx;
    }

    // Per-layer hidden capture for decode drift diagnostic (all layers)
    if (out_layer_hidden) {
      std::vector<uint16_t> layer_hid(HIDDEN);
      dev_.download_from_device(B.act_a, layer_hid.data(), HIDDEN * 2);
      out_layer_hidden->insert(out_layer_hidden->end(), layer_hid.begin(), layer_hid.end());
    }
  }
}

// ---------------------------------------------------------------------------
// diagnose_handoff: dump recurrent vs chunk state at prefill-decode boundary
// ---------------------------------------------------------------------------
void DecodeSession::diagnose_handoff(
    const std::vector<uint32_t>& tokens, uint32_t prompt_len) {
  const auto& B = *bufs_;
  const auto& D = *dsets_;
  const auto& P = *pipes_;

  auto barrier = [&](VkCommandBuffer cb, VkBuffer buf, VkDeviceSize size) {
    VkBufferMemoryBarrier bmb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    bmb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bmb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.buffer = buf;
    bmb.offset = 0;
    bmb.size = size;
    vkCmdPipelineBarrier(cb,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 1, &bmb, 0, nullptr);
  };

  // ---- Part 1: Run LM head + argmax on current act_a ----
  {
    auto cmd = dev_.allocate_command_buffer();
    dev_.begin_command_buffer(cmd);

    struct { uint32_t N; uint32_t eps_bits; } fn_push = { HIDDEN, float_to_bits(1e-6f) };
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.rmsnorm);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        P.pipeline_layout_3, 0, 1, &D.final_norm, 0, nullptr);
    vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &fn_push);
    vkCmdDispatch(cmd, 1, 1, 1);
    barrier(cmd, B.act_b.buffer, B.act_bytes);

    struct { uint32_t out_dim; uint32_t in_dim; } lm_push = { VOCAB, HIDDEN };
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.matvec);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        P.pipeline_layout_3, 0, 1, &D.lm_head, 0, nullptr);
    vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &lm_push);
    vkCmdDispatch(cmd, (VOCAB + 63) / 64, 1, 1);
    barrier(cmd, B.logits.buffer, VOCAB * 2);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.argmax);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        P.pipeline_layout_2, 0, 1, &D.argmax, 0, nullptr);
    uint32_t argmax_push = VOCAB;
    vkCmdPushConstants(cmd, P.pipeline_layout_2, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &argmax_push);
    vkCmdDispatch(cmd, 1, 1, 1);

    dev_.end_command_buffer(cmd);
    dev_.submit_and_wait(cmd);
  }

  uint32_t argmax_token = 0;
  dev_.download_from_device(B.argmax_result, &argmax_token, 4);

  std::vector<uint16_t> logit_dump(VOCAB);
  dev_.download_from_device(B.logits, logit_dump.data(), VOCAB * 2);

  // Top-5
  std::vector<std::pair<float, uint32_t>> top;
  top.reserve(5);
  for (uint32_t i = 0; i < VOCAB; ++i) {
    float value = half_to_float(logit_dump[i]);
    if (top.size() < 5) {
      top.emplace_back(value, i);
      if (top.size() == 5) std::sort(top.begin(), top.end(), std::greater<>());
    } else if (value > top.back().first) {
      top.back() = {value, i};
      std::sort(top.begin(), top.end(), std::greater<>());
    }
  }

  // Hidden state stats
  std::vector<uint16_t> hidden_fp16(HIDDEN);
  dev_.download_from_device(B.act_a, hidden_fp16.data(), HIDDEN * 2);
  double hidden_norm = 0.0;
  for (uint32_t i = 0; i < HIDDEN; ++i) {
    float v = half_to_float(hidden_fp16[i]);
    hidden_norm += static_cast<double>(v) * v;
  }
  hidden_norm = std::sqrt(hidden_norm);

  std::cout << "{\n";
  std::cout << "  \"diagnostic\": \"handoff_state\",\n";
  std::cout << "  \"prompt_len\": " << prompt_len << ",\n";
  std::cout << "  \"recurrent_hidden_argmax_token\": " << argmax_token << ",\n";
  std::cout << "  \"recurrent_hidden_norm\": " << hidden_norm << ",\n";
  std::cout << "  \"recurrent_hidden_top5_logits\": [";
  for (size_t i = 0; i < top.size(); ++i) {
    if (i > 0) std::cout << ", ";
    std::cout << "{\"token\":" << top[i].second << ",\"logit\":" << top[i].first << "}";
  }
  std::cout << "],\n";

  // ---- Part 2: Per DeltaNet layer chunk vs recurrent state ----
  const auto& schedule = model::Qwen35Config::layer_schedule();
  std::cout << "  \"deltanet_layers\": [\n";
  for (uint32_t dn_idx = 0; dn_idx < prefill_chunks_.size(); ++dn_idx) {
    if (dn_idx > 0) std::cout << ",\n";

    uint32_t model_layer = 0, dn_count = 0;
    for (uint32_t l = 0; l < LAYERS; ++l) {
      if (schedule[l] != model::LayerKind::FullAttention) {
        if (dn_count == dn_idx) { model_layer = l; break; }
        ++dn_count;
      }
    }

    std::cout << "    {\"dn_idx\": " << dn_idx << ", \"model_layer\": " << model_layer;

    // Chunk core_attn_out stats
    const auto& attn_last = chunk_core_attn_out_last_[dn_idx];
    if (!attn_last.empty()) {
      double am = 0.0, ami = 1e30, ama = -1e30, asq = 0.0;
      for (float v : attn_last) {
        am += v;
        asq += static_cast<double>(v) * v;
        if (v < ami) ami = v;
        if (v > ama) ama = v;
      }
      am /= attn_last.size();
      double ast = std::sqrt(asq / attn_last.size() - am * am);
      std::cout << ", \"chunk_attn_out_count\": " << attn_last.size()
                << ", \"chunk_attn_out_mean\": " << am
                << ", \"chunk_attn_out_std\": " << ast
                << ", \"chunk_attn_out_min\": " << ami
                << ", \"chunk_attn_out_max\": " << ama;

      // Per-head summary
      std::cout << ", \"heads\": [";
      for (uint32_t h = 0; h < DN_HEADS; ++h) {
        if (h > 0) std::cout << ", ";
        double hm = 0.0, hmi = 1e30, hma = -1e30;
        for (uint32_t vd = 0; vd < DN_V_DIM; ++vd) {
          float v = attn_last[h * DN_V_DIM + vd];
          hm += v;
          if (v < hmi) hmi = v;
          if (v > hma) hma = v;
        }
        hm /= DN_V_DIM;
        std::cout << "{\"h\":" << h << ",\"mean\":" << hm
                  << ",\"min\":" << hmi << ",\"max\":" << hma << "}";
      }
      std::cout << "]";
    }

    // GPU recurrent state norm
    {
      size_t state_matrix_bytes = DN_HEADS * DN_K_DIM * DN_V_DIM * 4;
      auto staging = dev_.create_host_visible_buffer(state_matrix_bytes);
      auto cp_cmd = dev_.allocate_command_buffer();
      dev_.begin_command_buffer(cp_cmd);
      VkDeviceSize state_off = dn_idx * B.dn_state_per_layer;
      VkBufferCopy cp{state_off, 0, state_matrix_bytes};
      vkCmdCopyBuffer(cp_cmd, B.dn_state.buffer, staging.buffer, 1, &cp);
      dev_.end_command_buffer(cp_cmd);
      dev_.submit_and_wait(cp_cmd);
      std::vector<float> state_data(DN_HEADS * DN_K_DIM * DN_V_DIM);
      dev_.download_from_device(staging, state_data.data(), state_matrix_bytes);
      dev_.destroy_buffer(staging);

      double sn = 0.0;
      for (float v : state_data) sn += static_cast<double>(v) * v;
      sn = std::sqrt(sn);
      std::cout << ", \"gpu_state_norm\": " << sn;
    }

    // Pre-norm hidden state stats
    {
      auto staging = dev_.create_host_visible_buffer(HIDDEN * 2);
      auto cp_cmd = dev_.allocate_command_buffer();
      dev_.begin_command_buffer(cp_cmd);
      VkDeviceSize snap_off = model_layer * HIDDEN * 2;
      VkBufferCopy cp{snap_off, 0, HIDDEN * 2};
      vkCmdCopyBuffer(cp_cmd, prefill_snapshots_.buffer, staging.buffer, 1, &cp);
      dev_.end_command_buffer(cp_cmd);
      dev_.submit_and_wait(cp_cmd);
      std::vector<uint16_t> snap_fp16(HIDDEN);
      dev_.download_from_device(staging, snap_fp16.data(), HIDDEN * 2);
      dev_.destroy_buffer(staging);

      double sm = 0.0, smi = 1e30, sma = -1e30, snsq = 0.0;
      for (uint32_t i = 0; i < HIDDEN; ++i) {
        float v = half_to_float(snap_fp16[i]);
        sm += v;
        snsq += static_cast<double>(v) * v;
        if (v < smi) smi = v;
        if (v > sma) sma = v;
      }
      sm /= HIDDEN;
      snsq = std::sqrt(snsq);
      std::cout << ", \"pre_norm_hidden_mean\": " << sm
                << ", \"pre_norm_hidden_norm\": " << snsq
                << ", \"pre_norm_hidden_min\": " << smi
                << ", \"pre_norm_hidden_max\": " << sma;
    }

    std::cout << "}";
  }
  std::cout << "\n  ]\n";
  std::cout << "}\n";
}

// ---------------------------------------------------------------------------
// diagnose_decode_drift: compare free-run vs rebuilt prefill state
// ---------------------------------------------------------------------------
void DecodeSession::diagnose_decode_drift(
    const std::vector<uint32_t>& full_prefix_tokens,
    uint32_t target_decode_step,
    const std::vector<uint16_t>& free_hidden,
    const std::vector<float>& free_logits,
    const std::vector<std::vector<float>>& free_dn_state,
    const std::vector<std::vector<uint16_t>>& free_kv_cache,
    const std::vector<uint16_t>& free_layer_hidden) {
  const auto& B = *bufs_;
  const auto& D = *dsets_;
  const auto& P = *pipes_;

  uint32_t prefix_len = static_cast<uint32_t>(full_prefix_tokens.size());

  // --- Rebuilt path: reset, reprefill, chunk-correct, compute logits ---
  reset();
  layer_major_prefill(full_prefix_tokens, prefix_len, false);
  run_chunk_prefill();
  std::vector<uint16_t> rebuilt_layer_hidden;
  if (!free_layer_hidden.empty()) {
    rebuilt_layer_hidden.reserve(LAYERS * HIDDEN);
  }
  correct_last_token_hidden(full_prefix_tokens, prefix_len, &rebuilt_layer_hidden);

  // --- Capture rebuilt KV cache (first prefix_len positions per layer) ---
  std::vector<std::vector<uint16_t>> rebuilt_kv_cache(NUM_ATTN_LAYERS);
  if (!free_kv_cache.empty()) {
    VkDeviceSize kv_bytes_per_layer = static_cast<VkDeviceSize>(prefix_len) * 2 * KV_HEADS * HEAD_DIM * 2;
    for (uint32_t kv_ai = 0; kv_ai < NUM_ATTN_LAYERS; ++kv_ai) {
      VkDeviceSize layer_off = static_cast<VkDeviceSize>(kv_ai) * MAX_SEQ * 2 * KV_HEADS * HEAD_DIM * 2;
      rebuilt_kv_cache[kv_ai].resize(prefix_len * 2 * KV_HEADS * HEAD_DIM);
      auto staging = dev_.create_host_visible_buffer(kv_bytes_per_layer);
      auto cp_cmd = dev_.allocate_command_buffer();
      dev_.begin_command_buffer(cp_cmd);
      VkBufferCopy cp{layer_off, 0, kv_bytes_per_layer};
      vkCmdCopyBuffer(cp_cmd, B.kv_cache.buffer, staging.buffer, 1, &cp);
      dev_.end_command_buffer(cp_cmd);
      dev_.submit_and_wait(cp_cmd);
      dev_.download_from_device(staging, rebuilt_kv_cache[kv_ai].data(), kv_bytes_per_layer);
      dev_.destroy_buffer(staging);
    }
  }

  // Run final_norm + LM head on the rebuilt hidden state (act_a)
  {
    auto cmd = dev_.allocate_command_buffer();
    dev_.begin_command_buffer(cmd);

    constexpr float RMS_EPS = 1e-6f;
    struct { uint32_t N; uint32_t eps_bits; } fn_push = { HIDDEN, float_to_bits(RMS_EPS) };
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.rmsnorm);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        P.pipeline_layout_3, 0, 1, &D.final_norm, 0, nullptr);
    vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &fn_push);
    vkCmdDispatch(cmd, 1, 1, 1);

    VkBufferMemoryBarrier bmb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    bmb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bmb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.buffer = B.act_b.buffer;
    bmb.offset = 0;
    bmb.size = B.act_bytes;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &bmb, 0, nullptr);

    struct { uint32_t out_dim; uint32_t in_dim; } lm_push = { VOCAB, HIDDEN };
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.matvec);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        P.pipeline_layout_3, 0, 1, &D.lm_head, 0, nullptr);
    vkCmdPushConstants(cmd, P.pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &lm_push);
    vkCmdDispatch(cmd, (VOCAB + 63) / 64, 1, 1);

    dev_.end_command_buffer(cmd);
    dev_.submit_and_wait(cmd);
  }

  // --- Read back rebuilt hidden (act_a) ---
  std::vector<uint16_t> rebuilt_hidden(HIDDEN);
  dev_.download_from_device(B.act_a, rebuilt_hidden.data(), HIDDEN * 2);

  // --- Read back rebuilt logits ---
  std::vector<float> rebuilt_logits(VOCAB);
  {
    std::vector<uint16_t> rebuilt_logits_fp16(VOCAB);
    dev_.download_from_device(B.logits, rebuilt_logits_fp16.data(), VOCAB * 2);
    for (uint32_t i = 0; i < VOCAB; ++i) {
      rebuilt_logits[i] = half_to_float(rebuilt_logits_fp16[i]);
    }
  }

  // --- Read back rebuilt DN state per layer ---
  std::vector<std::vector<float>> rebuilt_dn_state(NUM_DN_LAYERS);
  for (uint32_t dn = 0; dn < NUM_DN_LAYERS; ++dn) {
    VkDeviceSize state_off = dn * B.dn_state_per_layer;
    size_t matrix_floats = DN_HEADS * DN_K_DIM * DN_V_DIM;
    rebuilt_dn_state[dn].resize(matrix_floats);
    auto staging = dev_.create_host_visible_buffer(matrix_floats * 4);
    auto cp_cmd = dev_.allocate_command_buffer();
    dev_.begin_command_buffer(cp_cmd);
    VkBufferCopy cp{state_off, 0, matrix_floats * 4};
    vkCmdCopyBuffer(cp_cmd, B.dn_state.buffer, staging.buffer, 1, &cp);
    dev_.end_command_buffer(cp_cmd);
    dev_.submit_and_wait(cp_cmd);
    dev_.download_from_device(staging, rebuilt_dn_state[dn].data(), matrix_floats * 4);
    dev_.destroy_buffer(staging);
  }

  // --- Compare and emit JSON to stderr ---
  auto top5 = [](const std::vector<float>& logits_f32) {
    std::vector<std::pair<float, uint32_t>> top;
    top.reserve(5);
    for (uint32_t i = 0; i < VOCAB; ++i) {
      float v = logits_f32[i];
      if (top.size() < 5) {
        top.emplace_back(v, i);
        if (top.size() == 5) std::sort(top.begin(), top.end(), std::greater<>());
      } else if (v > top.back().first) {
        top.back() = {v, i};
        std::sort(top.begin(), top.end(), std::greater<>());
      }
    }
    return top;
  };

  auto hidden_stats = [](const std::vector<uint16_t>& h_fp16) {
    double norm = 0.0;
    for (uint32_t i = 0; i < HIDDEN; ++i) {
      float v = half_to_float(h_fp16[i]);
      norm += static_cast<double>(v) * v;
    }
    return std::sqrt(norm);
  };

  auto hidden_absdiff = [](const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
    double max_abs = 0.0;
    double mean_abs = 0.0;
    for (uint32_t i = 0; i < HIDDEN; ++i) {
      float da = half_to_float(a[i]);
      float db = half_to_float(b[i]);
      double ad = std::abs(static_cast<double>(da) - static_cast<double>(db));
      mean_abs += ad;
      if (ad > max_abs) max_abs = ad;
    }
    mean_abs /= HIDDEN;
    return std::make_pair(max_abs, mean_abs);
  };

  auto dn_state_absdiff = [](const std::vector<float>& a, const std::vector<float>& b) {
    double max_abs = 0.0;
    double mean_abs = 0.0;
    size_t n = a.size();
    for (size_t i = 0; i < n; ++i) {
      double ad = std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
      mean_abs += ad;
      if (ad > max_abs) max_abs = ad;
    }
    mean_abs /= static_cast<double>(n);
    return std::make_pair(max_abs, mean_abs);
  };

  auto dn_state_norm = [](const std::vector<float>& s) {
    double sn = 0.0;
    for (float v : s) sn += static_cast<double>(v) * v;
    return std::sqrt(sn);
  };

  auto [hid_max, hid_mean] = hidden_absdiff(free_hidden, rebuilt_hidden);

  std::cerr << "{\n";
  std::cerr << "  \"diagnostic\": \"decode_drift\",\n";
  std::cerr << "  \"target_decode_step\": " << target_decode_step << ",\n";
  std::cerr << "  \"prefix_len\": " << prefix_len << ",\n";
  std::cerr << "  \"prefix_tokens\": [";
  for (size_t i = 0; i < full_prefix_tokens.size(); ++i) {
    if (i > 0) std::cerr << ", ";
    std::cerr << full_prefix_tokens[i];
  }
  std::cerr << "],\n";
  std::cerr << "  \"free_run\": {\n";
  std::cerr << "    \"hidden_norm\": " << hidden_stats(free_hidden) << ",\n";
  auto free_top = top5(free_logits);
  std::cerr << "    \"top5_logits\": [";
  for (size_t i = 0; i < free_top.size(); ++i) {
    if (i > 0) std::cerr << ", ";
    std::cerr << "{\"token\":" << free_top[i].second << ",\"logit\":" << free_top[i].first << "}";
  }
  std::cerr << "],\n";
  std::cerr << "    \"dn_state_norms\": [";
  for (uint32_t dn = 0; dn < NUM_DN_LAYERS; ++dn) {
    if (dn > 0) std::cerr << ", ";
    if (dn < free_dn_state.size())
      std::cerr << dn_state_norm(free_dn_state[dn]);
    else
      std::cerr << "null";
  }
  std::cerr << "]\n";
  std::cerr << "  },\n";
  std::cerr << "  \"rebuilt\": {\n";
  std::cerr << "    \"hidden_norm\": " << hidden_stats(rebuilt_hidden) << ",\n";
  auto rebuilt_top = top5(rebuilt_logits);
  std::cerr << "    \"top5_logits\": [";
  for (size_t i = 0; i < rebuilt_top.size(); ++i) {
    if (i > 0) std::cerr << ", ";
    std::cerr << "{\"token\":" << rebuilt_top[i].second << ",\"logit\":" << rebuilt_top[i].first << "}";
  }
  std::cerr << "],\n";
  std::cerr << "    \"dn_state_norms\": [";
  for (uint32_t dn = 0; dn < NUM_DN_LAYERS; ++dn) {
    if (dn > 0) std::cerr << ", ";
    std::cerr << dn_state_norm(rebuilt_dn_state[dn]);
  }
  std::cerr << "]\n";
  std::cerr << "  },\n";
  std::cerr << "  \"diff\": {\n";
  std::cerr << "    \"hidden_max_abs_diff\": " << hid_max << ",\n";
  std::cerr << "    \"hidden_mean_abs_diff\": " << hid_mean << ",\n";

  // Per-DeltaNet-layer state diffs, find worst layers
  std::vector<std::pair<double, uint32_t>> dn_diffs;
  dn_diffs.reserve(NUM_DN_LAYERS);
  for (uint32_t dn = 0; dn < NUM_DN_LAYERS; ++dn) {
    if (dn < free_dn_state.size() && dn < rebuilt_dn_state.size()) {
      auto [dmax, dmean] = dn_state_absdiff(free_dn_state[dn], rebuilt_dn_state[dn]);
      dn_diffs.emplace_back(dmax, dn);
    }
  }
  std::sort(dn_diffs.begin(), dn_diffs.end(), std::greater<>());
  uint32_t show_layers = std::min<uint32_t>(5, static_cast<uint32_t>(dn_diffs.size()));
  std::cerr << "    \"worst_dn_layers\": [\n";
  for (uint32_t ri = 0; ri < show_layers; ++ri) {
    uint32_t dn = dn_diffs[ri].second;
    if (ri > 0) std::cerr << ",\n";
    auto [dmax, dmean] = dn_state_absdiff(free_dn_state[dn], rebuilt_dn_state[dn]);
    double fn = dn_state_norm(free_dn_state[dn]);
    double rn = dn_state_norm(rebuilt_dn_state[dn]);
    std::cerr << "      {\"dn_idx\": " << dn
              << ", \"free_norm\": " << fn
              << ", \"rebuilt_norm\": " << rn
              << ", \"max_abs_diff\": " << dmax
              << ", \"mean_abs_diff\": " << dmean << "}";
  }
  std::cerr << "\n    ]\n";

  // --- KV cache diff ---
  auto kv_cache_norm = [](const std::vector<uint16_t>& kv) {
    double sn = 0.0;
    for (uint16_t v : kv) {
      float f = half_to_float(v);
      sn += static_cast<double>(f) * f;
    }
    return std::sqrt(sn);
  };

  auto kv_cache_absdiff = [](const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
    double max_abs = 0.0;
    double mean_abs = 0.0;
    size_t n = a.size();
    for (size_t i = 0; i < n; ++i) {
      double ad = std::abs(static_cast<double>(half_to_float(a[i])) - static_cast<double>(half_to_float(b[i])));
      mean_abs += ad;
      if (ad > max_abs) max_abs = ad;
    }
    mean_abs /= static_cast<double>(n);
    return std::make_pair(max_abs, mean_abs);
  };

  std::cerr << ",\n";
  // Find worst KV cache layers
  std::vector<std::pair<double, uint32_t>> kv_diffs;
  kv_diffs.reserve(NUM_ATTN_LAYERS);
  for (uint32_t kv_ai = 0; kv_ai < NUM_ATTN_LAYERS; ++kv_ai) {
    if (kv_ai < free_kv_cache.size() && kv_ai < rebuilt_kv_cache.size()) {
      auto [dmax, dmean] = kv_cache_absdiff(free_kv_cache[kv_ai], rebuilt_kv_cache[kv_ai]);
      kv_diffs.emplace_back(dmax, kv_ai);
    }
  }
  std::sort(kv_diffs.begin(), kv_diffs.end(), std::greater<>());
  uint32_t show_kv = std::min<uint32_t>(6, static_cast<uint32_t>(kv_diffs.size()));
  std::cerr << "    \"worst_kv_layers\": [\n";
  for (uint32_t ri = 0; ri < show_kv; ++ri) {
    uint32_t kv_ai = kv_diffs[ri].second;
    if (ri > 0) std::cerr << ",\n";
    auto [dmax, dmean] = kv_cache_absdiff(free_kv_cache[kv_ai], rebuilt_kv_cache[kv_ai]);
    double fn = kv_cache_norm(free_kv_cache[kv_ai]);
    double rn = kv_cache_norm(rebuilt_kv_cache[kv_ai]);
    // Model layer: attn_idx 0 -> layer 3, 1 -> 7, etc.
    uint32_t model_layer = kv_ai * 4 + 3;
    std::cerr << "      {\"attn_idx\": " << kv_ai
              << ", \"model_layer\": " << model_layer
              << ", \"free_norm\": " << fn
              << ", \"rebuilt_norm\": " << rn
              << ", \"max_abs_diff\": " << dmax
              << ", \"mean_abs_diff\": " << dmean << "}";
  }
  if (!free_layer_hidden.empty()) {
    std::cerr << "    ],\n";
  } else {
    std::cerr << "    ]\n";
  }

  // --- Per-layer hidden diff ---
  if (!free_layer_hidden.empty()) {
  constexpr uint32_t kHIDDEN = HIDDEN;
  auto layer_norm = [&](const std::vector<uint16_t>& lh, uint32_t li) -> double {
    double sn = 0.0;
    for (uint32_t i = 0; i < kHIDDEN; ++i) {
      float v = half_to_float(lh[li * kHIDDEN + i]);
      sn += static_cast<double>(v) * v;
    }
    return std::sqrt(sn);
  };

  auto layer_absdiff = [&](const std::vector<uint16_t>& a, const std::vector<uint16_t>& b, uint32_t li) -> std::pair<double, double> {
    double max_abs = 0.0;
    double mean_abs = 0.0;
    size_t base = static_cast<size_t>(li) * kHIDDEN;
    for (uint32_t i = 0; i < kHIDDEN; ++i) {
      double ad = std::abs(static_cast<double>(half_to_float(a[base + i])) -
                           static_cast<double>(half_to_float(b[base + i])));
      mean_abs += ad;
      if (ad > max_abs) max_abs = ad;
    }
    mean_abs /= kHIDDEN;
    return {max_abs, mean_abs};
  };

  std::vector<std::pair<double, uint32_t>> hidden_layer_diffs;
  hidden_layer_diffs.reserve(LAYERS);
  for (uint32_t li = 0; li < LAYERS; ++li) {
    auto [dmax, dmean] = layer_absdiff(free_layer_hidden, rebuilt_layer_hidden, li);
    hidden_layer_diffs.emplace_back(dmax, li);
  }
  std::sort(hidden_layer_diffs.begin(), hidden_layer_diffs.end(), std::greater<>());

  std::cerr << "    \"worst_hidden_layers\": [\n";
  for (uint32_t ri = 0; ri < LAYERS; ++ri) {
    if (ri > 0) std::cerr << ",\n";
    uint32_t li = hidden_layer_diffs[ri].second;
    auto [dmax, dmean] = layer_absdiff(free_layer_hidden, rebuilt_layer_hidden, li);
    double fn = layer_norm(free_layer_hidden, li);
    double rn = layer_norm(rebuilt_layer_hidden, li);
    std::cerr << "      {\"layer\": " << li
              << ", \"free_norm\": " << fn
              << ", \"rebuilt_norm\": " << rn
              << ", \"max_abs_diff\": " << dmax
              << ", \"mean_abs_diff\": " << dmean << "}";
  }
  std::cerr << "    ]\n";
    std::cerr << "  }\n";
  } else {
    std::cerr << "  }\n";
  }
  std::cerr << "}\n";
}
}  // namespace spock::runtime

#endif  // SPOCK_HAS_VULKAN

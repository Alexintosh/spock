#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cstdint>
#include <cstdlib>

#include <cmath>
#include <cstring>

#include <vulkan/vulkan.h>

#include "runtime/deltanet_chunk.hpp"
#include "runtime/vk_device.hpp"

// ---------------------------------------------------------------------------
// Tiled chunk-prefill probe: single dispatch for all heads/v-tiles.
// Matches runtime-l2-padded-submit case.
// ---------------------------------------------------------------------------

namespace {

constexpr uint32_t TILE_V = 16;

std::vector<std::uint32_t> read_spirv() {
  auto try_load = [](const std::string& path) -> std::vector<std::uint32_t> {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    auto size = f.tellg();
    f.seekg(0);
    if (size % 4 != 0) return {};
    std::vector<std::uint32_t> code(static_cast<std::size_t>(size) / 4);
    f.read(reinterpret_cast<char*>(code.data()), size);
    return code;
  };

  auto spv = try_load("build/shaders/deltanet_chunk_prefill_tiled.comp.spv");
  if (!spv.empty()) return spv;

#ifdef SHADER_DIR
  spv = try_load(std::string(SHADER_DIR) + "/deltanet_chunk_prefill_tiled.comp.spv");
  if (!spv.empty()) return spv;
#endif

  throw std::runtime_error(
      "cannot open shader: deltanet_chunk_prefill_tiled.comp.spv "
      "(tried build/shaders/ and SHADER_DIR)");
}

}  // namespace

// ---------------------------------------------------------------------------

int main() {
  // Fixed probe dimensions matching runtime-l2-padded-submit
  constexpr uint32_t num_heads   = 16;
  constexpr uint32_t seq_len     = 104;
  constexpr uint32_t total_seq   = 128;
  constexpr uint32_t chunk_size  = 64;
  constexpr uint32_t chunk_count = 2;
  constexpr uint32_t k_dim       = 128;
  constexpr uint32_t v_dim       = 128;

  // Data ranges
  constexpr float q_lo = -1.0f,    q_hi = 1.0f;
  constexpr float k_lo = -1.0f,    k_hi = 1.0f;
  constexpr float v_lo = -20.0f,   v_hi = 20.0f;
  constexpr float g_lo = -9.0f,    g_hi = -1e-6f;
  constexpr float beta_lo = 0.0f,  beta_hi = 1.0f;

  const char* case_name = "runtime-l2-padded-tiled";
  uint32_t tile_count = (v_dim + TILE_V - 1) / TILE_V;

#if SPOCK_HAS_VULKAN && !defined(SPOCK_VULKAN_STUB)
  try {
    // --- Initialize Vulkan ---
    spock::runtime::VulkanDevice dev;
    dev.initialize();

    // --- Descriptor set layout: 7 storage-buffer bindings ---
    std::vector<VkDescriptorSetLayoutBinding> bindings(7);
    for (std::uint32_t i = 0; i < 7; ++i) {
      bindings[i].binding = i;
      bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings[i].descriptorCount = 1;
      bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      bindings[i].pImmutableSamplers = nullptr;
    }

    VkDescriptorSetLayout desc_layout =
        dev.create_descriptor_set_layout(bindings);

    // Push constants: 10 uint32 fields (same layout as original, pad0 unused)
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
      uint32_t pad0;
    };

    VkPipelineLayout pipe_layout =
        dev.create_pipeline_layout(desc_layout, sizeof(PushConsts));

    // --- Shader module ---
    auto spirv = read_spirv();
    VkShaderModule shader = dev.create_shader_module(spirv);

    // --- Compute pipeline ---
    VkPipeline pipeline = dev.create_compute_pipeline(shader, pipe_layout);

    // --- Allocate host-visible buffers ---
    auto host_buf = [&](VkDeviceSize sz) { return dev.create_host_visible_buffer(sz); };

    VkDeviceSize sz_q    = static_cast<VkDeviceSize>(num_heads * seq_len * k_dim) * sizeof(float);
    VkDeviceSize sz_k    = sz_q;
    VkDeviceSize sz_v    = static_cast<VkDeviceSize>(num_heads * seq_len * v_dim) * sizeof(float);
    VkDeviceSize sz_g    = static_cast<VkDeviceSize>(num_heads * seq_len) * sizeof(float);
    VkDeviceSize sz_beta = sz_g;
    VkDeviceSize sz_out  = (static_cast<VkDeviceSize>(num_heads * total_seq * v_dim) +
                            static_cast<VkDeviceSize>(num_heads * k_dim * v_dim)) * sizeof(float);
    VkDeviceSize sz_init = static_cast<VkDeviceSize>(num_heads * k_dim * v_dim) * sizeof(float);

    auto q_buf    = host_buf(sz_q);
    auto k_buf    = host_buf(sz_k);
    auto v_buf    = host_buf(sz_v);
    auto g_buf    = host_buf(sz_g);
    auto beta_buf = host_buf(sz_beta);
    auto out_buf  = host_buf(sz_out);
    auto init_buf = host_buf(sz_init);

    // --- Generate pseudo-random data ---
    auto xorshift32 = [](uint32_t& s) -> uint32_t {
      s ^= s << 13;
      s ^= s >> 17;
      s ^= s << 5;
      return s;
    };

    auto bounded_rand = [&xorshift32](size_t n, float lo, float hi,
                                       uint32_t seed) -> std::vector<float> {
      std::vector<float> v(n);
      uint32_t s = seed;
      for (size_t i = 0; i < n; ++i) {
        float t = static_cast<float>(xorshift32(s)) * (1.0f / 4294967296.0f);
        v[i] = lo + t * (hi - lo);
      }
      return v;
    };

    auto query = bounded_rand(num_heads * seq_len * k_dim, q_lo, q_hi, 0xDEADBEEFu);
    auto key   = bounded_rand(num_heads * seq_len * k_dim, k_lo, k_hi, 0xCAFEBABEu);
    auto value = bounded_rand(num_heads * seq_len * v_dim, v_lo, v_hi, 0xDECAFBADu);
    auto g     = bounded_rand(num_heads * seq_len, g_lo, g_hi, 0xABADCAFEu);
    auto beta  = bounded_rand(num_heads * seq_len, beta_lo, beta_hi, 0xBEEFCACEu);

    // L2-normalize q and k per [head][token]
    {
      const float eps = 1e-6f;
      for (uint32_t h = 0; h < num_heads; ++h) {
        for (uint32_t t = 0; t < seq_len; ++t) {
          auto normalize = [&](std::vector<float>& vec, std::size_t off) {
            float nrm = 0.0f;
            for (uint32_t d = 0; d < k_dim; ++d)
              nrm += vec[off + d] * vec[off + d];
            nrm = std::sqrt(nrm + eps);
            for (uint32_t d = 0; d < k_dim; ++d)
              vec[off + d] /= nrm;
          };
          const std::size_t off = (static_cast<std::size_t>(h) * seq_len + t) * k_dim;
          normalize(query, off);
          normalize(key, off);
        }
      }
    }

    std::vector<float> initial_state(num_heads * k_dim * v_dim, 0.0f);

    // --- Upload to mapped buffers ---
    std::memcpy(q_buf.mapped,    query.data(),         static_cast<size_t>(sz_q));
    std::memcpy(k_buf.mapped,    key.data(),           static_cast<size_t>(sz_k));
    std::memcpy(v_buf.mapped,    value.data(),         static_cast<size_t>(sz_v));
    std::memcpy(g_buf.mapped,    g.data(),             static_cast<size_t>(sz_g));
    std::memcpy(beta_buf.mapped, beta.data(),          static_cast<size_t>(sz_beta));
    std::memcpy(init_buf.mapped, initial_state.data(), static_cast<size_t>(sz_init));

    // --- Allocate descriptor set ---
    VkDescriptorSet desc_set = dev.allocate_descriptor_set(desc_layout);
    dev.update_descriptor_set(desc_set, 0, q_buf);
    dev.update_descriptor_set(desc_set, 1, k_buf);
    dev.update_descriptor_set(desc_set, 2, v_buf);
    dev.update_descriptor_set(desc_set, 3, g_buf);
    dev.update_descriptor_set(desc_set, 4, beta_buf);
    dev.update_descriptor_set(desc_set, 5, out_buf);
    dev.update_descriptor_set(desc_set, 6, init_buf);

    // --- Push constants ---
    float inv_sqrt_kdim = 1.0f / std::sqrt(static_cast<float>(k_dim));
    uint32_t q_scale_bits;
    std::memcpy(&q_scale_bits, &inv_sqrt_kdim, sizeof(q_scale_bits));

    PushConsts push_consts = {
        num_heads, seq_len, k_dim, v_dim, chunk_size,
        q_scale_bits, total_seq, chunk_count,
        1u,  // use_qk_l2norm: data is already L2-normalized
        0u   // pad0 (unused)
    };

    // --- Single dispatch: all heads, all v-tiles in one command ---
    {
      VkCommandBuffer cmd = dev.allocate_command_buffer();
      dev.begin_command_buffer(cmd);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              pipe_layout, 0, 1, &desc_set, 0, nullptr);
      vkCmdPushConstants(cmd, pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                         0, sizeof(PushConsts), &push_consts);
      // Dispatch: (num_heads, ceil(v_dim/TILE_V), 1): one workgroup per (head, v-tile)
      vkCmdDispatch(cmd, num_heads, tile_count, 1);
      dev.end_command_buffer(cmd);
      dev.submit_and_wait(cmd);
    }

    // --- Read output ---
    size_t out_floats = static_cast<size_t>(sz_out) / sizeof(float);
    const float* out_data = static_cast<const float*>(out_buf.mapped);
    double l2 = 0.0;
    uint32_t nan_count = 0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < out_floats; ++i) {
      float x = out_data[i];
      if (std::isnan(x) || std::isinf(x)) {
        ++nan_count;
        continue;
      }
      double d = static_cast<double>(x);
      l2 += d * d;
      float ax = std::abs(x);
      if (ax > max_abs) max_abs = ax;
    }
    l2 = std::sqrt(l2);

    // --- CPU reference comparison ---
    float max_abs_core   = 0.0f;
    float max_rel_core   = 0.0f;
    float max_abs_state  = 0.0f;
    float max_rel_state  = 0.0f;
    bool  compare_ok     = false;
    std::vector<float> head_core_abs, head_core_rel, head_state_abs, head_state_rel;
    int worst_core_head = -1, worst_state_head = -1;

    {
      const spock::runtime::DeltaNetChunkConfig cpu_cfg{
          num_heads, seq_len, k_dim, v_dim, chunk_size, false};

      const spock::runtime::DeltaNetChunkInputs cpu_inputs{
          query, key, value, g, beta, {}};

      const auto cpu_out =
          spock::runtime::run_deltanet_chunk_rule(cpu_cfg, cpu_inputs);

      const std::size_t state_offset = static_cast<std::size_t>(num_heads) * total_seq * v_dim;
      const std::size_t state_count  = static_cast<std::size_t>(num_heads) * k_dim * v_dim;

      // Core attention comparison: GPU [head][total_seq][v_dim] vs CPU [head][seq_len][v_dim]
      {
        max_abs_core = 0.0f;
        max_rel_core = 0.0f;
        for (uint32_t h = 0; h < num_heads; ++h) {
          for (uint32_t t = 0; t < seq_len; ++t) {
            for (uint32_t vd = 0; vd < v_dim; ++vd) {
              const std::size_t gpu_idx = (static_cast<std::size_t>(h) * total_seq + t) * v_dim + vd;
              const std::size_t cpu_idx = (static_cast<std::size_t>(h) * seq_len + t) * v_dim + vd;
              const float gv = out_data[gpu_idx];
              const float cv = cpu_out.core_attn_out[cpu_idx];
              if (std::isnan(gv) || std::isinf(gv)) continue;
              const float ab = std::abs(gv - cv);
              if (ab > max_abs_core) max_abs_core = ab;
              const float denom = std::max(1.0f, std::abs(cv));
              const float rel = ab / denom;
              if (rel > max_rel_core) max_rel_core = rel;
            }
          }
        }
      }

      // State comparison
      auto max_abs_and_rel = [](const float* gpu, const std::vector<float>& cpu,
                                 std::size_t count, float& out_abs, float& out_rel) {
        out_abs = 0.0f;
        out_rel = 0.0f;
        for (std::size_t i = 0; i < count; ++i) {
          const float gv = gpu[i];
          const float cv = cpu[i];
          if (std::isnan(gv) || std::isinf(gv)) continue;
          const float ab = std::abs(gv - cv);
          if (ab > out_abs) out_abs = ab;
          const float denom = std::max(1.0f, std::abs(cv));
          const float rel = ab / denom;
          if (rel > out_rel) out_rel = rel;
        }
      };

      max_abs_and_rel(out_data + state_offset, cpu_out.final_state, state_count,
                      max_abs_state, max_rel_state);

      compare_ok = (nan_count == 0) &&
                   (max_rel_core <= 1e-4f) &&
                   (max_rel_state <= 1e-4f);

      // Per-head diagnostics
      {
        const std::size_t gpu_head_stride = static_cast<std::size_t>(total_seq) * v_dim;
        const std::size_t cpu_head_stride = static_cast<std::size_t>(seq_len) * v_dim;
        const std::size_t core_per_head_valid = static_cast<std::size_t>(seq_len) * v_dim;
        const std::size_t state_per_head = static_cast<std::size_t>(k_dim) * v_dim;
        head_core_abs.assign(num_heads, 0.0f);
        head_core_rel.assign(num_heads, 0.0f);
        head_state_abs.assign(num_heads, 0.0f);
        head_state_rel.assign(num_heads, 0.0f);

        for (uint32_t h = 0; h < num_heads; ++h) {
          // Core attention per-head
          {
            float ha = 0.0f, hr = 0.0f;
            const float* gpu_core = out_data + h * gpu_head_stride;
            const float* cpu_core = cpu_out.core_attn_out.data() + h * cpu_head_stride;
            for (std::size_t i = 0; i < core_per_head_valid; ++i) {
              const float gv = gpu_core[i];
              const float cv = cpu_core[i];
              if (std::isnan(gv) || std::isinf(gv)) continue;
              const float ab = std::abs(gv - cv);
              if (ab > ha) ha = ab;
              const float denom = std::max(1.0f, std::abs(cv));
              const float r = ab / denom;
              if (r > hr) hr = r;
            }
            head_core_abs[h] = ha;
            head_core_rel[h] = hr;
          }
          // State per-head
          {
            float ha = 0.0f, hr = 0.0f;
            const float* gpu_state = out_data + state_offset + h * state_per_head;
            const float* cpu_state = cpu_out.final_state.data() + h * state_per_head;
            for (std::size_t i = 0; i < state_per_head; ++i) {
              const float gv = gpu_state[i];
              const float cv = cpu_state[i];
              if (std::isnan(gv) || std::isinf(gv)) continue;
              const float ab = std::abs(gv - cv);
              if (ab > ha) ha = ab;
              const float denom = std::max(1.0f, std::abs(cv));
              const float r = ab / denom;
              if (r > hr) hr = r;
            }
            head_state_abs[h] = ha;
            head_state_rel[h] = hr;
          }
        }
      }

      // Find worst heads
      for (uint32_t h = 0; h < num_heads; ++h) {
        if (worst_core_head < 0 || head_core_rel[h] > head_core_rel[static_cast<std::size_t>(worst_core_head)])
          worst_core_head = static_cast<int>(h);
        if (worst_state_head < 0 || head_state_rel[h] > head_state_rel[static_cast<std::size_t>(worst_state_head)])
          worst_state_head = static_cast<int>(h);
      }
    }

    // --- Cleanup ---
    dev.destroy_buffer(q_buf);
    dev.destroy_buffer(k_buf);
    dev.destroy_buffer(v_buf);
    dev.destroy_buffer(g_buf);
    dev.destroy_buffer(beta_buf);
    dev.destroy_buffer(out_buf);
    dev.destroy_buffer(init_buf);
    dev.destroy_pipeline(pipeline);
    dev.destroy_shader_module(shader);
    dev.destroy_pipeline_layout(pipe_layout);
    dev.destroy_descriptor_set_layout(desc_layout);

    // --- Output JSON ---
    std::cout << "{\n";
    std::cout << "  \"shader\": \"deltanet_chunk_prefill_tiled.comp\",\n";
    std::cout << "  \"case\": \"" << case_name << "\",\n";
    std::cout << "  \"status\": \"" << (compare_ok ? "compare-ok" : "compare-fail") << "\",\n";
    std::cout << "  \"dispatch_mode\": \"tiled-single\",\n";
    std::cout << "  \"num_heads\": " << num_heads << ",\n";
    std::cout << "  \"seq_len\": " << seq_len << ",\n";
    std::cout << "  \"total_seq\": " << total_seq << ",\n";
    std::cout << "  \"chunk_size\": " << chunk_size << ",\n";
    std::cout << "  \"chunk_count\": " << chunk_count << ",\n";
    std::cout << "  \"k_dim\": " << k_dim << ",\n";
    std::cout << "  \"v_dim\": " << v_dim << ",\n";
    std::cout << "  \"tile_v\": " << TILE_V << ",\n";
    std::cout << "  \"tile_count\": " << tile_count << ",\n";
    std::cout << "  \"output_l2\": " << l2 << ",\n";
    std::cout << "  \"nan_count\": " << nan_count << ",\n";
    std::cout << "  \"max_abs\": " << max_abs << ",\n";
    std::cout << "  \"max_abs_core\": " << max_abs_core << ",\n";
    std::cout << "  \"max_abs_state\": " << max_abs_state << ",\n";
    std::cout << "  \"max_rel_core\": " << max_rel_core << ",\n";
    std::cout << "  \"max_rel_state\": " << max_rel_state << ",\n";
    std::cout << "  \"head_error_summary\": [\n";
    for (uint32_t h = 0; h < num_heads; ++h) {
      std::cout << "    {\"head\":" << h
                << ",\"max_abs_core\":" << head_core_abs[h]
                << ",\"max_rel_core\":" << head_core_rel[h]
                << ",\"max_abs_state\":" << head_state_abs[h]
                << ",\"max_rel_state\":" << head_state_rel[h] << "}";
      if (h + 1 < num_heads) std::cout << ",";
      std::cout << "\n";
    }
    std::cout << "  ],\n";
    std::cout << "  \"worst_core_head\": " << worst_core_head << ",\n";
    std::cout << "  \"worst_state_head\": " << worst_state_head << "\n";
    std::cout << "}\n";

    return 0;
  } catch (const std::exception& e) {
    std::cout << "{\n";
    std::cout << "  \"shader\": \"deltanet_chunk_prefill_tiled.comp\",\n";
    std::cout << "  \"case\": \"" << case_name << "\",\n";
    std::cout << "  \"total_seq\": " << total_seq << ",\n";
    std::cout << "  \"chunk_count\": " << chunk_count << ",\n";
    std::cout << "  \"status\": \"error\",\n";
    std::cout << "  \"message\": \"" << e.what() << "\"\n";
    std::cout << "}\n";
    return 1;
  }
#else
  std::cout << "{\n";
  std::cout << "  \"shader\": \"deltanet_chunk_prefill_tiled.comp\",\n";
  std::cout << "  \"case\": \"" << case_name << "\",\n";
  std::cout << "  \"total_seq\": " << total_seq << ",\n";
  std::cout << "  \"chunk_count\": " << chunk_count << ",\n";
  std::cout << "  \"status\": \"error\",\n";
  std::cout << "  \"message\": \"Vulkan not available (SPOCK_VULKAN_STUB)\"\n";
  std::cout << "}\n";
  return 1;
#endif
}

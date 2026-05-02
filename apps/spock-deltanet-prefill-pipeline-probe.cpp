#include <fstream>
#include <iostream>
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

namespace {

std::vector<std::uint32_t> read_spirv(const char* fname) {
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

  auto spv = try_load(std::string("build/shaders/") + fname);
  if (!spv.empty()) return spv;

#ifdef SHADER_DIR
  spv = try_load(std::string(SHADER_DIR) + "/" + fname);
  if (!spv.empty()) return spv;
#endif

  throw std::runtime_error(
      std::string("cannot open shader: ") + fname +
      " (tried build/shaders/ and SHADER_DIR)");
}

// ---------------------------------------------------------------------------
// Half-precision helpers
// ---------------------------------------------------------------------------

std::uint16_t float_to_half(float f) {
  std::uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  std::uint32_t sign     = (bits >> 31) & 1;
  std::uint32_t exp_bits = (bits >> 23) & 0xff;
  std::uint32_t mant     = bits & 0x7fffff;

  if (exp_bits == 0xff) {
    std::uint16_t h = static_cast<std::uint16_t>(
        (sign << 15) | (0x1f << 10) | (mant >> 13));
    if (mant != 0) h |= 1;
    return h;
  }

  if (exp_bits == 0) {
    std::uint32_t half_mant = mant >> 13;
    return static_cast<std::uint16_t>((sign << 15) | half_mant);
  }

  std::int32_t exp_signed = static_cast<std::int32_t>(exp_bits) - 127;
  if (exp_signed < -14) {
    std::int32_t shift = -14 - exp_signed;
    if (shift > 31) shift = 31;
    std::uint32_t half_mant = (mant | 0x800000) >> (13 + static_cast<std::uint32_t>(shift));
    return static_cast<std::uint16_t>((sign << 15) | half_mant);
  }

  if (exp_signed > 15) {
    return static_cast<std::uint16_t>((sign << 15) | (0x1f << 10));
  }

  std::uint32_t half_exp = static_cast<std::uint32_t>(exp_signed + 15);
  std::uint32_t half_mant = mant >> 13;
  return static_cast<std::uint16_t>((sign << 15) | (half_exp << 10) | half_mant);
}

float half_to_float(std::uint16_t h) {
  std::uint32_t sign = (h >> 15) & 1;
  std::uint32_t exp  = (h >> 10) & 0x1f;
  std::uint32_t mant = h & 0x3ff;

  std::uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign << 31;
    } else {
      std::int32_t e = -14;
      while ((mant & 0x400) == 0) {
        mant <<= 1;
        --e;
      }
      mant &= 0x3ff;
      std::uint32_t f32_exp = static_cast<std::uint32_t>(e + 127) & 0xff;
      bits = (sign << 31) | (f32_exp << 23) | (mant << 13);
    }
  } else if (exp == 0x1f) {
    bits = (sign << 31) | (0xffu << 23) | (mant << 13);
  } else {
    std::uint32_t f32_exp = (exp + 127u - 15u) & 0xff;
    bits = (sign << 31) | (f32_exp << 23) | (mant << 13);
  }
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

}  // namespace

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
  // Fixed probe dimensions
  constexpr uint32_t num_heads  = 16;
  constexpr uint32_t seq_len    = 104;
  constexpr uint32_t total_seq  = 128;
  constexpr uint32_t chunk_size = 64;
  constexpr uint32_t chunk_count = 2;
  constexpr uint32_t k_dim      = 128;
  constexpr uint32_t v_dim      = 128;

  // Data ranges
  constexpr float q_lo = -1.0f,    q_hi = 1.0f;
  constexpr float k_lo = -1.0f,    k_hi = 1.0f;
  constexpr float v_lo = -20.0f,   v_hi = 20.0f;
  constexpr float g_lo = -9.0f,    g_hi = -1e-6f;
  constexpr float beta_lo = 0.0f,  beta_hi = 1.0f;

#if SPOCK_HAS_VULKAN && !defined(SPOCK_VULKAN_STUB)
  try {
    // --- Initialize Vulkan ---
    spock::runtime::VulkanDevice dev;
    dev.initialize();

    // =====================================================================
    // Stage 1: Load both shaders and create collect pipeline
    // =====================================================================

    auto collect_spv = read_spirv("deltanet_prefill_collect.comp.spv");
    auto chunk_spv   = read_spirv("deltanet_chunk_prefill.comp.spv");

    // Collect pipeline: 7 bindings, 5-uint push constants
    {
      std::vector<VkDescriptorSetLayoutBinding> cl_bindings(7);
      for (std::uint32_t i = 0; i < 7; ++i) {
        cl_bindings[i].binding = i;
        cl_bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        cl_bindings[i].descriptorCount = 1;
        cl_bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        cl_bindings[i].pImmutableSamplers = nullptr;
      }

      VkDescriptorSetLayout cl_desc_layout =
          dev.create_descriptor_set_layout(cl_bindings);

      VkPipelineLayout cl_pipe_layout =
          dev.create_pipeline_layout(cl_desc_layout, 5 * sizeof(uint32_t));

      VkShaderModule cl_shader = dev.create_shader_module(collect_spv);
      VkPipeline cl_pipeline = dev.create_compute_pipeline(cl_shader, cl_pipe_layout);

      // ===================================================================
      // Stage 2: Generate deterministic pseudo-random data
      // ===================================================================

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

      auto query  = bounded_rand(num_heads * seq_len * k_dim, q_lo, q_hi, 0xDEADBEEFu);
      auto key    = bounded_rand(num_heads * seq_len * k_dim, k_lo, k_hi, 0xCAFEBABEu);
      auto value  = bounded_rand(num_heads * seq_len * v_dim, v_lo, v_hi, 0xDECAFBADu);
      auto g      = bounded_rand(num_heads * seq_len, g_lo, g_hi, 0xABADCAFEu);
      auto beta   = bounded_rand(num_heads * seq_len, beta_lo, beta_hi, 0xBEEFCACEu);

      // L2-normalize q and k per [head][token] before fp16 conversion
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

      // Build CPU reference data: L2-normalized then fp16 roundtripped q/k/v,
      // g/beta as-is (collect shader copies fp32→fp32 for g/beta).
      auto cpu_query = [&]() {
        std::vector<float> out(num_heads * seq_len * k_dim);
        for (size_t i = 0; i < out.size(); ++i)
          out[i] = half_to_float(float_to_half(query[i]));
        return out;
      }();
      auto cpu_key = [&]() {
        std::vector<float> out(num_heads * seq_len * k_dim);
        for (size_t i = 0; i < out.size(); ++i)
          out[i] = half_to_float(float_to_half(key[i]));
        return out;
      }();
      auto cpu_value = [&]() {
        std::vector<float> out(num_heads * seq_len * v_dim);
        for (size_t i = 0; i < out.size(); ++i)
          out[i] = half_to_float(float_to_half(value[i]));
        return out;
      }();
      auto cpu_g    = g;     // fp32→fp32 passthrough
      auto cpu_beta = beta;  // fp32→fp32 passthrough

      // ===================================================================
      // Stage 3: Allocate buffers
      // ===================================================================

      auto host_buf = [&](VkDeviceSize sz) { return dev.create_host_visible_buffer(sz); };

      // Per-token input buffers (reused each token dispatch)
      VkDeviceSize sz_dn_qkv = static_cast<VkDeviceSize>(
          num_heads * k_dim * 2 + num_heads * v_dim) * sizeof(std::uint16_t);
      VkDeviceSize sz_g_beta = static_cast<VkDeviceSize>(num_heads * 2) * sizeof(float);

      // Head-major collection output buffers (fp32)
      VkDeviceSize sz_q_coll  = static_cast<VkDeviceSize>(num_heads * seq_len * k_dim) * sizeof(float);
      VkDeviceSize sz_k_coll  = sz_q_coll;
      VkDeviceSize sz_v_coll  = static_cast<VkDeviceSize>(num_heads * seq_len * v_dim) * sizeof(float);
      VkDeviceSize sz_g_coll  = static_cast<VkDeviceSize>(num_heads * seq_len) * sizeof(float);
      VkDeviceSize sz_beta_coll = sz_g_coll;

      // Chunk shader output + init
      VkDeviceSize sz_out  = (static_cast<VkDeviceSize>(num_heads * total_seq * v_dim) +
                              static_cast<VkDeviceSize>(num_heads * k_dim * v_dim)) * sizeof(float);
      VkDeviceSize sz_init = static_cast<VkDeviceSize>(num_heads * k_dim * v_dim) * sizeof(float);

      auto dn_qkv_buf   = host_buf(sz_dn_qkv);
      auto g_beta_buf   = host_buf(sz_g_beta);
      auto q_coll_buf   = host_buf(sz_q_coll);
      auto k_coll_buf   = host_buf(sz_k_coll);
      auto v_coll_buf   = host_buf(sz_v_coll);
      auto g_coll_buf   = host_buf(sz_g_coll);
      auto beta_coll_buf = host_buf(sz_beta_coll);
      auto out_buf      = host_buf(sz_out);
      auto init_buf     = host_buf(sz_init);

      // Zero-initialize collection + out + init buffers
      std::memset(q_coll_buf.mapped,   0, static_cast<size_t>(sz_q_coll));
      std::memset(k_coll_buf.mapped,   0, static_cast<size_t>(sz_k_coll));
      std::memset(v_coll_buf.mapped,   0, static_cast<size_t>(sz_v_coll));
      std::memset(g_coll_buf.mapped,   0, static_cast<size_t>(sz_g_coll));
      std::memset(beta_coll_buf.mapped, 0, static_cast<size_t>(sz_beta_coll));
      std::memset(out_buf.mapped,      0, static_cast<size_t>(sz_out));
      std::memset(init_buf.mapped,     0, static_cast<size_t>(sz_init));

      // --- Collect descriptor set ---
      VkDescriptorSet cl_desc_set = dev.allocate_descriptor_set(cl_desc_layout);
      dev.update_descriptor_set(cl_desc_set, 0, dn_qkv_buf);
      dev.update_descriptor_set(cl_desc_set, 1, g_beta_buf);
      dev.update_descriptor_set(cl_desc_set, 2, q_coll_buf);
      dev.update_descriptor_set(cl_desc_set, 3, k_coll_buf);
      dev.update_descriptor_set(cl_desc_set, 4, v_coll_buf);
      dev.update_descriptor_set(cl_desc_set, 5, g_coll_buf);
      dev.update_descriptor_set(cl_desc_set, 6, beta_coll_buf);

      // ===================================================================
      // Stage 4: Per-token collect dispatch loop
      // ===================================================================
      {
        uint32_t wg_count = num_heads;  // one workgroup per head

        for (uint32_t t = 0; t < seq_len; ++t) {
          // Fill per-token dn_qkv (fp16)
          {
            auto* dst = static_cast<std::uint16_t*>(dn_qkv_buf.mapped);
            // Q section: num_heads * k_dim
            size_t q_off = 0;
            for (uint32_t h = 0; h < num_heads; ++h) {
              for (uint32_t d = 0; d < k_dim; ++d) {
                size_t src = (static_cast<size_t>(h) * seq_len + t) * k_dim + d;
                dst[q_off + h * k_dim + d] = float_to_half(query[src]);
              }
            }
            // K section: num_heads * k_dim (offset = num_heads * k_dim)
            size_t k_off = static_cast<size_t>(num_heads) * k_dim;
            for (uint32_t h = 0; h < num_heads; ++h) {
              for (uint32_t d = 0; d < k_dim; ++d) {
                size_t src = (static_cast<size_t>(h) * seq_len + t) * k_dim + d;
                dst[k_off + h * k_dim + d] = float_to_half(key[src]);
              }
            }
            // V section: num_heads * v_dim (offset = num_heads * k_dim * 2)
            size_t v_off = static_cast<size_t>(num_heads) * k_dim * 2;
            for (uint32_t h = 0; h < num_heads; ++h) {
              for (uint32_t d = 0; d < v_dim; ++d) {
                size_t src = (static_cast<size_t>(h) * seq_len + t) * v_dim + d;
                dst[v_off + h * v_dim + d] = float_to_half(value[src]);
              }
            }
          }

          // Fill per-token g_beta (fp32)
          {
            auto* dst = static_cast<float*>(g_beta_buf.mapped);
            for (uint32_t h = 0; h < num_heads; ++h) {
              size_t src = static_cast<size_t>(h) * seq_len + t;
              dst[h] = g[src];
            }
            for (uint32_t h = 0; h < num_heads; ++h) {
              size_t src = static_cast<size_t>(h) * seq_len + t;
              dst[num_heads + h] = beta[src];
            }
          }

          // Record and submit collect dispatch
          {
            uint32_t push[5] = {num_heads, seq_len, t, k_dim, v_dim};

            VkCommandBuffer cmd = dev.allocate_command_buffer();
            dev.begin_command_buffer(cmd);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cl_pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    cl_pipe_layout, 0, 1, &cl_desc_set, 0, nullptr);
            vkCmdPushConstants(cmd, cl_pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(push), push);
            vkCmdDispatch(cmd, wg_count, 1, 1);
            dev.end_command_buffer(cmd);
            dev.submit_and_wait(cmd);
          }
        }
      }

      // ===================================================================
      // Stage 5: Chunk pipeline — consumes collect output directly
      // ===================================================================

      // Create chunk pipeline (7 bindings, 10-uint push constants)
      std::vector<VkDescriptorSetLayoutBinding> ch_bindings(7);
      for (std::uint32_t i = 0; i < 7; ++i) {
        ch_bindings[i].binding = i;
        ch_bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ch_bindings[i].descriptorCount = 1;
        ch_bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        ch_bindings[i].pImmutableSamplers = nullptr;
      }

      VkDescriptorSetLayout ch_desc_layout =
          dev.create_descriptor_set_layout(ch_bindings);

      VkPipelineLayout ch_pipe_layout =
          dev.create_pipeline_layout(ch_desc_layout, 10 * sizeof(uint32_t));

      VkShaderModule ch_shader = dev.create_shader_module(chunk_spv);
      VkPipeline ch_pipeline = dev.create_compute_pipeline(ch_shader, ch_pipe_layout);

      // Allocate chunk descriptor set and bind buffers
      VkDescriptorSet ch_desc_set = dev.allocate_descriptor_set(ch_desc_layout);
      dev.update_descriptor_set(ch_desc_set, 0, q_coll_buf);
      dev.update_descriptor_set(ch_desc_set, 1, k_coll_buf);
      dev.update_descriptor_set(ch_desc_set, 2, v_coll_buf);
      dev.update_descriptor_set(ch_desc_set, 3, g_coll_buf);
      dev.update_descriptor_set(ch_desc_set, 4, beta_coll_buf);
      dev.update_descriptor_set(ch_desc_set, 5, out_buf);
      dev.update_descriptor_set(ch_desc_set, 6, init_buf);

      // --- Chunk push constants ---
      float inv_sqrt_kdim = 1.0f / std::sqrt(static_cast<float>(k_dim));
      uint32_t q_scale_bits;
      std::memcpy(&q_scale_bits, &inv_sqrt_kdim, sizeof(q_scale_bits));

      uint32_t ch_push[10] = {
        num_heads, seq_len, k_dim, v_dim, chunk_size,
        q_scale_bits, total_seq, chunk_count,
        1u,  // use_qk_l2norm — data is already L2-normalized
        0u   // base_head (will be overridden per-head)
      };

      // Per-head separate-submit dispatch (proven workaround)
      for (uint32_t h = 0; h < num_heads; ++h) {
        ch_push[9] = h;  // base_head

        VkCommandBuffer cmd = dev.allocate_command_buffer();
        dev.begin_command_buffer(cmd);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ch_pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                ch_pipe_layout, 0, 1, &ch_desc_set, 0, nullptr);
        vkCmdPushConstants(cmd, ch_pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(ch_push), ch_push);
        vkCmdDispatch(cmd, 1, 1, 1);
        dev.end_command_buffer(cmd);
        dev.submit_and_wait(cmd);
      }

      // ===================================================================
      // Stage 6: CPU reference and comparison
      // ===================================================================

      // Run CPU reference with fp16-roundtripped L2-normalized data
      const spock::runtime::DeltaNetChunkConfig cpu_cfg{
          num_heads, seq_len, k_dim, v_dim, chunk_size, false};

      const spock::runtime::DeltaNetChunkInputs cpu_inputs{
          cpu_query, cpu_key, cpu_value, cpu_g, cpu_beta, {}};

      const auto cpu_out =
          spock::runtime::run_deltanet_chunk_rule(cpu_cfg, cpu_inputs);

      // GPU output layout:
      //   out_data[0..num_heads*total_seq*v_dim-1] = core_attn_out (padded)
      //   out_data[num_heads*total_seq*v_dim..] = final_state (contiguous)
      const float* gpu_data = static_cast<const float*>(out_buf.mapped);

      const std::size_t state_offset = static_cast<std::size_t>(num_heads) * total_seq * v_dim;
      const std::size_t state_count  = static_cast<std::size_t>(num_heads) * k_dim * v_dim;

      // Core comparison: GPU [head][total_seq][v_dim] vs CPU compact [head][seq_len][v_dim]
      float max_abs_core = 0.0f;
      float max_rel_core = 0.0f;
      uint32_t nan_count = 0;
      {
        for (uint32_t h = 0; h < num_heads; ++h) {
          for (uint32_t t = 0; t < seq_len; ++t) {
            for (uint32_t vd = 0; vd < v_dim; ++vd) {
              const std::size_t gpu_idx = (static_cast<std::size_t>(h) * total_seq + t) * v_dim + vd;
              const std::size_t cpu_idx = (static_cast<std::size_t>(h) * seq_len + t) * v_dim + vd;
              const float gv = gpu_data[gpu_idx];
              if (std::isnan(gv)) {
                ++nan_count;
                continue;
              }
              const float cv = cpu_out.core_attn_out[cpu_idx];
              const float ab = std::abs(gv - cv);
              if (ab > max_abs_core) max_abs_core = ab;
              const float denom = std::max(1.0f, std::abs(cv));
              const float rel = ab / denom;
              if (rel > max_rel_core) max_rel_core = rel;
            }
          }
        }
      }

      // State comparison: both [head][k_dim][v_dim] contiguous
      float max_abs_state = 0.0f;
      float max_rel_state = 0.0f;
      {
        for (std::size_t i = 0; i < state_count; ++i) {
          const float gv = gpu_data[state_offset + i];
          if (std::isnan(gv)) {
            ++nan_count;
            continue;
          }
          const float cv = cpu_out.final_state[i];
          const float ab = std::abs(gv - cv);
          if (ab > max_abs_state) max_abs_state = ab;
          const float denom = std::max(1.0f, std::abs(cv));
          const float rel = ab / denom;
          if (rel > max_rel_state) max_rel_state = rel;
        }
      }

      bool compare_ok = (nan_count == 0) &&
                        (max_rel_core <= 1e-4f) &&
                        (max_rel_state <= 1e-4f);

      // ===================================================================
      // Cleanup
      // ===================================================================

      dev.destroy_buffer(dn_qkv_buf);
      dev.destroy_buffer(g_beta_buf);
      dev.destroy_buffer(q_coll_buf);
      dev.destroy_buffer(k_coll_buf);
      dev.destroy_buffer(v_coll_buf);
      dev.destroy_buffer(g_coll_buf);
      dev.destroy_buffer(beta_coll_buf);
      dev.destroy_buffer(out_buf);
      dev.destroy_buffer(init_buf);

      dev.destroy_pipeline(cl_pipeline);
      dev.destroy_pipeline(ch_pipeline);
      dev.destroy_shader_module(cl_shader);
      dev.destroy_shader_module(ch_shader);
      dev.destroy_pipeline_layout(cl_pipe_layout);
      dev.destroy_pipeline_layout(ch_pipe_layout);
      dev.destroy_descriptor_set_layout(cl_desc_layout);
      dev.destroy_descriptor_set_layout(ch_desc_layout);

      // ===================================================================
      // JSON output
      // ===================================================================

      std::cout << "{\n";
      std::cout << "  \"shader\": \"deltanet_prefill_collect+deltanet_chunk_prefill\",\n";
      std::cout << "  \"status\": \"" << (compare_ok ? "compare-ok" : "compare-fail") << "\",\n";
      std::cout << "  \"num_heads\": " << num_heads << ",\n";
      std::cout << "  \"seq_len\": " << seq_len << ",\n";
      std::cout << "  \"total_seq\": " << total_seq << ",\n";
      std::cout << "  \"chunk_size\": " << chunk_size << ",\n";
      std::cout << "  \"chunk_count\": " << chunk_count << ",\n";
      std::cout << "  \"max_rel_core\": " << max_rel_core << ",\n";
      std::cout << "  \"max_rel_state\": " << max_rel_state << ",\n";
      std::cout << "  \"max_abs_core\": " << max_abs_core << ",\n";
      std::cout << "  \"max_abs_state\": " << max_abs_state << ",\n";
      std::cout << "  \"nan_count\": " << nan_count << "\n";
      std::cout << "}\n";

      return 0;
    }
  } catch (const std::exception& e) {
    std::cout << "{\n";
    std::cout << "  \"shader\": \"deltanet_prefill_collect+deltanet_chunk_prefill\",\n";
    std::cout << "  \"status\": \"error\",\n";
    std::cout << "  \"num_heads\": " << num_heads << ",\n";
    std::cout << "  \"seq_len\": " << seq_len << ",\n";
    std::cout << "  \"total_seq\": " << total_seq << ",\n";
    std::cout << "  \"chunk_size\": " << chunk_size << ",\n";
    std::cout << "  \"chunk_count\": " << chunk_count << ",\n";
    std::cout << "  \"message\": \"" << e.what() << "\"\n";
    std::cout << "}\n";
    return 1;
  }
#else
  std::cout << "{\n";
  std::cout << "  \"shader\": \"deltanet_prefill_collect+deltanet_chunk_prefill\",\n";
  std::cout << "  \"status\": \"error\",\n";
  std::cout << "  \"num_heads\": " << num_heads << ",\n";
  std::cout << "  \"seq_len\": " << seq_len << ",\n";
  std::cout << "  \"total_seq\": " << total_seq << ",\n";
  std::cout << "  \"chunk_size\": " << chunk_size << ",\n";
  std::cout << "  \"chunk_count\": " << chunk_count << ",\n";
  std::cout << "  \"message\": \"Vulkan not available (SPOCK_VULKAN_STUB)\"\n";
  std::cout << "}\n";
  return 1;
#endif
}

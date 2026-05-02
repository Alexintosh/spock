#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <cstdint>
#include <cstdlib>

#include <cmath>
#include <cstring>

#include <vulkan/vulkan.h>

#include "runtime/vk_device.hpp"

// ---------------------------------------------------------------------------

namespace {

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

  auto spv = try_load("build/shaders/deltanet_prefill_collect.comp.spv");
  if (!spv.empty()) return spv;

#ifdef SHADER_DIR
  spv = try_load(std::string(SHADER_DIR) + "/deltanet_prefill_collect.comp.spv");
  if (!spv.empty()) return spv;
#endif

  throw std::runtime_error(
      "cannot open shader: deltanet_prefill_collect.comp.spv "
      "(tried build/shaders/ and SHADER_DIR)");
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
    // Inf or NaN
    std::uint16_t h = static_cast<std::uint16_t>(
        (sign << 15) | (0x1f << 10) | (mant >> 13));
    if (mant != 0) h |= 1;  // ensure NaN stays NaN
    return h;
  }

  if (exp_bits == 0) {
    // Zero or subnormal
    std::uint32_t half_mant = mant >> 13;
    return static_cast<std::uint16_t>((sign << 15) | half_mant);
  }

  std::int32_t exp_signed = static_cast<std::int32_t>(exp_bits) - 127;
  if (exp_signed < -14) {
    // Subnormal in half
    std::int32_t shift = -14 - exp_signed;
    if (shift > 31) shift = 31;
    std::uint32_t half_mant = (mant | 0x800000) >> (13 + static_cast<std::uint32_t>(shift));
    return static_cast<std::uint16_t>((sign << 15) | half_mant);
  }

  if (exp_signed > 15) {
    // Overflow to inf
    return static_cast<std::uint16_t>((sign << 15) | (0x1f << 10));
  }

  // Normal half
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
    // zero / subnormal
    if (mant == 0) {
      bits = sign << 31;
    } else {
      // subnormal: normalize
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
    // inf / nan
    bits = (sign << 31) | (0xffu << 23) | (mant << 13);
  } else {
    // normal
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
  constexpr uint32_t num_heads = 16;
  constexpr uint32_t seq_len   = 104;
  constexpr uint32_t k_dim     = 128;
  constexpr uint32_t v_dim     = 128;

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

    // --- Descriptor set layout: 7 storage-buffer bindings, compute stage ---
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

    // --- Push constants: 5 uint32 fields ---
    struct PushConsts {
      uint32_t num_heads;
      uint32_t seq_len;
      uint32_t token_idx;
      uint32_t k_dim;
      uint32_t v_dim;
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

    // Per-token input buffers (reused each token dispatch)
    VkDeviceSize sz_dn_qkv = static_cast<VkDeviceSize>(
        num_heads * k_dim * 2 + num_heads * v_dim) * sizeof(std::uint16_t);
    VkDeviceSize sz_g_beta = static_cast<VkDeviceSize>(num_heads * 2) * sizeof(float);

    // Head-major output collection buffers
    VkDeviceSize sz_q_out  = static_cast<VkDeviceSize>(num_heads * seq_len * k_dim) * sizeof(float);
    VkDeviceSize sz_k_out  = sz_q_out;
    VkDeviceSize sz_v_out  = static_cast<VkDeviceSize>(num_heads * seq_len * v_dim) * sizeof(float);
    VkDeviceSize sz_g_out  = static_cast<VkDeviceSize>(num_heads * seq_len) * sizeof(float);
    VkDeviceSize sz_beta_out = sz_g_out;

    auto dn_qkv_buf  = host_buf(sz_dn_qkv);
    auto g_beta_buf  = host_buf(sz_g_beta);
    auto q_out_buf   = host_buf(sz_q_out);
    auto k_out_buf   = host_buf(sz_k_out);
    auto v_out_buf   = host_buf(sz_v_out);
    auto g_out_buf   = host_buf(sz_g_out);
    auto beta_out_buf = host_buf(sz_beta_out);

    // Zero-initialize output buffers
    std::memset(q_out_buf.mapped,   0, static_cast<size_t>(sz_q_out));
    std::memset(k_out_buf.mapped,   0, static_cast<size_t>(sz_k_out));
    std::memset(v_out_buf.mapped,   0, static_cast<size_t>(sz_v_out));
    std::memset(g_out_buf.mapped,   0, static_cast<size_t>(sz_g_out));
    std::memset(beta_out_buf.mapped, 0, static_cast<size_t>(sz_beta_out));

    // --- Allocate descriptor set ---
    VkDescriptorSet desc_set = dev.allocate_descriptor_set(desc_layout);

    // --- Update bindings 0..6 ---
    dev.update_descriptor_set(desc_set, 0, dn_qkv_buf);
    dev.update_descriptor_set(desc_set, 1, g_beta_buf);
    dev.update_descriptor_set(desc_set, 2, q_out_buf);
    dev.update_descriptor_set(desc_set, 3, k_out_buf);
    dev.update_descriptor_set(desc_set, 4, v_out_buf);
    dev.update_descriptor_set(desc_set, 5, g_out_buf);
    dev.update_descriptor_set(desc_set, 6, beta_out_buf);

    // --- Generate deterministic expected data on CPU ---
    // xorshift32 PRNG
    auto xorshift32 = [](uint32_t& s) -> uint32_t {
      s ^= s << 13;
      s ^= s >> 17;
      s ^= s << 5;
      return s;
    };

    // Generate n floats via xorshift, scaled to [lo, hi], seeded per-array
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

    // Generate per-token original float data
    // q: [num_heads * seq_len * k_dim]
    // k: [num_heads * seq_len * k_dim]
    // v: [num_heads * seq_len * v_dim]
    // g: [num_heads * seq_len]
    // beta: [num_heads * seq_len]
    auto query  = bounded_rand(num_heads * seq_len * k_dim, q_lo, q_hi, 0xDEADBEEFu);
    auto key    = bounded_rand(num_heads * seq_len * k_dim, k_lo, k_hi, 0xCAFEBABEu);
    auto value  = bounded_rand(num_heads * seq_len * v_dim, v_lo, v_hi, 0xDECAFBADu);
    auto g      = bounded_rand(num_heads * seq_len, g_lo, g_hi, 0xABADCAFEu);
    auto beta   = bounded_rand(num_heads * seq_len, beta_lo, beta_hi, 0xBEEFCACEu);

    // Build expected head-major output arrays.
    // q/k/v undergo fp16 roundtrip; g/beta stay exact.
    auto expected_q = [&]() {
      std::vector<float> exp(num_heads * seq_len * k_dim);
      for (uint32_t h = 0; h < num_heads; ++h) {
        for (uint32_t t = 0; t < seq_len; ++t) {
          for (uint32_t d = 0; d < k_dim; ++d) {
            // Original input is in token-major order: [head][token][dim]
            size_t src = (static_cast<size_t>(h) * seq_len + t) * k_dim + d;
            // Output is head-major: [head][seq_len][k_dim]
            size_t dst = (static_cast<size_t>(h) * seq_len + t) * k_dim + d;
            uint16_t hf = float_to_half(query[src]);
            exp[dst] = half_to_float(hf);
          }
        }
      }
      return exp;
    }();

    auto expected_k = [&]() {
      std::vector<float> exp(num_heads * seq_len * k_dim);
      for (uint32_t h = 0; h < num_heads; ++h) {
        for (uint32_t t = 0; t < seq_len; ++t) {
          for (uint32_t d = 0; d < k_dim; ++d) {
            size_t src = (static_cast<size_t>(h) * seq_len + t) * k_dim + d;
            size_t dst = (static_cast<size_t>(h) * seq_len + t) * k_dim + d;
            uint16_t hf = float_to_half(key[src]);
            exp[dst] = half_to_float(hf);
          }
        }
      }
      return exp;
    }();

    auto expected_v = [&]() {
      std::vector<float> exp(num_heads * seq_len * v_dim);
      for (uint32_t h = 0; h < num_heads; ++h) {
        for (uint32_t t = 0; t < seq_len; ++t) {
          for (uint32_t d = 0; d < v_dim; ++d) {
            size_t src = (static_cast<size_t>(h) * seq_len + t) * v_dim + d;
            size_t dst = (static_cast<size_t>(h) * seq_len + t) * v_dim + d;
            uint16_t hf = float_to_half(value[src]);
            exp[dst] = half_to_float(hf);
          }
        }
      }
      return exp;
    }();

    auto expected_g = [&]() {
      std::vector<float> exp(num_heads * seq_len);
      for (uint32_t h = 0; h < num_heads; ++h) {
        for (uint32_t t = 0; t < seq_len; ++t) {
          size_t src = static_cast<size_t>(h) * seq_len + t;
          size_t dst = static_cast<size_t>(h) * seq_len + t;
          exp[dst] = g[src];
        }
      }
      return exp;
    }();

    auto expected_beta = [&]() {
      std::vector<float> exp(num_heads * seq_len);
      for (uint32_t h = 0; h < num_heads; ++h) {
        for (uint32_t t = 0; t < seq_len; ++t) {
          size_t src = static_cast<size_t>(h) * seq_len + t;
          size_t dst = static_cast<size_t>(h) * seq_len + t;
          exp[dst] = beta[src];
        }
      }
      return exp;
    }();

    // --- Per-token dispatch loop ---
    // Workgroup count = num_heads
    uint32_t wg_count = num_heads;

    for (uint32_t t = 0; t < seq_len; ++t) {
      // --- Fill per-token dn_qkv (fp16) ---
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
        // K section: num_heads * k_dim
        size_t k_off = static_cast<size_t>(num_heads) * k_dim;
        for (uint32_t h = 0; h < num_heads; ++h) {
          for (uint32_t d = 0; d < k_dim; ++d) {
            size_t src = (static_cast<size_t>(h) * seq_len + t) * k_dim + d;
            dst[k_off + h * k_dim + d] = float_to_half(key[src]);
          }
        }
        // V section: num_heads * v_dim
        size_t v_off = static_cast<size_t>(num_heads) * k_dim * 2;
        for (uint32_t h = 0; h < num_heads; ++h) {
          for (uint32_t d = 0; d < v_dim; ++d) {
            size_t src = (static_cast<size_t>(h) * seq_len + t) * v_dim + d;
            dst[v_off + h * v_dim + d] = float_to_half(value[src]);
          }
        }
      }

      // --- Fill per-token g_beta (fp32) ---
      {
        auto* dst = static_cast<float*>(g_beta_buf.mapped);
        // g section: num_heads
        for (uint32_t h = 0; h < num_heads; ++h) {
          size_t src = static_cast<size_t>(h) * seq_len + t;
          dst[h] = g[src];
        }
        // beta section: num_heads
        for (uint32_t h = 0; h < num_heads; ++h) {
          size_t src = static_cast<size_t>(h) * seq_len + t;
          dst[num_heads + h] = beta[src];
        }
      }

      // --- Record command buffer: bind, push constants, dispatch, submit ---
      PushConsts push = {num_heads, seq_len, t, k_dim, v_dim};

      VkCommandBuffer cmd = dev.allocate_command_buffer();
      dev.begin_command_buffer(cmd);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              pipe_layout, 0, 1, &desc_set, 0, nullptr);
      vkCmdPushConstants(cmd, pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                         0, sizeof(PushConsts), &push);
      vkCmdDispatch(cmd, wg_count, 1, 1);
      dev.end_command_buffer(cmd);
      dev.submit_and_wait(cmd);
    }

    // --- Read back output buffers and compare ---
    const float* gpu_q    = static_cast<const float*>(q_out_buf.mapped);
    const float* gpu_k    = static_cast<const float*>(k_out_buf.mapped);
    const float* gpu_v    = static_cast<const float*>(v_out_buf.mapped);
    const float* gpu_g    = static_cast<const float*>(g_out_buf.mapped);
    const float* gpu_beta = static_cast<const float*>(beta_out_buf.mapped);

    auto compare_buf = [](const float* gpu, const float* cpu, size_t count,
                           float& max_abs, float& max_rel, uint32_t& nan_count) {
      max_abs = 0.0f;
      max_rel = 0.0f;
      nan_count = 0;
      for (size_t i = 0; i < count; ++i) {
        float gv = gpu[i];
        if (std::isnan(gv)) {
          ++nan_count;
          continue;
        }
        float cv = cpu[i];
        float ab = std::abs(gv - cv);
        if (ab > max_abs) max_abs = ab;
        float denom = std::max(1.0f, std::abs(cv));
        float rel = ab / denom;
        if (rel > max_rel) max_rel = rel;
      }
    };

    float max_abs_q = 0, max_rel_q = 0;
    float max_abs_k = 0, max_rel_k = 0;
    float max_abs_v = 0, max_rel_v = 0;
    float max_abs_g = 0, max_rel_g = 0;
    float max_abs_beta = 0, max_rel_beta = 0;
    uint32_t nan_q = 0, nan_k = 0, nan_v = 0, nan_g = 0, nan_beta = 0;

    compare_buf(gpu_q,    expected_q.data(),    expected_q.size(),    max_abs_q,    max_rel_q,    nan_q);
    compare_buf(gpu_k,    expected_k.data(),    expected_k.size(),    max_abs_k,    max_rel_k,    nan_k);
    compare_buf(gpu_v,    expected_v.data(),    expected_v.size(),    max_abs_v,    max_rel_v,    nan_v);
    compare_buf(gpu_g,    expected_g.data(),    expected_g.size(),    max_abs_g,    max_rel_g,    nan_g);
    compare_buf(gpu_beta, expected_beta.data(), expected_beta.size(), max_abs_beta, max_rel_beta, nan_beta);

    uint32_t total_nan = nan_q + nan_k + nan_v + nan_g + nan_beta;
    float max_rel_all = max_rel_q;
    if (max_rel_k  > max_rel_all) max_rel_all = max_rel_k;
    if (max_rel_v  > max_rel_all) max_rel_all = max_rel_v;
    if (max_rel_g  > max_rel_all) max_rel_all = max_rel_g;
    if (max_rel_beta > max_rel_all) max_rel_all = max_rel_beta;

    bool compare_ok = (total_nan == 0) && (max_rel_all <= 1e-5f);

    // --- Cleanup ---
    dev.destroy_buffer(dn_qkv_buf);
    dev.destroy_buffer(g_beta_buf);
    dev.destroy_buffer(q_out_buf);
    dev.destroy_buffer(k_out_buf);
    dev.destroy_buffer(v_out_buf);
    dev.destroy_buffer(g_out_buf);
    dev.destroy_buffer(beta_out_buf);
    dev.destroy_pipeline(pipeline);
    dev.destroy_shader_module(shader);
    dev.destroy_pipeline_layout(pipe_layout);
    dev.destroy_descriptor_set_layout(desc_layout);

    // --- JSON output ---
    std::cout << "{\n";
    std::cout << "  \"shader\": \"deltanet_prefill_collect.comp\",\n";
    std::cout << "  \"status\": \"" << (compare_ok ? "compare-ok" : "compare-fail") << "\",\n";
    std::cout << "  \"num_heads\": " << num_heads << ",\n";
    std::cout << "  \"seq_len\": " << seq_len << ",\n";
    std::cout << "  \"k_dim\": " << k_dim << ",\n";
    std::cout << "  \"v_dim\": " << v_dim << ",\n";
    std::cout << "  \"max_rel_q\": " << max_rel_q << ",\n";
    std::cout << "  \"max_rel_k\": " << max_rel_k << ",\n";
    std::cout << "  \"max_rel_v\": " << max_rel_v << ",\n";
    std::cout << "  \"max_rel_g\": " << max_rel_g << ",\n";
    std::cout << "  \"max_rel_beta\": " << max_rel_beta << ",\n";
    std::cout << "  \"max_abs_q\": " << max_abs_q << ",\n";
    std::cout << "  \"max_abs_k\": " << max_abs_k << ",\n";
    std::cout << "  \"max_abs_v\": " << max_abs_v << ",\n";
    std::cout << "  \"max_abs_g\": " << max_abs_g << ",\n";
    std::cout << "  \"max_abs_beta\": " << max_abs_beta << ",\n";
    std::cout << "  \"nan_count\": " << total_nan << "\n";
    std::cout << "}\n";

    return 0;
  } catch (const std::exception& e) {
    std::cout << "{\n";
    std::cout << "  \"shader\": \"deltanet_prefill_collect.comp\",\n";
    std::cout << "  \"status\": \"error\",\n";
    std::cout << "  \"num_heads\": " << num_heads << ",\n";
    std::cout << "  \"seq_len\": " << seq_len << ",\n";
    std::cout << "  \"k_dim\": " << k_dim << ",\n";
    std::cout << "  \"v_dim\": " << v_dim << ",\n";
    std::cout << "  \"message\": \"" << e.what() << "\"\n";
    std::cout << "}\n";
    return 1;
  }
#else
  std::cout << "{\n";
  std::cout << "  \"shader\": \"deltanet_prefill_collect.comp\",\n";
  std::cout << "  \"status\": \"error\",\n";
  std::cout << "  \"num_heads\": " << num_heads << ",\n";
  std::cout << "  \"seq_len\": " << seq_len << ",\n";
  std::cout << "  \"k_dim\": " << k_dim << ",\n";
  std::cout << "  \"v_dim\": " << v_dim << ",\n";
  std::cout << "  \"message\": \"Vulkan not available (SPOCK_VULKAN_STUB)\"\n";
  std::cout << "}\n";
  return 1;
#endif
}

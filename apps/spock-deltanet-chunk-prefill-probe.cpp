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

  auto spv = try_load("build/shaders/deltanet_chunk_prefill.comp.spv");
  if (!spv.empty()) return spv;

#ifdef SHADER_DIR
  spv = try_load(std::string(SHADER_DIR) + "/deltanet_chunk_prefill.comp.spv");
  if (!spv.empty()) return spv;
#endif

  throw std::runtime_error(
      "cannot open shader: deltanet_chunk_prefill.comp.spv "
      "(tried build/shaders/ and SHADER_DIR)");
}

}  // namespace

// ---------------------------------------------------------------------------
// Probe case definition
// ---------------------------------------------------------------------------

struct ProbeCase {
    const char* name;
    uint32_t num_heads, seq_len, k_dim, v_dim, chunk_size, total_seq, chunk_count;
    float q_lo, q_hi, k_lo, k_hi, v_lo, v_hi, g_lo, g_hi, beta_lo, beta_hi;
    bool repeat_per_head = false;
    bool pseudo_random = false;
    bool l2_normalize_qk = false;
    bool separate_head_submits = false;
};

static const ProbeCase kCases[] = {
    {"real-width", 1, 4, 128, 128, 4, 4, 1,
     -0.01f, 0.01f, -0.01f, 0.01f, -0.01f, 0.01f, -0.04f, -0.01f, 0.1f, 0.4f},
    {"real-chunk", 1, 64, 128, 128, 64, 64, 1,
     -0.005f, 0.005f, -0.005f, 0.005f, -0.005f, 0.005f, -0.015f, -0.002f, 0.05f, 0.2f},
    {"two-chunks", 1, 128, 128, 128, 64, 128, 2,
     -0.003f, 0.003f, -0.003f, 0.003f, -0.003f, 0.003f, -0.012f, -0.0015f, 0.04f, 0.16f},
    {"multi-head", 16, 64, 128, 128, 64, 64, 1,
     -0.002f, 0.002f, -0.002f, 0.002f, -0.002f, 0.002f, -0.010f, -0.001f, 0.03f, 0.12f},
    {"multi-head-repeat", 16, 64, 128, 128, 64, 64, 1,
     -0.002f, 0.002f, -0.002f, 0.002f, -0.002f, 0.002f, -0.010f, -0.001f, 0.03f, 0.12f,
     true},
    {"multi-head-padded", 16, 104, 128, 128, 64, 128, 2,
     -0.002f, 0.002f, -0.002f, 0.002f, -0.002f, 0.002f, -0.010f, -0.001f, 0.03f, 0.12f},
    {"runtime-range-padded", 16, 104, 128, 128, 64, 128, 2,
     -1.0f, 1.0f, -1.0f, 1.0f, -20.0f, 20.0f, -9.0f, -1e-6f, 0.0f, 1.0f,
     false, true},
    {"runtime-l2-padded", 16, 104, 128, 128, 64, 128, 2,
     -1.0f, 1.0f, -1.0f, 1.0f, -20.0f, 20.0f, -9.0f, -1e-6f, 0.0f, 1.0f,
     false, true, true},
    {"runtime-l2-padded-submit", 16, 104, 128, 128, 64, 128, 2,
     -1.0f, 1.0f, -1.0f, 1.0f, -20.0f, 20.0f, -9.0f, -1e-6f, 0.0f, 1.0f,
     false, true, true, true},
};

static const ProbeCase* find_case(const std::string& name) {
    for (const auto& c : kCases) {
        if (name == c.name) return &c;
    }
    return nullptr;
}

// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
  // --- Parse CLI ---
  std::string case_name = "real-width";

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help") {
      std::cout << "usage: spock-deltanet-chunk-prefill-probe [--help] [--case <name>]\n\n"
                << "cases:\n"
                << "  real-width   verified small case (default)\n"
                << "  real-chunk   real-width chunk-size=64 seq_len=64\n"
                << "  two-chunks   2 chunks of 64, seq_len=128 total_seq=128\n"
                << "  multi-head   16 heads, seq_len=64, 1 chunk, verify per-head dispatch\n"
               << "  multi-head-repeat 16 heads, repeated data per head, isolate indexing\n"
               << "  multi-head-padded  16 heads, seq_len=104, 2 chunks of 64, reproduce pp520_046 failure\n"
               << "  runtime-range-padded  16 heads, seq_len=104, 2 chunks of 64, pseudo-random ranges\n"
               << "  runtime-l2-padded  16 heads, seq_len=104, 2 chunks of 64, pseudo-random L2-normalized q/k\n"
               << "  runtime-l2-padded-submit  16 heads, seq_len=104, 2 chunks of 64, pseudo-random L2-normalized q/k, separate head submits\n";
      return 0;
    }
    if (arg == "--case") {
      if (i + 1 < argc) {
        case_name = argv[++i];
      } else {
        std::cout << "{\"status\":\"error\",\"message\":\"--case requires a value\"}\n";
        return 1;
      }
    } else {
      std::cout << "{\"status\":\"error\",\"message\":\"unknown argument: " << arg << "\"}\n";
      return 1;
    }
  }

  // --- Select case config ---
  const ProbeCase* pc = find_case(case_name);
  if (pc == nullptr) {
    std::cout << "{\"status\":\"error\",\"message\":\"unknown case: " << case_name << "\"}\n";
    return 1;
  }

  uint32_t num_heads = pc->num_heads;
  uint32_t seq_len   = pc->seq_len;
  uint32_t k_dim     = pc->k_dim;
  uint32_t v_dim     = pc->v_dim;
  uint32_t chunk_size = pc->chunk_size;
  uint32_t total_seq  = pc->total_seq;
  uint32_t chunk_count = pc->chunk_count;
  float q_lo = pc->q_lo, q_hi = pc->q_hi;
  float k_lo = pc->k_lo, k_hi = pc->k_hi;
  float v_lo = pc->v_lo, v_hi = pc->v_hi;
  float g_lo = pc->g_lo, g_hi = pc->g_hi;
  float beta_lo = pc->beta_lo, beta_hi = pc->beta_hi;

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

    // --- Pipeline layout: 40 bytes push constant (10 uint32 fields) ---
    VkPipelineLayout pipe_layout =
        dev.create_pipeline_layout(desc_layout, sizeof(PushConsts));

    // --- Shader module ---
    auto spirv = read_spirv();
    VkShaderModule shader = dev.create_shader_module(spirv);

    // --- Compute pipeline ---
    VkPipeline pipeline = dev.create_compute_pipeline(shader, pipe_layout);


    // --- Allocate host-visible buffers for all 7 bindings ---
    auto host_buf = [&](VkDeviceSize sz) { return dev.create_host_visible_buffer(sz); };

    VkDeviceSize sz_q    = static_cast<VkDeviceSize>(num_heads * seq_len * k_dim) * sizeof(float);
    VkDeviceSize sz_k    = sz_q;
    VkDeviceSize sz_v    = static_cast<VkDeviceSize>(num_heads * seq_len * v_dim) * sizeof(float);
    VkDeviceSize sz_g    = static_cast<VkDeviceSize>(num_heads * seq_len) * sizeof(float);
    VkDeviceSize sz_beta = sz_g;
    VkDeviceSize sz_out  = (static_cast<VkDeviceSize>(num_heads * total_seq * v_dim) +
                            static_cast<VkDeviceSize>(num_heads * k_dim * v_dim)) * sizeof(float);
    VkDeviceSize sz_init = static_cast<VkDeviceSize>(num_heads * k_dim * v_dim) * sizeof(float);

    auto q_buf     = host_buf(sz_q);
    auto k_buf     = host_buf(sz_k);
    auto v_buf     = host_buf(sz_v);
    auto g_buf     = host_buf(sz_g);
    auto beta_buf  = host_buf(sz_beta);
    auto out_buf   = host_buf(sz_out);
    auto init_buf  = host_buf(sz_init);

    // --- Prepare deterministic bounded data (shared CPU/GPU) ---
    // Deterministic xorshift32: returns uint32 in [0, 2^32-1]
    auto xorshift32 = [](uint32_t& s) -> uint32_t {
      s ^= s << 13;
      s ^= s >> 17;
      s ^= s << 5;
      return s;
    };

    // Fill n floats via xorshift, scaled to [lo, hi], seeded per-array
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

    auto bounded = [](size_t n, float lo, float hi) -> std::vector<float> {
      std::vector<float> v(n);
      if (n == 0) return v;
      if (n == 1) { v[0] = lo; return v; }
      float step = (hi - lo) / static_cast<float>(n - 1);
      for (size_t i = 0; i < n; ++i)
        v[i] = lo + step * static_cast<float>(i);
      return v;
    };

    std::vector<float> query, key, value, g, beta;
    if (pc->pseudo_random) {
      query  = bounded_rand(num_heads * seq_len * k_dim, q_lo, q_hi, 0xDEADBEEFu);
      key    = bounded_rand(num_heads * seq_len * k_dim, k_lo, k_hi, 0xCAFEBABEu);
      value  = bounded_rand(num_heads * seq_len * v_dim, v_lo, v_hi, 0xDECAFBADu);
      g      = bounded_rand(num_heads * seq_len, g_lo, g_hi, 0xABADCAFEu);
      beta   = bounded_rand(num_heads * seq_len, beta_lo, beta_hi, 0xBEEFCACEu);
      if (pc->l2_normalize_qk) {
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
            normalize(query,  off);
            normalize(key, off);
          }
        }
      }
    } else if (pc->repeat_per_head) {
      auto head_q = bounded(static_cast<size_t>(seq_len) * k_dim, q_lo, q_hi);
      auto head_k = bounded(static_cast<size_t>(seq_len) * k_dim, k_lo, k_hi);
      auto head_v = bounded(static_cast<size_t>(seq_len) * v_dim, v_lo, v_hi);
      auto head_g = bounded(static_cast<size_t>(seq_len), g_lo, g_hi);
      auto head_beta = bounded(static_cast<size_t>(seq_len), beta_lo, beta_hi);
      std::size_t q_stride  = static_cast<std::size_t>(seq_len) * k_dim;
      std::size_t k_stride  = q_stride;
      std::size_t v_stride  = static_cast<std::size_t>(seq_len) * v_dim;
      std::size_t g_stride  = static_cast<std::size_t>(seq_len);
      std::size_t beta_stride = g_stride;
      query .assign(num_heads * q_stride, 0.0f);
      key   .assign(num_heads * k_stride, 0.0f);
      value .assign(num_heads * v_stride, 0.0f);
      g     .assign(num_heads * g_stride, 0.0f);
      beta  .assign(num_heads * beta_stride, 0.0f);
      for (uint32_t h = 0; h < num_heads; ++h) {
        std::memcpy(&query [h * q_stride],  head_q.data(),   q_stride * sizeof(float));
        std::memcpy(&key   [h * k_stride],  head_k.data(),   k_stride * sizeof(float));
        std::memcpy(&value [h * v_stride],  head_v.data(),   v_stride * sizeof(float));
        std::memcpy(&g     [h * g_stride],  head_g.data(),   g_stride * sizeof(float));
        std::memcpy(&beta  [h * beta_stride], head_beta.data(), beta_stride * sizeof(float));
      }
    } else {
      query  = bounded(num_heads * seq_len * k_dim, q_lo, q_hi);
      key    = bounded(num_heads * seq_len * k_dim, k_lo, k_hi);
      value  = bounded(num_heads * seq_len * v_dim, v_lo, v_hi);
      g      = bounded(num_heads * seq_len, g_lo, g_hi);
      beta   = bounded(num_heads * seq_len, beta_lo, beta_hi);
    }
    std::vector<float> initial_state(num_heads * k_dim * v_dim, 0.0f);

    // --- Upload to mapped buffers ---
    std::memcpy(q_buf.mapped,    query.data(),         static_cast<size_t>(sz_q));
    std::memcpy(k_buf.mapped,    key.data(),           static_cast<size_t>(sz_k));
    std::memcpy(v_buf.mapped,    value.data(),         static_cast<size_t>(sz_v));
    std::memcpy(g_buf.mapped,    g.data(),             static_cast<size_t>(sz_g));
    std::memcpy(beta_buf.mapped, beta.data(),          static_cast<size_t>(sz_beta));
    std::memcpy(init_buf.mapped, initial_state.data(), static_cast<size_t>(sz_init));

    // --- Allocate descriptor set from 7-binding layout ---
    VkDescriptorSet desc_set = dev.allocate_descriptor_set(desc_layout);

    // --- Update bindings 0..6 ---
    dev.update_descriptor_set(desc_set, 0, q_buf);
    dev.update_descriptor_set(desc_set, 1, k_buf);
    dev.update_descriptor_set(desc_set, 2, v_buf);
    dev.update_descriptor_set(desc_set, 3, g_buf);
    dev.update_descriptor_set(desc_set, 4, beta_buf);
    dev.update_descriptor_set(desc_set, 5, out_buf);
    dev.update_descriptor_set(desc_set, 6, init_buf);

    // --- Push constants: 10 uint32 fields matching shader order ---
    float inv_sqrt_kdim = 1.0f / std::sqrt(static_cast<float>(k_dim));
    uint32_t q_scale_bits;
    std::memcpy(&q_scale_bits, &inv_sqrt_kdim, sizeof(q_scale_bits));

    PushConsts push_consts = {num_heads, seq_len, k_dim, v_dim, chunk_size,
                              q_scale_bits, total_seq, chunk_count,
                              pc->l2_normalize_qk ? 1u : 0u, 0};

    // --- Record command buffer: bind, push constants, dispatch, submit ---
    // Single dispatch for 1 head; serialized per-head dispatches for >1 head
    if (num_heads == 1) {
      VkCommandBuffer cmd = dev.allocate_command_buffer();
      dev.begin_command_buffer(cmd);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              pipe_layout, 0, 1, &desc_set, 0, nullptr);
      push_consts.base_head = 0;
      vkCmdPushConstants(cmd, pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                         0, sizeof(PushConsts), &push_consts);
      vkCmdDispatch(cmd, 1, 1, 1);
      dev.end_command_buffer(cmd);
      dev.submit_and_wait(cmd);
    } else if (pc->separate_head_submits) {
      // One command buffer per head, submitted and waited individually
      for (uint32_t h = 0; h < num_heads; ++h) {
        VkCommandBuffer cmd = dev.allocate_command_buffer();
        dev.begin_command_buffer(cmd);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipe_layout, 0, 1, &desc_set, 0, nullptr);
        push_consts.base_head = h;
        vkCmdPushConstants(cmd, pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(PushConsts), &push_consts);
        vkCmdDispatch(cmd, 1, 1, 1);
        dev.end_command_buffer(cmd);
        dev.submit_and_wait(cmd);
      }
    } else {
      const VkMemoryBarrier serial_head_barrier = {
          .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
          .pNext         = nullptr,
          .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
          .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
      };
      VkCommandBuffer cmd = dev.allocate_command_buffer();
      dev.begin_command_buffer(cmd);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              pipe_layout, 0, 1, &desc_set, 0, nullptr);
      for (uint32_t h = 0; h < num_heads; ++h) {
        push_consts.base_head = h;
        vkCmdPushConstants(cmd, pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(PushConsts), &push_consts);
        vkCmdDispatch(cmd, 1, 1, 1);
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            1, &serial_head_barrier,
            0, nullptr,
            0, nullptr);
      }
      dev.end_command_buffer(cmd);
      dev.submit_and_wait(cmd);
    }

    // --- Read output from mapped memory ---
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

      // Core attention comparison is strided: GPU layout is [head][total_seq][v_dim],
      // CPU layout is [head][seq_len][v_dim]. Compare only valid tokens (t < seq_len).
      {
        max_abs_core = 0.0f;
        max_rel_core = 0.0f;
        for (uint32_t h = 0; h < num_heads; ++h) {
          for (uint32_t t = 0; t < seq_len; ++t) {
            for (uint32_t vd = 0; vd < v_dim; ++vd) {
              const std::size_t gpu_idx = (static_cast<std::size_t>(h) * total_seq + t) * v_dim + vd;
              const std::size_t cpu_idx = (static_cast<std::size_t>(h) * seq_len + t) * v_dim + vd;
              const float g = out_data[gpu_idx];
              const float c = cpu_out.core_attn_out[cpu_idx];
              const float ab = std::abs(g - c);
              if (ab > max_abs_core) max_abs_core = ab;
              const float denom = std::max(1.0f, std::abs(c));
              const float rel = ab / denom;
              if (rel > max_rel_core) max_rel_core = rel;
            }
          }
        }
      }

      auto max_abs_and_rel = [](const float* gpu, const std::vector<float>& cpu,
                                 std::size_t count, float& out_abs, float& out_rel) {
        out_abs = 0.0f;
        out_rel = 0.0f;
        for (std::size_t i = 0; i < count; ++i) {
          const float g = gpu[i];
          const float c = cpu[i];
          const float ab = std::abs(g - c);
          if (ab > out_abs) out_abs = ab;
          const float denom = std::max(1.0f, std::abs(c));
          const float rel = ab / denom;
          if (rel > out_rel) out_rel = rel;
        }
      };

      max_abs_and_rel(out_data + state_offset, cpu_out.final_state, state_count,
                       max_abs_state, max_rel_state);

      compare_ok = (nan_count == 0) &&
                   (max_rel_core <= 1e-4f) &&
                   (max_rel_state <= 1e-4f);

      // Per-head diagnostics for multi-head cases
      if (num_heads > 1) {
        const std::size_t gpu_head_stride = static_cast<std::size_t>(total_seq) * v_dim;
        const std::size_t cpu_head_stride = static_cast<std::size_t>(seq_len) * v_dim;
        const std::size_t core_per_head_valid = static_cast<std::size_t>(seq_len) * v_dim;
        const std::size_t state_per_head = static_cast<std::size_t>(k_dim) * v_dim;
        head_core_abs.assign(num_heads, 0.0f);
        head_core_rel.assign(num_heads, 0.0f);
        head_state_abs.assign(num_heads, 0.0f);
        head_state_rel.assign(num_heads, 0.0f);

        for (uint32_t h = 0; h < num_heads; ++h) {
          // Core attention per-head: GPU is [head][total_seq][v_dim], CPU is [head][seq_len][v_dim].
          // Compare only the first seq_len*total_seq valid tokens within each head's GPU region.
          {
            float ha = 0.0f, hr = 0.0f;
            const float* gpu_core = out_data + h * gpu_head_stride;
            const float* cpu_core = cpu_out.core_attn_out.data() + h * cpu_head_stride;
            for (std::size_t i = 0; i < core_per_head_valid; ++i) {
              const float g = gpu_core[i];
              const float c = cpu_core[i];
              const float ab = std::abs(g - c);
              if (ab > ha) ha = ab;
              const float denom = std::max(1.0f, std::abs(c));
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
              const float g = gpu_state[i];
              const float c = cpu_state[i];
              const float ab = std::abs(g - c);
              if (ab > ha) ha = ab;
              const float denom = std::max(1.0f, std::abs(c));
              const float r = ab / denom;
              if (r > hr) hr = r;
            }
            head_state_abs[h] = ha;
            head_state_rel[h] = hr;
          }
        }
      }
    }

    // Find worst heads by relative error
    if (num_heads > 1) {
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
    std::cout << "  \"shader\": \"deltanet_chunk_prefill.comp\",\n";
    std::cout << "  \"case\": \"" << case_name << "\",\n";
    std::cout << "  \"total_seq\": " << total_seq << ",\n";
    std::cout << "  \"chunk_count\": " << chunk_count << ",\n";
    std::cout << "  \"k_dim\": " << k_dim << ",\n";
    std::cout << "  \"v_dim\": " << v_dim << ",\n";
    std::cout << "  \"seq_len\": " << seq_len << ",\n";
    std::cout << "  \"chunk_size\": " << chunk_size << ",\n";
    std::cout << "  \"head_dispatch_mode\": \""
              << (num_heads == 1 ? "single" : (pc->separate_head_submits ? "per-submit" : "serial")) << "\",\n";
    std::cout << "  \"serial_head_barrier\": "
              << (num_heads > 1 && !pc->separate_head_submits ? "true" : "false") << ",\n";
    std::cout << "  \"separate_head_submits\": "
              << (pc->separate_head_submits ? "true" : "false") << ",\n";
    std::cout << "  \"repeat_per_head\": "
              << (pc->repeat_per_head ? "true" : "false") << ",\n";
    std::cout << "  \"pseudo_random\": "
              << (pc->pseudo_random ? "true" : "false") << ",\n";
    std::cout << "  \"l2_normalize_qk\": "
              << (pc->l2_normalize_qk ? "true" : "false") << ",\n";
    std::cout << "  \"status\": \""
              << (compare_ok ? "compare-ok" : "compare-fail") << "\",\n";
    std::cout << "  \"output_l2\": " << l2 << ",\n";
    std::cout << "  \"nan_count\": " << nan_count << ",\n";
    std::cout << "  \"max_abs\": " << max_abs << ",\n";
    std::cout << "  \"max_abs_core\": " << max_abs_core << ",\n";
    std::cout << "  \"max_abs_state\": " << max_abs_state << ",\n";
    std::cout << "  \"max_rel_core\": " << max_rel_core << ",\n";
    std::cout << "  \"max_rel_state\": " << max_rel_state << ",\n";

    if (num_heads > 1) {
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
      std::cout << "  \"worst_state_head\": " << worst_state_head << ",\n";
    }

    std::cout << "  \"cpu_chunk_bridge_production\": true\n";
    std::cout << "}\n";

    return 0;
  } catch (const std::exception& e) {
    std::cout << "{\n";
    std::cout << "  \"shader\": \"deltanet_chunk_prefill.comp\",\n";
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
  std::cout << "  \"shader\": \"deltanet_chunk_prefill.comp\",\n";
    std::cout << "  \"case\": \"" << case_name << "\",\n";
    std::cout << "  \"total_seq\": " << total_seq << ",\n";
    std::cout << "  \"chunk_count\": " << chunk_count << ",\n";
  std::cout << "  \"status\": \"error\",\n";
  std::cout << "  \"message\": \"Vulkan not available (SPOCK_VULKAN_STUB)\"\n";
  std::cout << "}\n";
  return 1;
#endif
}

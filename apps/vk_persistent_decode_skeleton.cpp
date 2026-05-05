#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include "runtime/vk_device.hpp"
#include "runtime/weight_loader.hpp"

// ---- fp16 helpers (IEEE 754 binary16, no ARM/NEON intrinsics) ----

namespace {

static inline uint16_t fp32_to_fp16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  uint32_t sign = (u >> 16u) & 0x8000u;
  int32_t exp = static_cast<int32_t>((u >> 23u) & 0xFFu) - 127 + 15;
  uint32_t mant = (u & 0x007FFFFFu) >> 13u;
  if (exp <= 0) {
    // Flush denormals to zero for determinism.
    return static_cast<uint16_t>(sign);
  }
  if (exp >= 31) {
    // Clamp to max representable fp16 (no inf/nan for simplicity).
    return static_cast<uint16_t>(sign | 0x7BFFu);
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10u) | mant);
}

static inline float fp16_to_fp32(uint16_t h) {
  uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16u;
  uint32_t exp = (h >> 10u) & 0x1Fu;
  uint32_t mant = h & 0x3FFu;
  if (exp == 0) {
    if (mant != 0) {
      const float value = std::ldexp(static_cast<float>(mant), -24);
      return (h & 0x8000u) ? -value : value;
    }
    float f;
    uint32_t u = sign;
    std::memcpy(&f, &u, sizeof(f));
    return f;
  }
  uint32_t f_exp = exp + 127 - 15;
  uint32_t u = sign | (f_exp << 23u) | (mant << 13u);
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

// Deterministic input: small exact fp16 values cycling 1..8.
// Keeping the dot products small makes fp32 reduction order deterministic for
// the tested model-width geometries.
static inline uint16_t input_vec_val(uint32_t c) {
  return fp32_to_fp16(static_cast<float>((c % 8u) + 1u));
}

// Deterministic weight: small exact fp16 values cycling by group and column.
// Get weight value at (group g, column c) from either real or synthetic data.
static inline uint16_t weight_mat_val(uint32_t g, uint32_t c, uint32_t hidden,
                                       const std::vector<uint16_t>& real_wm) {
  if (!real_wm.empty()) {
    return real_wm[g * hidden + c];
  }
  const uint32_t group_scale = (g % 8u) + 1u;
  const uint32_t col_scale = (c % 8u) + 1u;
  return fp32_to_fp16(static_cast<float>(group_scale * col_scale));
}

// Shader-mirrored dot product for group g:
//   64 lanes, lane l accumulates columns l, l+64, l+128, ... in fp32.
//   Then tree reduction: stride 32,16,8,4,2,1.
// This must match the GPU reduction order exactly for bitwise checksum agreement.
static inline float shader_mirrored_dot(uint32_t g, uint32_t hidden,
                                         const std::vector<uint16_t>& real_wm) {
  // Per-lane fp32 partial sums (up to 64 lanes).
  float lane_sums[64] = {};
  for (uint32_t lane = 0; lane < 64; ++lane) {
    float acc = 0.0f;
    for (uint32_t c = lane; c < hidden; c += 64) {
      float iv = fp16_to_fp32(input_vec_val(c));
      float wv = fp16_to_fp32(weight_mat_val(g, c, hidden, real_wm));
      acc += iv * wv;
    }
    lane_sums[lane] = acc;
  }
  // Tree reduction (stride 32 -> 1).
  for (uint32_t stride = 32; stride >= 1; stride >>= 1) {
    for (uint32_t lane = 0; lane < stride; ++lane) {
      lane_sums[lane] += lane_sums[lane + stride];
    }
  }
  return lane_sums[0];
}

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

  auto spv = try_load("build/shaders/persistent_decode_skeleton.comp.spv");
  if (!spv.empty()) return spv;

#ifdef SHADER_DIR
  spv = try_load(std::string(SHADER_DIR) + "/persistent_decode_skeleton.comp.spv");
  if (!spv.empty()) return spv;
#endif

  throw std::runtime_error(
      "cannot open shader: persistent_decode_skeleton.comp.spv "
      "(tried build/shaders/ and SHADER_DIR)");
}

struct RepeatResult {
  bool ok = false;
  bool timestamp_valid = false;
  std::uint32_t arrived = 0;
  std::uint32_t generation = 0;
  std::uint32_t failures = 0;
  std::uint32_t checksum = 0;
  std::uint32_t trace_mismatches = 0;
  double gpu_dispatch_us = 0.0;
};

// Per-role data for multi-role mode.
struct RoleWeightData {
  std::string role;
  std::uint32_t rows = 0;
  std::uint32_t cols = 0;
  std::vector<uint16_t> wm;  // fp16 row-major: [workgroups * hidden]
};

// Emit a JSON error object and return exit code 2.
int json_error(const std::string& message) {
  std::cout << "{\n";
  std::cout << "  \"status\": \"error\",\n";
  std::cout << "  \"message\": \"" << message << "\"\n";
  std::cout << "}\n";
  return 2;
}

// Compute expected checksum, expected trace, and expected generation for
// a given (workgroups, iterations, hidden, real_wm) combination.
struct ExpectedResult {
  std::uint32_t expected_checksum;
  std::vector<std::uint32_t> expected_trace;
  std::uint32_t expected_generation;
};

ExpectedResult compute_expected(std::uint32_t workgroups,
                                std::uint32_t iterations,
                                std::uint32_t hidden,
                                const std::vector<uint16_t>& real_wm) {
  // Pre-compute dot bits per group using shader-mirrored reduction.
  std::vector<std::uint32_t> dot_bits(workgroups);
  for (std::uint32_t g = 0; g < workgroups; ++g) {
    float d = shader_mirrored_dot(g, hidden, real_wm);
    std::uint32_t bits;
    std::memcpy(&bits, &d, sizeof(bits));
    dot_bits[g] = bits;
  }

  std::uint64_t sum_g = 0;
  std::uint64_t sum_dot = 0;
  for (std::uint32_t g = 0; g < workgroups; ++g) {
    sum_g += (g + 1);
    sum_dot += dot_bits[g];
  }

  std::uint64_t sum_i = static_cast<std::uint64_t>(iterations) * (iterations + 1) / 2;
  std::uint32_t expected_checksum = static_cast<std::uint32_t>(
      static_cast<std::uint64_t>(workgroups) * (sum_i * sum_g + sum_dot * iterations));

  std::vector<std::uint32_t> expected_trace(iterations);
  for (std::uint32_t iter = 0; iter < iterations; ++iter) {
    expected_trace[iter] = static_cast<std::uint32_t>(
        static_cast<std::uint64_t>(iter + 1) * sum_g + sum_dot);
  }

  std::uint32_t expected_generation = iterations * 2u;

  return {expected_checksum, std::move(expected_trace), expected_generation};
}

}  // namespace

int main(int argc, char** argv) {
  std::uint32_t tokens = 2;
  std::uint32_t layers = 4;
  std::uint32_t hidden = 128;
  std::uint32_t workgroups = 8;
  std::uint32_t repeats = 1;
  bool do_timestamps = false;
  bool qwen35_preset = false;
  std::string repack_dir;
  std::vector<std::string> weight_roles;

  // Track explicit overrides for preset precedence.
  bool has_tokens = false;
  bool has_layers = false;
  bool has_hidden = false;
  bool has_workgroups = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--tokens" && i + 1 < argc) {
      tokens = std::stoul(argv[++i]);
      has_tokens = true;
    } else if (arg == "--layers" && i + 1 < argc) {
      layers = std::stoul(argv[++i]);
      has_layers = true;
    } else if (arg == "--hidden" && i + 1 < argc) {
      hidden = std::stoul(argv[++i]);
      has_hidden = true;
    } else if (arg == "--workgroups" && i + 1 < argc) {
      workgroups = std::stoul(argv[++i]);
      has_workgroups = true;
    } else if (arg == "--repeats" && i + 1 < argc) {
      repeats = std::stoul(argv[++i]);
    } else if (arg == "--timestamps") {
      do_timestamps = true;
    } else if (arg == "--qwen35-preset") {
      qwen35_preset = true;
    } else if (arg == "--repack-dir" && i + 1 < argc) {
      repack_dir = argv[++i];
    } else if (arg == "--weight-role" && i + 1 < argc) {
      weight_roles.push_back(argv[++i]);
    } else if (arg == "--help") {
      std::cout << "usage: vk_persistent_decode_skeleton [options]\n";
      std::cout << "  --tokens N        decode iterations = tokens * layers (default 2)\n";
      std::cout << "  --layers N        layer count (default 4)\n";
      std::cout << "  --hidden N        hidden / weight columns (default 128)\n";
      std::cout << "  --workgroups N    dispatch workgroup count (default 8)\n";
      std::cout << "  --repeats N       in-process repeated dispatches (default 1)\n";
      std::cout << "  --timestamps      record GPU timestamps around dispatch\n";
      std::cout << "  --qwen35-preset   preset: tokens=128, layers=24, hidden=1024, workgroups=82\n";
      std::cout << "                    Explicit --tokens/--layers/--hidden/--workgroups override.\n";
      std::cout << "  --repack-dir DIR  load real fp16 weights from repacked model artifact\n";
      std::cout << "  --weight-role ROLE tensor role to load (repeatable, e.g. layer.0.mlp_gate)\n";
      std::cout << "                    --repack-dir and --weight-role must both be specified together.\n";
      std::cout << "  --help            show this help\n";
      return 0;
    }
  }

  // Apply Qwen3.5 preset defaults (user flags take precedence).
  if (qwen35_preset) {
    if (!has_tokens)     { tokens = 128;     has_tokens = true; }
    if (!has_layers)     { layers = 24;      has_layers = true; }
    if (!has_hidden)     { hidden = 1024;    has_hidden = true; }
    if (!has_workgroups) { workgroups = 82;  has_workgroups = true; }
  }

  // Validate.
  if (tokens == 0 || layers == 0 || hidden == 0 || workgroups == 0) {
    return json_error("--tokens, --layers, --hidden, --workgroups must be > 0");
  }
  if (repeats == 0) {
    return json_error("--repeats must be > 0");
  }

  std::uint64_t product = static_cast<std::uint64_t>(tokens) * layers;
  if (product > std::numeric_limits<std::uint32_t>::max()) {
    return json_error("--tokens * --layers overflows uint32");
  }

  std::uint32_t iterations = static_cast<std::uint32_t>(product);

  // --- Real weight loading ---
  bool real_weight = false;
  bool multi_role = false;
  std::vector<uint16_t> real_wm;  // fp16 row-major: [workgroups * hidden] (single-role compat)
  std::uint32_t real_weight_rows = 0;
  std::uint32_t real_weight_cols = 0;
  std::vector<RoleWeightData> role_weights;

  if (!repack_dir.empty() && weight_roles.empty()) {
    return json_error("--repack-dir and --weight-role must both be specified");
  }

  if (!weight_roles.empty()) {
    if (repack_dir.empty()) {
      return json_error("--repack-dir is required when --weight-role is specified");
    }

    multi_role = (weight_roles.size() > 1);
    if (multi_role && repeats != 1) {
      return json_error("--repeats > 1 is not supported in multi-role mode yet");
    }
    if (multi_role && do_timestamps) {
      return json_error("--timestamps is not supported in multi-role mode yet");
    }
    real_weight = true;

    auto artifact = spock::runtime::WeightArtifact::load(repack_dir);

    for (std::size_t ri = 0; ri < weight_roles.size(); ++ri) {
      const auto& role = weight_roles[ri];
      const auto* info = artifact.find_by_role(role);
      if (!info) {
        return json_error("weight role not found: " + role);
      }
      if (info->dtype != "fp16") {
        return json_error("weight dtype must be fp16 for role '" + role +
                          "', got: " + info->dtype);
      }
      if (info->shape.size() != 2) {
        return json_error("weight must be rank-2 for role '" + role +
                          "', got rank: " + std::to_string(info->shape.size()));
      }

      std::uint32_t role_rows = static_cast<uint32_t>(info->shape[0]);
      std::uint32_t role_cols = static_cast<uint32_t>(info->shape[1]);

      if (workgroups > role_rows) {
        return json_error("workgroups (" + std::to_string(workgroups) +
                          ") > weight rows (" + std::to_string(role_rows) +
                          ") for role '" + role + "'");
      }

      // Infer hidden from first role's columns if not explicitly set.
      if (!has_hidden) {
        hidden = role_cols;
        has_hidden = true;
      }
      if (hidden > role_cols) {
        return json_error("hidden (" + std::to_string(hidden) +
                          ") > weight cols (" + std::to_string(role_cols) +
                          ") for role '" + role + "'");
      }

      // Read raw bytes and extract first workgroups rows x hidden cols.
      auto raw = spock::runtime::read_tensor_bytes(artifact, *info);
      RoleWeightData rwd;
      rwd.role = role;
      rwd.rows = role_rows;
      rwd.cols = role_cols;
      rwd.wm.resize(static_cast<std::size_t>(workgroups) * hidden);
      for (uint32_t g = 0; g < workgroups; ++g) {
        for (uint32_t c = 0; c < hidden; ++c) {
          const std::size_t src_index =
              (static_cast<std::size_t>(g) * role_cols + c) *
              sizeof(std::uint16_t);
          std::uint16_t value = 0;
          std::memcpy(&value, raw.data() + src_index, sizeof(value));
          rwd.wm[static_cast<std::size_t>(g) * hidden + c] = value;
        }
      }
      role_weights.push_back(std::move(rwd));
    }

    // For backward compatibility: when single role, populate old fields.
    if (weight_roles.size() == 1) {
      real_wm = std::move(role_weights[0].wm);
      real_weight_rows = role_weights[0].rows;
      real_weight_cols = role_weights[0].cols;
      real_weight = true;
    }
  }

  // Multi-role mode does not implement per-role repeats or timestamps.
  if (multi_role && repeats != 1) {
    return json_error("--repeats > 1 is not supported in multi-role mode");
  }
  if (multi_role && do_timestamps) {
    return json_error("--timestamps is not supported in multi-role mode");
  }


#if SPOCK_HAS_VULKAN && !defined(SPOCK_VULKAN_STUB)
  try {
    spock::runtime::VulkanDevice dev;
    dev.initialize();

    // --- Common GPU resources ---
    constexpr VkDeviceSize control_size = 4 * sizeof(std::uint32_t);
    const std::uint32_t zero_init[4] = {0, 0, 0, 0};

    // --- Descriptor set layout: 5 storage buffers ---
    int num_bindings = 5;
    std::vector<VkDescriptorSetLayoutBinding> bindings(num_bindings);
    for (int b = 0; b < num_bindings; ++b) {
      bindings[b].binding = b;
      bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings[b].descriptorCount = 1;
      bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      bindings[b].pImmutableSamplers = nullptr;
    }

    VkDescriptorSetLayout desc_layout =
        dev.create_descriptor_set_layout(bindings);

    // --- Pipeline layout: 16-byte push constants (4 x uint32) ---
    VkPipelineLayout pipe_layout =
        dev.create_pipeline_layout(desc_layout, 4 * sizeof(std::uint32_t));

    // --- Shader + pipeline ---
    auto spirv = read_spirv();
    VkShaderModule shader = dev.create_shader_module(spirv);
    VkPipeline pipeline = dev.create_compute_pipeline(shader, pipe_layout);

    struct PushConsts {
      std::uint32_t workgroup_count;
      std::uint32_t iteration_count;
      std::uint32_t hidden;
      std::uint32_t _pad;
    };

    // --- Timestamp query pool (optional, shared across roles) ---
    bool ts_valid = false;
    VkQueryPool ts_pool = VK_NULL_HANDLE;
    if (do_timestamps) {
      ts_valid = dev.capabilities().timestamp_valid;
      if (ts_valid) {
        ts_pool = dev.create_timestamp_query_pool(2);
      }
    }

    // ============================================================
    // Single-role or synthetic mode: backward-compatible path
    // ============================================================
    if (!multi_role) {
      ExpectedResult expected = compute_expected(
          workgroups, iterations, hidden, real_wm);

      // --- Control buffer ---
      auto control_buf = dev.create_device_local_buffer(control_size);
      dev.upload_to_device(control_buf, zero_init, control_size);

      // --- Trace buffer ---
      VkDeviceSize trace_count = static_cast<VkDeviceSize>(workgroups) * iterations;
      VkDeviceSize trace_size = trace_count * sizeof(std::uint32_t);
      auto trace_buf = dev.create_device_local_buffer(trace_size);
      std::vector<std::uint32_t> trace_zeros(trace_count, 0);
      dev.upload_to_device(trace_buf, trace_zeros.data(), trace_size);

      // --- Scratch buffer ---
      VkDeviceSize scratch_count = static_cast<VkDeviceSize>(workgroups);
      VkDeviceSize scratch_size = scratch_count * sizeof(std::uint32_t);
      auto scratch_buf = dev.create_device_local_buffer(scratch_size);
      std::vector<std::uint32_t> scratch_zeros(scratch_count, 0);
      dev.upload_to_device(scratch_buf, scratch_zeros.data(), scratch_size);

      // --- Input vector ---
      VkDeviceSize iv_count = hidden;
      VkDeviceSize iv_size = iv_count * sizeof(std::uint16_t);
      auto input_vec_buf = dev.create_device_local_buffer(iv_size);
      std::vector<std::uint16_t> iv_data(iv_count);
      for (std::uint32_t c = 0; c < hidden; ++c) {
        iv_data[c] = input_vec_val(c);
      }
      dev.upload_to_device(input_vec_buf, iv_data.data(), iv_size);

      // --- Weight matrix ---
      VkDeviceSize wm_count = static_cast<VkDeviceSize>(workgroups) * hidden;
      VkDeviceSize wm_size = wm_count * sizeof(std::uint16_t);
      auto weight_mat_buf = dev.create_device_local_buffer(wm_size);
      std::vector<std::uint16_t> wm_data(wm_count);
      if (real_weight) {
        std::memcpy(wm_data.data(), real_wm.data(), wm_size);
      } else {
        for (std::uint32_t g = 0; g < workgroups; ++g) {
          for (std::uint32_t c = 0; c < hidden; ++c) {
            wm_data[g * hidden + c] = weight_mat_val(g, c, hidden, real_wm);
          }
        }
      }
      dev.upload_to_device(weight_mat_buf, wm_data.data(), wm_size);

      // --- Descriptor set ---
      VkDescriptorSet desc_set = dev.allocate_descriptor_set(desc_layout);
      dev.update_descriptor_set(desc_set, 0, control_buf);
      dev.update_descriptor_set(desc_set, 1, trace_buf);
      dev.update_descriptor_set(desc_set, 2, scratch_buf);
      dev.update_descriptor_set(desc_set, 3, input_vec_buf);
      dev.update_descriptor_set(desc_set, 4, weight_mat_buf);

      PushConsts push{workgroups, iterations, hidden, 0};

      std::vector<RepeatResult> repeat_results;
      repeat_results.reserve(repeats);

      for (std::uint32_t repeat = 0; repeat < repeats; ++repeat) {
        dev.upload_to_device(control_buf, zero_init, control_size);
        dev.upload_to_device(trace_buf, trace_zeros.data(), trace_size);
        dev.upload_to_device(scratch_buf, scratch_zeros.data(), scratch_size);

        bool repeat_ts_valid = ts_valid;
        double repeat_gpu_dispatch_us = 0.0;

        VkCommandBuffer cmd = dev.allocate_command_buffer();
        dev.begin_command_buffer(cmd);

        if (do_timestamps && repeat_ts_valid) {
          dev.reset_query_pool(ts_pool, 0, 2);
          vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             ts_pool, 0);
        }

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipe_layout, 0, 1, &desc_set, 0, nullptr);
        vkCmdPushConstants(cmd, pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(PushConsts), &push);
        vkCmdDispatch(cmd, workgroups, 1, 1);

        if (do_timestamps && repeat_ts_valid) {
          vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             ts_pool, 1);
        }

        dev.end_command_buffer(cmd);
        dev.submit_and_wait(cmd);

        if (do_timestamps && repeat_ts_valid) {
          auto ts = dev.get_timestamp_results(ts_pool, 0, 2);
          if (ts.size() == 2 && ts[1] >= ts[0]) {
            float period_ns = dev.capabilities().timestamp_period;
            repeat_gpu_dispatch_us =
                static_cast<double>(ts[1] - ts[0]) * period_ns / 1000.0;
          } else {
            repeat_ts_valid = false;
          }
        }

        std::uint32_t control_out[4] = {};
        dev.download_from_device(control_buf, control_out, control_size);

        std::vector<std::uint32_t> trace_out(trace_count);
        dev.download_from_device(trace_buf, trace_out.data(), trace_size);

        RepeatResult result;
        result.arrived = control_out[0];
        result.generation = control_out[1];
        result.failures = control_out[2];
        result.checksum = control_out[3];
        result.timestamp_valid = repeat_ts_valid;
        result.gpu_dispatch_us = repeat_gpu_dispatch_us;

        for (std::uint32_t g = 0; g < workgroups; ++g) {
          for (std::uint32_t i = 0; i < iterations; ++i) {
            std::uint32_t actual = trace_out[g * iterations + i];
            if (actual != expected.expected_trace[i]) {
              ++result.trace_mismatches;
            }
          }
        }

        result.ok = (result.failures == 0) &&
                    (result.generation == expected.expected_generation) &&
                    (result.arrived == 0) &&
                    (result.checksum == expected.expected_checksum) &&
                    (result.trace_mismatches == 0);

        repeat_results.push_back(result);
      }

      RepeatResult first_result;
      if (!repeat_results.empty()) {
        first_result = repeat_results.front();
      }

      bool ok = true;
      std::uint64_t failures_total = 0;
      std::uint64_t trace_mismatches_total = 0;
      for (const auto& result : repeat_results) {
        ok = ok && result.ok;
        failures_total += result.failures;
        trace_mismatches_total += result.trace_mismatches;
      }

      // --- Cleanup ---
      if (do_timestamps && ts_pool != VK_NULL_HANDLE) {
        dev.destroy_query_pool(ts_pool);
      }
      dev.destroy_buffer(control_buf);
      dev.destroy_buffer(trace_buf);
      dev.destroy_buffer(scratch_buf);
      dev.destroy_buffer(input_vec_buf);
      dev.destroy_buffer(weight_mat_buf);
      dev.destroy_pipeline(pipeline);
      dev.destroy_shader_module(shader);
      dev.destroy_pipeline_layout(pipe_layout);
      dev.destroy_descriptor_set_layout(desc_layout);

      // --- JSON output (backward-compatible single-role / synthetic) ---
      std::cout << "{\n";
      std::cout << "  \"tokens\": " << tokens << ",\n";
      std::cout << "  \"layers\": " << layers << ",\n";
      std::cout << "  \"hidden\": " << hidden << ",\n";
      std::cout << "  \"workgroups\": " << workgroups << ",\n";
      std::cout << "  \"iterations\": " << iterations << ",\n";
      if (qwen35_preset) {
        std::cout << "  \"qwen35_preset\": \"active\",\n";
      }
      std::cout << "  \"real_weight\": " << (real_weight ? "true" : "false") << ",\n";
      if (real_weight) {
        std::cout << "  \"weight_role\": \"" << weight_roles[0] << "\",\n";
        std::cout << "  \"real_weight_rows\": " << real_weight_rows << ",\n";
        std::cout << "  \"real_weight_cols\": " << real_weight_cols << ",\n";
        std::cout << "  \"repack_dir\": \"" << repack_dir << "\",\n";
      }
      if (repeats > 1) {
        std::cout << "  \"repeats\": " << repeats << ",\n";
        std::cout << "  \"status\": \"" << (ok ? "ok" : "fail") << "\",\n";
        std::cout << "  \"failures\": " << failures_total << ",\n";
        std::cout << "  \"generation\": " << first_result.generation << ",\n";
        std::cout << "  \"expected_generation\": " << expected.expected_generation << ",\n";
        std::cout << "  \"checksum\": " << first_result.checksum << ",\n";
        std::cout << "  \"expected_checksum\": " << expected.expected_checksum << ",\n";
        std::cout << "  \"trace_mismatches\": " << trace_mismatches_total << ",\n";
        std::cout << "  \"repeat_results\": [\n";
        for (std::size_t i = 0; i < repeat_results.size(); ++i) {
          const auto& result = repeat_results[i];
          std::cout << "    {\n";
          std::cout << "      \"repeat\": " << i + 1 << ",\n";
          std::cout << "      \"status\": \"" << (result.ok ? "ok" : "fail") << "\",\n";
          std::cout << "      \"failures\": " << result.failures << ",\n";
          std::cout << "      \"generation\": " << result.generation << ",\n";
          std::cout << "      \"arrived\": " << result.arrived << ",\n";
          std::cout << "      \"checksum\": " << result.checksum << ",\n";
          std::cout << "      \"trace_mismatches\": " << result.trace_mismatches;
          if (do_timestamps) {
            std::cout << ",\n";
            std::cout << "      \"timestamp_valid\": "
                      << (result.timestamp_valid ? "true" : "false") << ",\n";
            if (result.timestamp_valid) {
              double per_barrier_us = result.gpu_dispatch_us /
                  static_cast<double>(expected.expected_generation);
              std::cout << "      \"gpu_dispatch_us\": " << result.gpu_dispatch_us << ",\n";
              std::cout << "      \"per_barrier_us\": " << per_barrier_us << ",\n";
              std::cout << "      \"barriers\": " << expected.expected_generation << "\n";
            } else {
              std::cout << "      \"gpu_dispatch_us\": null,\n";
              std::cout << "      \"per_barrier_us\": null,\n";
              std::cout << "      \"barriers\": " << expected.expected_generation << "\n";
            }
          } else {
            std::cout << "\n";
          }
          std::cout << "    }" << (i + 1 == repeat_results.size() ? "\n" : ",\n");
        }
        std::cout << "  ]\n";
        std::cout << "}\n";
        return ok ? 0 : 1;
      }

      std::cout << "  \"status\": \"" << (ok ? "ok" : "fail") << "\",\n";
      std::cout << "  \"failures\": " << first_result.failures << ",\n";
      std::cout << "  \"generation\": " << first_result.generation << ",\n";
      std::cout << "  \"expected_generation\": " << expected.expected_generation << ",\n";
      std::cout << "  \"checksum\": " << first_result.checksum << ",\n";
      std::cout << "  \"expected_checksum\": " << expected.expected_checksum << ",\n";
      std::cout << "  \"trace_mismatches\": " << first_result.trace_mismatches;
      if (do_timestamps) {
        std::cout << ",\n";
        std::cout << "  \"timestamp_valid\": "
                  << (first_result.timestamp_valid ? "true" : "false") << ",\n";
        if (first_result.timestamp_valid) {
          double per_barrier_us = first_result.gpu_dispatch_us /
              static_cast<double>(expected.expected_generation);
          std::cout << "  \"gpu_dispatch_us\": " << first_result.gpu_dispatch_us << ",\n";
          std::cout << "  \"per_barrier_us\": " << per_barrier_us << ",\n";
          std::cout << "  \"barriers\": " << expected.expected_generation << "\n";
        } else {
          std::cout << "  \"gpu_dispatch_us\": null,\n";
          std::cout << "  \"per_barrier_us\": null,\n";
          std::cout << "  \"barriers\": " << expected.expected_generation << "\n";
        }
      } else {
        std::cout << "\n";
      }
      std::cout << "}\n";

      return ok ? 0 : 1;
    }

    // ============================================================
    // Multi-role mode: per-role dispatch loop
    // ============================================================
    struct RoleResult {
      std::string role;
      std::uint32_t rows = 0;
      std::uint32_t cols = 0;
      std::uint32_t checksum = 0;
      std::uint32_t expected_checksum = 0;
      std::uint32_t trace_mismatches = 0;
      std::uint32_t failures = 0;
      std::string status;
    };

    std::vector<RoleResult> role_results;

    // Pre-allocate reusable buffers for the dispatch loop.
    auto control_buf = dev.create_device_local_buffer(control_size);

    VkDeviceSize trace_count = static_cast<VkDeviceSize>(workgroups) * iterations;
    VkDeviceSize trace_size = trace_count * sizeof(std::uint32_t);
    auto trace_buf = dev.create_device_local_buffer(trace_size);
    std::vector<std::uint32_t> trace_zeros(trace_count, 0);

    VkDeviceSize scratch_count = static_cast<VkDeviceSize>(workgroups);
    VkDeviceSize scratch_size = scratch_count * sizeof(std::uint32_t);
    auto scratch_buf = dev.create_device_local_buffer(scratch_size);
    std::vector<std::uint32_t> scratch_zeros(scratch_count, 0);

    VkDeviceSize iv_count = hidden;
    VkDeviceSize iv_size = iv_count * sizeof(std::uint16_t);
    auto input_vec_buf = dev.create_device_local_buffer(iv_size);
    std::vector<std::uint16_t> iv_data(iv_count);
    for (std::uint32_t c = 0; c < hidden; ++c) {
      iv_data[c] = input_vec_val(c);
    }
    dev.upload_to_device(input_vec_buf, iv_data.data(), iv_size);

    VkDeviceSize wm_count = static_cast<VkDeviceSize>(workgroups) * hidden;
    VkDeviceSize wm_size = wm_count * sizeof(std::uint16_t);
    auto weight_mat_buf = dev.create_device_local_buffer(wm_size);

    VkDescriptorSet desc_set = dev.allocate_descriptor_set(desc_layout);
    dev.update_descriptor_set(desc_set, 0, control_buf);
    dev.update_descriptor_set(desc_set, 1, trace_buf);
    dev.update_descriptor_set(desc_set, 2, scratch_buf);
    dev.update_descriptor_set(desc_set, 3, input_vec_buf);
    dev.update_descriptor_set(desc_set, 4, weight_mat_buf);

    PushConsts push{workgroups, iterations, hidden, 0};

    for (const auto& rwd : role_weights) {
      ExpectedResult expected = compute_expected(
          workgroups, iterations, hidden, rwd.wm);

      // Upload weight data for this role.
      dev.upload_to_device(weight_mat_buf, rwd.wm.data(), wm_size);

      // Reset control, trace, scratch.
      dev.upload_to_device(control_buf, zero_init, control_size);
      dev.upload_to_device(trace_buf, trace_zeros.data(), trace_size);
      dev.upload_to_device(scratch_buf, scratch_zeros.data(), scratch_size);

      VkCommandBuffer cmd = dev.allocate_command_buffer();
      dev.begin_command_buffer(cmd);

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              pipe_layout, 0, 1, &desc_set, 0, nullptr);
      vkCmdPushConstants(cmd, pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                         0, sizeof(PushConsts), &push);
      vkCmdDispatch(cmd, workgroups, 1, 1);

      dev.end_command_buffer(cmd);
      dev.submit_and_wait(cmd);

      // Read back results.
      std::uint32_t control_out[4] = {};
      dev.download_from_device(control_buf, control_out, control_size);

      std::vector<std::uint32_t> trace_out(trace_count);
      dev.download_from_device(trace_buf, trace_out.data(), trace_size);

      RoleResult rr;
      rr.role = rwd.role;
      rr.rows = rwd.rows;
      rr.cols = rwd.cols;
      rr.checksum = control_out[3];
      rr.expected_checksum = expected.expected_checksum;
      rr.failures = control_out[2];

      for (std::uint32_t g = 0; g < workgroups; ++g) {
        for (std::uint32_t i = 0; i < iterations; ++i) {
          std::uint32_t actual = trace_out[g * iterations + i];
          if (actual != expected.expected_trace[i]) {
            ++rr.trace_mismatches;
          }
        }
      }

      rr.status = (rr.failures == 0 &&
                   control_out[1] == expected.expected_generation &&
                   control_out[0] == 0 &&
                   rr.checksum == rr.expected_checksum &&
                   rr.trace_mismatches == 0) ? "ok" : "fail";

      role_results.push_back(std::move(rr));
    }

    // --- Cleanup ---
    if (do_timestamps && ts_pool != VK_NULL_HANDLE) {
      dev.destroy_query_pool(ts_pool);
    }
    dev.destroy_buffer(control_buf);
    dev.destroy_buffer(trace_buf);
    dev.destroy_buffer(scratch_buf);
    dev.destroy_buffer(input_vec_buf);
    dev.destroy_buffer(weight_mat_buf);
    dev.destroy_pipeline(pipeline);
    dev.destroy_shader_module(shader);
    dev.destroy_pipeline_layout(pipe_layout);
    dev.destroy_descriptor_set_layout(desc_layout);

    // --- Multi-role JSON output ---
    bool all_ok = true;
    for (const auto& rr : role_results) {
      if (rr.status != "ok") all_ok = false;
    }

    std::cout << "{\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"layers\": " << layers << ",\n";
    std::cout << "  \"hidden\": " << hidden << ",\n";
    std::cout << "  \"workgroups\": " << workgroups << ",\n";
    std::cout << "  \"iterations\": " << iterations << ",\n";
    std::cout << "  \"real_weight\": true,\n";
    std::cout << "  \"repack_dir\": \"" << repack_dir << "\",\n";
    std::cout << "  \"multi_role\": true,\n";
    std::cout << "  \"role_count\": " << role_weights.size() << ",\n";
    std::cout << "  \"status\": \"" << (all_ok ? "ok" : "fail") << "\",\n";
    std::cout << "  \"roles\": [\n";
    for (std::size_t i = 0; i < role_results.size(); ++i) {
      const auto& rr = role_results[i];
      std::cout << "    {\n";
      std::cout << "      \"role\": \"" << rr.role << "\",\n";
      std::cout << "      \"rows\": " << rr.rows << ",\n";
      std::cout << "      \"cols\": " << rr.cols << ",\n";
      std::cout << "      \"checksum\": " << rr.checksum << ",\n";
      std::cout << "      \"expected_checksum\": " << rr.expected_checksum << ",\n";
      std::cout << "      \"trace_mismatches\": " << rr.trace_mismatches << ",\n";
      std::cout << "      \"failures\": " << rr.failures << ",\n";
      std::cout << "      \"status\": \"" << rr.status << "\"\n";
      std::cout << "    }";
      if (i + 1 < role_results.size()) {
        std::cout << ",";
      }
      std::cout << "\n";
    }
    std::cout << "  ]\n";
    std::cout << "}\n";

    return all_ok ? 0 : 1;

  } catch (const std::exception& e) {
    std::cout << "{\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"layers\": " << layers << ",\n";
    std::cout << "  \"hidden\": " << hidden << ",\n";
    std::cout << "  \"workgroups\": " << workgroups << ",\n";
    std::cout << "  \"iterations\": " << iterations << ",\n";
    std::cout << "  \"status\": \"error\",\n";
    std::cout << "  \"message\": \"" << e.what() << "\"\n";
    std::cout << "}\n";
    return 2;
  }
#else
  std::cout << "{\n";
  std::cout << "  \"tokens\": " << tokens << ",\n";
  std::cout << "  \"layers\": " << layers << ",\n";
  std::cout << "  \"hidden\": " << hidden << ",\n";
  std::cout << "  \"workgroups\": " << workgroups << ",\n";
  std::cout << "  \"iterations\": " << iterations << ",\n";
  std::cout << "  \"status\": \"error\",\n";
  std::cout << "  \"message\": \"Vulkan not available (SPOCK_VULKAN_STUB)\"\n";
  std::cout << "}\n";
  return 2;
#endif
}

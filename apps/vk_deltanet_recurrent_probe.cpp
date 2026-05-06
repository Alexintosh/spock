#include <cstdint>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include "runtime/vk_device.hpp"

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

  auto spv = try_load("build/shaders/deltanet_recurrent.comp.spv");
  if (!spv.empty()) return spv;

#ifdef SHADER_DIR
  spv = try_load(std::string(SHADER_DIR) + "/deltanet_recurrent.comp.spv");
  if (!spv.empty()) return spv;
#endif

  throw std::runtime_error(
      "cannot open shader: deltanet_recurrent.comp.spv "
      "(tried build/shaders/ and SHADER_DIR)");
}

int json_error(const std::string& message) {
  std::cout << "{\n";
  std::cout << "  \"status\": \"error\",\n";
  std::cout << "  \"message\": \"" << message << "\"\n";
  std::cout << "}\n";
  return 2;
}

std::string parse_u32_option(const std::string& opt,
                             const std::string& value,
                             std::uint32_t* out) {
  if (value.empty()) {
    return opt + " must be a nonnegative integer, got empty string";
  }
  if (value[0] < '0' || value[0] > '9') {
    return opt + " must be a nonnegative integer, got: " + value;
  }
  std::size_t pos = 0;
  try {
    unsigned long val = std::stoul(value, &pos, 10);
    if (val > UINT32_MAX) {
      return opt + " value too large: " + value;
    }
    *out = static_cast<std::uint32_t>(val);
  } catch (...) {
    return opt + " must be a nonnegative integer, got: " + value;
  }
  if (pos != value.size()) {
    return opt + " must be a nonnegative integer, got: " + value;
  }
  return {};
}

std::vector<std::uint16_t> load_fp16_file(const std::string& path,
                                          std::uint32_t length,
                                          const std::string& opt) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    throw std::runtime_error("cannot open " + opt + ": " + path);
  }
  auto file_size = f.tellg();
  f.seekg(0);
  auto required = static_cast<std::streamsize>(length) * sizeof(std::uint16_t);
  if (file_size < required) {
    throw std::runtime_error(opt + " too small: need " + std::to_string(length) +
                             " x 2 = " + std::to_string(required) +
                             " bytes, got " + std::to_string(file_size));
  }
  std::vector<std::uint16_t> data(length);
  f.read(reinterpret_cast<char*>(data.data()), required);
  return data;
}

std::vector<std::uint32_t> load_u32_file(const std::string& path,
                                         std::uint32_t length,
                                         const std::string& opt) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    throw std::runtime_error("cannot open " + opt + ": " + path);
  }
  auto file_size = f.tellg();
  f.seekg(0);
  auto required = static_cast<std::streamsize>(length) * sizeof(std::uint32_t);
  if (file_size < required) {
    throw std::runtime_error(opt + " too small: need " + std::to_string(length) +
                             " x 4 = " + std::to_string(required) +
                             " bytes, got " + std::to_string(file_size));
  }
  std::vector<std::uint32_t> data(length);
  f.read(reinterpret_cast<char*>(data.data()), required);
  return data;
}

std::vector<float> load_fp32_file(const std::string& path,
                                  std::uint32_t length,
                                  const std::string& opt) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    throw std::runtime_error("cannot open " + opt + ": " + path);
  }
  auto file_size = f.tellg();
  f.seekg(0);
  auto required = static_cast<std::streamsize>(length) * sizeof(float);
  if (file_size < required) {
    throw std::runtime_error(opt + " too small: need " + std::to_string(length) +
                             " x 4 = " + std::to_string(required) +
                             " bytes, got " + std::to_string(file_size));
  }
  std::vector<float> data(length);
  f.read(reinterpret_cast<char*>(data.data()), required);
  return data;
}

struct PushConstants {
  std::uint32_t num_heads;
  std::uint32_t k_dim;
  std::uint32_t v_dim;
  std::uint32_t state_total;
  std::uint32_t q_scale_bits;
};

std::uint32_t fp16_ulp_diff(std::uint16_t a, std::uint16_t b) {
  // Both are 16-bit patterns; compute bit diff for same-sign values.
  if (a == b) return 0;
  bool sign_a = (a >> 15) & 1;
  bool sign_b = (b >> 15) & 1;
  if (sign_a != sign_b) return 0xFFFF;  // opposite sign = max
  std::uint16_t abs_a = a & 0x7FFF;
  std::uint16_t abs_b = b & 0x7FFF;
  return static_cast<std::uint32_t>(abs_a > abs_b ? abs_a - abs_b : abs_b - abs_a);
}

}  // namespace

int main(int argc, char** argv) {
  std::uint32_t num_heads = 16;
  std::uint32_t k_dim = 128;
  std::uint32_t v_dim = 128;
  std::string q_fp16_file;
  std::string k_fp16_file;
  std::string v_fp16_file;
  std::string g_beta_bits_file;
  std::string state_pre_f32_file;
  std::string expected_output_fp16_file;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--num-heads" && i + 1 < argc) {
      std::string err = parse_u32_option("--num-heads", argv[++i], &num_heads);
      if (!err.empty()) return json_error(err);
    } else if (arg == "--k-dim" && i + 1 < argc) {
      std::string err = parse_u32_option("--k-dim", argv[++i], &k_dim);
      if (!err.empty()) return json_error(err);
    } else if (arg == "--v-dim" && i + 1 < argc) {
      std::string err = parse_u32_option("--v-dim", argv[++i], &v_dim);
      if (!err.empty()) return json_error(err);
    } else if (arg == "--q-fp16-file" && i + 1 < argc) {
      q_fp16_file = argv[++i];
    } else if (arg == "--k-fp16-file" && i + 1 < argc) {
      k_fp16_file = argv[++i];
    } else if (arg == "--v-fp16-file" && i + 1 < argc) {
      v_fp16_file = argv[++i];
    } else if (arg == "--g-beta-bits-file" && i + 1 < argc) {
      g_beta_bits_file = argv[++i];
    } else if (arg == "--state-pre-f32-file" && i + 1 < argc) {
      state_pre_f32_file = argv[++i];
    } else if (arg == "--expected-output-fp16-file" && i + 1 < argc) {
      expected_output_fp16_file = argv[++i];
    } else if (arg == "--help") {
      std::cout << "usage: vk_deltanet_recurrent_probe [options]\n";
      std::cout << "  --q-fp16-file PATH              query vectors fp16 (num_heads * k_dim values)\n";
      std::cout << "  --k-fp16-file PATH              key vectors fp16 (num_heads * k_dim values)\n";
      std::cout << "  --v-fp16-file PATH              value vectors fp16 (num_heads * v_dim values)\n";
      std::cout << "  --g-beta-bits-file PATH         g/beta fp32 bits u32 (num_heads*2 values, g then beta)\n";
      std::cout << "  --state-pre-f32-file PATH       pre-recurrent state fp32 (num_heads*k_dim*v_dim + num_heads*2 values)\n";
      std::cout << "  --expected-output-fp16-file PATH expected recurrent core output fp16 (num_heads * v_dim values)\n";
      std::cout << "  --num-heads N                   number of heads (default 16)\n";
      std::cout << "  --k-dim N                       key dimension per head (default 128)\n";
      std::cout << "  --v-dim N                       value dimension per head (default 128)\n";
      std::cout << "  --help                          show this help\n";
      return 0;
    }
  }

  if (q_fp16_file.empty()) return json_error("--q-fp16-file is required");
  if (k_fp16_file.empty()) return json_error("--k-fp16-file is required");
  if (v_fp16_file.empty()) return json_error("--v-fp16-file is required");
  if (g_beta_bits_file.empty()) return json_error("--g-beta-bits-file is required");
  if (state_pre_f32_file.empty()) return json_error("--state-pre-f32-file is required");
  if (expected_output_fp16_file.empty()) return json_error("--expected-output-fp16-file is required");

  if (num_heads == 0) return json_error("--num-heads must be nonzero");
  if (k_dim == 0) return json_error("--k-dim must be nonzero");
  if (v_dim == 0) return json_error("--v-dim must be nonzero");
  if (k_dim != 128 || v_dim != 128)
    return json_error("--k-dim and --v-dim must be 128 (shader local_size_x and shared arrays are 128)");
  std::uint32_t q_total = num_heads * k_dim;
  std::uint32_t kv_total = num_heads * k_dim;   // K section
  std::uint32_t v_total = num_heads * v_dim;     // V section
  std::uint32_t state_matrix = num_heads * k_dim * v_dim;
  std::uint32_t state_total = state_matrix;       // offset for g/beta tail
  std::uint32_t state_with_tail = state_matrix + num_heads * 2;

  std::vector<std::uint16_t> q_data;
  std::vector<std::uint16_t> k_data;
  std::vector<std::uint16_t> v_data;
  std::vector<std::uint32_t> g_beta_bits;
  std::vector<float> state_data;
  std::vector<std::uint16_t> expected_output;
  try {
    q_data = load_fp16_file(q_fp16_file, q_total, "--q-fp16-file");
    k_data = load_fp16_file(k_fp16_file, q_total, "--k-fp16-file");
    v_data = load_fp16_file(v_fp16_file, v_total, "--v-fp16-file");
    g_beta_bits = load_u32_file(g_beta_bits_file, num_heads * 2, "--g-beta-bits-file");
    state_data = load_fp32_file(state_pre_f32_file, state_with_tail, "--state-pre-f32-file");
    expected_output = load_fp16_file(expected_output_fp16_file, v_total, "--expected-output-fp16-file");
  } catch (const std::exception& e) {
    return json_error(e.what());
  }

  // Validate/overwrite state tail from g/beta bits file.
  // The pre-state fixture already includes the tail; overwrite to make the
  // contract explicit. Layout: g[0..num_heads-1] then beta[0..num_heads-1].
  // Use memcpy to write fp32 bit patterns into state_data without violating
  // strict-aliasing rules (float* vs uint32_t* reinterpret is undefined).
  for (std::uint32_t h = 0; h < num_heads; ++h) {
    float gf;
    std::memcpy(&gf, &g_beta_bits[h], sizeof(gf));
    state_data[state_matrix + h] = gf;  // g

    float bf;
    std::memcpy(&bf, &g_beta_bits[num_heads + h], sizeof(bf));
    state_data[state_matrix + num_heads + h] = bf;  // beta
  }

  // Compute q_scale_bits = float bits for 1/sqrt(k_dim)
  float q_scale = 1.0f / std::sqrt(static_cast<float>(k_dim));
  std::uint32_t q_scale_bits_val = 0;
  std::memcpy(&q_scale_bits_val, &q_scale, sizeof(q_scale_bits_val));

#if SPOCK_HAS_VULKAN && !defined(SPOCK_VULKAN_STUB)
  try {
    spock::runtime::VulkanDevice dev;
    dev.initialize();

    // Descriptor layout: 3 storage buffer bindings
    std::vector<VkDescriptorSetLayoutBinding> bindings(3);
    for (std::uint32_t b = 0; b < bindings.size(); ++b) {
      bindings[b].binding = b;
      bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings[b].descriptorCount = 1;
      bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      bindings[b].pImmutableSamplers = nullptr;
    }

    VkDescriptorSetLayout desc_layout = dev.create_descriptor_set_layout(bindings);
    VkPipelineLayout pipe_layout =
        dev.create_pipeline_layout(desc_layout, sizeof(PushConstants));
    auto spirv = read_spirv();
    VkShaderModule shader = dev.create_shader_module(spirv);
    VkPipeline pipeline = dev.create_compute_pipeline(shader, pipe_layout);

    // Buffer sizes
    VkDeviceSize q_bytes = static_cast<VkDeviceSize>(q_total) * sizeof(std::uint16_t);
    // KV+out buffer: K section (num_heads * k_dim) + V section (num_heads * v_dim)
    VkDeviceSize kv_bytes = static_cast<VkDeviceSize>(kv_total + v_total) * sizeof(std::uint16_t);
    VkDeviceSize state_bytes = static_cast<VkDeviceSize>(state_with_tail) * sizeof(float);

    auto q_buf = dev.create_device_local_buffer(q_bytes);
    auto kv_buf = dev.create_device_local_buffer(kv_bytes);
    auto state_buf = dev.create_device_local_buffer(state_bytes);
    auto kv_staging = dev.create_host_visible_buffer(kv_bytes);

    // Upload Q
    dev.upload_to_device(q_buf, q_data.data(), q_bytes);

    // Build KV buffer: K section then V section
    std::vector<std::uint16_t> kv_data(kv_total + v_total);
    std::memcpy(kv_data.data(), k_data.data(), static_cast<std::size_t>(kv_total) * 2);
    std::memcpy(kv_data.data() + kv_total, v_data.data(), static_cast<std::size_t>(v_total) * 2);
    dev.upload_to_device(kv_buf, kv_data.data(), kv_bytes);

    // Upload state (matrix + overwritten g/beta tail)
    dev.upload_to_device(state_buf, state_data.data(), state_bytes);

    // Allocate descriptor set
    VkDescriptorSet desc_set = dev.allocate_descriptor_set(desc_layout);
    dev.update_descriptor_set(desc_set, 0, q_buf);
    dev.update_descriptor_set(desc_set, 1, kv_buf);
    dev.update_descriptor_set(desc_set, 2, state_buf);

    // Dispatch
    PushConstants pc{num_heads, k_dim, v_dim, state_total, q_scale_bits_val};
    VkCommandBuffer cmd = dev.allocate_command_buffer();
    dev.begin_command_buffer(cmd);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipe_layout, 0, 1, &desc_set, 0, nullptr);
    vkCmdPushConstants(cmd, pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(PushConstants), &pc);
    vkCmdDispatch(cmd, num_heads, 1, 1);

    // Barrier: shader write -> transfer read on kv_buf
    {
      VkBufferMemoryBarrier barrier{};
      barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
      barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.buffer = kv_buf.buffer;
      barrier.offset = 0;
      barrier.size = kv_bytes;
      vkCmdPipelineBarrier(cmd,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT,
                           0,
                           0, nullptr,
                           1, &barrier,
                           0, nullptr);
    }

    // Copy kv output to host-visible staging
    VkBufferCopy kv_copy{0, 0, kv_bytes};
    vkCmdCopyBuffer(cmd, kv_buf.buffer, kv_staging.buffer, 1, &kv_copy);

    dev.end_command_buffer(cmd);
    dev.submit_and_wait(cmd);

    // Download from staging — V/output section starts after K section
    std::vector<std::uint16_t> gpu_kv(kv_total + v_total);
    std::memcpy(gpu_kv.data(), kv_staging.mapped, static_cast<std::size_t>(kv_bytes));

    // Output is in the V section: gpu_kv[kv_total .. kv_total + v_total)
    std::uint32_t output_mismatches = 0;
    int first_mismatch_idx = -1;
    std::uint16_t first_mismatch_gpu = 0, first_mismatch_expected = 0;
    std::uint32_t max_ulp = 0;
    for (std::uint32_t i = 0; i < v_total; ++i) {
      std::uint16_t gpu_val = gpu_kv[kv_total + i];
      std::uint16_t exp_val = expected_output[i];
      if (gpu_val != exp_val) {
        ++output_mismatches;
        std::uint32_t ulp = fp16_ulp_diff(gpu_val, exp_val);
        if (ulp > max_ulp) max_ulp = ulp;
        if (first_mismatch_idx < 0) {
          first_mismatch_idx = static_cast<int>(i);
          first_mismatch_gpu = gpu_val;
          first_mismatch_expected = exp_val;
        }
      }
    }

    dev.destroy_buffer(q_buf);
    dev.destroy_buffer(kv_buf);
    dev.destroy_buffer(state_buf);
    dev.destroy_buffer(kv_staging);
    dev.destroy_pipeline(pipeline);
    dev.destroy_shader_module(shader);
    dev.destroy_pipeline_layout(pipe_layout);
    dev.destroy_descriptor_set_layout(desc_layout);

    const bool ok = output_mismatches == 0;
    std::cout << "{\n";
    std::cout << "  \"status\": \"" << (ok ? "ok" : "fail") << "\",\n";
    std::cout << "  \"num_heads\": " << num_heads << ",\n";
    std::cout << "  \"k_dim\": " << k_dim << ",\n";
    std::cout << "  \"v_dim\": " << v_dim << ",\n";
    std::cout << "  \"output_count\": " << v_total << ",\n";
    std::cout << "  \"output_mismatches\": " << output_mismatches << ",\n";
    std::cout << "  \"max_fp16_ulp_diff\": " << max_ulp;
    if (first_mismatch_idx >= 0) {
      std::cout << ",\n  \"first_mismatch_index\": " << first_mismatch_idx;
      std::cout << ",\n  \"first_mismatch_gpu\": " << first_mismatch_gpu;
      std::cout << ",\n  \"first_mismatch_expected\": " << first_mismatch_expected;
    }
    std::cout << "\n}\n";
    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    return json_error(std::string("vulkan failure: ") + e.what());
  }
#else
  return json_error("Vulkan disabled");
#endif
}

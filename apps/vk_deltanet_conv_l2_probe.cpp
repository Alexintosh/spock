#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include "runtime/vk_device.hpp"
#include "runtime/weight_loader.hpp"

namespace {

std::vector<std::uint32_t> read_spirv(const std::string& shader_name) {
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

  auto spv = try_load("build/shaders/" + shader_name + ".spv");
  if (!spv.empty()) return spv;

#ifdef SHADER_DIR
  spv = try_load(std::string(SHADER_DIR) + "/" + shader_name + ".spv");
  if (!spv.empty()) return spv;
#endif

  throw std::runtime_error(
      "cannot open shader: " + shader_name +
      ".comp.spv (tried build/shaders/ and SHADER_DIR)");
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

std::vector<std::uint16_t> load_conv_weights(
    const spock::runtime::WeightArtifact& artifact,
    std::uint32_t conv_dim,
    std::uint32_t kernel_size) {
  const auto* info = artifact.find_by_role("layer.0.delta_conv");
  if (!info) {
    throw std::runtime_error("weight role not found: layer.0.delta_conv");
  }
  if (info->dtype != "fp16") {
    throw std::runtime_error("layer.0.delta_conv must be fp16");
  }
  std::uint32_t expected = conv_dim * kernel_size;
  std::uint32_t actual = static_cast<std::uint32_t>(info->nbytes / 2);
  if (actual < expected) {
    throw std::runtime_error(
        "layer.0.delta_conv too small: need " + std::to_string(expected) +
        " fp16 values, got " + std::to_string(actual));
  }
  auto raw = spock::runtime::read_tensor_bytes(artifact, *info);
  std::vector<std::uint16_t> weights(expected);
  std::memcpy(weights.data(), raw.data(), static_cast<std::size_t>(expected) * 2);
  return weights;
}

struct ConvPushConstants {
  std::uint32_t conv_dim;
  std::uint32_t kernel_size;
};

struct L2PushConstants {
  std::uint32_t num_heads;
  std::uint32_t head_dim;
};

}  // namespace

int main(int argc, char** argv) {
  std::uint32_t conv_dim = 6144;
  std::uint32_t kernel_size = 4;
  std::uint32_t num_heads = 16;
  std::uint32_t head_dim = 128;
  std::string repack_dir;
  std::string raw_qkv_fp16_file;
  std::string conv_state_pre_fp16_file;
  std::string expected_q_fp16_file;
  std::string expected_k_fp16_file;
  std::string expected_v_fp16_file;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--conv-dim" && i + 1 < argc) {
      std::string err = parse_u32_option("--conv-dim", argv[++i], &conv_dim);
      if (!err.empty()) return json_error(err);
    } else if (arg == "--kernel-size" && i + 1 < argc) {
      std::string err = parse_u32_option("--kernel-size", argv[++i], &kernel_size);
      if (!err.empty()) return json_error(err);
    } else if (arg == "--num-heads" && i + 1 < argc) {
      std::string err = parse_u32_option("--num-heads", argv[++i], &num_heads);
      if (!err.empty()) return json_error(err);
    } else if (arg == "--head-dim" && i + 1 < argc) {
      std::string err = parse_u32_option("--head-dim", argv[++i], &head_dim);
      if (!err.empty()) return json_error(err);
    } else if (arg == "--repack-dir" && i + 1 < argc) {
      repack_dir = argv[++i];
    } else if (arg == "--raw-qkv-fp16-file" && i + 1 < argc) {
      raw_qkv_fp16_file = argv[++i];
    } else if (arg == "--conv-state-pre-fp16-file" && i + 1 < argc) {
      conv_state_pre_fp16_file = argv[++i];
    } else if (arg == "--expected-q-fp16-file" && i + 1 < argc) {
      expected_q_fp16_file = argv[++i];
    } else if (arg == "--expected-k-fp16-file" && i + 1 < argc) {
      expected_k_fp16_file = argv[++i];
    } else if (arg == "--expected-v-fp16-file" && i + 1 < argc) {
      expected_v_fp16_file = argv[++i];
    } else if (arg == "--help") {
      std::cout << "usage: vk_deltanet_conv_l2_probe [options]\n";
      std::cout << "  --repack-dir DIR                    load delta_conv weights from repacked artifact\n";
      std::cout << "  --raw-qkv-fp16-file PATH            raw fp16 qkv input (conv_dim values)\n";
      std::cout << "  --conv-state-pre-fp16-file PATH     pre-conv rolling state (conv_dim*kernel_size values)\n";
      std::cout << "  --expected-q-fp16-file PATH         expected post-conv+L2 query (num_heads*head_dim values)\n";
      std::cout << "  --expected-k-fp16-file PATH         expected post-conv+L2 key (num_heads*head_dim values)\n";
      std::cout << "  --expected-v-fp16-file PATH         expected post-conv value (num_heads*head_dim values)\n";
      std::cout << "  --conv-dim N                        total qkv dimension (default 6144)\n";
      std::cout << "  --kernel-size N                     conv1d kernel size (default 4)\n";
      std::cout << "  --num-heads N                       number of heads (default 16)\n";
      std::cout << "  --head-dim N                        per-head dimension (default 128)\n";
      std::cout << "  --help                              show this help\n";
      return 0;
    }
  }

  if (repack_dir.empty()) return json_error("--repack-dir is required");
  if (raw_qkv_fp16_file.empty()) return json_error("--raw-qkv-fp16-file is required");
  if (conv_state_pre_fp16_file.empty()) return json_error("--conv-state-pre-fp16-file is required");
  if (expected_q_fp16_file.empty()) return json_error("--expected-q-fp16-file is required");
  if (expected_k_fp16_file.empty()) return json_error("--expected-k-fp16-file is required");
  if (expected_v_fp16_file.empty()) return json_error("--expected-v-fp16-file is required");

  std::uint32_t key_total = num_heads * head_dim;
  std::uint32_t val_total = num_heads * head_dim;
  // conv_dim = key_total * 2 + val_total, but accept override
  (void)val_total;

  std::vector<std::uint16_t> raw_qkv;
  std::vector<std::uint16_t> conv_state_pre;
  std::vector<std::uint16_t> expected_q;
  std::vector<std::uint16_t> expected_k;
  std::vector<std::uint16_t> expected_v;
  std::vector<std::uint16_t> conv_weights;
  try {
    raw_qkv = load_fp16_file(raw_qkv_fp16_file, conv_dim, "--raw-qkv-fp16-file");
    conv_state_pre = load_fp16_file(conv_state_pre_fp16_file,
                                    conv_dim * kernel_size,
                                    "--conv-state-pre-fp16-file");
    expected_q = load_fp16_file(expected_q_fp16_file, key_total, "--expected-q-fp16-file");
    expected_k = load_fp16_file(expected_k_fp16_file, key_total, "--expected-k-fp16-file");
    expected_v = load_fp16_file(expected_v_fp16_file, key_total, "--expected-v-fp16-file");
    auto artifact = spock::runtime::WeightArtifact::load(repack_dir);
    conv_weights = load_conv_weights(artifact, conv_dim, kernel_size);
  } catch (const std::exception& e) {
    return json_error(e.what());
  }

#if SPOCK_HAS_VULKAN && !defined(SPOCK_VULKAN_STUB)
  try {
    spock::runtime::VulkanDevice dev;
    dev.initialize();

    // --- Descriptor layout: 3 storage buffer bindings ---
    // Used by both conv1d_step.comp and l2_norm_per_head.comp.
    // conv1d: binding 0 = qkv (in/out), 1 = conv_state (in/out), 2 = weight (readonly)
    // l2_norm: binding 0 = input (readonly), 1 = output (writeonly), 2 = dummy
    std::vector<VkDescriptorSetLayoutBinding> bindings(3);
    for (std::uint32_t b = 0; b < bindings.size(); ++b) {
      bindings[b].binding = b;
      bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings[b].descriptorCount = 1;
      bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      bindings[b].pImmutableSamplers = nullptr;
    }

    VkDescriptorSetLayout desc_layout = dev.create_descriptor_set_layout(bindings);

    // --- Conv1d pipeline ---
    VkDeviceSize conv_push_size = sizeof(ConvPushConstants);
    VkPipelineLayout conv_pipe_layout =
        dev.create_pipeline_layout(desc_layout, conv_push_size);
    auto conv_spirv = read_spirv("conv1d_step.comp");
    VkShaderModule conv_shader = dev.create_shader_module(conv_spirv);
    VkPipeline conv_pipeline = dev.create_compute_pipeline(conv_shader, conv_pipe_layout);

    // --- L2-norm pipeline ---
    VkDeviceSize l2_push_size = sizeof(L2PushConstants);
    VkPipelineLayout l2_pipe_layout =
        dev.create_pipeline_layout(desc_layout, l2_push_size);
    auto l2_spirv = read_spirv("l2_norm_per_head.comp");
    VkShaderModule l2_shader = dev.create_shader_module(l2_spirv);
    VkPipeline l2_pipeline = dev.create_compute_pipeline(l2_shader, l2_pipe_layout);

    // --- Buffers ---
    VkDeviceSize qkv_bytes = static_cast<VkDeviceSize>(conv_dim) * sizeof(std::uint16_t);
    VkDeviceSize state_bytes =
        static_cast<VkDeviceSize>(conv_dim) * kernel_size * sizeof(std::uint16_t);
    VkDeviceSize weight_bytes =
        static_cast<VkDeviceSize>(conv_dim) * kernel_size * sizeof(std::uint16_t);
    // Dummy buffer for l2_norm binding 2 (unused but required by layout)
    VkDeviceSize key_bytes = static_cast<VkDeviceSize>(key_total) * sizeof(std::uint16_t);

    auto qkv_buf = dev.create_device_local_buffer(qkv_bytes);
    auto state_buf = dev.create_device_local_buffer(state_bytes);
    auto weight_buf = dev.create_device_local_buffer(weight_bytes);
    auto dummy_buf = dev.create_device_local_buffer(key_bytes);

    dev.upload_to_device(qkv_buf, raw_qkv.data(), qkv_bytes);
    dev.upload_to_device(state_buf, conv_state_pre.data(), state_bytes);
    dev.upload_to_device(weight_buf, conv_weights.data(), weight_bytes);

    // --- Conv1d dispatch ---
    VkDescriptorSet conv_desc = dev.allocate_descriptor_set(desc_layout);
    dev.update_descriptor_set(conv_desc, 0, qkv_buf);
    dev.update_descriptor_set(conv_desc, 1, state_buf);
    dev.update_descriptor_set(conv_desc, 2, weight_buf);

    auto qkv_staging = dev.create_host_visible_buffer(qkv_bytes);

    ConvPushConstants conv_pc{conv_dim, kernel_size};
    VkCommandBuffer cmd = dev.allocate_command_buffer();
    dev.begin_command_buffer(cmd);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, conv_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            conv_pipe_layout, 0, 1, &conv_desc, 0, nullptr);
    vkCmdPushConstants(cmd, conv_pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(ConvPushConstants), &conv_pc);
    vkCmdDispatch(cmd, 1, 1, 1);

    // --- Barrier: conv1d write → L2 Q read on qkv_buf ---
    {
      VkMemoryBarrier barrier{};
      barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      vkCmdPipelineBarrier(cmd,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           0,
                           1, &barrier,
                           0, nullptr,
                           0, nullptr);
    }

    // --- L2-norm Q (in-place on first key_total values of qkv) ---
    VkDescriptorSet l2q_desc = dev.allocate_descriptor_set(desc_layout);
    dev.update_descriptor_set(l2q_desc, 0, qkv_buf, 0, key_bytes);
    dev.update_descriptor_set(l2q_desc, 1, qkv_buf, 0, key_bytes);
    dev.update_descriptor_set(l2q_desc, 2, dummy_buf);

    L2PushConstants l2q_pc{num_heads, head_dim};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, l2_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            l2_pipe_layout, 0, 1, &l2q_desc, 0, nullptr);
    vkCmdPushConstants(cmd, l2_pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(L2PushConstants), &l2q_pc);
    vkCmdDispatch(cmd, num_heads, 1, 1);

    // --- Barrier: L2 Q write → L2 K read on qkv_buf ---
    {
      VkMemoryBarrier barrier{};
      barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      vkCmdPipelineBarrier(cmd,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           0,
                           1, &barrier,
                           0, nullptr,
                           0, nullptr);
    }

    // --- L2-norm K (in-place on second key_total values of qkv) ---
    VkDeviceSize k_offset = static_cast<VkDeviceSize>(key_total) * sizeof(std::uint16_t);
    VkDescriptorSet l2k_desc = dev.allocate_descriptor_set(desc_layout);
    dev.update_descriptor_set(l2k_desc, 0, qkv_buf, k_offset, key_bytes);
    dev.update_descriptor_set(l2k_desc, 1, qkv_buf, k_offset, key_bytes);
    dev.update_descriptor_set(l2k_desc, 2, dummy_buf);

    L2PushConstants l2k_pc{num_heads, head_dim};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, l2_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            l2_pipe_layout, 0, 1, &l2k_desc, 0, nullptr);
    vkCmdPushConstants(cmd, l2_pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(L2PushConstants), &l2k_pc);
    vkCmdDispatch(cmd, num_heads, 1, 1);


    // --- Barrier: L2 K write → transfer read on qkv_buf ---
    {
      VkBufferMemoryBarrier barrier{};
      barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
      barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.buffer = qkv_buf.buffer;
      barrier.offset = 0;
      barrier.size = qkv_bytes;
      vkCmdPipelineBarrier(cmd,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT,
                           0,
                           0, nullptr,
                           1, &barrier,
                           0, nullptr);
    }


    // --- Copy qkv to host-visible staging in same command buffer ---
    VkBufferCopy qkv_copy{0, 0, qkv_bytes};
    vkCmdCopyBuffer(cmd, qkv_buf.buffer, qkv_staging.buffer, 1, &qkv_copy);

    dev.end_command_buffer(cmd);
    dev.submit_and_wait(cmd);

    // --- Download from staging ---
    std::vector<std::uint16_t> gpu_qkv(conv_dim);
    std::memcpy(gpu_qkv.data(), qkv_staging.mapped, static_cast<std::size_t>(qkv_bytes));

    // Q slice: [0, key_total)
    std::uint32_t q_mismatches = 0;
    int first_q_mismatch = -1;
    std::uint16_t first_q_gpu = 0, first_q_expected = 0;
    for (std::uint32_t i = 0; i < key_total; ++i) {
      if (gpu_qkv[i] != expected_q[i]) {
        ++q_mismatches;
        if (first_q_mismatch < 0) {
          first_q_mismatch = static_cast<int>(i);
          first_q_gpu = gpu_qkv[i];
          first_q_expected = expected_q[i];
        }
      }
    }

    // K slice: [key_total, 2*key_total)
    std::uint32_t k_mismatches = 0;
    int first_k_mismatch = -1;
    std::uint16_t first_k_gpu = 0, first_k_expected = 0;
    for (std::uint32_t i = 0; i < key_total; ++i) {
      if (gpu_qkv[key_total + i] != expected_k[i]) {
        ++k_mismatches;
        if (first_k_mismatch < 0) {
          first_k_mismatch = static_cast<int>(i);
          first_k_gpu = gpu_qkv[key_total + i];
          first_k_expected = expected_k[i];
        }
      }
    }

    // V slice: [2*key_total, 3*key_total)
    std::uint32_t v_mismatches = 0;
    int first_v_mismatch = -1;
    std::uint16_t first_v_gpu = 0, first_v_expected = 0;
    for (std::uint32_t i = 0; i < key_total; ++i) {
      if (gpu_qkv[2 * key_total + i] != expected_v[i]) {
        ++v_mismatches;
        if (first_v_mismatch < 0) {
          first_v_mismatch = static_cast<int>(i);
          first_v_gpu = gpu_qkv[2 * key_total + i];
          first_v_expected = expected_v[i];
        }
      }
    }

    dev.destroy_buffer(qkv_buf);
    dev.destroy_buffer(state_buf);
    dev.destroy_buffer(weight_buf);
    dev.destroy_buffer(dummy_buf);
    dev.destroy_buffer(qkv_staging);
    dev.destroy_pipeline(conv_pipeline);
    dev.destroy_pipeline(l2_pipeline);
    dev.destroy_shader_module(conv_shader);
    dev.destroy_shader_module(l2_shader);
    dev.destroy_pipeline_layout(conv_pipe_layout);
    dev.destroy_pipeline_layout(l2_pipe_layout);
    dev.destroy_descriptor_set_layout(desc_layout);

    const bool ok = (q_mismatches == 0 && k_mismatches == 0 && v_mismatches == 0);
    std::cout << "{\n";
    std::cout << "  \"status\": \"" << (ok ? "ok" : "fail") << "\",\n";
    std::cout << "  \"q_mismatches\": " << q_mismatches << ",\n";
    std::cout << "  \"k_mismatches\": " << k_mismatches << ",\n";
    std::cout << "  \"v_mismatches\": " << v_mismatches;
    if (first_q_mismatch >= 0) {
      std::cout << ",\n  \"first_q_mismatch_index\": " << first_q_mismatch;
      std::cout << ",\n  \"first_q_mismatch_gpu\": " << first_q_gpu;
      std::cout << ",\n  \"first_q_mismatch_expected\": " << first_q_expected;
    }
    if (first_k_mismatch >= 0) {
      std::cout << ",\n  \"first_k_mismatch_index\": " << first_k_mismatch;
      std::cout << ",\n  \"first_k_mismatch_gpu\": " << first_k_gpu;
      std::cout << ",\n  \"first_k_mismatch_expected\": " << first_k_expected;
    }
    if (first_v_mismatch >= 0) {
      std::cout << ",\n  \"first_v_mismatch_index\": " << first_v_mismatch;
      std::cout << ",\n  \"first_v_mismatch_gpu\": " << first_v_gpu;
      std::cout << ",\n  \"first_v_mismatch_expected\": " << first_v_expected;
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
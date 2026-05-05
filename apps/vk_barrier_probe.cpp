#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
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

  auto spv = try_load("build/shaders/persistent_barrier_probe.comp.spv");
  if (!spv.empty()) return spv;

#ifdef SHADER_DIR
  spv = try_load(std::string(SHADER_DIR) + "/persistent_barrier_probe.comp.spv");
  if (!spv.empty()) return spv;
#endif

  throw std::runtime_error(
      "cannot open shader: persistent_barrier_probe.comp.spv "
      "(tried build/shaders/ and SHADER_DIR)");
}

}  // namespace

int main(int argc, char** argv) {
  std::uint32_t iterations = 10000;
  std::uint32_t workgroups = 8;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--iterations" && i + 1 < argc) {
      iterations = std::stoul(argv[++i]);
    } else if (arg == "--workgroups" && i + 1 < argc) {
      workgroups = std::stoul(argv[++i]);
    } else if (arg == "--help") {
      std::cout << "usage: vk_barrier_probe [options]\n";
      std::cout << "  --iterations N   iterations per workgroup (default 10000)\n";
      std::cout << "  --workgroups N   dispatch workgroup count (default 8)\n";
      std::cout << "  --help           show this help\n";
      return 0;
    }
  }

#if SPOCK_HAS_VULKAN && !defined(SPOCK_VULKAN_STUB)
  try {
    spock::runtime::VulkanDevice dev;
    dev.initialize();

    // --- Buffers ---
    // Control: 4 x uint32 (arrived, generation, failures, checksum)
    constexpr VkDeviceSize control_size = 4 * sizeof(std::uint32_t);
    auto control_buf = dev.create_device_local_buffer(control_size);

    const std::uint32_t zero_init[4] = {0, 0, 0, 0};
    dev.upload_to_device(control_buf, zero_init, control_size);

    // Trace: workgroups * iterations x uint32
    VkDeviceSize trace_count = static_cast<VkDeviceSize>(workgroups) * iterations;
    VkDeviceSize trace_size = trace_count * sizeof(std::uint32_t);
    auto trace_buf = dev.create_device_local_buffer(trace_size);

    std::vector<std::uint32_t> trace_zeros(trace_count, 0);
    dev.upload_to_device(trace_buf, trace_zeros.data(), trace_size);

    // --- Descriptor set layout: 2 storage buffers ---
    VkDescriptorSetLayoutBinding bindings[2];
    for (int b = 0; b < 2; ++b) {
      bindings[b].binding = b;
      bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings[b].descriptorCount = 1;
      bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      bindings[b].pImmutableSamplers = nullptr;
    }

    VkDescriptorSetLayout desc_layout =
        dev.create_descriptor_set_layout({bindings[0], bindings[1]});

    // --- Pipeline layout: 8-byte push constants (2 x uint32) ---
    VkPipelineLayout pipe_layout =
        dev.create_pipeline_layout(desc_layout, 2 * sizeof(std::uint32_t));

    // --- Shader + pipeline ---
    auto spirv = read_spirv();
    VkShaderModule shader = dev.create_shader_module(spirv);
    VkPipeline pipeline = dev.create_compute_pipeline(shader, pipe_layout);

    // --- Descriptor set ---
    VkDescriptorSet desc_set = dev.allocate_descriptor_set(desc_layout);
    dev.update_descriptor_set(desc_set, 0, control_buf);
    dev.update_descriptor_set(desc_set, 1, trace_buf);

    // --- Record, submit, wait ---
    struct PushConsts {
      std::uint32_t workgroup_count;
      std::uint32_t iteration_count;
    };

    PushConsts push{workgroups, iterations};

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

    // --- Download results ---
    std::uint32_t control_out[4] = {};
    dev.download_from_device(control_buf, control_out, control_size);

    std::vector<std::uint32_t> trace_out(trace_count);
    dev.download_from_device(trace_buf, trace_out.data(), trace_size);

    std::uint32_t arrived    = control_out[0];
    std::uint32_t generation = control_out[1];
    std::uint32_t failures   = control_out[2];
    std::uint32_t checksum   = control_out[3];

    // Expected checksum: sum_{g=1..wg} g * sum_{i=1..it} i
    // Using uint64 to avoid overflow during computation, then truncate to uint32.
    std::uint64_t sum_g = (static_cast<std::uint64_t>(workgroups) *
                           (workgroups + 1)) / 2;
    std::uint64_t sum_i = (static_cast<std::uint64_t>(iterations) *
                           (iterations + 1)) / 2;
    std::uint32_t expected_checksum =
        static_cast<std::uint32_t>(sum_g * sum_i);

    // Validate trace: trace[g * iterations + i] should equal (g+1)*(i+1)
    std::uint32_t trace_mismatches = 0;
    for (std::uint32_t g = 0; g < workgroups; ++g) {
      for (std::uint32_t i = 0; i < iterations; ++i) {
        std::uint32_t expected_val = (g + 1) * (i + 1);
        std::uint32_t actual_val = trace_out[g * iterations + i];
        if (actual_val != expected_val) {
          ++trace_mismatches;
        }
      }
    }

    bool ok = (failures == 0) && (generation == iterations) &&
              (arrived == 0) && (checksum == expected_checksum) &&
              (trace_mismatches == 0);

    // --- Cleanup ---
    dev.destroy_buffer(control_buf);
    dev.destroy_buffer(trace_buf);
    dev.destroy_pipeline(pipeline);
    dev.destroy_shader_module(shader);
    dev.destroy_pipeline_layout(pipe_layout);
    dev.destroy_descriptor_set_layout(desc_layout);

    // --- JSON output ---
    std::cout << "{\n";
    std::cout << "  \"iterations\": " << iterations << ",\n";
    std::cout << "  \"workgroups\": " << workgroups << ",\n";
    std::cout << "  \"status\": \"" << (ok ? "ok" : "fail") << "\",\n";
    std::cout << "  \"failures\": " << failures << ",\n";
    std::cout << "  \"generation\": " << generation << ",\n";
    std::cout << "  \"arrived\": " << arrived << ",\n";
    std::cout << "  \"checksum\": " << checksum << ",\n";
    std::cout << "  \"expected_checksum\": " << expected_checksum << ",\n";
    std::cout << "  \"trace_mismatches\": " << trace_mismatches << "\n";
    std::cout << "}\n";

    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::cout << "{\n";
    std::cout << "  \"iterations\": " << iterations << ",\n";
    std::cout << "  \"workgroups\": " << workgroups << ",\n";
    std::cout << "  \"status\": \"error\",\n";
    std::cout << "  \"message\": \"" << e.what() << "\"\n";
    std::cout << "}\n";
    return 2;
  }
#else
  std::cout << "{\n";
  std::cout << "  \"iterations\": " << iterations << ",\n";
  std::cout << "  \"workgroups\": " << workgroups << ",\n";
  std::cout << "  \"status\": \"error\",\n";
  std::cout << "  \"message\": \"Vulkan not available (SPOCK_VULKAN_STUB)\"\n";
  std::cout << "}\n";
  return 2;
#endif
}

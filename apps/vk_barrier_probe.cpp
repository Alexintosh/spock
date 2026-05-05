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

std::uint32_t payload_lane(std::uint32_t group,
                           std::uint32_t lane,
                           std::uint32_t payload_iters) {
  std::uint32_t x = ((group + 1u) * 747796405u) ^
                    ((lane + 1u) * 277803737u);
  std::uint32_t acc = 0;
  for (std::uint32_t p = 0; p < payload_iters; ++p) {
    x = x * 1664525u + 1013904223u + p;
    acc += (x ^ (x >> 16u)) + ((lane + 1u) * (p + 1u));
  }
  return acc;
}

std::uint32_t payload_group(std::uint32_t group,
                            std::uint32_t payload_iters) {
  std::uint32_t sum = 0;
  for (std::uint32_t lane = 0; lane < 64; ++lane) {
    sum += payload_lane(group, lane, payload_iters);
  }
  return sum;
}


// Deterministic hash for payload input vector and weight matrix.
// input_vec[c] = input_vec_hash(c)
std::uint32_t input_vec_hash(std::uint32_t c) {
  std::uint32_t x = c * 2654435761u;
  x ^= x >> 16u;
  x *= 2246822519u;
  x ^= x >> 13u;
  return x;
}

// weight_mat[g * payload_cols + c] = weight_mat_hash(g, c)
std::uint32_t weight_mat_hash(std::uint32_t g, std::uint32_t c) {
  std::uint32_t x = ((g + 1u) * 2246822519u) ^ (c * 3266489917u);
  x ^= x >> 16u;
  x *= 668265263u;
  x ^= x >> 13u;
  return x;
}

// Compute the dot-like memory payload for a single group:
// sum_{c=0..payload_cols-1} input_vec[c] * weight_mat[g*payload_cols + c]
// All arithmetic is uint32 wraparound.
std::uint32_t payload_dot(std::uint32_t group, std::uint32_t payload_cols) {
  std::uint32_t acc = 0;
  for (std::uint32_t c = 0; c < payload_cols; ++c) {
    acc += input_vec_hash(c) * weight_mat_hash(group, c);
  }
  return acc;
}
}  // namespace

int main(int argc, char** argv) {
  std::uint32_t iterations = 10000;
  std::uint32_t workgroups = 8;
  std::uint32_t payload_iters = 0;
  std::uint32_t payload_cols = 0;
  bool do_timestamps = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--iterations" && i + 1 < argc) {
      iterations = std::stoul(argv[++i]);
    } else if (arg == "--workgroups" && i + 1 < argc) {
      workgroups = std::stoul(argv[++i]);
    } else if (arg == "--payload-iters" && i + 1 < argc) {
      payload_iters = std::stoul(argv[++i]);
    } else if (arg == "--payload-cols" && i + 1 < argc) {
      payload_cols = std::stoul(argv[++i]);
    } else if (arg == "--timestamps") {
      do_timestamps = true;
    } else if (arg == "--help") {
      std::cout << "usage: vk_barrier_probe [options]\n";
      std::cout << "  --iterations N   iterations per workgroup (default 10000)\n";
      std::cout << "  --workgroups N   dispatch workgroup count (default 8)\n";
      std::cout << "  --payload-iters N  per-lane deterministic ALU payload (default 0)\n";
      std::cout << "  --payload-cols N  per-lane deterministic memory-traffic payload (default 0)\n";
      std::cout << "  --timestamps     record GPU timestamps around dispatch\n";
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

    // Scratch: workgroups x uint32 (inter-group communication)
    VkDeviceSize scratch_count = static_cast<VkDeviceSize>(workgroups);
    VkDeviceSize scratch_size = scratch_count * sizeof(std::uint32_t);
    auto scratch_buf = dev.create_device_local_buffer(scratch_size);

    std::vector<std::uint32_t> scratch_zeros(scratch_count, 0);
    dev.upload_to_device(scratch_buf, scratch_zeros.data(), scratch_size);

    // --- Payload-cols buffers ---
    spock::runtime::VulkanDevice::Buffer input_vec_buf;
    spock::runtime::VulkanDevice::Buffer weight_mat_buf;
    const std::uint32_t payload_cols_alloc = payload_cols == 0 ? 1 : payload_cols;

    VkDeviceSize iv_count = payload_cols_alloc;
    VkDeviceSize iv_size = iv_count * sizeof(std::uint32_t);
    input_vec_buf = dev.create_device_local_buffer(iv_size);
    std::vector<std::uint32_t> iv_data(iv_count);
    for (std::uint32_t c = 0; c < payload_cols_alloc; ++c) {
      iv_data[c] = payload_cols == 0 ? 0u : input_vec_hash(c);
    }
    dev.upload_to_device(input_vec_buf, iv_data.data(), iv_size);

    VkDeviceSize wm_count = payload_cols == 0
        ? 1
        : static_cast<VkDeviceSize>(workgroups) * payload_cols;
    VkDeviceSize wm_size = wm_count * sizeof(std::uint32_t);
    weight_mat_buf = dev.create_device_local_buffer(wm_size);
    std::vector<std::uint32_t> wm_data(wm_count);
    if (payload_cols != 0) {
      for (std::uint32_t g = 0; g < workgroups; ++g) {
        for (std::uint32_t c = 0; c < payload_cols; ++c) {
          wm_data[g * payload_cols + c] = weight_mat_hash(g, c);
        }
      }
    }
    dev.upload_to_device(weight_mat_buf, wm_data.data(), wm_size);

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

    // --- Descriptor set ---
    VkDescriptorSet desc_set = dev.allocate_descriptor_set(desc_layout);
    dev.update_descriptor_set(desc_set, 0, control_buf);
    dev.update_descriptor_set(desc_set, 1, trace_buf);
    dev.update_descriptor_set(desc_set, 2, scratch_buf);
    dev.update_descriptor_set(desc_set, 3, input_vec_buf);
    dev.update_descriptor_set(desc_set, 4, weight_mat_buf);

    // --- Timestamp query pool (optional) ---
    bool ts_valid = false;
    double gpu_dispatch_us = 0.0;
    VkQueryPool ts_pool = VK_NULL_HANDLE;
    if (do_timestamps) {
      ts_valid = dev.capabilities().timestamp_valid;
      if (ts_valid) {
        ts_pool = dev.create_timestamp_query_pool(2);
      }
    }

    // --- Record, submit, wait ---
    struct PushConsts {
      std::uint32_t workgroup_count;
      std::uint32_t iteration_count;
      std::uint32_t payload_iters;
      std::uint32_t payload_cols;
    };

    PushConsts push{workgroups, iterations, payload_iters, payload_cols};

    VkCommandBuffer cmd = dev.allocate_command_buffer();
    dev.begin_command_buffer(cmd);

    if (do_timestamps && ts_valid) {
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

    if (do_timestamps && ts_valid) {
      vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         ts_pool, 1);
    }

    dev.end_command_buffer(cmd);
    dev.submit_and_wait(cmd);

    // --- Retrieve timestamp results ---
    if (do_timestamps && ts_valid) {
      auto ts = dev.get_timestamp_results(ts_pool, 0, 2);
      if (ts.size() == 2 && ts[1] >= ts[0]) {
        float period_ns = dev.capabilities().timestamp_period;
        gpu_dispatch_us =
            static_cast<double>(ts[1] - ts[0]) * period_ns / 1000.0;
      } else {
        ts_valid = false;
      }
    }

    // --- Download results ---
    std::uint32_t control_out[4] = {};
    dev.download_from_device(control_buf, control_out, control_size);

    std::vector<std::uint32_t> trace_out(trace_count);
    dev.download_from_device(trace_buf, trace_out.data(), trace_size);

    std::uint32_t arrived    = control_out[0];
    std::uint32_t generation = control_out[1];
    std::uint32_t failures   = control_out[2];
    std::uint32_t checksum   = control_out[3];

    // Two-stage shader: 2 global barriers per iteration.
    std::uint32_t expected_generation = iterations * 2u;

    // sum_{k=1..workgroups} k = workgroups*(workgroups+1)/2
    std::uint64_t sum_g = (static_cast<std::uint64_t>(workgroups) *
                           (workgroups + 1)) / 2;
    // sum_{i=1..iterations} i = iterations*(iterations+1)/2
    std::uint64_t sum_i = (static_cast<std::uint64_t>(iterations) *
                           (iterations + 1)) / 2;

    std::uint32_t payload_total = 0;
    if (payload_iters != 0) {
      for (std::uint32_t g = 0; g < workgroups; ++g) {
        payload_total += payload_group(g, payload_iters);
      }
    }
    if (payload_cols != 0) {
      for (std::uint32_t g = 0; g < workgroups; ++g) {
        payload_total += payload_dot(g, payload_cols);
      }
    }

    // Each trace cell equals:
    //   (i+1) * sum_{k=1..workgroups} k + sum(payload(group))
    // Same value for every group. All arithmetic is uint32 wraparound.
    std::uint32_t trace_mismatches = 0;
    for (std::uint32_t g = 0; g < workgroups; ++g) {
      for (std::uint32_t i = 0; i < iterations; ++i) {
        std::uint32_t expected_val = static_cast<std::uint32_t>(
            sum_g * (i + 1) + payload_total);
        std::uint32_t actual_val = trace_out[g * iterations + i];
        if (actual_val != expected_val) {
          ++trace_mismatches;
        }
      }
    }

    // Expected checksum:
    //   workgroups * (sum_g * sum_i + payload_total * iterations) mod uint32
    // Each group accumulates that per-iteration trace sum in local_checksum,
    // then atomicAdds into global checksum.
    std::uint32_t expected_checksum =
        static_cast<std::uint32_t>(
            static_cast<std::uint64_t>(workgroups) *
            (sum_g * sum_i +
             static_cast<std::uint64_t>(payload_total) * iterations));

    bool ok = (failures == 0) && (generation == expected_generation) &&
              (arrived == 0) && (checksum == expected_checksum) &&
              (trace_mismatches == 0);

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

    // --- JSON output ---
    std::cout << "{\n";
    std::cout << "  \"iterations\": " << iterations << ",\n";
    std::cout << "  \"workgroups\": " << workgroups << ",\n";
    if (payload_iters != 0) {
      std::cout << "  \"payload_iters\": " << payload_iters << ",\n";
    }
    if (payload_cols != 0) {
      std::cout << "  \"payload_cols\": " << payload_cols << ",\n";
    }
    std::cout << "  \"status\": \"" << (ok ? "ok" : "fail") << "\",\n";
    std::cout << "  \"failures\": " << failures << ",\n";
    std::cout << "  \"generation\": " << generation << ",\n";
    std::cout << "  \"expected_generation\": " << expected_generation << ",\n";
    std::cout << "  \"arrived\": " << arrived << ",\n";
    std::cout << "  \"checksum\": " << checksum << ",\n";
    std::cout << "  \"expected_checksum\": " << expected_checksum << ",\n";
    std::cout << "  \"trace_mismatches\": " << trace_mismatches;
    if (do_timestamps) {
      std::cout << ",\n";
      std::cout << "  \"timestamp_valid\": " << (ts_valid ? "true" : "false") << ",\n";
      if (ts_valid) {
        double per_barrier_us = gpu_dispatch_us /
            static_cast<double>(expected_generation);
        std::cout << "  \"gpu_dispatch_us\": " << gpu_dispatch_us << ",\n";
        std::cout << "  \"per_barrier_us\": " << per_barrier_us << ",\n";
        std::cout << "  \"barriers\": " << expected_generation << "\n";
      } else {
        std::cout << "  \"gpu_dispatch_us\": null,\n";
        std::cout << "  \"per_barrier_us\": null,\n";
        std::cout << "  \"barriers\": " << expected_generation << "\n";
      }
    } else {
      std::cout << "\n";
    }
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

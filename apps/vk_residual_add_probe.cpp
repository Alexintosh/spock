#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
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

  auto spv = try_load("build/shaders/residual_add.comp.spv");
  if (!spv.empty()) return spv;

#ifdef SHADER_DIR
  spv = try_load(std::string(SHADER_DIR) + "/residual_add.comp.spv");
  if (!spv.empty()) return spv;
#endif

  throw std::runtime_error(
      "cannot open shader: residual_add.comp.spv "
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

bool is_negative_fp16(std::uint16_t value) {
  return (value & 0x8000u) != 0u && (value & 0x7fffu) != 0u;
}

std::uint32_t fp16_ulp_diff(std::uint16_t a, std::uint16_t b) {
  if (a == b) return 0;
  if (is_negative_fp16(a) != is_negative_fp16(b)) {
    return UINT32_MAX;
  }
  return (a > b) ? (a - b) : (b - a);
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

}  // namespace

int main(int argc, char** argv) {
  std::uint32_t length = 1024;
  std::uint32_t output_fp16_ulp_tolerance = 0;
  std::string input_a_fp16_file;
  std::string input_b_fp16_file;
  std::string expected_output_fp16_file;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--length" && i + 1 < argc) {
      std::string err = parse_u32_option("--length", argv[++i], &length);
      if (!err.empty()) return json_error(err);
    } else if (arg == "--input-a-fp16-file" && i + 1 < argc) {
      input_a_fp16_file = argv[++i];
    } else if (arg == "--input-b-fp16-file" && i + 1 < argc) {
      input_b_fp16_file = argv[++i];
    } else if (arg == "--expected-output-fp16-file" && i + 1 < argc) {
      expected_output_fp16_file = argv[++i];
    } else if (arg == "--output-fp16-ulp-tolerance" && i + 1 < argc) {
      std::string err = parse_u32_option("--output-fp16-ulp-tolerance",
                                         argv[++i],
                                         &output_fp16_ulp_tolerance);
      if (!err.empty()) return json_error(err);
    } else if (arg == "--help") {
      std::cout << "usage: vk_residual_add_probe [options]\n";
      std::cout << "  --length N                         vector length (default 1024)\n";
      std::cout << "  --input-a-fp16-file PATH            raw fp16 input A vector\n";
      std::cout << "  --input-b-fp16-file PATH            raw fp16 input B vector\n";
      std::cout << "  --expected-output-fp16-file PATH    raw fp16 expected output vector\n";
      std::cout << "  --output-fp16-ulp-tolerance N       allowed output fp16 ULP diff (default 0)\n";
      std::cout << "  --help                              show this help\n";
      return 0;
    }
  }

  if (length == 0) {
    return json_error("--length must be > 0");
  }
  if (input_a_fp16_file.empty()) {
    return json_error("--input-a-fp16-file is required");
  }
  if (input_b_fp16_file.empty()) {
    return json_error("--input-b-fp16-file is required");
  }
  if (expected_output_fp16_file.empty()) {
    return json_error("--expected-output-fp16-file is required");
  }

  std::vector<std::uint16_t> input_a;
  std::vector<std::uint16_t> input_b;
  std::vector<std::uint16_t> expected_output;
  try {
    input_a = load_fp16_file(input_a_fp16_file, length, "--input-a-fp16-file");
    input_b = load_fp16_file(input_b_fp16_file, length, "--input-b-fp16-file");
    expected_output = load_fp16_file(expected_output_fp16_file, length, "--expected-output-fp16-file");
  } catch (const std::exception& e) {
    return json_error(e.what());
  }

#if SPOCK_HAS_VULKAN && !defined(SPOCK_VULKAN_STUB)
  try {
    spock::runtime::VulkanDevice dev;
    dev.initialize();

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
        dev.create_pipeline_layout(desc_layout, sizeof(std::uint32_t));
    auto spirv = read_spirv();
    VkShaderModule shader = dev.create_shader_module(spirv);
    VkPipeline pipeline = dev.create_compute_pipeline(shader, pipe_layout);

    VkDeviceSize bytes = static_cast<VkDeviceSize>(length) * sizeof(std::uint16_t);
    auto input_a_buf = dev.create_device_local_buffer(bytes);
    auto input_b_buf = dev.create_device_local_buffer(bytes);
    auto output_buf = dev.create_device_local_buffer(bytes);
    std::vector<std::uint16_t> output_zeros(length, 0);
    dev.upload_to_device(input_a_buf, input_a.data(), bytes);
    dev.upload_to_device(input_b_buf, input_b.data(), bytes);
    dev.upload_to_device(output_buf, output_zeros.data(), bytes);

    VkDescriptorSet desc_set = dev.allocate_descriptor_set(desc_layout);
    dev.update_descriptor_set(desc_set, 0, input_a_buf);
    dev.update_descriptor_set(desc_set, 1, input_b_buf);
    dev.update_descriptor_set(desc_set, 2, output_buf);

    VkCommandBuffer cmd = dev.allocate_command_buffer();
    dev.begin_command_buffer(cmd);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipe_layout, 0, 1, &desc_set, 0, nullptr);
    vkCmdPushConstants(cmd, pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(std::uint32_t), &length);
    vkCmdDispatch(cmd, 1, 1, 1);
    dev.end_command_buffer(cmd);
    dev.submit_and_wait(cmd);

    std::vector<std::uint16_t> gpu_output(length);
    dev.download_from_device(output_buf, gpu_output.data(), bytes);

    std::uint32_t output_exact_mismatches = 0;
    std::uint32_t output_within_tolerance = 0;
    std::uint32_t output_mismatches = 0;
    std::uint32_t max_fp16_ulp_diff = 0;
    int first_mismatch_row = -1;
    for (std::uint32_t row = 0; row < length; ++row) {
      std::uint32_t ulp = fp16_ulp_diff(gpu_output[row], expected_output[row]);
      if (ulp > max_fp16_ulp_diff) {
        max_fp16_ulp_diff = ulp;
      }
      if (ulp == 0) continue;
      ++output_exact_mismatches;
      if (ulp <= output_fp16_ulp_tolerance) {
        ++output_within_tolerance;
      } else {
        ++output_mismatches;
        if (first_mismatch_row < 0) {
          first_mismatch_row = static_cast<int>(row);
        }
      }
    }

    dev.destroy_buffer(input_a_buf);
    dev.destroy_buffer(input_b_buf);
    dev.destroy_buffer(output_buf);
    dev.destroy_pipeline(pipeline);
    dev.destroy_shader_module(shader);
    dev.destroy_pipeline_layout(pipe_layout);
    dev.destroy_descriptor_set_layout(desc_layout);

    const bool ok = output_mismatches == 0;
    std::cout << "{\n";
    std::cout << "  \"length\": " << length << ",\n";
    std::cout << "  \"input_a_fp16_file\": \"" << input_a_fp16_file << "\",\n";
    std::cout << "  \"input_b_fp16_file\": \"" << input_b_fp16_file << "\",\n";
    std::cout << "  \"expected_output_fp16_file\": \"" << expected_output_fp16_file << "\",\n";
    std::cout << "  \"status\": \"" << (ok ? "ok" : "fail") << "\",\n";
    std::cout << "  \"output_exact_mismatches\": " << output_exact_mismatches << ",\n";
    std::cout << "  \"output_within_tolerance\": " << output_within_tolerance << ",\n";
    std::cout << "  \"output_mismatches\": " << output_mismatches << ",\n";
    std::cout << "  \"max_fp16_ulp_diff\": " << max_fp16_ulp_diff << ",\n";
    std::cout << "  \"output_fp16_ulp_tolerance\": " << output_fp16_ulp_tolerance;
    if (first_mismatch_row >= 0) {
      std::cout << ",\n  \"first_mismatch_row\": " << first_mismatch_row;
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

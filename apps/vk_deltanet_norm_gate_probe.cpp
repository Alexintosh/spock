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
#include "runtime/weight_loader.hpp"

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

  auto spv = try_load("build/shaders/deltanet_norm_gate.comp.spv");
  if (!spv.empty()) return spv;

#ifdef SHADER_DIR
  spv = try_load(std::string(SHADER_DIR) + "/deltanet_norm_gate.comp.spv");
  if (!spv.empty()) return spv;
#endif

  throw std::runtime_error(
      "cannot open shader: deltanet_norm_gate.comp.spv "
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
  if ((a == 0x0000 || a == 0x8000) && (b == 0x0000 || b == 0x8000)) return 0;
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

std::vector<float> load_fp32_weight_vector(
    const spock::runtime::WeightArtifact& artifact,
    const std::string& role,
    std::uint32_t extract_len) {
  const auto* info = artifact.find_by_role(role);
  if (!info) {
    throw std::runtime_error("weight role not found: " + role);
  }
  if (info->dtype != "fp32") {
    throw std::runtime_error("weight dtype must be fp32 for role '" +
                             role + "', got: " + info->dtype);
  }
  if (info->shape.size() != 1) {
    throw std::runtime_error("weight must be rank-1 for role '" +
                             role + "', got rank: " +
                             std::to_string(info->shape.size()));
  }
  std::uint32_t tensor_len = static_cast<std::uint32_t>(info->shape[0]);
  if (extract_len > tensor_len) {
    throw std::runtime_error("extract_len (" + std::to_string(extract_len) +
                             ") > tensor len (" + std::to_string(tensor_len) +
                             ") for role '" + role + "'");
  }

  auto raw = spock::runtime::read_tensor_bytes(artifact, *info);
  std::vector<float> result(extract_len);
  std::memcpy(result.data(), raw.data(), static_cast<std::size_t>(extract_len) * sizeof(float));
  return result;
}

std::uint32_t float_to_bits(float v) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &v, sizeof(bits));
  return bits;
}

struct PushConstants {
  std::uint32_t num_heads;
  std::uint32_t head_dim;
  std::uint32_t epsilon_bits;
  std::uint32_t output_offset;
};

}  // namespace

int main(int argc, char** argv) {
  std::uint32_t num_heads = 0;
  std::uint32_t head_dim = 0;
  std::uint32_t output_fp16_ulp_tolerance = 0;
  std::string repack_dir;
  std::string weight_role;
  std::string core_fp16_file;
  std::string gate_fp16_file;
  std::string expected_output_fp16_file;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--num-heads" && i + 1 < argc) {
      std::string err = parse_u32_option("--num-heads", argv[++i], &num_heads);
      if (!err.empty()) return json_error(err);
    } else if (arg == "--head-dim" && i + 1 < argc) {
      std::string err = parse_u32_option("--head-dim", argv[++i], &head_dim);
      if (!err.empty()) return json_error(err);
    } else if (arg == "--repack-dir" && i + 1 < argc) {
      repack_dir = argv[++i];
    } else if (arg == "--weight-role" && i + 1 < argc) {
      weight_role = argv[++i];
    } else if (arg == "--core-fp16-file" && i + 1 < argc) {
      core_fp16_file = argv[++i];
    } else if (arg == "--gate-fp16-file" && i + 1 < argc) {
      gate_fp16_file = argv[++i];
    } else if (arg == "--expected-output-fp16-file" && i + 1 < argc) {
      expected_output_fp16_file = argv[++i];
    } else if (arg == "--output-fp16-ulp-tolerance" && i + 1 < argc) {
      std::string err = parse_u32_option("--output-fp16-ulp-tolerance",
                                         argv[++i],
                                         &output_fp16_ulp_tolerance);
      if (!err.empty()) return json_error(err);
    } else if (arg == "--help") {
      std::cout << "usage: vk_deltanet_norm_gate_probe [options]\n";
      std::cout << "  --num-heads N                       number of DeltaNet value heads\n";
      std::cout << "  --head-dim N                        per-head value dimension\n";
      std::cout << "  --repack-dir DIR                    load fp32 norm weight from repacked artifact\n";
      std::cout << "  --weight-role ROLE                  fp32 norm weight role, e.g. layer.0.delta_norm\n";
      std::cout << "  --core-fp16-file PATH               raw fp16 recurrent core vector\n";
      std::cout << "  --gate-fp16-file PATH               raw fp16 z gate vector\n";
      std::cout << "  --expected-output-fp16-file PATH    raw fp16 expected gated output vector\n";
      std::cout << "  --output-fp16-ulp-tolerance N       allowed output fp16 ULP diff (default 0)\n";
      std::cout << "  --help                              show this help\n";
      return 0;
    }
  }

  if (num_heads == 0) {
    return json_error("--num-heads is required and must be > 0");
  }
  if (head_dim == 0) {
    return json_error("--head-dim is required and must be > 0");
  }
  if (repack_dir.empty()) {
    return json_error("--repack-dir is required");
  }
  if (weight_role.empty()) {
    return json_error("--weight-role is required");
  }
  if (core_fp16_file.empty()) {
    return json_error("--core-fp16-file is required");
  }
  if (gate_fp16_file.empty()) {
    return json_error("--gate-fp16-file is required");
  }
  if (expected_output_fp16_file.empty()) {
    return json_error("--expected-output-fp16-file is required");
  }

  std::uint32_t length = 0;
  if (num_heads > UINT32_MAX / head_dim) {
    return json_error("--num-heads * --head-dim overflows uint32");
  }
  length = num_heads * head_dim;

  std::vector<float> weight_data;
  try {
    auto artifact = spock::runtime::WeightArtifact::load(repack_dir);
    weight_data = load_fp32_weight_vector(artifact, weight_role, head_dim);
  } catch (const std::exception& e) {
    return json_error(std::string("weight loading failed: ") + e.what());
  }

  std::vector<std::uint16_t> core_data;
  std::vector<std::uint16_t> gate_data;
  std::vector<std::uint16_t> expected_output;
  try {
    core_data = load_fp16_file(core_fp16_file, length, "--core-fp16-file");
    gate_data = load_fp16_file(gate_fp16_file, length, "--gate-fp16-file");
    expected_output = load_fp16_file(expected_output_fp16_file, length,
                                     "--expected-output-fp16-file");
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
        dev.create_pipeline_layout(desc_layout, sizeof(PushConstants));
    auto spirv = read_spirv();
    VkShaderModule shader = dev.create_shader_module(spirv);
    VkPipeline pipeline = dev.create_compute_pipeline(shader, pipe_layout);

    VkDeviceSize fp16_bytes = static_cast<VkDeviceSize>(length) * sizeof(std::uint16_t);
    VkDeviceSize weight_bytes = static_cast<VkDeviceSize>(head_dim) * sizeof(float);
    auto core_buf = dev.create_device_local_buffer(fp16_bytes);
    auto gate_buf = dev.create_device_local_buffer(fp16_bytes);
    auto weight_buf = dev.create_device_local_buffer(weight_bytes);
    dev.upload_to_device(core_buf, core_data.data(), fp16_bytes);
    dev.upload_to_device(gate_buf, gate_data.data(), fp16_bytes);
    dev.upload_to_device(weight_buf, weight_data.data(), weight_bytes);

    VkDescriptorSet desc_set = dev.allocate_descriptor_set(desc_layout);
    dev.update_descriptor_set(desc_set, 0, core_buf);
    dev.update_descriptor_set(desc_set, 1, gate_buf);
    dev.update_descriptor_set(desc_set, 2, weight_buf);

    PushConstants pc{};
    pc.num_heads = num_heads;
    pc.head_dim = head_dim;
    pc.epsilon_bits = float_to_bits(1.0e-6f);
    pc.output_offset = 0;

    VkCommandBuffer cmd = dev.allocate_command_buffer();
    dev.begin_command_buffer(cmd);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipe_layout, 0, 1, &desc_set, 0, nullptr);
    vkCmdPushConstants(cmd, pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(PushConstants), &pc);
    vkCmdDispatch(cmd, num_heads, 1, 1);
    dev.end_command_buffer(cmd);
    dev.submit_and_wait(cmd);

    std::vector<std::uint16_t> gpu_output(length);
    dev.download_from_device(core_buf, gpu_output.data(), fp16_bytes);

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

    dev.destroy_buffer(core_buf);
    dev.destroy_buffer(gate_buf);
    dev.destroy_buffer(weight_buf);
    dev.destroy_pipeline(pipeline);
    dev.destroy_shader_module(shader);
    dev.destroy_pipeline_layout(pipe_layout);
    dev.destroy_descriptor_set_layout(desc_layout);

    const bool ok = output_mismatches == 0;
    std::cout << "{\n";
    std::cout << "  \"num_heads\": " << num_heads << ",\n";
    std::cout << "  \"head_dim\": " << head_dim << ",\n";
    std::cout << "  \"weight_role\": \"" << weight_role << "\",\n";
    std::cout << "  \"core_fp16_file\": \"" << core_fp16_file << "\",\n";
    std::cout << "  \"gate_fp16_file\": \"" << gate_fp16_file << "\",\n";
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

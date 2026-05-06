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

  auto spv = try_load("build/shaders/deltanet_compute_g_beta.comp.spv");
  if (!spv.empty()) return spv;

#ifdef SHADER_DIR
  spv = try_load(std::string(SHADER_DIR) + "/deltanet_compute_g_beta.comp.spv");
  if (!spv.empty()) return spv;
#endif

  throw std::runtime_error(
      "cannot open shader: deltanet_compute_g_beta.comp.spv "
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

float half_to_float(std::uint16_t h) {
  std::uint32_t sign = (h >> 15) & 1;
  std::uint32_t exponent = (h >> 10) & 0x1f;
  std::uint32_t mantissa = h & 0x3ff;

  std::uint32_t f = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      f = sign << 31;
    } else {
      exponent = 127 - 15 + 1;
      while ((mantissa & 0x400) == 0) {
        mantissa <<= 1;
        exponent--;
      }
      mantissa &= 0x3ff;
      f = (sign << 31) | (exponent << 23) | (mantissa << 13);
    }
  } else if (exponent == 31) {
    f = (sign << 31) | (0xff << 23) | (mantissa << 13);
  } else {
    f = (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13);
  }

  float result = 0.0f;
  std::memcpy(&result, &f, sizeof(result));
  return result;
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

std::vector<float> load_g_beta_weights(
    const spock::runtime::WeightArtifact& artifact,
    std::uint32_t num_heads) {
  const auto* a_info = artifact.find_by_role("layer.0.delta_a_log");
  const auto* dt_info = artifact.find_by_role("layer.0.delta_dt_bias");
  if (!a_info) throw std::runtime_error("weight role not found: layer.0.delta_a_log");
  if (!dt_info) throw std::runtime_error("weight role not found: layer.0.delta_dt_bias");
  if (a_info->dtype != "fp32") {
    throw std::runtime_error("layer.0.delta_a_log must be fp32");
  }
  if (dt_info->dtype != "fp16") {
    throw std::runtime_error("layer.0.delta_dt_bias must be fp16");
  }
  if (a_info->shape.size() != 1 || dt_info->shape.size() != 1 ||
      a_info->shape[0] < num_heads || dt_info->shape[0] < num_heads) {
    throw std::runtime_error("g/beta weights must be rank-1 and cover num_heads");
  }

  auto a_raw = spock::runtime::read_tensor_bytes(artifact, *a_info);
  auto dt_raw = spock::runtime::read_tensor_bytes(artifact, *dt_info);
  const auto* a_vals = reinterpret_cast<const float*>(a_raw.data());
  const auto* dt_vals = reinterpret_cast<const std::uint16_t*>(dt_raw.data());

  std::vector<float> packed(static_cast<std::size_t>(num_heads) * 2);
  for (std::uint32_t h = 0; h < num_heads; ++h) {
    packed[h * 2 + 0] = a_vals[h];
    packed[h * 2 + 1] = half_to_float(dt_vals[h]);
  }
  return packed;
}

struct PushConstants {
  std::uint32_t num_heads;
  std::uint32_t layer_idx;
};

}  // namespace

int main(int argc, char** argv) {
  std::uint32_t num_heads = 0;
  std::uint32_t layer_index = 0;
  std::string repack_dir;
  std::string a_fp16_file;
  std::string b_fp16_file;
  std::string expected_g_beta_bits_file;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--num-heads" && i + 1 < argc) {
      std::string err = parse_u32_option("--num-heads", argv[++i], &num_heads);
      if (!err.empty()) return json_error(err);
    } else if (arg == "--layer-index" && i + 1 < argc) {
      std::string err = parse_u32_option("--layer-index", argv[++i], &layer_index);
      if (!err.empty()) return json_error(err);
    } else if (arg == "--repack-dir" && i + 1 < argc) {
      repack_dir = argv[++i];
    } else if (arg == "--a-fp16-file" && i + 1 < argc) {
      a_fp16_file = argv[++i];
    } else if (arg == "--b-fp16-file" && i + 1 < argc) {
      b_fp16_file = argv[++i];
    } else if (arg == "--expected-g-beta-bits-file" && i + 1 < argc) {
      expected_g_beta_bits_file = argv[++i];
    } else if (arg == "--help") {
      std::cout << "usage: vk_deltanet_g_beta_probe [options]\n";
      std::cout << "  --num-heads N                       number of DeltaNet heads\n";
      std::cout << "  --layer-index N                     DeltaNet layer index in packed a_log/dt_bias buffer\n";
      std::cout << "  --repack-dir DIR                    load delta_a_log and delta_dt_bias from repacked artifact\n";
      std::cout << "  --a-fp16-file PATH                  raw fp16 projected a vector\n";
      std::cout << "  --b-fp16-file PATH                  raw fp16 projected b vector\n";
      std::cout << "  --expected-g-beta-bits-file PATH    raw uint32 expected fp32 bits, g then beta\n";
      std::cout << "  --help                              show this help\n";
      return 0;
    }
  }

  if (num_heads == 0) return json_error("--num-heads is required and must be > 0");
  if (layer_index != 0) return json_error("only --layer-index 0 is supported by this probe");
  if (repack_dir.empty()) return json_error("--repack-dir is required");
  if (a_fp16_file.empty()) return json_error("--a-fp16-file is required");
  if (b_fp16_file.empty()) return json_error("--b-fp16-file is required");
  if (expected_g_beta_bits_file.empty()) {
    return json_error("--expected-g-beta-bits-file is required");
  }

  std::vector<std::uint16_t> a_data;
  std::vector<std::uint16_t> b_data;
  std::vector<std::uint32_t> expected_bits;
  std::vector<float> ab_data;
  try {
    a_data = load_fp16_file(a_fp16_file, num_heads, "--a-fp16-file");
    b_data = load_fp16_file(b_fp16_file, num_heads, "--b-fp16-file");
    expected_bits = load_u32_file(expected_g_beta_bits_file, num_heads * 2,
                                  "--expected-g-beta-bits-file");
    auto artifact = spock::runtime::WeightArtifact::load(repack_dir);
    ab_data = load_g_beta_weights(artifact, num_heads);
  } catch (const std::exception& e) {
    return json_error(e.what());
  }

#if SPOCK_HAS_VULKAN && !defined(SPOCK_VULKAN_STUB)
  try {
    spock::runtime::VulkanDevice dev;
    dev.initialize();

    std::vector<VkDescriptorSetLayoutBinding> bindings(4);
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

    VkDeviceSize fp16_bytes = static_cast<VkDeviceSize>(num_heads) * sizeof(std::uint16_t);
    VkDeviceSize fp32_pair_bytes = static_cast<VkDeviceSize>(num_heads) * 2 * sizeof(float);
    auto a_buf = dev.create_device_local_buffer(fp16_bytes);
    auto b_buf = dev.create_device_local_buffer(fp16_bytes);
    auto ab_buf = dev.create_device_local_buffer(fp32_pair_bytes);
    auto out_buf = dev.create_device_local_buffer(fp32_pair_bytes);
    std::vector<float> zeros(static_cast<std::size_t>(num_heads) * 2, 0.0f);
    dev.upload_to_device(a_buf, a_data.data(), fp16_bytes);
    dev.upload_to_device(b_buf, b_data.data(), fp16_bytes);
    dev.upload_to_device(ab_buf, ab_data.data(), fp32_pair_bytes);
    dev.upload_to_device(out_buf, zeros.data(), fp32_pair_bytes);

    VkDescriptorSet desc_set = dev.allocate_descriptor_set(desc_layout);
    dev.update_descriptor_set(desc_set, 0, a_buf);
    dev.update_descriptor_set(desc_set, 1, b_buf);
    dev.update_descriptor_set(desc_set, 2, ab_buf);
    dev.update_descriptor_set(desc_set, 3, out_buf);

    PushConstants pc{num_heads, layer_index};
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

    std::vector<std::uint32_t> gpu_bits(num_heads * 2);
    dev.download_from_device(out_buf, gpu_bits.data(), fp32_pair_bytes);

    std::uint32_t exact_mismatches = 0;
    int first_mismatch_row = -1;
    for (std::uint32_t i = 0; i < num_heads * 2; ++i) {
      if (gpu_bits[i] != expected_bits[i]) {
        ++exact_mismatches;
        if (first_mismatch_row < 0) first_mismatch_row = static_cast<int>(i);
      }
    }

    dev.destroy_buffer(a_buf);
    dev.destroy_buffer(b_buf);
    dev.destroy_buffer(ab_buf);
    dev.destroy_buffer(out_buf);
    dev.destroy_pipeline(pipeline);
    dev.destroy_shader_module(shader);
    dev.destroy_pipeline_layout(pipe_layout);
    dev.destroy_descriptor_set_layout(desc_layout);

    const bool ok = exact_mismatches == 0;
    std::cout << "{\n";
    std::cout << "  \"num_heads\": " << num_heads << ",\n";
    std::cout << "  \"layer_index\": " << layer_index << ",\n";
    std::cout << "  \"a_fp16_file\": \"" << a_fp16_file << "\",\n";
    std::cout << "  \"b_fp16_file\": \"" << b_fp16_file << "\",\n";
    std::cout << "  \"expected_g_beta_bits_file\": \"" << expected_g_beta_bits_file << "\",\n";
    std::cout << "  \"status\": \"" << (ok ? "ok" : "fail") << "\",\n";
    std::cout << "  \"output_exact_mismatches\": " << exact_mismatches;
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

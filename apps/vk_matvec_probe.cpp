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

  auto spv = try_load("build/shaders/matvec.comp.spv");
  if (!spv.empty()) return spv;

#ifdef SHADER_DIR
  spv = try_load(std::string(SHADER_DIR) + "/matvec.comp.spv");
  if (!spv.empty()) return spv;
#endif

  throw std::runtime_error(
      "cannot open shader: matvec.comp.spv "
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
  // Both zeros: +0 (0x0000) and -0 (0x8000) are equivalent.
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

// Load a rank-2 fp16 weight matrix from the repack artifact, extracting
// the first extract_rows x extract_cols in row-major order.
std::vector<uint16_t> load_weight_matrix(
    const spock::runtime::WeightArtifact& artifact,
    const std::string& role,
    uint32_t extract_rows, uint32_t extract_cols) {
  const auto* info = artifact.find_by_role(role);
  if (!info) {
    throw std::runtime_error("weight role not found: " + role);
  }
  if (info->dtype != "fp16") {
    throw std::runtime_error("weight dtype must be fp16 for role '" +
                             role + "', got: " + info->dtype);
  }
  if (info->shape.size() != 2) {
    throw std::runtime_error("weight must be rank-2 for role '" +
                             role + "', got rank: " +
                             std::to_string(info->shape.size()));
  }
  uint32_t tensor_rows = static_cast<uint32_t>(info->shape[0]);
  uint32_t tensor_cols = static_cast<uint32_t>(info->shape[1]);

  if (extract_rows > tensor_rows) {
    throw std::runtime_error("extract_rows (" + std::to_string(extract_rows) +
                             ") > tensor rows (" + std::to_string(tensor_rows) +
                             ") for role '" + role + "'");
  }
  if (extract_cols > tensor_cols) {
    throw std::runtime_error("extract_cols (" + std::to_string(extract_cols) +
                             ") > tensor cols (" + std::to_string(tensor_cols) +
                             ") for role '" + role + "'");
  }

  auto raw = spock::runtime::read_tensor_bytes(artifact, *info);
  std::vector<uint16_t> result(static_cast<std::size_t>(extract_rows) * extract_cols);
  for (uint32_t row = 0; row < extract_rows; ++row) {
    std::size_t dst_offset = static_cast<std::size_t>(row) * extract_cols;
    std::size_t src_offset =
        (static_cast<std::size_t>(row) * tensor_cols) * sizeof(uint16_t);
    std::memcpy(result.data() + dst_offset,
                raw.data() + src_offset,
                extract_cols * sizeof(uint16_t));
  }
  return result;
}

struct PushConstants {
  std::uint32_t out_dim;
  std::uint32_t in_dim;
};

}  // namespace

int main(int argc, char** argv) {
  std::uint32_t in_dim = 0;
  std::uint32_t out_dim = 0;
  std::string repack_dir;
  std::string weight_role;
  std::string input_fp16_file;
  std::string expected_output_fp16_file;
  std::uint32_t output_fp16_ulp_tolerance = 0;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--in-dim" && i + 1 < argc) {
      std::string err = parse_u32_option("--in-dim", argv[++i], &in_dim);
      if (!err.empty()) return json_error(err);
    } else if (arg == "--out-dim" && i + 1 < argc) {
      std::string err = parse_u32_option("--out-dim", argv[++i], &out_dim);
      if (!err.empty()) return json_error(err);
    } else if (arg == "--repack-dir" && i + 1 < argc) {
      repack_dir = argv[++i];
    } else if (arg == "--weight-role" && i + 1 < argc) {
      weight_role = argv[++i];
    } else if (arg == "--input-fp16-file" && i + 1 < argc) {
      input_fp16_file = argv[++i];
    } else if (arg == "--expected-output-fp16-file" && i + 1 < argc) {
      expected_output_fp16_file = argv[++i];
    } else if (arg == "--output-fp16-ulp-tolerance" && i + 1 < argc) {
      std::string err = parse_u32_option("--output-fp16-ulp-tolerance",
                                         argv[++i],
                                         &output_fp16_ulp_tolerance);
      if (!err.empty()) return json_error(err);
    } else if (arg == "--help") {
      std::cout << "usage: vk_matvec_probe [options]\n";
      std::cout << "  --in-dim N                          input vector dimension (required)\n";
      std::cout << "  --out-dim N                         output vector dimension (required)\n";
      std::cout << "  --repack-dir DIR                    load real fp16 weight from repacked model artifact\n";
      std::cout << "  --weight-role ROLE                  weight role for the matvec matrix (e.g. layer.0.delta_out_proj)\n";
      std::cout << "  --input-fp16-file PATH              raw fp16 input vector file (required)\n";
      std::cout << "  --expected-output-fp16-file PATH    raw fp16 expected output vector file (required)\n";
      std::cout << "  --output-fp16-ulp-tolerance N       allowed output fp16 ULP diff (default 0, exact)\n";
      std::cout << "  --help                              show this help\n";
      return 0;
    }
  }

  if (in_dim == 0) {
    return json_error("--in-dim is required and must be > 0");
  }
  if (out_dim == 0) {
    return json_error("--out-dim is required and must be > 0");
  }
  if (input_fp16_file.empty()) {
    return json_error("--input-fp16-file is required");
  }
  if (expected_output_fp16_file.empty()) {
    return json_error("--expected-output-fp16-file is required");
  }
  if (repack_dir.empty() && weight_role.empty()) {
    return json_error("--repack-dir and --weight-role are required (no synthetic weight mode)");
  }
  if (!weight_role.empty() && repack_dir.empty()) {
    return json_error("--weight-role requires --repack-dir");
  }
  if (repack_dir.empty()) {
    return json_error("--repack-dir is required");
  }

  // Load weight matrix.
  std::vector<std::uint16_t> weight_data;
  try {
    auto artifact = spock::runtime::WeightArtifact::load(repack_dir);
    weight_data = load_weight_matrix(artifact, weight_role, out_dim, in_dim);
  } catch (const std::exception& e) {
    return json_error(std::string("weight loading failed: ") + e.what());
  }

  // Load input vector.
  std::vector<std::uint16_t> input_data;
  try {
    input_data = load_fp16_file(input_fp16_file, in_dim, "--input-fp16-file");
  } catch (const std::exception& e) {
    return json_error(e.what());
  }

  // Load expected output vector.
  std::vector<std::uint16_t> expected_output;
  try {
    expected_output = load_fp16_file(expected_output_fp16_file, out_dim,
                                     "--expected-output-fp16-file");
  } catch (const std::exception& e) {
    return json_error(e.what());
  }

#if SPOCK_HAS_VULKAN && !defined(SPOCK_VULKAN_STUB)
  try {
    spock::runtime::VulkanDevice dev;
    dev.initialize();

    // 3 bindings: weight (read), input (read), output (write).
    std::vector<VkDescriptorSetLayoutBinding> bindings(3);
    for (std::uint32_t b = 0; b < bindings.size(); ++b) {
      bindings[b].binding = b;
      bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings[b].descriptorCount = 1;
      bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      bindings[b].pImmutableSamplers = nullptr;
    }

    VkDescriptorSetLayout desc_layout = dev.create_descriptor_set_layout(bindings);
    // Push constants: 2 x uint32 = 8 bytes.
    VkPipelineLayout pipe_layout =
        dev.create_pipeline_layout(desc_layout, sizeof(PushConstants));
    auto spirv = read_spirv();
    VkShaderModule shader = dev.create_shader_module(spirv);
    VkPipeline pipeline = dev.create_compute_pipeline(shader, pipe_layout);

    VkDeviceSize weight_bytes = static_cast<VkDeviceSize>(out_dim) * in_dim * sizeof(std::uint16_t);
    VkDeviceSize input_bytes = static_cast<VkDeviceSize>(in_dim) * sizeof(std::uint16_t);
    VkDeviceSize output_bytes = static_cast<VkDeviceSize>(out_dim) * sizeof(std::uint16_t);

    auto weight_buf = dev.create_device_local_buffer(weight_bytes);
    auto input_buf = dev.create_device_local_buffer(input_bytes);
    auto output_buf = dev.create_device_local_buffer(output_bytes);

    dev.upload_to_device(weight_buf, weight_data.data(), weight_bytes);
    dev.upload_to_device(input_buf, input_data.data(), input_bytes);
    // Zero output buffer.
    std::vector<std::uint16_t> output_zeros(out_dim, 0);
    dev.upload_to_device(output_buf, output_zeros.data(), output_bytes);

    VkDescriptorSet desc_set = dev.allocate_descriptor_set(desc_layout);
    dev.update_descriptor_set(desc_set, 0, weight_buf);
    dev.update_descriptor_set(desc_set, 1, input_buf);
    dev.update_descriptor_set(desc_set, 2, output_buf);

    PushConstants pc{};
    pc.out_dim = out_dim;
    pc.in_dim = in_dim;

    // matvec.comp: local_size_x = 64, dispatch ceil(out_dim/64) workgroups.
    std::uint32_t dispatch_x = (out_dim + 63u) / 64u;

    VkCommandBuffer cmd = dev.allocate_command_buffer();
    dev.begin_command_buffer(cmd);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipe_layout, 0, 1, &desc_set, 0, nullptr);
    vkCmdPushConstants(cmd, pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(PushConstants), &pc);
    vkCmdDispatch(cmd, dispatch_x, 1, 1);
    dev.end_command_buffer(cmd);
    dev.submit_and_wait(cmd);

    std::vector<std::uint16_t> gpu_output(out_dim);
    dev.download_from_device(output_buf, gpu_output.data(), output_bytes);

    // Compare.
    std::uint32_t output_exact_mismatches = 0;
    std::uint32_t output_within_tolerance = 0;
    std::uint32_t output_mismatches = 0;
    std::uint32_t max_fp16_ulp_diff = 0;
    int first_mismatch_row = -1;
    for (std::uint32_t row = 0; row < out_dim; ++row) {
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

    dev.destroy_buffer(weight_buf);
    dev.destroy_buffer(input_buf);
    dev.destroy_buffer(output_buf);
    dev.destroy_pipeline(pipeline);
    dev.destroy_shader_module(shader);
    dev.destroy_pipeline_layout(pipe_layout);
    dev.destroy_descriptor_set_layout(desc_layout);

    const bool ok = output_mismatches == 0;
    std::cout << "{\n";
    std::cout << "  \"in_dim\": " << in_dim << ",\n";
    std::cout << "  \"out_dim\": " << out_dim << ",\n";
    std::cout << "  \"weight_role\": \"" << weight_role << "\",\n";
    std::cout << "  \"input_fp16_file\": \"" << input_fp16_file << "\",\n";
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

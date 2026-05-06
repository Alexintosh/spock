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

  auto spv = try_load("build/shaders/" + shader_name + ".comp.spv");
  if (!spv.empty()) return spv;

#ifdef SHADER_DIR
  spv = try_load(std::string(SHADER_DIR) + "/" + shader_name + ".comp.spv");
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

std::vector<std::uint16_t> load_weight_matrix(
    const spock::runtime::WeightArtifact& artifact,
    const std::string& role,
    std::uint32_t extract_rows, std::uint32_t extract_cols) {
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
  std::uint32_t tensor_rows = static_cast<std::uint32_t>(info->shape[0]);
  std::uint32_t tensor_cols = static_cast<std::uint32_t>(info->shape[1]);
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
  std::vector<std::uint16_t> result(static_cast<std::size_t>(extract_rows) * extract_cols);
  for (std::uint32_t row = 0; row < extract_rows; ++row) {
    std::size_t dst_offset = static_cast<std::size_t>(row) * extract_cols;
    std::size_t src_offset =
        (static_cast<std::size_t>(row) * tensor_cols) * sizeof(std::uint16_t);
    std::memcpy(result.data() + dst_offset,
                raw.data() + src_offset,
                extract_cols * sizeof(std::uint16_t));
  }
  return result;
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

struct MatvecPushConstants {
  std::uint32_t out_dim;
  std::uint32_t in_dim;
};

struct ConvPushConstants {
  std::uint32_t conv_dim;
  std::uint32_t kernel_size;
};

struct L2PushConstants {
  std::uint32_t num_heads;
  std::uint32_t head_dim;
};

struct GBetaPushConstants {
  std::uint32_t num_heads;
  std::uint32_t layer_idx;
};

struct RecurrentPushConstants {
  std::uint32_t num_heads;
  std::uint32_t k_dim;
  std::uint32_t v_dim;
  std::uint32_t state_total;
  std::uint32_t q_scale_bits;
};

struct NormGatePushConstants {
  std::uint32_t num_heads;
  std::uint32_t head_dim;
  std::uint32_t epsilon_bits;
  std::uint32_t output_offset;
};

}  // namespace

int main(int argc, char** argv) {
  // --- Dimension constants (layer-0 Qwen 3.5 0.8B) ---
  const std::uint32_t hidden = 1024;
  const std::uint32_t num_heads = 16;
  const std::uint32_t head_dim = 128;
  const std::uint32_t qkv_dim = 6144;
  const std::uint32_t key_total = num_heads * head_dim;   // 2048
  const std::uint32_t val_total = num_heads * head_dim;   // 2048
  const std::uint32_t conv_kernel = 4;
  const std::uint32_t state_matrix = num_heads * head_dim * head_dim;  // 262144
  const std::uint32_t state_total = state_matrix;                       // offset for g/beta
  const std::uint32_t state_with_tail = state_matrix + num_heads * 2;   // 262176

  // --- CLI args ---
  std::string repack_dir;
  std::string input_norm_fp16_file;
  std::string input_hidden_fp16_file;
  std::string conv_state_pre_fp16_file;
  std::string state_pre_f32_file;
  std::string expected_mixer_output_fp16_file;
  std::string expected_mixer_residual_fp16_file;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--repack-dir" && i + 1 < argc) {
      repack_dir = argv[++i];
    } else if (arg == "--input-norm-fp16-file" && i + 1 < argc) {
      input_norm_fp16_file = argv[++i];
    } else if (arg == "--input-hidden-fp16-file" && i + 1 < argc) {
      input_hidden_fp16_file = argv[++i];
    } else if (arg == "--conv-state-pre-fp16-file" && i + 1 < argc) {
      conv_state_pre_fp16_file = argv[++i];
    } else if (arg == "--state-pre-f32-file" && i + 1 < argc) {
      state_pre_f32_file = argv[++i];
    } else if (arg == "--expected-mixer-output-fp16-file" && i + 1 < argc) {
      expected_mixer_output_fp16_file = argv[++i];
    } else if (arg == "--expected-mixer-residual-fp16-file" && i + 1 < argc) {
      expected_mixer_residual_fp16_file = argv[++i];
    } else if (arg == "--help") {
      std::cout << "usage: vk_deltanet_mixer_probe [options]\n";
      std::cout << "  --repack-dir DIR                          load repacked model weights\n";
      std::cout << "  --input-norm-fp16-file PATH               fp16 input norm vector (hidden=1024 values)\n";
      std::cout << "  --input-hidden-fp16-file PATH             fp16 pre-norm hidden vector (hidden=1024 values)\n";
      std::cout << "  --conv-state-pre-fp16-file PATH           fp16 pre-conv rolling state (qkv_dim*kernel values)\n";
      std::cout << "  --state-pre-f32-file PATH                 fp32 pre-recurrent state (state_with_tail values)\n";
      std::cout << "  --expected-mixer-output-fp16-file PATH    fp16 expected mixer output (hidden values)\n";
      std::cout << "  --expected-mixer-residual-fp16-file PATH  fp16 expected mixer residual (hidden values)\n";
      std::cout << "  --help                                    show this help\n";
      return 0;
    }
  }

  if (repack_dir.empty()) return json_error("--repack-dir is required");
  if (input_norm_fp16_file.empty()) return json_error("--input-norm-fp16-file is required");
  if (input_hidden_fp16_file.empty()) return json_error("--input-hidden-fp16-file is required");
  if (conv_state_pre_fp16_file.empty()) return json_error("--conv-state-pre-fp16-file is required");
  if (state_pre_f32_file.empty()) return json_error("--state-pre-f32-file is required");
  if (expected_mixer_output_fp16_file.empty()) return json_error("--expected-mixer-output-fp16-file is required");
  if (expected_mixer_residual_fp16_file.empty()) return json_error("--expected-mixer-residual-fp16-file is required");

  // --- Load fixtures and weights ---
  std::vector<std::uint16_t> input_norm_data;
  std::vector<std::uint16_t> input_hidden_data;
  std::vector<std::uint16_t> conv_state_pre_data;
  std::vector<float> state_pre_data;
  std::vector<std::uint16_t> expected_mixer_output;
  std::vector<std::uint16_t> expected_mixer_residual;

  std::vector<std::uint16_t> weight_qkv;
  std::vector<std::uint16_t> weight_z;
  std::vector<std::uint16_t> weight_a;
  std::vector<std::uint16_t> weight_b;
  std::vector<std::uint16_t> weight_conv;
  std::vector<float> weight_ab;
  std::vector<float> weight_norm;
  std::vector<std::uint16_t> weight_out_proj;

  try {
    auto artifact = spock::runtime::WeightArtifact::load(repack_dir);
    weight_qkv = load_weight_matrix(artifact, "layer.0.delta_in_proj_qkv", qkv_dim, hidden);
    weight_z = load_weight_matrix(artifact, "layer.0.delta_in_proj_z", val_total, hidden);
    weight_a = load_weight_matrix(artifact, "layer.0.delta_in_proj_a", num_heads, hidden);
    weight_b = load_weight_matrix(artifact, "layer.0.delta_in_proj_b", num_heads, hidden);
    weight_conv = load_conv_weights(artifact, qkv_dim, conv_kernel);
    weight_ab = load_g_beta_weights(artifact, num_heads);
    weight_norm = load_fp32_weight_vector(artifact, "layer.0.delta_norm", head_dim);
    weight_out_proj = load_weight_matrix(artifact, "layer.0.delta_out_proj", hidden, val_total);
  } catch (const std::exception& e) {
    return json_error(std::string("weight loading failed: ") + e.what());
  }

  try {
    input_norm_data = load_fp16_file(input_norm_fp16_file, hidden, "--input-norm-fp16-file");
    input_hidden_data = load_fp16_file(input_hidden_fp16_file, hidden, "--input-hidden-fp16-file");
    conv_state_pre_data = load_fp16_file(conv_state_pre_fp16_file,
                                         qkv_dim * conv_kernel,
                                         "--conv-state-pre-fp16-file");
    state_pre_data = load_fp32_file(state_pre_f32_file, state_with_tail,
                                    "--state-pre-f32-file");
    expected_mixer_output = load_fp16_file(expected_mixer_output_fp16_file, hidden,
                                           "--expected-mixer-output-fp16-file");
    expected_mixer_residual = load_fp16_file(expected_mixer_residual_fp16_file, hidden,
                                             "--expected-mixer-residual-fp16-file");
  } catch (const std::exception& e) {
    return json_error(e.what());
  }

  // q_scale = 1/sqrt(k_dim) for recurrent shader
  float q_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  std::uint32_t q_scale_bits_val = 0;
  std::memcpy(&q_scale_bits_val, &q_scale, sizeof(q_scale_bits_val));

  float eps = 1.0e-6f;
  std::uint32_t epsilon_bits = float_to_bits(eps);


#if SPOCK_HAS_VULKAN && !defined(SPOCK_VULKAN_STUB)
  try {
    spock::runtime::VulkanDevice dev;
    dev.initialize();

    // --- Descriptor set layouts: 3-binding and 4-binding ---
    auto make_bindings = [](std::uint32_t count) {
      std::vector<VkDescriptorSetLayoutBinding> bindings(count);
      for (std::uint32_t b = 0; b < count; ++b) {
        bindings[b].binding = b;
        bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[b].descriptorCount = 1;
        bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[b].pImmutableSamplers = nullptr;
      }
      return bindings;
    };

    auto bindings3 = make_bindings(3);
    auto bindings4 = make_bindings(4);
    VkDescriptorSetLayout layout3 = dev.create_descriptor_set_layout(bindings3);
    VkDescriptorSetLayout layout4 = dev.create_descriptor_set_layout(bindings4);

    // --- Pipeline layouts ---
    VkPipelineLayout matvec_layout = dev.create_pipeline_layout(layout3, sizeof(MatvecPushConstants));
    VkPipelineLayout conv_layout = dev.create_pipeline_layout(layout3, sizeof(ConvPushConstants));
    VkPipelineLayout l2_layout = dev.create_pipeline_layout(layout3, sizeof(L2PushConstants));
    VkPipelineLayout gbeta_layout = dev.create_pipeline_layout(layout4, sizeof(GBetaPushConstants));
    VkPipelineLayout recurrent_layout = dev.create_pipeline_layout(layout3, sizeof(RecurrentPushConstants));
    VkPipelineLayout norm_gate_layout = dev.create_pipeline_layout(layout3, sizeof(NormGatePushConstants));
    VkPipelineLayout residual_layout = dev.create_pipeline_layout(layout3, sizeof(std::uint32_t));

    // --- Shader modules and pipelines ---
    auto matvec_spirv = read_spirv("matvec");
    auto conv_spirv = read_spirv("conv1d_step");
    auto l2_spirv = read_spirv("l2_norm_per_head");
    auto gbeta_spirv = read_spirv("deltanet_compute_g_beta");
    auto recurrent_spirv = read_spirv("deltanet_recurrent");
    auto norm_gate_spirv = read_spirv("deltanet_norm_gate");
    auto residual_spirv = read_spirv("residual_add");

    VkShaderModule matvec_sm = dev.create_shader_module(matvec_spirv);
    VkShaderModule conv_sm = dev.create_shader_module(conv_spirv);
    VkShaderModule l2_sm = dev.create_shader_module(l2_spirv);
    VkShaderModule gbeta_sm = dev.create_shader_module(gbeta_spirv);
    VkShaderModule recurrent_sm = dev.create_shader_module(recurrent_spirv);
    VkShaderModule norm_gate_sm = dev.create_shader_module(norm_gate_spirv);
    VkShaderModule residual_sm = dev.create_shader_module(residual_spirv);

    VkPipeline matvec_pipe = dev.create_compute_pipeline(matvec_sm, matvec_layout);
    VkPipeline conv_pipe = dev.create_compute_pipeline(conv_sm, conv_layout);
    VkPipeline l2_pipe = dev.create_compute_pipeline(l2_sm, l2_layout);
    VkPipeline gbeta_pipe = dev.create_compute_pipeline(gbeta_sm, gbeta_layout);
    VkPipeline recurrent_pipe = dev.create_compute_pipeline(recurrent_sm, recurrent_layout);
    VkPipeline norm_gate_pipe = dev.create_compute_pipeline(norm_gate_sm, norm_gate_layout);
    VkPipeline residual_pipe = dev.create_compute_pipeline(residual_sm, residual_layout);

    // --- Buffers ---
    // Weight buffers (device-local, uploaded once)
    VkDeviceSize weight_qkv_bytes = static_cast<VkDeviceSize>(qkv_dim) * hidden * 2;
    VkDeviceSize weight_z_bytes = static_cast<VkDeviceSize>(val_total) * hidden * 2;
    VkDeviceSize weight_a_bytes = static_cast<VkDeviceSize>(num_heads) * hidden * 2;
    VkDeviceSize weight_b_bytes = static_cast<VkDeviceSize>(num_heads) * hidden * 2;
    VkDeviceSize weight_conv_bytes = static_cast<VkDeviceSize>(qkv_dim) * conv_kernel * 2;
    VkDeviceSize weight_ab_bytes = static_cast<VkDeviceSize>(num_heads) * 2 * 4;
    VkDeviceSize weight_norm_bytes = static_cast<VkDeviceSize>(head_dim) * 4;
    VkDeviceSize weight_out_bytes = static_cast<VkDeviceSize>(hidden) * val_total * 2;

    auto w_qkv_buf = dev.create_device_local_buffer(weight_qkv_bytes);
    auto w_z_buf = dev.create_device_local_buffer(weight_z_bytes);
    auto w_a_buf = dev.create_device_local_buffer(weight_a_bytes);
    auto w_b_buf = dev.create_device_local_buffer(weight_b_bytes);
    auto w_conv_buf = dev.create_device_local_buffer(weight_conv_bytes);
    auto w_ab_buf = dev.create_device_local_buffer(weight_ab_bytes);
    auto w_norm_buf = dev.create_device_local_buffer(weight_norm_bytes);
    auto w_out_buf = dev.create_device_local_buffer(weight_out_bytes);

    dev.upload_to_device(w_qkv_buf, weight_qkv.data(), weight_qkv_bytes);
    dev.upload_to_device(w_z_buf, weight_z.data(), weight_z_bytes);
    dev.upload_to_device(w_a_buf, weight_a.data(), weight_a_bytes);
    dev.upload_to_device(w_b_buf, weight_b.data(), weight_b_bytes);
    dev.upload_to_device(w_conv_buf, weight_conv.data(), weight_conv_bytes);
    dev.upload_to_device(w_ab_buf, weight_ab.data(), weight_ab_bytes);
    dev.upload_to_device(w_norm_buf, weight_norm.data(), weight_norm_bytes);
    dev.upload_to_device(w_out_buf, weight_out_proj.data(), weight_out_bytes);

    // Activation / state buffers
    VkDeviceSize input_norm_bytes = static_cast<VkDeviceSize>(hidden) * 2;
    VkDeviceSize qkv_bytes = static_cast<VkDeviceSize>(qkv_dim) * 2;
    VkDeviceSize z_bytes = static_cast<VkDeviceSize>(val_total) * 2;
    VkDeviceSize a_bytes = static_cast<VkDeviceSize>(num_heads) * 2;
    VkDeviceSize b_bytes = static_cast<VkDeviceSize>(num_heads) * 2;
    VkDeviceSize conv_state_bytes = static_cast<VkDeviceSize>(qkv_dim) * conv_kernel * 2;
    VkDeviceSize state_bytes = static_cast<VkDeviceSize>(state_with_tail) * 4;
    VkDeviceSize input_hidden_bytes = static_cast<VkDeviceSize>(hidden) * 2;
    VkDeviceSize mixer_output_bytes = static_cast<VkDeviceSize>(hidden) * 2;
    VkDeviceSize mixer_residual_bytes = static_cast<VkDeviceSize>(hidden) * 2;
    // Dummy for l2_norm binding 2 (unused but required)
    VkDeviceSize dummy_bytes = static_cast<VkDeviceSize>(key_total) * 2;
    // KV+out buffer for recurrent: K section + V section = key_total + val_total
    VkDeviceSize kv_bytes = static_cast<VkDeviceSize>(key_total + val_total) * 2;

    auto input_norm_buf = dev.create_device_local_buffer(input_norm_bytes);
    auto qkv_buf = dev.create_device_local_buffer(qkv_bytes);
    auto z_buf = dev.create_device_local_buffer(z_bytes);
    auto a_buf = dev.create_device_local_buffer(a_bytes);
    auto b_buf = dev.create_device_local_buffer(b_bytes);
    auto conv_state_buf = dev.create_device_local_buffer(conv_state_bytes);
    auto state_buf = dev.create_device_local_buffer(state_bytes);
    auto input_hidden_buf = dev.create_device_local_buffer(input_hidden_bytes);
    auto mixer_output_buf = dev.create_device_local_buffer(mixer_output_bytes);
    auto mixer_residual_buf = dev.create_device_local_buffer(mixer_residual_bytes);
    auto dummy_buf = dev.create_device_local_buffer(dummy_bytes);

    // Staging for output download (mixer_output + mixer_residual)
    VkDeviceSize staging_size = mixer_output_bytes + mixer_residual_bytes;
    auto output_staging = dev.create_host_visible_buffer(staging_size);

    // Upload fixture data
    dev.upload_to_device(input_norm_buf, input_norm_data.data(), input_norm_bytes);
    dev.upload_to_device(input_hidden_buf, input_hidden_data.data(), input_hidden_bytes);
    dev.upload_to_device(conv_state_buf, conv_state_pre_data.data(), conv_state_bytes);
    dev.upload_to_device(state_buf, state_pre_data.data(), state_bytes);

    // Zero output buffers
    std::uint32_t max_buf_dim = qkv_dim;
    if (val_total > max_buf_dim) max_buf_dim = val_total;
    if (hidden > max_buf_dim) max_buf_dim = hidden;
    if (key_total + val_total > max_buf_dim) max_buf_dim = key_total + val_total;
    std::vector<std::uint16_t> zeros_fp16(max_buf_dim, 0);
    dev.upload_to_device(qkv_buf, zeros_fp16.data(), qkv_bytes);
    dev.upload_to_device(z_buf, zeros_fp16.data(), z_bytes);
    dev.upload_to_device(a_buf, zeros_fp16.data(), a_bytes);
    dev.upload_to_device(b_buf, zeros_fp16.data(), b_bytes);
    dev.upload_to_device(mixer_output_buf, zeros_fp16.data(), mixer_output_bytes);
    dev.upload_to_device(mixer_residual_buf, zeros_fp16.data(), mixer_residual_bytes);

    // --- Allocate descriptor sets ---
    // Matvec sets (3-binding): stages 1-4 and 10
    // Each needs: weight(input_norm), input_buf, output_buf
    // But weight roles change each time. We'll allocate fresh sets for each matvec stage.
    VkDescriptorSet ds_matvec1 = dev.allocate_descriptor_set(layout3);  // input_norm -> qkv_raw
    VkDescriptorSet ds_matvec2 = dev.allocate_descriptor_set(layout3);  // input_norm -> z
    VkDescriptorSet ds_matvec3 = dev.allocate_descriptor_set(layout3);  // input_norm -> a
    VkDescriptorSet ds_matvec4 = dev.allocate_descriptor_set(layout3);  // input_norm -> b
    VkDescriptorSet ds_matvec10 = dev.allocate_descriptor_set(layout3); // gated -> mixer_output

    dev.update_descriptor_set(ds_matvec1, 0, w_qkv_buf);
    dev.update_descriptor_set(ds_matvec1, 1, input_norm_buf);
    dev.update_descriptor_set(ds_matvec1, 2, qkv_buf);

    dev.update_descriptor_set(ds_matvec2, 0, w_z_buf);
    dev.update_descriptor_set(ds_matvec2, 1, input_norm_buf);
    dev.update_descriptor_set(ds_matvec2, 2, z_buf);

    dev.update_descriptor_set(ds_matvec3, 0, w_a_buf);
    dev.update_descriptor_set(ds_matvec3, 1, input_norm_buf);
    dev.update_descriptor_set(ds_matvec3, 2, a_buf);

    dev.update_descriptor_set(ds_matvec4, 0, w_b_buf);
    dev.update_descriptor_set(ds_matvec4, 1, input_norm_buf);
    dev.update_descriptor_set(ds_matvec4, 2, b_buf);

    // Stage 10: out_proj reads gated from V section of qkv_buf
    // Recurrent binding 1 starts at K section; inside that view, V/output is at key_total * 2 bytes,
    // so globally the V section is at 2 * key_total * sizeof(uint16_t).
    VkDeviceSize v_section_offset_bytes = static_cast<VkDeviceSize>(2) * key_total * 2;
    dev.update_descriptor_set(ds_matvec10, 0, w_out_buf);
    dev.update_descriptor_set(ds_matvec10, 1, qkv_buf, v_section_offset_bytes, z_bytes);
    dev.update_descriptor_set(ds_matvec10, 2, mixer_output_buf);

    // Conv set (3-binding)
    VkDescriptorSet ds_conv = dev.allocate_descriptor_set(layout3);
    dev.update_descriptor_set(ds_conv, 0, qkv_buf);
    dev.update_descriptor_set(ds_conv, 1, conv_state_buf);
    dev.update_descriptor_set(ds_conv, 2, w_conv_buf);

    // L2 Q set (3-binding, in-place on Q section)
    VkDeviceSize q_section_bytes = static_cast<VkDeviceSize>(key_total) * 2;
    VkDescriptorSet ds_l2q = dev.allocate_descriptor_set(layout3);
    dev.update_descriptor_set(ds_l2q, 0, qkv_buf, 0, q_section_bytes);
    dev.update_descriptor_set(ds_l2q, 1, qkv_buf, 0, q_section_bytes);
    dev.update_descriptor_set(ds_l2q, 2, dummy_buf);

    // L2 K set (3-binding, in-place on K section)
    VkDeviceSize k_offset_bytes = static_cast<VkDeviceSize>(key_total) * 2;
    VkDescriptorSet ds_l2k = dev.allocate_descriptor_set(layout3);
    dev.update_descriptor_set(ds_l2k, 0, qkv_buf, k_offset_bytes, q_section_bytes);
    dev.update_descriptor_set(ds_l2k, 1, qkv_buf, k_offset_bytes, q_section_bytes);
    dev.update_descriptor_set(ds_l2k, 2, dummy_buf);

    // G/beta set (4-binding)
    VkDescriptorSet ds_gbeta = dev.allocate_descriptor_set(layout4);
    dev.update_descriptor_set(ds_gbeta, 0, a_buf);
    dev.update_descriptor_set(ds_gbeta, 1, b_buf);
    dev.update_descriptor_set(ds_gbeta, 2, w_ab_buf);
    // binding 3: writable state, pointing at g/beta tail
    VkDeviceSize state_tail_offset = static_cast<VkDeviceSize>(state_matrix) * 4;
    VkDeviceSize state_tail_bytes = static_cast<VkDeviceSize>(num_heads) * 2 * 4;
    dev.update_descriptor_set(ds_gbeta, 3, state_buf, state_tail_offset, state_tail_bytes);

    // Recurrent set (3-binding)
    // binding 0: Q section of qkv_buf (first key_total fp16 values)
    // binding 1: KV+out buffer - we create a combined K+V view of qkv_buf
    //   The recurrent shader expects: K [num_heads*k_dim] then V [num_heads*v_dim]
    //   In qkv_buf: Q [key_total] then K [key_total] then V [val_total]
    //   So binding 1 should point at K section start with size kv_bytes
    // binding 2: state
    VkDescriptorSet ds_recurrent = dev.allocate_descriptor_set(layout3);
    dev.update_descriptor_set(ds_recurrent, 0, qkv_buf, 0, q_section_bytes);
    dev.update_descriptor_set(ds_recurrent, 1, qkv_buf, k_offset_bytes, kv_bytes);
    dev.update_descriptor_set(ds_recurrent, 2, state_buf);

    // Norm-gate set (3-binding)
    // binding 0: io - V section of qkv_buf (core output, overwritten with gated)
    // binding 1: gate - z_buf
    // binding 2: norm weight - w_norm_buf
    VkDescriptorSet ds_norm_gate = dev.allocate_descriptor_set(layout3);
    dev.update_descriptor_set(ds_norm_gate, 0, qkv_buf, v_section_offset_bytes, z_bytes);
    dev.update_descriptor_set(ds_norm_gate, 1, z_buf);
    dev.update_descriptor_set(ds_norm_gate, 2, w_norm_buf);

    // Residual set (3-binding)
    VkDescriptorSet ds_residual = dev.allocate_descriptor_set(layout3);
    dev.update_descriptor_set(ds_residual, 0, input_hidden_buf);
    dev.update_descriptor_set(ds_residual, 1, mixer_output_buf);
    dev.update_descriptor_set(ds_residual, 2, mixer_residual_buf);

    // --- Record command buffer ---
    VkCommandBuffer cmd = dev.allocate_command_buffer();
    dev.begin_command_buffer(cmd);

    // Helper: compute-shader -> compute-shader barrier
    auto cs_barrier = [&]() {
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
    };

    // --- Stage 1: matvec input_norm -> qkv_raw ---
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, matvec_pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            matvec_layout, 0, 1, &ds_matvec1, 0, nullptr);
    MatvecPushConstants pc1{qkv_dim, hidden};
    vkCmdPushConstants(cmd, matvec_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(MatvecPushConstants), &pc1);
    vkCmdDispatch(cmd, (qkv_dim + 63u) / 64u, 1, 1);

    cs_barrier();

    // --- Stage 2: matvec input_norm -> z ---
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            matvec_layout, 0, 1, &ds_matvec2, 0, nullptr);
    MatvecPushConstants pc2{val_total, hidden};
    vkCmdPushConstants(cmd, matvec_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(MatvecPushConstants), &pc2);
    vkCmdDispatch(cmd, (val_total + 63u) / 64u, 1, 1);

    // Stage 2 output not read until stage 9 (norm_gate), no barrier needed now

    // --- Stage 3: matvec input_norm -> a ---
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            matvec_layout, 0, 1, &ds_matvec3, 0, nullptr);
    MatvecPushConstants pc3{num_heads, hidden};
    vkCmdPushConstants(cmd, matvec_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(MatvecPushConstants), &pc3);
    vkCmdDispatch(cmd, (num_heads + 63u) / 64u, 1, 1);

    // --- Stage 4: matvec input_norm -> b ---
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            matvec_layout, 0, 1, &ds_matvec4, 0, nullptr);
    MatvecPushConstants pc4{num_heads, hidden};
    vkCmdPushConstants(cmd, matvec_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(MatvecPushConstants), &pc4);
    vkCmdDispatch(cmd, (num_heads + 63u) / 64u, 1, 1);

    // Barrier: stages 1,3,4 writes complete before conv reads qkv, and g/beta reads a,b
    cs_barrier();

    // --- Stage 5: conv1d_step on qkv ---
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, conv_pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            conv_layout, 0, 1, &ds_conv, 0, nullptr);
    ConvPushConstants pc5{qkv_dim, conv_kernel};
    vkCmdPushConstants(cmd, conv_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(ConvPushConstants), &pc5);
    vkCmdDispatch(cmd, 1, 1, 1);

    cs_barrier();

    // --- Stage 6a: L2-norm Q (in-place on first key_total of qkv) ---
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, l2_pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            l2_layout, 0, 1, &ds_l2q, 0, nullptr);
    L2PushConstants pc6a{num_heads, head_dim};
    vkCmdPushConstants(cmd, l2_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(L2PushConstants), &pc6a);
    vkCmdDispatch(cmd, num_heads, 1, 1);

    cs_barrier();

    // --- Stage 6b: L2-norm K (in-place on second key_total of qkv) ---
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            l2_layout, 0, 1, &ds_l2k, 0, nullptr);
    L2PushConstants pc6b{num_heads, head_dim};
    vkCmdPushConstants(cmd, l2_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(L2PushConstants), &pc6b);
    vkCmdDispatch(cmd, num_heads, 1, 1);


    // --- Stage 7: g/beta computation ---
    // Need barrier: stage 6b writes qkv, stage 7 doesn't read qkv but reads a,b
    // Actually stage 7 reads a_buf and b_buf which were written in stages 3,4.
    // The barrier after stages 3,4 covers that. But state_buf also needs to be ready.
    // State was uploaded before dispatch. So the barrier before stage 5 already covered a,b.
    // We do need to ensure stages 6a/6b don't interfere with stage 7's state write.
    // Stage 7 writes to state_buf tail. No conflict with stages 6a/6b (which only touch qkv_buf).
    // But we need to ensure stage 7's write is visible before stage 8 reads state.
    // We'll add a barrier after stage 7.

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, gbeta_pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            gbeta_layout, 0, 1, &ds_gbeta, 0, nullptr);
    GBetaPushConstants pc7{num_heads, 0};
    vkCmdPushConstants(cmd, gbeta_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(GBetaPushConstants), &pc7);
    vkCmdDispatch(cmd, num_heads, 1, 1);

    cs_barrier();

    // --- Stage 8: recurrent core ---
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, recurrent_pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            recurrent_layout, 0, 1, &ds_recurrent, 0, nullptr);
    RecurrentPushConstants pc8{num_heads, head_dim, head_dim, state_total, q_scale_bits_val};
    vkCmdPushConstants(cmd, recurrent_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(RecurrentPushConstants), &pc8);
    vkCmdDispatch(cmd, num_heads, 1, 1);

    cs_barrier();

    // --- Stage 9: norm-gate ---
    // Reads core (V section of qkv_buf, written by stage 8) and z_buf (written by stage 2).
    // Writes gated output in-place to V section.
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, norm_gate_pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            norm_gate_layout, 0, 1, &ds_norm_gate, 0, nullptr);
    NormGatePushConstants pc9{num_heads, head_dim, epsilon_bits, 0};
    vkCmdPushConstants(cmd, norm_gate_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(NormGatePushConstants), &pc9);
    vkCmdDispatch(cmd, num_heads, 1, 1);

    cs_barrier();

    // --- Stage 10: out_proj matvec ---
    // Reads gated output from V section of qkv_buf, writes mixer_output_buf.
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, matvec_pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            matvec_layout, 0, 1, &ds_matvec10, 0, nullptr);
    MatvecPushConstants pc10{hidden, val_total};
    vkCmdPushConstants(cmd, matvec_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(MatvecPushConstants), &pc10);
    vkCmdDispatch(cmd, (hidden + 63u) / 64u, 1, 1);

    cs_barrier();

    // --- Stage 11: residual add ---
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, residual_pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            residual_layout, 0, 1, &ds_residual, 0, nullptr);
    vkCmdPushConstants(cmd, residual_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(std::uint32_t), &hidden);
    vkCmdDispatch(cmd, 1, 1, 1);

    // --- Copy outputs to staging ---
    {
      VkBufferMemoryBarrier barrier{};
      barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
      barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.buffer = mixer_output_buf.buffer;
      barrier.offset = 0;
      barrier.size = mixer_output_bytes;
      vkCmdPipelineBarrier(cmd,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT,
                           0,
                           0, nullptr,
                           1, &barrier,
                           0, nullptr);
    }
    {
      VkBufferMemoryBarrier barrier{};
      barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
      barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.buffer = mixer_residual_buf.buffer;
      barrier.offset = 0;
      barrier.size = mixer_residual_bytes;
      vkCmdPipelineBarrier(cmd,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT,
                           0,
                           0, nullptr,
                           1, &barrier,
                           0, nullptr);
    }

    VkBufferCopy copy_output{0, 0, mixer_output_bytes};
    vkCmdCopyBuffer(cmd, mixer_output_buf.buffer, output_staging.buffer, 1, &copy_output);
    VkBufferCopy copy_residual{0, mixer_output_bytes, mixer_residual_bytes};
    vkCmdCopyBuffer(cmd, mixer_residual_buf.buffer, output_staging.buffer, 1, &copy_residual);


    dev.end_command_buffer(cmd);
    dev.submit_and_wait(cmd);

    // --- Download from staging ---
    std::vector<std::uint16_t> gpu_mixer_output(hidden);
    std::memcpy(gpu_mixer_output.data(), output_staging.mapped,
                static_cast<std::size_t>(mixer_output_bytes));
    std::vector<std::uint16_t> gpu_mixer_residual(hidden);
    std::memcpy(gpu_mixer_residual.data(),
                static_cast<const char*>(output_staging.mapped) + mixer_output_bytes,
                static_cast<std::size_t>(mixer_residual_bytes));

    // --- Compare mixer output ---
    std::uint32_t output_mismatches = 0;
    std::uint32_t output_max_ulp = 0;
    int output_first_mismatch = -1;
    std::uint16_t output_first_gpu = 0, output_first_expected = 0;
    for (std::uint32_t i = 0; i < hidden; ++i) {
      if (gpu_mixer_output[i] != expected_mixer_output[i]) {
        ++output_mismatches;
        std::uint32_t ulp = fp16_ulp_diff(gpu_mixer_output[i], expected_mixer_output[i]);
        if (ulp > output_max_ulp) output_max_ulp = ulp;
        if (output_first_mismatch < 0) {
          output_first_mismatch = static_cast<int>(i);
          output_first_gpu = gpu_mixer_output[i];
          output_first_expected = expected_mixer_output[i];
        }
      }
    }

    // --- Compare mixer residual ---
    std::uint32_t residual_mismatches = 0;
    std::uint32_t residual_max_ulp = 0;
    int residual_first_mismatch = -1;
    std::uint16_t residual_first_gpu = 0, residual_first_expected = 0;
    for (std::uint32_t i = 0; i < hidden; ++i) {
      if (gpu_mixer_residual[i] != expected_mixer_residual[i]) {
        ++residual_mismatches;
        std::uint32_t ulp = fp16_ulp_diff(gpu_mixer_residual[i], expected_mixer_residual[i]);
        if (ulp > residual_max_ulp) residual_max_ulp = ulp;
        if (residual_first_mismatch < 0) {
          residual_first_mismatch = static_cast<int>(i);
          residual_first_gpu = gpu_mixer_residual[i];
          residual_first_expected = expected_mixer_residual[i];
        }
      }
    }

    // --- Cleanup ---
    auto destroy_bufs = [&](auto&... bufs) { (dev.destroy_buffer(bufs), ...); };
    destroy_bufs(w_qkv_buf, w_z_buf, w_a_buf, w_b_buf, w_conv_buf,
                 w_ab_buf, w_norm_buf, w_out_buf);
    destroy_bufs(input_norm_buf, qkv_buf, z_buf, a_buf, b_buf,
                 conv_state_buf, state_buf, input_hidden_buf,
                 mixer_output_buf, mixer_residual_buf, dummy_buf, output_staging);

    auto destroy_pipes = [&](auto&... pipes) { (dev.destroy_pipeline(pipes), ...); };
    destroy_pipes(matvec_pipe, conv_pipe, l2_pipe, gbeta_pipe,
                  recurrent_pipe, norm_gate_pipe, residual_pipe);

    auto destroy_sms = [&](auto&... sms) { (dev.destroy_shader_module(sms), ...); };
    destroy_sms(matvec_sm, conv_sm, l2_sm, gbeta_sm,
                recurrent_sm, norm_gate_sm, residual_sm);

    auto destroy_layouts = [&](auto&... ls) { (dev.destroy_pipeline_layout(ls), ...); };
    destroy_layouts(matvec_layout, conv_layout, l2_layout, gbeta_layout,
                    recurrent_layout, norm_gate_layout, residual_layout);

    dev.destroy_descriptor_set_layout(layout3);
    dev.destroy_descriptor_set_layout(layout4);

    // --- JSON output ---
    const bool ok = (output_mismatches == 0 && residual_mismatches == 0);
    std::cout << "{\n";
    std::cout << "  \"status\": \"" << (ok ? "ok" : "fail") << "\",\n";
    std::cout << "  \"hidden\": " << hidden << ",\n";
    std::cout << "  \"num_heads\": " << num_heads << ",\n";
    std::cout << "  \"head_dim\": " << head_dim << ",\n";
    std::cout << "  \"mixer_output_mismatches\": " << output_mismatches << ",\n";
    std::cout << "  \"mixer_output_max_fp16_ulp_diff\": " << output_max_ulp << ",";
    if (output_first_mismatch >= 0) {
      std::cout << "\n  \"mixer_output_first_mismatch_index\": " << output_first_mismatch << ",";
      std::cout << "\n  \"mixer_output_first_mismatch_gpu\": " << output_first_gpu << ",";
      std::cout << "\n  \"mixer_output_first_mismatch_expected\": " << output_first_expected << ",";
    }
    std::cout << "\n  \"mixer_residual_mismatches\": " << residual_mismatches << ",\n";
    std::cout << "  \"mixer_residual_max_fp16_ulp_diff\": " << residual_max_ulp;
    if (residual_first_mismatch >= 0) {
      std::cout << ",\n  \"mixer_residual_first_mismatch_index\": " << residual_first_mismatch;
      std::cout << ",\n  \"mixer_residual_first_mismatch_gpu\": " << residual_first_gpu;
      std::cout << ",\n  \"mixer_residual_first_mismatch_expected\": " << residual_first_expected;
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

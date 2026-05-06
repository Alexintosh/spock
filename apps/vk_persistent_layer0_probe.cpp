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
  uint32_t exp = (u >> 23u) & 0xFFu;
  uint32_t mant = u & 0x007FFFFFu;

  if (exp == 0xFFu) {
    if (mant != 0) {
      return static_cast<uint16_t>(sign | 0x7E00u);
    }
    return static_cast<uint16_t>(sign | 0x7C00u);
  }

  int32_t half_exp = static_cast<int32_t>(exp) - 127 + 15;
  if (half_exp >= 31) {
    return static_cast<uint16_t>(sign | 0x7C00u);
  }

  auto round_shift = [](uint32_t value, uint32_t shift) -> uint32_t {
    if (shift == 0) return value;
    const uint32_t halfway = 1u << (shift - 1u);
    const uint32_t mask = (1u << shift) - 1u;
    const uint32_t remainder = value & mask;
    uint32_t rounded = value >> shift;
    if (remainder > halfway || (remainder == halfway && (rounded & 1u))) {
      ++rounded;
    }
    return rounded;
  };

  if (half_exp <= 0) {
    if (half_exp < -10) {
      return static_cast<uint16_t>(sign);
    }
    uint32_t mantissa = mant | 0x00800000u;
    uint32_t half_mant = round_shift(mantissa, static_cast<uint32_t>(14 - half_exp));
    return static_cast<uint16_t>(sign | half_mant);
  }

  uint32_t half_mant = round_shift(mant, 13u);
  if (half_mant == 0x400u) {
    half_mant = 0;
    ++half_exp;
    if (half_exp >= 31) {
      return static_cast<uint16_t>(sign | 0x7C00u);
    }
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(half_exp) << 10u) | half_mant);
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

// 128-lane shader-mirrored dot product for row g against input_vec.
// Lane strides by 128, tree reduction stride 64,32,16,8,4,2,1.
static inline float shader_mirrored_dot_128(
    uint32_t g, uint32_t dim,
    const uint16_t* weight_data,
    const uint16_t* input_data) {
  float lane_sums[128] = {};
  for (uint32_t lane = 0; lane < 128; ++lane) {
    float acc = 0.0f;
    uint32_t row_offset = g * dim;
    for (uint32_t c = lane; c < dim; c += 128) {
      float iv = fp16_to_fp32(input_data[c]);
      float wv = fp16_to_fp32(weight_data[row_offset + c]);
      acc += iv * wv;
    }
    lane_sums[lane] = acc;
  }
  for (uint32_t stride = 64; stride >= 1; stride >>= 1) {
    for (uint32_t lane = 0; lane < stride; ++lane) {
      lane_sums[lane] += lane_sums[lane + stride];
    }
  }
  return lane_sums[0];
}

// 128-lane shader-mirrored RMSNorm sum-of-squares reduction.
static inline float shader_mirrored_sum_sq_128(uint32_t dim, const uint16_t* input_data) {
  float lane_sums[128] = {};
  for (uint32_t lane = 0; lane < 128; ++lane) {
    float acc = 0.0f;
    for (uint32_t c = lane; c < dim; c += 128) {
      float v = fp16_to_fp32(input_data[c]);
      acc += v * v;
    }
    lane_sums[lane] = acc;
  }
  for (uint32_t stride = 64; stride >= 1; stride >>= 1) {
    for (uint32_t lane = 0; lane < stride; ++lane) {
      lane_sums[lane] += lane_sums[lane + stride];
    }
  }
  return lane_sums[0];
}

// 128-lane shader-mirrored down dot product.
static inline float shader_mirrored_down_dot_128(
    uint32_t out_row, uint32_t intermediate_count,
    const uint16_t* weight_down_data,
    const uint16_t* activated_data) {
  float lane_sums[128] = {};
  uint32_t row_offset = out_row * intermediate_count;
  for (uint32_t lane = 0; lane < 128; ++lane) {
    float acc = 0.0f;
    for (uint32_t c = lane; c < intermediate_count; c += 128) {
      float wv = fp16_to_fp32(weight_down_data[row_offset + c]);
      float av = fp16_to_fp32(activated_data[c]);
      acc += wv * av;
    }
    lane_sums[lane] = acc;
  }
  for (uint32_t stride = 64; stride >= 1; stride >>= 1) {
    for (uint32_t lane = 0; lane < stride; ++lane) {
      lane_sums[lane] += lane_sums[lane + stride];
    }
  }
  return lane_sums[0];
}

// Compute the unsigned ULP distance between two fp16 values.
static inline uint32_t fp16_ulp_diff(uint16_t a, uint16_t b) {
  if (a == b) return 0;
  if ((a == 0x0000 || a == 0x8000) && (b == 0x0000 || b == 0x8000)) return 0;
  bool a_sign = (a & 0x8000u) != 0;
  bool b_sign = (b & 0x8000u) != 0;
  if (a_sign != b_sign) return UINT32_MAX;
  return a > b ? static_cast<uint32_t>(a) - static_cast<uint32_t>(b)
               : static_cast<uint32_t>(b) - static_cast<uint32_t>(a);
}

// SiLU activation.
static inline float silu(float x) {
  return x / (1.0f + std::exp(-x));
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

  auto spv = try_load("build/shaders/persistent_layer0_probe.comp.spv");
  if (!spv.empty()) return spv;

#ifdef SHADER_DIR
  spv = try_load(std::string(SHADER_DIR) + "/persistent_layer0_probe.comp.spv");
  if (!spv.empty()) return spv;
#endif

  throw std::runtime_error(
      "cannot open shader: persistent_layer0_probe.comp.spv "
      "(tried build/shaders/ and SHADER_DIR)");
}

// Emit a JSON error object and return exit code 2.
int json_error(const std::string& message) {
  std::cout << "{\n";
  std::cout << "  \"status\": \"error\",\n";
  std::cout << "  \"message\": \"" << message << "\"\n";
  std::cout << "}\n";
  return 2;
}

// Load a weight role from the artifact, extracting first extract_rows x extract_cols.
std::vector<uint16_t> load_weight_slice(
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

// Load a rank-1 fp16 weight vector.
std::vector<uint16_t> load_weight_vector(
    const spock::runtime::WeightArtifact& artifact,
    const std::string& role,
    uint32_t extract_len) {
  const auto* info = artifact.find_by_role(role);
  if (!info) {
    throw std::runtime_error("weight role not found: " + role);
  }
  if (info->dtype != "fp16") {
    throw std::runtime_error("weight dtype must be fp16 for role '" +
                             role + "', got: " + info->dtype);
  }
  if (info->shape.size() != 1) {
    throw std::runtime_error("weight must be rank-1 for role '" +
                             role + "', got rank: " +
                             std::to_string(info->shape.size()));
  }
  uint32_t tensor_len = static_cast<uint32_t>(info->shape[0]);
  if (extract_len > tensor_len) {
    throw std::runtime_error("extract_len (" + std::to_string(extract_len) +
                             ") > tensor len (" + std::to_string(tensor_len) +
                             ") for role '" + role + "'");
  }
  auto raw = spock::runtime::read_tensor_bytes(artifact, *info);
  std::vector<uint16_t> result(extract_len);
  std::memcpy(result.data(), raw.data(), extract_len * sizeof(uint16_t));
  return result;
}

// Load a rank-1 fp32 weight vector as raw uint32 bits for shader payloads.
std::vector<std::uint32_t> load_fp32_weight_bits_vector(
    const spock::runtime::WeightArtifact& artifact,
    const std::string& role,
    uint32_t extract_len) {
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
  uint32_t tensor_len = static_cast<uint32_t>(info->shape[0]);
  if (extract_len > tensor_len) {
    throw std::runtime_error("extract_len (" + std::to_string(extract_len) +
                             ") > tensor len (" + std::to_string(tensor_len) +
                             ") for role '" + role + "'");
  }
  auto raw = spock::runtime::read_tensor_bytes(artifact, *info);
  std::vector<std::uint32_t> result(extract_len);
  std::memcpy(result.data(), raw.data(), extract_len * sizeof(std::uint32_t));
  return result;
}

// Parse a CLI option as a nonnegative uint32. Returns empty string on success.
std::string parse_u32_option(const std::string& opt,
                             const std::string& value,
                             uint32_t* out) {
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
    *out = static_cast<uint32_t>(val);
  } catch (...) {
    return opt + " must be a nonnegative integer, got: " + value;
  }
  if (pos != value.size()) {
    return opt + " must be a nonnegative integer, got: " + value;
  }
  return {};
}

// Load a raw fp16 file into a vector.
std::vector<uint16_t> load_fp16_file(const std::string& path, uint32_t expected_len) {
  std::vector<uint16_t> data(expected_len);
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    throw std::runtime_error("cannot open fp16 file: " + path);
  }
  auto file_size = f.tellg();
  f.seekg(0);
  auto required = static_cast<std::streamsize>(expected_len) * sizeof(uint16_t);
  if (file_size < required) {
    throw std::runtime_error("fp16 file too small: " + path +
      " need " + std::to_string(expected_len) +
      " x 2 = " + std::to_string(required) +
      " bytes, got " + std::to_string(file_size));
  }
  f.read(reinterpret_cast<char*>(data.data()), required);
  return data;
}

std::vector<float> load_fp32_file(const std::string& path, uint32_t expected_len) {
  std::vector<float> data(expected_len);
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    throw std::runtime_error("cannot open fp32 file: " + path);
  }
  auto file_size = f.tellg();
  f.seekg(0);
  auto required = static_cast<std::streamsize>(expected_len) * sizeof(float);
  if (file_size < required) {
    throw std::runtime_error("fp32 file too small: " + path +
      " need " + std::to_string(expected_len) +
      " x 4 = " + std::to_string(required) +
      " bytes, got " + std::to_string(file_size));
  }
  f.read(reinterpret_cast<char*>(data.data()), required);
  return data;
}

// Per-output projection comparison result.
struct ProjectionCompareResult {
  uint32_t exact_mismatches;
  uint32_t max_fp16_ulp;
  bool all_exact() const { return exact_mismatches == 0; }
};

struct BitsCompareResult {
  uint32_t exact_mismatches;
  int32_t first_mismatch_index;
};

std::vector<std::uint32_t> load_u32_file(
    const std::string& path,
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

struct GBetaControlPayload {
  std::uint32_t arrived;
  std::uint32_t generation;
  std::uint32_t failures;
  std::uint32_t checksum;
  float delta_a_log[16];
  float delta_dt_bias[16];
  std::uint32_t g_beta_bits[32];
};

GBetaControlPayload load_g_beta_control_payload(
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

  GBetaControlPayload payload{};
  for (std::uint32_t h = 0; h < num_heads; ++h) {
    payload.delta_a_log[h] = a_vals[h];
    payload.delta_dt_bias[h] = half_to_float(dt_vals[h]);
  }
  return payload;
}

std::vector<uint16_t> load_conv_weights(
    const spock::runtime::WeightArtifact& artifact,
    uint32_t conv_dim,
    uint32_t kernel_size) {
  const auto* info = artifact.find_by_role("layer.0.delta_conv");
  if (!info) {
    throw std::runtime_error("weight role not found: layer.0.delta_conv");
  }
  if (info->dtype != "fp16") {
    throw std::runtime_error("layer.0.delta_conv must be fp16");
  }
  uint32_t expected = conv_dim * kernel_size;
  uint32_t actual = static_cast<uint32_t>(info->nbytes / 2);
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

ProjectionCompareResult compare_fp16_output(
    const std::vector<uint16_t>& gpu,
    const std::vector<uint16_t>& expected,
    uint32_t len) {
  ProjectionCompareResult result{0, 0};
  for (uint32_t i = 0; i < len; ++i) {
    uint32_t ulp = fp16_ulp_diff(gpu[i], expected[i]);
    if (ulp > result.max_fp16_ulp) {
      result.max_fp16_ulp = ulp;
    }
    if (ulp != 0) {
      ++result.exact_mismatches;
    }
  }
  return result;
}

BitsCompareResult compare_u32_output(
    const std::vector<std::uint32_t>& gpu,
    const std::vector<std::uint32_t>& expected) {
  BitsCompareResult result{0, -1};
  for (std::size_t i = 0; i < gpu.size(); ++i) {
    if (gpu[i] != expected[i]) {
      ++result.exact_mismatches;
      if (result.first_mismatch_index < 0) {
        result.first_mismatch_index = static_cast<int32_t>(i);
      }
    }
  }
  return result;
}

int run_g_beta_mode(
    uint32_t workgroups,
    const std::string& repack_dir,
    const std::string& a_fp16_file,
    const std::string& b_fp16_file,
    const std::string& expected_g_beta_bits_file) {
  constexpr uint32_t num_heads = 16;

  if (a_fp16_file.empty()) {
    return json_error("--a-fp16-file is required for g-beta mode");
  }
  if (b_fp16_file.empty()) {
    return json_error("--b-fp16-file is required for g-beta mode");
  }
  if (expected_g_beta_bits_file.empty()) {
    return json_error("--expected-g-beta-bits-file is required for g-beta mode");
  }

  std::vector<uint16_t> a_data;
  std::vector<uint16_t> b_data;
  std::vector<std::uint32_t> expected_bits;
  GBetaControlPayload control_payload{};
  try {
    a_data = load_fp16_file(a_fp16_file, num_heads);
    b_data = load_fp16_file(b_fp16_file, num_heads);
    expected_bits = load_u32_file(expected_g_beta_bits_file, num_heads * 2,
                                  "--expected-g-beta-bits-file");
    auto artifact = spock::runtime::WeightArtifact::load(repack_dir);
    control_payload = load_g_beta_control_payload(artifact, num_heads);
  } catch (const std::exception& e) {
    return json_error(std::string("g-beta load failed: ") + e.what());
  }

#if SPOCK_HAS_VULKAN && !defined(SPOCK_VULKAN_STUB)
  try {
    spock::runtime::VulkanDevice dev;
    dev.initialize();

    constexpr int num_bindings = 10;
    std::vector<VkDescriptorSetLayoutBinding> bindings(num_bindings);
    for (int b = 0; b < num_bindings; ++b) {
      bindings[b].binding = b;
      bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings[b].descriptorCount = 1;
      bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      bindings[b].pImmutableSamplers = nullptr;
    }
    VkDescriptorSetLayout desc_layout = dev.create_descriptor_set_layout(bindings);
    VkPipelineLayout pipe_layout =
        dev.create_pipeline_layout(desc_layout, 5 * sizeof(std::uint32_t));

    auto spirv = read_spirv();
    VkShaderModule shader = dev.create_shader_module(spirv);
    VkPipeline pipeline = dev.create_compute_pipeline(shader, pipe_layout);

    struct PushConsts {
      std::uint32_t workgroup_count;
      std::uint32_t hidden;
      std::uint32_t intermediate_count;
      std::uint32_t output_rows;
      std::uint32_t mode;
    };

    VkDeviceSize control_size = sizeof(GBetaControlPayload);
    VkDeviceSize fp16_bytes =
        static_cast<VkDeviceSize>(num_heads) * sizeof(std::uint16_t);
    VkDeviceSize dummy_bytes = sizeof(std::uint16_t);
    std::uint16_t dummy_zero = 0;

    auto control_buf = dev.create_device_local_buffer(control_size);
    dev.upload_to_device(control_buf, &control_payload, control_size);
    auto a_buf = dev.create_device_local_buffer(fp16_bytes);
    dev.upload_to_device(a_buf, a_data.data(), fp16_bytes);
    auto b_buf = dev.create_device_local_buffer(fp16_bytes);
    dev.upload_to_device(b_buf, b_data.data(), fp16_bytes);
    auto dummy3_buf = dev.create_device_local_buffer(dummy_bytes);
    auto dummy4_buf = dev.create_device_local_buffer(dummy_bytes);
    auto dummy5_buf = dev.create_device_local_buffer(dummy_bytes);
    auto dummy6_buf = dev.create_device_local_buffer(dummy_bytes);
    auto dummy7_buf = dev.create_device_local_buffer(dummy_bytes);
    auto dummy8_buf = dev.create_device_local_buffer(dummy_bytes);
    auto dummy9_buf = dev.create_device_local_buffer(dummy_bytes);
    dev.upload_to_device(dummy3_buf, &dummy_zero, dummy_bytes);
    dev.upload_to_device(dummy4_buf, &dummy_zero, dummy_bytes);
    dev.upload_to_device(dummy5_buf, &dummy_zero, dummy_bytes);
    dev.upload_to_device(dummy6_buf, &dummy_zero, dummy_bytes);
    dev.upload_to_device(dummy7_buf, &dummy_zero, dummy_bytes);
    dev.upload_to_device(dummy8_buf, &dummy_zero, dummy_bytes);
    dev.upload_to_device(dummy9_buf, &dummy_zero, dummy_bytes);

    VkDescriptorSet desc_set = dev.allocate_descriptor_set(desc_layout);
    dev.update_descriptor_set(desc_set, 0, control_buf);
    dev.update_descriptor_set(desc_set, 1, a_buf);
    dev.update_descriptor_set(desc_set, 2, b_buf);
    dev.update_descriptor_set(desc_set, 3, dummy3_buf);
    dev.update_descriptor_set(desc_set, 4, dummy4_buf);
    dev.update_descriptor_set(desc_set, 5, dummy5_buf);
    dev.update_descriptor_set(desc_set, 6, dummy6_buf);
    dev.update_descriptor_set(desc_set, 7, dummy7_buf);
    dev.update_descriptor_set(desc_set, 8, dummy8_buf);
    dev.update_descriptor_set(desc_set, 9, dummy9_buf);

    PushConsts push{workgroups, 0, 0, 0, 3};
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

    GBetaControlPayload control_out{};
    dev.download_from_device(control_buf, &control_out, control_size);
    std::vector<std::uint32_t> gpu_bits(std::begin(control_out.g_beta_bits),
                                        std::end(control_out.g_beta_bits));
    auto compare = compare_u32_output(gpu_bits, expected_bits);

    dev.destroy_buffer(control_buf);
    dev.destroy_buffer(a_buf);
    dev.destroy_buffer(b_buf);
    dev.destroy_buffer(dummy3_buf);
    dev.destroy_buffer(dummy4_buf);
    dev.destroy_buffer(dummy5_buf);
    dev.destroy_buffer(dummy6_buf);
    dev.destroy_buffer(dummy7_buf);
    dev.destroy_buffer(dummy8_buf);
    dev.destroy_buffer(dummy9_buf);
    dev.destroy_pipeline(pipeline);
    dev.destroy_shader_module(shader);
    dev.destroy_pipeline_layout(pipe_layout);
    dev.destroy_descriptor_set_layout(desc_layout);

    constexpr uint32_t expected_generation = 0;
    bool ok = control_out.failures == 0 &&
              control_out.arrived == 0 &&
              control_out.generation == expected_generation &&
              compare.exact_mismatches == 0;
    std::cout << "{\n";
    std::cout << "  \"probe\": \"persistent_layer0_g_beta\",\n";
    std::cout << "  \"mode\": \"g-beta\",\n";
    std::cout << "  \"status\": \"" << (ok ? "ok" : "fail") << "\",\n";
    std::cout << "  \"failures\": " << control_out.failures << ",\n";
    std::cout << "  \"arrived\": " << control_out.arrived << ",\n";
    std::cout << "  \"generation\": " << control_out.generation << ",\n";
    std::cout << "  \"expected_generation\": " << expected_generation << ",\n";
    std::cout << "  \"g_beta_bit_mismatches\": " << compare.exact_mismatches;
    if (compare.first_mismatch_index >= 0) {
      std::cout << ",\n  \"first_mismatch_index\": " << compare.first_mismatch_index
                << ",\n  \"expected_bits\": " << expected_bits[compare.first_mismatch_index]
                << ",\n  \"actual_bits\": " << gpu_bits[compare.first_mismatch_index];
    }
    std::cout << "\n}\n";
    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::cout << "{\n";
    std::cout << "  \"probe\": \"persistent_layer0_g_beta\",\n";
    std::cout << "  \"mode\": \"g-beta\",\n";
    std::cout << "  \"status\": \"error\",\n";
    std::cout << "  \"message\": \"" << e.what() << "\"\n";
    std::cout << "}\n";
    return 2;
  }
#else
  return json_error("Vulkan not available (SPOCK_VULKAN_STUB)");
#endif
}

int run_conv_l2_mode(
    uint32_t qkv_rows, uint32_t z_rows, uint32_t workgroups,
    const std::string& repack_dir,
    const std::string& raw_qkv_fp16_file,
    const std::string& conv_state_pre_fp16_file,
    const std::string& expected_q_fp16_file,
    const std::string& expected_k_fp16_file,
    const std::string& expected_v_fp16_file,
    uint32_t conv_l2_fp16_ulp_tolerance) {
  constexpr uint32_t kernel_size = 4;
  constexpr uint32_t head_dim = 128;
  constexpr uint32_t num_heads = 16;

  if (raw_qkv_fp16_file.empty()) {
    return json_error("--raw-qkv-fp16-file is required for conv-l2 mode");
  }
  if (conv_state_pre_fp16_file.empty()) {
    return json_error("--conv-state-pre-fp16-file is required for conv-l2 mode");
  }
  if (expected_q_fp16_file.empty()) {
    return json_error("--expected-q-fp16-file is required for conv-l2 mode");
  }
  if (expected_k_fp16_file.empty()) {
    return json_error("--expected-k-fp16-file is required for conv-l2 mode");
  }
  if (expected_v_fp16_file.empty()) {
    return json_error("--expected-v-fp16-file is required for conv-l2 mode");
  }

  std::vector<uint16_t> raw_qkv_data;
  std::vector<uint16_t> conv_state_pre_data;
  std::vector<uint16_t> expected_q;
  std::vector<uint16_t> expected_k;
  std::vector<uint16_t> expected_v;
  std::vector<uint16_t> conv_weights;
  try {
    raw_qkv_data = load_fp16_file(raw_qkv_fp16_file, qkv_rows);
    conv_state_pre_data = load_fp16_file(conv_state_pre_fp16_file, qkv_rows * kernel_size);
    expected_q = load_fp16_file(expected_q_fp16_file, z_rows);
    expected_k = load_fp16_file(expected_k_fp16_file, z_rows);
    expected_v = load_fp16_file(expected_v_fp16_file, z_rows);
    auto artifact = spock::runtime::WeightArtifact::load(repack_dir);
    conv_weights = load_conv_weights(artifact, qkv_rows, kernel_size);
  } catch (const std::exception& e) {
    return json_error(std::string("conv-l2 load failed: ") + e.what());
  }

#if SPOCK_HAS_VULKAN && !defined(SPOCK_VULKAN_STUB)
  try {
    spock::runtime::VulkanDevice dev;
    dev.initialize();

    constexpr int num_bindings = 10;
    std::vector<VkDescriptorSetLayoutBinding> bindings(num_bindings);
    for (int b = 0; b < num_bindings; ++b) {
      bindings[b].binding = b;
      bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings[b].descriptorCount = 1;
      bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      bindings[b].pImmutableSamplers = nullptr;
    }
    VkDescriptorSetLayout desc_layout = dev.create_descriptor_set_layout(bindings);
    VkPipelineLayout pipe_layout =
        dev.create_pipeline_layout(desc_layout, 5 * sizeof(std::uint32_t));

    auto spirv = read_spirv();
    VkShaderModule shader = dev.create_shader_module(spirv);
    VkPipeline pipeline = dev.create_compute_pipeline(shader, pipe_layout);

    struct PushConsts {
      std::uint32_t workgroup_count;
      std::uint32_t hidden;
      std::uint32_t intermediate_count;
      std::uint32_t output_rows;
      std::uint32_t mode;
    };

    constexpr VkDeviceSize control_size = 4 * sizeof(std::uint32_t);
    const std::uint32_t zero_init[4] = {0, 0, 0, 0};
    VkDeviceSize qkv_bytes =
        static_cast<VkDeviceSize>(qkv_rows) * sizeof(std::uint16_t);
    VkDeviceSize conv_state_bytes =
        static_cast<VkDeviceSize>(qkv_rows) * kernel_size * sizeof(std::uint16_t);
    VkDeviceSize delta_conv_bytes = conv_state_bytes;
    VkDeviceSize q_out_bytes =
        static_cast<VkDeviceSize>(z_rows) * sizeof(std::uint16_t);
    VkDeviceSize dummy_bytes = sizeof(std::uint16_t);
    std::uint16_t dummy_zero = 0;

    auto control_buf = dev.create_device_local_buffer(control_size);
    dev.upload_to_device(control_buf, zero_init, control_size);
    auto qkv_buf = dev.create_device_local_buffer(qkv_bytes);
    dev.upload_to_device(qkv_buf, raw_qkv_data.data(), qkv_bytes);
    auto conv_state_buf = dev.create_device_local_buffer(conv_state_bytes);
    dev.upload_to_device(conv_state_buf, conv_state_pre_data.data(), conv_state_bytes);
    auto delta_conv_buf = dev.create_device_local_buffer(delta_conv_bytes);
    dev.upload_to_device(delta_conv_buf, conv_weights.data(), delta_conv_bytes);
    auto output_q_buf = dev.create_device_local_buffer(q_out_bytes);
    auto output_k_buf = dev.create_device_local_buffer(q_out_bytes);
    auto output_v_buf = dev.create_device_local_buffer(q_out_bytes);
    std::vector<std::uint16_t> zeros(z_rows, 0);
    dev.upload_to_device(output_q_buf, zeros.data(), q_out_bytes);
    dev.upload_to_device(output_k_buf, zeros.data(), q_out_bytes);
    dev.upload_to_device(output_v_buf, zeros.data(), q_out_bytes);
    auto dummy7_buf = dev.create_device_local_buffer(dummy_bytes);
    auto dummy8_buf = dev.create_device_local_buffer(dummy_bytes);
    auto dummy9_buf = dev.create_device_local_buffer(dummy_bytes);
    dev.upload_to_device(dummy7_buf, &dummy_zero, dummy_bytes);
    dev.upload_to_device(dummy8_buf, &dummy_zero, dummy_bytes);
    dev.upload_to_device(dummy9_buf, &dummy_zero, dummy_bytes);

    VkDescriptorSet desc_set = dev.allocate_descriptor_set(desc_layout);
    dev.update_descriptor_set(desc_set, 0, control_buf);
    dev.update_descriptor_set(desc_set, 1, qkv_buf);
    dev.update_descriptor_set(desc_set, 2, conv_state_buf);
    dev.update_descriptor_set(desc_set, 3, delta_conv_buf);
    dev.update_descriptor_set(desc_set, 4, output_q_buf);
    dev.update_descriptor_set(desc_set, 5, output_k_buf);
    dev.update_descriptor_set(desc_set, 6, output_v_buf);
    dev.update_descriptor_set(desc_set, 7, dummy7_buf);
    dev.update_descriptor_set(desc_set, 8, dummy8_buf);
    dev.update_descriptor_set(desc_set, 9, dummy9_buf);

    PushConsts push{workgroups, head_dim, qkv_rows, z_rows, 2};
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

    std::uint32_t control_out[4] = {};
    dev.download_from_device(control_buf, control_out, control_size);
    std::uint32_t arrived = control_out[0];
    std::uint32_t generation = control_out[1];
    std::uint32_t failures = control_out[2];

    std::vector<std::uint16_t> gpu_q(z_rows);
    std::vector<std::uint16_t> gpu_k(z_rows);
    std::vector<std::uint16_t> gpu_v(z_rows);
    dev.download_from_device(output_q_buf, gpu_q.data(), q_out_bytes);
    dev.download_from_device(output_k_buf, gpu_k.data(), q_out_bytes);
    dev.download_from_device(output_v_buf, gpu_v.data(), q_out_bytes);

    auto q_result = compare_fp16_output(gpu_q, expected_q, z_rows);
    auto k_result = compare_fp16_output(gpu_k, expected_k, z_rows);
    auto v_result = compare_fp16_output(gpu_v, expected_v, z_rows);

    dev.destroy_buffer(control_buf);
    dev.destroy_buffer(qkv_buf);
    dev.destroy_buffer(conv_state_buf);
    dev.destroy_buffer(delta_conv_buf);
    dev.destroy_buffer(output_q_buf);
    dev.destroy_buffer(output_k_buf);
    dev.destroy_buffer(output_v_buf);
    dev.destroy_buffer(dummy7_buf);
    dev.destroy_buffer(dummy8_buf);
    dev.destroy_buffer(dummy9_buf);
    dev.destroy_pipeline(pipeline);
    dev.destroy_shader_module(shader);
    dev.destroy_pipeline_layout(pipe_layout);
    dev.destroy_descriptor_set_layout(desc_layout);

    constexpr uint32_t expected_generation = 1;
    auto within_tolerance = [&](const ProjectionCompareResult& r) -> bool {
      return r.max_fp16_ulp <= conv_l2_fp16_ulp_tolerance;
    };
    bool ok = failures == 0 && arrived == 0 && generation == expected_generation &&
              within_tolerance(q_result) &&
              within_tolerance(k_result) &&
              within_tolerance(v_result);
    std::cout << "{\n";
    std::cout << "  \"probe\": \"persistent_layer0_conv_l2\",\n";
    std::cout << "  \"mode\": \"conv-l2\",\n";
    std::cout << "  \"status\": \"" << (ok ? "ok" : "fail") << "\",\n";
    std::cout << "  \"failures\": " << failures << ",\n";
    std::cout << "  \"arrived\": " << arrived << ",\n";
    std::cout << "  \"generation\": " << generation << ",\n";
    std::cout << "  \"expected_generation\": " << expected_generation << ",\n";
    std::cout << "  \"q_exact_mismatches\": " << q_result.exact_mismatches << ",\n";
    std::cout << "  \"q_max_fp16_ulp\": " << q_result.max_fp16_ulp << ",\n";
    std::cout << "  \"k_exact_mismatches\": " << k_result.exact_mismatches << ",\n";
    std::cout << "  \"k_max_fp16_ulp\": " << k_result.max_fp16_ulp << ",\n";
    std::cout << "  \"v_exact_mismatches\": " << v_result.exact_mismatches << ",\n";
    std::cout << "  \"v_max_fp16_ulp\": " << v_result.max_fp16_ulp << ",\n";
    std::cout << "  \"tolerance\": " << conv_l2_fp16_ulp_tolerance << "\n";
    std::cout << "}\n";
    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::cout << "{\n";
    std::cout << "  \"probe\": \"persistent_layer0_conv_l2\",\n";
    std::cout << "  \"mode\": \"conv-l2\",\n";
    std::cout << "  \"status\": \"error\",\n";
    std::cout << "  \"message\": \"" << e.what() << "\"\n";
    std::cout << "}\n";
    return 2;
  }
#else
  return json_error("Vulkan not available (SPOCK_VULKAN_STUB)");
#endif
}

int run_projection_mode(
    int /*argc*/, char** /*argv*/,
    uint32_t hidden, uint32_t qkv_rows, uint32_t z_rows, uint32_t ab_rows,
    uint32_t workgroups, const std::string& repack_dir,
    const std::string& input_norm_fp16_file,
    const std::string& expected_qkv_raw_fp16_file,
    const std::string& expected_z_fp16_file,
    const std::string& expected_a_fp16_file,
    const std::string& expected_b_fp16_file,
    uint32_t projection_fp16_ulp_tolerance) {

  // Validate projection-specific args.
  if (input_norm_fp16_file.empty()) {
    return json_error("--input-norm-fp16-file is required for projection mode");
  }
  if (expected_qkv_raw_fp16_file.empty()) {
    return json_error("--expected-qkv-raw-fp16-file is required for projection mode");
  }
  if (expected_z_fp16_file.empty()) {
    return json_error("--expected-z-fp16-file is required for projection mode");
  }
  if (expected_a_fp16_file.empty()) {
    return json_error("--expected-a-fp16-file is required for projection mode");
  }
  if (expected_b_fp16_file.empty()) {
    return json_error("--expected-b-fp16-file is required for projection mode");
  }

  // --- Load input ---
  std::vector<uint16_t> input_norm_data;
  try {
    input_norm_data = load_fp16_file(input_norm_fp16_file, hidden);
  } catch (const std::exception& e) {
    return json_error(std::string("input norm load failed: ") + e.what());
  }

  // --- Load projection weights ---
  std::vector<uint16_t> weight_qkv_data;
  std::vector<uint16_t> weight_z_data;
  std::vector<uint16_t> weight_a_data;
  std::vector<uint16_t> weight_b_data;
  try {
    auto artifact = spock::runtime::WeightArtifact::load(repack_dir);
    weight_qkv_data = load_weight_slice(artifact, "layer.0.delta_in_proj_qkv", qkv_rows, hidden);
    weight_z_data = load_weight_slice(artifact, "layer.0.delta_in_proj_z", z_rows, hidden);
    weight_a_data = load_weight_slice(artifact, "layer.0.delta_in_proj_a", ab_rows, hidden);
    weight_b_data = load_weight_slice(artifact, "layer.0.delta_in_proj_b", ab_rows, hidden);
  } catch (const std::exception& e) {
    return json_error(std::string("weight loading failed: ") + e.what());
  }

  // --- Load expected outputs ---
  std::vector<uint16_t> expected_qkv_raw;
  std::vector<uint16_t> expected_z;
  std::vector<uint16_t> expected_a;
  std::vector<uint16_t> expected_b;
  try {
    expected_qkv_raw = load_fp16_file(expected_qkv_raw_fp16_file, qkv_rows);
    expected_z = load_fp16_file(expected_z_fp16_file, z_rows);
    expected_a = load_fp16_file(expected_a_fp16_file, ab_rows);
    expected_b = load_fp16_file(expected_b_fp16_file, ab_rows);
  } catch (const std::exception& e) {
    return json_error(std::string("expected output load failed: ") + e.what());
  }

#if SPOCK_HAS_VULKAN && !defined(SPOCK_VULKAN_STUB)
  try {
    spock::runtime::VulkanDevice dev;
    dev.initialize();

    // --- Descriptor set layout: 10 storage buffers ---
    constexpr int num_bindings = 10;
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

    // --- Pipeline layout: 20-byte push constants (5 x uint32) ---
    VkPipelineLayout pipe_layout =
        dev.create_pipeline_layout(desc_layout, 5 * sizeof(std::uint32_t));

    // --- Shader + pipeline ---
    auto spirv = read_spirv();
    VkShaderModule shader = dev.create_shader_module(spirv);
    VkPipeline pipeline = dev.create_compute_pipeline(shader, pipe_layout);

    struct PushConsts {
      std::uint32_t workgroup_count;
      std::uint32_t hidden;
      std::uint32_t intermediate_count;  // qkv_rows in projection mode
      std::uint32_t output_rows;         // z_rows in projection mode
      std::uint32_t mode;
    };

    // --- Buffer sizes ---
    constexpr VkDeviceSize control_size = 4 * sizeof(std::uint32_t);
    const std::uint32_t zero_init[4] = {0, 0, 0, 0};

    VkDeviceSize weight_qkv_size =
        static_cast<VkDeviceSize>(qkv_rows) * hidden * sizeof(std::uint16_t);
    VkDeviceSize weight_z_size =
        static_cast<VkDeviceSize>(z_rows) * hidden * sizeof(std::uint16_t);
    VkDeviceSize weight_a_size =
        static_cast<VkDeviceSize>(ab_rows) * hidden * sizeof(std::uint16_t);
    VkDeviceSize weight_b_size =
        static_cast<VkDeviceSize>(ab_rows) * hidden * sizeof(std::uint16_t);
    VkDeviceSize input_size =
        static_cast<VkDeviceSize>(hidden) * sizeof(std::uint16_t);
    VkDeviceSize output_qkv_size =
        static_cast<VkDeviceSize>(qkv_rows) * sizeof(std::uint16_t);
    VkDeviceSize output_z_size =
        static_cast<VkDeviceSize>(z_rows) * sizeof(std::uint16_t);
    VkDeviceSize output_a_size =
        static_cast<VkDeviceSize>(ab_rows) * sizeof(std::uint16_t);
    VkDeviceSize output_b_size =
        static_cast<VkDeviceSize>(ab_rows) * sizeof(std::uint16_t);

    // --- Create buffers ---
    auto control_buf = dev.create_device_local_buffer(control_size);
    dev.upload_to_device(control_buf, zero_init, control_size);

    auto weight_qkv_buf = dev.create_device_local_buffer(weight_qkv_size);
    dev.upload_to_device(weight_qkv_buf, weight_qkv_data.data(), weight_qkv_size);

    auto weight_z_buf = dev.create_device_local_buffer(weight_z_size);
    dev.upload_to_device(weight_z_buf, weight_z_data.data(), weight_z_size);

    auto weight_a_buf = dev.create_device_local_buffer(weight_a_size);
    dev.upload_to_device(weight_a_buf, weight_a_data.data(), weight_a_size);

    auto weight_b_buf = dev.create_device_local_buffer(weight_b_size);
    dev.upload_to_device(weight_b_buf, weight_b_data.data(), weight_b_size);

    auto input_buf = dev.create_device_local_buffer(input_size);
    dev.upload_to_device(input_buf, input_norm_data.data(), input_size);

    auto output_qkv_buf = dev.create_device_local_buffer(output_qkv_size);
    std::vector<std::uint16_t> output_qkv_zeros(qkv_rows, 0);
    dev.upload_to_device(output_qkv_buf, output_qkv_zeros.data(), output_qkv_size);

    auto output_z_buf = dev.create_device_local_buffer(output_z_size);
    std::vector<std::uint16_t> output_z_zeros(z_rows, 0);
    dev.upload_to_device(output_z_buf, output_z_zeros.data(), output_z_size);

    auto output_a_buf = dev.create_device_local_buffer(output_a_size);
    std::vector<std::uint16_t> output_a_zeros(ab_rows, 0);
    dev.upload_to_device(output_a_buf, output_a_zeros.data(), output_a_size);

    auto output_b_buf = dev.create_device_local_buffer(output_b_size);
    std::vector<std::uint16_t> output_b_zeros(ab_rows, 0);
    dev.upload_to_device(output_b_buf, output_b_zeros.data(), output_b_size);

    // --- Descriptor set ---
    VkDescriptorSet desc_set = dev.allocate_descriptor_set(desc_layout);
    dev.update_descriptor_set(desc_set, 0, control_buf);
    dev.update_descriptor_set(desc_set, 1, weight_qkv_buf);    // weight_qkv
    dev.update_descriptor_set(desc_set, 2, weight_z_buf);      // weight_z
    dev.update_descriptor_set(desc_set, 3, weight_a_buf);      // weight_a
    dev.update_descriptor_set(desc_set, 4, weight_b_buf);      // weight_b
    dev.update_descriptor_set(desc_set, 5, input_buf);          // input (dn_input_norm)
    dev.update_descriptor_set(desc_set, 6, output_qkv_buf);     // output_qkv_raw
    dev.update_descriptor_set(desc_set, 7, output_z_buf);       // output_z
    dev.update_descriptor_set(desc_set, 8, output_a_buf);       // output_a
    dev.update_descriptor_set(desc_set, 9, output_b_buf);       // output_b

    PushConsts push{workgroups, hidden, qkv_rows, z_rows, 1};

    // --- Dispatch ---
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

    // --- Read back control buffer ---
    std::uint32_t control_out[4] = {};
    dev.download_from_device(control_buf, control_out, control_size);

    std::uint32_t arrived = control_out[0];
    std::uint32_t generation = control_out[1];
    std::uint32_t failures = control_out[2];
    std::uint32_t checksum = control_out[3];

    // --- Download and compare outputs ---
    std::vector<std::uint16_t> gpu_qkv_raw(qkv_rows);
    dev.download_from_device(output_qkv_buf, gpu_qkv_raw.data(), output_qkv_size);

    std::vector<std::uint16_t> gpu_z(z_rows);
    dev.download_from_device(output_z_buf, gpu_z.data(), output_z_size);

    std::vector<std::uint16_t> gpu_a(ab_rows);
    dev.download_from_device(output_a_buf, gpu_a.data(), output_a_size);

    std::vector<std::uint16_t> gpu_b(ab_rows);
    dev.download_from_device(output_b_buf, gpu_b.data(), output_b_size);

    auto qkv_result = compare_fp16_output(gpu_qkv_raw, expected_qkv_raw, qkv_rows);
    auto z_result = compare_fp16_output(gpu_z, expected_z, z_rows);
    auto a_result = compare_fp16_output(gpu_a, expected_a, ab_rows);
    auto b_result = compare_fp16_output(gpu_b, expected_b, ab_rows);

    // --- Cleanup ---
    dev.destroy_buffer(control_buf);
    dev.destroy_buffer(weight_qkv_buf);
    dev.destroy_buffer(weight_z_buf);
    dev.destroy_buffer(weight_a_buf);
    dev.destroy_buffer(weight_b_buf);
    dev.destroy_buffer(input_buf);
    dev.destroy_buffer(output_qkv_buf);
    dev.destroy_buffer(output_z_buf);
    dev.destroy_buffer(output_a_buf);
    dev.destroy_buffer(output_b_buf);
    dev.destroy_pipeline(pipeline);
    dev.destroy_shader_module(shader);
    dev.destroy_pipeline_layout(pipe_layout);
    dev.destroy_descriptor_set_layout(desc_layout);

    // --- Status check ---
    // Projection mode uses no barriers, so generation stays 0.
    constexpr uint32_t expected_generation = 0;
    bool structural_ok = (failures == 0) &&
                         (generation == expected_generation) &&
                         (arrived == 0);
    auto within_tolerance = [&](const ProjectionCompareResult& r) -> bool {
      return r.max_fp16_ulp <= projection_fp16_ulp_tolerance;
    };
    bool output_ok = within_tolerance(qkv_result) && within_tolerance(z_result) &&
                     within_tolerance(a_result) && within_tolerance(b_result);
    bool ok = structural_ok && output_ok;
    std::string status = ok ? "ok" : "fail";

    // --- JSON output ---
    std::cout << "{\n";
    std::cout << "  \"probe\": \"persistent_layer0_projection_prefix\",\n";
    std::cout << "  \"mode\": \"projections\",\n";
    std::cout << "  \"hidden\": " << hidden << ",\n";
    std::cout << "  \"qkv_rows\": " << qkv_rows << ",\n";
    std::cout << "  \"z_rows\": " << z_rows << ",\n";
    std::cout << "  \"ab_rows\": " << ab_rows << ",\n";
    std::cout << "  \"local_size_x\": 128,\n";
    std::cout << "  \"workgroups\": " << workgroups << ",\n";
    std::cout << "  \"layer\": 0,\n";
    std::cout << "  \"repack_dir\": \"" << repack_dir << "\",\n";
    std::cout << "  \"status\": \"" << status << "\",\n";
    std::cout << "  \"failures\": " << failures << ",\n";
    std::cout << "  \"arrived\": " << arrived << ",\n";
    std::cout << "  \"generation\": " << generation << ",\n";
    std::cout << "  \"expected_generation\": " << expected_generation << ",\n";
    std::cout << "  \"projection_fp16_ulp_tolerance\": " << projection_fp16_ulp_tolerance << ",\n";
    std::cout << "  \"qkv_raw_exact_mismatches\": " << qkv_result.exact_mismatches << ",\n";
    std::cout << "  \"qkv_raw_max_fp16_ulp\": " << qkv_result.max_fp16_ulp << ",\n";
    std::cout << "  \"z_exact_mismatches\": " << z_result.exact_mismatches << ",\n";
    std::cout << "  \"z_max_fp16_ulp\": " << z_result.max_fp16_ulp << ",\n";
    std::cout << "  \"a_exact_mismatches\": " << a_result.exact_mismatches << ",\n";
    std::cout << "  \"a_max_fp16_ulp\": " << a_result.max_fp16_ulp << ",\n";
    std::cout << "  \"b_exact_mismatches\": " << b_result.exact_mismatches << ",\n";
    std::cout << "  \"b_max_fp16_ulp\": " << b_result.max_fp16_ulp << "\n";
    std::cout << "}\n";

    if (!ok) return 1;
    return 0;

  } catch (const std::exception& e) {
    std::cout << "{\n";
    std::cout << "  \"probe\": \"persistent_layer0_projection_prefix\",\n";
    std::cout << "  \"status\": \"error\",\n";
    std::cout << "  \"message\": \"" << e.what() << "\"\n";
    std::cout << "}\n";
    return 2;
  }
#else
  return json_error("Vulkan not available (SPOCK_VULKAN_STUB)");
#endif


}  // closes run_projection_mode

int run_recurrent_mode(
    uint32_t workgroups,
    const std::string& q_fp16_file,
    const std::string& k_fp16_file,
    const std::string& v_fp16_file,
    const std::string& g_beta_bits_file,
    const std::string& state_pre_f32_file,
    const std::string& expected_output_fp16_file) {
  constexpr uint32_t num_heads = 16;
  constexpr uint32_t head_dim = 128;
  constexpr uint32_t qkv_total = num_heads * head_dim;
  constexpr uint32_t state_matrix = num_heads * head_dim * head_dim;
  constexpr uint32_t state_with_tail = state_matrix + num_heads * 2;

  if (q_fp16_file.empty()) {
    return json_error("--q-fp16-file is required for recurrent mode");
  }
  if (k_fp16_file.empty()) {
    return json_error("--k-fp16-file is required for recurrent mode");
  }
  if (v_fp16_file.empty()) {
    return json_error("--v-fp16-file is required for recurrent mode");
  }
  if (g_beta_bits_file.empty()) {
    return json_error("--g-beta-bits-file is required for recurrent mode");
  }
  if (state_pre_f32_file.empty()) {
    return json_error("--state-pre-f32-file is required for recurrent mode");
  }
  if (expected_output_fp16_file.empty()) {
    return json_error("--expected-output-fp16-file is required for recurrent mode");
  }

  std::vector<uint16_t> q_data;
  std::vector<uint16_t> k_data;
  std::vector<uint16_t> v_data;
  std::vector<std::uint32_t> g_beta_bits;
  std::vector<float> state_data;
  std::vector<uint16_t> expected_output;
  try {
    q_data = load_fp16_file(q_fp16_file, qkv_total);
    k_data = load_fp16_file(k_fp16_file, qkv_total);
    v_data = load_fp16_file(v_fp16_file, qkv_total);
    g_beta_bits = load_u32_file(g_beta_bits_file, num_heads * 2, "--g-beta-bits-file");
    state_data = load_fp32_file(state_pre_f32_file, state_with_tail);
    expected_output = load_fp16_file(expected_output_fp16_file, qkv_total);
  } catch (const std::exception& e) {
    return json_error(std::string("recurrent load failed: ") + e.what());
  }

#if SPOCK_HAS_VULKAN && !defined(SPOCK_VULKAN_STUB)
  try {
    spock::runtime::VulkanDevice dev;
    dev.initialize();
    constexpr int num_bindings = 10;
    std::vector<VkDescriptorSetLayoutBinding> bindings(num_bindings);
    for (int b = 0; b < num_bindings; ++b) {
      bindings[b].binding = b;
      bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings[b].descriptorCount = 1;
      bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      bindings[b].pImmutableSamplers = nullptr;
    }
    VkDescriptorSetLayout desc_layout = dev.create_descriptor_set_layout(bindings);
    VkPipelineLayout pipe_layout =
        dev.create_pipeline_layout(desc_layout, 5 * sizeof(std::uint32_t));

    auto spirv = read_spirv();
    VkShaderModule shader = dev.create_shader_module(spirv);
    VkPipeline pipeline = dev.create_compute_pipeline(shader, pipe_layout);

    struct PushConsts {
      std::uint32_t workgroup_count;
      std::uint32_t hidden;
      std::uint32_t intermediate_count;
      std::uint32_t output_rows;
      std::uint32_t mode;
    };
    if (workgroups < num_heads) {
      throw std::runtime_error("--workgroups must be at least 16 for recurrent mode");
    }

    constexpr std::size_t control_arrived_offset = 0;
    constexpr std::size_t control_generation_offset = 1;
    constexpr std::size_t control_failures_offset = 2;
    constexpr std::size_t control_g_beta_offset = 4 + 16 + 16;
    constexpr std::size_t control_extra_offset = control_g_beta_offset + num_heads * 2;
    std::vector<std::uint32_t> control_payload(control_extra_offset + state_matrix, 0);
    for (std::size_t i = 0; i < num_heads * 2; ++i) {
      control_payload[control_g_beta_offset + i] = g_beta_bits[i];
    }
    for (std::size_t i = 0; i < state_matrix; ++i) {
      std::uint32_t bits = 0;
      std::memcpy(&bits, &state_data[i], sizeof(bits));
      control_payload[control_extra_offset + i] = bits;
    }

    VkDeviceSize qkv_bytes = static_cast<VkDeviceSize>(qkv_total) * sizeof(std::uint16_t);
    VkDeviceSize control_bytes =
        static_cast<VkDeviceSize>(control_payload.size()) * sizeof(std::uint32_t);
    VkDeviceSize dummy_bytes = sizeof(std::uint16_t);
    std::uint16_t dummy_zero = 0;

    auto control_buf = dev.create_device_local_buffer(control_bytes);
    dev.upload_to_device(control_buf, control_payload.data(), control_bytes);
    auto q_buf = dev.create_device_local_buffer(qkv_bytes);
    dev.upload_to_device(q_buf, q_data.data(), qkv_bytes);
    auto k_buf = dev.create_device_local_buffer(qkv_bytes);
    dev.upload_to_device(k_buf, k_data.data(), qkv_bytes);
    auto v_buf = dev.create_device_local_buffer(qkv_bytes);
    dev.upload_to_device(v_buf, v_data.data(), qkv_bytes);
    auto dummy4_buf = dev.create_device_local_buffer(dummy_bytes);
    auto dummy5_buf = dev.create_device_local_buffer(dummy_bytes);
    auto output_buf = dev.create_device_local_buffer(qkv_bytes);
    std::vector<std::uint16_t> output_zero(qkv_total, 0);
    dev.upload_to_device(output_buf, output_zero.data(), qkv_bytes);
    auto dummy7_buf = dev.create_device_local_buffer(dummy_bytes);
    auto dummy8_buf = dev.create_device_local_buffer(dummy_bytes);
    auto dummy9_buf = dev.create_device_local_buffer(dummy_bytes);
    dev.upload_to_device(dummy4_buf, &dummy_zero, dummy_bytes);
    dev.upload_to_device(dummy5_buf, &dummy_zero, dummy_bytes);
    dev.upload_to_device(dummy7_buf, &dummy_zero, dummy_bytes);
    dev.upload_to_device(dummy8_buf, &dummy_zero, dummy_bytes);
    dev.upload_to_device(dummy9_buf, &dummy_zero, dummy_bytes);

    VkDescriptorSet desc_set = dev.allocate_descriptor_set(desc_layout);
    dev.update_descriptor_set(desc_set, 0, control_buf);
    dev.update_descriptor_set(desc_set, 1, q_buf);
    dev.update_descriptor_set(desc_set, 2, k_buf);
    dev.update_descriptor_set(desc_set, 3, v_buf);
    dev.update_descriptor_set(desc_set, 4, dummy4_buf);
    dev.update_descriptor_set(desc_set, 5, dummy5_buf);
    dev.update_descriptor_set(desc_set, 6, output_buf);
    dev.update_descriptor_set(desc_set, 7, dummy7_buf);
    dev.update_descriptor_set(desc_set, 8, dummy8_buf);
    dev.update_descriptor_set(desc_set, 9, dummy9_buf);

    PushConsts push{workgroups, 0, 0, 0, 4};
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

    std::vector<std::uint32_t> control_out(control_payload.size(), 0);
    dev.download_from_device(control_buf, control_out.data(), control_bytes);
    std::uint32_t arrived = control_out[control_arrived_offset];
    std::uint32_t generation = control_out[control_generation_offset];
    std::uint32_t failures = control_out[control_failures_offset];
    std::vector<std::uint16_t> gpu_output(qkv_total);
    dev.download_from_device(output_buf, gpu_output.data(), qkv_bytes);
    auto output_result = compare_fp16_output(gpu_output, expected_output, qkv_total);

    dev.destroy_buffer(control_buf);
    dev.destroy_buffer(q_buf);
    dev.destroy_buffer(k_buf);
    dev.destroy_buffer(v_buf);
    dev.destroy_buffer(dummy4_buf);
    dev.destroy_buffer(dummy5_buf);
    dev.destroy_buffer(output_buf);
    dev.destroy_buffer(dummy7_buf);
    dev.destroy_buffer(dummy8_buf);
    dev.destroy_buffer(dummy9_buf);
    dev.destroy_pipeline(pipeline);
    dev.destroy_shader_module(shader);
    dev.destroy_pipeline_layout(pipe_layout);
    dev.destroy_descriptor_set_layout(desc_layout);

    constexpr uint32_t expected_generation = 0;
    bool ok = failures == 0 &&
              arrived == 0 &&
              generation == expected_generation &&
              output_result.exact_mismatches == 0;
    std::cout << "{\n";
    std::cout << "  \"probe\": \"persistent_layer0_recurrent\",\n";
    std::cout << "  \"mode\": \"recurrent\",\n";
    std::cout << "  \"status\": \"" << (ok ? "ok" : "fail") << "\",\n";
    std::cout << "  \"failures\": " << failures << ",\n";
    std::cout << "  \"arrived\": " << arrived << ",\n";
    std::cout << "  \"generation\": " << generation << ",\n";
    std::cout << "  \"expected_generation\": " << expected_generation << ",\n";
    std::cout << "  \"output_exact_mismatches\": " << output_result.exact_mismatches << ",\n";
    std::cout << "  \"output_max_fp16_ulp\": " << output_result.max_fp16_ulp << "\n";
    std::cout << "}\n";
    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::cout << "{\n";
    std::cout << "  \"probe\": \"persistent_layer0_recurrent\",\n";
    std::cout << "  \"mode\": \"recurrent\",\n";
    std::cout << "  \"status\": \"error\",\n";
    std::cout << "  \"message\": \"" << e.what() << "\"\n";
    std::cout << "}\n";
    return 2;
  }
#else
  return json_error("Vulkan not available (SPOCK_VULKAN_STUB)");
#endif
}

int run_mixer_tail_mode(
    uint32_t workgroups,
    const std::string& repack_dir,
    const std::string& core_fp16_file,
    const std::string& z_fp16_file,
    const std::string& input_hidden_fp16_file,
    const std::string& expected_mixer_output_fp16_file,
    const std::string& expected_mixer_residual_fp16_file,
    uint32_t mixer_tail_fp16_ulp_tolerance) {
  constexpr uint32_t hidden = 1024;
  constexpr uint32_t dn_dim = 2048;
  constexpr uint32_t num_heads = 16;

  if (repack_dir.empty()) {
    return json_error("--repack-dir is required for mixer-tail mode");
  }
  if (core_fp16_file.empty()) {
    return json_error("--core-fp16-file is required for mixer-tail mode");
  }
  if (z_fp16_file.empty()) {
    return json_error("--z-fp16-file is required for mixer-tail mode");
  }
  if (input_hidden_fp16_file.empty()) {
    return json_error("--input-hidden-fp16-file is required for mixer-tail mode");
  }
  if (expected_mixer_output_fp16_file.empty()) {
    return json_error("--expected-mixer-output-fp16-file is required for mixer-tail mode");
  }
  if (expected_mixer_residual_fp16_file.empty()) {
    return json_error("--expected-mixer-residual-fp16-file is required for mixer-tail mode");
  }

  std::vector<uint16_t> core_data;
  std::vector<uint16_t> z_data;
  std::vector<uint16_t> input_hidden_data;
  std::vector<uint16_t> expected_mixer_output;
  std::vector<uint16_t> expected_mixer_residual;
  std::vector<std::uint32_t> delta_norm_bits;
  std::vector<uint16_t> delta_out_proj;
  try {
    core_data = load_fp16_file(core_fp16_file, dn_dim);
    z_data = load_fp16_file(z_fp16_file, dn_dim);
    input_hidden_data = load_fp16_file(input_hidden_fp16_file, hidden);
    expected_mixer_output = load_fp16_file(expected_mixer_output_fp16_file, hidden);
    expected_mixer_residual = load_fp16_file(expected_mixer_residual_fp16_file, hidden);
    auto artifact = spock::runtime::WeightArtifact::load(repack_dir);
    delta_norm_bits = load_fp32_weight_bits_vector(artifact, "layer.0.delta_norm", 128);
    delta_out_proj = load_weight_slice(artifact, "layer.0.delta_out_proj", hidden, dn_dim);
  } catch (const std::exception& e) {
    return json_error(std::string("mixer-tail load failed: ") + e.what());
  }

#if SPOCK_HAS_VULKAN && !defined(SPOCK_VULKAN_STUB)
  try {
    if (workgroups < num_heads) {
      throw std::runtime_error("--workgroups must be at least 16 for mixer-tail mode");
    }

    spock::runtime::VulkanDevice dev;
    dev.initialize();
    constexpr int num_bindings = 10;
    std::vector<VkDescriptorSetLayoutBinding> bindings(num_bindings);
    for (int b = 0; b < num_bindings; ++b) {
      bindings[b].binding = b;
      bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings[b].descriptorCount = 1;
      bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      bindings[b].pImmutableSamplers = nullptr;
    }
    VkDescriptorSetLayout desc_layout = dev.create_descriptor_set_layout(bindings);
    VkPipelineLayout pipe_layout =
        dev.create_pipeline_layout(desc_layout, 5 * sizeof(std::uint32_t));

    auto spirv = read_spirv();
    VkShaderModule shader = dev.create_shader_module(spirv);
    VkPipeline pipeline = dev.create_compute_pipeline(shader, pipe_layout);

    struct PushConsts {
      std::uint32_t workgroup_count;
      std::uint32_t hidden;
      std::uint32_t intermediate_count;
      std::uint32_t output_rows;
      std::uint32_t mode;
    };

    constexpr std::size_t control_arrived_offset = 0;
    constexpr std::size_t control_generation_offset = 1;
    constexpr std::size_t control_failures_offset = 2;
    constexpr std::size_t control_extra_offset = 4 + 16 + 16 + 32;
    std::vector<std::uint32_t> control_payload(control_extra_offset + 128, 0);
    for (std::size_t i = 0; i < 128; ++i) {
      control_payload[control_extra_offset + i] = delta_norm_bits[i];
    }

    VkDeviceSize control_bytes =
        static_cast<VkDeviceSize>(control_payload.size()) * sizeof(std::uint32_t);
    VkDeviceSize dn_bytes = static_cast<VkDeviceSize>(dn_dim) * sizeof(std::uint16_t);
    VkDeviceSize hidden_bytes = static_cast<VkDeviceSize>(hidden) * sizeof(std::uint16_t);
    VkDeviceSize out_proj_bytes =
        static_cast<VkDeviceSize>(hidden) * dn_dim * sizeof(std::uint16_t);
    VkDeviceSize dummy_bytes = sizeof(std::uint16_t);
    std::uint16_t dummy_zero = 0;

    auto control_buf = dev.create_device_local_buffer(control_bytes);
    dev.upload_to_device(control_buf, control_payload.data(), control_bytes);
    auto core_buf = dev.create_device_local_buffer(dn_bytes);
    dev.upload_to_device(core_buf, core_data.data(), dn_bytes);
    auto z_buf = dev.create_device_local_buffer(dn_bytes);
    dev.upload_to_device(z_buf, z_data.data(), dn_bytes);
    auto input_hidden_buf = dev.create_device_local_buffer(hidden_bytes);
    dev.upload_to_device(input_hidden_buf, input_hidden_data.data(), hidden_bytes);
    auto out_proj_buf = dev.create_device_local_buffer(out_proj_bytes);
    dev.upload_to_device(out_proj_buf, delta_out_proj.data(), out_proj_bytes);
    auto mixer_output_buf = dev.create_device_local_buffer(hidden_bytes);
    auto mixer_residual_buf = dev.create_device_local_buffer(hidden_bytes);
    std::vector<std::uint16_t> hidden_zero(hidden, 0);
    dev.upload_to_device(mixer_output_buf, hidden_zero.data(), hidden_bytes);
    dev.upload_to_device(mixer_residual_buf, hidden_zero.data(), hidden_bytes);
    auto dummy7_buf = dev.create_device_local_buffer(dummy_bytes);
    auto dummy8_buf = dev.create_device_local_buffer(dummy_bytes);
    auto dummy9_buf = dev.create_device_local_buffer(dummy_bytes);
    dev.upload_to_device(dummy7_buf, &dummy_zero, dummy_bytes);
    dev.upload_to_device(dummy8_buf, &dummy_zero, dummy_bytes);
    dev.upload_to_device(dummy9_buf, &dummy_zero, dummy_bytes);

    VkDescriptorSet desc_set = dev.allocate_descriptor_set(desc_layout);
    dev.update_descriptor_set(desc_set, 0, control_buf);
    dev.update_descriptor_set(desc_set, 1, core_buf);
    dev.update_descriptor_set(desc_set, 2, z_buf);
    dev.update_descriptor_set(desc_set, 3, input_hidden_buf);
    dev.update_descriptor_set(desc_set, 4, out_proj_buf);
    dev.update_descriptor_set(desc_set, 5, mixer_output_buf);
    dev.update_descriptor_set(desc_set, 6, mixer_residual_buf);
    dev.update_descriptor_set(desc_set, 7, dummy7_buf);
    dev.update_descriptor_set(desc_set, 8, dummy8_buf);
    dev.update_descriptor_set(desc_set, 9, dummy9_buf);

    PushConsts push{workgroups, 0, 0, 0, 5};
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

    std::vector<std::uint32_t> control_out(control_payload.size(), 0);
    dev.download_from_device(control_buf, control_out.data(), control_bytes);
    std::uint32_t arrived = control_out[control_arrived_offset];
    std::uint32_t generation = control_out[control_generation_offset];
    std::uint32_t failures = control_out[control_failures_offset];
    std::vector<std::uint16_t> gpu_mixer_output(hidden);
    std::vector<std::uint16_t> gpu_mixer_residual(hidden);
    dev.download_from_device(mixer_output_buf, gpu_mixer_output.data(), hidden_bytes);
    dev.download_from_device(mixer_residual_buf, gpu_mixer_residual.data(), hidden_bytes);

    auto mixer_output_result =
        compare_fp16_output(gpu_mixer_output, expected_mixer_output, hidden);
    auto mixer_residual_result =
        compare_fp16_output(gpu_mixer_residual, expected_mixer_residual, hidden);

    dev.destroy_buffer(control_buf);
    dev.destroy_buffer(core_buf);
    dev.destroy_buffer(z_buf);
    dev.destroy_buffer(input_hidden_buf);
    dev.destroy_buffer(out_proj_buf);
    dev.destroy_buffer(mixer_output_buf);
    dev.destroy_buffer(mixer_residual_buf);
    dev.destroy_buffer(dummy7_buf);
    dev.destroy_buffer(dummy8_buf);
    dev.destroy_buffer(dummy9_buf);
    dev.destroy_pipeline(pipeline);
    dev.destroy_shader_module(shader);
    dev.destroy_pipeline_layout(pipe_layout);
    dev.destroy_descriptor_set_layout(desc_layout);

    constexpr uint32_t expected_generation = 2;
    bool ok = failures == 0 &&
              arrived == 0 &&
              generation == expected_generation &&
              mixer_output_result.max_fp16_ulp <= mixer_tail_fp16_ulp_tolerance &&
              mixer_residual_result.max_fp16_ulp <= mixer_tail_fp16_ulp_tolerance;
    std::cout << "{\n";
    std::cout << "  \"probe\": \"persistent_layer0_mixer_tail\",\n";
    std::cout << "  \"mode\": \"mixer-tail\",\n";
    std::cout << "  \"status\": \"" << (ok ? "ok" : "fail") << "\",\n";
    std::cout << "  \"failures\": " << failures << ",\n";
    std::cout << "  \"arrived\": " << arrived << ",\n";
    std::cout << "  \"generation\": " << generation << ",\n";
    std::cout << "  \"expected_generation\": " << expected_generation << ",\n";
    std::cout << "  \"mixer_tail_fp16_ulp_tolerance\": " << mixer_tail_fp16_ulp_tolerance << ",\n";
    std::cout << "  \"mixer_output_exact_mismatches\": " << mixer_output_result.exact_mismatches << ",\n";
    std::cout << "  \"mixer_output_max_fp16_ulp\": " << mixer_output_result.max_fp16_ulp << ",\n";
    std::cout << "  \"mixer_residual_exact_mismatches\": " << mixer_residual_result.exact_mismatches << ",\n";
    std::cout << "  \"mixer_residual_max_fp16_ulp\": " << mixer_residual_result.max_fp16_ulp << "\n";
    std::cout << "}\n";
    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::cout << "{\n";
    std::cout << "  \"probe\": \"persistent_layer0_mixer_tail\",\n";
    std::cout << "  \"mode\": \"mixer-tail\",\n";
    std::cout << "  \"status\": \"error\",\n";
    std::cout << "  \"message\": \"" << e.what() << "\"\n";
    std::cout << "}\n";
    return 2;
  }
#else
  return json_error("Vulkan not available (SPOCK_VULKAN_STUB)");
#endif
}
}  // namespace

int main(int argc, char** argv) {
  // Fixed dimensions for layer 0.
  constexpr uint32_t hidden = 1024;
  constexpr uint32_t intermediate = 3584;
  constexpr uint32_t output_rows = 1024;
  // Projection-mode dimensions.
  constexpr uint32_t qkv_rows = 6144;
  constexpr uint32_t z_rows = 2048;
  constexpr uint32_t ab_rows = 16;

  uint32_t workgroups = 82;
  uint32_t mode = 0;  // 0 = tail, 1 = projections, 2 = conv-l2, 3 = g-beta, 4 = recurrent, 5 = mixer-tail
  std::string repack_dir;

  // Tail-mode args.
  std::string input_fp16_file;
  std::string expected_output_fp16_file;
  uint32_t output_fp16_ulp_tolerance = 0;
  uint32_t output_population_ulp_threshold = 0;
  bool output_population_ulp_threshold_set = false;
  uint32_t output_population_max_rows = 0;
  bool output_population_max_rows_set = false;

  // Projection-mode args.
  std::string input_norm_fp16_file;
  std::string expected_qkv_raw_fp16_file;
  std::string expected_z_fp16_file;
  std::string expected_a_fp16_file;
  std::string expected_b_fp16_file;
  uint32_t projection_fp16_ulp_tolerance = 0;

  // Conv/L2 mode args.
  std::string raw_qkv_fp16_file;
  std::string conv_state_pre_fp16_file;
  std::string expected_q_fp16_file;
  std::string expected_k_fp16_file;
  std::string expected_v_fp16_file;
  uint32_t conv_l2_fp16_ulp_tolerance = 0;

  // G/Beta mode args.
  std::string a_fp16_file;
  std::string b_fp16_file;
  std::string expected_g_beta_bits_file;

  // Recurrent mode args.
  std::string q_fp16_file;
  std::string k_fp16_file;
  std::string v_fp16_file;
  std::string g_beta_bits_file;
  std::string state_pre_f32_file;

  // Mixer-tail mode args.
  std::string core_fp16_file;
  std::string z_fp16_file;
  std::string input_hidden_fp16_file;
  std::string expected_mixer_output_fp16_file;
  std::string expected_mixer_residual_fp16_file;
  uint32_t mixer_tail_fp16_ulp_tolerance = 0;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--repack-dir" && i + 1 < argc) {
      repack_dir = argv[++i];
    } else if (arg == "--mode" && i + 1 < argc) {
      const std::string mode_str = argv[++i];
      if (mode_str == "tail") {
        mode = 0;
      } else if (mode_str == "projections") {
        mode = 1;
      } else if (mode_str == "conv-l2") {
        mode = 2;
      } else if (mode_str == "g-beta") {
        mode = 3;
      } else if (mode_str == "recurrent") {
        mode = 4;
      } else if (mode_str == "mixer-tail") {
        mode = 5;
      } else {
        return json_error("--mode must be 'tail' or 'projections' or 'conv-l2' or 'g-beta' or 'recurrent' or 'mixer-tail', got: " + mode_str);
      }
    } else if (arg == "--input-fp16-file" && i + 1 < argc) {
      input_fp16_file = argv[++i];
    } else if (arg == "--expected-output-fp16-file" && i + 1 < argc) {
      expected_output_fp16_file = argv[++i];
    } else if (arg == "--input-norm-fp16-file" && i + 1 < argc) {
      input_norm_fp16_file = argv[++i];
    } else if (arg == "--expected-qkv-raw-fp16-file" && i + 1 < argc) {
      expected_qkv_raw_fp16_file = argv[++i];
    } else if (arg == "--expected-z-fp16-file" && i + 1 < argc) {
      expected_z_fp16_file = argv[++i];
    } else if (arg == "--expected-a-fp16-file" && i + 1 < argc) {
      expected_a_fp16_file = argv[++i];
    } else if (arg == "--expected-b-fp16-file" && i + 1 < argc) {
      expected_b_fp16_file = argv[++i];
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
    } else if (arg == "--a-fp16-file" && i + 1 < argc) {
      a_fp16_file = argv[++i];
    } else if (arg == "--b-fp16-file" && i + 1 < argc) {
      b_fp16_file = argv[++i];
    } else if (arg == "--expected-g-beta-bits-file" && i + 1 < argc) {
      expected_g_beta_bits_file = argv[++i];
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
    } else if (arg == "--core-fp16-file" && i + 1 < argc) {
      core_fp16_file = argv[++i];
    } else if (arg == "--z-fp16-file" && i + 1 < argc) {
      z_fp16_file = argv[++i];
    } else if (arg == "--input-hidden-fp16-file" && i + 1 < argc) {
      input_hidden_fp16_file = argv[++i];
    } else if (arg == "--expected-mixer-output-fp16-file" && i + 1 < argc) {
      expected_mixer_output_fp16_file = argv[++i];
    } else if (arg == "--expected-mixer-residual-fp16-file" && i + 1 < argc) {
      expected_mixer_residual_fp16_file = argv[++i];
    } else if (arg == "--projection-fp16-ulp-tolerance" && i + 1 < argc) {
      const std::string value = argv[++i];
      std::string err = parse_u32_option("--projection-fp16-ulp-tolerance", value,
                                          &projection_fp16_ulp_tolerance);
      if (!err.empty()) return json_error(err);
    } else if (arg == "--conv-l2-fp16-ulp-tolerance" && i + 1 < argc) {
      const std::string value = argv[++i];
      std::string err = parse_u32_option("--conv-l2-fp16-ulp-tolerance", value,
                                          &conv_l2_fp16_ulp_tolerance);
      if (!err.empty()) return json_error(err);
    } else if (arg == "--mixer-tail-fp16-ulp-tolerance" && i + 1 < argc) {
      const std::string value = argv[++i];
      std::string err = parse_u32_option("--mixer-tail-fp16-ulp-tolerance", value,
                                          &mixer_tail_fp16_ulp_tolerance);
      if (!err.empty()) return json_error(err);
    } else if (arg == "--workgroups" && i + 1 < argc) {
      const std::string value = argv[++i];
      std::string err = parse_u32_option("--workgroups", value, &workgroups);
      if (!err.empty()) return json_error(err);
    } else if (arg == "--output-fp16-ulp-tolerance" && i + 1 < argc) {
      const std::string value = argv[++i];
      std::string err = parse_u32_option("--output-fp16-ulp-tolerance", value,
                                          &output_fp16_ulp_tolerance);
      if (!err.empty()) return json_error(err);
    } else if (arg == "--output-fp16-population-ulp-threshold" && i + 1 < argc) {
      const std::string value = argv[++i];
      std::string err = parse_u32_option("--output-fp16-population-ulp-threshold",
                                          value, &output_population_ulp_threshold);
      if (!err.empty()) return json_error(err);
      output_population_ulp_threshold_set = true;
    } else if (arg == "--output-fp16-max-rows-above-population-threshold" && i + 1 < argc) {
      const std::string value = argv[++i];
      std::string err = parse_u32_option("--output-fp16-max-rows-above-population-threshold",
                                          value, &output_population_max_rows);
      if (!err.empty()) return json_error(err);
      output_population_max_rows_set = true;
    } else if (arg == "--help") {
      std::cout << "usage: vk_persistent_layer0_probe [options]\n";
      std::cout << "  --mode tail|projections|conv-l2|g-beta|recurrent|mixer-tail   probe mode (default: tail)\n";
      std::cout << "  --repack-dir DIR   load real fp16 weights from repacked model artifact (required for tail/projections/conv-l2/g-beta)\n";
      std::cout << "\n  Tail mode options:\n";
      std::cout << "  --input-fp16-file PATH  load mixer_residual fp16 input (required)\n";
      std::cout << "  --expected-output-fp16-file PATH  load expected post_mlp fp16 output for comparison\n";
      std::cout << "  --output-fp16-ulp-tolerance N  allow up to N fp16 ULP diff (default 0, exact)\n";
      std::cout << "  --output-fp16-population-ulp-threshold N  count output rows with ULP diff above N\n";
      std::cout << "  --output-fp16-max-rows-above-population-threshold N  fail when rows above threshold exceed N\n";
      std::cout << "\n  Projection mode options:\n";
      std::cout << "  --projection-fp16-ulp-tolerance N  allow up to N fp16 ULP diff for all projection outputs (default 0, exact)\n";
      std::cout << "  --input-norm-fp16-file PATH  load dn_input_norm fp16 input\n";
      std::cout << "  --expected-qkv-raw-fp16-file PATH  expected qkv_raw output\n";
      std::cout << "  --expected-z-fp16-file PATH  expected z output\n";
      std::cout << "  --expected-a-fp16-file PATH  expected a output\n";
      std::cout << "  --expected-b-fp16-file PATH  expected b output\n";
      std::cout << "\n  Conv/L2 mode options:\n";
      std::cout << "  --raw-qkv-fp16-file PATH  load raw qkv fp16 input\n";
      std::cout << "  --conv-state-pre-fp16-file PATH  load pre-conv rolling state\n";
      std::cout << "  --expected-q-fp16-file PATH  expected normalized q output\n";
      std::cout << "  --expected-k-fp16-file PATH  expected normalized k output\n";
      std::cout << "  --expected-v-fp16-file PATH  expected v output\n";
      std::cout << "  --conv-l2-fp16-ulp-tolerance N  allow up to N fp16 ULP diff for q/k/v outputs (default 0, exact)\n";
      std::cout << "\n  G/Beta mode options:\n";
      std::cout << "  --a-fp16-file PATH  load projected a fp16 input\n";
      std::cout << "  --b-fp16-file PATH  load projected b fp16 input\n";
      std::cout << "  --expected-g-beta-bits-file PATH  expected g/beta fp32 bit patterns as uint32, g then beta\n";
      std::cout << "\n  Recurrent mode options:\n";
      std::cout << "  --q-fp16-file PATH  load q fp16 input\n";
      std::cout << "  --k-fp16-file PATH  load k fp16 input\n";
      std::cout << "  --v-fp16-file PATH  load v fp16 input\n";
      std::cout << "  --g-beta-bits-file PATH  load g/beta fp32 bit patterns as uint32, g then beta\n";
      std::cout << "  --state-pre-f32-file PATH  load recurrent state pre-update fp32\n";
      std::cout << "  --expected-output-fp16-file PATH  expected dn_core fp16 output\n";
      std::cout << "\n  Mixer-tail mode options:\n";
      std::cout << "  --core-fp16-file PATH  load dn_core fp16 input\n";
      std::cout << "  --z-fp16-file PATH  load dn_z fp16 input\n";
      std::cout << "  --input-hidden-fp16-file PATH  load residual input hidden fp16\n";
      std::cout << "  --expected-mixer-output-fp16-file PATH  expected mixer output fp16\n";
      std::cout << "  --expected-mixer-residual-fp16-file PATH  expected mixer residual fp16\n";
      std::cout << "  --mixer-tail-fp16-ulp-tolerance N  allow up to N fp16 ULP diff for mixer output/residual (default 0, exact)\n";
      std::cout << "\n  Common options:\n";
      std::cout << "  --workgroups N     dispatch workgroup count (default 82)\n";
      std::cout << "  --help             show this help\n";
      std::cout << "\nFixed: hidden=1024, local_size_x=128, layer=0.\n";
      return 0;
    } else {
      return json_error("unknown option: " + arg);
    }
  }

  // Validate common.
  if (mode != 4 && repack_dir.empty()) {
    return json_error("--repack-dir is required");
  }
  if (workgroups == 0) {
    return json_error("--workgroups must be > 0");
  }
  if (output_population_max_rows_set && !output_population_ulp_threshold_set) {
    return json_error("--output-fp16-max-rows-above-population-threshold requires --output-fp16-population-ulp-threshold");
  }

  // Dispatch to mode-specific implementation.
  if (mode == 1) {
    return run_projection_mode(argc, argv,
                               hidden, qkv_rows, z_rows, ab_rows,
                               workgroups, repack_dir,
                               input_norm_fp16_file,
                               expected_qkv_raw_fp16_file,
                               expected_z_fp16_file,
                               expected_a_fp16_file,
                               expected_b_fp16_file,
                               projection_fp16_ulp_tolerance);
  }
  if (mode == 2) {
    return run_conv_l2_mode(qkv_rows, z_rows, workgroups, repack_dir,
                            raw_qkv_fp16_file,
                            conv_state_pre_fp16_file,
                            expected_q_fp16_file,
                            expected_k_fp16_file,
                            expected_v_fp16_file,
                            conv_l2_fp16_ulp_tolerance);
  }
  if (mode == 3) {
    return run_g_beta_mode(workgroups, repack_dir, a_fp16_file, b_fp16_file,
                           expected_g_beta_bits_file);
  }
  if (mode == 4) {
    return run_recurrent_mode(workgroups, q_fp16_file, k_fp16_file, v_fp16_file,
                              g_beta_bits_file, state_pre_f32_file,
                              expected_output_fp16_file);
  }
  if (mode == 5) {
    return run_mixer_tail_mode(workgroups, repack_dir,
                               core_fp16_file,
                               z_fp16_file,
                               input_hidden_fp16_file,
                               expected_mixer_output_fp16_file,
                               expected_mixer_residual_fp16_file,
                               mixer_tail_fp16_ulp_tolerance);
  }
  // mode == 0: tail-only path continues below.

  // Validate tail-specific args.
  if (input_fp16_file.empty()) {
    return json_error("--input-fp16-file is required for tail mode");
  }

  // --- Load input ---
  std::vector<uint16_t> input_data(hidden);
  {
    std::ifstream f(input_fp16_file, std::ios::binary | std::ios::ate);
    if (!f) {
      return json_error("cannot open --input-fp16-file: " + input_fp16_file);
    }
    auto file_size = f.tellg();
    f.seekg(0);
    auto required = static_cast<std::streamsize>(hidden) * sizeof(uint16_t);
    if (file_size < required) {
      return json_error("--input-fp16-file too small: need " + std::to_string(hidden) +
                       " x 2 = " + std::to_string(required) +
                       " bytes, got " + std::to_string(file_size));
    }
    f.read(reinterpret_cast<char*>(input_data.data()), required);
  }

  // --- Load weights ---
  std::vector<uint16_t> weight_norm_data;
  std::vector<uint16_t> weight_gate_data;
  std::vector<uint16_t> weight_up_data;
  std::vector<uint16_t> weight_down_data;

  try {
    auto artifact = spock::runtime::WeightArtifact::load(repack_dir);

    weight_norm_data = load_weight_vector(artifact, "layer.0.post_norm", hidden);
    weight_gate_data = load_weight_slice(artifact, "layer.0.mlp_gate", intermediate, hidden);
    weight_up_data = load_weight_slice(artifact, "layer.0.mlp_up", intermediate, hidden);
    weight_down_data = load_weight_slice(artifact, "layer.0.mlp_down", output_rows, intermediate);
  } catch (const std::exception& e) {
    return json_error(std::string("weight loading failed: ") + e.what());
  }

  // --- CPU reference computation (128-lane mirrors shader) ---
  // Stage 0: RMSNorm.
  std::vector<uint16_t> normalized_input(hidden);
  {
    float sum_sq = shader_mirrored_sum_sq_128(hidden, input_data.data());
    float mean_sq = sum_sq / static_cast<float>(hidden);
    float inv_rms = 1.0f / std::sqrt(mean_sq + 1e-6f);
    for (uint32_t c = 0; c < hidden; ++c) {
      float v = fp16_to_fp32(input_data[c]);
      float w = fp16_to_fp32(weight_norm_data[c]);
      normalized_input[c] = fp32_to_fp16(v * inv_rms * (1.0f + w));
    }
  }

  // Stage 1: gate/up dots.
  std::vector<uint16_t> gate_scratch(intermediate);
  std::vector<uint16_t> up_scratch(intermediate);
  for (uint32_t row = 0; row < intermediate; ++row) {
    float gate_dot = shader_mirrored_dot_128(row, hidden,
                                              weight_gate_data.data(),
                                              normalized_input.data());
    float up_dot = shader_mirrored_dot_128(row, hidden,
                                            weight_up_data.data(),
                                            normalized_input.data());
    gate_scratch[row] = fp32_to_fp16(gate_dot);
    up_scratch[row] = fp32_to_fp16(up_dot);
  }

  // Stage 2: activation.
  std::vector<uint16_t> activated(intermediate);
  for (uint32_t row = 0; row < intermediate; ++row) {
    float g = fp16_to_fp32(gate_scratch[row]);
    float u = fp16_to_fp32(up_scratch[row]);
    float act = silu(g) * u;
    activated[row] = fp32_to_fp16(act);
  }

  // Stage 3: down dots + residual add.
  std::uint32_t expected_checksum = 0;
  std::vector<uint16_t> expected_output(output_rows);
  for (uint32_t row = 0; row < output_rows; ++row) {
    float down_total = shader_mirrored_down_dot_128(
        row, intermediate,
        weight_down_data.data(),
        activated.data());
    down_total += fp16_to_fp32(input_data[row]);
    std::uint32_t bits;
    std::memcpy(&bits, &down_total, sizeof(bits));
    expected_checksum += bits;
    expected_output[row] = fp32_to_fp16(down_total);
  }

  // --- Load expected output if provided ---
  if (!expected_output_fp16_file.empty()) {
    try {
      std::ifstream f(expected_output_fp16_file, std::ios::binary | std::ios::ate);
      if (!f) {
        return json_error("cannot open --expected-output-fp16-file: " + expected_output_fp16_file);
      }
      auto file_size = f.tellg();
      f.seekg(0);
      auto required = static_cast<std::streamsize>(output_rows) * sizeof(uint16_t);
      if (file_size < required) {
        return json_error("--expected-output-fp16-file too small: need " + std::to_string(output_rows) +
                         " x 2 = " + std::to_string(required) +
                         " bytes, got " + std::to_string(file_size));
      }
      f.read(reinterpret_cast<char*>(expected_output.data()), required);
    } catch (const std::exception& e) {
      return json_error(std::string("--expected-output-fp16-file read failed: ") + e.what());
    }
  }

  // 3 global barriers (Stage0->Stage1, Stage1->Stage2, Stage2->Stage3).
  constexpr uint32_t expected_generation = 3u;

#if SPOCK_HAS_VULKAN && !defined(SPOCK_VULKAN_STUB)
  try {
    spock::runtime::VulkanDevice dev;
    dev.initialize();

    // --- Descriptor set layout: 10 storage buffers ---
    constexpr int num_bindings = 10;
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

    // --- Pipeline layout: 20-byte push constants (5 x uint32) ---
    VkPipelineLayout pipe_layout =
        dev.create_pipeline_layout(desc_layout, 5 * sizeof(std::uint32_t));

    // --- Shader + pipeline ---
    auto spirv = read_spirv();
    VkShaderModule shader = dev.create_shader_module(spirv);
    VkPipeline pipeline = dev.create_compute_pipeline(shader, pipe_layout);

    struct PushConsts {
      std::uint32_t workgroup_count;
      std::uint32_t hidden;
      std::uint32_t intermediate_count;
      std::uint32_t output_rows;
      std::uint32_t mode;
    };

    // --- Buffer sizes ---
    constexpr VkDeviceSize control_size = 4 * sizeof(std::uint32_t);
    const std::uint32_t zero_init[4] = {0, 0, 0, 0};

    VkDeviceSize gate_scratch_size =
        static_cast<VkDeviceSize>(intermediate) * sizeof(std::uint16_t);
    VkDeviceSize up_scratch_size =
        static_cast<VkDeviceSize>(intermediate) * sizeof(std::uint16_t);
    VkDeviceSize input_size =
        static_cast<VkDeviceSize>(hidden) * sizeof(std::uint16_t);
    VkDeviceSize weight_gate_size =
        static_cast<VkDeviceSize>(intermediate) * hidden * sizeof(std::uint16_t);
    VkDeviceSize weight_up_size =
        static_cast<VkDeviceSize>(intermediate) * hidden * sizeof(std::uint16_t);
    VkDeviceSize weight_down_size =
        static_cast<VkDeviceSize>(output_rows) * intermediate * sizeof(std::uint16_t);
    VkDeviceSize output_size =
        static_cast<VkDeviceSize>(output_rows) * sizeof(std::uint16_t);
    VkDeviceSize norm_output_size = input_size;
    VkDeviceSize weight_norm_size = input_size;

    // --- Create buffers ---
    auto control_buf = dev.create_device_local_buffer(control_size);
    dev.upload_to_device(control_buf, zero_init, control_size);

    auto gate_scratch_buf = dev.create_device_local_buffer(gate_scratch_size);
    std::vector<std::uint16_t> gate_scratch_zeros(intermediate, 0);
    dev.upload_to_device(gate_scratch_buf, gate_scratch_zeros.data(), gate_scratch_size);

    auto up_scratch_buf = dev.create_device_local_buffer(up_scratch_size);
    std::vector<std::uint16_t> up_scratch_zeros(intermediate, 0);
    dev.upload_to_device(up_scratch_buf, up_scratch_zeros.data(), up_scratch_size);

    auto input_buf = dev.create_device_local_buffer(input_size);
    dev.upload_to_device(input_buf, input_data.data(), input_size);

    auto weight_gate_buf = dev.create_device_local_buffer(weight_gate_size);
    dev.upload_to_device(weight_gate_buf, weight_gate_data.data(), weight_gate_size);

    auto weight_up_buf = dev.create_device_local_buffer(weight_up_size);
    dev.upload_to_device(weight_up_buf, weight_up_data.data(), weight_up_size);

    auto weight_down_buf = dev.create_device_local_buffer(weight_down_size);
    dev.upload_to_device(weight_down_buf, weight_down_data.data(), weight_down_size);

    auto output_buf = dev.create_device_local_buffer(output_size);
    std::vector<std::uint16_t> output_zeros(output_rows, 0);
    dev.upload_to_device(output_buf, output_zeros.data(), output_size);

    auto norm_output_buf = dev.create_device_local_buffer(norm_output_size);
    std::vector<std::uint16_t> norm_output_zeros(hidden, 0);
    dev.upload_to_device(norm_output_buf, norm_output_zeros.data(), norm_output_size);

    auto weight_norm_buf = dev.create_device_local_buffer(weight_norm_size);
    dev.upload_to_device(weight_norm_buf, weight_norm_data.data(), weight_norm_size);

    // --- Descriptor set ---
    VkDescriptorSet desc_set = dev.allocate_descriptor_set(desc_layout);
    dev.update_descriptor_set(desc_set, 0, control_buf);
    dev.update_descriptor_set(desc_set, 1, gate_scratch_buf);
    dev.update_descriptor_set(desc_set, 2, up_scratch_buf);
    dev.update_descriptor_set(desc_set, 3, input_buf);
    dev.update_descriptor_set(desc_set, 4, weight_gate_buf);
    dev.update_descriptor_set(desc_set, 5, weight_up_buf);
    dev.update_descriptor_set(desc_set, 6, weight_down_buf);
    dev.update_descriptor_set(desc_set, 7, output_buf);
    dev.update_descriptor_set(desc_set, 8, norm_output_buf);
    dev.update_descriptor_set(desc_set, 9, weight_norm_buf);

    PushConsts push{workgroups, hidden, intermediate, output_rows, 0};

    // --- Dispatch ---
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

    // --- Read back results ---
    std::uint32_t control_out[4] = {};
    dev.download_from_device(control_buf, control_out, control_size);

    std::uint32_t arrived = control_out[0];
    std::uint32_t generation = control_out[1];
    std::uint32_t failures = control_out[2];
    std::uint32_t checksum = control_out[3];

    // --- Download GPU output and validate ---
    std::vector<std::uint16_t> gpu_output(output_rows);
    dev.download_from_device(output_buf, gpu_output.data(), output_size);

    std::uint32_t output_exact_mismatches = 0;
    std::uint32_t output_within_tolerance = 0;
    std::uint32_t output_mismatches = 0;
    std::uint32_t max_fp16_ulp_diff = 0;
    std::uint32_t output_ulp_le_1 = 0;
    std::uint32_t output_ulp_le_2 = 0;
    std::uint32_t output_ulp_le_4 = 0;
    std::uint32_t output_ulp_le_8 = 0;
    std::uint32_t output_ulp_le_16 = 0;
    std::uint32_t output_ulp_le_32 = 0;
    std::uint32_t output_ulp_le_64 = 0;
    std::uint32_t output_ulp_gt_64 = 0;
    std::uint32_t output_rows_above_population_ulp_threshold = 0;
    int first_mismatch_row = -1;
    for (std::uint32_t row = 0; row < output_rows; ++row) {
      uint16_t gpu_val = gpu_output[row];
      uint16_t exp_val = expected_output[row];
      uint32_t ulp = fp16_ulp_diff(gpu_val, exp_val);
      if (ulp <= 1) ++output_ulp_le_1;
      if (ulp <= 2) ++output_ulp_le_2;
      if (ulp <= 4) ++output_ulp_le_4;
      if (ulp <= 8) ++output_ulp_le_8;
      if (ulp <= 16) ++output_ulp_le_16;
      if (ulp <= 32) ++output_ulp_le_32;
      if (ulp <= 64) ++output_ulp_le_64;
      else ++output_ulp_gt_64;
      if (output_population_ulp_threshold_set &&
          ulp > output_population_ulp_threshold) {
        ++output_rows_above_population_ulp_threshold;
      }
      if (ulp > max_fp16_ulp_diff) {
        max_fp16_ulp_diff = ulp;
      }
      if (ulp == 0) continue;
      ++output_exact_mismatches;
      if (ulp <= output_fp16_ulp_tolerance) {
        ++output_within_tolerance;
      } else {
        ++output_mismatches;
        if (first_mismatch_row < 0) first_mismatch_row = static_cast<int>(row);
      }
    }

    // --- Cleanup ---
    dev.destroy_buffer(control_buf);
    dev.destroy_buffer(gate_scratch_buf);
    dev.destroy_buffer(up_scratch_buf);
    dev.destroy_buffer(input_buf);
    dev.destroy_buffer(weight_gate_buf);
    dev.destroy_buffer(weight_up_buf);
    dev.destroy_buffer(weight_down_buf);
    dev.destroy_buffer(output_buf);
    dev.destroy_buffer(norm_output_buf);
    dev.destroy_buffer(weight_norm_buf);
    dev.destroy_pipeline(pipeline);
    dev.destroy_shader_module(shader);
    dev.destroy_pipeline_layout(pipe_layout);
    dev.destroy_descriptor_set_layout(desc_layout);

    // --- Status check ---
    bool structural_ok = (failures == 0) &&
                         (generation == expected_generation) &&
                         (arrived == 0);
    bool output_ok = (output_mismatches == 0);
    bool output_population_ok = !output_population_max_rows_set ||
                                (output_rows_above_population_ulp_threshold <=
                                 output_population_max_rows);
    bool ok = structural_ok && output_ok && output_population_ok;

    std::string status = ok ? "ok" : "fail";

    // --- JSON output ---
    std::cout << "{\n";
    std::cout << "  \"probe\": \"persistent_layer0_tail\",\n";
    std::cout << "  \"hidden\": " << hidden << ",\n";
    std::cout << "  \"intermediate_count\": " << intermediate << ",\n";
    std::cout << "  \"output_rows\": " << output_rows << ",\n";
    std::cout << "  \"local_size_x\": 128,\n";
    std::cout << "  \"workgroups\": " << workgroups << ",\n";
    std::cout << "  \"layer\": 0,\n";
    std::cout << "  \"repack_dir\": \"" << repack_dir << "\",\n";
    std::cout << "  \"input_fp16_file\": \"" << input_fp16_file << "\",\n";
    if (!expected_output_fp16_file.empty()) {
      std::cout << "  \"expected_output_fp16_file\": \"" << expected_output_fp16_file << "\",\n";
    }
    std::cout << "  \"status\": \"" << status << "\",\n";
    std::cout << "  \"failures\": " << failures << ",\n";
    std::cout << "  \"arrived\": " << arrived << ",\n";
    std::cout << "  \"generation\": " << generation << ",\n";
    std::cout << "  \"expected_generation\": " << expected_generation << ",\n";
    std::cout << "  \"checksum\": " << checksum << ",\n";
    std::cout << "  \"expected_checksum\": " << expected_checksum << ",\n";
    std::cout << "  \"output_exact_mismatches\": " << output_exact_mismatches << ",\n";
    std::cout << "  \"output_within_tolerance\": " << output_within_tolerance << ",\n";
    std::cout << "  \"output_mismatches\": " << output_mismatches << ",\n";
    std::cout << "  \"max_fp16_ulp_diff\": " << max_fp16_ulp_diff << ",\n";
    std::cout << "  \"output_ulp_le_1\": " << output_ulp_le_1 << ",\n";
    std::cout << "  \"output_ulp_le_2\": " << output_ulp_le_2 << ",\n";
    std::cout << "  \"output_ulp_le_4\": " << output_ulp_le_4 << ",\n";
    std::cout << "  \"output_ulp_le_8\": " << output_ulp_le_8 << ",\n";
    std::cout << "  \"output_ulp_le_16\": " << output_ulp_le_16 << ",\n";
    std::cout << "  \"output_ulp_le_32\": " << output_ulp_le_32 << ",\n";
    std::cout << "  \"output_ulp_le_64\": " << output_ulp_le_64 << ",\n";
    std::cout << "  \"output_ulp_gt_64\": " << output_ulp_gt_64 << ",\n";
    if (output_population_ulp_threshold_set) {
      std::cout << "  \"output_fp16_population_ulp_threshold\": " << output_population_ulp_threshold << ",\n";
      std::cout << "  \"output_rows_above_population_ulp_threshold\": " << output_rows_above_population_ulp_threshold << ",\n";
    }
    if (output_population_max_rows_set) {
      std::cout << "  \"output_fp16_max_rows_above_population_threshold\": " << output_population_max_rows << ",\n";
      std::cout << "  \"output_population_ok\": " << (output_population_ok ? "true" : "false") << ",\n";
    }
    std::cout << "  \"output_fp16_ulp_tolerance\": " << output_fp16_ulp_tolerance;
    if (first_mismatch_row >= 0) {
      std::cout << ",\n  \"first_mismatch_row\": " << first_mismatch_row;
    }
    std::cout << "\n}\n";

    if (!ok) return 1;
    return 0;

  } catch (const std::exception& e) {
    std::cout << "{\n";
    std::cout << "  \"probe\": \"persistent_layer0_tail\",\n";
    std::cout << "  \"status\": \"error\",\n";
    std::cout << "  \"message\": \"" << e.what() << "\"\n";
    std::cout << "}\n";
    return 2;
  }
#else
  std::cout << "{\n";
  std::cout << "  \"probe\": \"persistent_layer0_tail\",\n";
  std::cout << "  \"status\": \"error\",\n";
  std::cout << "  \"message\": \"Vulkan not available (SPOCK_VULKAN_STUB)\"\n";
  std::cout << "}\n";
  return 2;
#endif
}

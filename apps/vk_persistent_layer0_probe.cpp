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

}  // namespace

int main(int argc, char** argv) {
  // Fixed dimensions for layer 0.
  constexpr uint32_t hidden = 1024;
  constexpr uint32_t intermediate = 3584;
  constexpr uint32_t output_rows = 1024;

  uint32_t workgroups = 82;
  std::string repack_dir;
  std::string input_fp16_file;
  std::string expected_output_fp16_file;
  uint32_t output_fp16_ulp_tolerance = 0;
  uint32_t output_population_ulp_threshold = 0;
  bool output_population_ulp_threshold_set = false;
  uint32_t output_population_max_rows = 0;
  bool output_population_max_rows_set = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--repack-dir" && i + 1 < argc) {
      repack_dir = argv[++i];
    } else if (arg == "--input-fp16-file" && i + 1 < argc) {
      input_fp16_file = argv[++i];
    } else if (arg == "--expected-output-fp16-file" && i + 1 < argc) {
      expected_output_fp16_file = argv[++i];
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
      std::cout << "  --repack-dir DIR   load real fp16 weights from repacked model artifact (required)\n";
      std::cout << "  --input-fp16-file PATH  load mixer_residual fp16 input (required)\n";
      std::cout << "  --expected-output-fp16-file PATH  load expected post_mlp fp16 output for comparison\n";
      std::cout << "  --workgroups N     dispatch workgroup count (default 82)\n";
      std::cout << "  --output-fp16-ulp-tolerance N  allow up to N fp16 ULP diff (default 0, exact)\n";
      std::cout << "  --output-fp16-population-ulp-threshold N  count output rows with ULP diff above N\n";
      std::cout << "  --output-fp16-max-rows-above-population-threshold N  fail when rows above threshold exceed N\n";
      std::cout << "  --help             show this help\n";
      std::cout << "\nFixed: hidden=1024, intermediate=3584, output_rows=1024, layer=0,\n";
      std::cout << "       RMSNorm=post_norm, residual=enabled, local_size_x=128.\n";
      return 0;
    } else {
      return json_error("unknown option: " + arg);
    }
  }

  // Validate.
  if (repack_dir.empty()) {
    return json_error("--repack-dir is required");
  }
  if (input_fp16_file.empty()) {
    return json_error("--input-fp16-file is required");
  }
  if (workgroups == 0) {
    return json_error("--workgroups must be > 0");
  }
  if (output_population_max_rows_set && !output_population_ulp_threshold_set) {
    return json_error("--output-fp16-max-rows-above-population-threshold requires --output-fp16-population-ulp-threshold");
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

    // --- Pipeline layout: 16-byte push constants (4 x uint32) ---
    VkPipelineLayout pipe_layout =
        dev.create_pipeline_layout(desc_layout, 4 * sizeof(std::uint32_t));

    // --- Shader + pipeline ---
    auto spirv = read_spirv();
    VkShaderModule shader = dev.create_shader_module(spirv);
    VkPipeline pipeline = dev.create_compute_pipeline(shader, pipe_layout);

    struct PushConsts {
      std::uint32_t workgroup_count;
      std::uint32_t hidden;
      std::uint32_t intermediate_count;
      std::uint32_t output_rows;
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

    PushConsts push{workgroups, hidden, intermediate, output_rows};

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

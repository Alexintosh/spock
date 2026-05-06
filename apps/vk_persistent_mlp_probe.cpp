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

// Deterministic input: small exact fp16 values cycling 1..8.
static inline uint16_t input_vec_val(uint32_t c) {
  return fp32_to_fp16(static_cast<float>((c % 8u) + 1u));
}

// Deterministic weight: small exact fp16 values cycling by group and column.
static inline uint16_t weight_mat_val(uint32_t g, uint32_t c) {
  const uint32_t group_scale = (g % 8u) + 1u;
  const uint32_t col_scale = (c % 8u) + 1u;
  return fp32_to_fp16(static_cast<float>(group_scale * col_scale) / 1024.0f);
}

// Shader-mirrored lane-strided dot product for row g against input_vec.
// 64 lanes, lane l accumulates columns l, l+64, l+128, ... in fp32.
// Tree reduction: stride 32,16,8,4,2,1.
// Returns fp32 dot value (same as shader lane_sums[0]).
static inline float shader_mirrored_dot(
    uint32_t g, uint32_t dim,
    const uint16_t* weight_data,
    const uint16_t* input_data) {
  float lane_sums[64] = {};
  for (uint32_t lane = 0; lane < 64; ++lane) {
    float acc = 0.0f;
    uint32_t row_offset = g * dim;
    for (uint32_t c = lane; c < dim; c += 64) {
      float iv = fp16_to_fp32(input_data[c]);
      float wv = fp16_to_fp32(weight_data[row_offset + c]);
      acc += iv * wv;
    }
    lane_sums[lane] = acc;
  }
  for (uint32_t stride = 32; stride >= 1; stride >>= 1) {
    for (uint32_t lane = 0; lane < stride; ++lane) {
      lane_sums[lane] += lane_sums[lane + stride];
    }
  }
  return lane_sums[0];
}

// Shader-mirrored RMSNorm sum-of-squares reduction.
// Stage 0 uses the same 64-lane lane-strided fp32 accumulation and tree
// reduction as the matvec stages, so the host reference has to preserve that
// order at model width.
static inline float shader_mirrored_sum_sq(uint32_t dim, const uint16_t* input_data) {
  float lane_sums[64] = {};
  for (uint32_t lane = 0; lane < 64; ++lane) {
    float acc = 0.0f;
    for (uint32_t c = lane; c < dim; c += 64) {
      float v = fp16_to_fp32(input_data[c]);
      acc += v * v;
    }
    lane_sums[lane] = acc;
  }
  for (uint32_t stride = 32; stride >= 1; stride >>= 1) {
    for (uint32_t lane = 0; lane < stride; ++lane) {
      lane_sums[lane] += lane_sums[lane + stride];
    }
  }
  return lane_sums[0];
}

// Shader-mirrored down dot product: row g of weight_down dotted with
// activated scratch (length intermediate_count), lane-strided.
static inline float shader_mirrored_down_dot(
    uint32_t out_row, uint32_t intermediate_count,
    const uint16_t* weight_down_data,
    const uint16_t* activated_data) {
  float lane_sums[64] = {};
  uint32_t row_offset = out_row * intermediate_count;
  for (uint32_t lane = 0; lane < 64; ++lane) {
    float acc = 0.0f;
    for (uint32_t c = lane; c < intermediate_count; c += 64) {
      float wv = fp16_to_fp32(weight_down_data[row_offset + c]);
      float av = fp16_to_fp32(activated_data[c]);
      acc += wv * av;
    }
    lane_sums[lane] = acc;
  }
  for (uint32_t stride = 32; stride >= 1; stride >>= 1) {
    for (uint32_t lane = 0; lane < stride; ++lane) {
      lane_sums[lane] += lane_sums[lane + stride];
    }
  }
  return lane_sums[0];
}

// Compute the unsigned ULP distance between two fp16 values.
// Returns 0 for identical values (including both zeros).
// For same-sign non-zero values, returns the difference in their integer representations.
// For opposite-sign non-zero values, returns a large sentinel (UINT32_MAX) since
// such a difference cannot be expressed as a bounded ULP count.
// For +/-0 comparisons, returns 0 (signed zeros are equivalent at fp16 ULP level).
static inline uint32_t fp16_ulp_diff(uint16_t a, uint16_t b) {
  if (a == b) return 0;
  // Both zeros: +0 (0x0000) and -0 (0x8000) are equivalent.
  if ((a == 0x0000 || a == 0x8000) && (b == 0x0000 || b == 0x8000)) return 0;
  bool a_sign = (a & 0x8000u) != 0;
  bool b_sign = (b & 0x8000u) != 0;
  // Opposite signs (and at least one is nonzero, since we handled zeros above).
  if (a_sign != b_sign) return UINT32_MAX;
  // Same sign: ULP diff is the absolute difference of the raw integer representations.
  // For same-sign fp16 values, integer ordering matches magnitude ordering.
  return a > b ? static_cast<uint32_t>(a) - static_cast<uint32_t>(b)
               : static_cast<uint32_t>(b) - static_cast<uint32_t>(a);
}
// SiLU activation: x / (1 + exp(-x)).
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

  auto spv = try_load("build/shaders/persistent_mlp_probe.comp.spv");
  if (!spv.empty()) return spv;

#ifdef SHADER_DIR
  spv = try_load(std::string(SHADER_DIR) + "/persistent_mlp_probe.comp.spv");
  if (!spv.empty()) return spv;
#endif

  throw std::runtime_error(
      "cannot open shader: persistent_mlp_probe.comp.spv "
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

// Load a weight role from the artifact, extracting first extract_rows x extract_cols
// in row-major order from the raw fp16 tensor.
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
  // Extract first extract_rows rows x extract_cols cols via memcpy from raw bytes.
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

// Load a rank-1 fp16 weight vector, extracting the first extract_len elements.
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

}  // namespace

int main(int argc, char** argv) {
  std::uint32_t hidden = 128;
  std::uint32_t intermediate = 16;
  std::uint32_t output_rows = 8;
  std::uint32_t workgroups = 8;
  int layer = 0;
  std::string repack_dir;
  bool residual = false;
  std::string input_fp16_file;
  int input_token = -1;
  bool input_token_set = false;
  uint32_t output_fp16_ulp_tolerance = 0;
  bool pre_mlp_rmsnorm = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--hidden" && i + 1 < argc) {
      hidden = std::stoul(argv[++i]);
    } else if (arg == "--intermediate" && i + 1 < argc) {
      intermediate = std::stoul(argv[++i]);
    } else if (arg == "--output-rows" && i + 1 < argc) {
      output_rows = std::stoul(argv[++i]);
    } else if (arg == "--workgroups" && i + 1 < argc) {
      workgroups = std::stoul(argv[++i]);
    } else if (arg == "--layer" && i + 1 < argc) {
      const std::string layer_str = argv[++i];
      if (layer_str.empty()) {
        return json_error("--layer must be a nonnegative integer, got empty string");
      }
      // Reject leading whitespace or sign characters.
      if (layer_str[0] < '0' || layer_str[0] > '9') {
        return json_error("--layer must be a nonnegative integer, got: " + layer_str);
      }
      std::size_t pos = 0;
      try {
        layer = std::stoi(layer_str, &pos, 10);
      } catch (...) {
        return json_error("--layer must be a nonnegative integer, got: " + layer_str);
      }
      if (pos != layer_str.size()) {
        return json_error("--layer must be a nonnegative integer, got: " + layer_str);
      }
    } else if (arg == "--repack-dir" && i + 1 < argc) {
      repack_dir = argv[++i];
    } else if (arg == "--input-token" && i + 1 < argc) {
      input_token = std::stoi(argv[++i]);
      input_token_set = true;
    } else if (arg == "--input-fp16-file" && i + 1 < argc) {
      input_fp16_file = argv[++i];
    } else if (arg == "--residual") {
      residual = true;
    } else if (arg == "--pre-mlp-rmsnorm") {
      pre_mlp_rmsnorm = true;
    } else if (arg == "--output-fp16-ulp-tolerance" && i + 1 < argc) {
      const std::string tol_str = argv[++i];
      if (tol_str.empty()) {
        return json_error("--output-fp16-ulp-tolerance must be a nonnegative integer, got empty string");
      }
      if (tol_str[0] < '0' || tol_str[0] > '9') {
        return json_error("--output-fp16-ulp-tolerance must be a nonnegative integer, got: " + tol_str);
      }
      std::size_t pos = 0;
      try {
        unsigned long val = std::stoul(tol_str, &pos, 10);
        if (val > UINT32_MAX) {
          return json_error("--output-fp16-ulp-tolerance value too large: " + tol_str);
        }
        output_fp16_ulp_tolerance = static_cast<uint32_t>(val);
      } catch (...) {
        return json_error("--output-fp16-ulp-tolerance must be a nonnegative integer, got: " + tol_str);
      }
      if (pos != tol_str.size()) {
        return json_error("--output-fp16-ulp-tolerance must be a nonnegative integer, got: " + tol_str);
      }
    } else if (arg == "--help") {
      std::cout << "usage: vk_persistent_mlp_probe [options]\n";
      std::cout << "  --hidden N         hidden dimension / weight columns (default 128)\n";
      std::cout << "  --intermediate N   intermediate (FFN) dimension (default 16)\n";
      std::cout << "  --output-rows N    output rows from down projection (default 8)\n";
      std::cout << "  --workgroups N     dispatch workgroup count (default 8)\n";
      std::cout << "  --layer N          layer index for real-weight roles (default 0)\n";
      std::cout << "  --repack-dir DIR   load real fp16 weights from repacked model artifact\n";
      std::cout << "  --residual         enable residual mode (output += input)\n";
      std::cout << "  --input-token ID   use real token embedding row as input (requires --repack-dir)\n";
      std::cout << "  --input-fp16-file PATH  load raw fp16 input vector from file (mutually exclusive with --input-token)\n";
      std::cout << "  --pre-mlp-rmsnorm  apply RMSNorm to input before gate/up projections (requires --repack-dir)\n";
      std::cout << "  --output-fp16-ulp-tolerance N  allow up to N fp16 ULP diff between GPU and CPU output (default 0, exact)\n";
      std::cout << "  --help             show this help\n";
      return 0;
    }
  }

  // Validate.
  if (layer < 0) {
    return json_error("--layer must be >= 0, got: " + std::to_string(layer));
  }
  if (hidden == 0 || intermediate == 0 || output_rows == 0 || workgroups == 0) {
    return json_error("--hidden, --intermediate, --output-rows, --workgroups must be > 0");
  }
  if (input_token_set && input_token < 0) {
    return json_error("--input-token must be >= 0, got: " + std::to_string(input_token));
  }
  if (!input_fp16_file.empty() && input_token_set) {
    return json_error("--input-fp16-file and --input-token are mutually exclusive");
  }
  if (input_token >= 0 && repack_dir.empty()) {
    return json_error("--input-token requires --repack-dir");
  }
  if (residual && output_rows > hidden) {
    return json_error("--residual requires --output-rows <= --hidden");
  }
  if (pre_mlp_rmsnorm && repack_dir.empty()) {
    return json_error("--pre-mlp-rmsnorm requires --repack-dir");
  }

  // --- Weight data ---
  bool real_weight = !repack_dir.empty();

  // Weight matrices in fp16 row-major.
  std::vector<uint16_t> weight_gate_data;
  std::vector<uint16_t> weight_up_data;
  std::vector<uint16_t> weight_down_data;
  std::vector<uint16_t> weight_norm_data;  // RMSNorm weight (post_norm), [hidden]

  std::vector<uint16_t> input_data(hidden);
  bool use_embedding_input = false;
  bool use_fp16_file_input = false;

  if (real_weight && input_token >= 0) {
    try {
      auto artifact = spock::runtime::WeightArtifact::load(repack_dir);
      const auto* emb_info = artifact.find_by_role("global.token_embedding");
      if (!emb_info) {
        return json_error("weight role not found: global.token_embedding");
      }
      if (emb_info->dtype != "fp16") {
        return json_error("token_embedding dtype must be fp16, got: " + emb_info->dtype);
      }
      if (emb_info->shape.size() != 2) {
        return json_error("token_embedding must be rank-2, got rank: " + std::to_string(emb_info->shape.size()));
      }
      uint32_t vocab_size = static_cast<uint32_t>(emb_info->shape[0]);
      uint32_t emb_dim = static_cast<uint32_t>(emb_info->shape[1]);
      if (static_cast<uint32_t>(input_token) >= vocab_size) {
        return json_error("--input-token " + std::to_string(input_token) + " >= vocab_size " + std::to_string(vocab_size));
      }
      if (emb_dim < hidden) {
        return json_error("token_embedding dim " + std::to_string(emb_dim) + " < hidden " + std::to_string(hidden));
      }
      auto raw = spock::runtime::read_tensor_bytes(artifact, *emb_info);
      // Extract exactly one row of length hidden starting at row input_token.
      std::size_t src_offset = static_cast<std::size_t>(input_token) * emb_dim * sizeof(uint16_t);
      std::memcpy(input_data.data(), raw.data() + src_offset, hidden * sizeof(uint16_t));
      use_embedding_input = true;
    } catch (const std::exception& e) {
      return json_error(std::string("token embedding loading failed: ") + e.what());
    }
  } else if (!input_fp16_file.empty()) {
    try {
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
      use_fp16_file_input = true;
    } catch (const std::exception& e) {
      return json_error(std::string("--input-fp16-file read failed: ") + e.what());
    }
  } else {
    for (std::uint32_t c = 0; c < hidden; ++c) {
      input_data[c] = input_vec_val(c);
    }
  }

  if (real_weight) {
    try {
      auto artifact = spock::runtime::WeightArtifact::load(repack_dir);

      // Load gate: intermediate rows x hidden cols from layer.N.mlp_gate.
      weight_gate_data = load_weight_slice(
          artifact, "layer." + std::to_string(layer) + ".mlp_gate", intermediate, hidden);

      // Load up: intermediate rows x hidden cols from layer.N.mlp_up.
      weight_up_data = load_weight_slice(
          artifact, "layer." + std::to_string(layer) + ".mlp_up", intermediate, hidden);

      // Load down: output_rows rows x intermediate cols from layer.N.mlp_down.
      weight_down_data = load_weight_slice(
          artifact, "layer." + std::to_string(layer) + ".mlp_down", output_rows, intermediate);

      // Load RMSNorm weight (post_norm) if pre-mlp-rmsnorm enabled.
      if (pre_mlp_rmsnorm) {
        weight_norm_data = load_weight_vector(
            artifact, "layer." + std::to_string(layer) + ".post_norm", hidden);
      }
    } catch (const std::exception& e) {
      return json_error(std::string("weight loading failed: ") + e.what());
    }
  } else {
    // Synthetic weights.
    weight_gate_data.resize(static_cast<std::size_t>(intermediate) * hidden);
    weight_up_data.resize(static_cast<std::size_t>(intermediate) * hidden);
    weight_down_data.resize(static_cast<std::size_t>(output_rows) * intermediate);

    for (std::uint32_t row = 0; row < intermediate; ++row) {
      for (std::uint32_t c = 0; c < hidden; ++c) {
        weight_gate_data[static_cast<std::size_t>(row) * hidden + c] =
            weight_mat_val(row, c);
        weight_up_data[static_cast<std::size_t>(row) * hidden + c] =
            weight_mat_val(row, c);
      }
    }
    for (std::uint32_t row = 0; row < output_rows; ++row) {
      for (std::uint32_t c = 0; c < intermediate; ++c) {
        weight_down_data[static_cast<std::size_t>(row) * intermediate + c] =
            weight_mat_val(row, c);
      }
    }
  }

  // --- CPU RMSNorm computation (mirrors shader Stage 0) ---
  // out[i] = fp16(input[i] * rsqrt(mean(input^2) + 1e-6) * (1 + weight[i]))
  // Preserves raw input_data for residual addition.
  std::vector<uint16_t> normalized_input;
  if (pre_mlp_rmsnorm) {
    normalized_input.resize(hidden);
    float sum_sq = shader_mirrored_sum_sq(hidden, input_data.data());
    float mean_sq = sum_sq / static_cast<float>(hidden);
    float inv_rms = 1.0f / std::sqrt(mean_sq + 1e-6f);

    for (std::uint32_t c = 0; c < hidden; ++c) {
      float v = fp16_to_fp32(input_data[c]);
      float w = fp16_to_fp32(weight_norm_data[c]);
      normalized_input[c] = fp32_to_fp16(v * inv_rms * (1.0f + w));
    }
  }

  // Select the input for gate/up projections.
  const uint16_t* proj_input = pre_mlp_rmsnorm ? normalized_input.data()
                                                : input_data.data();

  // --- CPU expected computation (mirrors shader) ---
  // Stage 1: gate dot and up dot per intermediate row.
  //   Per row: lane-strided fp32 dot, tree-reduce, convert dot to fp16.
  std::vector<uint16_t> gate_scratch(intermediate);
  std::vector<uint16_t> up_scratch(intermediate);

  for (std::uint32_t row = 0; row < intermediate; ++row) {
    float gate_dot = shader_mirrored_dot(row, hidden,
                                          weight_gate_data.data(),
                                          proj_input);
    float up_dot = shader_mirrored_dot(row, hidden,
                                         weight_up_data.data(),
                                         proj_input);
    gate_scratch[row] = fp32_to_fp16(gate_dot);
    up_scratch[row] = fp32_to_fp16(up_dot);
  }

  // Stage 2: activation fp16 = fp16(silu(fp16(gate_dot)) * fp16(up_dot)).
  // Stored back into gate_scratch.
  std::vector<uint16_t> activated(intermediate);
  for (std::uint32_t row = 0; row < intermediate; ++row) {
    float g = fp16_to_fp32(gate_scratch[row]);
    float u = fp16_to_fp32(up_scratch[row]);
    float act = silu(g) * u;
    activated[row] = fp32_to_fp16(act);
  }

  // Stage 3: down dot per output row, lane-strided over intermediate.
  //   expected_checksum = sum floatBitsToUint(down_total fp32) over output_rows.
  //   NOTE: This fp32-bit checksum is a structural diagnostic.  The authoritative
  //   pass/fail gate is the per-row fp16 output comparison below.
  std::uint32_t expected_checksum = 0;
  std::vector<uint16_t> expected_output(output_rows);
  for (std::uint32_t row = 0; row < output_rows; ++row) {
    float down_total = shader_mirrored_down_dot(
        row, intermediate,
        weight_down_data.data(),
        activated.data());
    if (residual) {
      down_total += fp16_to_fp32(input_data[row]);
    }
    std::uint32_t bits;
    std::memcpy(&bits, &down_total, sizeof(bits));
    expected_checksum += bits;
    expected_output[row] = fp32_to_fp16(down_total);
  }

  // Global barrier count:
  //   Without pre_mlp_rmsnorm: 2 barriers (Stage1→Stage2, Stage2→Stage3).
  //   With pre_mlp_rmsnorm:    3 barriers (Stage0→Stage1, Stage1→Stage2, Stage2→Stage3).
  std::uint32_t expected_generation = pre_mlp_rmsnorm ? 3u : 2u;

#if SPOCK_HAS_VULKAN && !defined(SPOCK_VULKAN_STUB)
  try {
    spock::runtime::VulkanDevice dev;
    dev.initialize();

    // --- Descriptor set layout: 10 storage buffers ---
    // Bindings 0-7: original buffers.
    // Binding 8: norm_output (float16_t[hidden]).
    // Binding 9: weight_norm (float16_t[hidden]).
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

    // --- Pipeline layout: 24-byte push constants (6 x uint32) ---
    VkPipelineLayout pipe_layout =
        dev.create_pipeline_layout(desc_layout, 6 * sizeof(std::uint32_t));

    // --- Shader + pipeline ---
    auto spirv = read_spirv();
    VkShaderModule shader = dev.create_shader_module(spirv);
    VkPipeline pipeline = dev.create_compute_pipeline(shader, pipe_layout);

    struct PushConsts {
      std::uint32_t workgroup_count;
      std::uint32_t hidden;
      std::uint32_t intermediate_count;
      std::uint32_t output_rows;
      std::uint32_t residual_enabled;
      std::uint32_t pre_mlp_rmsnorm;
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

    // Norm output buffer: initialized to zeros. Shader Stage 0 writes normalized values.
    auto norm_output_buf = dev.create_device_local_buffer(norm_output_size);
    std::vector<std::uint16_t> norm_output_zeros(hidden, 0);
    dev.upload_to_device(norm_output_buf, norm_output_zeros.data(), norm_output_size);

    // Weight norm buffer: RMSNorm weight (post_norm).
    // If not pre_mlp_rmsnorm, upload zeros (shader won't read it).
    auto weight_norm_buf = dev.create_device_local_buffer(weight_norm_size);
    if (pre_mlp_rmsnorm) {
      dev.upload_to_device(weight_norm_buf, weight_norm_data.data(), weight_norm_size);
    } else {
      std::vector<std::uint16_t> weight_norm_zeros(hidden, 0);
      dev.upload_to_device(weight_norm_buf, weight_norm_zeros.data(), weight_norm_size);
    }

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

    PushConsts push{workgroups, hidden, intermediate, output_rows,
                    residual ? 1u : 0u,
                    pre_mlp_rmsnorm ? 1u : 0u};

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

    // --- Download GPU output and validate per-row fp16 values ---
    std::vector<std::uint16_t> gpu_output(output_rows);
    dev.download_from_device(output_buf, gpu_output.data(), output_size);

    std::uint32_t output_exact_mismatches = 0;
    std::uint32_t output_within_tolerance = 0;
    std::uint32_t output_mismatches = 0;
    std::uint32_t max_fp16_ulp_diff = 0;
    int first_mismatch_row = -1;
    for (std::uint32_t row = 0; row < output_rows; ++row) {
      uint16_t gpu_val = gpu_output[row];
      uint16_t exp_val = expected_output[row];
      uint32_t ulp = fp16_ulp_diff(gpu_val, exp_val);
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
    // Structural checks: barrier correctness.
    bool structural_ok = (failures == 0) &&
                         (generation == expected_generation) &&
                         (arrived == 0);
    // Per-row fp16 output match (respecting tolerance).
    bool output_ok = (output_mismatches == 0);
    bool ok = structural_ok && output_ok;

    std::string status = ok ? "ok" : "fail";

    // --- JSON output ---
    std::cout << "{\n";
    std::cout << "  \"hidden\": " << hidden << ",\n";
    std::cout << "  \"intermediate_count\": " << intermediate << ",\n";
    std::cout << "  \"output_rows\": " << output_rows << ",\n";
    std::cout << "  \"workgroups\": " << workgroups << ",\n";
    std::cout << "  \"real_weight\": " << (real_weight ? "true" : "false") << ",\n";
    std::cout << "  \"layer\": " << layer << ",\n";
    if (residual) {
      std::cout << "  \"residual\": true,\n";
    }
    if (pre_mlp_rmsnorm) {
      std::cout << "  \"pre_mlp_rmsnorm\": true,\n";
    }
    if (use_embedding_input) {
      std::cout << "  \"input_token\": " << input_token << ",\n";
    }
    if (use_fp16_file_input) {
      std::cout << "  \"input_fp16_file\": \"" << input_fp16_file << "\",\n";
    }
    if (real_weight) {
      std::cout << "  \"repack_dir\": \"" << repack_dir << "\",\n";
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
    std::cout << "  \"output_fp16_ulp_tolerance\": " << output_fp16_ulp_tolerance;
    if (first_mismatch_row >= 0) {
      std::cout << ",\n  \"first_mismatch_row\": " << first_mismatch_row;
    }
    std::cout << "\n";
    std::cout << "}\n";

    if (!ok) return 1;
    return 0;

  } catch (const std::exception& e) {
    std::cout << "{\n";
    std::cout << "  \"hidden\": " << hidden << ",\n";
    std::cout << "  \"intermediate_count\": " << intermediate << ",\n";
    std::cout << "  \"output_rows\": " << output_rows << ",\n";
    std::cout << "  \"workgroups\": " << workgroups << ",\n";
    std::cout << "  \"status\": \"error\",\n";
    std::cout << "  \"message\": \"" << e.what() << "\"\n";
    std::cout << "}\n";
    return 2;
  }
#else
  std::cout << "{\n";
  std::cout << "  \"hidden\": " << hidden << ",\n";
  std::cout << "  \"intermediate_count\": " << intermediate << ",\n";
  std::cout << "  \"output_rows\": " << output_rows << ",\n";
  std::cout << "  \"workgroups\": " << workgroups << ",\n";
  std::cout << "  \"status\": \"error\",\n";
  std::cout << "  \"message\": \"Vulkan not available (SPOCK_VULKAN_STUB)\"\n";
  std::cout << "}\n";
  return 2;
#endif
}

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace spock::runtime {

/// Describes one packed tensor in the repacked weight artifact.
struct TensorInfo {
  std::string role_path;   // e.g. "layer.0.input_norm"
  std::string name;        // e.g. "model.language_model.layers.0.input_layernorm.weight"
  std::uint64_t offset;    // byte offset in text_weights.bin
  std::uint64_t nbytes;    // byte size in file
  std::string dtype;       // "fp16" or "fp32"
  std::vector<std::size_t> shape;
};

/// Parsed repack artifact: manifest + binary file path.
class WeightArtifact {
 public:
  /// Load manifest from a repack directory (contains text_repack_manifest.json + text_weights.bin).
  static WeightArtifact load(const std::string& repack_dir);

  /// Find tensor by PyTorch-compatible state_dict key.
  /// The name is converted from repack format ("model.language_model.layers.X...")
  /// to PyTorch format ("model.layers.X...").
  const TensorInfo* find_by_state_dict_key(const std::string& key) const;

  /// Find tensor by repack role path, e.g. "layer.0.input_norm".
  const TensorInfo* find_by_role(const std::string& role_path) const;

  /// Find tensor by manifest name (as-is).
  const TensorInfo* find_by_name(const std::string& name) const;

  /// Get the global token embedding tensor info.
  const TensorInfo& token_embedding() const;
  /// Get the global final norm tensor info.
  const TensorInfo& final_norm() const;

  const std::string& weights_file_path() const { return weights_file_path_; }
  const std::vector<TensorInfo>& tensors() const { return tensors_; }
  std::size_t total_bytes() const { return total_bytes_; }
  std::size_t tensor_count() const { return tensors_.size(); }

 private:
  std::vector<TensorInfo> tensors_;
  std::unordered_map<std::string, std::size_t> name_index_;
  std::unordered_map<std::string, std::size_t> role_index_;
  std::string weights_file_path_;
  std::size_t total_bytes_ = 0;
};

/// Read raw tensor bytes from the weight file at the given offset.
std::vector<std::byte> read_tensor_bytes(const WeightArtifact& artifact,
                                          const TensorInfo& info);

}  // namespace spock::runtime

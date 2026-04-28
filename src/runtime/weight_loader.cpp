#include "weight_loader.hpp"
#include "json_parse.hpp"

#include <filesystem>
#include <fstream>

namespace spock::runtime {

namespace {

constexpr std::string_view kLanguageModelPrefix = "model.language_model.";

std::string state_dict_key_from_name(const std::string& name) {
  // Convert "model.language_model.layers.X.Y" → "model.layers.X.Y"
  if (name.starts_with(kLanguageModelPrefix)) {
    return "model." + name.substr(kLanguageModelPrefix.size());
  }
  return name;
}

}  // anonymous namespace

WeightArtifact WeightArtifact::load(const std::string& repack_dir) {
  namespace fs = std::filesystem;

  auto manifest_path = fs::path(repack_dir) / "text_repack_manifest.json";
  auto root = read_json_file(manifest_path.string());

  WeightArtifact artifact;
  artifact.weights_file_path_ = (fs::path(repack_dir) / "text_weights.bin").string();

  auto* tensors_arr = root.get("tensors");
  if (!tensors_arr || !tensors_arr->is_array()) {
    throw std::runtime_error("manifest missing 'tensors' array");
  }

  for (const auto& entry : tensors_arr->as_array()) {
    TensorInfo info;
    info.role_path = entry.get_string("role_path");
    info.name = entry.get_string("name");
    info.offset = static_cast<std::uint64_t>(entry.get_int("offset"));
    info.nbytes = static_cast<std::uint64_t>(entry.get_int("nbytes"));
    info.dtype = entry.get_string("dtype");

    auto* shape_val = entry.get("shape");
    if (shape_val && shape_val->is_array()) {
      for (const auto& dim : shape_val->as_array()) {
        info.shape.push_back(static_cast<std::size_t>(dim.as_int()));
      }
    }

    std::size_t idx = artifact.tensors_.size();
    artifact.name_index_[info.name] = idx;
    artifact.role_index_[info.role_path] = idx;
    artifact.total_bytes_ += info.nbytes;
    artifact.tensors_.push_back(std::move(info));
  }

  return artifact;
}

const TensorInfo* WeightArtifact::find_by_state_dict_key(const std::string& key) const {
  // Convert from "model.layers.X.Y" → "model.language_model.layers.X.Y"
  // then look up by manifest name.
  std::string converted;
  if (key.starts_with("model.layers.")) {
    converted = "model.language_model." + key.substr(6);  // skip "model."
  } else {
    converted = key;
  }
  return find_by_name(converted);
}

const TensorInfo* WeightArtifact::find_by_role(const std::string& role_path) const {
  auto it = role_index_.find(role_path);
  if (it == role_index_.end()) return nullptr;
  return &tensors_[it->second];
}

const TensorInfo* WeightArtifact::find_by_name(const std::string& name) const {
  auto it = name_index_.find(name);
  if (it == name_index_.end()) return nullptr;
  return &tensors_[it->second];
}

const TensorInfo& WeightArtifact::token_embedding() const {
  auto* t = find_by_role("global.token_embedding");
  if (!t) throw std::runtime_error("token embedding tensor not found");
  return *t;
}

const TensorInfo& WeightArtifact::final_norm() const {
  auto* t = find_by_role("global.final_norm");
  if (!t) throw std::runtime_error("final norm tensor not found");
  return *t;
}

std::vector<std::byte> read_tensor_bytes(const WeightArtifact& artifact,
                                          const TensorInfo& info) {
  std::ifstream ifs(artifact.weights_file_path(), std::ios::binary);
  if (!ifs) {
    throw std::runtime_error("cannot open weights file: " + artifact.weights_file_path());
  }
  ifs.seekg(static_cast<std::streamoff>(info.offset), std::ios::beg);
  if (!ifs) {
    throw std::runtime_error("seek failed in weights file");
  }
  std::vector<std::byte> buf(info.nbytes);
  ifs.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(info.nbytes));
  if (!ifs) {
    throw std::runtime_error("read failed in weights file");
  }
  return buf;
}

}  // namespace spock::runtime

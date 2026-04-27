#include "runtime/vk_allocator.hpp"

namespace spock::runtime {

std::string_view to_string(const BufferRole role) {
  switch (role) {
    case BufferRole::Weights:
      return "weights";
    case BufferRole::Activations:
      return "activations";
    case BufferRole::DeltaState:
      return "delta_state";
    case BufferRole::KvCache:
      return "kv_cache";
    case BufferRole::Scratch:
      return "scratch";
    case BufferRole::Staging:
      return "staging";
  }
  return "unknown";
}

BufferPlan make_buffer_plan(const BufferRole role, const std::uint64_t size_bytes) {
  const bool staging = role == BufferRole::Staging;
  return BufferPlan{
      .role = role,
      .size_bytes = size_bytes,
      .alignment_bytes = 256,
      .device_local = !staging,
      .host_visible = staging,
  };
}

}  // namespace spock::runtime

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace spock::runtime {

enum class BufferRole {
  Weights,
  Activations,
  DeltaState,
  KvCache,
  Scratch,
  Staging,
};

struct BufferPlan {
  BufferRole role;
  std::uint64_t size_bytes;
  std::uint64_t alignment_bytes;
  bool device_local;
  bool host_visible;
};

std::string_view to_string(BufferRole role);
BufferPlan make_buffer_plan(BufferRole role, std::uint64_t size_bytes);

}  // namespace spock::runtime

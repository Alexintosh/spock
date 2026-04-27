#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace spock::runtime {

struct VulkanCapabilities {
  bool vulkan_available = false;
  std::string device_name = "unavailable";
  std::uint32_t api_version = 0;
  std::uint32_t subgroup_size = 0;
  std::uint64_t max_shared_memory_bytes = 0;
  std::uint32_t max_workgroup_invocations = 0;
  std::vector<std::string> notes;
};

class VulkanContext {
 public:
  static VulkanCapabilities query_default_device();
  static std::string render_capabilities_json(const VulkanCapabilities& caps);
};

}  // namespace spock::runtime

#include "runtime/vk_context.hpp"

#include <sstream>

#if SPOCK_HAS_VULKAN && !defined(SPOCK_VULKAN_STUB)
#include <vulkan/vulkan.h>
#endif

namespace spock::runtime {

VulkanCapabilities VulkanContext::query_default_device() {
  VulkanCapabilities caps{};
#if !SPOCK_HAS_VULKAN || defined(SPOCK_VULKAN_STUB)
  caps.notes.push_back("built without Vulkan SDK");
#else
  try {
    VulkanDevice device;
    device.initialize();
    caps = device.capabilities();
  } catch (const std::exception& error) {
    caps.notes.push_back(error.what());
  }
#endif
  return caps;
}

std::string VulkanContext::render_capabilities_json(const VulkanCapabilities& caps) {
  std::ostringstream out;
  out << "{\n";
  out << "  \"vulkan_available\": " << (caps.vulkan_available ? "true" : "false") << ",\n";
  out << "  \"device_name\": \"" << caps.device_name << "\",\n";
  out << "  \"api_version\": " << caps.api_version << ",\n";
  out << "  \"subgroup_size\": " << caps.subgroup_size << ",\n";
  out << "  \"max_shared_memory_bytes\": " << caps.max_shared_memory_bytes << ",\n";
  out << "  \"max_workgroup_invocations\": " << caps.max_workgroup_invocations << ",\n";
  out << "  \"timestamp_period_ns\": " << caps.timestamp_period << ",\n";
  out << "  \"timestamp_valid\": " << (caps.timestamp_valid ? "true" : "false") << ",\n";
  out << "  \"notes\": [";
  for (std::size_t i = 0; i < caps.notes.size(); ++i) {
    out << (i == 0 ? "" : ", ") << '"' << caps.notes[i] << '"';
  }
  out << "]\n";
  out << "}\n";
  return out.str();
}

}  // namespace spock::runtime

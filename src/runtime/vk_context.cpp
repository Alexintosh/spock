#include "runtime/vk_context.hpp"

#include <sstream>

#if SPOCK_HAS_VULKAN && !defined(SPOCK_VULKAN_STUB)
#include <vulkan/vulkan.h>
#endif

namespace spock::runtime {

VulkanCapabilities VulkanContext::query_default_device() {
  VulkanCapabilities caps{};
#if SPOCK_HAS_VULKAN && !defined(SPOCK_VULKAN_STUB)
  VkApplicationInfo app_info{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app_info.pApplicationName = "spock";
  app_info.apiVersion = VK_API_VERSION_1_2;

  VkInstanceCreateInfo create_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  create_info.pApplicationInfo = &app_info;

  VkInstance instance = VK_NULL_HANDLE;
  if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS) {
    caps.notes.push_back("vkCreateInstance failed");
    return caps;
  }

  std::uint32_t count = 0;
  vkEnumeratePhysicalDevices(instance, &count, nullptr);
  if (count == 0) {
    caps.notes.push_back("no physical Vulkan devices found");
    vkDestroyInstance(instance, nullptr);
    return caps;
  }

  std::vector<VkPhysicalDevice> devices(count);
  vkEnumeratePhysicalDevices(instance, &count, devices.data());
  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(devices.front(), &props);

  caps.vulkan_available = true;
  caps.device_name = props.deviceName;
  caps.api_version = props.apiVersion;
  caps.max_shared_memory_bytes = props.limits.maxComputeSharedMemorySize;
  caps.max_workgroup_invocations = props.limits.maxComputeWorkGroupInvocations;

  VkPhysicalDeviceSubgroupProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
  VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  props2.pNext = &subgroup;
  vkGetPhysicalDeviceProperties2(devices.front(), &props2);
  caps.subgroup_size = subgroup.subgroupSize;

  vkDestroyInstance(instance, nullptr);
#else
  caps.notes.push_back("built without Vulkan SDK");
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
  out << "  \"notes\": [";
  for (std::size_t i = 0; i < caps.notes.size(); ++i) {
    out << (i == 0 ? "" : ", ") << '"' << caps.notes[i] << '"';
  }
  out << "]\n";
  out << "}\n";
  return out.str();
}

}  // namespace spock::runtime

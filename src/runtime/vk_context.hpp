#pragma once

#include "runtime/vk_device.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace spock::runtime {

class VulkanContext {
 public:
  static VulkanCapabilities query_default_device();
  static std::string render_capabilities_json(const VulkanCapabilities& caps);
};

}  // namespace spock::runtime

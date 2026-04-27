#include "runtime/vk_context.hpp"

#include <iostream>

int main() {
  const auto caps = spock::runtime::VulkanContext::query_default_device();
  std::cout << spock::runtime::VulkanContext::render_capabilities_json(caps);
}

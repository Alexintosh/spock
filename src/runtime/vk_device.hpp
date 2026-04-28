#pragma once

#include <cstdint>
#include <string>
#include <vector>

#if SPOCK_HAS_VULKAN && !defined(SPOCK_VULKAN_STUB)
#include <vulkan/vulkan.h>
#endif

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

#if SPOCK_HAS_VULKAN && !defined(SPOCK_VULKAN_STUB)

/// RAII Vulkan compute device. Owns instance, device, queue, command pool,
/// and provides buffer management, pipeline creation, and submit helpers.
class VulkanDevice {
 public:
  VulkanDevice();
  ~VulkanDevice();

  VulkanDevice(const VulkanDevice&) = delete;
  VulkanDevice& operator=(const VulkanDevice&) = delete;
  VulkanDevice(VulkanDevice&&) noexcept;
  VulkanDevice& operator=(VulkanDevice&&) noexcept;

  /// Initialize instance, select device, create compute queue and command pool.
  void initialize();

  /// Tear down all Vulkan objects. Called automatically by destructor.
  void destroy();

  const VulkanCapabilities& capabilities() const { return caps_; }

  // --- Buffer management ---

  /// Create a device-local storage buffer. Returns (buffer, device_memory).
  struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* mapped = nullptr;  // non-null if host-visible
  };

  Buffer create_device_local_buffer(VkDeviceSize size);
  Buffer create_host_visible_buffer(VkDeviceSize size);

  /// Upload data to a device-local buffer via a staging buffer and copy cmd.
  void upload_to_device(const Buffer& dst, const void* data, VkDeviceSize size);

  /// Upload data to a host-visible buffer (memcpy).
  void upload_to_host_visible(const Buffer& dst, const void* data, VkDeviceSize size);

  /// Download data from a device-local buffer via staging.
  void download_from_device(const Buffer& src, void* data, VkDeviceSize size);

  void destroy_buffer(Buffer& buf);

  // --- Command buffers ---

  VkCommandBuffer allocate_command_buffer();
  void begin_command_buffer(VkCommandBuffer cmd);
  void end_command_buffer(VkCommandBuffer cmd);
  void submit_and_wait(VkCommandBuffer cmd);

  // --- Pipeline ---

  VkShaderModule create_shader_module(const std::vector<std::uint32_t>& spirv);
  void destroy_shader_module(VkShaderModule module);

  VkDescriptorSetLayout create_descriptor_set_layout(
      const std::vector<VkDescriptorSetLayoutBinding>& bindings);
  void destroy_descriptor_set_layout(VkDescriptorSetLayout layout);

  VkPipelineLayout create_pipeline_layout(
      VkDescriptorSetLayout descriptor_set_layout,
      VkDeviceSize push_constant_range_size = 0);
  void destroy_pipeline_layout(VkPipelineLayout layout);

  VkPipeline create_compute_pipeline(
      VkShaderModule shader,
      VkPipelineLayout pipeline_layout,
      const char* entry_point = "main");
  void destroy_pipeline(VkPipeline pipeline);

  VkDescriptorSet allocate_descriptor_set(VkDescriptorSetLayout layout);

  void update_descriptor_set(
      VkDescriptorSet set,
      uint32_t binding,
      const Buffer& buffer,
      VkDeviceSize offset = 0,
      VkDeviceSize range = VK_WHOLE_SIZE);

  // --- Raw handles for advanced use ---

  VkDevice device() const { return device_; }
  VkPhysicalDevice physical_device() const { return physical_device_; }

 private:
  VkInstance instance_ = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue compute_queue_ = VK_NULL_HANDLE;
  uint32_t compute_queue_family_ = 0;
  VkCommandPool command_pool_ = VK_NULL_HANDLE;
  VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
  VulkanCapabilities caps_;
  bool initialized_ = false;

  uint32_t find_memory_type(uint32_t type_filter,
                             VkMemoryPropertyFlags properties) const;
};

#else  // Vulkan stub

/// Stub when Vulkan is not available.
class VulkanDevice {
 public:
  VulkanDevice() = default;
  ~VulkanDevice() = default;
  void initialize() {}
  void destroy() {}
  VulkanCapabilities capabilities() const { return {}; }
};

#endif  // SPOCK_HAS_VULKAN

std::string render_capabilities_json(const VulkanCapabilities& caps);

}  // namespace spock::runtime

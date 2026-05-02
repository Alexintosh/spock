#include "runtime/vk_device.hpp"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#if SPOCK_HAS_VULKAN && !defined(SPOCK_VULKAN_STUB)
#include <vulkan/vulkan.h>
#endif

namespace spock::runtime {

#if SPOCK_HAS_VULKAN && !defined(SPOCK_VULKAN_STUB)

namespace {

struct DevicePick {
  VkPhysicalDevice physical_device = VK_NULL_HANDLE;
  uint32_t compute_queue_family = UINT32_MAX;
  VkPhysicalDeviceProperties properties{};
  uint64_t score = 0;
};

uint64_t device_score(const VkPhysicalDeviceProperties& props) {
  uint64_t score = 0;

  switch (props.deviceType) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
      score += 1'000'000'000ull;
      break;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
      score += 100'000'000ull;
      break;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
      score += 10'000'000ull;
      break;
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
      score += 1'000'000ull;
      break;
    default:
      break;
  }

  // This project is explicitly targeting the local RX 6750 XT.
  if (props.vendorID == 0x1002) score += 100'000ull;
  if (std::string(props.deviceName).find("6750 XT") != std::string::npos) score += 10'000ull;
  if (std::string(props.deviceName).find("NAVI22") != std::string::npos) score += 5'000ull;
  if (std::string(props.deviceName).find("llvmpipe") != std::string::npos) score = 0;

  score += static_cast<uint64_t>(props.apiVersion);
  return score;
}

}  // namespace

VulkanDevice::VulkanDevice() = default;

VulkanDevice::~VulkanDevice() {
  if (initialized_) {
    destroy();
  }
}

VulkanDevice::VulkanDevice(VulkanDevice&& other) noexcept
    : instance_(other.instance_),
      physical_device_(other.physical_device_),
      device_(other.device_),
      compute_queue_(other.compute_queue_),
      compute_queue_family_(other.compute_queue_family_),
      command_pool_(other.command_pool_),
      descriptor_pool_(other.descriptor_pool_),
      caps_(std::move(other.caps_)),
      initialized_(other.initialized_) {
  other.instance_ = VK_NULL_HANDLE;
  other.physical_device_ = VK_NULL_HANDLE;
  other.device_ = VK_NULL_HANDLE;
  other.compute_queue_ = VK_NULL_HANDLE;
  other.command_pool_ = VK_NULL_HANDLE;
  other.descriptor_pool_ = VK_NULL_HANDLE;
  other.initialized_ = false;
}

VulkanDevice& VulkanDevice::operator=(VulkanDevice&& other) noexcept {
  if (this != &other) {
    if (initialized_) {
      destroy();
    }
    instance_ = other.instance_;
    physical_device_ = other.physical_device_;
    device_ = other.device_;
    compute_queue_ = other.compute_queue_;
    compute_queue_family_ = other.compute_queue_family_;
    command_pool_ = other.command_pool_;
    descriptor_pool_ = other.descriptor_pool_;
    caps_ = std::move(other.caps_);
    initialized_ = other.initialized_;

    other.instance_ = VK_NULL_HANDLE;
    other.physical_device_ = VK_NULL_HANDLE;
    other.device_ = VK_NULL_HANDLE;
    other.compute_queue_ = VK_NULL_HANDLE;
    other.command_pool_ = VK_NULL_HANDLE;
    other.descriptor_pool_ = VK_NULL_HANDLE;
    other.initialized_ = false;
  }
  return *this;
}

void VulkanDevice::initialize() {
  // --- Create instance ---
  VkApplicationInfo app_info{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app_info.pApplicationName = "spock";
  app_info.apiVersion = VK_API_VERSION_1_2;

  VkInstanceCreateInfo inst_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  inst_info.pApplicationInfo = &app_info;

  if (vkCreateInstance(&inst_info, nullptr, &instance_) != VK_SUCCESS) {
    throw std::runtime_error("vkCreateInstance failed");
  }

  // --- Enumerate physical devices ---
  uint32_t gpu_count = 0;
  vkEnumeratePhysicalDevices(instance_, &gpu_count, nullptr);
  if (gpu_count == 0) {
    vkDestroyInstance(instance_, nullptr);
    instance_ = VK_NULL_HANDLE;
    throw std::runtime_error("no Vulkan physical devices found");
  }

  std::vector<VkPhysicalDevice> gpus(gpu_count);
  vkEnumeratePhysicalDevices(instance_, &gpu_count, gpus.data());

  // --- Pick the best compute-capable device, preferring the target RX 6750 XT ---
  DevicePick best{};
  best.score = 0;

  for (auto gpu : gpus) {
    uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &family_count, families.data());

    uint32_t compute_family = UINT32_MAX;
    for (uint32_t i = 0; i < family_count; ++i) {
      if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
        compute_family = i;
        break;
      }
    }
    if (compute_family == UINT32_MAX) continue;

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(gpu, &props);
    const uint64_t score = device_score(props);
    if (score > best.score) {
      best.physical_device = gpu;
      best.compute_queue_family = compute_family;
      best.properties = props;
      best.score = score;
    }
  }

  if (best.physical_device == VK_NULL_HANDLE) {
    vkDestroyInstance(instance_, nullptr);
    instance_ = VK_NULL_HANDLE;
    throw std::runtime_error("no physical device with compute queue found");
  }

  physical_device_ = best.physical_device;
  compute_queue_family_ = best.compute_queue_family;

  // --- Populate capabilities ---
  caps_.vulkan_available = true;
  caps_.device_name = best.properties.deviceName;
  caps_.api_version = best.properties.apiVersion;
  caps_.max_shared_memory_bytes = best.properties.limits.maxComputeSharedMemorySize;
  caps_.max_workgroup_invocations = best.properties.limits.maxComputeWorkGroupInvocations;
  if (best.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) {
    caps_.notes.push_back("Selected CPU Vulkan device; hardware GPU target is unavailable.");
  }

  VkPhysicalDeviceSubgroupProperties subgroup{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
  VkPhysicalDeviceProperties2 props2{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  props2.pNext = &subgroup;
  vkGetPhysicalDeviceProperties2(physical_device_, &props2);
  caps_.subgroup_size = subgroup.subgroupSize;

  // --- Create logical device ---
  float queue_priority = 1.0f;
  VkDeviceQueueCreateInfo queue_info{
      VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  queue_info.queueFamilyIndex = compute_queue_family_;
  queue_info.queueCount = 1;
  queue_info.pQueuePriorities = &queue_priority;

  VkDeviceCreateInfo dev_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dev_info.queueCreateInfoCount = 1;
  dev_info.pQueueCreateInfos = &queue_info;

  if (vkCreateDevice(physical_device_, &dev_info, nullptr, &device_) !=
      VK_SUCCESS) {
    vkDestroyInstance(instance_, nullptr);
    instance_ = VK_NULL_HANDLE;
    physical_device_ = VK_NULL_HANDLE;
    throw std::runtime_error("vkCreateDevice failed");
  }

  vkGetDeviceQueue(device_, compute_queue_family_, 0, &compute_queue_);

  // --- Create command pool ---
  VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pool_info.queueFamilyIndex = compute_queue_family_;

  if (vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_) !=
      VK_SUCCESS) {
    vkDestroyDevice(device_, nullptr);
    vkDestroyInstance(instance_, nullptr);
    device_ = VK_NULL_HANDLE;
    instance_ = VK_NULL_HANDLE;
    physical_device_ = VK_NULL_HANDLE;
    throw std::runtime_error("vkCreateCommandPool failed");
  }

  // --- Create descriptor pool ---
  std::vector<VkDescriptorPoolSize> pool_sizes = {
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 64},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 192},
  };

  VkDescriptorPoolCreateInfo desc_pool_info{
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  desc_pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  desc_pool_info.maxSets = 192;
  desc_pool_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
  desc_pool_info.pPoolSizes = pool_sizes.data();

  if (vkCreateDescriptorPool(device_, &desc_pool_info, nullptr,
                              &descriptor_pool_) != VK_SUCCESS) {
    vkDestroyCommandPool(device_, command_pool_, nullptr);
    vkDestroyDevice(device_, nullptr);
    vkDestroyInstance(instance_, nullptr);
    command_pool_ = VK_NULL_HANDLE;
    device_ = VK_NULL_HANDLE;
    instance_ = VK_NULL_HANDLE;
    physical_device_ = VK_NULL_HANDLE;
    throw std::runtime_error("vkCreateDescriptorPool failed");
  }

  initialized_ = true;
}

void VulkanDevice::destroy() {
  if (device_ != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(device_);
  }

  if (descriptor_pool_ != VK_NULL_HANDLE) {
    vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
    descriptor_pool_ = VK_NULL_HANDLE;
  }

  if (command_pool_ != VK_NULL_HANDLE) {
    vkDestroyCommandPool(device_, command_pool_, nullptr);
    command_pool_ = VK_NULL_HANDLE;
  }

  if (device_ != VK_NULL_HANDLE) {
    vkDestroyDevice(device_, nullptr);
    device_ = VK_NULL_HANDLE;
  }

  compute_queue_ = VK_NULL_HANDLE;
  physical_device_ = VK_NULL_HANDLE;

  if (instance_ != VK_NULL_HANDLE) {
    vkDestroyInstance(instance_, nullptr);
    instance_ = VK_NULL_HANDLE;
  }

  initialized_ = false;
}

// --- Buffer management ---

VulkanDevice::Buffer VulkanDevice::create_device_local_buffer(VkDeviceSize size) {
  VkBufferCreateInfo buf_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  buf_info.size = size;
  buf_info.usage =
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
      VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  Buffer buf;
  buf.size = size;

  if (vkCreateBuffer(device_, &buf_info, nullptr, &buf.buffer) != VK_SUCCESS) {
    throw std::runtime_error("vkCreateBuffer (device-local) failed");
  }

  VkMemoryRequirements mem_reqs{};
  vkGetBufferMemoryRequirements(device_, buf.buffer, &mem_reqs);

  VkMemoryAllocateInfo alloc_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  alloc_info.allocationSize = mem_reqs.size;
  alloc_info.memoryTypeIndex = find_memory_type(
      mem_reqs.memoryTypeBits,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  if (vkAllocateMemory(device_, &alloc_info, nullptr, &buf.memory) !=
      VK_SUCCESS) {
    vkDestroyBuffer(device_, buf.buffer, nullptr);
    buf.buffer = VK_NULL_HANDLE;
    throw std::runtime_error("vkAllocateMemory (device-local) failed");
  }

  vkBindBufferMemory(device_, buf.buffer, buf.memory, 0);
  return buf;
}

VulkanDevice::Buffer VulkanDevice::create_host_visible_buffer(VkDeviceSize size) {
  VkBufferCreateInfo buf_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  buf_info.size = size;
  buf_info.usage =
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
      VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  Buffer buf;
  buf.size = size;

  if (vkCreateBuffer(device_, &buf_info, nullptr, &buf.buffer) != VK_SUCCESS) {
    throw std::runtime_error("vkCreateBuffer (host-visible) failed");
  }

  VkMemoryRequirements mem_reqs{};
  vkGetBufferMemoryRequirements(device_, buf.buffer, &mem_reqs);

  VkMemoryAllocateInfo alloc_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  alloc_info.allocationSize = mem_reqs.size;
  alloc_info.memoryTypeIndex = find_memory_type(
      mem_reqs.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  if (vkAllocateMemory(device_, &alloc_info, nullptr, &buf.memory) !=
      VK_SUCCESS) {
    vkDestroyBuffer(device_, buf.buffer, nullptr);
    buf.buffer = VK_NULL_HANDLE;
    throw std::runtime_error("vkAllocateMemory (host-visible) failed");
  }

  vkBindBufferMemory(device_, buf.buffer, buf.memory, 0);

  vkMapMemory(device_, buf.memory, 0, size, 0, &buf.mapped);
  return buf;
}

void VulkanDevice::upload_to_device(const Buffer& dst, const void* data,
                                     VkDeviceSize size) {
  Buffer staging = create_host_visible_buffer(size);
  std::memcpy(staging.mapped, data, static_cast<size_t>(size));

  VkCommandBuffer cmd = allocate_command_buffer();
  begin_command_buffer(cmd);

  VkBufferCopy region{};
  region.srcOffset = 0;
  region.dstOffset = 0;
  region.size = size;
  vkCmdCopyBuffer(cmd, staging.buffer, dst.buffer, 1, &region);

  end_command_buffer(cmd);
  submit_and_wait(cmd);

  destroy_buffer(staging);
}

void VulkanDevice::upload_to_host_visible(const Buffer& dst, const void* data,
                                           VkDeviceSize size) {
  if (!dst.mapped) {
    throw std::runtime_error("upload_to_host_visible: buffer is not mapped");
  }
  std::memcpy(dst.mapped, data, static_cast<size_t>(size));
}

void VulkanDevice::download_from_device(const Buffer& src, void* data,
                                         VkDeviceSize size) {
  Buffer staging = create_host_visible_buffer(size);

  VkCommandBuffer cmd = allocate_command_buffer();
  begin_command_buffer(cmd);

  VkBufferCopy region{};
  region.srcOffset = 0;
  region.dstOffset = 0;
  region.size = size;
  vkCmdCopyBuffer(cmd, src.buffer, staging.buffer, 1, &region);

  end_command_buffer(cmd);
  submit_and_wait(cmd);

  std::memcpy(data, staging.mapped, static_cast<size_t>(size));
  destroy_buffer(staging);
}

void VulkanDevice::destroy_buffer(Buffer& buf) {
  if (buf.mapped != nullptr && buf.memory != VK_NULL_HANDLE) {
    vkUnmapMemory(device_, buf.memory);
    buf.mapped = nullptr;
  }
  if (buf.buffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(device_, buf.buffer, nullptr);
    buf.buffer = VK_NULL_HANDLE;
  }
  if (buf.memory != VK_NULL_HANDLE) {
    vkFreeMemory(device_, buf.memory, nullptr);
    buf.memory = VK_NULL_HANDLE;
  }
  buf.size = 0;
}

// --- Command buffers ---

VkCommandBuffer VulkanDevice::allocate_command_buffer() {
  VkCommandBufferAllocateInfo alloc_info{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  alloc_info.commandPool = command_pool_;
  alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandBufferCount = 1;

  VkCommandBuffer cmd = VK_NULL_HANDLE;
  if (vkAllocateCommandBuffers(device_, &alloc_info, &cmd) != VK_SUCCESS) {
    throw std::runtime_error("vkAllocateCommandBuffers failed");
  }
  return cmd;
}

void VulkanDevice::begin_command_buffer(VkCommandBuffer cmd) {
  VkCommandBufferBeginInfo begin_info{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &begin_info);
}

void VulkanDevice::end_command_buffer(VkCommandBuffer cmd) {
  vkEndCommandBuffer(cmd);
}

void VulkanDevice::submit_and_wait(VkCommandBuffer cmd) {
  VkSubmitInfo submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &cmd;

  VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence = VK_NULL_HANDLE;
  vkCreateFence(device_, &fence_info, nullptr, &fence);

  vkQueueSubmit(compute_queue_, 1, &submit_info, fence);
  vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX);

  vkDestroyFence(device_, fence, nullptr);
  vkFreeCommandBuffers(device_, command_pool_, 1, &cmd);
}

// --- Pipeline ---

VkShaderModule VulkanDevice::create_shader_module(
    const std::vector<std::uint32_t>& spirv) {
  VkShaderModuleCreateInfo module_info{
      VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  module_info.codeSize = spirv.size() * sizeof(std::uint32_t);
  module_info.pCode = spirv.data();

  VkShaderModule module = VK_NULL_HANDLE;
  if (vkCreateShaderModule(device_, &module_info, nullptr, &module) !=
      VK_SUCCESS) {
    throw std::runtime_error("vkCreateShaderModule failed");
  }
  return module;
}

void VulkanDevice::destroy_shader_module(VkShaderModule module) {
  if (module != VK_NULL_HANDLE) {
    vkDestroyShaderModule(device_, module, nullptr);
  }
}

VkDescriptorSetLayout VulkanDevice::create_descriptor_set_layout(
    const std::vector<VkDescriptorSetLayoutBinding>& bindings) {
  VkDescriptorSetLayoutCreateInfo layout_info{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
  layout_info.pBindings = bindings.data();

  VkDescriptorSetLayout layout = VK_NULL_HANDLE;
  if (vkCreateDescriptorSetLayout(device_, &layout_info, nullptr, &layout) !=
      VK_SUCCESS) {
    throw std::runtime_error("vkCreateDescriptorSetLayout failed");
  }
  return layout;
}

void VulkanDevice::destroy_descriptor_set_layout(VkDescriptorSetLayout layout) {
  if (layout != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(device_, layout, nullptr);
  }
}

VkPipelineLayout VulkanDevice::create_pipeline_layout(
    VkDescriptorSetLayout descriptor_set_layout,
    VkDeviceSize push_constant_range_size) {
  VkPipelineLayoutCreateInfo layout_info{
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  layout_info.setLayoutCount = 1;
  layout_info.pSetLayouts = &descriptor_set_layout;

  VkPushConstantRange push_range{};
  if (push_constant_range_size > 0) {
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.offset = 0;
    push_range.size = static_cast<uint32_t>(push_constant_range_size);
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push_range;
  }

  VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
  if (vkCreatePipelineLayout(device_, &layout_info, nullptr,
                              &pipeline_layout) != VK_SUCCESS) {
    throw std::runtime_error("vkCreatePipelineLayout failed");
  }
  return pipeline_layout;
}

void VulkanDevice::destroy_pipeline_layout(VkPipelineLayout layout) {
  if (layout != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(device_, layout, nullptr);
  }
}

VkPipeline VulkanDevice::create_compute_pipeline(
    VkShaderModule shader,
    VkPipelineLayout pipeline_layout,
    const char* entry_point) {
  VkPipelineShaderStageCreateInfo stage_info{
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stage_info.module = shader;
  stage_info.pName = entry_point;

  VkComputePipelineCreateInfo pipeline_info{
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  pipeline_info.stage = stage_info;
  pipeline_info.layout = pipeline_layout;

  VkPipeline pipeline = VK_NULL_HANDLE;
  if (vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info,
                                nullptr, &pipeline) != VK_SUCCESS) {
    throw std::runtime_error("vkCreateComputePipelines failed");
  }
  return pipeline;
}

void VulkanDevice::destroy_pipeline(VkPipeline pipeline) {
  if (pipeline != VK_NULL_HANDLE) {
    vkDestroyPipeline(device_, pipeline, nullptr);
  }
}

VkDescriptorSet VulkanDevice::allocate_descriptor_set(
    VkDescriptorSetLayout layout) {
  VkDescriptorSetAllocateInfo alloc_info{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  alloc_info.descriptorPool = descriptor_pool_;
  alloc_info.descriptorSetCount = 1;
  alloc_info.pSetLayouts = &layout;

  VkDescriptorSet set = VK_NULL_HANDLE;
  if (vkAllocateDescriptorSets(device_, &alloc_info, &set) != VK_SUCCESS) {
    throw std::runtime_error("vkAllocateDescriptorSets failed");
  }
  return set;
}

void VulkanDevice::update_descriptor_set(
    VkDescriptorSet set,
    uint32_t binding,
    const Buffer& buffer,
    VkDeviceSize offset,
    VkDeviceSize range) {
  VkDescriptorBufferInfo buf_info{};
  buf_info.buffer = buffer.buffer;
  buf_info.offset = offset;
  buf_info.range = range;

  VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  write.dstSet = set;
  write.dstBinding = binding;
  write.dstArrayElement = 0;
  write.descriptorCount = 1;
  write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  write.pBufferInfo = &buf_info;

  vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
}

// --- Private helpers ---

uint32_t VulkanDevice::find_memory_type(
    uint32_t type_filter,
    VkMemoryPropertyFlags properties) const {
  VkPhysicalDeviceMemoryProperties mem_props{};
  vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_props);

  for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
    if ((type_filter & (1u << i)) &&
        (mem_props.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }
  throw std::runtime_error("find_memory_type: no suitable memory type found");
}

#endif  // SPOCK_HAS_VULKAN && !defined(SPOCK_VULKAN_STUB)

}  // namespace spock::runtime

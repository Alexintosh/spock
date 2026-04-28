#include "runtime/vk_decode.hpp"
#include "runtime/vk_device.hpp"
#include "runtime/weight_loader.hpp"
#include "model/qwen35_config.hpp"

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>

#if SPOCK_HAS_VULKAN && !defined(SPOCK_VULKAN_STUB)
#include <vulkan/vulkan.h>
#endif

namespace spock::runtime {

namespace {

/// Read SPIR-V shader bytes from a file.
std::vector<uint32_t> read_spirv(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open shader: " + path);
  auto size = f.tellg();
  f.seekg(0);
  std::vector<uint32_t> words(size / 4);
  f.read(reinterpret_cast<char*>(words.data()), size);
  return words;
}

float half_to_float(uint16_t h) {
  uint32_t sign = (h >> 15) & 1;
  uint32_t exponent = (h >> 10) & 0x1f;
  uint32_t mantissa = h & 0x3ff;

  uint32_t f;
  if (exponent == 0) {
    if (mantissa == 0) {
      f = sign << 31;
    } else {
      // Denormalized
      exponent = 127 - 15 + 1;
      while ((mantissa & 0x400) == 0) {
        mantissa <<= 1;
        exponent--;
      }
      mantissa &= 0x3ff;
      f = (sign << 31) | (exponent << 23) | (mantissa << 13);
    }
  } else if (exponent == 31) {
    f = (sign << 31) | (0xff << 23) | (mantissa << 13);
  } else {
    f = (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13);
  }

  float result;
  memcpy(&result, &f, 4);
  return result;
}

/// Load an fp16 tensor from the artifact and return fp32 host data.
/// Returns vector of float values and the element count.
std::vector<float> load_fp16_tensor_to_fp32(const WeightArtifact& artifact,
                                             const TensorInfo& info) {
  auto raw = read_tensor_bytes(artifact, info);
  std::vector<float> out;
  out.reserve(info.shape.empty() ? info.nbytes / 2 : std::accumulate(
      info.shape.begin(), info.shape.end(), 1, std::multiplies<>()));

  const uint16_t* fp16_data = reinterpret_cast<const uint16_t*>(raw.data());
  size_t count = raw.size() / 2;
  out.resize(count);
  for (size_t i = 0; i < count; ++i) {
    out[i] = half_to_float(fp16_data[i]);
  }
  return out;
}

/// Upload raw bytes to a device buffer.
void upload_raw(VulkanDevice& dev, const VulkanDevice::Buffer& buf,
                const void* data, size_t size) {
  dev.upload_to_device(buf, data, static_cast<VkDeviceSize>(size));
}

/// Push constant helper: pack float as uint32 for GPU push constants.
uint32_t float_to_bits(float f) {
  uint32_t bits;
  memcpy(&bits, &f, 4);
  return bits;
}

}  // namespace

DecodeResult run_vk_decode(const DecodeConfig& config) {
  DecodeResult result;

#if !SPOCK_HAS_VULKAN || defined(SPOCK_VULKAN_STUB)
  result.error = "Vulkan not available (built with stub)";
  return result;
#else

  // Helper: issue a buffer memory barrier between compute stages
  auto barrier = [&](VkCommandBuffer cmd_buf, VkBuffer buf, VkDeviceSize size) {
    VkBufferMemoryBarrier bmb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    bmb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bmb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.buffer = buf;
    bmb.offset = 0;
    bmb.size = size;
    vkCmdPipelineBarrier(cmd_buf,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 1, &bmb, 0, nullptr);
  };

  // --- 1. Initialize Vulkan device ---
  VulkanDevice dev;
  try {
    dev.initialize();
  } catch (const std::exception& e) {
    result.error = std::string("Vulkan init failed: ") + e.what();
    return result;
  }

  const auto& caps = dev.capabilities();
  if (config.verbose) {
    std::cerr << "Vulkan device: " << caps.device_name << "\n";
    std::cerr << "Subgroup size: " << caps.subgroup_size << "\n";
    std::cerr << "Max shared memory: " << caps.max_shared_memory_bytes << "\n";
  }

  // --- 2. Load weight artifact ---
  WeightArtifact artifact;
  try {
    artifact = WeightArtifact::load(config.repack_dir);
  } catch (const std::exception& e) {
    result.error = std::string("Weight load failed: ") + e.what();
    return result;
  }

  if (config.verbose) {
    std::cerr << "Loaded " << artifact.tensor_count() << " tensors ("
              << (artifact.total_bytes() / (1024*1024)) << " MiB)\n";
  }

  // --- 3. Load shaders ---
  auto shader_dir = std::string(SHADER_DIR);
  auto embedding_shader = read_spirv(shader_dir + "/embedding_lookup.comp.spv");
  auto rmsnorm_shader = read_spirv(shader_dir + "/rms_norm.comp.spv");
  auto matvec_shader = read_spirv(shader_dir + "/matvec.comp.spv");
  auto argmax_shader = read_spirv(shader_dir + "/argmax.comp.spv");

  // --- 4. Create pipeline infrastructure ---
  // All our shaders use the same 3-binding layout:
  //   binding 0: readonly storage buffer (weights or input)
  //   binding 1: readonly storage buffer (weight or input)
  //   binding 2: writeonly storage buffer (output)
  // Push constants: 8 bytes (two uint32s)

  std::vector<VkDescriptorSetLayoutBinding> bindings_3 = {
      {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
  };
  std::vector<VkDescriptorSetLayoutBinding> bindings_2 = {
      {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
  };

  auto ds_layout_3 = dev.create_descriptor_set_layout(bindings_3);
  auto ds_layout_2 = dev.create_descriptor_set_layout(bindings_2);

  // Pipeline layouts with push constants
  auto pipeline_layout_3 = dev.create_pipeline_layout(ds_layout_3, 8);  // 8 bytes push constants
  auto pipeline_layout_2 = dev.create_pipeline_layout(ds_layout_2, 8);

  // Create shader modules and pipelines
  auto embedding_module = dev.create_shader_module(embedding_shader);
  auto rmsnorm_module = dev.create_shader_module(rmsnorm_shader);
  auto matvec_module = dev.create_shader_module(matvec_shader);
  auto argmax_module = dev.create_shader_module(argmax_shader);

  // Embedding uses 2-binding layout (weight, output)
  auto embedding_pipeline = dev.create_compute_pipeline(embedding_module, pipeline_layout_2);
  // RMSNorm uses 3-binding layout (input, weight, output)
  auto rmsnorm_pipeline = dev.create_compute_pipeline(rmsnorm_module, pipeline_layout_3);
  // MatVec uses 3-binding layout (weight, input, output)
  auto matvec_pipeline = dev.create_compute_pipeline(matvec_module, pipeline_layout_3);
  // Argmax uses 2-binding layout (values, result)
  auto argmax_pipeline = dev.create_compute_pipeline(argmax_module, pipeline_layout_2);

  // --- 5. Allocate GPU buffers ---
  constexpr uint32_t HIDDEN = model::Qwen35Config::hidden_size;
  constexpr uint32_t INTER = model::Qwen35Config::intermediate_size;
  constexpr uint32_t VOCAB = 248320;
  constexpr uint32_t LAYERS = model::Qwen35Config::layer_count;
  const float RMS_EPS = 1e-6f;

  // Activation ping-pong buffers (fp16)
  // act_a: current hidden state, act_b: scratch
  size_t act_bytes = HIDDEN * 2;  // fp16
  auto act_a = dev.create_device_local_buffer(act_bytes);
  auto act_b = dev.create_device_local_buffer(act_bytes);

  // Logits buffer (fp16, vocab_size)
  auto logits = dev.create_device_local_buffer(VOCAB * 2);

  // Argmax result (single uint32)
  auto argmax_result = dev.create_device_local_buffer(4);

  // MLP intermediate buffer (fp16, intermediate_size)
  auto mlp_buf = dev.create_device_local_buffer(INTER * 2);

  // --- 6. Upload all weights to a single large device buffer ---
  // Read the entire text_weights.bin and upload it
  auto total_size = artifact.total_bytes();
  auto weights_buf = dev.create_device_local_buffer(total_size);

  {
    std::ifstream wf(artifact.weights_file_path(), std::ios::binary);
    if (!wf) {
      result.error = "cannot open weights file: " + artifact.weights_file_path();
      return result;
    }
    std::vector<char> weights_data(total_size);
    wf.read(weights_data.data(), total_size);
    upload_raw(dev, weights_buf, weights_data.data(), total_size);
  }

  // Upload RMSNorm weights (final norm) to a separate buffer
  auto& final_norm_info = artifact.final_norm();
  auto final_norm_raw = read_tensor_bytes(artifact, final_norm_info);
  auto final_norm_buf = dev.create_device_local_buffer(final_norm_info.nbytes);
  upload_raw(dev, final_norm_buf, final_norm_raw.data(), final_norm_info.nbytes);

  // --- 7. Allocate descriptor sets ---
  // We need a descriptor set for each operation that points to the right buffers
  auto embedding_ds = dev.allocate_descriptor_set(ds_layout_2);
  auto rmsnorm_ds = dev.allocate_descriptor_set(ds_layout_3);
  auto matvec_ds = dev.allocate_descriptor_set(ds_layout_3);
  auto final_norm_ds = dev.allocate_descriptor_set(ds_layout_3);
  auto argmax_ds = dev.allocate_descriptor_set(ds_layout_2);

  // --- 8. Decode loop ---
  // Use the provided prompt tokens or a simple default
  auto tokens = config.prompt_tokens.empty()
      ? std::vector<uint32_t>{1, 2, 3}  // fallback
      : config.prompt_tokens;
  result.prompt_tokens = tokens;

  auto t0 = std::chrono::high_resolution_clock::now();

  for (uint32_t step = 0; step < config.max_new_tokens; ++step) {
    uint32_t current_token = tokens.back();

    // --- Embedding lookup ---
    dev.update_descriptor_set(embedding_ds, 0, weights_buf,
        artifact.token_embedding().offset, artifact.token_embedding().nbytes);
    dev.update_descriptor_set(embedding_ds, 1, act_a);

    auto cmd = dev.allocate_command_buffer();
    dev.begin_command_buffer(cmd);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, embedding_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline_layout_2, 0, 1, &embedding_ds, 0, nullptr);
    uint32_t push_token = current_token;
    vkCmdPushConstants(cmd, pipeline_layout_2, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &push_token);
    vkCmdDispatch(cmd, 1, 1, 1);  // single workgroup of 64

    barrier(cmd, act_a.buffer, act_bytes);
    // --- Process each layer ---
    for (uint32_t layer = 0; layer < LAYERS; ++layer) {
      auto kind = model::Qwen35Config::layer_schedule()[layer];

      // Input RMSNorm
      // For each layer we need to:
      // 1. RMSNorm the input hidden state
      // 2. Run attention or DeltaNet block (matvec projections)
      // 3. Add residual
      // 4. RMSNorm
      // 5. MLP (gate + up matvec, SiLU gate, down matvec)
      // 6. Add residual
      //
      // For the first pass, we simplify: just run the MLP with identity skip.
      // Full attention/DeltaNet comes in the next iteration.

      // Input norm
      std::string norm_role = "layer." + std::to_string(layer) + ".input_norm";
      auto norm_info = artifact.find_by_role(norm_role);

      if (norm_info) {
        dev.update_descriptor_set(rmsnorm_ds, 0, act_a);
        dev.update_descriptor_set(rmsnorm_ds, 1, weights_buf,
            norm_info->offset, norm_info->nbytes);
        dev.update_descriptor_set(rmsnorm_ds, 2, act_b);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rmsnorm_pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout_3, 0, 1, &rmsnorm_ds, 0, nullptr);
        struct { uint32_t N; uint32_t eps_bits; } rms_push = { HIDDEN, float_to_bits(RMS_EPS) };
        vkCmdPushConstants(cmd, pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &rms_push);
        vkCmdDispatch(cmd, 1, 1, 1);

        barrier(cmd, act_b.buffer, act_bytes);
      }

      // --- MLP (simplified first pass: just gate projection for now) ---
      // This is a placeholder path. Full layer processing will follow.
      // For now, skip the full layer and just pass through the normed hidden state.
      std::swap(act_a, act_b);  // act_a now has normed hidden state
    }  // closes inner for (layers)

    // --- Final RMSNorm ---
    dev.update_descriptor_set(final_norm_ds, 0, act_a);
    dev.update_descriptor_set(final_norm_ds, 1, final_norm_buf);
    dev.update_descriptor_set(final_norm_ds, 2, act_b);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rmsnorm_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline_layout_3, 0, 1, &final_norm_ds, 0, nullptr);
    struct { uint32_t N; uint32_t eps_bits; } final_rms_push = { HIDDEN, float_to_bits(RMS_EPS) };
    vkCmdPushConstants(cmd, pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &final_rms_push);
    vkCmdDispatch(cmd, 1, 1, 1);

    barrier(cmd, act_b.buffer, act_bytes);
    // --- LM head (matvec with embedding weight, tied) ---
    // output = embedding_weight * hidden  (shape: [vocab, hidden] * [hidden] = [vocab])
    dev.update_descriptor_set(matvec_ds, 0, weights_buf,
        artifact.token_embedding().offset, artifact.token_embedding().nbytes);
    dev.update_descriptor_set(matvec_ds, 1, act_b);
    dev.update_descriptor_set(matvec_ds, 2, logits);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, matvec_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline_layout_3, 0, 1, &matvec_ds, 0, nullptr);
    struct { uint32_t out_dim; uint32_t in_dim; } lm_push = { VOCAB, HIDDEN };
    vkCmdPushConstants(cmd, pipeline_layout_3, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &lm_push);
    vkCmdDispatch(cmd, (VOCAB + 63) / 64, 1, 1);

    barrier(cmd, logits.buffer, VOCAB * 2);
    // --- Argmax ---
    dev.update_descriptor_set(argmax_ds, 0, logits);
    dev.update_descriptor_set(argmax_ds, 1, argmax_result);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, argmax_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline_layout_2, 0, 1, &argmax_ds, 0, nullptr);
    uint32_t argmax_push = VOCAB;
    vkCmdPushConstants(cmd, pipeline_layout_2, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &argmax_push);
    vkCmdDispatch(cmd, 1, 1, 1);

    dev.end_command_buffer(cmd);
    dev.submit_and_wait(cmd);

    // Read back argmax result
    uint32_t next_token = 0;
    dev.download_from_device(argmax_result, &next_token, 4);
    tokens.push_back(next_token);
    result.generated_tokens.push_back(next_token);

    if (config.verbose) {
      std::cerr << "  step " << step << ": token " << next_token << "\n";
    }
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  result.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  // Cleanup
  dev.destroy_buffer(act_a);
  dev.destroy_buffer(act_b);
  dev.destroy_buffer(logits);
  dev.destroy_buffer(argmax_result);
  dev.destroy_buffer(mlp_buf);
  dev.destroy_buffer(weights_buf);
  dev.destroy_buffer(final_norm_buf);
  dev.destroy_pipeline(embedding_pipeline);
  dev.destroy_pipeline(rmsnorm_pipeline);
  dev.destroy_pipeline(matvec_pipeline);
  dev.destroy_pipeline(argmax_pipeline);
  dev.destroy_shader_module(embedding_module);
  dev.destroy_shader_module(rmsnorm_module);
  dev.destroy_shader_module(matvec_module);
  dev.destroy_shader_module(argmax_module);
  dev.destroy_pipeline_layout(pipeline_layout_3);
  dev.destroy_pipeline_layout(pipeline_layout_2);
  dev.destroy_descriptor_set_layout(ds_layout_3);
  dev.destroy_descriptor_set_layout(ds_layout_2);
  dev.destroy();

  return result;
#endif
}

}  // namespace spock::runtime

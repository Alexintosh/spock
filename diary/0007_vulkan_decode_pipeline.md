# 0007: Vulkan Decode Pipeline Bring-Up

## Goal

The seventh phase brings up the Vulkan compute runtime, weight loading pipeline, and first end-to-end GPU dispatch. The previous phases proved the weight artifact is correct (0006). This phase proves the GPU can receive those weights, compile shaders, and execute a decode loop — even though per-layer processing is not yet complete.

This is the first phase that runs code on the RX 6750 XT.

## Why A Vulkan Device Wrapper Matters

Raw Vulkan is a C API that requires manual lifecycle management for every object: instances, physical devices, logical devices, queues, command pools, descriptor pools, buffers, memories, shader modules, pipeline layouts, pipelines, descriptor sets, and command buffers. Each has a create call and a destroy call. Missing a destroy leaks GPU resources. Calling destroy in the wrong order causes validation errors or crashes.

The `VulkanDevice` class wraps all of this in RAII. It owns every Vulkan object it creates. The destructor tears them down. Copy is deleted; move is supported. This eliminates an entire class of bugs before the first shader runs.

## The Vulkan Device

`src/runtime/vk_device.hpp` and `src/runtime/vk_device.cpp` provide the `VulkanDevice` class. It encapsulates:

- **Instance creation** with validation layers in debug builds.
- **Physical device selection** — picks the first discrete GPU that exposes a compute queue.
- **Logical device** with a single compute queue family.
- **Command pool** for allocating and recycling command buffers.
- **Descriptor pool** with a generous allocation for storage buffer descriptors.
- **Buffer management** — `create_device_local_buffer` for GPU-resident data, `create_host_visible_buffer` for CPU-accessible data, `upload_to_device` for staging uploads, `download_from_device` for readback.
- **Pipeline creation** — shader module compilation, descriptor set layout, pipeline layout with optional push constants, and compute pipeline assembly.
- **Submit helpers** — `allocate_command_buffer`, `begin_command_buffer`, `end_command_buffer`, `submit_and_wait` for synchronous dispatch.

The class also exposes `VulkanCapabilities`: device name, API version, subgroup size, max shared memory, max workgroup invocations. These are queried at init time and used to validate shader dispatch parameters.

When Vulkan is not available at build time, a stub class compiles instead, allowing the rest of the project to build without a Vulkan SDK.

## JSON Parser

The repack manifest is JSON. The project has a policy of zero external dependencies for the C++ runtime. Rather than adding nlohmann/json or rapidjson, `src/runtime/json_parse.hpp` and `src/runtime/json_parse.cpp` implement a minimal recursive-descent parser.

It supports exactly the subset the manifest needs:

- Objects (string keys, any values).
- Arrays.
- Strings (with escape sequences).
- Numbers (integers and doubles).
- Booleans and null.

The manifest does not use floating-point numbers for offsets or sizes, so the parser does not need to handle precision edge cases. It produces a `JsonValue` variant that supports typed access (`as_string`, `as_int`, `as_array`, `as_object`) and optional field lookup (`get("key")` returns nullptr for missing keys).

## Weight Loader

`src/runtime/weight_loader.hpp` and `src/runtime/weight_loader.cpp` implement `WeightArtifact`. This class:

1. Reads `text_repack_manifest.json` from the repack directory.
2. Parses the tensor list: role path, name, offset, byte size, dtype, shape.
3. Builds two indexes: by role path (`"layer.0.input_norm"`) and by manifest name.
4. Provides typed accessors: `find_by_role`, `find_by_name`, `find_by_state_dict_key`.
5. Exposes convenience methods: `token_embedding()`, `final_norm()`, `total_bytes()`, `tensor_count()`.
6. The companion function `read_tensor_bytes` reads raw bytes from `text_weights.bin` at the aligned offset recorded in the manifest.

The weight loader does not interpret tensor values. It provides byte ranges and metadata. The decode pipeline interprets those bytes as FP16 or FP32 depending on the tensor role.

## Compute Shaders

Four GLSL compute shaders implement the initial decode kernels.

### embedding_lookup.comp

Fetches one row from the `[vocab_size, hidden_size]` FP16 embedding matrix. Dispatched as a single workgroup of 64 invocations. Each invocation loads 16 FP16 values (1024 / 64 = 16). The token ID is passed as a push constant. The shader is simple by design: no reduction, no accumulation, just a strided copy from the weight buffer to the output buffer.

### rms_norm.comp

Computes `output = input * rsqrt(mean(input^2) + eps) * (1 + weight)` where weight is the learned RMSNorm scale. Dispatched as a single workgroup of 256 invocations.

Key design choices:

- FP32 accumulation: all partial sums and the norm factor are computed in FP32, even though input and output are FP16. This prevents precision loss in the normalization.
- Shared-memory tree reduction: 256 partial sums are written to shared memory, then reduced with a log-step tree (stride 128, 64, 32, ..., 1). This avoids requiring subgroup operations that depend on specific SPIR-V extensions.
- Epsilon packed as `uint` in push constants: push constants are 32-bit words. Passing `1e-6f` as `uintBitsToFloat(epsilon_bits)` avoids introducing a float uniform block for a single scalar.

### matvec.comp

Matrix-vector multiply for decode (batch-1): `output[i] = dot(W[i,*], x)`. The weight matrix is row-major `[out_dim, in_dim]`. Dispatched as `ceil(out_dim / 64)` workgroups. Each invocation computes one output element by performing a full dot product across the input dimension.

This is the workhorse kernel. Every projection in the model — QKV, output projection, gate, up, down, LM head — routes through matvec. The first pass uses a simple scalar loop over the input dimension. Later optimizations can tile for cache locality or use subgroup operations, but correctness comes first.

### argmax.comp

Finds the index of the maximum FP16 value in a buffer of length N. Dispatched as a single workgroup of 256 invocations. Each invocation scans `ceil(N / 256)` values, tracking the local maximum and its index. A shared-memory tree reduction selects the global maximum.

This kernel produces the token ID for each decode step.

## Decode Pipeline

`src/runtime/vk_decode.cpp` implements `run_vk_decode`, and `apps/spock-decode.cpp` provides the CLI entry point.

The decode loop for each token:

1. **Embedding lookup**: Read the row for `current_token` from the embedding weight matrix into `act_a`.
2. **Per-layer processing** (24 layers): For each layer, run input RMSNorm on the hidden state. The normed result goes to `act_b`. In this first pass, the rest of the layer (projections, attention/DeltaNet, MLP, residuals) is skipped. The normed hidden state is passed through as-is.
3. **Final RMSNorm**: Apply the model's final layer norm.
4. **LM head**: Matvec with the tied embedding weight matrix `[vocab_size, hidden_size]` times the hidden state `[hidden_size]`, producing logits `[vocab_size]`.
5. **Argmax**: Select the token with the highest logit. Read it back to the host. Append to the token sequence.

All GPU work for one token is submitted as a single command buffer with pipeline barriers between stages. The host waits for completion after argmax before reading back the result.

## Weight Upload

The entire `text_weights.bin` (approximately 1.4 GiB, 320 tensors) is read into host memory and uploaded to a single device-local buffer via a staging buffer. Individual tensors are accessed by binding descriptor sets with byte offsets into this large buffer. This avoids creating 320 separate GPU buffers and simplifies descriptor management.

The final norm weight is uploaded to a separate small buffer because it is used in a different descriptor set binding pattern than the per-layer norms.

## Observed GPU Properties

Running on the AMD RX 6750 XT:

- **Subgroup size**: 64. The original project plan assumed subgroup size 32. The observed value of 64 affects workgroup sizing and shared-memory layout. All current shaders use explicit workgroup sizes (64 or 256) and shared-memory tree reduction, so this is not a correctness issue, but future subgroup-based optimizations must account for subgroup size 64.
- **Max shared memory per workgroup**: 65,536 bytes (64 KiB). Confirmed via `VK_API_VERSION_1_3` properties.
- **Device name**: Reported as the full AMD board name via Vulkan physical device properties.

## Current Limitations

The per-layer processing is a skeleton. Each layer runs input RMSNorm but skips all projections, attention, DeltaNet recurrence, MLP, and residual connections. The normed hidden state passes through unchanged.

Consequences:

- The hidden state reaching the final norm is a function of only the embedding and repeated RMSNorm, not of the model's learned representations.
- All generated tokens are 0. This is expected and correct given the current pipeline: the LM head projects a meaningless hidden state, and argmax picks whatever index happens to be numerically largest.
- The pipeline proves that Vulkan dispatch works end-to-end (shader compilation, buffer management, command submission, readback). It does not yet produce valid tokens.

## What This Enables

The GPU pipeline is proven. The skeleton demonstrates:

- Vulkan device initialization and capability query.
- Weight artifact loading and GPU upload.
- SPIR-V shader compilation and pipeline creation.
- Multi-stage command buffer dispatch with barriers.
- Host readback of compute results.

The next step is wiring the per-layer compute:

- QKV projections via matvec.
- DeltaNet recurrence for 18 linear-attention layers.
- Full self-attention with KV cache for 6 attention layers.
- MLP with gate/up projections, SiLU activation, and down projection.
- Residual adds between sublayers.
- FP16 activations with FP32 accumulation throughout.

The matvec kernel already exists. The work is in orchestrating the correct sequence of dispatches and managing intermediate buffers for each layer's activations.

## Verification

Run the GPU decode pipeline:

```sh
./build/spock-decode --repack-dir artifacts/spock-text-repack-qwen35-0p8b --max-new-tokens 4 --verbose
```

Expected output: 4 generated tokens, all 0, with verbose logging showing Vulkan device properties and per-step dispatch.

Full build and test:

```sh
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

All 11 tests pass. The existing test suite covers the weight pipeline, repack validation, and P0 parity. The Vulkan decode path is an application, not a test, and produces the expected all-zero output given the current skeleton.

# 0026: GPU-Side Chunk Init Clear — Removing the CPU Zero-Fill/Staging Bridge

## Goal

Eliminate the last CPU data touch for chunk-prefill computation on the
no-compare GPU-collected+tiled fast path: the allocation and zero-fill of
the chunk-prefill **init state** buffer.

Before this entry, the gated path (diaries 0020–0025) still initialized
`init_buf` — the zero-fp32 initial recurrent state consumed by the tiled
shader as binding 6 — via a host-visible buffer with CPU `memset`:

```cpp
auto init_buf = dev_.create_host_visible_buffer(sz_init);
memset(init_buf.mapped, 0, static_cast<size_t>(sz_init));
```

This CPU memset was the *last* host-side data write for the chunk-prefill
computation on the fast path. The input bridge (Q/K/V/g/beta collection)
was GPU-resident since diaries 0020/0021. The output bridge (final_state
readback, fp32→fp16 attn conversion, upload) was GPU-resident since
diary 0025. But the init state was still zeroed on CPU and staged across
the PCIe bus.

This entry replaces that with:

- A **device-local `init_buf`** allocation on the no-compare fast path,
  bypassing host-visible memory.
- A **`vkCmdFillBuffer`** call in the command buffer to zero the buffer
  on-device before the tiled shader dispatches.
- A **`VkBufferMemoryBarrier`** from `VK_PIPELINE_STAGE_TRANSFER_BIT` to
  `VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT` to ensure the fill completes
  before the shader reads `init_state`.

The host-visible + CPU memset path is preserved as a fallback for compare
diagnostics, non-tiled paths, and CPU-collected chunk input paths —
identical to the fallback strategy from diary 0025.

## Inference Concepts

### What is "init state" and why is it zero?

The DeltaNet chunk-rule computes a recurrent state update:

```
state[t+1] = decay(t) ⊙ state[t] + update(t)
```

For the current runtime path, each layer's prefill chunk-rule invocation
starts from the zero recurrent state for the prompt. The padded internal
chunk loop is handled inside `deltanet_chunk_prefill_tiled.comp`; binding
6 provides the initial `state[t]` baseline for that shader invocation.

Before this entry, that zero state was produced by CPU `memset`. After
this entry, on the no-compare fast path, it is produced by
`vkCmdFillBuffer` on the GPU — no CPU data touch.

### What this changes about the data flow

Before (all paths):

```
CPU: allocate host-visible init_buf
CPU: memset(init_buf, 0, sz_init)
CPU: (later) bind init_buf to shader binding 6
GPU: tiled shader reads init_state from host-visible init_buf
      → host-visible read by GPU (suboptimal, but functional)
```

After (no-compare fast path):

```
CPU: allocate device-local init_buf (no host mapping)
GPU: vkCmdFillBuffer(init_buf, 0, sz_init, 0)
GPU: barrier: fill → shader read
GPU: tiled shader reads init_state from device-local init_buf
      → purely device-local, no PCIe traffic for init data
```

After (fallback/compare paths): unchanged — host-visible + CPU memset.

### What this does NOT change

- The host **does not stop iterating layers or recording command buffers**.
  The runtime still calls `gpu_chunk_prefill_from_gpu_collect()` per layer,
  allocates per-layer `init_buf` and `out_buf`, records the fill → barrier →
  shader dispatch → copy → fp16 extraction sequence, submits, and waits.
- The kernel itself is unchanged. The tiled shader was not modified. It
  reads `init_state` from binding 6 exactly as before. The only difference
  is whether that binding points to a device-local or host-visible buffer,
  and whether its zero content was produced by CPU `memset` or GPU
  `vkCmdFillBuffer`.
- The init state lifetime is the same: per-layer, per-chunk. After the
  chunk completes, the buffer is freed (it is scoped to the function call).
- This is still **not full GPU offload**. The remaining CPU/host pieces
  documented in diary 0025 are unchanged: per-layer orchestration and
  submission, decode argmax/logit computation, diagnostic readbacks,
  host-visible handoff flags, and the broader megakernel fusion gap.

## Implementation Work Completed

### Single change: `src/runtime/vk_session.cpp`

The change is confined to `gpu_chunk_prefill_from_gpu_collect()` in
`src/runtime/vk_session.cpp`. Two edits:

**1. Buffer allocation split (lines ~3236–3244)**

Before:
```cpp
auto init_buf = dev_.create_host_visible_buffer(sz_init);
memset(init_buf.mapped, 0, static_cast<size_t>(sz_init));
```

After:
```cpp
VulkanDevice::Buffer init_buf;
if (use_gpu_handoff) {
  init_buf = dev_.create_device_local_buffer(sz_init);
} else {
  init_buf = dev_.create_host_visible_buffer(sz_init);
  memset(init_buf.mapped, 0, static_cast<size_t>(sz_init));
}
```

The `use_gpu_handoff` boolean was already defined earlier in the function
(added in diary 0025) as `tiled && !chunk_compare_active && !collect_compare_active`.
It gates both the device-local `out_buf` allocation (diary 0025) and now
the device-local `init_buf` allocation.

On the fast path (device-local), no `memset` call is made — the buffer is
uninitialized at allocation time. Zero-initialization happens via
`vkCmdFillBuffer` in the command buffer before the shader dispatch.

**2. Command-buffer fill + barrier (lines ~3283–3296)**

After the tiled-path command buffer is opened, before the shader pipeline
bind and dispatch:

```cpp
if (use_gpu_handoff) {
  vkCmdFillBuffer(cmd, init_buf.buffer, 0, sz_init, 0);
  VkBufferMemoryBarrier bmb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  bmb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  bmb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  bmb.buffer = init_buf.buffer;
  bmb.offset = 0;
  bmb.size = VK_WHOLE_SIZE;
  vkCmdPipelineBarrier(cmd,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0, 0, nullptr, 1, &bmb, 0, nullptr);
}
```

The barrier is essential: without it, the tiled shader could read stale or
undefined memory before the fill completes. Vulkan guarantees that
`vkCmdFillBuffer` writes are visible to subsequent commands in the same
command buffer after a correct barrier, but the barrier must be explicit.

The barrier transitions:
- **src**: `VK_PIPELINE_STAGE_TRANSFER_BIT` with `VK_ACCESS_TRANSFER_WRITE_BIT`
  — the fill-buffer write.
- **dst**: `VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT` with
  `VK_ACCESS_SHADER_READ_BIT` — the subsequent shader read of init_state.

Without this, on some implementations, the shader may observe partial or
no zero-fill.

### No other files modified

The change is 24 insertions, 2 deletions in a single file. No shaders were
modified, no pipelines added, no descriptor layouts changed. The shader's
binding 6 (`init_state`) already existed and is unchanged.

### Gate conditions (unchanged from diary 0025)

The device-local `init_buf` + `vkCmdFillBuffer` path activates under the
same conditions as the device-local `out_buf` and GPU handoff:

- `SPOCK_GPU_CHUNK_PREFILL=1`
- `SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1`
- `SPOCK_GPU_CHUNK_PREFILL_TILED=1`
- Neither `SPOCK_GPU_CHUNK_PREFILL_COMPARE=1` nor
  `SPOCK_GPU_COLLECT_PREFILL_COMPARE=1` is set

If any condition is false, the runtime falls back to host-visible `init_buf`
with CPU `memset` — identical to the behavior before this change.

## Verification

All commands run on the target RADV RX 6750 XT (NAVI22) hardware. The
verification was performed after the patch was applied as a working-tree
change.

### Build

```sh
source ~/.zshrc && cmake -S . -B build && cmake --build build -j
```

Passed cleanly. No compilation or linking errors.

### git diff --check

```sh
source ~/.zshrc && git diff --check
```

No whitespace errors.

### No-compare fast path (device-local init_buf + vkCmdFillBuffer active)

```sh
SPOCK_GPU_CHUNK_PREFILL=1 SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1 \
  SPOCK_GPU_CHUNK_PREFILL_TILED=1 \
  python3 tests/run_vk_decode_parity.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --reference tests/data/reference_tokens.jsonl \
  --ids short_correctness_001 \
  --max-new-tokens 16
```

```
{"status":"ok","checked":1,"failures":[]}
```

The device-local init_buf with GPU-side zero-fill produces correct decoded
output for `short_correctness_001` through 16 generated tokens. The init
state is zeroed entirely on-device with no CPU data touch.

### Compare fallback path (host-visible init_buf + CPU memset preserved)

```sh
SPOCK_GPU_CHUNK_PREFILL=1 SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1 \
  SPOCK_GPU_CHUNK_PREFILL_TILED=1 SPOCK_GPU_CHUNK_PREFILL_COMPARE=1 \
  python3 tests/run_vk_decode_parity.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --reference tests/data/reference_tokens.jsonl \
  --ids short_correctness_001 \
  --max-new-tokens 1
```

```
{"status":"ok","checked":1,"failures":[]}
```

The fallback path (host-visible init_buf + CPU memset) still works when
the compare flag is active. The device-local init_buf is disabled and the
existing CPU zero-fill path is used.

### CTest regression: GPU-collect suite (3/3 passed)

```sh
ctest --test-dir build --output-on-failure -R spock_vk_decode_gpu_collect_chunk_prefill
```

| Test | Status | Time |
|------|--------|------|
| `spock_vk_decode_gpu_collect_chunk_prefill_short` | Passed | 114.89 sec |
| `spock_vk_decode_gpu_collect_chunk_prefill_tiled` | Passed | 8.97 sec |
| `spock_vk_decode_gpu_collect_chunk_prefill_short_baseline` | Passed | 5.08 sec |

Key observations:

- The tiled CTest runtime (8.97 sec) is within noise of the 8.95 sec
  recorded in diary 0025 — this is expected, as the change replaces a
  CPU memset (negligible at this buffer size) with a GPU fill (negligible
  relative to the shader dispatch). No performance claim is made.
- The per-head submit test (`_short`) remained at ~115 sec.
- The baseline (no gates) remained at ~5 sec.

## Known Limitations

1. **Does not make full GPU offload complete.** This entry removes the
   last CPU data touch for chunk-prefill compute on the no-compare fast
   path. The remaining CPU/host pieces (unchanged since diary 0025) are:

   - **Per-layer orchestration and submission.** The host still iterates
     layers, records command-buffer sequences, binds descriptors, submits
     queue work, and waits for fences. Each DeltaNet layer requires host
     intervention to issue the `vkCmdFillBuffer`, the barrier, the tiled
     dispatch, the GPU copy, and the `deltanet_chunk_last_to_fp16.comp`
     dispatch.
   - **Decode argmax/logit.** The final token selection (argmax over
     logits) happens on CPU after GPU logit computation.
   - **Diagnostic/fallback paths.** The compare-diagnostic paths and
     non-tiled paths still use host-visible readback and CPU memset.
   - **Host-visible handoff flag.** The `gpu_chunk_handoff_ready_` boolean
     is a host-memory control flag, not device-native signaling.
   - **Broader megakernel fusion and persistent dispatch.** The per-layer
     orchestration model will remain a serial bottleneck until the entire
     decode pass is fused into a single persistent dispatch.

2. **Single-use scoped buffer.** The device-local `init_buf` is allocated
   per-layer invocation and freed after `gpu_chunk_prefill_from_gpu_collect()`
   returns. This is identical to the lifetime of the previous host-visible
   `init_buf` and the device-local `out_buf` from diary 0025. A future
   optimization could reuse a pool of device-local init buffers, but the
   allocation cost is negligible relative to the shader dispatch.

3. **Barrier is non-optional.** The `vkCmdFillBuffer` → shader-read
   barrier is strictly required for correctness on Vulkan. Without it, the
   shader may observe undefined memory. The cost of the barrier (one
   `vkCmdPipelineBarrier` call) is paid per layer, per chunk — small
   relative to the tiled dispatch itself, but not free.

4. **Change is trivial in isolation.** This is a 24-line change replacing
   a CPU memset with a GPU fill call. Its significance is architectural:
   it completes the removal of CPU data touches for the chunk-prefill
   computation on the fast path. The init state was the last buffer the
   CPU wrote to for chunk-prefill compute after diaries 0020/0021 (input)
   and 0025 (output).

5. **Coverage still limited.** Verified on `short_correctness_001` at
   16 tokens. Broader P0 coverage (including 512+ token prompts) is
   still pending, as in diary 0025.

## Next Work

### Near-term: Prompt coverage expansion

Run the no-compare GPU-collected+tiled fast path (with device-local init
and output handoff) on additional P0 prompts and longer sequences
(>512 tokens). Both the init clear and the output handoff are functionally
identical to their CPU-mediated predecessors, but empirical verification
across the prompt corpus is needed before defaulting.

### Medium-term: Toward the fused megakernel

The fast path is now fully GPU-resident for chunk-prefill *data* — no CPU
data touch from input collection through output handoff. But the
per-layer host orchestration (iterate, record, submit, wait) remains the
dominant overhead. The path to removing it is the same megakernel roadmap
described in diary 0025:

- Eliminate per-layer host submission by fusing all layers into a single
  persistent dispatch.
- Fold decode argmax/logit computation onto GPU.
- Remove host-visible control flags in favor of device timeline semaphores.

This entry does not make progress on that roadmap — it cleans up a
remaining data-touch detail on the existing path.

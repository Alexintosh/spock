# 0025: GPU-Resident Chunk-Prefill Output Handoff — Removing the CPU Readback/Upload Bridge

## Goal

Eliminate the CPU readback/re-upload bridge for chunk-prefill output and
state on the fastest gated path — where GPU collection, tiled single-dispatch,
and the no-compare fast path are all active
(`SPOCK_GPU_CHUNK_PREFILL=1`, `SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1`,
`SPOCK_GPU_CHUNK_PREFILL_TILED=1`, no compare-diagnostic flag set).

Before this entry, the gated GPU-collected+tiled path still went through a
CPU intermediary for chunk *output*: the tiled shader wrote final_state and
core_attn_out to `out_buf` (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT), the host
read them back, converted the last-token fp32 core_attn_out slice to fp16 via
`float_to_half` on the CPU, and uploaded `dn_state` and the fp16
attention slice back to device memory for `correct_last_token_hidden()`. This
CPU bridge was the last remaining host-side data touch for the chunk-prefill
computation itself on the no-compare path — the *input* bridge (Q/K/V/g/beta
collection) had been removed in diaries 0020/0021, but the *output* bridge
was still in place.

This entry replaces that bridge with:

- A **device-local chunk output buffer** (no host-visible round-trip) for the
  tiled shader to write final_state and core_attn_out.
- A **GPU-to-GPU `vkCmdCopyBuffer`** from that device-local buffer into the
  persistent `dn_state` buffer — no CPU state host readback.
- A **new helper shader `deltanet_chunk_last_to_fp16.comp`** that extracts
  the last-token fp32 core_attn_out slice on-device and converts it to fp16
  in a per-layer device-local `dn_chunk_attn_out_` buffer.
- A **device-local handoff path in `correct_last_token_hidden()`** that uses
  `gpu_chunk_handoff_ready_[dn_idx]` to copy that fp16 slice GPU-to-GPU into
  the B.dn_qkv V region, bypassing CPU vector conversion and upload entirely.

The CPU bridge is preserved as a fallback for the compare-diagnostic paths
(when `SPOCK_GPU_CHUNK_PREFILL_COMPARE=1` or
`SPOCK_GPU_COLLECT_PREFILL_COMPARE=1` is set and the host-visible readback
is needed for verification), for non-tiled paths, and for CPU-collected chunk
input paths that still go through the host side of the chunk-prefill
orchestration.

## Inference Concepts

### The output handoff problem

Deltanet chunk-prefill produces two output values per layer:

1. **final_state** — the recurrent state after processing all tokens in the
   chunk. This is a `[num_heads × k_dim × v_dim]` fp32 matrix per layer.
   It must feed the next autoregressive recurrent decode step.

2. **core_attn_out** — the per-token attention output for the entire chunk,
   `[num_heads × seq_len × v_dim]` fp32. The autoregressive decode loop
   only needs the **last token's** slice of this. That slice is converted to
   fp16 and placed into the B.dn_qkv V section where `correct_last_token_hidden()`
   expects it.

Before this entry, both outputs went through:
```
GPU: tiled shader writes to host-visible out_buf
CPU: vkMapMemory → read final_state into std::vector<float>
CPU: read core_attn_out, extract last-token slice, float_to_half convert
CPU: upload final_state to bufs_->dn_state (staging → device)
CPU: upload fp16 attn slice to B.dn_qkv V region (staging → device)
```

Each CPU step is a full frame boundary — staging buffer allocation, memory
barrier, queue submit for upload, fence wait. For 24 DeltaNet layers, this
is 24 pairs of download + upload per chunk, each paying driver overhead.

### Device-local handoff

After this entry, the no-compare GPU-collected+tiled path does:
```
GPU: tiled shader writes to device-local chunk output buffer
GPU: vkCmdCopyBuffer final_state slice → bufs_->dn_state
GPU: deltanet_chunk_last_to_fp16.comp: extract last-token fp32 → fp16
GPU: vkCmdCopyBuffer fp16 attn slice → B.dn_qkv V region
```

All GPU, no host data touch for the chunk-prefill output. The host still
orchestrates (records the command-buffer steps, issues submissions,
binds descriptors) but does not touch chunk-prefill output data.

### Why this is not full GPU offload

Removing the output data bridge is a qualitative step — the host no longer
reads or writes chunk-prefill output on the fast path. But the host still:

- Iterates layers and records per-layer command sequences.
- Submits command buffers and waits for fences per orchestration step.
- Zeros init state via staging (CPU zero-fill → staging upload).
- Handles decode argmax/logit computation (CPU).
- Reads back diagnostic/fallback paths.

Full GPU offload would require fusing the per-layer orchestration into
persistent workgroup dispatch and eliminating all host-visible staging for
the hot decode loop. This entry does not claim that.

## Implementation Work Completed

### New helper shader: `deltanet_chunk_last_to_fp16.comp`

A new compute shader was written at `shaders/deltanet_chunk_last_to_fp16.comp`:

- Reads the last-token fp32 core_attn_out slice from the device-local chunk
  output buffer.
- Converts each fp32 element to fp16.
- Writes the fp16 slice into a per-layer device-local output buffer
  (`dn_chunk_attn_out_[dn_idx]`).
- Workgroup decomposition: dispatches `num_heads × ceil(v_dim / 64), 1, 1`.

This is a pure data-movement + format-conversion shader, not a computation
kernel. It exists because Vulkan does not offer a built-in fp32→fp16
buffer copy operation, and the host-side `half_to_float` reverse path
required a CPU round-trip.

### Device-local chunk output buffer

`gpu_chunk_prefill_from_gpu_collect()` previously wrote final_state and
core_attn_out into `out_buf` — a host-visible buffer allocated by the caller
for each layer. On the no-compare path, this is replaced with a **device-local
chunk output buffer** (`out_buf`) that is:

- Allocated per layer invocation (device-local, not host-visible).
- Sized to hold the same per-layer final_state + core_attn_out data layout
  as the previous `out_buf`, but in device-local memory.

The shader writes to this buffer in the same layout as before. The only
difference is the memory property — no host visibility, no staging mapping.

### GPU-to-GPU state copy

After the tiled shader completes for a layer, the runtime now issues a
`vkCmdCopyBuffer` from the device-local chunk output buffer into the
persistent `bufs_->dn_state` buffer, copying only the `final_state` region:

```cpp
VkBufferCopy copy{};
copy.srcOffset = final_state_offset;    // offset in chunk output buffer
copy.dstOffset = layer_dn_state_offset; // offset in bufs_->dn_state
copy.size      = final_state_bytes;     // num_heads * k_dim * v_dim * sizeof(float)
vkCmdCopyBuffer(cmd, out_buf.buffer, bufs_->dn_state.buffer, 1, &copy);
```

No CPU involvement. The `final_state` data stays on-device.

### `deltanet_chunk_last_to_fp16.comp` dispatch

After the state copy, the runtime dispatches the new shader:

```cpp
vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, chunk_last_to_fp16_pipeline_);
vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ...);
vkCmdPushConstants(cmd, ...);
vkCmdDispatch(cmd, (DN_VAL_TOTAL + 63) / 64, 1, 1);
```

The shader reads `core_attn_out` from the device-local chunk output buffer
(already written by the preceding tiled dispatch), extracts the last-token
fp32 slice, converts to fp16, and writes to `dn_chunk_attn_out_[dn_idx]`.

### `correct_last_token_hidden()` device-local handoff

`correct_last_token_hidden()` is the function that places the per-layer
fp16 attention output into the B.dn_qkv V region for the subsequent decode
or prefill phase. Before this entry, it read from CPU-side temporaries
(`chunk_output_cache_`, `chunk_vec_cache_`).

A new handoff path was added:

1. **`gpu_chunk_handoff_ready_[dn_idx]`** — a per-layer `bool` flag (in host
   memory, set by the runtime after the device-local dispatch sequence
   completes for that layer) that signals `correct_last_token_hidden()` to
   use the device-local copy path.

2. When the flag is set, `correct_last_token_hidden()` issues a
   `vkCmdCopyBuffer` from `dn_chunk_attn_out_[dn_idx]` into the
   `B.dn_qkv` V region at the appropriate layer offset — the fp16 slice is
   already the correct format, so no CPU conversion is needed.

3. The flag is cleared after the copy, so subsequent calls (or fallback
   paths) use the regular CPU upload path.

### Buffer usage fix in `vk_device.cpp`

The device-local buffer allocation in `src/runtime/vk_device.cpp` was
updated to include `VK_BUFFER_USAGE_TRANSFER_SRC_BIT` in the usage flags
for buffers that are used as source in `vkCmdCopyBuffer` operations. This
is because:

- Existing code uses `vkCmdCopyBuffer` from device-local buffers
  (`bufs_->dn_state` is copied into during decode).
- The new handoff code uses `vkCmdCopyBuffer` from the device-local chunk
  output buffer and from `dn_chunk_attn_out_[dn_idx]`.

Without this flag, Vulkan validation would report transfer-source usage
violations on the affected buffers — the driver requires explicit
transfer-source declaration for buffers used in copy commands, even when
the copy is GPU-internal.

The affected buffer allocations were those that previously only specified
`VK_BUFFER_USAGE_STORAGE_BUFFER_BIT` (for shader access) and now also
needed transfer-source access for the device-local copy operations.

### Fallback path preserved

The existing host-visible readback/upload path is preserved for:

- **Compare diagnostics**: When `SPOCK_GPU_CHUNK_PREFILL_COMPARE=1` or
  `SPOCK_GPU_COLLECT_PREFILL_COMPARE=1` is set, the runtime still uses
  `out_buf` (host-visible) so the CPU can read back the shader output and
  compare against the expected result. The device-local handoff would
  bypass this comparison, so it is disabled when any compare flag is active.
- **Non-tiled paths**: Per-head submit uses the existing `out_buf` path.
  The device-local handoff was implemented for the tiled path only.
- **CPU-collected chunk input paths**: When GPU collection is not active
  (`SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1` is not set), the chunk
  prefill input comes from CPU staging and the output path remains the
  existing host-visible readback.

The fallback path uses the same `out_buf` and CPU conversion/upload
sequence as in diaries 0020–0024. It is unmodified.

### Gate conditions for the fast path

The no-compare device-local handoff fast path activates only when **all** of
the following are true:

- `SPOCK_GPU_CHUNK_PREFILL=1`
- `SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1`
- `SPOCK_GPU_CHUNK_PREFILL_TILED=1`
- Neither `SPOCK_GPU_CHUNK_PREFILL_COMPARE=1` nor
  `SPOCK_GPU_COLLECT_PREFILL_COMPARE=1` is set.

If any condition is false, the runtime falls back to the existing
host-visible `out_buf` path with CPU readback, conversion, and upload.

## Verification

All commands run on the target RADV RX 6750 XT (NAVI22) hardware.

### Build

```sh
source ~/.zshrc && cmake -S . -B build && cmake --build build -j
```

Passed cleanly. No compilation or linking errors. The new shader compiles to
a SPIR-V module loaded at session init alongside the existing pipelines.

### git diff --check

```sh
source ~/.zshrc && git diff --check
```

No whitespace errors.

### Standalone tiled probe (synthetic verification)

```sh
timeout 600s ./build/spock-deltanet-chunk-prefill-tiled-probe
```

```
{"status":"compare-ok","nan_count":0,
 "max_rel_core":1.19209e-07,"max_rel_state":1.19208e-07,
 "max_abs_core":1.19209e-07,"max_abs_state":2.38419e-07}
```

The standalone probe confirms the tiled shader output is unchanged (same
synthetic case as diaries 0023/0024). The probe itself was not modified —
it uses the host-visible out_buf path, which is unchanged. This is a
regression check that the build and shader compilation are sound.

### No-compare fast path parity (device-local handoff active)

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

The device-local handoff path produces the correct decoded output for
`short_correctness_001` through 16 generated tokens. No host-side data is
touched for chunk-prefill output after the shader dispatch completes.

### No-compare longer prompts

```sh
SPOCK_GPU_CHUNK_PREFILL=1 SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1 \
  SPOCK_GPU_CHUNK_PREFILL_TILED=1 \
  python3 tests/run_vk_decode_parity.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --reference tests/data/reference_tokens.jsonl \
  --ids mixed_correctness_023,pp520_046 \
  --max-new-tokens 4
```

```
{"status":"ok","checked":2,"failures":[]}
```

The device-local handoff path also passes on longer prompts
(`mixed_correctness_023` and `pp520_046`, which had been prefillsensitive
in earlier phases).

### Compare fallback path (host-visible readback preserved)

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

The compare-diagnostic fallback path (host-visible readback, CPU comparison)
still works when the compare flag is active. The device-local handoff is
disabled and the existing `out_buf` readback path is used.

### CTest regression: GPU-collect suite (3/3 passed)

```sh
ctest --test-dir build --output-on-failure -R spock_vk_decode_gpu_collect_chunk_prefill
```

| Test | Status | Time |
|------|--------|------|
| `spock_vk_decode_gpu_collect_chunk_prefill_short` | Passed | 114.88 sec |
| `spock_vk_decode_gpu_collect_chunk_prefill_tiled` | Passed | 8.95 sec |
| `spock_vk_decode_gpu_collect_chunk_prefill_short_baseline` | Passed | 5.11 sec |

Key observations:

- The tiled CTest runtime has moved between 10.67 sec (diary 0024),
  16.82 sec (first diary 0025 run), and 8.95 sec (latest rerun after
  handoff flag cleanup). These single timings are not enough to classify
  the change as a regression or improvement: they were taken under
  different ambient conditions, and the new handoff intentionally adds a
  small GPU copy + dispatch per layer to remove a CPU bridge.
- The per-head submit test (`_short`) remained in the ~100 sec range.
- The baseline (no gates) remained under 10 sec.
- The significance of this entry is not measured speedup. It is the
  **reduced CPU bridge**: the host no longer touches chunk-prefill output
  data on the fast path. This is a qualitative architectural change,
  not a performance optimization.

### CTest regression: Targeted gates (4/4 passed)

```sh
ctest --test-dir build --output-on-failure \
  -R "spock_capabilities|spock_deltanet_chunk_unit|spock_vk_decode_prefill_handoff_mismatch|spock_diagnose_handoff_mc023"
```

All 4/4 passed:
- `spock_capabilities` — 0.03 sec
- `spock_deltanet_chunk_unit` — 0.91 sec
- `spock_vk_decode_prefill_handoff_mismatch` — 39.58 sec
- `spock_diagnose_handoff_mc023` — 7.78 sec

Total: 7 CTest tests passed (3 GPU-collect suite + 4 targeted gate tests),
plus the standalone probe, plus 3 parity harness runs (no-compare fast path
at 16 tokens, no-compare longer prompts at 4 tokens, compare fallback at
1 token).

## Known Limitations

1. **Does not make full GPU offload complete.** This entry removes a
   specific CPU bridge (chunk-prefill output handoff for the no-compare
   GPU-collected+tiled path). It does not eliminate CPU orchestration or
   all host-side data touches. Remaining CPU/host pieces include:

   - **Per-layer orchestration and submission.** The host still iterates
     layers, records command-buffer sequences, binds descriptors, submits
     queue work, and waits for fences. Each DeltaNet layer requires host
     intervention to issue the tiled dispatch, the GPU copy, and the
     `deltanet_chunk_last_to_fp16.comp` dispatch.
   - **Init zero staging.** The initial (zero) state for chunk prefill is
     staged via CPU zero-fill → staging buffer upload. A device-local
     clear (vkCmdFillBuffer) would eliminate this host touch.
   - **Diagnostic/fallback paths.** The compare-diagnostic paths and
     non-tiled paths still use host-visible readback.
   - **Decode argmax/logit.** The final token selection (argmax over logits)
     happens on CPU after GPU logit computation.
   - **Broader megakernel fusion and persistent dispatch.** The per-layer
     orchestration model will remain a serial bottleneck until the entire
     decode pass is fused into a single persistent dispatch.

2. **New shader adds complexity.** `deltanet_chunk_last_to_fp16.comp` is a
   narrow-purpose shader (fp32→fp16 extraction from a specific buffer
   layout). It adds a pipeline, descriptor set, and dispatch to the
   session init and per-layer command sequence. A future fused megakernel
   would subsume this operation into the larger computation.

3. **Per-layer handoff flag is host-visible.** The
   `gpu_chunk_handoff_ready_[dn_idx]` boolean array is in host memory, set
   by the runtime after the device-side dispatch completes. This is not a
   data touch (it is a control flag), but it is still a host-mediated
   handshake. A true GPU-native design would signal readiness via device
   timelines or semaphores.

4. **Device-local buffer usage flag fix is a prerequisite.** The
   `VK_BUFFER_USAGE_TRANSFER_SRC_BIT` addition in `vk_device.cpp` is
   required for correctness, not a new feature. It affects existing buffers
   (like `dn_state`) that are used as copy sources. Without it, validation
   errors would occur when the new GPU-to-GPU copy operations run.

5. **Performance characterization deferred.** The CTest tiled runtime has
   moved between 10.67 sec, 16.82 sec, and 8.95 sec across recent runs.
   These timings were collected under different run conditions and are not
   enough to classify the change as a regression or improvement. The handoff
   itself adds a GPU copy + dispatch per layer to remove the CPU output
   bridge. Meaningful
   performance analysis requires controlled benchmarking on representative
   workloads, not the minimal CTest prompt at seq_len=9.

6. **Coverage still limited.** The new handoff path has been verified on
   `short_correctness_001` (16 tokens), `mixed_correctness_023`, and
   `pp520_046` (4 tokens each). Broader P0 coverage and 512+ token prompts
   are still pending.

## Next Work

### Near-term: Prompt coverage expansion

Run the no-compare device-local handoff path on additional P0 prompts,
including longer sequences (>512 tokens). The handoff path is fully
GPU-resident for chunk-prefill output, so its correctness on longer prompts
depends on the same shader correctness already verified — but empirical
verification across the prompt corpus is needed before considering default.

### Near-term: Init zero staging cleanup

Replace the CPU zero-fill + staging upload for chunk-prefill init state
with a `vkCmdFillBuffer` call. This is a small change that removes the
last CPU data touch for the chunk-prefill compute path (on the no-compare
GPU-collected+tiled gate).

### Medium-term: Eliminate host-visible staging for diagnostic paths

The compare-diagnostic paths still require host-visible readback of shader
output. If the diagnostic is kept as a permanent verification, it could
be run as a separate GPU-only comparison dispatch — but the current
CPU-comparison approach (download fp32 → compare against CPU reference)
is simpler and may be acceptable for a non-hot-path diagnostic.

### Medium-term: Toward the fused megakernel

- Merge the new `deltanet_chunk_last_to_fp16.comp` operation into the tiled
  shader (write fp16 output directly instead of post-processing fp32).
- Precompute `k_cumdecay` once per head across v-tiles.
- Eliminate per-layer host submission by fusing all layers into a single
  persistent dispatch.
- Fold decode argmax/logit computation onto GPU.

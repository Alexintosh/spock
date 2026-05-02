# Vulkan Runtime Strategy

The runtime exists to make the RX 6750 XT execution path measurable, reproducible, and honest about operational parity.

## Target Runtime Modes

The runtime supports three increasingly fused modes:

- `layer_by_layer`: correctness-first path with separate dispatches for kernels or layers.
- `single_submit`: a recorded command buffer executes one token with no host mediation between layers.
- `persistent_dispatch`: one persistent Vulkan dispatch owns the decode pass.

The project should not claim full megakernel parity unless `persistent_dispatch` is correct and stable on the target RADV stack.

## Device Bring-Up

Capability detection must record:

- physical device name
- Vulkan API version
- driver name and version
- subgroup size and supported subgroup operations
- max workgroup size
- shared memory / LDS limits
- storage-buffer limits
- timestamp query support and period
- fp16 shader support
- bf16 support, expected to be unavailable for this target
- cooperative matrix support, expected to be unavailable for this target

The capability dump is benchmark metadata, not debug-only output.

## Memory Model

Required allocation classes:

- device-local weights
- device-local activations
- device-local DeltaNet recurrent state
- device-local KV cache
- host-visible staging buffers
- persistent mapped upload ring
- scratch buffers sized by pipeline mode

The hot decode path must not depend on host-visible model weights.

## DeltaNet Prefill Offload Status

Current production prefill is still not fully GPU-native. The env-gated
`SPOCK_GPU_CHUNK_PREFILL=1` path moves the DeltaNet chunk-rule computation to
Vulkan, but by default runtime Q/K/V/g/beta collection remains CPU-hosted.

**GPU-collected chunk-prefill path** is now wired behind the additional env gate
`SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1` (meaningful only with
`SPOCK_GPU_CHUNK_PREFILL=1`). When both gates are active, the runtime preserves
GPU-collected Q/K/V/g/beta buffers for all DeltaNet layers in device-local
per-layer segments and feeds them directly into `deltanet_chunk_prefill.comp`
via `gpu_chunk_prefill_from_gpu_collect()`, avoiding CPU intermediate
packing/upload for the chunk-prefill inputs. Default behavior unchanged.

Two standalone proofs and one runtime diagnostic confirm the collection
shader is correct and can be driven from real session activations:

- `spock-deltanet-prefill-collect-probe` proves per-token fp16 dn_qkv plus
  fp32 g/beta can be collected into fp32 head-major Q/K/V/g/beta buffers with
  exact CPU agreement.
- `spock-deltanet-prefill-pipeline-probe` proves those collected buffers feed
  `deltanet_chunk_prefill.comp` directly and match `run_deltanet_chunk_rule`
  at heads=16, seq_len=104, total_seq=128, chunk_size=64.
- `SPOCK_GPU_COLLECT_PREFILL_COMPARE=1` dispatches
  `deltanet_prefill_collect.comp` from real `DecodeSession` per-token
  QKV/g/beta activation buffers during layer-major prefill, downloads the
  GPU-collected buffers, and compares against the existing
  CPU-collected `PrefillChunkState`. Verified exact match on
  `short_correctness_001` (all 18 DeltaNet layers, seq_len=9, max_rel=0,
  max_abs=0, nan_count=0).

**CTest regression gate** (diary 0022) protects this gated path from
accidental regression: `spock_vk_decode_gpu_collect_chunk_prefill_short`
runs `short_correctness_001 --max-new-tokens 1` with the double-gated
env vars, and `spock_vk_decode_gpu_collect_chunk_prefill_short_baseline`
runs the same prompt with no env vars as a quick diagnostic reference.

Both tests are registered in CMakeLists.txt and execute via
`tests/run_vk_decode_parity.py`.

The CPU collection bridge is now bypassed on the no-compare gated path
(diary 0021): when `SPOCK_GPU_CHUNK_PREFILL=1` and
`SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1` are active and neither
`SPOCK_GPU_COLLECT_PREFILL_COMPARE=1` nor
`SPOCK_GPU_CHUNK_PREFILL_COMPARE=1` is set, the per-token staging downloads,
half_to_float conversion, and prefill_chunks_ population are skipped entirely.
CPU collection remains for fallback and diagnostics when either compare flag
is active.

**Tiled single-dispatch chunk-prefill gate integrated** (diary 0024). The
tiled shader (`deltanet_chunk_prefill_tiled.comp`) is now wired into the
runtime behind `SPOCK_GPU_CHUNK_PREFILL_TILED=1` (only with
`SPOCK_GPU_CHUNK_PREFILL=1`). The new gate removes the per-head submit
loop: each DeltaNet layer dispatches a single
`vkCmdDispatch(num_heads, ceil(v_dim/TILE_V), 1)` instead of 16 per-head
submits. Both the CPU-collected and GPU-collected data paths support the
tiled dispatch mode. Verified on `short_correctness_001` at
`--max-new-tokens 1` with parity OK, compare diagnostics reporting `nan_count=0`, and CTest suite
3/3 passed (tiled: 10.67 sec in diary 0024; 16.82 sec in the first diary
0025 run; 8.95 sec in the latest diary 0025 rerun after handoff flag
cleanup; not directly comparable). Still env-gated, not default. A new CTest test
`spock_vk_decode_gpu_collect_chunk_prefill_tiled` protects the
GPU-collected + tiled path from regression.

**GPU-resident chunk-prefill output handoff + init clear** (diaries 0025/0026).
On the no-compare GPU-collected+tiled path (no diagnostic compare flag
active), all chunk-prefill compute data stays on-device.

- **Output handoff** (diary 0025): The tiled shader writes to a device-local
  chunk output buffer; `final_state` is copied GPU-to-GPU via `vkCmdCopyBuffer`
  into `bufs_->dn_state`; the last-token fp32 `core_attn_out` slice is extracted
  and converted to fp16 by `deltanet_chunk_last_to_fp16.comp` into per-layer
  `dn_chunk_attn_out_` buffers; and `correct_last_token_hidden()` copies that
  fp16 slice GPU-to-GPU into the `B.dn_qkv` V region. This eliminates CPU
  readback, float_to_half conversion, and upload.
- **Init clear** (diary 0026): The chunk-prefill init state buffer is now
  device-local and zeroed via `vkCmdFillBuffer` instead of host-visible +
  CPU `memset`. This removes the last CPU data touch for chunk-prefill
  compute on the fast path.

A prerequisite buffer usage fix in `src/runtime/vk_device.cpp` added
`VK_BUFFER_USAGE_TRANSFER_SRC_BIT` to device-local buffers used as copy
sources (existing and new `vkCmdCopyBuffer` targets). Without this flag,
the new GPU-to-GPU copies and existing copies from device-local buffers
would violate Vulkan usage constraints.

The fallback host-visible paths are preserved for compare diagnostics,
non-tiled paths, and CPU-collected chunk input paths — both for `out_buf`
(CPU readback) and `init_buf` (CPU memset).

This does NOT make full GPU offload complete. Remaining CPU/host pieces
include per-layer orchestration/submission, diagnostic/fallback paths,
decode argmax/logit/diagnostic readbacks, and broader megakernel
fusion/persistent dispatch.

Still env-gated, not default.

**Opt-in device-resident decode token embedding** (diary 0027). Behind
`SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1`, the per-step embedding lookup reads
token_id directly from the device-local `argmax_result` buffer (binding 0)
instead of from a CPU push constant. The CPU still downloads
`argmax_result` each decode step for external output and parity — this is
NOT full GPU offload and NOT the megakernel. It only removes the CPU token
value as the *source* for the next step's embedding; the current serial
loop still downloads before the next iteration. A new shader
`embedding_lookup_from_buffer.comp` performs the same row lookup as the
existing `embedding_lookup.comp` but reads the index from a storage buffer
rather than a push constant. The shader uses the existing
`pipeline_layout_3` and is dispatched as a single workgroup of 64
invocations. The existing push-constant path remains the default. The gate
is independent of the GPU chunk-prefill gates and composes correctly with
all of them. Still env-gated, not default.

**Opt-in deferred generated-token download** (diary 0028). Behind
`SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1 SPOCK_GPU_DEFER_TOKEN_DOWNLOAD=1`,
the per-step CPU download of `argmax_result` is replaced by a device-local
`vkCmdCopyBuffer` (4 bytes, same-queue, no host round-trip) that writes
into a pre-allocated device-local `gen_tokens` buffer at step offset
`decode_step * 4`. After the decode loop completes, all generated tokens
are downloaded in a single batch via `download_from_device(gen_tokens,
num_generated * 4)` and pushed into `result.generated_tokens` and
`tokens`. The gate is disabled when `verbose`, `debug_dump`, or
`diagnose_decode_drift` is active, because those paths need per-token
values at each step. Guards `max_new_tokens > 0` to avoid zero-sized
Vulkan buffer allocation; zero-token parity now passes. Default behavior
remains per-step download; the gate is disabled for `verbose`,
`debug_dump`, and `diagnose_decode_drift`. Does not restructure the
submit-wait loop — no performance speedup claimed. Still env-gated, not
default.

## Descriptor Model

The baseline descriptor layout exposes:

- packed weights
- activation buffers
- DeltaNet state
- KV cache
- scratch buffers
- runtime constants

Pipeline layouts may specialize by runtime mode, but benchmark output must
identify the mode.

### Per-layer stable descriptor sets

The `layer_by_layer` decode path initially mutated a shared set of 24 (later
26) covered `VkDescriptorSet` handles for each of the 28 layers, adjusting
weight offsets and per-layer buffer offsets (KV cache slot, conv1d state)
with every dispatch. This is correct but incompatible with single-submit
recording, where the command buffer must be recorded ahead of time
and cannot mutate descriptors between recording and submission.

When `SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1` (diary 0029, extended in 0032),
the constructor pre-allocates 28 x 26 = 728 `VkDescriptorSet` handles from
pool `ds_layout_3` and pre-binds each with its layer-specific weight offset,
activation buffer references, and static per-layer buffer offsets. The
decode loop then skips the per-layer mutation block and selects the
pre-bound set for the current layer via alias:

```
VkDescriptorSet ds_input_norm = per_layer_sets_enabled_
    ? per_layer_sets_->input_norm[layer]
    : D.input_norm;
vkCmdBindDescriptorSets(cmd, ..., &ds_input_norm, ...);
```

Covered descriptor sets include common MLP/norm (9 sets), attention
(10 sets), first-stage DeltaNet (5 sets), and L2-norm DeltaNet (2 sets).
RoPE descriptors (D.rope, D.rope_k) are pre-bound once at session
construction (diary 0031); the per-step position is communicated via push
constant freq_offset. Intra-DeltaNet sub-step L2-norm descriptors
(dn_l2_q, dn_l2_k) are now pre-bound (diary 0032). Remaining inner
DeltaNet dispatch-target descriptors (dn_recurrent, dn_norm_gate,
dn_out_proj, dn_compute_g_beta) are NOT covered and remain on the old
mutation path. Internal decomposition descriptors (dn_split_q,
dn_split_kv) are also listed as uncovered. These six descriptors are the
remaining descriptor-mutation blocker for full single-submit recording.

**Negative result (diary 0030):** A naive all-six pre-binding attempt
(dn_l2_q, dn_l2_k, dn_recurrent, dn_norm_gate, dn_out_proj,
dn_compute_g_beta) was reverted after decode-state corruption at step 1.
The L2-norm pair was independently safe and has been successfully
pre-bound in diary 0032. The four stateful descriptors (dn_recurrent,
dn_norm_gate, dn_out_proj, dn_compute_g_beta) remain unresolved: naive
pre-binding causes decode-state corruption, consistent with a
state-offset or descriptor-aliasing bug that does not produce a
Vulkan-level error. Root cause was not pursued; a deeper rework or
kernel fusion is required for the stateful subset.

Descriptor pool capacity was increased to accommodate the per-layer sets:
- maxSets: 192 → 1024
- storage buffer slots: 192 → 4096
- combined-image-sampler: 64 (unchanged)

The pool retains `VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT` so
sets can be destroyed per-session and the pool reused.

This is NOT single-submit. It removes the per-layer descriptor mutation
blocker for covered sets, enabling future command-buffer pre-recording.

## Synchronization Strategy

`layer_by_layer` may use command-buffer ordering and explicit barriers between operations.

`single_submit` records a full token schedule into one command buffer and submits once per token. This is the minimum operational target for "no host mediation between layers."

`persistent_dispatch` requires a proven cross-workgroup coordination strategy. If a software global barrier is used, it must have:

- a bounded progress argument
- a timeout or watchdog-safe failure mode during validation
- stress tests across repeated long decode runs
- exact-token parity against `layer_by_layer`

## Measurement Hooks

The runtime must expose timing boundaries for:

- prefill
- one-token decode
- `tg128` decode loop
- individual major blocks during tuning
- command-buffer submission overhead
- GPU timestamp regions when supported

Every benchmark must state whether reported timing is GPU-only, host end-to-end, or both.

## Go / No-Go Rule

If the runtime cannot prove stable cross-workgroup synchronization on this GPU and driver, pivot to `single_submit` and benchmark that path. A stable single-submit engine is a valid outcome; an unstable persistent dispatch is not.

## Observed Device Properties

Values recorded from the local RADV stack during decode pipeline bring-up:

| Property | Expected | Observed |
| --- | --- | --- |
| Device name | AMD Radeon RX 6750 XT | AMD Radeon RX 6750 XT (RADV NAVI22) |
| Subgroup size | 32 | 64 |
| Max shared memory | 64 KiB | 64 KiB (65536 bytes) |
| Max workgroup invocations | 1024 | 1024 |
| Vulkan API version | 1.2+ | 1.4.318 |
| fp16 shader support | yes | yes |
| bf16 shader support | no | no |
| Cooperative matrix | no | no |

Note that the subgroup size is 64, not the originally assumed 32. This affects:
- matvec workgroup sizing (currently 64, which is correct)
- rms_norm shared-memory reduction array sizing (256, independent of subgroup)
- Future persistent kernel design (barrier probe needs re-evaluation with wg=64)

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

## Descriptor Model

The baseline descriptor layout should expose:

- packed weights
- activation buffers
- DeltaNet state
- KV cache
- scratch buffers
- runtime constants

Pipeline layouts may specialize by runtime mode, but benchmark output must identify the mode.

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

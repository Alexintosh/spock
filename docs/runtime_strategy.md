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
Vulkan, but runtime Q/K/V/g/beta collection is still CPU-hosted.

Two standalone proofs and one runtime diagnostic now confirm the collection
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

The diagnostic proof clears the main layout-risk argument. The next runtime
step is to feed the GPU-collected buffers (now proven numerically identical to
the CPU-collected data on the checked runtime prompt) into the env-gated
chunk-prefill path, then remove the CPU collection bridge from that path.

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

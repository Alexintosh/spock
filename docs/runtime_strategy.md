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


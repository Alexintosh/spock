# 0113: DeltaNet Full Token-Mixer End-to-End Probe

## Goal

Prove that the complete DeltaNet token-mixer pipeline, assembled from
individually gated compute shaders, produces exact fp16-bit-identical
mixer output and residual values for layer 0 of Qwen 3.5 0.8B when given
the same input activations, weights, and recurrent state as the captured
runtime reference.

This closes the final gate in the DeltaNet backward-validation ladder:
instead of testing each stage in isolation, the entire 11-stage mixer
pipeline runs in a single Vulkan submit and the two final outputs --
mixer_output and mixer_residual -- are compared against the captured
reference tensors.

```
input_norm (1024 fp16) + input_hidden (1024 fp16)
  + weights (qkv_proj, z_proj, a_proj, b_proj, conv, g_beta, norm, out_proj)
  + conv_state (24576 fp16) + recurrent_state (262176 f32)
  --> [11-stage Vulkan compute pipeline]
  --> mixer_output (1024 fp16) + mixer_residual (1024 fp16)
```

## Why this was needed

Diaries 0099 through 0112 proved each individual stage of the DeltaNet
mixer in isolation. Each probe loaded its specific fixture inputs and
verified its specific output against captured reference tensors. Every
single-stage gate passed with zero mismatches.

However, an end-to-end probe serves a different purpose than a collection
of unit gates:

1. **Binding layout integration.** Individual probes use simple buffer
   layouts where each binding points to a dedicated buffer. The mixer
   pipeline reuses a single `qkv_buf` across multiple stages with
   sub-allocated sections (Q, K, V). The recurrent shader's output
   placement and the subsequent norm-gate and out-proj reads depend on
   the exact byte offsets within this shared buffer. A layout bug in any
   offset computation would pass individual probes but fail in the
   integrated pipeline.

2. **Inter-stage data flow.** The probes verify each stage independently
   with pre-captured intermediate fixtures. The mixer probe verifies that
   the actual GPU-produced intermediate values chain correctly through
   all eleven stages, including potential precision effects from
   in-place operations like L2 normalization.

3. **Command buffer correctness.** The full pipeline records eleven
   dispatch calls with memory barriers between dependent stages. Missing
   or misordered barriers can produce non-deterministic failures that
   individual probes never see.

4. **Weight loading integration.** Individual probes may use simplified
   weight loading. The mixer probe loads all eight weight matrices from
   the repacked artifact using the production weight loader, verifying
   that role-based weight extraction works for the complete mixer.

## The layout bug this probe found

During initial bringup, the mixer probe failed with large mismatches
in both output and residual. Investigation revealed a buffer offset
calculation error in how the V section of `qkv_buf` was referenced
after the recurrent shader stage.

### qkv_buf memory layout

The `qkv_buf` buffer holds the combined QKV projection output:

```
offset 0:                   Q section [key_total = 2048 fp16 values]
offset key_total * 2 bytes: K section [key_total = 2048 fp16 values]
offset 2 * key_total * 2:   V section [val_total = 2048 fp16 values]
```

With key_total = 2048 and sizeof(fp16) = 2:
- Q section: bytes [0, 4096)
- K section: bytes [4096, 8192)
- V section: bytes [8192, 12288)

### Recurrent shader binding

The recurrent shader (`deltanet_recurrent.comp`) has binding 1 pointed
at the K section start (byte offset 4096) with a size covering both
K and V sections (8192 bytes). Inside the shader's local view:

- K is at local offset 0
- V/output is at local offset key_total * 2 = 4096 bytes

Globally, the recurrent output lands at K section start + key_total * 2 =
4096 + 4096 = 8192 bytes. This is the V section start.

### The bug

The code originally set:

```cpp
VkDeviceSize v_offset_bytes = static_cast<VkDeviceSize>(key_total) * 2;
// = 2048 * 2 = 4096 bytes  --> points at K section!
```

This pointed at the K section, not the V section. Both the norm-gate
shader (which reads the recurrent core output) and the output projection
(which reads the gated result) were reading from the wrong memory region.

### The fix

```cpp
VkDeviceSize v_section_offset_bytes = static_cast<VkDeviceSize>(2) * key_total * 2;
// = 2 * 2048 * 2 = 8192 bytes  --> correctly points at V section
```

After this fix, both the norm-gate and out-proj stages correctly read
from the V section of qkv_buf, and the probe passes with zero mismatches.

## Implementation

### Probe app

`apps/vk_deltanet_mixer_probe.cpp` -- single-submit Vulkan compute
pipeline that chains all eleven mixer stages:

| Stage | Shader | Input | Output |
|-------|--------|-------|--------|
| 1 | matvec | input_norm, weight_qkv | qkv_buf (Q section) |
| 2 | matvec | input_norm, weight_z | z_buf |
| 3 | matvec | input_norm, weight_a | a_buf |
| 4 | matvec | input_norm, weight_b | b_buf |
| 5 | conv1d_step | qkv_buf, conv_state, weight_conv | qkv_buf (in-place) |
| 6a | l2_norm_per_head | qkv_buf Q section | qkv_buf Q section (in-place) |
| 6b | l2_norm_per_head | qkv_buf K section | qkv_buf K section (in-place) |
| 7 | deltanet_compute_g_beta | a_buf, b_buf, weight_ab | state_buf tail |
| 8 | deltanet_recurrent | qkv_buf Q, KV+out view, state_buf | qkv_buf V section |
| 9 | deltanet_norm_gate | qkv_buf V section, z_buf, weight_norm | qkv_buf V section (in-place) |
| 10 | matvec | qkv_buf V section, weight_out_proj | mixer_output_buf |
| 11 | residual_add | input_hidden, mixer_output | mixer_residual_buf |

### Buffer layout

Key layout decisions:

- **qkv_buf**: single buffer of 6144 fp16 values. Q/K/V sections are
  sub-allocated via descriptor set offset+range. The L2 norm stages
  operate in-place on their respective sections. The recurrent stage
  writes output to the V section.

- **Descriptor sub-allocation**: Each stage's descriptor set binds to
  the appropriate offset and range within shared buffers rather than
  using dedicated per-stage buffers. This mirrors the production
  decode path's memory-efficient layout.

- **Staging**: A host-visible staging buffer holds only the two final
  outputs (mixer_output + mixer_residual) for download and comparison.
  Intermediate results remain device-local.

### Command buffer synchronization

Memory barriers (`vkCmdPipelineBarrier` with `VkMemoryBarrier`) are
inserted between dependent stages. The barrier pattern follows
compute-write to compute-read ordering:

- After stage 1: qkv written before conv reads it
- After stages 3+4: a/b written before g_beta reads them
- After stage 5: conv output before L2 norms read it
- After stage 6a: Q norm before K norm (sequential in-place on same buffer)
- After stage 7: g_beta state tail written before recurrent reads it
- After stage 8: recurrent output before norm_gate reads it
- After stage 9: gated output before out_proj reads it
- After stage 10: mixer_output before residual_add reads it
- After stage 11: transfer barrier for staging copy

### Fixture files

Seven input fixtures plus two expected output fixtures:

Input fixtures:
- `layer0_step1_dn_input_norm_1024.fp16` (2048 bytes)
- `layer0_step1_input_hidden_1024.fp16` (2048 bytes)
- `layer0_step1_dn_conv_state_pre_24576.fp16` (49152 bytes)
- `layer0_step1_dn_recurrent_state_pre_262176.f32` (1048704 bytes)

Expected output fixtures:
- `layer0_step1_mixer_output_1024.fp16` (2048 bytes)
- `layer0_step1_mixer_residual_1024.fp16` (2048 bytes)

Weight matrices loaded from the repacked model artifact at
`artifacts/spock-text-repack-qwen35-0p8b`.

### CMake integration

Two CTest targets:
- `spock_deltanet_mixer_probe_help` -- smoke test (--help exits 0)
- `spock_deltanet_mixer_probe_layer0_exact` -- full layer-0 exact match

## Verification

Build and run:

```
$ cmake --build build -j
$ build/vk_deltanet_mixer_probe \
    --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
    --input-norm-fp16-file tests/data/layer0_step1_dn_input_norm_1024.fp16 \
    --input-hidden-fp16-file tests/data/layer0_step1_input_hidden_1024.fp16 \
    --conv-state-pre-fp16-file tests/data/layer0_step1_dn_conv_state_pre_24576.fp16 \
    --state-pre-f32-file tests/data/layer0_step1_dn_recurrent_state_pre_262176.f32 \
    --expected-mixer-output-fp16-file tests/data/layer0_step1_mixer_output_1024.fp16 \
    --expected-mixer-residual-fp16-file tests/data/layer0_step1_mixer_residual_1024.fp16
```

Output:

```json
{
  "status": "ok",
  "hidden": 1024,
  "num_heads": 16,
  "head_dim": 128,
  "mixer_output_mismatches": 0,
  "mixer_output_max_fp16_ulp_diff": 0,
  "mixer_residual_mismatches": 0,
  "mixer_residual_max_fp16_ulp_diff": 0
}
```

Full ctest suite (19 tests including all DeltaNet gates):

```
100% tests passed, 0 tests failed out of 19
```

Individual gate results:
- matvec_probe (5 variants): all passed
- conv_l2_probe: passed
- g_beta_probe: passed
- recurrent_probe: passed
- norm_gate_probe: passed
- residual_add_probe: passed
- mixer_probe: passed
- diary_check: passed

## What this proves

The complete DeltaNet token-mixer pipeline for layer 0 of Qwen 3.5 0.8B
produces exact fp16-bit-identical output when:

1. All eight weight matrices are loaded from the repacked artifact via
   the production weight loader.
2. All eleven compute stages execute in a single Vulkan submit with
   correct memory barriers.
3. The shared qkv_buf layout with sub-allocated Q/K/V sections works
   correctly across all stages that read and write it.
4. The recurrent shader's output placement in the V section is correctly
   consumed by the norm-gate and output projection stages.

Combined with the individual stage probes (diaries 0099-0112), the
DeltaNet backward-validation ladder is fully closed for layer 0 at both
the unit-gate and end-to-end levels.

## What this does not prove

- Not multi-layer correctness (only layer 0 is tested).
- Not the fused shader variants (deltanet_recurrent_gbeta,
  deltanet_recurrent_gbeta_norm_gate).
- Not multi-token decode sequencing.
- Not production megakernel scheduling or persistent dispatch.
- Not weight quantization or compression effects.
- Not other GPU architectures or Vulkan drivers.

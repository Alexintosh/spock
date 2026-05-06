# 0109: DeltaNet Conv1d + Q/K L2 Probe

## Goal

Prove the layer-0 DeltaNet conv1d mutation and q/k L2 normalization produce
exact fp16-bit-identical output to the captured runtime handoff tensors. This
closes gate 6 from the DeltaNet backward-validation ladder:

```
raw qkv (6144) --> [conv1d_step] --> [L2-norm Q] --> [L2-norm K] --> [V pass-through]
```

## Implementation

New probe app: `apps/vk_deltanet_conv_l2_probe.cpp`.

The probe loads five fixtures and one weight tensor:

- `layer0_step1_dn_qkv_raw_6144.fp16` — raw projected QKV before conv
- `layer0_step1_dn_conv_state_pre_24576.fp16` — pre-mutation conv rolling state
- `layer0_step1_dn_q_2048.fp16`, `layer0_step1_dn_k_2048.fp16`,
  `layer0_step1_dn_v_2048.fp16` — expected post-conv + L2-normalized q/k/v
- `layer.0.delta_conv` weights from the repacked artifact (24576 fp16 values,
  shape [6144,1,4])

The probe runs the unfused runtime shader sequence in one command buffer:

1. `conv1d_step.comp` — dispatches 1 workgroup with `conv_dim=6144, kernel_size=4`.
   Reads QKV buffer + conv state, applies depthwise conv1d + SiLU, writes QKV in-place.
2. `l2_norm_per_head.comp` on Q slice — dispatches `num_heads=16` workgroups with
   `head_dim=128`. All three bindings (input, output, dummy) point to the Q region
   of the QKV buffer at offset 0, range `key_total * 2` bytes, achieving in-place L2 norm.
3. `l2_norm_per_head.comp` on K slice — same push constants, but bindings at offset
   `key_total * 2` bytes into the QKV buffer.
4. V slice is not L2-normalized; post-conv V is the expected output.

The full QKV buffer is downloaded after the command buffer completes. Q, K, and V
slices are compared as raw fp16 bits against the expected fixtures. No tolerance.

CLI required args: `--repack-dir`, `--raw-qkv-fp16-file`, `--conv-state-pre-fp16-file`,
`--expected-q-fp16-file`, `--expected-k-fp16-file`, `--expected-v-fp16-file`.
Optional numeric args: `--conv-dim` (6144), `--kernel-size` (4), `--num-heads` (16),
`--head-dim` (128).

## Shader dispatch pattern

Both `conv1d_step.comp` and `l2_norm_per_head.comp` use the same 3-binding
descriptor layout (`pipeline_layout_3` in the runtime):

- conv1d: binding 0 = QKV (in/out), 1 = conv_state (in/out), 2 = weight (readonly)
- l2_norm: binding 0 = input (readonly), 1 = output (writeonly), 2 = dummy (writeonly)

The runtime uses this same shared layout for all three dispatches. The probe
reproduces this pattern with separate descriptor sets per dispatch but the same
descriptor set layout and two pipeline layouts (one per shader, since push constant
structs differ).

JSON output includes `status`, `q_mismatches`, `k_mismatches`, `v_mismatches`, and
first-mismatch diagnostics (index, gpu bits, expected bits) per slice when present.

## CTest gates

- `spock_deltanet_conv_l2_probe_help` — verifies --help exits cleanly
- `spock_deltanet_conv_l2_probe_layer0_exact` — runs the full conv+L2 sequence with
  layer-0 fixtures and asserts exact fp16 match

## Verification

The checked-in `layer0_step1_dn_conv_state_pre_24576.fp16` fixture was
regenerated from a fresh rebuilt dump after adding a shader-write →
transfer-read buffer barrier before `vkCmdCopyBuffer` for the pre-conv
state capture.
- `cmake --build build -j` compiles cleanly
- `ctest --test-dir build -R spock_deltanet_conv_l2_probe --output-on-failure` passes
- `python3 tests/run_diary_check.py` passes
- `git diff --check` clean

## What this proves

If the exact gate passes, the conv1d depthwise convolution + SiLU and q/k L2
normalization produce bit-identical output to the unfused runtime path for layer 0.
This eliminates conv mutation and L2 normalization as possible causes for any
downstream recurrent-core mismatch.

## What this does not prove

- Not recurrent core parity.
- Not other-layer conv/L2 correctness.
- Not the fused `deltanet_conv_l2_qk.comp` shader.
- Not inference or megakernel completion.
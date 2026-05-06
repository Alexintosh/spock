# 0117: Persistent Layer-0 Conv/L2 Gate

## Goal

Validate that `vk_persistent_layer0_probe` can advance the next narrow DeltaNet mixer boundary inside the 128-lane persistent layer shader: `dn_qkv_raw_fp16 + conv_state_pre + layer.0.delta_conv -> conv-mutated qkv -> L2(q) + L2(k) + copy(v)`.

This is not full mixer computation, not full layer composition, not inference, and not the megakernel. It validates only the conv/L2 handoff that follows the projection-prefix gate from diary 0116.

## Why Conv/L2 Next

Conv/L2 is the first stateful consumer of the projection-prefix qkv output. It advances the rolling convolution state, mutates qkv in place, and defines the q/k/v vectors that g/beta and recurrent stages consume. Gating this boundary in the persistent execution shape removes a large class of “later stage” ambiguity before recurrent math is added.

The conventional non-persistent `vk_deltanet_conv_l2_probe` already proved this boundary exactly. The remaining question was whether the persistent 128-lane layout, row-strided channel coverage, and one software global barrier could reproduce the same result.

## Implementation

### Shader: `shaders/persistent_layer0_probe.comp`

- Added `mode=2` alongside the existing `mode=0` tail path and `mode=1` projection-prefix path.
- Reused the existing 10-binding descriptor set:
  - 0 control
  - 1 qkv scratch/input
  - 2 conv_state
  - 3 delta_conv
  - 4 output_q
  - 5 output_k
  - 6 output_v
  - 7-9 dummy
- Stage A runs row-strided depthwise conv1d over all 6144 qkv channels:
  - shift `conv_state[ch, :]` left
  - append the raw qkv value
  - apply the 4-tap depthwise conv
  - apply SiLU
  - overwrite qkv scratch in place
- One `global_barrier()` separates Stage A from Stage B. `expected_generation == 1`.
- Stage B uses one 128-lane workgroup per head:
  - groups 0..15 normalize q heads from qkv `[0..2047]` into `output_q`
  - groups 16..31 normalize k heads from qkv `[2048..4095]` into `output_k`
  - formula: `inversesqrt(sum_sq + 1e-6)`
- V is copied from qkv `[4096..6143]` into `output_v` with row-strided work assignment.

### Host App: `apps/vk_persistent_layer0_probe.cpp`

- Added `--mode conv-l2`.
- Added required fixture inputs:
  - `--raw-qkv-fp16-file`
  - `--conv-state-pre-fp16-file`
  - `--expected-q-fp16-file`
  - `--expected-k-fp16-file`
  - `--expected-v-fp16-file`
- Added optional `--conv-l2-fp16-ulp-tolerance N` (default 0, exact).
- Added `layer.0.delta_conv` loading with the same flat fp16 layout and validation rules used by `apps/vk_deltanet_conv_l2_probe.cpp`.
- Added JSON reporting for the new probe name, mode, structural barrier counters, per-output exact mismatch counts, max fp16 ULP, and tolerance.

### CMake

- Added exact CTest gate `spock_persistent_layer0_probe_conv_l2_exact`.

Exact comparison passes on this RX 6750 XT validation path, so no bounded fallback gate was needed.

## Verification

### Direct Command

```sh
./build/vk_persistent_layer0_probe --mode conv-l2 \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --raw-qkv-fp16-file tests/data/layer0_step1_dn_qkv_raw_6144.fp16 \
  --conv-state-pre-fp16-file tests/data/layer0_step1_dn_conv_state_pre_24576.fp16 \
  --expected-q-fp16-file tests/data/layer0_step1_dn_q_2048.fp16 \
  --expected-k-fp16-file tests/data/layer0_step1_dn_k_2048.fp16 \
  --expected-v-fp16-file tests/data/layer0_step1_dn_v_2048.fp16
```

Result:

- `status = ok`
- `failures = 0`
- `arrived = 0`
- `generation = 1`
- `q_exact_mismatches = 0`, `q_max_fp16_ulp = 0`
- `k_exact_mismatches = 0`, `k_max_fp16_ulp = 0`
- `v_exact_mismatches = 0`, `v_max_fp16_ulp = 0`

### CTest

The targeted regression set passes with the new exact gate included:

- `spock_persistent_layer0_probe_projection_prefix_exact_fails`
- `spock_persistent_layer0_probe_projection_prefix_ulp1`
- `spock_persistent_layer0_probe_conv_l2_exact`
- `spock_persistent_layer0_probe_post_mlp_exact_fails`
- `spock_persistent_layer0_probe_post_mlp_bounded`
- `spock_deltanet_conv_l2_probe_layer0_exact`
- `spock_diary_check`

## Known Limitations

- This stops at q/k/v after conv/L2. It does not add g/beta, recurrent, norm-gate, output projection, residual add, or full `post_mlp`.
- Single layer (0) only. Not multi-layer, not representative-layer sweep.
- It proves the persistent conv/L2 handoff exactly for the current captured fixture boundary, not the whole decode loop.

## Next Work

1. Add persistent g/beta computation after q/k/v.
2. Add persistent recurrent state update and core output.
3. Add persistent norm-gate, output projection, and residual add.
4. Compose the existing persistent post-mixer tail after the full mixer path.
5. Compare full layer-0 persistent output against captured `post_mlp`.

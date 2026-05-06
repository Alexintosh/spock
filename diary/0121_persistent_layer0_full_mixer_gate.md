# 0121: Persistent Layer-0 Full-Mixer Gate

## Goal

Validate that `vk_persistent_layer0_probe` can run the complete composed DeltaNet mixer inside `persistent_layer0_probe.comp` as a single persistent dispatch:

```text
input_norm_fp16[1024] + input_hidden_fp16[1024]
  + all projection weights (qkv, z, a, b)
  + conv_state[24576] + delta_conv[24576]
  + a_log/dt_bias/g_beta control payload
  + recurrent_state[262144 fp32 bits] + delta_norm[128 fp32 bits]
  + delta_out_proj[1024 x 2048]
  -> [single 128-lane 82-workgroup persistent dispatch, 6 global barriers]
  -> mixer_output_fp16[1024] + mixer_residual_fp16[1024]
```

This composes all five previously gated persistent sub-blocks (projection-prefix, conv/L2, g-beta, recurrent, mixer-tail) into one `mode=6` dispatch. It is the persistent counterpart to the standalone 11-dispatch mixer pipeline from diary 0113.

## Why This Gate Matters

Diaries 0116-0120 validated each persistent sub-block independently against captured intermediate fixtures. The full-mixer gate proves they compose correctly when chained through 6 global barriers in a single dispatch with shared scratch buffers and in-place mutations. This is the first time the complete DeltaNet mixer runs as one persistent kernel on the RX 6750 XT.

The key difference from the standalone mixer (diary 0113) is that the standalone path uses 11 separate dispatches with Vulkan memory barriers between stages. The persistent path uses 6 software global barriers inside one dispatch. The reduction-order differences compound across the chain, producing bounded fp16 ULP deviations from the multi-dispatch reference.

## Implementation

### Shader: `shaders/persistent_layer0_probe.comp`

- Added `mode=6` (full-mixer) alongside the existing modes 0-5.
- The mode=6 path chains 7 stages with 6 global barriers (expected_generation = 6):

| Stage | Sub-block | Barrier | Generation |
|-------|-----------|---------|------------|
| 1 | Projections: QKV/Z/A/B row-strided matvecs | none | 0 |
| 2 | g-beta (16 wg) + conv phase A (all wg strided) | gen 0->1 | 1 |
| 3 | L2 normalize q/k heads (32 wg) | gen 1->2 | 2 |
| 4 | Recurrent core (16 wg), writes dn_core into buf2 | gen 2->3 | 3 |
| 5 | Norm-gate (16 wg): per-head RMSNorm + SiLU gating | gen 3->4 | 4 |
| 6 | Output projection: row-strided 128-lane matvec | gen 4->5 | 5 |
| 7 | Residual add: input_hidden + mixer_output | gen 5->6 | 6 |

- Binding layout (10 bindings):

| Binding | Contents |
|---------|----------|
| 0 | Control: arrived/gen/failures/checksum + a_log[16] + dt_bias[16] + g_beta_bits[32] + extra[262144 state + 128 delta_norm] |
| 1 | input_norm fp16[1024] |
| 2 | qkv scratch fp16[6144] (projection -> conv -> L2 -> recurrent -> norm-gate) |
| 3 | z scratch fp16[2048] (projection -> norm-gate) |
| 4 | ab scratch fp16[32] (a[16] then b[16]) |
| 5 | conv_state fp16[24576] |
| 6 | delta_conv fp16[24576] |
| 7 | packed projection weights: qkv[6144x1024] + z[2048x1024] + a[16x1024] + b[16x1024] |
| 8 | delta_out_proj fp16[1024 x 2048] |
| 9 | hidden pack fp16[3072]: input_hidden[1024] + mixer_output[1024] + mixer_residual[1024] |

### Host App: `apps/vk_persistent_layer0_probe.cpp`

- Added `run_full_mixer_mode()` function.
- Added `--mode full-mixer` CLI parsing (mode=6).
- Added `--full-mixer-fp16-ulp-tolerance N` CLI argument.
- Added full-mixer help section to `--help` output.
- Added `if (mode == 6)` dispatch in `main()`.
- The host loads all 8 weight matrices from the repacked artifact, packs them into a single weight buffer, constructs the control payload with recurrent state and delta_norm, and dispatches a single compute command.
- Outputs: `mixer_output` and `mixer_residual` read from the hidden pack buffer (binding 9).

### CMake

Added paired gates:

- `spock_persistent_layer0_probe_full_mixer_exact_fails` (WILL_FAIL)
- `spock_persistent_layer0_probe_full_mixer_ulp16`

## Verification

### Direct Exact Command

```sh
./build/vk_persistent_layer0_probe --mode full-mixer \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --input-norm-fp16-file tests/data/layer0_step1_dn_input_norm_1024.fp16 \
  --input-hidden-fp16-file tests/data/layer0_step1_input_hidden_1024.fp16 \
  --conv-state-pre-fp16-file tests/data/layer0_step1_dn_conv_state_pre_24576.fp16 \
  --state-pre-f32-file tests/data/layer0_step1_dn_recurrent_state_pre_262176.f32 \
  --expected-mixer-output-fp16-file tests/data/layer0_step1_mixer_output_1024.fp16 \
  --expected-mixer-residual-fp16-file tests/data/layer0_step1_mixer_residual_1024.fp16
```

Result:

- `status = fail`
- `failures = 0`
- `arrived = 0`
- `generation = 6`
- `expected_generation = 6`
- `mixer_output_exact_mismatches = 28`
- `mixer_output_max_fp16_ulp = 6`
- `mixer_residual_exact_mismatches = 8`
- `mixer_residual_max_fp16_ulp = 16`

### Direct Bounded Command

```sh
./build/vk_persistent_layer0_probe --mode full-mixer \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --input-norm-fp16-file tests/data/layer0_step1_dn_input_norm_1024.fp16 \
  --input-hidden-fp16-file tests/data/layer0_step1_input_hidden_1024.fp16 \
  --conv-state-pre-fp16-file tests/data/layer0_step1_dn_conv_state_pre_24576.fp16 \
  --state-pre-f32-file tests/data/layer0_step1_dn_recurrent_state_pre_262176.f32 \
  --expected-mixer-output-fp16-file tests/data/layer0_step1_mixer_output_1024.fp16 \
  --expected-mixer-residual-fp16-file tests/data/layer0_step1_mixer_residual_1024.fp16 \
  --full-mixer-fp16-ulp-tolerance 16
```

Result:

- `status = ok`
- `mixer_output_exact_mismatches = 28`
- `mixer_output_max_fp16_ulp = 6`
- `mixer_residual_exact_mismatches = 8`
- `mixer_residual_max_fp16_ulp = 16`

The ULP deviation is expected. The reference fixtures were captured from the standalone multi-dispatch mixer (`vk_deltanet_mixer_probe`) which produces exact results. The persistent single-dispatch path accumulates reduction-order divergence across the 6-barrier chain: the projection-prefix stage produces 1 ULP on ~9 rows, and these propagate through conv, L2, recurrent, norm-gate, and output projection stages. The residual add then accumulates the output projection's ULP deviation with the original input_hidden.

### Regression Coverage

All 15 targeted tests pass:

- `spock_deltanet_mixer_probe_help`
- `spock_deltanet_mixer_probe_layer0_exact`
- `spock_persistent_layer0_probe_help`
- `spock_persistent_layer0_probe_post_mlp_exact_fails`
- `spock_persistent_layer0_probe_post_mlp_bounded`
- `spock_persistent_layer0_probe_projection_prefix_exact_fails`
- `spock_persistent_layer0_probe_projection_prefix_ulp1`
- `spock_persistent_layer0_probe_conv_l2_exact`
- `spock_persistent_layer0_probe_g_beta_exact`
- `spock_persistent_layer0_probe_recurrent_exact`
- `spock_persistent_layer0_probe_mixer_tail_exact_fails`
- `spock_persistent_layer0_probe_mixer_tail_ulp1`
- `spock_persistent_layer0_probe_full_mixer_exact_fails`
- `spock_persistent_layer0_probe_full_mixer_ulp16`
- `spock_diary_check`

## Known Limitations

- Single layer (0) and one captured decode step only. Not a representative-layer sweep.
- The persistent full-mixer produces bounded fp16 ULP deviations (max 6 ULP on mixer_output, max 16 ULP on mixer_residual) compared to the multi-dispatch reference. This is a reduction-order boundary, not a correctness bug.
- The conv stage mutates conv_state as a side effect; multi-step persistence would need state preservation across dispatches.
- Not full layer persistence (mixer + post-mixer tail in one pass), not inference, and not the megakernel.

## Next Work

1. Compose the persistent full-mixer output into the existing persistent post-mixer tail (`mode=0`) as a single layer-0 pass.
2. Compare full layer-0 persistent output (mixer + tail) against captured `post_mlp`.
3. Widen from layer 0 to representative DeltaNet layers.

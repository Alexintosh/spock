# 0123: Persistent Layer-0 Full-Layer Tap Localization

## Goal

Localize the `vk_persistent_layer0_probe --mode layer0` post-MLP drift from
diary 0122 without weakening the existing full-layer gate.

Diary 0122 proved that the single-dispatch layer-0 path is structurally
correct and bounded:

```text
dn_input_norm -> persistent DeltaNet mixer -> mixer_residual
  -> post_norm -> MLP -> post_mlp
```

The remaining question was where the final `post_mlp` spread of 105 fp16 ULP
starts inside the post-mixer RMSNorm+MLP tail when the tail consumes the
persistent mixer's bounded `mixer_residual` instead of the captured runtime
`mixer_residual`.

## Implementation

### Host App: `apps/vk_persistent_layer0_probe.cpp`

Extended mode 7 with optional internal tap fixture comparisons:

- `--expected-norm-output-fp16-file PATH`
- `--expected-up-scratch-fp16-file PATH`
- `--expected-mlp-product-fp16-file PATH`
- `--tap-norm-fp16-ulp-tolerance N`
- `--tap-up-fp16-ulp-tolerance N`
- `--tap-product-fp16-ulp-tolerance N`

The implementation deliberately exposes only taps that are still valid after
the single dispatch completes:

| Tap | Shader storage after mode 7 | Fixture |
|-----|-----------------------------|---------|
| post_norm output | `buf3[0..1023]` | `layer0_step1_mlp_normed_1024.fp16` |
| MLP up projection | `buf7[0..3583]` | `layer0_step1_mlp_up_3584.fp16` |
| SiLU(gate) * up product | `buf2[0..3583]` | `layer0_step1_mlp_product_3584.fp16` |

Two requested boundaries are intentionally reported as blocked:

- Gate scratch is overwritten by the Stage 10 activation product.
- Standalone down output is not stored separately because Stage 11 fuses down
  projection with residual add into `post_mlp`.

The JSON output now reports:

- normal mode-7 mixer and final `post_mlp` metrics;
- tap metrics only when the corresponding expected fixture is supplied;
- explicit blocked status strings for gate and standalone down-output taps.

### CMake

Added paired tap gates:

- `spock_persistent_layer0_probe_layer0_taps_exact_fails` (`WILL_FAIL`)
- `spock_persistent_layer0_probe_layer0_taps_bounded`

The bounded tap gate is additive. The existing
`spock_persistent_layer0_probe_layer0_bounded` gate remains unchanged and does
not require tap fixtures.

## Verification

### Build

```sh
cmake --build build --target vk_persistent_layer0_probe -j2
```

Result: target built successfully.

### Direct Exact Tap Command

```sh
./build/vk_persistent_layer0_probe --mode layer0 \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --input-norm-fp16-file tests/data/layer0_step1_dn_input_norm_1024.fp16 \
  --input-hidden-fp16-file tests/data/layer0_step1_input_hidden_1024.fp16 \
  --conv-state-pre-fp16-file tests/data/layer0_step1_dn_conv_state_pre_24576.fp16 \
  --state-pre-f32-file tests/data/layer0_step1_dn_recurrent_state_pre_262176.f32 \
  --expected-mixer-output-fp16-file tests/data/layer0_step1_mixer_output_1024.fp16 \
  --expected-mixer-residual-fp16-file tests/data/layer0_step1_mixer_residual_1024.fp16 \
  --expected-output-fp16-file tests/data/layer0_step1_post_mlp_1024.fp16 \
  --full-mixer-fp16-ulp-tolerance 16 \
  --output-fp16-ulp-tolerance 105 \
  --expected-norm-output-fp16-file tests/data/layer0_step1_mlp_normed_1024.fp16 \
  --expected-up-scratch-fp16-file tests/data/layer0_step1_mlp_up_3584.fp16 \
  --expected-mlp-product-fp16-file tests/data/layer0_step1_mlp_product_3584.fp16
```

Result: `status = fail`, as intended for the exact tap gate.

Tap metrics:

- `tap_norm_exact_mismatches = 10`
- `tap_norm_max_fp16_ulp = 29`
- `tap_up_exact_mismatches = 668`
- `tap_up_max_fp16_ulp = 253`
- `tap_product_exact_mismatches = 1004`
- `tap_product_max_fp16_ulp = 62`

The inherited layer metrics remained the diary 0122 values under their bounded
tolerances:

- `mixer_output_max_fp16_ulp = 6`
- `mixer_residual_max_fp16_ulp = 16`
- `output_max_fp16_ulp = 105`

### Direct Bounded Tap Command

```sh
./build/vk_persistent_layer0_probe --mode layer0 \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --input-norm-fp16-file tests/data/layer0_step1_dn_input_norm_1024.fp16 \
  --input-hidden-fp16-file tests/data/layer0_step1_input_hidden_1024.fp16 \
  --conv-state-pre-fp16-file tests/data/layer0_step1_dn_conv_state_pre_24576.fp16 \
  --state-pre-f32-file tests/data/layer0_step1_dn_recurrent_state_pre_262176.f32 \
  --expected-mixer-output-fp16-file tests/data/layer0_step1_mixer_output_1024.fp16 \
  --expected-mixer-residual-fp16-file tests/data/layer0_step1_mixer_residual_1024.fp16 \
  --expected-output-fp16-file tests/data/layer0_step1_post_mlp_1024.fp16 \
  --full-mixer-fp16-ulp-tolerance 16 \
  --output-fp16-ulp-tolerance 105 \
  --expected-norm-output-fp16-file tests/data/layer0_step1_mlp_normed_1024.fp16 \
  --tap-norm-fp16-ulp-tolerance 29 \
  --expected-up-scratch-fp16-file tests/data/layer0_step1_mlp_up_3584.fp16 \
  --tap-up-fp16-ulp-tolerance 253 \
  --expected-mlp-product-fp16-file tests/data/layer0_step1_mlp_product_3584.fp16 \
  --tap-product-fp16-ulp-tolerance 62
```

Result: `status = ok`.

## Interpretation

The final 105 ULP `post_mlp` bound is not caused only by the final residual
add. The drift is already present immediately after the post-mixer RMSNorm
(`post_norm` max 29 ULP), grows substantially at the MLP up projection
boundary (max 253 ULP), and remains visible at the activation-product boundary
(max 62 ULP).

This differs from the earlier captured-MLP boundary stack in diaries 0094-0097,
where captured `mixer_residual` input gave:

- RMSNorm exact;
- gate projection max 1 ULP;
- up projection max 2 ULP;
- product max 2 ULP;
- down output max 2 ULP;
- post-residual max 87 ULP.

The new result says the persistent layer0 path has a different upstream
numerical condition: the persistent DeltaNet mixer feeds a bounded but not exact
`mixer_residual` into a reduction-sensitive RMSNorm and then into large MLP
matvecs. The next quality step is therefore a precision experiment around the
post-mixer RMSNorm and MLP projection accumulation, especially the up projection
tap. It is not appropriate to widen to representative layers until this
precision decision is documented.

## Current Status

The mode-7 drift is localized enough to choose the next experiment:

1. Preserve the current structural layer0 full-layer gate.
2. Add a precision variant or diagnostic for post-mixer RMSNorm/MLP projection
   accumulation.
3. Decide whether the persistent-mixer-to-post-MLP tap bounds are acceptable
   for decode parity, or whether the shader must change before widening beyond
   layer 0.

This remains a captured one-step layer-0 persistent pass. It is still not
multi-layer decode, not all 24 layers, not the token loop, and not archived
basic inference.

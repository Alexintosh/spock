# 0125: Persistent Layer-0 Mixer Residual Derived Diagnostic

## Goal

Decide whether the persistent layer-0 `mixer_residual` drift is caused by the
residual-add step itself or inherited from the persistent DeltaNet
`mixer_output`.

Diary 0124 proved that the mode-7 post-mixer tail is correct when fed the
captured `mixer_residual` fixture. The remaining local precision question was
inside the full persistent mixer:

- `mixer_output_max_fp16_ulp = 6`
- `mixer_residual_max_fp16_ulp = 16`

The diagnostic added here recomputes the residual on the host from the actual
GPU `mixer_output`:

```text
derived_residual[row] =
  fp16_round(float(input_hidden[row]) + float(gpu_mixer_output[row]))
```

It then compares that derived residual to both the GPU-written
`mixer_residual` and the captured expected `mixer_residual`.

## Implementation

### Host App: `apps/vk_persistent_layer0_probe.cpp`

`run_full_mixer_mode` now emits four additional JSON fields for both
`--mode full-mixer` and `--mode layer0`:

```text
derived_mixer_residual_exact_mismatches
derived_mixer_residual_max_fp16_ulp
derived_expected_mixer_residual_exact_mismatches
derived_expected_mixer_residual_max_fp16_ulp
```

No shader descriptor layout changed. No shader code changed. No pass/fail gate
was weakened. Existing full-mixer and full-layer gates continue to enforce the
same `--full-mixer-fp16-ulp-tolerance 16` and mode-7 tail bounds.

## Verification

### Build

```sh
cmake --build build -j2
```

Result: build completed and relinked `vk_persistent_layer0_probe`.

### Full Mixer

Command shape:

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

Result: `status = ok`.

Metrics:

- `mixer_output_exact_mismatches = 28`
- `mixer_output_max_fp16_ulp = 6`
- `mixer_residual_exact_mismatches = 8`
- `mixer_residual_max_fp16_ulp = 16`
- `derived_mixer_residual_exact_mismatches = 0`
- `derived_mixer_residual_max_fp16_ulp = 0`
- `derived_expected_mixer_residual_exact_mismatches = 8`
- `derived_expected_mixer_residual_max_fp16_ulp = 16`

### Full Layer

Command shape:

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

Result: `status = ok`, `tail_input_source = persistent_mixer_residual`.

New derived-residual metrics:

- `derived_mixer_residual_exact_mismatches = 0`
- `derived_mixer_residual_max_fp16_ulp = 0`
- `derived_expected_mixer_residual_exact_mismatches = 8`
- `derived_expected_mixer_residual_max_fp16_ulp = 16`

Existing layer metrics remain:

- `mixer_output_max_fp16_ulp = 6`
- `mixer_residual_max_fp16_ulp = 16`
- `output_max_fp16_ulp = 105`
- `tap_norm_max_fp16_ulp = 29`
- `tap_up_max_fp16_ulp = 253`
- `tap_product_max_fp16_ulp = 62`

### Focused Tests

```sh
git diff --check
ctest --test-dir build -R 'persistent_layer0_probe|deltanet_mixer_probe|diary_check' --output-on-failure
```

Result: `git diff --check` clean, `22/22` focused tests passed.

## Interpretation

The residual-add step in the persistent full-mixer path is not the source of
the 16 ULP `mixer_residual` drift. The GPU-written `mixer_residual` exactly
matches the host-derived residual computed from the actual GPU
`mixer_output`.

Therefore, the persistent mixer residual spread is inherited from the
bounded-not-exact `mixer_output` and the fp16 residual-add rounding boundary
against the captured reference. This narrows the next precision work: inspect
or improve the persistent DeltaNet mixer-output path before changing the
residual add or post-mixer tail.

## Current Status

The diagnostic is host-side only and does not change shader behavior. It is
still a captured one-step layer-0 persistent pass. It is not multi-layer
decode, not all 24 layers, not attention coverage, not the token loop, and not
archived basic inference.

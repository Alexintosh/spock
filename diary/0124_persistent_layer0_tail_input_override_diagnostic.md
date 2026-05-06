# 0124: Persistent Layer-0 Tail-Input Override Diagnostic

## Goal

Add a bounded diagnostic to decide whether diary 0123 mode-7 post-mixer
RMSNorm/MLP tap drift is caused by the persistent mixer residual input or by a
bug in the mode-7 tail implementation.

Diary 0123 localized the layer-0 `post_mlp` spread to the post-mixer RMSNorm
and MLP projection boundaries:

- post_norm max 29 fp16 ULP;
- up projection max 253 fp16 ULP;
- activation product max 62 fp16 ULP;
- final `post_mlp` max 105 fp16 ULP.

The missing distinction was whether those larger bounds came from the tail
implementation itself, or from feeding the tail the persistent mixer's
bounded-but-not-exact `mixer_residual`.

## Implementation

### Host App: `apps/vk_persistent_layer0_probe.cpp`

Added:

```text
--layer0-tail-input-override-fp16-file PATH
```

The option is valid only with `--mode layer0`. The host loads exactly 1024 fp16
values and places them in hidden-pack slot 3 before dispatch:

```text
slot 0: input_hidden
slot 1: mixer_output
slot 2: persistent mixer_residual
slot 3: tail-input override before Stage 8, post_mlp output after Stage 11
```

For mode 7, `params.output_rows` is repurposed as a boolean because the mode-7
shader path does not otherwise use it:

- `0`: tail reads slot 2, the normal persistent mixer residual;
- `1`: tail reads slot 3, the override file.

The JSON output reports `tail_input_source` as either
`persistent_mixer_residual` or `override_file`.

### Shader: `shaders/persistent_layer0_probe.comp`

No descriptor layout changed. Mode 7 now computes:

```glsl
const uint tail_residual_offset = (params.output_rows != 0u)
    ? hidden_dim * 3u
    : hidden_dim * 2u;
```

Stage 8 RMSNorm and Stage 11 residual add read from `tail_residual_offset`.
Stage 11 still writes final `post_mlp` to slot 3.

When the override is active, Stage 11 reads and writes the same slot. This is
safe for this diagnostic because each output row is assigned to one workgroup,
and lane 0 reads that row's residual before writing that row's `post_mlp`.

### CMake

Added:

- `spock_persistent_layer0_probe_layer0_override_exact_fails` (`WILL_FAIL`)
- `spock_persistent_layer0_probe_layer0_override_bounded`
- `spock_persistent_layer0_probe_layer0_override_taps_bounded`

The bounded override gates use the measured captured-tail bounds:

- output max 87 ULP;
- post_norm max 0 ULP;
- up projection max 1 ULP;
- activation product max 2 ULP.

## Verification

### Build

The full build was required so the shader SPIR-V was regenerated:

```sh
cmake --build build -j2
```

Result: `persistent_layer0_probe.comp.spv` regenerated and build completed.

### Normal Mode 7 With Persistent Mixer Residual

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

Metrics:

- `mixer_output_max_fp16_ulp = 6`
- `mixer_residual_max_fp16_ulp = 16`
- `output_exact_mismatches = 433`
- `output_max_fp16_ulp = 105`
- `tap_norm_exact_mismatches = 10`
- `tap_norm_max_fp16_ulp = 29`
- `tap_up_exact_mismatches = 668`
- `tap_up_max_fp16_ulp = 253`
- `tap_product_exact_mismatches = 1004`
- `tap_product_max_fp16_ulp = 62`

### Override Mode 7 With Captured Mixer Residual

Same command plus:

```text
--layer0-tail-input-override-fp16-file tests/data/layer0_step1_mixer_residual_1024.fp16
```

Result: `status = ok`, `tail_input_source = override_file`.

Metrics:

- `mixer_output_max_fp16_ulp = 6`
- `mixer_residual_max_fp16_ulp = 16`
- `output_exact_mismatches = 314`
- `output_max_fp16_ulp = 87`
- `tap_norm_exact_mismatches = 0`
- `tap_norm_max_fp16_ulp = 0`
- `tap_up_exact_mismatches = 5`
- `tap_up_max_fp16_ulp = 1`
- `tap_product_exact_mismatches = 11`
- `tap_product_max_fp16_ulp = 2`

### Comparison

| Metric | Normal persistent residual | Captured residual override |
|--------|----------------------------|----------------------------|
| `mixer_output_max_fp16_ulp` | 6 | 6 |
| `mixer_residual_max_fp16_ulp` | 16 | 16 |
| `tap_norm_max_fp16_ulp` | 29 | 0 |
| `tap_up_max_fp16_ulp` | 253 | 1 |
| `tap_product_max_fp16_ulp` | 62 | 2 |
| `output_max_fp16_ulp` | 105 | 87 |

The override does not change the computed mixer metrics. It changes only the
tail input. Feeding captured `mixer_residual` restores the mode-7 tail to the
old captured-tail behavior: exact post_norm, tiny up/product drift, and the
known 87 ULP final `post_mlp` bound.

## Interpretation

The mode-7 tail implementation is not the source of the widened diary 0123
tap bounds. When the same mode-7 shader consumes captured `mixer_residual`, its
post_norm/up/product/output metrics return to the captured-tail envelope.

Therefore, the widened normal mode-7 bounds are caused by the persistent
DeltaNet mixer feeding a bounded-but-not-exact `mixer_residual` into a
reduction-sensitive RMSNorm and large MLP projections. The next quality
decision is not a tail rewrite. It is one of:

1. improve the persistent DeltaNet mixer residual precision;
2. accept and document the downstream amplification from `mixer_residual` max
   16 ULP to post_norm/up/product/post_mlp bounds;
3. run a focused mixer precision experiment before widening mode 7 beyond
   layer 0.

## Current Status

The tail-input override diagnostic is implemented and gated. Existing mode-7
behavior is unchanged when the override is absent.

This remains a captured one-step layer-0 persistent pass. It is still not
multi-layer decode, not all 24 layers, not the token loop, and not archived
basic inference.

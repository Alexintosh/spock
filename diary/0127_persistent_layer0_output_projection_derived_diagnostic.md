# 0127: Persistent Layer-0 Output Projection Derived Diagnostic

## Goal

Decide whether the persistent full-mixer `mixer_output` drift is caused by the
shader output projection implementation itself, or by the reference/reduction
boundary around the projection.

Diary 0126 showed that the input to the DeltaNet output projection is nearly
exact:

- `tap_dn_gated_exact_mismatches = 1`
- `tap_dn_gated_max_fp16_ulp = 1`

but the projection output remained wider:

- `mixer_output_exact_mismatches = 28`
- `mixer_output_max_fp16_ulp = 6`

The diagnostic added here recomputes the output projection on the host from
the actual GPU `dn_gated` values:

```text
derived_mixer_output[row] =
  fp16_round(sum_c float(delta_out_proj[row, c]) * float(gpu_dn_gated[c]))
```

It then compares that derived output to both the GPU `mixer_output` and the
captured expected `mixer_output`.

## Implementation

### Host App: `apps/vk_persistent_layer0_probe.cpp`

When the full-mixer-only `--expected-dn-gated-fp16-file` tap is supplied,
`run_full_mixer_mode` now also emits:

```text
derived_mixer_output_exact_mismatches
derived_mixer_output_max_fp16_ulp
derived_expected_mixer_output_exact_mismatches
derived_expected_mixer_output_max_fp16_ulp
```

No shader code changed. No descriptor layout changed. No existing pass/fail
criteria were weakened. The derived projection is host-side evidence only; it
does not replace the existing `mixer_output` gate.

The host loop follows the same row-major weight access as shader Stage 6:

```text
row_offset = row * 2048
acc += delta_out_proj[row_offset + c] * gpu_dn_gated[c]
```

This is intentionally scalar host code. It is not a performance path. It is a
diagnostic to decide whether the shader output projection is internally
self-consistent with the values it actually consumes.

## Verification

### Build

```sh
cmake --build build -j2
```

Result: build completed and relinked `vk_persistent_layer0_probe`.

### Full-Mixer Derived Output Projection

Command:

```sh
./build/vk_persistent_layer0_probe --mode full-mixer \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --input-norm-fp16-file tests/data/layer0_step1_dn_input_norm_1024.fp16 \
  --input-hidden-fp16-file tests/data/layer0_step1_input_hidden_1024.fp16 \
  --conv-state-pre-fp16-file tests/data/layer0_step1_dn_conv_state_pre_24576.fp16 \
  --state-pre-f32-file tests/data/layer0_step1_dn_recurrent_state_pre_262176.f32 \
  --expected-mixer-output-fp16-file tests/data/layer0_step1_mixer_output_1024.fp16 \
  --expected-mixer-residual-fp16-file tests/data/layer0_step1_mixer_residual_1024.fp16 \
  --full-mixer-fp16-ulp-tolerance 16 \
  --expected-dn-gated-fp16-file tests/data/layer0_step1_dn_gated_2048.fp16 \
  --tap-dn-gated-fp16-ulp-tolerance 1
```

Result: `status = ok`.

Metrics:

- `tap_dn_gated_exact_mismatches = 1`
- `tap_dn_gated_max_fp16_ulp = 1`
- `derived_mixer_output_exact_mismatches = 2`
- `derived_mixer_output_max_fp16_ulp = 1`
- `derived_expected_mixer_output_exact_mismatches = 28`
- `derived_expected_mixer_output_max_fp16_ulp = 6`
- `mixer_output_exact_mismatches = 28`
- `mixer_output_max_fp16_ulp = 6`
- `derived_mixer_residual_max_fp16_ulp = 0`
- `derived_expected_mixer_residual_max_fp16_ulp = 16`

### Focused Tests

```sh
git diff --check
ctest --test-dir build -R 'persistent_layer0_probe|deltanet_mixer_probe|diary_check' --output-on-failure
```

Result: `git diff --check` clean, `24/24` focused tests passed.

## Interpretation

The shader output projection is internally consistent with the actual GPU
`dn_gated` input. A scalar host recomputation from that same GPU input differs
from the GPU `mixer_output` by only two exact mismatches and max 1 ULP. The same
host-derived output still has the full 28 exact mismatches and max 6 ULP
against the captured reference.

That strongly suggests the remaining `mixer_output` spread is a
CPU/reference-vs-GPU projection reduction boundary, plus the one 1 ULP
`dn_gated` input difference, not an output-projection shader bug. The
persistent full-mixer path is therefore self-consistent at the local diagnostic
level:

- norm-gate output is bounded at 1 ULP;
- output projection from actual GPU `dn_gated` matches GPU output at 1 ULP;
- residual add from actual GPU `mixer_output` is exact.

The remaining quality decision is whether this bounded CPU/reference-vs-GPU
projection envelope is acceptable before widening beyond layer 0, or whether to
add a reduction-order-matched reference/gate for the persistent path.

## Current Status

This is still a captured one-step layer-0 persistent pass. It is not
multi-layer decode, not all 24 layers, not attention coverage, not the token
loop, and not archived basic inference.

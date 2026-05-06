# 0126: Persistent Layer-0 Mixer Output dn_gated Tap

## Goal

Localize the persistent full-mixer `mixer_output` drift after diary 0125 proved
the residual add itself is exact relative to the actual GPU `mixer_output`.

The remaining question was whether the full-mixer drift enters before the
DeltaNet output projection, or inside the output projection accumulation:

- `dn_gated -> delta_out_proj -> mixer_output`

## Implementation

### Host App: `apps/vk_persistent_layer0_probe.cpp`

Added an optional full-mixer-only tap:

```text
--expected-dn-gated-fp16-file PATH
--tap-dn-gated-fp16-ulp-tolerance N
```

The tap compares `buf2[0..2047]` after the mode-6 norm-gate stage to the
captured fixture `tests/data/layer0_step1_dn_gated_2048.fp16` and emits:

```text
tap_dn_gated_exact_mismatches
tap_dn_gated_max_fp16_ulp
tap_dn_gated_fp16_ulp_tolerance
```

No shader code changed. The tap is valid only for `--mode full-mixer`.
Mode 7 later reuses `buf2` for the post-mixer MLP product, so the option is
rejected outside full-mixer mode rather than reporting stale data.

This restriction is deliberate. The diagnostic is trying to inspect the value
that feeds the DeltaNet output projection, not add another long-lived debug
buffer to the layer0 path. In mode 6, `buf2` still holds `dn_gated` after the
residual add because the output projection only reads it. In mode 7, the
post-mixer tail reuses the same buffer for later MLP work, so an end-of-dispatch
host read would no longer represent the mixer boundary. Keeping the tap
full-mixer-only avoids a descriptor-layout change and keeps the measurement
tied to one surviving buffer with clear ownership.

### CMake

Added two focused gates:

- `spock_persistent_layer0_probe_full_mixer_dn_gated_exact_fails`
- `spock_persistent_layer0_probe_full_mixer_dn_gated_ulp1`

The exact gate uses the normal full-mixer `16` ULP tolerance for
`mixer_output/mixer_residual`, but leaves the dn-gated tap exact, proving the
single 1 ULP tap mismatch is caught. The bounded gate allows 1 ULP for the tap.

## Verification

### Build

```sh
cmake --build build -j2
```

Result: build completed and relinked `vk_persistent_layer0_probe`.

### Full-Mixer dn_gated Exact Diagnostic

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
  --full-mixer-fp16-ulp-tolerance 16 \
  --expected-dn-gated-fp16-file tests/data/layer0_step1_dn_gated_2048.fp16
```

Result: `status = fail`, as expected for the exact tap gate.

Metrics:

- `tap_dn_gated_exact_mismatches = 1`
- `tap_dn_gated_max_fp16_ulp = 1`
- `mixer_output_exact_mismatches = 28`
- `mixer_output_max_fp16_ulp = 6`
- `mixer_residual_exact_mismatches = 8`
- `mixer_residual_max_fp16_ulp = 16`

### Full-Mixer dn_gated Bounded Diagnostic

Same command plus:

```text
--tap-dn-gated-fp16-ulp-tolerance 1
```

Result: `status = ok`.

Metrics:

- `tap_dn_gated_exact_mismatches = 1`
- `tap_dn_gated_max_fp16_ulp = 1`
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

The persistent full-mixer norm-gate output is nearly exact against the captured
`dn_gated` fixture: one element differs by 1 ULP. The final `mixer_output`
still has 28 exact mismatches and max 6 ULP.

This localizes the remaining mixer-output spread to the DeltaNet output
projection boundary, or to amplification of the one 1 ULP `dn_gated` input
difference through that projection. The residual add and post-mixer tail remain
ruled out by diaries 0125 and 0124 respectively.

The practical consequence is that the next diagnostic should not widen the
full-layer tolerance or rewrite the post-mixer tail. The useful next boundary is
inside `delta_out_proj`: either compare the output projection against a
host-derived projection from the actual GPU `dn_gated`, or add a controlled
output-projection accumulation experiment that can distinguish reduction-order
drift from input drift. If host-derived projection from actual GPU `dn_gated`
matches the GPU `mixer_output`, the shader is internally self-consistent and
the remaining mismatch is a CPU/reference reduction-order boundary. If it does
not match, the output projection implementation itself needs inspection.

This also explains why diary 0125 saw exact derived residuals while final
layer0 drift stayed larger. The residual add faithfully consumes the persistent
`mixer_output`; it is not a corrective boundary. Any bounded error entering
`mixer_output` becomes the tail input, where RMSNorm and MLP projections can
amplify it into the larger mode-7 tap bounds observed in diary 0123.

## Current Status

This is still a captured one-step layer-0 persistent pass. It is not
multi-layer decode, not all 24 layers, not attention coverage, not the token
loop, and not archived basic inference.

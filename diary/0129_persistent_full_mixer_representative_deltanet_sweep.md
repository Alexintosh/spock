# 0129: Persistent Full-Mixer Representative DeltaNet Sweep

## Goal

Finish the representative DeltaNet full-mixer widening that diary 0128 started.

The target model repeats three DeltaNet layers followed by one full-attention
layer. The representative DeltaNet sweep is therefore:

```text
layers 0, 4, 8, 12, 16, 20
```

Layer 0 was already the deeply localized seed case. Layer 4 proved the
`--layer-index` loader path. This entry adds the remaining representative
DeltaNet layers: 8, 12, 16, and 20.

## Implementation

### Fixtures

Generated fresh step-1 component dumps and recurrent-state sidecars for layers
8, 12, 16, and 20:

```sh
build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --tokens /tmp/spock_step1_tokens.txt \
  --max-new-tokens 2 \
  --dump-step-components 1 \
  --dump-dn-recurrent-state-pre-layer L \
  --dump-dn-recurrent-state-pre-file /tmp/spock_layerL_step1_dn_recurrent_state_pre_262176.f32 \
  > /tmp/spock_layerL_components_stdout.txt \
  2> /tmp/spock_layerL_components_stderr.json
```

For each layer, extracted:

```text
dn_input_norm_fp16
input_hidden_fp16
dn_conv_state_pre_fp16
mixer_output_fp16
mixer_residual_fp16
dn_gated_fp16
```

and copied the fp32 recurrent-state sidecar into `tests/data`.

Layer 20 already had `layer20_step1_mixer_residual_1024.fp16` from earlier MLP
work; the newly generated value matched the existing tracked fixture, so only
the missing layer-20 full-mixer fixtures are new in this checkpoint.

### CMake

Added paired exact-fails and bounded gates for layers 8, 12, 16, and 20:

```text
spock_persistent_layer0_probe_full_mixer_layer8_exact_fails
spock_persistent_layer0_probe_full_mixer_layer8_bounded
spock_persistent_layer0_probe_full_mixer_layer12_exact_fails
spock_persistent_layer0_probe_full_mixer_layer12_bounded
spock_persistent_layer0_probe_full_mixer_layer16_exact_fails
spock_persistent_layer0_probe_full_mixer_layer16_bounded
spock_persistent_layer0_probe_full_mixer_layer20_exact_fails
spock_persistent_layer0_probe_full_mixer_layer20_bounded
```

The bounded gate tolerances are layer-specific and measured from direct runs:

| Layer | full-mixer tolerance | dn_gated tap tolerance |
|-------|----------------------|------------------------|
| 8 | 4 | 2 |
| 12 | 16 | 15 |
| 16 | 32 | 2 |
| 20 | 4 | 0 |

## Verification

### Direct Measurements

Commands used the same shape for each layer:

```sh
./build/vk_persistent_layer0_probe --mode full-mixer --layer-index L \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --input-norm-fp16-file tests/data/layerL_step1_dn_input_norm_1024.fp16 \
  --input-hidden-fp16-file tests/data/layerL_step1_input_hidden_1024.fp16 \
  --conv-state-pre-fp16-file tests/data/layerL_step1_dn_conv_state_pre_24576.fp16 \
  --state-pre-f32-file tests/data/layerL_step1_dn_recurrent_state_pre_262176.f32 \
  --expected-mixer-output-fp16-file tests/data/layerL_step1_mixer_output_1024.fp16 \
  --expected-mixer-residual-fp16-file tests/data/layerL_step1_mixer_residual_1024.fp16 \
  --full-mixer-fp16-ulp-tolerance 1000 \
  --expected-dn-gated-fp16-file tests/data/layerL_step1_dn_gated_2048.fp16 \
  --tap-dn-gated-fp16-ulp-tolerance 1000
```

Metrics:

| Layer | mixer_output mismatches / max ULP | mixer_residual mismatches / max ULP | dn_gated mismatches / max ULP | derived output max ULP | derived residual max ULP |
|-------|-----------------------------------|-------------------------------------|-------------------------------|------------------------|--------------------------|
| 8 | 66 / 2 | 10 / 4 | 1 / 2 | 1 | 0 |
| 12 | 127 / 6 | 19 / 16 | 3 / 15 | 1 | 0 |
| 16 | 188 / 25 | 50 / 32 | 7 / 2 | 1 | 0 |
| 20 | 3 / 1 | 2 / 4 | 0 / 0 | 1 | 0 |

### Focused Tests

```sh
cmake --build build -j2
git diff --check
ctest --test-dir build -R 'persistent_layer0_probe|deltanet_mixer_probe|diary_check' --output-on-failure
```

Result: `git diff --check` clean, `34/34` focused tests passed.

## Interpretation

The persistent full-mixer path now has representative DeltaNet coverage across
layers 0, 4, 8, 12, 16, and 20. The same self-consistency pattern holds across
the sweep:

- residual add from actual GPU `mixer_output` remains exact (`derived residual
  max 0 ULP`);
- host-derived output projection from actual GPU `dn_gated` remains max 1 ULP;
- the wider differences are against the captured CPU/reference fixture.

Layer 16 is the widest representative DeltaNet case so far:

- `mixer_output_max_fp16_ulp = 25`
- `mixer_residual_max_fp16_ulp = 32`

Layer 12 has the widest `dn_gated` tap:

- `tap_dn_gated_max_fp16_ulp = 15`

These are still bounded one-step captured validations, not an inference claim.
They give a concrete tolerance envelope for moving from single-layer proof to
bounded multi-layer persistent decode.

## Current Status

Representative DeltaNet full-mixer coverage is closed for layers 0, 4, 8, 12,
16, and 20. This is still not attention coverage, not multi-layer decode, not
all 24 layers, not the token loop, and not archived basic inference.

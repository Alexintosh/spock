# 0128: Persistent Full-Mixer Layer-4 Gate

## Goal

Start widening the persistent DeltaNet full-mixer gate beyond layer 0.

Diaries 0124-0127 reduced the layer-0 precision question to an explainable
CPU/reference-vs-GPU projection envelope. The next milestone is not another
layer-0 diagnostic. It is proving the same persistent full-mixer shader can
load a different DeltaNet layer's weights and captured state, then run the
same six-barrier mixer composition against that layer's runtime fixtures.

Layer 4 is the first representative widening target because the model schedule
repeats three DeltaNet layers followed by one full-attention layer. Layer 4 is
the first DeltaNet layer in the second block.

## Implementation

### Host App: `apps/vk_persistent_layer0_probe.cpp`

Added:

```text
--layer-index N
```

For `--mode full-mixer` and `--mode layer0`, the host now loads weights from:

```text
layer.N.delta_in_proj_qkv
layer.N.delta_in_proj_z
layer.N.delta_in_proj_a
layer.N.delta_in_proj_b
layer.N.delta_conv
layer.N.delta_a_log
layer.N.delta_dt_bias
layer.N.delta_norm
layer.N.delta_out_proj
```

The default remains `--layer-index 0`, so existing layer-0 behavior is
unchanged. Nonzero `--layer-index` is rejected outside full-mixer/layer0 modes.
The JSON output now reports `layer_index`.

No shader code changed. The persistent shader already receives all layer
specific tensors through buffers and control payloads; the missing piece was
host-side role selection.

### Fixtures

Generated a fresh step-1 component dump and layer-4 recurrent-state sidecar:

```sh
build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --tokens /tmp/spock_step1_tokens.txt \
  --max-new-tokens 2 \
  --dump-step-components 1 \
  --dump-dn-recurrent-state-pre-layer 4 \
  --dump-dn-recurrent-state-pre-file /tmp/spock_layer4_step1_dn_recurrent_state_pre_262176.f32 \
  > /tmp/spock_layer4_components_stdout.txt \
  2> /tmp/spock_layer4_components_stderr.json
```

Extracted full-mixer fixtures:

```sh
python3 tools/extract_component_fp16.py --input /tmp/spock_layer4_components_stderr.json --layer 4 --field dn_input_norm_fp16 --output tests/data/layer4_step1_dn_input_norm_1024.fp16
python3 tools/extract_component_fp16.py --input /tmp/spock_layer4_components_stderr.json --layer 4 --field input_hidden_fp16 --output tests/data/layer4_step1_input_hidden_1024.fp16
python3 tools/extract_component_fp16.py --input /tmp/spock_layer4_components_stderr.json --layer 4 --field dn_conv_state_pre_fp16 --output tests/data/layer4_step1_dn_conv_state_pre_24576.fp16
python3 tools/extract_component_fp16.py --input /tmp/spock_layer4_components_stderr.json --layer 4 --field mixer_output_fp16 --output tests/data/layer4_step1_mixer_output_1024.fp16
python3 tools/extract_component_fp16.py --input /tmp/spock_layer4_components_stderr.json --layer 4 --field mixer_residual_fp16 --output tests/data/layer4_step1_mixer_residual_1024.fp16
python3 tools/extract_component_fp16.py --input /tmp/spock_layer4_components_stderr.json --layer 4 --field dn_gated_fp16 --output tests/data/layer4_step1_dn_gated_2048.fp16
cp /tmp/spock_layer4_step1_dn_recurrent_state_pre_262176.f32 tests/data/layer4_step1_dn_recurrent_state_pre_262176.f32
```

Checked-in files:

- `tests/data/layer4_step1_dn_input_norm_1024.fp16`
- `tests/data/layer4_step1_input_hidden_1024.fp16`
- `tests/data/layer4_step1_dn_conv_state_pre_24576.fp16`
- `tests/data/layer4_step1_dn_recurrent_state_pre_262176.f32`
- `tests/data/layer4_step1_mixer_output_1024.fp16`
- `tests/data/layer4_step1_mixer_residual_1024.fp16`
- `tests/data/layer4_step1_dn_gated_2048.fp16`

### CMake

Added paired gates:

- `spock_persistent_layer0_probe_full_mixer_layer4_exact_fails`
- `spock_persistent_layer0_probe_full_mixer_layer4_bounded`

The exact gate proves the layer-4 differences are caught. The bounded gate
uses `--full-mixer-fp16-ulp-tolerance 8` and
`--tap-dn-gated-fp16-ulp-tolerance 9`.

## Verification

### Build

```sh
cmake --build build -j2
```

Result: build completed and relinked `vk_persistent_layer0_probe`.

### Layer 4 Full-Mixer Bounded

Command:

```sh
./build/vk_persistent_layer0_probe --mode full-mixer --layer-index 4 \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --input-norm-fp16-file tests/data/layer4_step1_dn_input_norm_1024.fp16 \
  --input-hidden-fp16-file tests/data/layer4_step1_input_hidden_1024.fp16 \
  --conv-state-pre-fp16-file tests/data/layer4_step1_dn_conv_state_pre_24576.fp16 \
  --state-pre-f32-file tests/data/layer4_step1_dn_recurrent_state_pre_262176.f32 \
  --expected-mixer-output-fp16-file tests/data/layer4_step1_mixer_output_1024.fp16 \
  --expected-mixer-residual-fp16-file tests/data/layer4_step1_mixer_residual_1024.fp16 \
  --full-mixer-fp16-ulp-tolerance 8 \
  --expected-dn-gated-fp16-file tests/data/layer4_step1_dn_gated_2048.fp16 \
  --tap-dn-gated-fp16-ulp-tolerance 9
```

Result: `status = ok`.

Metrics:

- `layer_index = 4`
- `mixer_output_exact_mismatches = 59`
- `mixer_output_max_fp16_ulp = 7`
- `mixer_residual_exact_mismatches = 14`
- `mixer_residual_max_fp16_ulp = 8`
- `derived_mixer_residual_exact_mismatches = 0`
- `derived_mixer_residual_max_fp16_ulp = 0`
- `derived_expected_mixer_residual_exact_mismatches = 14`
- `derived_expected_mixer_residual_max_fp16_ulp = 8`
- `tap_dn_gated_exact_mismatches = 2`
- `tap_dn_gated_max_fp16_ulp = 9`
- `derived_mixer_output_exact_mismatches = 1`
- `derived_mixer_output_max_fp16_ulp = 1`
- `derived_expected_mixer_output_exact_mismatches = 59`
- `derived_expected_mixer_output_max_fp16_ulp = 7`

### Focused Tests

```sh
git diff --check
ctest --test-dir build -R 'persistent_layer0_probe|deltanet_mixer_probe|diary_check' --output-on-failure
```

Result: `git diff --check` clean, `26/26` focused tests passed.

## Interpretation

The same persistent full-mixer shader now runs a non-layer-0 DeltaNet layer
with layer-specific weights and captured state. Layer 4 preserves the same
self-consistency pattern discovered on layer 0:

- residual add from actual GPU `mixer_output` is exact;
- host-derived output projection from actual GPU `dn_gated` matches GPU
  `mixer_output` within max 1 ULP;
- the wider mismatch is against the captured CPU/reference fixture.

Layer 4 has a larger `dn_gated` tap envelope than layer 0: max 9 ULP versus
layer 0's max 1 ULP. That means representative widening cannot assume the
layer-0 tap bound globally. The output and residual still remain tightly
bounded: `mixer_output` max 7 ULP and `mixer_residual` max 8 ULP.

The next widening step is to repeat this for the remaining representative
DeltaNet layers: 8, 12, 16, and 20. Full attention layers remain uncovered.

## Current Status

This is still captured one-step full-mixer validation, now covering layers 0
and 4. It is not multi-layer decode, not all 24 layers, not attention coverage,
not the token loop, and not archived basic inference.

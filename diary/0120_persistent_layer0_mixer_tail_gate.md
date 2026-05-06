# 0120: Persistent Layer-0 Mixer-Tail Gate

## Goal

Validate the downstream DeltaNet mixer tail inside `persistent_layer0_probe.comp`:

```text
dn_core_fp16 + dn_z_fp16 + layer.0.delta_norm
  -> dn_gated_fp16
  -> layer.0.delta_out_proj
  -> mixer_output_fp16
  -> input_hidden_fp16 + mixer_output_fp16
  -> mixer_residual_fp16
```

This is the persistent counterpart to the standalone downstream gates from
diaries 0099, 0101, and 0102. It is not the full DeltaNet mixer because it
still consumes captured `dn_core_fp16` and `dn_z_fp16`.

## Why This Gate Matters

After diary 0119, the persistent probe had the recurrent core output. The next
contract downstream is the transformation from recurrent output to the first
layer residual handoff:

- norm-gate converts `dn_core` and `dn_z` into the gated DeltaNet vector;
- output projection maps the 2048-wide DeltaNet vector back to hidden size;
- residual add produces the `mixer_residual` input for the post-mixer MLP tail.

This gate closes that downstream path in the persistent shader shape before
attempting a single composed persistent mixer from `dn_input_norm`.

## Implementation

### Shader: `shaders/persistent_layer0_probe.comp`

- Added `mode=5`, named `mixer-tail` at the CLI.
- Binding layout:
  - binding 0: control payload, with `layer.0.delta_norm` fp32 bits in
    `control.extra[0..127]`
  - binding 1: `dn_core_fp16[2048]`, overwritten in-place with `dn_gated_fp16`
  - binding 2: `dn_z_fp16[2048]`
  - binding 3: `input_hidden_fp16[1024]`
  - binding 4: `layer.0.delta_out_proj` fp16 matrix `[1024 x 2048]`
  - binding 5: `mixer_output_fp16[1024]`
  - binding 6: `mixer_residual_fp16[1024]`
- Stage 1 runs per-head RMSNorm + SiLU gate with 16 resident 128-lane
  workgroups.
- A software global barrier separates norm-gate from output projection.
- Stage 2 runs the output projection with row-strided 128-lane reductions
  across the resident workgroups.
- A second software global barrier separates output projection from residual
  add.
- Stage 3 computes `input_hidden + mixer_output -> mixer_residual`.
- Expected barrier generation is `2`.

### Host App: `apps/vk_persistent_layer0_probe.cpp`

- Added `--mode mixer-tail`.
- Added required inputs:
  - `--core-fp16-file`
  - `--z-fp16-file`
  - `--input-hidden-fp16-file`
  - `--expected-mixer-output-fp16-file`
  - `--expected-mixer-residual-fp16-file`
- Added `--mixer-tail-fp16-ulp-tolerance N`, defaulting to exact.
- Loads `layer.0.delta_norm` as fp32 rank-1 length 128 and packs raw bits into
  the control payload.
- Loads `layer.0.delta_out_proj` as fp16 `[1024 x 2048]`.

### CMake

Added paired gates:

- `spock_persistent_layer0_probe_mixer_tail_exact_fails`
- `spock_persistent_layer0_probe_mixer_tail_ulp1`

The exact gate is intentionally marked `WILL_FAIL`.

## Verification

### Direct Exact Command

```sh
./build/vk_persistent_layer0_probe --mode mixer-tail \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --core-fp16-file tests/data/layer0_step1_dn_core_2048.fp16 \
  --z-fp16-file tests/data/layer0_step1_dn_z_2048.fp16 \
  --input-hidden-fp16-file tests/data/layer0_step1_input_hidden_1024.fp16 \
  --expected-mixer-output-fp16-file tests/data/layer0_step1_mixer_output_1024.fp16 \
  --expected-mixer-residual-fp16-file tests/data/layer0_step1_mixer_residual_1024.fp16
```

Result:

- `status = fail`
- `failures = 0`
- `arrived = 0`
- `generation = 2`
- `expected_generation = 2`
- `mixer_output_exact_mismatches = 2`
- `mixer_output_max_fp16_ulp = 1`
- `mixer_residual_exact_mismatches = 0`
- `mixer_residual_max_fp16_ulp = 0`

### Direct Bounded Command

```sh
./build/vk_persistent_layer0_probe --mode mixer-tail \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --core-fp16-file tests/data/layer0_step1_dn_core_2048.fp16 \
  --z-fp16-file tests/data/layer0_step1_dn_z_2048.fp16 \
  --input-hidden-fp16-file tests/data/layer0_step1_input_hidden_1024.fp16 \
  --expected-mixer-output-fp16-file tests/data/layer0_step1_mixer_output_1024.fp16 \
  --expected-mixer-residual-fp16-file tests/data/layer0_step1_mixer_residual_1024.fp16 \
  --mixer-tail-fp16-ulp-tolerance 1
```

Result:

- `status = ok`
- `mixer_output_exact_mismatches = 2`
- `mixer_output_max_fp16_ulp = 1`
- `mixer_residual_exact_mismatches = 0`
- `mixer_residual_max_fp16_ulp = 0`

The 1-ULP mixer-output boundary is expected from using the persistent
128-lane row-strided output projection instead of the standalone serial
matvec accumulation order. The final residual handoff remains exact for this
fixture.

## Known Limitations

- This mode starts from captured `dn_core` and `dn_z`; it does not yet compose
  projection-prefix, conv/L2, g/beta, recurrent, and mixer-tail into one pass.
- The output projection has a documented 1-ULP reduction-order boundary.
- Single layer (0) and one captured step only. Not a representative-layer sweep.
- Not full layer persistence, not inference, not the megakernel.

## Next Work

1. Compose persistent projection-prefix, conv/L2, g/beta, recurrent, and
   mixer-tail in one `persistent_layer0_probe` mode.
2. Feed that composed mixer output into the existing persistent post-mixer tail.
3. Compare full layer-0 persistent output against captured `post_mlp`.

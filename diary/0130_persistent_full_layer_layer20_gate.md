# 0130: Persistent Full-Layer Layer-20 Gate

## Goal

Widen the composed persistent full-layer path beyond layer 0.

Diary 0129 closed representative DeltaNet mixer coverage for layers 0, 4, 8,
12, 16, and 20, but those gates intentionally stopped at `mixer_residual`.
That was enough to validate the token mixer envelope. It was not enough to prove
that a mid-network layer can continue through the post-mixer RMSNorm and MLP
tail in the same persistent dispatch.

Layer 20 is the first useful widening target because it already has both:

- representative full-mixer fixtures from diary 0129;
- a checked-in `post_mlp` fixture from the earlier layer-20 MLP precision work.

## Implementation

### Full-Layer Mode Alias

`vk_persistent_layer0_probe` now accepts:

```text
--mode full-layer
```

as the explicit spelling for the existing mode-7 composed path. The legacy
spelling remains accepted:

```text
--mode layer0
```

The executable and shader names still contain `layer0` because they grew from
the layer-0 seed probe, but the mode is now layer-selectable through
`--layer-index` for both `full-mixer` and `full-layer`.

The JSON mode string for the composed path is now:

```json
"mode": "full-layer"
```

This reduces ambiguity before moving to bounded multi-layer work.

### CTest Gates

Added paired layer-20 full-layer tests:

```text
spock_persistent_layer0_probe_full_layer_layer20_exact_fails
spock_persistent_layer0_probe_full_layer_layer20_bounded
```

The exact gate is marked `WILL_FAIL`. The bounded gate requires:

```text
full-mixer max <= 4 fp16 ULP
post_mlp max <= 265 fp16 ULP
```

The layer-20 post-MLP bound is wider than layer 0, but it is consistent with the
known MLP residual-tail behavior recorded before the persistent full-mixer path
was composed. This gate does not weaken any existing layer-0 gate.

## Verification

### Direct Measurement

```sh
./build/vk_persistent_layer0_probe --mode full-layer --layer-index 20 \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --input-norm-fp16-file tests/data/layer20_step1_dn_input_norm_1024.fp16 \
  --input-hidden-fp16-file tests/data/layer20_step1_input_hidden_1024.fp16 \
  --conv-state-pre-fp16-file tests/data/layer20_step1_dn_conv_state_pre_24576.fp16 \
  --state-pre-f32-file tests/data/layer20_step1_dn_recurrent_state_pre_262176.f32 \
  --expected-mixer-output-fp16-file tests/data/layer20_step1_mixer_output_1024.fp16 \
  --expected-mixer-residual-fp16-file tests/data/layer20_step1_mixer_residual_1024.fp16 \
  --expected-output-fp16-file tests/data/layer20_step1_post_mlp_1024.fp16 \
  --full-mixer-fp16-ulp-tolerance 10000 \
  --output-fp16-ulp-tolerance 10000
```

Observed metrics:

```text
failures = 0
arrived = 0
generation = 10
expected_generation = 10
mixer_output = 3 exact mismatches, max 1 ULP
mixer_residual = 2 exact mismatches, max 4 ULP
derived mixer residual vs GPU residual = max 0 ULP
post_mlp = 236 exact mismatches, max 265 ULP
```

### Focused Tests

```sh
cmake --build build -j2
ctest --test-dir build -R 'full_layer_layer20|persistent_layer0_probe_layer0_bounded|persistent_layer0_probe_layer0_override_bounded|persistent_layer0_probe_help' --output-on-failure
```

Result before diary/doc updates: `5/5` tests passed.

## Interpretation

The project now has a mid-network DeltaNet layer running through the composed
persistent full-layer path:

```text
dn_input_norm
-> full DeltaNet mixer
-> mixer_residual
-> post_norm
-> MLP
-> post_mlp
```

inside one persistent dispatch with 10 software global barriers.

This is still not multi-layer decode. The hidden handoff between adjacent
layers is not produced by this probe; it is still supplied from captured
fixtures. The value of this checkpoint is narrower and important: the
single-layer composed path is no longer a layer-0-only artifact, so bounded
multi-layer work can start from a less fragile assumption.

## Current Status

Layer 20 full-layer coverage is now gated. The remaining target-path work is:

- bounded multi-layer DeltaNet handoff;
- attention-layer persistent coverage;
- all-24-layer persistent or strongest honest fused Vulkan path;
- final RMSNorm, LM head, token selection, and GPU-resident token loop;
- archived basic inference from the target path.

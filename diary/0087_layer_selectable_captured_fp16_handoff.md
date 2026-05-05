# 0087: Layer-Selectable Captured FP16 Handoff -- Layer 1 Validated

## Goal

Extend the captured fp16 handoff validation beyond layer 0 by making `vk_persistent_mlp_probe` layer-selectable and validating layer 1.

## Background

Diary 0086 validated the persistent MLP probe against a real layer 0 step 1 `mixer_residual` capture, establishing per-row fp16 output equality as the authoritative pass/fail gate. The probe was hardcoded to layer 0 (`layer.0.mlp_gate/up/down`). To build confidence that the handoff is correct across the full model, validation must cover multiple layers.

## Implementation

### `--layer N` CLI option

Added `--layer N` to `vk_persistent_mlp_probe` (default 0). Validates `N >= 0`, rejects non-integer input with a JSON error and exit code 2. Weight role construction now uses `"layer." + std::to_string(layer) + ".mlp_gate"` etc. JSON output always includes `"layer": N`.

### Layer 1 fixture: `tests/data/layer1_step1_mixer_residual_1024.fp16`

A 2048-byte raw little-endian fp16 file containing 1024 fp16 values extracted from layer 1 step 1 `mixer_residual` in the same capture JSON used for layer 0 in diary 0086.

### Layer 1 probe run

```
vk_persistent_mlp_probe --hidden 1024 --intermediate 3584 --output-rows 1024 --workgroups 82 \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --input-fp16-file tests/data/layer1_step1_mixer_residual_1024.fp16 \
  --residual --layer 1
```

Result:

- output_mismatches: 0
- checksum: 2888553996
- expected_checksum: 2888553996
- layer: 1

All 1024 final fp16 output rows match exactly. Unlike layer 0's SiLU rounding divergence (checksum 67820897 vs expected 67824746), layer 1 produces an exact checksum match -- no fp32 diagnostic divergence at all.

### New CTest gates

- `spock_persistent_mlp_probe_layer1_captured_fp16_handoff` -- full real-weight layer 1 captured handoff.
- `spock_persistent_mlp_probe_layer_invalid_negative` -- validates `--layer -1` returns exit 2 (`WILL_FAIL TRUE`).
- `spock_persistent_mlp_probe_layer_invalid_partial` -- validates `--layer 1abc` returns exit 2 (`WILL_FAIL TRUE`).

## Verification

### Build

`cmake --build build -j` succeeded.

### Layer 0 default preserved

```
vk_persistent_mlp_probe --hidden 1024 --intermediate 3584 --output-rows 1024 --workgroups 82 \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --input-fp16-file tests/data/layer0_step1_mixer_residual_1024.fp16 \
  --residual
```

Result: output_mismatches == 0, checksum 67820897. Layer 0 behavior unchanged by the `--layer` addition.

### Layer 1

output_mismatches == 0, checksum 2888553996 == expected_checksum 2888553996 (exact match).

### CTest suite

All 12 `spock_persistent_mlp_probe` + `spock_extract_component_fp16` tests pass (100%).

## Interpretation

Layer 1's exact checksum match (no SiLU rounding divergence) shows that the GLSL `exp` vs `std::exp` divergence observed in diary 0086 is input-dependent and layer-specific, not systematic. Both layers produce exact fp16 output agreement. The fp16 output gate holds across two layers with qualitatively different intermediate-stage behavior.

## What This Is

- **Layer-selectable MLP probe CLI** via `--layer N`.
- **Layer 1 captured handoff validated** with exact fp16 output equality.
- **Confirmation that the fp16 output gate holds across two layers** (0 and 1).

## What This Is Not

- **Not inference.** No token generation, no RMSNorm, no attention/DeltaNet, no LM head.
- **Not all layers validated.** Only layers 0 and 1.
- **Not the megakernel.** This is a standalone probe with captured real data.

## Next Work

1. Validate additional layers (2--23) to confirm the gate holds across the full model.
2. Add RMSNorm-before-MLP with real norm weights.
3. Compose the MLP probe with the token-mixer side of a layer-shaped persistent probe.

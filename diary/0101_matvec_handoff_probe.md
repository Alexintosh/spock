# 0101: Vulkan Matvec Handoff Probe -- DeltaNet Output Projection Gate

## Goal

Add a generic Vulkan matvec handoff probe that exercises `shaders/matvec.comp` with
real model weights and captured activations, targeting the DeltaNet layer 0 output
projection:

```
dn_gated_fp16 [2048] x delta_out_proj [1024, 2048] -> dn_out_fp16 [1024]
```

This is the first standalone probe that validates the matvec shader outside the
decode loop, using real fp16 weights loaded from the repacked artifact and real
captured fp16 activations as inputs.

## Probe design

Added `vk_matvec_probe`, a minimal Vulkan compute probe that:

1. Loads an fp16 weight matrix from the repack artifact via `--weight-role`
2. Loads an fp16 input vector from a file via `--input-fp16-file`
3. Loads an fp16 expected output vector from a file via `--expected-output-fp16-file`
4. Dispatches `matvec.comp` with appropriate push constants (`out_dim`, `in_dim`)
5. Downloads the output and compares fp16 bit patterns row-by-row
6. Emits JSON diagnostics: `status`, `output_exact_mismatches`,
   `output_within_tolerance`, `output_mismatches`, `max_fp16_ulp_diff`,
   `output_fp16_ulp_tolerance`, and optional `first_mismatch_row`

The probe reuses:
- code style from `vk_residual_add_probe.cpp` (argument parsing, ULP accounting,
  JSON output)
- weight loading from `vk_persistent_mlp_probe.cpp` (artifact loading,
  `load_weight_matrix`)
- the existing `matvec.comp` shader unchanged

## Fixtures

Extracted from the same deterministic step-1 component dump used by diary 0099:

```
tests/data/layer0_step1_dn_gated_2048.fp16   (2048 fp16 values)
tests/data/layer0_step1_dn_out_1024.fp16     (1024 fp16 values)
```

These are extracted from `/tmp/spock_components1_mixer_stderr.txt` using
`tools/extract_component_fp16.py` with fields `dn_gated_fp16` and `dn_out_fp16`.
The checked-in fixtures were byte-compared against fresh extractions from that
dump, and `layer0_step1_dn_out_1024.fp16` was byte-compared against the existing
`layer0_step1_mixer_output_1024.fp16` fixture.

The weight role `layer.0.delta_out_proj` is loaded from the existing repacked
artifact at `artifacts/spock-text-repack-qwen35-0p8b`.

## Direct command

```
build/vk_matvec_probe \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --weight-role layer.0.delta_out_proj \
  --in-dim 2048 \
  --out-dim 1024 \
  --input-fp16-file tests/data/layer0_step1_dn_gated_2048.fp16 \
  --expected-output-fp16-file tests/data/layer0_step1_dn_out_1024.fp16
```

Result:

```json
{
  "in_dim": 2048,
  "out_dim": 1024,
  "weight_role": "layer.0.delta_out_proj",
  "status": "ok",
  "output_exact_mismatches": 0,
  "output_mismatches": 0,
  "max_fp16_ulp_diff": 0,
  "output_fp16_ulp_tolerance": 0
}
```

All 1024 output rows match bit-for-bit.

## CTest gates

Two tests:

- `spock_matvec_probe_help` -- validates --help exit
- `spock_matvec_probe_layer0_delta_out_proj_exact` -- validates exact fp16 output
  for layer 0 DeltaNet output projection

The exact test is a GPU-vs-runtime-GPU comparison: both the captured expected output
and the probe output come from the same matvec shader (just different codepaths:
runtime decode vs standalone probe). No CPU math tolerance is justified.

## What this proves

- `matvec.comp` produces bit-identical output to the runtime decode path for the
  layer 0 DeltaNet output projection at model width.
- The weight loading path from repacked artifact to matvec weight buffer is correct
  for `layer.0.delta_out_proj` [1024, 2048].
- The captured `dn_gated_fp16` -> `dn_out_fp16` equation is algebraically closed
  for this single layer/step.

## What this does not prove

- Not token-mixer computation parity (no DeltaNet recurrent/norm/gate pieces).
- Not multi-layer matvec correctness.
- Not attention output projection.
- Not MLP gate/up/down projections (those are already covered by
  `vk_persistent_mlp_probe`).
- Not full layer-shaped persistent decode.
- Not inference or megakernel completion.

## Interpretation

This probe fills a specific gap in the dependency ladder. Before this entry, the
project had:

- runtime captures of `dn_gated_fp16` and `dn_out_fp16` from the decode dump,
  but no standalone probe that exercised the matvec shader at that boundary.
- `vk_residual_add_probe` closing the post-mixer residual equation.
- `vk_persistent_mlp_probe` closing the RMSNorm+MLP equation.

Now the DeltaNet output projection is independently validated at the matvec shader
level. The next token-mixer work can walk backward through the DeltaNet pipeline
with confidence that the final projection is correct.

## Verification

- `cmake --build build -j` clean
- Direct `vk_matvec_probe` command passes exact (0 mismatches)
- CTest suite: `spock_matvec_probe_help`, `spock_matvec_probe_layer0_delta_out_proj_exact`,
  plus existing `spock_residual_add_probe`, `spock_extract_component_fp16`,
  `spock_diary_check`
- fresh `dn_gated_fp16` and `dn_out_fp16` extractions byte-match the checked-in
  fixtures
- `layer0_step1_dn_out_1024.fp16` byte-matches
  `layer0_step1_mixer_output_1024.fp16`

## Remaining scope

- Not DeltaNet recurrent/norm/gate persistent probe.
- Not attention persistent probe.
- Not full layer-shaped persistent decode.
- Not inference or megakernel completion.

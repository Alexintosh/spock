# 0114: Persistent Layer-0 Tail Probe -- 128-Lane Post-Mixer Scaffold

## Goal

Establish the first layer-shaped persistent scaffold with `local_size_x=128` and 82 workgroups, validating the 128-lane persistent execution shape for the post-mixer half before adding DeltaNet mixer stages into the same shader.

## Why 128 Lanes

The existing `vk_persistent_mlp_probe` uses 64 lanes, which matches the hardware subgroup size. However, DeltaNet recurrent computation uses one 128-lane workgroup per head (16 heads x 128 head_dim = 2048, processed as 128 lanes per head). The future layer-shaped persistent probe must use 128 lanes so the same workgroup shape handles both the recurrent mixer and the post-mixer MLP tail.

This probe validates that 128-lane persistent dispatch works correctly for the post-mixer path before the mixer stages are added.

## Implementation

### Shader: `shaders/persistent_layer0_probe.comp`

- `local_size_x = 128` (not 64)
- `shared float lane_sums[128]` (not 64)
- Lane strides by 128, tree reduction stride 64->32->16->8->4->2->1
- Four stages separated by software global barriers:
  1. **Stage 0: RMSNorm** over `mixer_residual` input, writing `norm_output` fp16
  2. **Stage 1: gate/up matvec** from `norm_output` into scratch buffers
  3. **Stage 2: SiLU(gate) * up** activation product
  4. **Stage 3: down matvec + residual add** with raw `mixer_residual` -> `post_mlp` output
- Push constants: `workgroup_count`, `hidden`, `intermediate_count`, `output_rows`
- 10 storage buffer bindings (control, gate scratch, up scratch, input, gate weight, up weight, down weight, output, norm output, norm weight)
- Control buffer tracks `arrived`, `generation`, `failures`, `checksum`
- No optional branches: always runs RMSNorm, always runs residual add

### Host App: `apps/vk_persistent_layer0_probe.cpp`

- Fixed dimensions: hidden=1024, intermediate=3584, output_rows=1024
- Fixed layer 0: loads `layer.0.post_norm`, `layer.0.mlp_gate`, `layer.0.mlp_up`, `layer.0.mlp_down`
- 128-lane CPU reference functions mirror the shader's reduction order
- CLI: `--repack-dir`, `--input-fp16-file`, `--expected-output-fp16-file`, `--workgroups N`, `--output-fp16-ulp-tolerance N`, `--output-fp16-population-ulp-threshold N`, `--output-fp16-max-rows-above-population-threshold N`, `--help`
- JSON output: status, workgroups, generation, failures, checksum, exact/tolerance/mismatch counts, ULP distribution buckets, population gate

### CMake

- Shader added to `SPOCK_SHADER_SOURCES`
- App added to executable foreach
- Three CTest entries:
  - `spock_persistent_layer0_probe_help` -- CLI help
  - `spock_persistent_layer0_probe_post_mlp_exact_fails` -- WILL_FAIL, exact comparison against captured runtime output
  - `spock_persistent_layer0_probe_post_mlp_bounded` -- passing bounded gate with tolerance 87, population threshold 16, max rows above threshold 10

## Verification

### Structural

- `generation == 3` (3 global barriers: Stage0->Stage1, Stage1->Stage2, Stage2->Stage3)
- `failures == 0`
- `arrived == 0` after completion
- GPU checksum matches 128-lane CPU reference exactly (3274228010)

### Output Comparison Against Captured Runtime

- Input: `tests/data/layer0_step1_mixer_residual_1024.fp16`
- Expected: `tests/data/layer0_step1_post_mlp_1024.fp16`
- Exact: 314 mismatches, max 87 ULP (fails as expected -- WILL_FAIL test)
- Bounded: tolerance 87, population threshold 16, max 10 rows above threshold -> **passes**

### ULP Distribution (1024 output rows)

| ULP range | Count |
|-----------|-------|
| <= 1       | 955   |
| <= 2       | 982   |
| <= 4       | 1005  |
| <= 8       | 1013  |
| <= 16      | 1014  |
| <= 32      | 1019  |
| <= 64      | 1023  |
| > 64      | 1     |

The distribution is identical to the 64-lane `vk_persistent_mlp_probe` with the same captured input. The 128-lane accumulation order does not change the precision profile relative to the captured runtime output (which uses a different accumulation path from the conventional Vulkan runtime).

### CTest Suite

All targeted tests pass:
- `spock_persistent_layer0_probe_help`
- `spock_persistent_layer0_probe_post_mlp_exact_fails` (WILL_FAIL)
- `spock_persistent_layer0_probe_post_mlp_bounded`
- `spock_deltanet_mixer_probe_help`
- `spock_deltanet_mixer_probe_layer0_exact`
- `spock_persistent_mlp_probe_layer0_captured_rmsnorm_mlp_*` (4 tests)
- `spock_diary_check` (113 entries)

## Known Limitations

- This is a tail-only probe: it does not implement DeltaNet mixer stages. The mixer_residual is loaded from a captured fixture, not computed by persistent shader stages.
- The 128-lane checksum matches the 128-lane CPU reference exactly, but differs from the captured runtime `post_mlp` because the runtime uses different accumulation order. This is the same precision boundary documented in diaries 0091-0098 for the 64-lane probe.
- Single layer (0) only. Not multi-layer, not representative-layer sweep.
- Not inference, not full layer persistence, not the megakernel.

## Next Work

1. Add DeltaNet mixer stages (projections, conv/L2, g/beta, recurrent, norm-gate, out-proj, residual-add) before the post-mixer tail in the same 128-lane persistent shader.
2. Validate the full layer-0 pipeline: `dn_input_norm -> mixer -> mixer_residual -> post_norm -> MLP -> post_mlp` in a single persistent dispatch.
3. Sweep the full layer probe across representative layers before attempting 24-layer persistent decode.

## Bounded Policy

The passing bounded gate uses:
- `--output-fp16-ulp-tolerance 87`
- `--output-fp16-population-ulp-threshold 16`
- `--output-fp16-max-rows-above-population-threshold 10`

These are identical to the existing `vk_persistent_mlp_probe` layer-0 policy (diary 0097). The 128-lane execution shape does not change the precision profile.

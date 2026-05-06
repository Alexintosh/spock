# 0116: Persistent Layer-0 Projection-Prefix Gate

## Goal

Validate that the persistent layer-0 shader can compute the stateless DeltaNet projection prefix -- `dn_input_norm -> qkv_raw(6144), z(2048), a(16), b(16)` -- inside a 128-lane persistent dispatch with `local_size_x=128` and 82 workgroups. This is the first persistent DeltaNet projection-prefix stage, preceding conv/L2, g/beta, and recurrent stages.

This is not full mixer computation, not full layer composition, not inference, and not the megakernel. It validates only the projection fanout from `dn_input_norm_fp16` through four weight matrices in a single persistent dispatch.

## Why Projection Prefix First

The projection prefix is the stateless, per-layer-independent head of the DeltaNet mixer. It has no recurrent state, no conv state, no inter-head interaction, and no normalization dependencies beyond the input. Every later mixer stage (conv/L2, g/beta, recurrent, norm-gate) consumes outputs these projections produce. Gating projections first means a future recurrent failure cannot be blamed on a projection bug.

This is also the simplest stage to add to the existing persistent shader because it reuses the same lane-strided fp16 dot-product and tree-reduction pattern already proven in the tail probe (diary 0114).

## Implementation

### Shader: `shaders/persistent_layer0_probe.comp`

- Added `mode=1` projection-prefix path alongside the existing `mode=0` tail path.
- Mode selection via push constant `mode` (0 = tail, 1 = projections).
- Projection mode loads four weight matrices: `layer.0.delta_in_proj_qkv` (6144x1024), `layer.0.delta_in_proj_z` (2048x1024), `layer.0.delta_in_proj_a` (16x1024), `layer.0.delta_in_proj_b` (16x1024).
- Input: `dn_input_norm_fp16` (1024 values).
- Output: qkv_raw (6144), z (2048), a (16), b (16) written to coherent scratch buffers.
- Projection mode uses no software global barriers; command submission completion/queue wait precedes host readback. Generation stays 0 because `global_barrier()` is not called.
- Row-strided coverage: each workgroup covers multiple output rows via `row = group; row < total_rows; row += workgroups`.

### Host App: `apps/vk_persistent_layer0_probe.cpp`

- Added `--mode tail|projections` (default `tail`).
- Projection mode accepts `--input-norm-fp16-file`, `--expected-qkv-raw-fp16-file`, `--expected-z-fp16-file`, `--expected-a-fp16-file`, `--expected-b-fp16-file`.
- Added `--projection-fp16-ulp-tolerance N` (default 0, exact) with per-output exact mismatch counts and max ULP reporting.
- JSON output includes per-output exact mismatch counts and max ULP, plus structural checks (generation, failures, arrived).

### CMake

- Two new CTest entries:
  - `spock_persistent_layer0_probe_projection_prefix_exact_fails` (WILL_FAIL) -- proves exact comparison detects 8 qkv + 1 z mismatches at 1 ULP.
  - `spock_persistent_layer0_probe_projection_prefix_ulp1` -- proves ULP-1 tolerance passes.
- Comment updated from "GPU-vs-CPU boundary" to "128-lane reduction-order boundary" to accurately describe the precision boundary.

## Verification

### Structural

- `generation == 0` (projection mode calls no barriers; generation remains at its initial value).
- `failures == 0`.
- `arrived == 0` after completion.
- Structural checks identical to tail mode pass for the projection path.

### Output Comparison Against Captured Runtime

- Input: `tests/data/layer0_step1_dn_input_norm_1024.fp16`.
- Expected qkv_raw: `tests/data/layer0_step1_dn_qkv_raw_6144.fp16`.
- Expected z: `tests/data/layer0_step1_dn_z_2048.fp16`.
- Expected a: `tests/data/layer0_step1_dn_a_16.fp16`.
- Expected b: `tests/data/layer0_step1_dn_b_16.fp16`.

Exact comparison results:
- qkv_raw: 8 exact mismatches, max 1 ULP.
- z: 1 exact mismatch, max 1 ULP.
- a: 0 exact mismatches.
- b: 0 exact mismatches.
- Total mismatches concentrated at the 1-ULP boundary; `--projection-fp16-ulp-tolerance 1` clears all.

The 1-ULP mismatches on qkv_raw and z are caused by the 128-lane reduction order differing from the conventional runtime's 64-lane dispatch order. The conventional runtime and the single-submit `vk_matvec_probe` produce exact results because they share the same reduction width. The persistent shader's wider 128-lane tree reduction changes the fp32 accumulation order for intermediate partial sums, which can shift fp16 rounding at a small number of output positions. This is a documented reduction-order boundary, not a correctness bug.

### CTest Suite

All 11 targeted tests pass:
- `spock_matvec_probe_layer0_delta_z_proj_exact`
- `spock_matvec_probe_layer0_delta_qkv_proj_exact`
- `spock_matvec_probe_layer0_delta_a_proj_exact`
- `spock_matvec_probe_layer0_delta_b_proj_exact`
- `spock_deltanet_mixer_probe_layer0_exact`
- `spock_persistent_layer0_probe_help`
- `spock_persistent_layer0_probe_post_mlp_exact_fails`
- `spock_persistent_layer0_probe_post_mlp_bounded`
- `spock_persistent_layer0_probe_projection_prefix_exact_fails`
- `spock_persistent_layer0_probe_projection_prefix_ulp1`
- `spock_diary_check`

## Known Limitations

- This is projection-prefix only. It does not implement conv/L2, g/beta, recurrent, norm-gate, output projection, or residual add. The outputs stop at raw qkv, z, a, and b.
- The 1-ULP boundary on qkv_raw and z is a reduction-order artifact specific to the 128-lane persistent shader vs the 64-lane conventional runtime. It is not a general precision guarantee for later stages.
- Single layer (0) only. Not multi-layer, not representative-layer sweep.
- Not full mixer, not full layer, not inference, not the megakernel.

## Next Work

1. Add persistent conv/L2 stages that consume qkv_raw and produce q/k/v, using the already captured and gated q/k/v fixtures.
2. Add persistent g/beta computation and recurrent core stages.
3. Add persistent norm-gate, output projection, and first residual add.
4. Compose the existing persistent post-mixer tail after the mixer stages in the same shader.
5. Compare full layer-0 persistent output against captured `post_mlp`.
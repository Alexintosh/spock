# 0077: Persistent Decode Skeleton — Multi-Role Real-Weight Probe

## Goal

Extend `vk_persistent_decode_skeleton` with a multi-real-weight probe mode that validates several fp16 rank-2 weight roles in one invocation, exercising more of the weight artifact format while preserving existing single-role and synthetic behavior.

## Background

Diary 0076 added single-role real-weight support: `--repack-dir DIR --weight-role ROLE` loads one fp16 rank-2 tensor, validates it, and dispatches the persistent skeleton against it. The natural next step is to validate multiple weight roles in one run, confirming that the persistent compute path handles distinct real weight matrices correctly and that the validation, dispatch, and checksum machinery generalizes.

## Implementation

### Repeatable `--weight-role` CLI option

`--weight-role ROLE` is now repeatable. Each occurrence appends to an internal role list:

```
--weight-role layer.0.mlp_gate --weight-role layer.0.mlp_up
```

When zero roles are specified, the synthetic path runs unchanged. When one role is specified, the output is identical to the diary 0076 single-role format. When two or more roles are specified, multi-role mode activates.

### Multi-role validation

When multiple roles are active:

1. The WeightArtifact is loaded once from `--repack-dir`.
2. Each role is validated independently: dtype fp16, rank 2, workgroups <= rows.
3. If `--hidden` is not explicitly set, hidden is inferred from the first role's column count. All subsequent roles must have `cols >= hidden`.
4. Validation errors produce clear JSON messages and exit 2.

### Per-role dispatch loop

The multi-role path:

1. Creates one VulkanDevice, one pipeline, one descriptor set layout.
2. Allocates reusable buffers (control, trace, scratch, input vector, weight matrix).
3. For each role: uploads the role's weight data, resets control/trace/scratch, dispatches, reads back results, validates checksum and trace.
4. The input vector is shared across roles (same deterministic synthetic data). The weight matrix buffer is overwritten per role.

This reuses the same shader, pipeline, and descriptor set across roles — only the weight buffer contents and CPU-side expected checksums change per role.

### Per-role JSON output

Multi-role JSON output includes:

- Top-level fields: `tokens`, `layers`, `hidden`, `workgroups`, `iterations`, `real_weight`, `repack_dir`, `multi_role`, `role_count`, `status`.
- A `roles` array with per-role objects containing: `role`, `rows`, `cols`, `checksum`, `expected_checksum`, `trace_mismatches`, `failures`, `status`.
- Top-level `status` is "ok" only if all roles pass.

Single-role output is unchanged from diary 0076. Synthetic output is unchanged.

### Backward compatibility

- `--weight-role` used once produces the exact same JSON shape as diary 0076.
- `--weight-role` omitted produces the exact same synthetic JSON.
- The help text now says `--weight-role ROLE tensor role to load (repeatable, e.g. layer.0.mlp_gate)`.

### Code refactoring

The main function was refactored to extract:

- `json_error()`: helper for emitting JSON error objects with exit code 2.
- `compute_expected()`: extracts the checksum/trace/generation computation into a reusable function, called once for single-role/synthetic and once per role in multi-role mode.
- `RoleWeightData` and `RoleResult` structs for clean per-role data flow.

The GPU dispatch loop was split into two code paths (single-role vs multi-role) to avoid entangling the multi-role logic with the existing repeat-results output.

### CTest

A new CTest gate `spock_persistent_decode_skeleton_multi_role_smoke` exercises two-role real-weight loading:

- Artifact: `artifacts/spock-text-repack-qwen35-0p8b`
- Roles: `layer.0.mlp_gate` and `layer.0.mlp_up`
- tokens=1, layers=1, hidden=128, workgroups=4

## Verification

### Focused CTest suite (5/5 passed)

```
ctest --test-dir build -R "spock_persistent_decode_skeleton|spock_diary_check" --output-on-failure
```

Tests: `spock_persistent_decode_skeleton_help`, `spock_persistent_decode_skeleton_smoke`, `spock_persistent_decode_skeleton_real_weight_smoke`, `spock_persistent_decode_skeleton_multi_role_smoke`, and `spock_diary_check`.

The focused synthetic, single-role real-weight, multi-role real-weight, and diary gates passed after the refactor.

### Direct multi-role run

```
build/vk_persistent_decode_skeleton \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --weight-role layer.0.mlp_gate \
  --weight-role layer.0.mlp_up \
  --tokens 1 --layers 1 --hidden 128 --workgroups 4
```

Result: status ok, both roles pass with exact checksum agreement and trace_mismatches=0. The mlp_gate checksum (4086921960) matches the diary 0076 single-role result. The mlp_up checksum (4047383920) is a distinct value from a different weight matrix, also matching its expected value exactly.

### Diary check

```
ctest --test-dir build -R "spock_diary_check" --output-on-failure
```

Passed with 77 entries valid after adding this diary entry.

## What This Is

- **First multi-weight validation inside the persistent skeleton.** Multiple real fp16 weight roles flow through the persistent fp16/fp32 compute + barrier path in one invocation.
- **Validated artifact generality.** The weight loading, validation, and checksum machinery works correctly across different weight matrices from the same artifact.
- **Backward-compatible.** Single-role and synthetic modes produce identical output to diary 0076.

## What This Is Not

- **Not inference.** No attention, no DeltaNet, no KV cache, no LM head, no token generation.
- **Not the megakernel.** No persistent dispatch of actual inference work.
- **Not production spock-decode.** This is a standalone probe.
- **Not layer semantics.** The probe treats each weight as a flat matrix; it does not interpret roles in the model architecture.
- **Not a performance benchmark.** Roles dispatch sequentially, reusing GPU buffers. No attempt to fuse multi-role work.

## Known Limitations

- **Sequential dispatch.** Each role dispatches separately. This is correct but not optimized for throughput.
- **Prefix rows/cols only.** Each role still uses `workgroups` rows and `hidden` columns from potentially larger weight matrices.
- **Synthetic input.** The input vector is the same deterministic data for all roles.
- **No timestamps in multi-role mode.** Multi-role mode now rejects `--timestamps` with an explicit error (exit 2) instead of silently ignoring it.
- **No repeats in multi-role mode.** Multi-role mode now rejects `--repeats > 1` with an explicit error (exit 2) instead of silently ignoring it.
- **Shared hidden dimension.** All roles must have at least `hidden` columns. Roles with different row counts are supported as long as `workgroups <= rows`.

## Next Work

1. Add per-role GPU timestamps to the multi-role path.
2. Extend to more weight roles (e.g., attention QKV, DeltaNet projections) and larger row/column slices.
3. Explore batched multi-role dispatch where feasible.
4. Progress toward persistent decode megakernel with real multi-weight layer compute.

# 0076: Persistent Decode Skeleton — Real Repacked fp16 Model-Weight Support

## Goal

Extend `vk_persistent_decode_skeleton` to load and verify real repacked fp16 model weights from the spock-text-repack artifact directory, making this the first time a real model weight flows through the persistent skeleton's fp16/fp32 compute path. This is NOT inference, NOT the megakernel, and NOT production decode. It validates that the persistent fp16 dot-product + barrier coordination works correctly against actual model data before investing in full layer compute.

## Background

Diary 0075 introduced the persistent decode skeleton probe with deterministic synthetic fp16 data. All dot products, shared-memory reductions, cross-reads, and barrier coordination were verified at Qwen3.5 preset geometry (82 workgroups, 128 tokens x 24 layers, hidden=1024). The probe used synthetic input and weight values designed to stay in a safe fp16 range.

The natural next step is to replace synthetic weights with real fp16 model weights from the existing repacked artifact format, confirming that the persistent compute path handles actual model data correctly — including subnormal fp16 values and the full range of real weight magnitudes.

## Implementation

### Real-weight loading via `--repack-dir` and `--weight-role`

The app now accepts two new CLI options used together:

- `--repack-dir DIR` — path to a repacked model artifact directory (e.g., `artifacts/spock-text-repack-qwen35-0p8b`).
- `--weight-role ROLE` — weight role name to load (e.g., `layer.0.mlp_gate`).

When both are supplied, the app loads the corresponding `WeightArtifact` from the repacked manifest and validates:

1. **dtype is fp16** — real weights must be float16.
2. **rank is 2** — the weight must be a matrix `[rows, cols]`.
3. **Shape constraints** — the shader uses `workgroups` rows and `hidden` columns from the weight matrix. If `--hidden` is not explicitly supplied, hidden is inferred from the weight's column count.
4. **Bounds checks** — errors if `hidden > cols` or `workgroups > rows`.

The weight buffer replaces the synthetic weight matrix. The input vector remains synthetic (deterministic fp16 values cycling 1..8).

### JSON output extensions

When real weights are active, the JSON output includes:

- `real_weight: true`
- `weight_role`
- `real_weight_rows`
- `real_weight_cols`
- `repack_dir`

When real weights are not active (no `--repack-dir` / `--weight-role`), `real_weight` is `false` and the synthetic path runs unchanged.

### CPU expected-checksum fix: matching shader reduction order

The initial real-weight test showed a checksum mismatch. The root cause was that the CPU reference computation did not mirror the shader's exact reduction order. The shader performs:

1. 64 lane-strided fp32 partial sums (each lane sums elements at stride 64).
2. Tree reduction in shared memory: stride 32, 16, 8, 4, 2, 1.

The CPU reference was re-implemented to follow this exact order. With synthetic data this produces the same result as a naive sum (values are small and exactly representable), but real weights have a wider magnitude range where reduction order matters for fp32 accumulation.

### Subnormal preservation fix

A second mismatch was traced to the host-side fp16 decoder. The existing `half_to_float` conversion flushed fp16 subnormals to zero. Real model weights contain subnormal fp16 values, and flushing them changed the dot-product result. The fix: the host fp16 decoder now preserves subnormals when computing expected checksums for real-weight mode (converting the subnormal fp16 to its correct fp32 value). This fixed the initial mismatch between GPU and CPU checksums for the `layer.0.mlp_gate` weight at hidden=128 and hidden=1024.

### CTest

A new CTest gate `spock_persistent_decode_skeleton_real_weight_smoke` exercises real-weight loading:

- Artifact: `artifacts/spock-text-repack-qwen35-0p8b`
- Role: `layer.0.mlp_gate`
- tokens=1, layers=1, hidden=128, workgroups=4

This validates that real weight loading, fp16 subnormal handling, and checksum verification all work in the CTest environment.

## Verification

### CTest suite

```
ctest --test-dir build -R "spock_persistent_decode_skeleton|spock_diary_check" --output-on-failure
```

4/4 tests passed: `spock_persistent_decode_skeleton_help`, `spock_persistent_decode_skeleton_smoke`, `spock_persistent_decode_skeleton_real_weight_smoke`, and `spock_diary_check`.

### Direct run: explicit hidden=128

```
build/vk_persistent_decode_skeleton \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --weight-role layer.0.mlp_gate \
  --tokens 1 --layers 1 --hidden 128 --workgroups 4 --timestamps
```

Result: status ok, hidden 128, workgroups 4, real_weight true, rows 3584, cols 1024, generation 2, expected_generation 2, checksum 4086921960, expected_checksum 4086921960, trace_mismatches 0, gpu_dispatch_us 17.08, per_barrier_us 8.54.

### Direct run: inferred hidden=1024

```
build/vk_persistent_decode_skeleton \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --weight-role layer.0.mlp_gate \
  --tokens 1 --layers 1 --workgroups 4 --timestamps
```

Result: status ok, hidden inferred to 1024, workgroups 4, real_weight true, rows 3584, cols 1024, generation 2, expected_generation 2, checksum 4142172704, expected_checksum 4142172704, trace_mismatches 0, gpu_dispatch_us 29.76, per_barrier_us 14.88.

The two checksums differ (4086921960 vs 4142172704) because different column counts produce different dot products over different slices of the same real weight matrix. Both match their expected values exactly.

## What This Is

- **First real repacked fp16 model-weight use inside the persistent skeleton.** Real weights from the Qwen3.5-0.8B repacked artifact now flow through the persistent fp16/fp32 compute + barrier path.
- **Validated subnormal handling.** The host fp16 decoder correctly preserves subnormal values when computing expected checksums against real model data.

## What This Is Not

- **Not inference.** No attention, no DeltaNet, no KV cache, no LM head, no token generation.
- **Not the megakernel.** No persistent dispatch of actual inference work.
- **Not production spock-decode.** This is a standalone probe.
- **Not a performance benchmark.** The probe still uses only prefix rows/cols of the weight matrix — `workgroups` rows and `hidden` columns from a potentially much larger matrix. It does not exercise the full matvec that inference would require.
- **Not layer semantics.** The probe treats the weight as a flat matrix; it does not interpret its role in the model architecture.

## Known Limitations

- **Prefix rows/cols only.** The shader dot-product covers `workgroups` rows and `hidden` columns, not the full weight shape. At hidden=128, only 128 of 1024 columns are used. At hidden=1024, all columns are used but only 4 of 3584 rows.
- **No layer semantics.** The weight role name is loaded but not interpreted. The probe does not know whether the weight is a gate, up, down, Q, K, V, or output projection.
- **Synthetic input.** The input vector remains deterministic synthetic data (cycling fp16 values 1..8), not real activations from model inference.
- **Subnormal preservation is host-side only.** The GPU shader's fp16 arithmetic handles subnormals per hardware behavior; the fix ensures the CPU reference matches. No guarantee of subnormal-preserving arithmetic on all GPU vendors, but this probe targets the local RADV stack.

## Next Work

1. Extend the real-weight path to exercise more weight roles (e.g., `layer.0.mlp_up`, `layer.0.attn_qkv`) and larger row/column slices.
2. Add a multi-weight-role mode that loads and verifies several weights in one run, exercising more of the weight artifact format.
3. Progress toward a persistent decode megakernel that uses real weights for actual layer compute (RMSNorm, projections, recurrent state updates, attention, MLP).

# 0078: Persistent Decode Skeleton — Row-Strided Weight Coverage

## Goal

Extend `vk_persistent_decode_skeleton` with `--row-count N` so that a bounded set of resident workgroups can cover more matrix rows than workgroups via row-strided assignment (`row = group; row < row_count; row += workgroups`). This is the first step toward a persistent megakernel where a small resident pool covers a large weight matrix.

## Background

Diary 0077 validated multi-role real-weight dispatch, but each workgroup still processes exactly one row. A weight matrix with hundreds of rows would require hundreds of resident workgroups, which is infeasible on real GPUs. Row-striding lets each workgroup hop across rows in lockstep, covering `row_count` rows with only `workgroups` workgroups.

## Implementation

### New CLI option: `--row-count N`

`--row-count N` sets the number of matrix rows the persistent shader covers. When omitted, `row_count` defaults to `workgroups`, preserving the old one-row-per-workgroup behavior and all existing checksums.

The shader now computes its initial row assignment as `row = group` and advances by `workgroups` inside each persistent iteration:

```
for (row = group; row < row_count; row += workgroups) {
    // fp16 dot product, reduction, barrier
}
```

### Default backward compatibility

When `row_count == workgroups`, each workgroup processes exactly one row — the same behavior as diaries 0075–0077. The synthetic default checksum (2229282944 for tokens=2, layers=4, hidden=128, workgroups=8) is unchanged. Single-role default runs produce checksum 4086921960; multi-role default gate/up produce 4086921960 and 4047383920 — all unchanged.

### CPU reference update

The host-side expected-checksum computation mirrors the shader's row coverage and per-row dot reduction. When `row_count > workgroups`, workgroups process multiple rows each, and the CPU reference includes every covered row in the same checksum model.

## Verification

### CTest gate

A new test `spock_persistent_decode_skeleton_row_count_real_weight_smoke` exercises row-strided real-weight coverage:

- Artifact: `artifacts/spock-text-repack-qwen35-0p8b`
- Role: `layer.0.mlp_gate`
- tokens=1, layers=1, hidden=128, workgroups=4, row-count=16

### Direct run

```
build/vk_persistent_decode_skeleton \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --weight-role layer.0.mlp_gate \
  --tokens 1 --layers 1 --hidden 128 --workgroups 4 --row-count 16
```

Result: status ok, row_count 16, checksum 3002794576, expected_checksum 3002794576, trace_mismatches 0.

### Direct multi-role row-count run

```
build/vk_persistent_decode_skeleton \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --weight-role layer.0.mlp_gate \
  --weight-role layer.0.mlp_up \
  --tokens 1 --layers 1 --hidden 128 --workgroups 4 --row-count 16
```

Result: status ok. `layer.0.mlp_gate` checksum 3002794576 matched expected_checksum 3002794576; `layer.0.mlp_up` checksum 2880359528 matched expected_checksum 2880359528; both roles had trace_mismatches 0.

### Direct timestamped row-count run

```
build/vk_persistent_decode_skeleton \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --weight-role layer.0.mlp_gate \
  --tokens 1 --layers 1 --hidden 128 --workgroups 4 --row-count 16 --timestamps
```

Result: status ok, checksum 3002794576, expected_checksum 3002794576, trace_mismatches 0, timestamp_valid true, gpu_dispatch_us 26.2, per_barrier_us 13.1, barriers 2.

### Existing tests unchanged

```
ctest --test-dir build -R "spock_persistent_decode_skeleton|spock_diary_check" --output-on-failure
```

All prior tests pass with unchanged checksums.

## What This Is

- **Row-strided projection coverage.** A bounded pool of resident workgroups covers more matrix rows than there are workgroups.
- **Backward-compatible.** Default `row_count == workgroups` preserves all prior checksums and behavior.
- **First step toward megakernel weight coverage.** Validates the row-striding pattern before investing in full persistent decode.

## What This Is Not

- **Not inference.** No attention, no DeltaNet, no KV cache, no LM head, no token generation.
- **Not layer semantics.** The probe treats the weight as a flat matrix with no architectural interpretation.
- **Not the megakernel.** No persistent dispatch of actual inference work.
- **Not a performance benchmark.** Row-striding adds more row work inside each persistent iteration, but this probe is correctness-only.

## Known Limitations

- **Single-role CTest only.** The committed row-count CTest covers one real role. A direct multi-role row-count run passed, but multi-role row-count should get its own CTest before relying on it as a regression gate.
- **Prefix columns only.** Each row still uses `hidden` columns from potentially wider weight matrices.
- **Synthetic input.** The input vector is deterministic, not from an actual model.
- **No row-count timing characterization yet.** A timestamped single-role row-count run passed, but `per_barrier_us` is not a row-strided throughput metric.

## Next Work

1. Validate row-striding with multi-role dispatch.
2. Extend row-count coverage to larger weight slices (e.g., full hidden=1024, all rows).
3. Adapt GPU timestamps to the row-strided barrier pattern.
4. Progress toward persistent decode megakernel with row-strided weight coverage.

# 0079: Persistent Decode Skeleton -- Model-Width Row-Strided Coverage

## Goal

Exercise the row-strided persistent skeleton at model-width hidden size with a decode-relevant resident workgroup count. Diary 0078 proved `--row-count` on a small hidden=128 slice. This entry checks hidden=1024, workgroups=82, row_count=128 against real repacked MLP weights.

## Background

The persistent megakernel target cannot rely on one resident workgroup per output row. Real MLP matrices such as `layer.0.mlp_gate` and `layer.0.mlp_up` have shape `[3584,1024]`. Row-striding lets a bounded resident pool cover more rows than there are resident workgroups while preserving the software-global-barrier pattern.

The important detail is that this test stresses the same two constraints that matter for the eventual megakernel. First, `hidden=1024` means every row dot product traverses the real model width rather than the smaller smoke-test prefix used in diary 0078. Second, `workgroups=82` keeps the resident workgroup count at the Qwen3.5 decode-shape geometry already used by the barrier and persistent skeleton probes. The shader then covers `row_count=128`, so some workgroups process a second matrix row while others process one row. That validates the uneven row distribution case instead of only the trivial `row_count == workgroups` case.

This still deliberately stops short of all 3584 MLP rows. The goal is not to benchmark full projection throughput yet; it is to prove that the resident-workgroup row-striding pattern remains bit-exact when the dot product uses the actual hidden width and real fp16 values. Full-row coverage should come after this path has stable correctness gates and after we decide how to report timing in terms of rows, columns, roles, and barriers.

## Verification

### Single-role model-width row-count run

```
build/vk_persistent_decode_skeleton \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --weight-role layer.0.mlp_gate \
  --tokens 1 --layers 1 --hidden 1024 --workgroups 82 --row-count 128 --timestamps
```

Result: status ok, row_count 128, checksum 2755310530, expected_checksum 2755310530, trace_mismatches 0, timestamp_valid true, gpu_dispatch_us 90.96, per_barrier_us 45.48, barriers 2.

### Multi-role model-width row-count run

```
build/vk_persistent_decode_skeleton \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --weight-role layer.0.mlp_gate \
  --weight-role layer.0.mlp_up \
  --tokens 1 --layers 1 --hidden 1024 --workgroups 82 --row-count 128
```

Result: status ok. `layer.0.mlp_gate` checksum 2755310530 matched expected_checksum 2755310530; `layer.0.mlp_up` checksum 3034398318 matched expected_checksum 3034398318; both roles had trace_mismatches 0.

## CTest

A new CTest gate `spock_persistent_decode_skeleton_row_count_model_width_smoke` covers:

- Artifact: `artifacts/spock-text-repack-qwen35-0p8b`
- Role: `layer.0.mlp_gate`
- tokens=1, layers=1, hidden=1024, workgroups=82, row-count=128

This makes model-width row-strided real-weight coverage part of the focused persistent skeleton regression set.

The CTest intentionally gates the single-role `layer.0.mlp_gate` path rather than the full multi-role run. The single-role gate is enough to protect the shader, host row extraction, CPU checksum model, and row-count JSON/output wiring at model width. The multi-role command remains a direct verification sample for now; it should become a CTest once the next layer-shaped probe decides whether multi-role row-count should execute as independent role dispatches or as a more fused MLP-oriented path.

## Interpretation

The result is a meaningful step beyond "can load several weights." The persistent skeleton now demonstrates that a fixed resident workgroup pool can cover more projection rows than the number of workgroups launched, while still using real fp16 model data and exact CPU/GPU checksum agreement. That is the structural pattern the megakernel will need for large projections: resident workgroups cannot scale linearly with matrix rows, so rows must be assigned inside the persistent work loop.

The timestamped single-role sample also gives a rough sanity check that the row-strided work is actually executing in the shader. The model-width row-count run reported `gpu_dispatch_us` around 90 us for one persistent iteration with two software-global-barrier generations. This number should not be compared directly to decode throughput, because the probe does not materialize output vectors, does not run all rows, and does not execute surrounding layer operations. It is useful only as a local timing sample attached to a correctness gate.

The checksum values also tell us the rows are not being accidentally collapsed to the old prefix behavior. The default single-role checksum from diary 0078 was 4086921960 for four rows at hidden=128, while the model-width row-count checksum is 2755310530 for 128 rows at hidden=1024. The multi-role run produced a distinct `mlp_up` checksum, 3034398318, with exact expected agreement. Different roles and row counts are therefore flowing through the row-strided path rather than reusing stale buffers.

## What This Is

- **Model-width real-weight coverage.** The shader covers 128 rows at the actual 1024-column hidden width for a real MLP projection.
- **Decode-relevant resident geometry.** The run uses 82 workgroups, matching the current Qwen3.5 decode-shape probe geometry.
- **A stronger row-striding gate.** The new CTest protects the row-strided path beyond the small hidden=128 smoke test.

## What This Is Not

- **Not inference.** There is still no real activation vector from a model step, no MLP activation, no down projection, no residual path, and no token generation.
- **Not full projection coverage.** `row_count=128` covers more rows than resident workgroups, but not all 3584 rows.
- **Not the megakernel.** This is still a standalone probe, not production decode.
- **Not a throughput benchmark.** The timestamped run is a correctness-timed sample only; it is not normalized to row work or compared against the current decode path.

## Next Work

1. Add a real MLP micro-probe: gate/up row-strided projections, SiLU/gated multiply, then down projection.
2. Increase row_count toward full MLP row coverage once the checksum and timing model are clear.
3. Connect row-strided projection coverage to layer-shaped persistent decode work.

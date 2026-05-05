# 0070: Chunked Decode GPU-Timestamped Controlled Sweep

## Goal

Run a controlled GPU-timestamped sweep across chunk sizes 1, 2, 4, 8, 16 using the `--gpu-timestamps` extension from diary 0069. This produces the first multi-size GPU-side timing comparison for chunked decode, separating device execution time from host overhead visible in the diary 0066 host-only sweep.

## Background

Diary 0066 and the refreshed diary 0067 run established the host-side sweep
pattern: `decode_ms` moved from about 353.1 ms at chunk size 1 to about
350.5 ms at chunk size 16, with size 8 slightly best in that short sample.
However, host-side timing conflates GPU execution, driver submit overhead, and
host scheduling.

Diary 0068 added GPU timestamp recording inside chunked decode command buffers. Diary 0069 extended the sweep tool with `--gpu-timestamps` to capture `gpu_decode_us` and `per_token_gpu_us` from `spock-decode` JSON output. A short single-size run at chunk size 16 confirmed the pipeline worked.

This diary records the results of a full multi-size sweep with GPU timestamps enabled.

## Command

```
python3 tools/run_chunked_decode_sweep.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --ids short_correctness_001 \
  --max-new-tokens 16 \
  --chunk-sizes 1,2,4,8,16 \
  --warmup-runs 1 \
  --timed-runs 3 \
  --gpu-timestamps
```

Results written to `/tmp/spock_chunked_gpu_timestamp_sweep_0069.json`.

## Results

### Correctness

All aggregate records report `match=true`. All per-run records match. Reference parity holds at every chunk size.

### Submit counts

| Chunk size | decode_submit_count | chunked_decode_submit_count |
|------------|--------------------|-----------------------------|
| 1          | 16                 | 16                          |
| 2          | 8                  | 8                           |
| 4          | 4                  | 4                           |
| 8          | 2                  | 2                           |
| 16         | 1                  | 1                           |

Submit counts match the post-0067 structural model `ceil(N/C)` exactly, with `decode_submit_count == chunked_decode_submit_count` at all sizes.

### Host-side decode timing

| Chunk size | decode_ms_mean (3 runs) |
|------------|------------------------|
| 1          | 353.060                |
| 2          | 351.659                |
| 4          | 350.803                |
| 8          | 350.153                |
| 16         | 349.475                |

Host-side decode time decreases modestly and monotonically with chunk size, consistent with the diary 0066 finding. The total reduction from size 1 to size 16 is about 3.6 ms (~1.0%).

### GPU-side decode timing

| Chunk size | gpu_decode_us_mean | per_token_gpu_us_mean_mean | gpu_decode_us min | gpu_decode_us max |
|------------|-------------------|---------------------------|-------------------|-------------------|
| 1          | 348295.7          | 21768.49                  | 348120            | 348475            |
| 2          | 348124.3          | 21757.77                  | 347912            | 348322            |
| 4          | 347836.3          | 21739.79                  | 347735            | 347915            |
| 8          | 347688.3          | 21730.53                  | 347522            | 347772            |
| 16         | 347333.0          | 21708.31                  | 347205            | 347471            |

GPU decode time is nearly flat across chunk sizes. The total reduction from size 1 to size 16 is about 963 us (~0.28%). Per-token GPU time decreases from about 21.77 ms to about 21.71 ms, a ~0.3% reduction. The min/max spread within each chunk size is under 0.1%, indicating stable measurement.

### Interpretation

GPU time is nearly flat across chunk sizes while host-side decode_ms shows a modest monotonic improvement with submit-count reduction. This suggests that for this short run (16 tokens, one prompt), the decode is dominated by actual GPU work rather than submit overhead. The host-side improvement visible in `decode_ms` is primarily explained by reduced host-side wait/schedule overhead per chunk boundary, not by faster GPU execution.

In other words: at this geometry, the GPU does the same work regardless of how many command buffers the host submits. The host saves a small amount of scheduling overhead by batching, but the savings are under 1% of total decode time.

### Comparison with diary 0066

Diary 0066 saw host-side decode_ms means: size1 353.09, size2 351.98, size4 351.24, size8 350.14, size16 350.49. Size8 was best and size16 slightly regressed. This sweep shows monotonic improvement through size16 (349.475). The difference is within run-to-run variance and does not indicate a structural change. The GPU timing confirms the underlying device work is essentially constant.

## Verification

### Command invocation

```
python3 tools/run_chunked_decode_sweep.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --ids short_correctness_001 \
  --max-new-tokens 16 \
  --chunk-sizes 1,2,4,8,16 \
  --warmup-runs 1 \
  --timed-runs 3 \
  --gpu-timestamps
```

Completed successfully. All 15 timed runs (3 per chunk size) reported `match=true`.
All aggregate records report `match=true`. GPU timestamp validation passed for all
runs (gpu_decode_us present and positive, per_token_gpu_us length matches
generated_count). Results written to
`/tmp/spock_chunked_gpu_timestamp_sweep_0069.json`.

### Consistency checks

- Submit counts match the `ceil(16/C)` model exactly at all sizes.
- gpu_decode_us values are consistent with the single-size diary 0069 run:
  347708 us at size 16 is close to this sweep's size-16 range of
  347205-347471 us, but not inside that three-run min/max window.
- Per-token GPU timing is within the range seen in diary 0068's single-run sample.
- Host-side decode_ms values are consistent with diary 0066's sweep at the same geometry.

### Diary and doc consistency

```
ctest --test-dir build -R "spock_diary_check" --output-on-failure
```

```
git diff --check
```

Both pass after this change.

## What This Does Not Show

- **Not a throughput benchmark.** This is one prompt, 16 tokens, 3 timed runs. It is measurement evidence, not a performance claim.
- **Not persistent dispatch.** Chunked decode is host-orchestrated command-buffer batching. The host still submits per chunk and waits.
- **Not the megakernel.** No persistent workgroups, no cross-workgroup synchronization, no in-kernel token loop.
- **Not a long-run or high-token characterization.** 16 tokens may not capture effects visible at 128 or 512 tokens.
- **Not multi-prompt.** Only `short_correctness_001` was tested.
- **Not block-level timing.** Block-level GPU timestamps remain excluded from the chunked path.

## Known Limitations

- **Single prompt, short run.** Statistical power is limited. The ~0.28% GPU timing spread could be noise.
- **No host overhead decomposition.** The difference between `decode_ms` and `gpu_decode_us/1000` includes host submit, wait, and scheduling overhead, but we did not instrument each component separately.
- **GPU timestamp resolution.** Timestamp period is device-dependent. The near-flat GPU timing could mask small per-submit overhead that is below timestamp granularity.
- **Warmup may be insufficient.** One warmup run may not fully stabilize GPU clocks and thermal state.

## Next Work

1. Run at higher token counts (e.g., `--max-new-tokens 128`) to see whether submit overhead becomes more significant with longer decode sequences.
2. Add a second prompt (`mixed_correctness_023`) to check whether the flat-GPU-timing pattern holds across different prefill geometries.
3. Consider block-level GPU timestamps inside chunked command buffers to identify which layer/block dominates the flat GPU time.
4. Evaluate whether the host-side improvement justifies larger chunk sizes for production, or whether chunk size 1 is already near the GPU-determined floor.

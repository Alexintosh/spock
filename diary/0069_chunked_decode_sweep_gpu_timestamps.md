# 0069: Chunked Decode Sweep GPU Timestamp Extension

## Goal

Extend `tools/run_chunked_decode_sweep.py` with an opt-in `--gpu-timestamps` flag that enables `SPOCK_GPU_TIMESTAMPS=1` during sweep runs and records GPU timing fields from `spock-decode` JSON output in per-run and aggregate records. This gives the sweep tool a first GPU-side measurement dimension alongside the existing host-side timing, without changing default behavior.

## Background

Diary 0068 added GPU timestamp support inside chunked decode command buffers. When `SPOCK_GPU_TIMESTAMPS=1` and `SPOCK_GPU_CHUNKED_DECODE=1` are both active, `spock-decode` JSON output includes `gpu_decode_us` (total GPU decode time) and `per_token_gpu_us` (per-token array). A manual run at chunk size 16 produced `gpu_decode_us=347611` with 16 per-token values.

The sweep tool (diary 0064, extended in 0065 with warmup/repeat support) currently captures host-side timing only: `elapsed_ms`, `prefill_ms`, `decode_ms`. Per diary 0066, host-side timing showed a modest monotonic decrease with chunk size, but this conflates GPU execution, driver submit overhead, and host scheduling. The GPU timestamp fields allow separating device work duration from host overhead in sweep output.

Diary 0068's "Next Work" section explicitly called out running a GPU-timestamped sweep across chunk sizes. This change wires the infrastructure for that: the sweep tool itself does not run any sweeps, but it can now be invoked with `--gpu-timestamps` to collect and validate GPU timing alongside host timing.

## Implementation Work Completed

### `--gpu-timestamps` CLI flag

The sweep tool gains `--gpu-timestamps`, a `store_true` argument defaulting to `False`. When set, `build_env` includes `SPOCK_GPU_TIMESTAMPS=1` in the decode environment alongside the existing fast-path and chunked gates.

### Per-run GPU timing records

When `--gpu-timestamps` is active, `_make_run_record` adds:

- `gpu_decode_us`: total GPU decode command buffer execution time from `spock-decode` JSON
- `per_token_gpu_us_count`: count of per-token GPU timing values
- `per_token_gpu_us_mean`: mean of per-token values
- `per_token_gpu_us_min`: minimum per-token value
- `per_token_gpu_us_max`: maximum per-token value

The raw `per_token_gpu_us` array is summarized rather than emitted in full to keep output manageable at high token counts. The count/mean/min/max summary is sufficient to verify timestamp data was recorded and to compare distributions across chunk sizes.

### Aggregate GPU timing records

`_make_aggregate_record` adds mean/min/max for both `gpu_decode_us` and `per_token_gpu_us_mean` across timed runs when `--gpu-timestamps` is active, following the same pattern as the existing `elapsed_ms`, `prefill_ms`, and `decode_ms` aggregation.

### GPU timestamp validation

When `--gpu-timestamps` is active, each timed run is validated:

1. `gpu_decode_us` must be present and > 0. Missing or non-positive values indicate timestamps were not recorded (gate misconfiguration or driver issue).
2. `per_token_gpu_us` must be a list. Missing or non-list values indicate the output structure is unexpected.
3. `per_token_gpu_us` length must match `generated_count` (falling back to `max_new_tokens` if `generated_count` is absent). A length mismatch indicates incomplete timestamp recording.

Each validation failure produces an error record and marks the run as failed (consistent with existing token-mismatch and decode-failure handling). The sweep exits nonzero on any validation failure.

### Summary metadata

The top-level JSON summary includes `"gpu_timestamps": true/false` and includes `SPOCK_GPU_TIMESTAMPS` in `env_gates` when active.

### Default behavior preserved

Without `--gpu-timestamps`, the sweep tool behaves identically to before: no `SPOCK_GPU_TIMESTAMPS` env var, no GPU timing fields in records, no GPU timing validation. The flag is purely additive.

## Verification

### Syntax and unit checks

```
python3 -m py_compile tools/run_chunked_decode_sweep.py tests/run_sweep_gpu_timestamp_unit.py
python3 tests/run_sweep_gpu_timestamp_unit.py
```

Passed.

The standalone unit script checks conditional timestamp env setup, per-run GPU
summary fields, aggregate GPU timing fields, and timestamp validation fallback
from missing `generated_count` to `max_new_tokens`. It is wired into CTest as
`spock_chunked_sweep_gpu_timestamp_unit`.

```
ctest --test-dir build -R "spock_chunked_sweep_gpu_timestamp_unit|spock_diary_check" --output-on-failure
```

Passed.

### Short GPU timestamp sweep

```
python3 tools/run_chunked_decode_sweep.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --ids short_correctness_001 \
  --max-new-tokens 16 \
  --chunk-sizes 16 \
  --warmup-runs 0 \
  --timed-runs 1 \
  --gpu-timestamps
```

Passed with `match=true`, `decode_submit_count=1`,
`chunked_decode_submit_count=1`, `gpu_decode_us=347708`,
`per_token_gpu_us_count=16`, `per_token_gpu_us_mean=21731.765`,
`per_token_gpu_us_min=5987.24`, and `per_token_gpu_us_max=23739.3`.

## What This Does Not Show

- **No broad sweep results.** This change adds infrastructure and verifies one
  short GPU-timestamped sweep at chunk size 16. A controlled multi-size sweep
  across chunk sizes 1, 2, 4, 8, 16 remains separate work.
- **No performance claims.** GPU timing fields are measurement instruments, not optimizations.
- **No persistent dispatch.** The sweep runs the chunked decode path, which is host-orchestrated command-buffer batching.
- **No megakernel progress.** This is tooling, not an architectural change.

## Known Limitations

- **Per-token summary only.** The raw `per_token_gpu_us` array is summarized into count/mean/min/max rather than emitted in full. If full per-token data is needed, the tool would need a separate verbose mode.
- **Validation depends on generated_count.** If `spock-decode` does not emit `generated_count`, validation falls back to `max_new_tokens`. This is correct for the current tool usage where all prompts produce exactly `max_new_tokens` tokens, but could be wrong if early stopping is introduced.
- **Hard-coded gate stack.** Same limitation as diaries 0064–0068. The GPU timestamp gate composes with the existing fast-path env configuration but is not independently configurable.
- **No block-level timestamps.** Block-level GPU timing (`SPOCK_GPU_BLOCK_TIMESTAMPS=1`) is still excluded from the chunked path and not wired into the sweep tool.

## Next Work

1. Run a GPU-timestamped sweep across chunk sizes 1, 2, 4, 8, 16 at higher token counts using `--gpu-timestamps` to characterize GPU-side per-token time vs. chunk size.
2. Compare GPU-side timing against host-side timing from diary 0066 to separate device execution from host overhead.
3. Consider adding `--gpu-block-timestamps` if block-level timestamp support is extended to the chunked path.
4. Evaluate whether per-token GPU timing data should feed into automatic chunk size recommendations.

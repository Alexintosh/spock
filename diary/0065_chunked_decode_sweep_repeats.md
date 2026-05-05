# 0065: Chunked Decode Sweep Repeat/Warmup Extension

## Goal

Extend `tools/run_chunked_decode_sweep.py` with repeat-run and warmup support
so that controlled host-side timing across multiple timed runs per
configuration is possible without manual scripting. This gives the sweep tool
the structure needed to characterize submission-overhead amortization across
chunk sizes with warmup, while preserving the existing single-run behavior as
the default.

This is a measurement-tool extension. It does not modify runtime, shader, or
test code.

## Background

Diary 0064 introduced `tools/run_chunked_decode_sweep.py` as a convenience
tool for sweeping chunk sizes across reference prompt IDs. It ran each
configuration exactly once, emitting per-run JSON records with host-side timing
and submit counts. The entry explicitly called out single-run timing as a
limitation and listed "controlled repeated timing across chunk sizes with
warmup" as the first item of next work.

The need for warmup runs is straightforward: the first invocation of a Vulkan
pipeline on this RADV stack incurs one-time compilation and caching overhead
that is not representative of steady-state decode timing. Without discarding
warmup runs, any aggregate timing over repeated executions would conflate
startup cost with the per-run cost the project actually wants to measure.

Similarly, a single timed run cannot estimate variance. Two or more timed runs
make it possible to compute mean, minimum, and maximum for elapsed, prefill,
and decode wall-clock times. This does not produce GPU-side execution time —
that remains a separate future step requiring `SPOCK_GPU_TIMESTAMPS=1` support
in multi-step command buffers — but it does give the project a controlled
host-side measurement structure that was missing.

## Implementation Work Completed

### New CLI arguments

The tool now accepts two additional arguments:

- `--warmup-runs N` (default 0): number of warmup executions per id/chunk_size
  combination. Warmup runs invoke the same `spock-decode` command with the
  full fast-path gate stack, compare generated tokens against references, and
  exit nonzero on mismatch. Their timing is recorded for diagnostic purposes
  but excluded from aggregate statistics.

- `--timed-runs N` (default 1): number of timed executions per id/chunk_size
  combination. Each timed run must match the reference. Per-run records in the
  JSON output now include a `run_index` field (0-based) identifying which timed
  run produced that record.

### Execution order

For each (id, chunk_size) pair, the tool executes warmup runs first, then
timed runs. This ordering is deliberate: warmup runs exist to absorb one-time
pipeline compilation and caching costs so that timed runs measure steady-state
behavior. Running warmups after timed runs would defeat this purpose.

Both warmup and timed runs must match the stored reference. A mismatch in any
run — warmup or timed — causes the tool to exit nonzero. This preserves the
tool's correctness-gating behavior: a zero exit code means every run at every
configuration produced reference-matching output.

### Aggregate records

After all timed runs complete for a given (id, chunk_size) pair, the tool
emits an aggregate record with the following fields:

- `aggregate`: `true` (distinguishes aggregate records from per-run records)
- `match`: `true` only if all timed runs matched the reference
- `timed_runs`: count of timed runs in the aggregate
- `decode_submit_count`, `chunked_decode_submit_count`, `generated_count`:
  copied only when all timed runs report the same non-null value (these values
  should be deterministic for a given configuration)
- `elapsed_ms_mean`, `elapsed_ms_min`, `elapsed_ms_max`: mean, minimum, and
  maximum host-side elapsed wall-clock milliseconds across timed runs
- `prefill_ms_mean`, `prefill_ms_min`, `prefill_ms_max`: same for prefill
  phase
- `decode_ms_mean`, `decode_ms_min`, `decode_ms_max`: same for decode phase

The aggregate record makes it possible to compare timing distributions across
chunk sizes without post-processing multiple per-run records. The mean/min/max
triplet is deliberately chosen over standard deviation: for the small run
counts this tool targets (typically 2-5 timed runs), mean/min/max is a more
honest summary than a sample standard deviation that would carry an inflated
uncertainty estimate.

### Top-level JSON output

The top-level JSON object now includes `warmup_runs` and `timed_runs` fields
stating the configured counts. This makes the execution configuration
self-describing in the output, so a reader of the JSON knows how many runs
contributed to each aggregate without consulting the original command line.

### Backward compatibility

With the default arguments (`--warmup-runs 0 --timed-runs 1`), the tool
produces one per-run record and one aggregate record per (id, chunk_size)
pair. The aggregate record for a single timed run has identical mean, min, and
max values. This preserves the existing one-run usage pattern while adding the
aggregate record that was previously absent.

### Design decisions

Warmup records are excluded from the `results` array entirely and are not
included in aggregate statistics. The rationale is that warmup runs serve a
specific purpose — absorbing compilation overhead — and their timing is not
representative of steady-state behavior. Including them in aggregates would make
the aggregates harder to interpret.

The tool still does not enable `SPOCK_GPU_TIMESTAMPS=1`. The repeated-run
structure is host-side only. GPU-side timing for multi-step command buffers
requires separate runtime work to accumulate timestamp queries across chunked
decode steps, which remains future work.

## Verification

### Syntax check

```
python3 -m py_compile tools/run_chunked_decode_sweep.py
```

Passed. No import or syntax errors.

### Multi-run sweep with warmup

```
python3 tools/run_chunked_decode_sweep.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --ids short_correctness_001 \
  --max-new-tokens 16 \
  --chunk-sizes 8,16 \
  --warmup-runs 1 \
  --timed-runs 2
```

Passed. The tool executed one warmup run and two timed runs per chunk size.

- Chunk size 8 aggregate: `decode_submit_count` 3, `chunked_decode_submit_count`
  2, all timed runs matched. Mean/min/max elapsed, prefill, and decode times
  were populated in the aggregate record.
- Chunk size 16 aggregate: `decode_submit_count` 2, `chunked_decode_submit_count`
  1, all timed runs matched. Same aggregate field structure.

The warmup runs matched references. Timed runs matched references. The tool
exited zero.

### Default-run backward compatibility

```
python3 tools/run_chunked_decode_sweep.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --ids short_correctness_001 \
  --max-new-tokens 16 \
  --chunk-sizes 8
```

Passed with default `--warmup-runs 0 --timed-runs 1`. The tool produced one
per-run record and one aggregate record for chunk size 8. The aggregate
record's mean/min/max values were identical (single run), confirming backward
compatibility with the one-run usage pattern established in diary 0064.

### Existing CTest suite

No CTest registrations were added or modified. The existing chunked decode
CTest suite was not re-run because this entry does not modify runtime code,
shader code, or test infrastructure.

## Interpretation

This entry extends the sweep tool with the repeat/warmup structure called out
in diary 0064's next-work section. The tool can now produce controlled
host-side timing measurements that distinguish warmup overhead from steady-state
timing, and it reports enough statistics (mean/min/max) to surface variance
across runs without claiming a sample size large enough for rigorous inference.

The submit-count geometry is unchanged from diary 0064: chunk size 8 produces
3 decode submits (2 chunked), chunk size 16 produces 2 decode submits (1
chunked). The repeat/warmup extension confirms these counts are stable across
multiple runs of the same configuration, which is expected given that the
counts are deterministic functions of `max_new_tokens` and `chunk_size`, but
it is useful to observe this directly rather than assume it.

This is controlled host-side single-machine timing structure. It is not final
performance proof (no GPU timestamps, no multi-machine replication, no
controlled thermal/power environment), not persistent dispatch, and not the
megakernel.

## Known Limitations

- **Host-side timing only.** Elapsed, prefill, and decode times are wall-clock
  host-side measurements from `spock-decode` JSON output. GPU-side execution
  time is not captured. The tool does not set `SPOCK_GPU_TIMESTAMPS=1`.

- **Not a rigorous benchmark.** Mean/min/max across 2-5 timed runs is a useful
  directional indicator, not a statistically rigorous estimate. Variance
  sources include host scheduling, thermal throttling, and background load.
  The tool makes no attempt to control these.

- **Warmup effectiveness is unquantified.** One warmup run is a reasonable
  heuristic for absorbing pipeline compilation on this RADV stack, but the
  tool does not measure how much warmup timing differs from timed-run timing.
  The caller must judge warmup adequacy from the raw per-run records if they
  are emitted.

- **Hard-coded fast-path gate stack.** Same as diary 0064. The environment
  gates are hard-coded in the tool.

- **Not persistent dispatch.** The sweep runs the chunked decode path, which
  is bounded command-buffer batching on the host. No persistent workgroups, no
  software global barrier, no in-kernel layer loop.

- **Not the megakernel.** This is a measurement convenience extension. It does
  not advance the persistent-dispatch or megakernel milestones.

## Next Work

1. Run a controlled sweep with `--warmup-runs 2 --timed-runs 5` across chunk
   sizes 1, 2, 4, 8, 16 to characterize host-side submission-overhead
   amortization and produce a timing comparison table for the diary.
2. Extend GPU timestamp bookkeeping to multi-step command buffers so per-chunk
   GPU execution time is observable alongside host-side timing.
3. Evaluate whether the RADV bounded-dispatch limits from diary 0053 constrain
   viable chunk sizes for the real decode workload at higher token counts.
4. Consider adding the sweep tool to CI as an optional broad-correctness gate
   once the project has a preferred default chunk size.

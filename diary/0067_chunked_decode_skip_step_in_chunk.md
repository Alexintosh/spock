# 0067: Chunked Decode — First skip_layers Decode Step Absorbed Into Chunk

## Goal

Move the first post-prefill `skip_layers` decode step (final-norm + LM-head +
argmax) into the chunked command buffer so that every decode step in the loop
is eligible for batched submission under `SPOCK_GPU_CHUNKED_DECODE=1`. This
eliminates the always-separate skip-layers submit that existed since diary 0061,
reducing total host submissions by one for every decode invocation.

## Background

Since diary 0061, the chunked decode path treated the first post-prefill
`skip_layers` step as a special case. That step performs final-norm, LM-head,
and argmax on the prefill output to produce the first generated token. It was
always submitted in its own command buffer, outside the chunked batching loop.
The remaining eligible decode steps (steps 2 through `max_new_tokens`) were
batched into chunked command buffers of size `SPOCK_GPU_DECODE_CHUNK_SIZE`.

This split meant the submit-count formula was:

```
decode_submit_count = 1 + ceil((max_new_tokens - 1) / chunk_size)
chunked_decode_submit_count = ceil((max_new_tokens - 1) / chunk_size)
```

The leading "1" was the skip-layers submit, always separate. At large chunk
sizes this was a significant fraction of total submits. For `max_new_tokens=16`
with `chunk_size=16`, the previous formula gave 2 total submits (1 skip-layers +
1 full chunk). But the skip-layers step is not fundamentally different from any
other decode step — it runs the same final-norm + LM-head + argmax sequence,
just on the prefill output rather than on a prior decode output. The only
distinction was that it did not need the full per-layer decode pass (the
`skip_layers` flag short-circuits the layer loop). There is no architectural
reason it cannot share a chunked command buffer with subsequent steps.

## Implementation

The runtime change is confined to `src/runtime/vk_session.cpp`. When
`chunked_decode_enabled` is true and the decode step is the first post-prefill
`skip_layers` step, the code now opens and records into `chunk_cmd` instead of
allocating a separate command buffer. Specifically:

1. **Chunk open**: The skip-layers step opens `chunk_cmd` the same way any other
   eligible step does when no chunk is already open.

2. **Barrier insertion**: After the deferred token copy, the same
   `argmax_result` next-token barrier is inserted. This ensures the generated
   token is visible to the next decode step within the same chunk, matching the
   intra-chunk dependency contract established in diary 0061.

3. **Step counter**: `chunk_recorded_steps` is incremented, making this step
   count toward the chunk fill check.

4. **Deferred submit**: The chunk is not submitted immediately. It remains open
   for subsequent steps until either the chunk is full
   (`chunk_recorded_steps == chunk_size`) or the decode loop ends with a partial
   chunk.

5. **Non-chunked path unchanged**: When `chunked_decode_enabled` is false, the
   skip-layers step follows the original single-submit path. No behavioral
   change outside the gated chunked path.

6. **Timestamp exclusion**: GPU timestamp queries remain excluded from the
   chunked path as established in diary 0061. The `chunked_decode_enabled`
   check still suppresses per-token timestamp recording.

The new submit-count formula is:

```
decode_submit_count = ceil(max_new_tokens / chunk_size)
chunked_decode_submit_count = decode_submit_count
```

Both counters are now identical because every step — including the first
skip-layers step — goes through the chunked path. There is no longer a separate
ungrouped submit.

## Verification

### CTest updates

Existing CTests were updated to reflect the new submit counts:

- **size-4 partial** (`max_new_tokens=6`, chunk size 4): Previously
  `decode_submit_count=3, chunked_decode_submit_count=2`. Now
  `decode_submit_count=2, chunked_decode_submit_count=2`. The six steps fill
  one chunk of four and one partial chunk of two: `ceil(6/4) = 2`.

- **size-8 multiprompt** (`max_new_tokens=16`, chunk size 8, two prompts):
  Previously `decode_submit_count=3, chunked_decode_submit_count=2` per prompt.
  Now `decode_submit_count=2, chunked_decode_submit_count=2`. Sixteen steps fill
  two chunks of eight: `ceil(16/8) = 2`.

### New CTest

Added `spock_vk_decode_chunked_gate_size16_singlechunk_16` — a size-16 single
chunk gate covering `short_correctness_001` and `mixed_correctness_023` with
`max_new_tokens=16` and chunk size 16. This test asserts
`decode_submit_count=1` and `chunked_decode_submit_count=1`, confirming that all
16 decode steps (including the skip-layers first step) fit in a single chunked
command buffer. This is the first test where the entire decode loop is one
submit.

### Build and test results

Build passed. Full CTest suite passed:

- `ctest full fast` — all tests pass
- `ctest size1` — pass
- `ctest size4 partial` — pass
- `ctest size8 multiprompt` — pass
- `ctest size16 singlechunk` — pass

### Refreshed sweep

```
python3 tools/run_chunked_decode_sweep.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --ids short_correctness_001 \
  --max-new-tokens 16 \
  --chunk-sizes 1,2,4,8,16 \
  --warmup-runs 1 \
  --timed-runs 3
```

All chunk sizes produced reference-matching output.

### Refreshed submit counts

| Chunk size | decode_submit_count | chunked_decode_submit_count |
| ---------- | ------------------- | --------------------------- |
| 1          | 16                  | 16                          |
| 2          | 8                   | 8                           |
| 4          | 4                   | 4                           |
| 8          | 2                   | 2                           |
| 16         | 1                   | 1                           |

Every step is now chunked. The `decode_submit_count` and
`chunked_decode_submit_count` columns are identical. The previous version had a
one-submit gap between them (the separate skip-layers submit). That gap is gone.

### Refreshed decode_ms (host-side, 3 timed runs, mean)

| Chunk size | decode_ms (mean) |
| ---------- | ---------------- |
| 1          | 353.09           |
| 2          | 351.98           |
| 4          | 351.24           |
| 8          | 350.14           |
| 16         | 350.49           |

Size 8 produced the lowest mean decode time in this short host-side sample. The
trend is not monotonic after size 8 — size 16 is slightly slower than size 8.
With only 3 timed runs per size at 16 tokens, this difference is well within
run-to-run variance and does not support a conclusion about optimal chunk size.
The direction of decreasing submits is structural; the timing data is
correlating host-side evidence at small sample size.

## What This Does Not Show

- **GPU-side execution time per chunk.** GPU timestamps remain excluded from
  the chunked path. Whether absorbing the skip-layers step into the chunk
  changes GPU-side behavior (e.g., command buffer recording overhead, pipeline
  stalls) is not addressed.

- **Performance at higher token counts.** Sixteen tokens is a short decode. The
  submit-count reduction from 16 to 1 is a large structural change, but the
  wall-clock benefit at 16 tokens is a few milliseconds. Higher token counts
  would amplify the signal.

- **Persistent dispatch viability.** This is bounded command-buffer batching on
  the host. The skip-layers step is now batched like every other step, but the
  underlying mechanism is still host-orchestrated command buffer record and
  submit. Not persistent workgroups, not a software global barrier, not
  in-kernel layer iteration.

- **Megakernel progress.** This change reduces host submissions, which is a
  prerequisite for any future persistent-dispatch design. It does not advance
  the persistent-dispatch or megakernel milestones directly.

## Known Limitations

- **Host-side timing only.** No GPU timestamps. Wall-clock measurements conflate
  GPU execution, driver overhead, and host scheduling.

- **Small sample.** Three timed runs per chunk size is a directional indicator.
  Not sufficient for statistical inference about the true mean at each size.

- **Single prompt.** Only `short_correctness_001` was swept. The submit-count
  geometry is deterministic and prompt-independent, but timing variance could
  differ across prompts.

- **16 generated tokens.** Short decode loop. Longer sequences stress the
  chunked path differently.

- **Non-monotonic timing at large chunk sizes.** The size-16 mean is slightly
  higher than size-8. This is not interpretable at this sample size. It may
  reflect host-side variance, driver queue behavior, or measurement noise. Do
  not overclaim monotonic improvement.

- **Hard-coded gate stack.** Same limitation as diaries 0064–0066. The sweep
  tool uses a fixed fast-path environment gate configuration.

- **Not persistent dispatch.** Bounded command-buffer batching only. No
  persistent workgroups, no software global barrier, no in-kernel layer loop.

- **Not the megakernel.** This reduces host submissions to their theoretical
  minimum for the current architecture (one submit per chunk), but it does not
  eliminate host involvement in the decode loop.

## Next Work

1. Run a longer sweep at `max_new_tokens=128` or higher to characterize whether
   the submit-overhead amortization signal grows with token count and whether
   the non-monotonic size-16 observation persists.

2. Extend GPU timestamp bookkeeping to multi-step chunked command buffers so
   per-chunk GPU execution time is observable alongside host-side timing. This
   would separate GPU work duration from host submit overhead.

3. Sweep across more prompt IDs to confirm timing trends are not
   prompt-specific.

4. Evaluate whether a default chunk size should be recommended once GPU-timing
   and longer-sequence sweeps provide enough data.

5. Consider whether further submit reduction requires moving to persistent
   dispatch or whether the current one-submit-per-N-steps ceiling is
   sufficient for production decode.

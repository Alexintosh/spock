# 0068: Chunked Decode GPU Timestamp Instrumentation

## Goal

Extend GPU timestamp recording into the chunked decode command buffers so that
per-token GPU execution time is observable alongside host-side timing when
`SPOCK_GPU_TIMESTAMPS=1` and `SPOCK_GPU_CHUNKED_DECODE=1` are both active.
This gives the first device-side timing view of chunked decode, replacing the
previous blanket exclusion of timestamps from the chunked path.

## Background

Diary 0040 introduced `SPOCK_GPU_TIMESTAMPS=1` for the non-chunked single-submit
decode path, bracketing each full decode step with Vulkan timestamp queries and
emitting `gpu_decode_us` and `per_token_gpu_us` in JSON output. Diary 0044 added
block-level timestamps behind `SPOCK_GPU_BLOCK_TIMESTAMPS=1` for per-layer
region breakdown.

Diary 0061 introduced chunked decode: one host-allocated command buffer records
up to `SPOCK_GPU_DECODE_CHUNK_SIZE` decode steps and submits on full chunk or
final partial chunk. The initial implementation excluded GPU timestamps entirely
from the chunked path via `!gpu_timestamps` in the chunked-enable condition. This
was the conservative choice — chunked command buffers span multiple decode steps,
and timestamp query management inside multi-step command buffers required careful
placement.

Diary 0067 absorbed the first post-prefill `skip_layers` step into the chunked
command buffer, making every decode step eligible for chunked batching. The
submit-count formula simplified to `ceil(N/C)`. A size-16 single-chunk test with
`max_new_tokens=16` now produces exactly one host submit for all 16 tokens. But
the diary noted explicitly that GPU timestamps were still excluded and that
GPU-side execution time per chunk was unobservable.

The host-side timing from diary 0067 showed decode_ms clustering around 350 ms
for 16 tokens across all chunk sizes, with size 8 slightly best. That is
host-side wall-clock time, conflating GPU execution, driver submit overhead, and
host scheduling. Without GPU timestamps inside the chunked command buffers,
there was no way to separate device work duration from host overhead.

## Implementation

### Removing the timestamp exclusion

The chunked-decode enable condition in `src/runtime/vk_session.cpp` previously
required `!gpu_timestamps` among its prerequisites. This meant `SPOCK_GPU_TIMESTAMPS=1`
and `SPOCK_GPU_CHUNKED_DECODE=1` were mutually exclusive. The exclusion is
removed: the chunked path now coexists with `SPOCK_GPU_TIMESTAMPS=1`.

The block-level timestamp gate `SPOCK_GPU_BLOCK_TIMESTAMPS=1` remains excluded
from the chunked path via `!gpu_block_timestamps`. Block-level timestamps record
per-layer regions inside a single decode step. Inside a chunked command buffer
that spans multiple steps, the region boundaries would multiply and the query
pool sizing logic would need rework. This is a deliberate scope boundary: the
current change covers per-step start/end timestamps only, not per-layer regions.

### Timestamp writes inside chunked command buffers

The existing full-step timestamp mechanism writes a `TIMESTAMP_TOP_OF_PIPE`
query at the start and end of each eligible decode step's command buffer
recording. When the chunked path opens a `chunk_cmd` command buffer, these same
timestamp write operations now occur inside the chunked command buffer:

1. **Step start**: `vkCmdWriteTimestamp` is recorded at the beginning of each
   step's contribution to the chunked command buffer, using the same query pool
   and two-query-per-token index as the non-chunked path.

2. **Step end**: `vkCmdWriteTimestamp` is recorded after each step's argmax and
   deferred token copy, before the next step begins (or before chunk submit if
   this is the last step in the chunk).

3. **First skip_layers step**: Diary 0067 moved the first post-prefill
   skip-layers step into the chunked command buffer. This change adds the
   timestamp start recording for that step as well, so `ts_decode_steps` has
   one entry per generated token — including the first token produced by the
   skip-layers step.

The `ts_decode_steps` counter tracks the number of timestamped decode steps.
Under chunked decode with timestamps enabled, this counter now increments once
per generated token, matching the per-token structure of `per_token_gpu_us`.

### Test infrastructure

`tests/run_vk_decode_parity.py` gains `--expect-gpu-decode-us-positive`, an
assertion that `gpu_decode_us` exists in the JSON output and is greater than
zero. This flag is used by the new timestamp-specific CTest to validate that
GPU timestamps were actually recorded and returned a plausible value.

### New CTest

`spock_vk_decode_chunked_gate_size16_timestamps_short` exercises the combined
chunked + timestamps path:

- Chunk size: 16
- `SPOCK_GPU_TIMESTAMPS=1`
- `--max-new-tokens 16`
- Prompt: `short_correctness_001`
- Assertions: `decode_submit_count=1`, `chunked_decode_submit_count=1`, and
  positive `gpu_decode_us`

This is the first test that validates GPU timestamp data from the chunked decode
path. The size-16 chunk with 16 tokens means all steps fit in a single command
buffer, giving one clean `gpu_decode_us` measurement for the entire decode loop.

## Verification

### Build

CMake build passed with the new code.

### CTest

Three test configurations passed:

- **size-8 multiprompt**: Correctness and submit counts confirmed for chunk size 8
  with two prompts.
- **size-16 singlechunk**: Correctness and submit counts confirmed for chunk size
  16 with 16 tokens, one submit.
- **size-16 timestamps**: The new combined test passed, confirming decode and
  chunked submit counts of 1 each, plus positive `gpu_decode_us`.

### Direct timestamp decode JSON

A manual run with chunk size 16, `SPOCK_GPU_TIMESTAMPS=1`, and 16 generated
tokens produced:

- `decode_submit_count`: 1
- `chunked_decode_submit_count`: 1
- `gpu_decode_us`: 347611
- `per_token_gpu_us`: 16 values, one per generated token

The total GPU decode time of about 348 ms for 16 tokens is consistent with the
host-side decode_ms figures from diary 0067 (around 350 ms mean), which is
expected since the dominant cost is GPU execution time for this single-submit
configuration.

### Host vs GPU timing distinction

The host-side `per_token_ms` values remain chunk-flush-shaped: non-flush steps
mostly show host-side command recording time, while the flush step receives the
chunk submit/wait cost. These are not useful device-timing measurements.

The GPU-side `per_token_gpu_us` values are the useful device timing. Each value
represents the GPU execution time for one complete decode step (embedding lookup,
24 layers, final norm, LM head, argmax) as measured by Vulkan timestamp queries
inside the chunked command buffer. The GPU and host values are not directly
comparable because they measure different things at different points in the
pipeline.

## What This Does Not Show

- **Per-chunk GPU timing.** The timestamps are per-step, not per-chunk. When
  multiple steps share a chunked command buffer, the per-step timestamps can be
  summed to approximate per-chunk GPU time, but no aggregate is emitted
  automatically.

- **Block-level timestamps.** `SPOCK_GPU_BLOCK_TIMESTAMPS=1` is still excluded
  from the chunked path. Per-layer timing within a chunked command buffer is not
  available.

- **Final benchmark proof.** One sample at 16 tokens on one prompt is a
  structural verification, not a performance benchmark. The `gpu_decode_us`
  value confirms timestamps work inside chunked command buffers; it does not
  establish optimal chunk size or throughput.

- **Persistent dispatch.** This is still host-orchestrated command buffer
  batching. The timestamps measure GPU execution time of recorded work inside
  the chunked command buffer. There are no persistent workgroups, no software
  global barrier, and no in-kernel layer iteration.

- **Megakernel progress.** GPU timestamps inside chunked command buffers are a
  measurement capability, not an architectural change. The decode loop is still
  host-driven.

- **Optimal chunk size.** The timestamp data from one 16-token run at chunk
  size 16 does not support conclusions about which chunk size minimizes GPU-side
  overhead. That requires a GPU-timestamped sweep across chunk sizes with
  sufficient repetition.

## Known Limitations

- **Single chunk size tested with timestamps.** The new CTest uses chunk size 16
  only. Other chunk sizes have not been exercised with timestamps in automated
  tests, though the per-step timestamp recording should work at any chunk size
  because it follows the same code path inside the chunked command buffer.

- **One prompt.** The timestamp verification used `short_correctness_001` only.
  Different prompts with different prefill lengths may produce different GPU-side
  timing profiles, but the structural correctness of timestamp placement is
  prompt-independent.

- **16 generated tokens.** Short decode loop. Longer sequences would exercise
  the per-step timestamp recording across more submit boundaries and partial
  chunks.

- **No block-level coverage.** Block timestamps remain excluded. Users who need
  per-layer GPU timing must use the non-chunked path with
  `SPOCK_GPU_BLOCK_TIMESTAMPS=1`.

- **Timestamp query pool sizing.** The timestamp implementation uses two
  queries per timestamped decode token, sized for the maximum expected decode
  steps. If `max_new_tokens` exceeds the pool capacity, timestamp data would be
  silently truncated. The pool sizing should be revisited if longer decode loops
  become standard.

- **Host per_token_ms is chunk-flush-shaped.** This is a known limitation from
  the chunked decode design, not fixed by this change. The useful per-token
  timing is `per_token_gpu_us`; the host-side values reflect submit boundaries,
  not per-step GPU work.

- **Hard-coded gate stack.** Same limitation as diaries 0064–0067. The
  timestamp gate composes with the existing fast-path env configuration but is
  not independently configurable.

- **Not persistent dispatch.** Bounded command-buffer batching only. No
  persistent workgroups, no software global barrier, no in-kernel layer loop.

- **Not the megakernel.** This adds measurement capability to the chunked path.
  It does not advance the persistent-dispatch or megakernel milestones.

## Next Work

1. Run a GPU-timestamped sweep across chunk sizes 1, 2, 4, 8, 16 at higher
   token counts (e.g., `max_new_tokens=64` or `128`) to characterize whether
   GPU-side per-token time varies with chunk size and whether the host-side
   size-8 optimum from diary 0067 reflects GPU behavior or host overhead.

2. Extend block-level timestamp support to the chunked path so that per-layer
   GPU timing is available inside chunked command buffers. This requires query
   pool sizing rework to account for `num_layers * num_steps_per_chunk` queries.

3. Sweep across more prompt IDs with GPU timestamps enabled to confirm that
   per-token GPU timing trends are not prompt-specific.

4. Evaluate whether `per_token_gpu_us` data should feed into an automatic chunk
   size recommendation or remain a manual diagnostic output.

5. Consider whether the host-side `per_token_ms` output should be annotated or
   renamed to clarify that it reflects chunk-flush boundaries, not per-step
   device timing, when chunked decode is active.

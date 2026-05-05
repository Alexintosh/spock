# 0066: Chunked Decode Controlled Sweep — Host-Side Timing Evidence

## Goal

Run a controlled, warmup-guarded sweep across chunk sizes 1, 2, 4, 8, and 16
using the extended sweep tool from diary 0065 to collect host-side submit-count
and decode-timing data for a single prompt. This is the first use of the
repeat/warmup infrastructure to produce a comparable timing table across chunk
sizes, addressing the first next-work item from diary 0065.

This is a measurement-only entry. It does not modify runtime, shader, test, or
tool code.

## Background

Diary 0064 introduced the sweep tool. Diary 0065 extended it with `--warmup-runs`
and `--timed-runs` arguments, producing aggregate mean/min/max timing records.
Both entries called out the need for a controlled sweep across multiple chunk
sizes with warmup and repeated timed runs to characterize host-side
submission-overhead amortization.

The structural prediction for submit counts is deterministic: for
`max_new_tokens=16`, the first decode step is a `skip_layers` single submit, and
the remaining 15 eligible steps are batched into chunked command buffers. The
expected geometry is `ceil(15 / chunk_size)` chunked submits plus one skip-layers
submit, giving total `decode_submit_count = 1 + ceil(15 / chunk_size)`.

What the structural model does not predict is how wall-clock decode time behaves
as submit count decreases. Host-side wall-clock time conflates GPU execution,
driver submission overhead, host scheduling, and any queue synchronization
latency between submits. A controlled sweep with warmup cannot fully separate
these components — that requires GPU timestamps — but it can show whether a
directional trend exists and whether the trend magnitude is consistent with the
submit-count reduction.

## Verification

### Command Run

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

Git revision in output: `7b10974`. The sweep exited zero, meaning every run at
every chunk size produced reference-matching output.

## Results

### Submit counts

| Chunk size | decode_submit_count | chunked_decode_submit_count |
| ---------- | ------------------- | --------------------------- |
| 1          | 16                  | 15                          |
| 2          | 9                   | 8                           |
| 4          | 5                   | 4                           |
| 8          | 3                   | 2                           |
| 16         | 2                   | 1                           |

These counts are exactly as predicted by the structural model
`1 + ceil(15 / chunk_size)` for total submits and `ceil(15 / chunk_size)` for
chunked submits. They are deterministic functions of `max_new_tokens` and
`chunk_size` and do not depend on the specific prompt content. The counts are
stable across all three timed runs at each chunk size.

### Decode timing

All times are host-side wall-clock milliseconds from `spock-decode` JSON output,
aggregated across 3 timed runs with 1 warmup run discarded. The decode phase
timing (`decode_ms`) covers the full decode loop including the skip-layers step
and all chunked submits.

| Chunk size | decode_ms (mean) | decode_ms (min) | decode_ms (max) | Spread (max - min) |
| ---------- | ---------------- | --------------- | --------------- | ------------------ |
| 1          | 353.018          | 352.844         | 353.326         | 0.482              |
| 2          | 351.709          | 351.340         | 351.942         | 0.602              |
| 4          | 350.854          | 350.537         | 351.163         | 0.626              |
| 8          | 350.261          | 350.151         | 350.476         | 0.325              |
| 16         | 349.876          | 349.784         | 350.026         | 0.242              |

The total spread from chunk size 1 to chunk size 16 is about 3.1 ms, or roughly
0.9% of the chunk-size-1 decode time. Within each chunk size, the spread across
three timed runs is under 0.7 ms, indicating low run-to-run variance at this
token count on this machine.

### Elapsed and prefill timing

Host-side `elapsed_ms` and `prefill_ms` were also captured. These values are
noisy and dominated by prefill overhead that is independent of chunk size. The
important signal for chunk-size characterization is the `decode_ms` trend and
the structural submit-count reduction, not the total elapsed time.

## Interpretation

The data shows a modest monotonic decrease in host-side decode time as chunk
size increases, from about 353 ms at chunk size 1 to about 350 ms at chunk size
16. This is consistent with the hypothesis that reducing host submit/wait
round-trips saves a small amount of wall-clock time, but the magnitude is small
at 16 generated tokens.

Several important caveats:

1. **This is host-side timing, not GPU execution time.** The wall-clock
   measurement includes driver submission overhead, host scheduling, and queue
   synchronization. GPU timestamp data would be needed to determine whether the
   actual GPU work duration changes with chunk size or whether the entire
   observed difference is host-side overhead amortization.

2. **The effect size is small at 16 tokens.** A 3 ms reduction over 14
   submit-removals suggests roughly 0.2 ms per eliminated submit round-trip at
   this token count. This is a directional estimate, not a precise measurement
   of per-submit overhead. Higher token counts would amplify the signal.

3. **Single prompt, single machine, single session.** The sweep covered one
   prompt ID (`short_correctness_001`) on one machine. No attempt was made to
   control for thermal state, background load, or power management. The results
   are evidence that the tooling works and the trend direction is plausible, not
   benchmark-quality data.

4. **The submit-count reduction is the robust signal.** The deterministic
   submit-count geometry — from 16 submits at chunk size 1 down to 2 submits at
   chunk size 16 — is the structural guarantee. The timing trend is a
   correlating observation that is consistent with reduced submit overhead but
   does not independently prove a specific overhead amount.

5. **All runs matched reference.** This confirms correctness across all five
   chunk sizes with the full fast-path gate stack active. No degradation was
   introduced by the chunked decode path at any tested size.

## What This Does Not Show

- **GPU-side execution time per chunk.** The tool does not enable
  `SPOCK_GPU_TIMESTAMPS=1`. Whether chunking changes GPU-side execution time
  (e.g., through better cache locality or pipeline occupancy) is not addressed.

- **Performance at higher token counts.** Sixteen tokens is a short decode.
  The submit-overhead fraction at `tg128` (128 generated tokens) may be
  substantially different, and the amortization curve may look different at
  longer sequences.

- **Multi-prompt behavior.** Only `short_correctness_001` was tested. Different
  prompt lengths or content should not change submit-count geometry but could
  affect timing variance.

- **Thermal or power steady state.** No thermal soak or controlled power
  environment. The modest timing spread (under 0.7 ms within each chunk size)
  suggests the machine was reasonably stable during the sweep, but this is not
  a controlled thermal experiment.

- **Persistent dispatch viability.** This is bounded command-buffer batching on
  the host, not persistent workgroups, not a software global barrier, and not
  in-kernel layer iteration.

- **Megakernel progress.** This measurement does not advance the
  persistent-dispatch or megakernel milestones.

## Known Limitations

- **Host-side timing only.** No GPU timestamps. The wall-clock measurements
  conflate GPU execution, driver overhead, and host scheduling.

- **Not a rigorous benchmark.** Three timed runs per configuration is a
  directional indicator. Sample size is too small for statistical inference
  about the true mean at each chunk size.

- **Single prompt.** Only `short_correctness_001` was swept. Multi-prompt
  sweeps are straightforward with the tool but were not run for this entry.

- **16 generated tokens.** The decode loop is short. Longer decode sequences
  would stress the chunked path differently.

- **Hard-coded gate stack.** Same limitation as diaries 0064 and 0065. The tool
  uses a fixed fast-path environment gate configuration.

- **Not persistent dispatch.** Bounded command-buffer batching only. No
  persistent workgroups, no software global barrier, no in-kernel layer loop.

- **Not the megakernel.** This is measurement evidence from the chunked decode
  path. It does not advance the persistent-dispatch or megakernel milestones.

## Next Work

1. Extend GPU timestamp bookkeeping to multi-step chunked command buffers so
   per-chunk GPU execution time is observable alongside host-side timing. This
   would separate GPU work duration from host submit overhead.
2. Run a longer sweep at `max_new_tokens=128` across chunk sizes to characterize
   whether the submit-overhead amortization signal grows with token count.
3. Sweep across more prompt IDs to confirm that the timing trend is not
   prompt-specific.
4. Evaluate whether the RADV bounded-dispatch limits from diary 0053 constrain
   viable chunk sizes for the real decode workload at higher token counts.
5. Consider selecting a preferred default chunk size once GPU-timing and
   longer-sequence sweeps provide enough data.

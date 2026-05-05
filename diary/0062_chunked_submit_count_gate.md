# 0062: Submit-Count Instrumentation for Chunked Decode

## Goal

Add submit-count instrumentation to the decode loop so that the host-side
submission reduction from chunked decode is observable rather than theoretical.
`DecodeResult` and `spock-decode` JSON output now expose
`decode_submit_count` and `chunked_decode_submit_count`. A new CTest asserts
the expected submission geometry for the size-4 partial-boundary case
established in diary 0061.

This is instrumentation, not a performance optimization. No wall-clock timing
is claimed.

## Background

Diary 0061 implemented the first active multi-token chunked decode path. One
command buffer now records up to `SPOCK_GPU_DECODE_CHUNK_SIZE` eligible decode
steps and submits on a full chunk or the final partial chunk. The
implementation was verified manually and with CTests at several token counts,
but the diary explicitly noted:

> The host-side submit reduction is real but unmeasured. The project does not
> yet have submit-count instrumentation or wall-clock comparison between
> chunked and non-chunked paths on the same prompt.

Without instrumentation, the submit-reduction claim rests entirely on code
reading. That is fine for an implementation diary, but it is not sufficient for
the next step: evaluating whether larger chunk sizes (8, 16) produce further
reduction, and whether that reduction translates to wall-clock improvement.

Submit-count instrumentation closes the gap between "the code should produce
fewer submits" and "the code does produce fewer submits, and here are the
exact numbers." It does not measure wall-clock time. It does not prove
performance. It proves structural amortization.

## Implementation Work Completed

### Counters added to the decode loop

Two counters are tracked during the main decode loop:

- `decode_submit_count` increments every time the host submits a command
  buffer and waits on a fence during the decode phase. This includes the
  initial `skip_layers` step (which always follows the single-submit path)
  and every subsequent eligible decode step under whatever path is active
  (single-submit or chunked).

- `chunked_decode_submit_count` increments only when a chunked command buffer
  is submitted. When chunked decode is not active, this counter stays at zero.

The scoping is intentional. The decode loop contains several submit points:
the `skip_layers` first-step path, legacy diagnostic submits, prefill submits,
and the main per-step (or per-chunk) submit. These counters track only the
submits in the main decode loop that fall under the full-fast or chunked path.
They do not count every `submit_and_wait` call in the session. The scope is
narrow because the goal is to measure the submission amortization that chunked
decode specifically provides, not to produce a total submit census.

### Output in DecodeResult and spock-decode JSON

Both counters are written into `DecodeResult` as new fields. When
`spock-decode` emits JSON, the output includes:

```json
{
  "decode_submit_count": 3,
  "chunked_decode_submit_count": 2,
  ...
}
```

The fields are always present, not gated behind an environment variable. When
chunked decode is inactive, `chunked_decode_submit_count` is zero and
`decode_submit_count` equals the number of eligible decode steps plus one for
the skip-layers step. The always-present design was chosen over opt-in output
because the counters have negligible overhead (two integer increments per
submit) and because the absence of a field is less informative than a
counter at zero.

### Size-4 partial CTest now asserts submission geometry

The size-4 partial-boundary CTest from diary 0061 (chunk size 4,
`max_new_tokens=6`) now asserts the expected submission counts. The geometry
for this test case is:

1. **Step 0**: skip-layers path produces the first generated token. This is
   one single-submit command buffer. `decode_submit_count` = 1,
   `chunked_decode_submit_count` = 0.

2. **Steps 1-4**: four eligible decode steps fill one complete chunk of size 4.
   One chunked command buffer submit. `decode_submit_count` = 2,
   `chunked_decode_submit_count` = 1.

3. **Step 5**: the remaining eligible step submits as a partial chunk of size 1.
   One chunked command buffer submit. `decode_submit_count` = 3,
   `chunked_decode_submit_count` = 2.

The CTest asserts `decode_submit_count=3` and
`chunked_decode_submit_count=2`. This verifies that the chunked path produces
exactly the submission amortization the implementation intends: 6 decode tokens
(1 skip-layers + 5 eligible) require only 3 host submissions instead of the 6
that single-submit would produce.

This is a structural assertion, not a performance measurement. It proves the
submission geometry is correct, not that fewer submissions are faster. The
wall-clock effect of reducing submissions is a separate measurement that
requires benchmarking infrastructure not yet in place.

## Verification

All verification was run locally on the RX 6750 XT (RADV NAVI22).

### Full fast gate CTest (unchanged baseline)

```
ctest --test-dir build -R "spock_vk_decode_full_fast_gate_short" --output-on-failure
```

Passed. No regression from the instrumentation.

### Size-1 equivalence CTest (unchanged)

```
ctest --test-dir build -R "spock_vk_decode_chunked_gate_size1_fast_gate_short" --output-on-failure
```

Passed. Size-1 equivalence still holds with the new counters present.

### Size-4 partial-boundary CTest (updated)

```
ctest --test-dir build -R "spock_vk_decode_chunked_gate_size4_partial_short" --output-on-failure
```

Passed. The test now asserts `decode_submit_count=3` and
`chunked_decode_submit_count=2` in addition to checking generated-token output.

## Interpretation

Submit-count instrumentation makes the chunked decode amortization observable
without requiring wall-clock measurement. The three passing CTests confirm:

1. The counters do not regress existing paths (full fast and size-1 equivalence
   pass unchanged).
2. The counters report the expected submission geometry for the size-4
   partial-boundary case.

The key design decision was scoping the counters to the main decode-loop
submissions under the full-fast/chunked path rather than counting every
`submit_and_wait` in the session. A broader scope would include prefill
submits, diagnostic submits, and legacy-path submits that are irrelevant to
the chunked decode amortization question. The narrow scope produces counters
that directly answer "how many host submissions did the decode loop require?"
and "how many of those were chunked submissions?"

The counters are not a performance claim. Fewer submissions should be faster,
but "should" is not measurement. Wall-clock benchmarking comparing chunked and
non-chunked paths on the same prompt is the next step.

## Known Limitations

- **Not a performance measurement.** Submit counts prove structural
  amortization, not wall-clock improvement. The actual time saved per
  eliminated submission depends on driver overhead, fence latency, and GPU
  utilization characteristics that are not captured by these counters.
- **Narrow scope.** The counters do not track prefill submits, diagnostic
  submits, or any submit outside the main decode loop. A total submit census
  would require a different instrumentation point.
- **No per-chunk breakdown.** The counters report totals, not per-chunk
  submit details. For chunk size 4 and 12 eligible steps, the output says
  `decode_submit_count=4` (1 skip-layers + 3 chunks), not "chunk 1 submitted
  at step 4, chunk 2 at step 8, chunk 3 at step 12." Per-chunk detail can be
  added if needed for tuning.
- **Skip-layers step always adds one.** The first post-prefill step follows
  the single-submit path and contributes 1 to `decode_submit_count` regardless
  of chunk size. This is inherent to the current design.

## Next Work

1. Measure wall-clock decode time for the chunked path against the non-chunked
   single-submit path on the same prompt and token count, using the existing
   GPU timestamp instrumentation where applicable.
2. Sweep chunk sizes (4, 8, 16) and record submit-count reduction and
   wall-clock delta for each.
3. Evaluate whether the RADV bounded-dispatch limits observed in diary 0053
   constrain maximum viable chunk size for the real decode workload.
4. Extend GPU timestamp bookkeeping to multi-step command buffers so that
   per-chunk GPU execution time is observable alongside submit counts.

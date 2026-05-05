# 0058: Chunked Decode Gate Scaffold

## Goal

Reserve the environment gates for bounded chunked decode without changing
runtime behavior yet. The next actual inference-path feature is expected to be
bounded multi-token decode chunks, but the implementation is large enough that
the first step should be explicit scaffolding plus regression checks.

This entry does not claim chunked decode is implemented.

## Background

The persistent barrier probe has pushed the project away from an unbounded
single-dispatch megakernel as the immediate target. Memory-heavy barrier probes
crossed a RADV context-loss boundary around long single-dispatch durations,
while bounded repeated dispatches remained stable. The safer next runtime target
is therefore a bounded chunk of GPU-owned decode work, not an unlimited resident
dispatch.

The runtime audit identified the smallest useful inference-path direction:
start from the existing single-submit decode path and eventually record several
decode tokens into one bounded command buffer. That will still be host-submitted,
but it can reduce per-token submit overhead and exercise token-to-token GPU
handoff inside a bounded interval.

Before changing that orchestration, the gate names should exist in code and be
documented as parse-only. This avoids ambiguity in future commits and makes it
clear that setting the variable today must not change execution.

The force-disabled scaffold also prevents a common failure mode in staged GPU
runtime work: accidentally introducing a user-visible gate before all of the
state dependencies are handled. Chunked decode will have to preserve RoPE
positioning, KV cache offsets, DeltaNet recurrent state updates, argmax-to-next
embedding handoff, timestamp query bookkeeping, and deferred generated-token
copy offsets across multiple decode steps recorded into one command buffer. A
gate that is parsed but inert gives later commits a stable place to attach the
real implementation without creating a half-working mode.

## Implementation Work Completed

`src/runtime/vk_session.cpp` now parses:

- `SPOCK_GPU_CHUNKED_DECODE=1`
- `SPOCK_GPU_DECODE_CHUNK_SIZE=N`

near the existing `SPOCK_GPU_SINGLE_SUBMIT` gate in `DecodeSession::decode()`.

The parsed chunk size clamps invalid or missing values back to `1`, with a
temporary upper bound of `1024`. The scaffold then force-disables execution via:

```
const bool chunked_decode_enabled = false;
```

and explicitly marks the parsed values used. This means the commit reserves the
gate names and records the intended constraints, but does not change control
flow, command-buffer recording, token handoff, timestamps, or diagnostics.

## Verification

All verification was run locally on the RX 6750 XT (RADV NAVI22).

### Build

```
cmake --build build -j
```

Passed.

### Existing full fast gate

```
ctest --test-dir build -R "spock_vk_decode_full_fast_gate_short" --output-on-failure
```

Passed:

```
1/1 Test #16: spock_vk_decode_full_fast_gate_short ...   Passed    7.45 sec
```

### Scaffold env remains behavior-free

```
SPOCK_GPU_CHUNKED_DECODE=1 SPOCK_GPU_DECODE_CHUNK_SIZE=4 \
ctest --test-dir build -R "spock_vk_decode_full_fast_gate_short" --output-on-failure
```

Passed:

```
1/1 Test #16: spock_vk_decode_full_fast_gate_short ...   Passed    7.56 sec
```

Because the full fast CTest already sets the current fast decode gates, this
second run confirms that adding the reserved chunked-decode env vars does not
change behavior yet.

## Known Limitations

- **No chunked decode execution.** The gate is parsed but force-disabled.
- **No submit-count reduction.** The runtime still submits once per decode
  token on the single-submit path.
- **No token-to-token in-command-buffer handoff.** The future implementation
  still needs to record multiple decode steps into one bounded command buffer.
- **No new performance data.** This is a correctness-preserving scaffold only.

## Next Work

1. Refactor `DecodeSession::decode()` so the single-token single-submit command
   recording can be reused for K-token bounded chunks.
2. Enable `SPOCK_GPU_CHUNKED_DECODE` first for `DECODE_CHUNK_SIZE=1` equivalence
   before attempting larger chunks.
3. Add chunk-boundary parity tests such as max-new-tokens 5 with chunk size 4.

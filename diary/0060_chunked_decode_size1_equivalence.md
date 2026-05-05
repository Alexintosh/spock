# 0060: Chunked Decode Size-1 Equivalence Gate

## Goal

Move the chunked-decode gate from a purely inert scaffold to a live but
behavior-preserving equivalence mode. `SPOCK_GPU_CHUNKED_DECODE=1` is now
eligible only when `SPOCK_GPU_DECODE_CHUNK_SIZE=1`, so the runtime can validate
the gate wiring without changing command-buffer structure.

This still does NOT implement multi-token chunked decode.

## Background

Diary 0058 reserved the chunked-decode environment variables and force-disabled
the feature. Diary 0059 added a CTest proving that setting the variables did not
change behavior while the scaffold was inert. That was useful as a guardrail,
but it left the gate entirely disconnected from runtime eligibility.

The next safe step is size-1 equivalence. A chunk size of 1 is semantically the
same as the current single-submit fast path: one command buffer per decode
token. It does not reduce submission count, but it proves the gate can be
enabled under the exact fast-path prerequisites without changing output. Once
that is covered by CTest, a future chunk-size-4 implementation has a smaller
delta: change command-buffer lifetime and token-to-token barriers, not gate
parsing and eligibility at the same time.

## Implementation Work Completed

`src/runtime/vk_session.cpp` now computes `chunked_decode_enabled` after the
timestamp gates. It is true only when all of the following are true:

- `SPOCK_GPU_CHUNKED_DECODE=1`
- `SPOCK_GPU_DECODE_CHUNK_SIZE=1`
- the existing `can_single_submit_base` fast-path predicate is true
- `SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1`
- `SPOCK_GPU_DEFER_TOKEN_DOWNLOAD=1`
- GPU timestamp instrumentation is off
- block timestamp instrumentation is off

The value is still not used to alter control flow. This is intentional: size 1
equivalence validates the active gate predicate while leaving the current
one-token single-submit recording path untouched.

`CMakeLists.txt` now renames the scaffold test to:

```
spock_vk_decode_chunked_gate_size1_fast_gate_short
```

and sets:

```
SPOCK_GPU_CHUNKED_DECODE=1
SPOCK_GPU_DECODE_CHUNK_SIZE=1
```

alongside the full fast decode gate stack.

## Verification

All verification was run locally on the RX 6750 XT (RADV NAVI22).

### Build

```
cmake --build build -j
```

Passed.

### Baseline full fast gate

```
ctest --test-dir build -R "spock_vk_decode_full_fast_gate_short" --output-on-failure
```

Passed:

```
1/1 Test #16: spock_vk_decode_full_fast_gate_short ...   Passed    6.17 sec
```

### Chunked gate size-1 equivalence

```
ctest --test-dir build -R "spock_vk_decode_chunked_gate_size1_fast_gate_short" --output-on-failure
```

Passed:

```
1/1 Test #17: spock_vk_decode_chunked_gate_size1_fast_gate_short ...   Passed    6.50 sec
```

The two tests cover the same prompt and token count. The second test proves the
chunked env gate can be set and admitted under size-1 equivalence without
breaking the current full fast decode path.

## Interpretation

This is a small but useful staging point. The project now has:

- a full fast decode baseline test without chunked variables
- a live chunked-gate size-1 equivalence test
- clear separation between gate eligibility and future command-buffer chunking

The next implementation should not change eligibility again. It should focus on
recording several decode steps into one bounded command buffer, starting with a
small chunk size such as 4 and with timestamps disabled.

## Known Limitations

- **No submit reduction.** Chunk size 1 is still one submit per decode token.
- **No multi-token command buffer.** Token-to-token barriers inside one command
  buffer are not implemented yet.
- **No performance claim.** This is a correctness and gate-wiring step.
- **Timestamps excluded.** The first real chunked implementation should keep GPU
  timestamp bookkeeping off until basic parity is proven.

## Next Work

1. Implement chunk size 4 by keeping one command buffer open across multiple
   decode steps.
2. Add explicit barriers from argmax/deferred copy to the next embedding read.
3. Add a chunk-boundary parity test with max-new-tokens 5 and chunk size 4.

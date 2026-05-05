# 0057: Full Fast Decode CTest Gate

## Goal

Add a regression test for the current fully gated fast Vulkan decode path before
attempting bounded multi-token decode chunks. The bounded-chunk refactor will
touch the most complicated part of `DecodeSession::decode()`, so the existing
best fast path needs a direct CTest target.

This is a test gate, not a new runtime feature.

## Background

The project now has strong probe evidence that the next persistent-runtime
direction should be bounded chunks rather than an unbounded memory-heavy single
dispatch. The next inference-path slice recommended by the runtime audit is a
chunked decode command-buffer mode: record several decode tokens into one
bounded command buffer and submit once per chunk. That work depends on the
existing fast decode stack:

- per-layer descriptor sets
- merged DeltaNet command buffers
- fused DeltaNet sub-blocks
- single-submit decode
- device-resident decode token
- deferred token download
- tiled decode matvec
- tiled LM head

Many of these gates were verified when they were introduced, but there was no
single CTest entry covering the combined "full fast" stack. Adding that test is
the safe prerequisite before modifying host-side decode orchestration.

This is especially important because the chunked-decode idea is a host-side
recording change, not a shader replacement. A bug in chunked recording could
look like a precision issue, a descriptor lifetime issue, a token handoff issue,
or a deferred-download indexing issue. A small baseline test for the existing
combined gate stack lets future work answer the first question quickly: did the
known one-token-at-a-time fast path still work before the new chunking logic was
enabled?

## Implementation Work Completed

`CMakeLists.txt` now registers:

```
spock_vk_decode_full_fast_gate_short
```

The test runs `tests/run_vk_decode_parity.py` on:

```
ids: short_correctness_001
max-new-tokens: 4
```

with this environment:

```
SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1
SPOCK_GPU_MERGED_DELTANET=1
SPOCK_GPU_FUSED_DN_CONV_L2=1
SPOCK_GPU_FUSED_DN_GBETA_RECURRENT=1
SPOCK_GPU_FUSED_DN_REC_NORM_GATE=1
SPOCK_GPU_SINGLE_SUBMIT=1
SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1
SPOCK_GPU_DEFER_TOKEN_DOWNLOAD=1
SPOCK_GPU_MATVEC_TILED=1
SPOCK_GPU_LM_HEAD_TILED=1
```

No runtime code changed.

## Verification

All verification was run locally on the RX 6750 XT (RADV NAVI22).

### Build

```
cmake --build build -j
```

Passed.

### New CTest

```
ctest --test-dir build -R "spock_vk_decode_full_fast_gate_short" --output-on-failure
```

Passed:

```
1/1 Test #16: spock_vk_decode_full_fast_gate_short ...   Passed    4.77 sec
```

## Interpretation

This gives the next decode-orchestration work a concrete guardrail. If a future
`SPOCK_GPU_CHUNKED_DECODE` path breaks parity, we can compare against this test
to distinguish a new chunking bug from a pre-existing fast-stack problem.

The test also documents the current strongest gated decode path in CMake rather
than leaving it as an ad hoc command from the diary history.

The selected fixture is intentionally small. `short_correctness_001` with four
generated tokens is enough to exercise more than one decode step, the
device-resident token handoff from each argmax into the next embedding, and the
deferred generated-token buffer. It is not meant to replace longer mixed-prompt
or chunk-prefill sweeps. Its job is to be cheap enough that it can run whenever
the decode orchestration changes.

## Known Limitations

- **Short prompt only.** The test uses one short fixture and four generated
  tokens. It is a fast regression gate, not broad coverage.
- **No chunked decode yet.** This only protects the existing one-submit-per-token
  fast path.
- **Still not full GPU offload.** The host still records and submits per token,
  and final token download remains host-visible after the deferred batch.

## Next Work

1. Use this CTest as the baseline while implementing bounded multi-token decode
   chunks.
2. Add a chunk-boundary test once `SPOCK_GPU_CHUNKED_DECODE` exists.
3. Keep the test small enough to run routinely; use longer parity sweeps for
   milestone validation.

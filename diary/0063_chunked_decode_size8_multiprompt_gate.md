# 0063: Size-8 Chunked Decode Multiprompt CTest Gate

## Goal

Extend chunked decode correctness coverage to a larger chunk size (8) across
multiple prompts with a full 16-token reference. A new CTest,
`spock_vk_decode_chunked_gate_size8_multiprompt_16`, asserts both generated-token
parity and submit-count geometry for two prompts simultaneously under the chunked
decode path.

This is a correctness broadening step, not a performance proof. It does not
introduce new runtime code. It adds one test that exercises existing code along a
previously untested axis: chunk size 8, multiple prompts, full 16-token
reference output.

## Background

Diary 0061 implemented the first active multi-token chunked decode path. One
command buffer records up to `SPOCK_GPU_DECODE_CHUNK_SIZE` eligible decode steps
and submits on a full chunk or the final partial chunk. Diary 0062 added
submit-count instrumentation (`decode_submit_count` and
`chunked_decode_submit_count`) and a CTest asserting the expected submission
geometry for chunk size 4 with 6 generated tokens.

The existing chunked decode CTest coverage before this entry:

- **Size-1 equivalence** (`spock_vk_decode_chunked_gate_size1_fast_gate_short`):
  chunk size 1 on `short_correctness_001` with 4 tokens. Proves the chunked path
  produces identical output to single-submit at the trivial chunk size.

- **Size-4 partial** (`spock_vk_decode_chunked_gate_size4_partial_short`):
  chunk size 4 on `short_correctness_001` with 6 tokens. Proves full-chunk and
  partial-chunk boundary behavior, with submit-count assertions.

Both tests use a single prompt. Neither exercises chunk size 8, and neither
verifies against a reference longer than 6 tokens. The gap matters because
larger chunk sizes produce more aggressive command-buffer batching and longer
intra-chunk token-propagation chains. A 7-step partial chunk at size 8 means
seven consecutive argmax-result barriers within one command buffer, each feeding
the next step's embedding lookup. This is the longest token-dependency chain
tested under chunked decode so far, and it runs across two structurally
different prompts.

## Implementation Work Completed

### New CTest: size-8 multiprompt with 16 tokens

The test `spock_vk_decode_chunked_gate_size8_multiprompt_16` runs
`run_vk_decode_parity.py` with the following parameters:

- **Prompt IDs**: `short_correctness_001,mixed_correctness_023`
- **max_new_tokens**: 16
- **Chunk size**: 8
- **Environment**: the full fast-path gate stack
  (`SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1`,
  `SPOCK_GPU_MERGED_DELTANET=1`,
  `SPOCK_GPU_SINGLE_SUBMIT=1`,
  `SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1`,
  `SPOCK_GPU_DEFER_TOKEN_DOWNLOAD=1`,
  `SPOCK_GPU_MATVEC_TILED=1`,
  `SPOCK_GPU_LM_HEAD_TILED=1`,
  `SPOCK_GPU_FUSED_DN_CONV_L2=1`,
  `SPOCK_GPU_FUSED_DN_GBETA_RECURRENT=1`,
  `SPOCK_GPU_FUSED_DN_REC_NORM_GATE=1`)
  plus `SPOCK_GPU_CHUNKED_DECODE=1` and
  `SPOCK_GPU_DECODE_CHUNK_SIZE=8`.

The test asserts generated-token parity against the stored reference for both
prompts at the full 16-token depth. It also asserts
`decode_submit_count=3` and `chunked_decode_submit_count=2` for each prompt.

### Expected submission geometry

For each prompt with `max_new_tokens=16` and `chunk_size=8`:

1. **Step 0**: The initial `skip_layers` step follows the single-submit path.
   This produces the first generated token. `decode_submit_count` = 1,
   `chunked_decode_submit_count` = 0.

2. **Steps 1-8**: Eight eligible decode steps fill one complete chunk of size 8.
   One chunked command buffer submit. `decode_submit_count` = 2,
   `chunked_decode_submit_count` = 1.

3. **Steps 9-15**: Seven remaining eligible steps submit as a partial chunk of
   size 7. One chunked command buffer submit. `decode_submit_count` = 3,
   `chunked_decode_submit_count` = 2.

Total: 16 generated tokens (1 skip-layers + 8 full chunk + 7 partial chunk) from
3 host submissions instead of 16 under single-submit. The submit-count assertion
is structural, not a wall-clock claim.

### Why two prompts

Using `short_correctness_001` and `mixed_correctness_023` exercises two different
token trajectories through the model. The first is the standard short-prompt
correctness check used throughout the project. The second is a mixed-type prompt
with different token patterns. Running both in one CTest ensures the chunked
path handles diverse decode sequences correctly at chunk size 8, not just the
single well-worn prompt.

The 16-token depth means the test verifies correct argmax output across a full
8-step chunk and a 7-step partial chunk for each prompt. This is the longest
reference output exercised under chunked decode to date. Previous tests stopped
at 4 or 6 tokens.

## Verification

All verification was run locally on the RX 6750 XT (RADV NAVI22).

### Full fast gate CTest (baseline unchanged)

```
ctest --test-dir build -R "spock_vk_decode_full_fast_gate_short" --output-on-failure
```

Passed. No regression.

### Size-1 equivalence CTest (unchanged)

```
ctest --test-dir build -R "spock_vk_decode_chunked_gate_size1_fast_gate_short" --output-on-failure
```

Passed. Size-1 equivalence still holds.

### Size-4 partial CTest (unchanged)

```
ctest --test-dir build -R "spock_vk_decode_chunked_gate_size4_partial_short" --output-on-failure
```

Passed. The size-4 boundary test continues to pass.

### Size-8 multiprompt CTest (new)

```
ctest --test-dir build -R "spock_vk_decode_chunked_gate_size8_multiprompt_16" --output-on-failure
```

Passed. Both prompts produce the expected 16-token reference output. Submit-count
assertions hold: `decode_submit_count=3`, `chunked_decode_submit_count=2` for
each prompt.

## Interpretation

This entry adds a CTest registration. It does not modify runtime code or shader
code. The value is coverage breadth: the chunked decode path is now tested at
chunk sizes 1, 4, and 8, across two prompts, with reference depths up to 16
tokens, and with submit-count assertions at sizes 4 and 8.

The size-8 geometry is particularly relevant because it contains the longest
intra-chunk dependency chain tested so far: seven consecutive
argmax-result-to-embedding barriers within one command buffer during the partial
chunk. If the barrier or deferred-token-copy logic had a subtle ordering bug
that only manifests at higher step counts, this test would catch it.

The test does not prove that chunk size 8 is faster than chunk size 4 or 1. It
proves that it is correct. Wall-clock comparison across chunk sizes remains
future work.

## Known Limitations

- **Not a performance measurement.** The test asserts correctness and submit
  counts. It does not measure wall-clock decode time. Whether larger chunks are
  faster depends on driver command-buffer submission overhead and GPU utilization
  characteristics not captured here.

- **Still not persistent dispatch.** The host submits per chunk. Chunk size 8
  reduces submissions from 16 to 3 for a 16-token decode, but the host is still
  in the loop between chunks.

- **Still not the megakernel.** No persistent workgroups, no software global
  barrier, no in-kernel layer loop. This is bounded command-buffer batching on
  the host.

- **Two prompts, not a corpus.** The test covers two specific prompts. It does
  not replace a broader correctness sweep across dozens of prompts or a fuzz-style
  token-sequence generator.

- **Single chunk size.** The test covers chunk size 8. Chunk sizes 2, 16, and 32
  remain untested by CTest, though the runtime supports any positive integer.

## Next Work

1. Wall-clock benchmarking comparing chunk sizes 1, 4, 8, and 16 on the same
   prompt and token count. Chunked GPU timestamp instrumentation remains
   separate work because timestamps are currently excluded from the chunked gate.
2. Evaluate whether the RADV bounded-dispatch limits from diary 0053 constrain
   viable chunk sizes for the real decode workload.
3. Extend GPU timestamp bookkeeping to multi-step command buffers so per-chunk
   execution time is observable.
4. Consider a broader multiprompt CTest covering more than two prompts if the
   chunk size sweep identifies a preferred default.

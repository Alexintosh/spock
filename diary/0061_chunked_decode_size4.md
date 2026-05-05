# 0061: Bounded Chunked Decode — First Active Multi-Token Path

## Goal

Implement the first active bounded chunked decode path under
`SPOCK_GPU_CHUNKED_DECODE=1` and `SPOCK_GPU_DECODE_CHUNK_SIZE=N`. When these
gates are active with chunk size greater than 1, one command buffer stays open
across up to N eligible decode steps and submits on a full chunk or the final
partial chunk. This is the first multi-token command-buffer recording in the
project, but it is not the final megakernel or full GPU offload.

## Background

Diary 0058 reserved the `SPOCK_GPU_CHUNKED_DECODE` and
`SPOCK_GPU_DECODE_CHUNK_SIZE` environment variables as a force-disabled scaffold.
Diary 0059 added a CTest proving the scaffold did not change behavior. Diary 0060
moved the gate to a live size-1 equivalence mode: `chunked_decode_enabled` was
true only when chunk size was 1, and CTest verified parity against the existing
fast path.

Size-1 equivalence proved the gate wiring was correct, but it did not exercise
the hard part: keeping one command buffer open across multiple decode steps with
correct inter-step barriers, deferred token propagation, and partial-chunk
submission at the end of the decode loop.

The bounded approach comes from the barrier-probe results in diaries 0053-0056.
Long unbounded persistent dispatches fail on this RADV stack once meaningful
memory traffic is present, but bounded dispatches of 100k iterations pass
repeatedly with stable timing. Chunked decode applies the same bounding
principle to the real decode workload: each chunk is one command buffer
containing at most N full decode steps, and the host re-enters between chunks.

## Implementation Work Completed

### Eligibility

`chunked_decode_enabled` is now true when all of the following hold:

- `SPOCK_GPU_CHUNKED_DECODE=1`
- `SPOCK_GPU_DECODE_CHUNK_SIZE` is a positive integer (no longer restricted to 1)
- the existing `can_single_submit_base` fast-path predicate is true
- `SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1`
- `SPOCK_GPU_DEFER_TOKEN_DOWNLOAD=1`
- GPU timestamps are disabled (both regular and block-level)

The timestamp exclusion is deliberate. GPU timestamp queries inject
`vkCmdWriteTimestamp` calls and query-pool resets into the command buffer, and
their bookkeeping assumes one command buffer per decode step. Extending that
bookkeeping to multi-step command buffers adds complexity that would obscure the
core chunking logic. Timestamps can be re-enabled once the chunked path is
stable and a dedicated measurement pass is warranted.

### Skip-layers step exclusion

The first decode step after chunk prefill is a `skip_layers` step that runs only
the final RMSNorm, LM-head matvec, and argmax to produce the first generated
token. This step does not follow the full 24-layer decode schedule and is not
compatible with chunked recording. The implementation keeps this step on the old
single-submit path unconditionally. Chunking begins only on the second decode
step, when `can_single_submit` is true and the step runs the full layer stack.

This means the chunk boundary is relative to eligible decode steps. If chunk
size is 4 and max new tokens is 5 (first token from skip-layers, then four
eligible decode steps), the skip-layers step produces token 0 and the four
eligible steps fill one full chunk. If max new tokens is 6, the skip-layers step
produces token 0, four eligible steps fill one full chunk, and the remaining
step submits as a partial chunk of size 1.

### Command-buffer lifetime

When chunked decode is active, the implementation allocates one command buffer
per chunk rather than per step. The command buffer is recorded across multiple
decode steps: embedding lookup, all 24 layers, final norm, LM head, and argmax
are appended for each step within the chunk. RoPE freq_offsets, KV cache
positions, and other step-varying push constants are recorded inline at each
dispatch point, just as the single-submit path does.

The command buffer submits when:

- the chunk is full (step count reaches `chunk_size`), or
- the decode loop ends with a partial chunk (remaining eligible steps are fewer
  than `chunk_size`)

After submission, the host waits on the fence as usual. The next chunk allocates
a new command buffer.

### Deferred token copy and argmax-result barrier

The existing deferred token download path (diary 0028) copies the argmax result
from the per-step `argmax_result` buffer into a device-local `gen_tokens` buffer
at offset `decode_step * 4` using `vkCmdCopyBuffer`. In the chunked path, this
copy happens inside the shared command buffer at each step.

The next step's embedding lookup reads from the same `argmax_result` buffer via
`embedding_from_buffer` (diary 0027), which reads the token index from device
memory. Without an explicit barrier, the GPU may reorder the deferred copy
relative to the next embedding read, or reorder subsequent argmax writes (from
later steps in the same chunk) relative to the prior copy.

The implementation already had a pre-copy barrier from the argmax shader write
to the transfer read used by `vkCmdCopyBuffer`. The new chunked-only barrier is
inserted after that deferred token copy at each step. It is scoped to the first
4 bytes of `argmax_result` and specifies:

- `srcAccessMask`: `VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT`
- `dstAccessMask`: `VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT`
- `srcStageMask`: `VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT`
- `dstStageMask`: `VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT`
- `srcQueueFamilyIndex` / `dstQueueFamilyIndex`: ignored (same-queue)
- `buffer`: the `argmax_result` buffer
- `offset`: 0, `size`: 4

This ensures that:

1. The deferred copy into `gen_tokens` at the current step's offset completes
   before the next step's embedding read observes the `argmax_result` value.
2. Any prior transfer reads from `argmax_result` (from earlier copies in the
   same chunk) finish before later argmax writes overwrite the buffer.

Without this barrier, step-to-step token propagation inside a single command
buffer would be a data race: the GPU scheduler could overlap the copy from step
N with the argmax write from step N+1, producing an incorrect embedding lookup
and corrupting all downstream computation.

### Tiled matvec and LM-head env stack

The chunked path operates under the full existing tiled matvec and tiled LM-head
environment gates. These are not optional for the chunked path; they are part of
the `can_single_submit_base` prerequisites. The chunked path does not introduce
new shaders, new descriptor layouts, or new buffer classes. It reuses the
existing single-submit recording logic, but calls it repeatedly into the same
command buffer instead of allocating a new one per step.

### Host orchestration

The host still submits per chunk and waits on a fence between chunks. This is
not persistent dispatch and not the megakernel. The submit count for N eligible
decode steps is `ceil(N / chunk_size)` instead of N. For chunk size 4 and
12 eligible steps, the submit count drops from 12 to 3. The actual submit-count
reduction and its wall-clock effect have not been measured yet.

## Verification

All verification was run locally on the RX 6750 XT (RADV NAVI22).

### Full fast gate CTest (unchanged baseline)

```
ctest --test-dir build -R "spock_vk_decode_full_fast_gate_short" --output-on-failure
```

Passed. No regression from the chunked implementation.

### Chunked gate size-1 equivalence CTest (unchanged)

```
ctest --test-dir build -R "spock_vk_decode_chunked_gate_size1_fast_gate_short" --output-on-failure
```

Passed. Size-1 equivalence still holds after the chunk-size-greater-than-1
implementation.

### Manual chunk size 4, max_new_tokens 4

```
SPOCK_GPU_CHUNKED_DECODE=1 SPOCK_GPU_DECODE_CHUNK_SIZE=4 \
  <full fast gate stack> \
  spock-decode short_correctness_001 --max-new-tokens 4
```

Passed. With the first generated token produced by the skip-layers path, this
records the remaining eligible decode steps into one chunk and flushes that
chunk at the end of the loop. It is an end-of-loop partial-chunk check, not a
full-chunk performance claim.

### Manual chunk size 4, max_new_tokens 5

```
SPOCK_GPU_CHUNKED_DECODE=1 SPOCK_GPU_DECODE_CHUNK_SIZE=4 \
  <full fast gate stack> \
  spock-decode short_correctness_001 --max-new-tokens 5
```

Passed. The skip-layers step produces the first generated token, then four
eligible steps fill one complete chunk. The token sequence is correct,
confirming that a full size-4 chunk preserves token propagation and final output.

### Size-4 partial-boundary CTest

A CTest for chunk size 4 with max_new_tokens 6 is being added alongside this
diary entry. It validates the full-chunk-plus-partial boundary: the first
generated token is produced by the skip-layers path, four eligible steps fill
one chunk, and the final eligible step submits as a partial chunk of size 1.

## Interpretation

This is the first time the project records more than one decode step into a
single Vulkan command buffer and gets correct output. The key engineering
insight is the argmax-result barrier: without it, intra-command-buffer
token-to-token propagation is a data race, and the GPU's out-of-order execution
model makes that race manifest as incorrect embeddings and downstream
corruption.

The implementation reuses existing infrastructure almost entirely. No new
shaders, no new descriptor layouts, no new buffer allocations. The chunked path
calls the same single-submit recording function multiple times into one command
buffer, with one additional `vkCmdPipelineBarrier` per step to synchronize the
deferred copy against the next embedding read.

The host-side submit reduction is real but unmeasured. The project does not yet
have submit-count instrumentation or wall-clock comparison between chunked and
non-chunked paths on the same prompt. That measurement is next work, not a
current claim.

## Known Limitations

- **Not the megakernel.** The host still enters between chunks, waits on a
  fence, and allocates a new command buffer. This is bounded multi-step
  recording, not persistent dispatch.
- **No performance measurement.** Submit-count reduction and wall-clock timing
  for the chunked path have not been measured. No performance claim is made.
- **Timestamps excluded.** GPU timestamp bookkeeping assumes one command buffer
  per step. Extending it to multi-step command buffers is deferred.
- **Skip-layers step not chunked.** The first post-prefill step always follows
  the old single-submit path. This adds one unavoidable submit per decode call
  regardless of chunk size.
- **Single queue.** The implementation assumes all work runs on the same Vulkan
  queue. Queue-family ownership transfer is not needed.
- **No early termination.** The chunked path does not support stopping mid-chunk
  on EOS. The entire chunk runs to completion. EOS detection happens at the
  host level between chunks.

## Next Work

1. Add a CTest for chunk size 4 with max_new_tokens 6 to protect the partial-
   chunk boundary from regression.
2. Measure submit count and wall-clock time for the chunked path against the
   non-chunked single-submit path on the same prompt and token count.
3. Evaluate whether larger chunk sizes (8, 16) produce further submit reduction
   without hitting the RADV bounded-dispatch limits observed in diary 0053.
4. Extend GPU timestamp bookkeeping to multi-step command buffers for
   measurement.
5. Begin exploring persistent dispatch within a single chunk: keeping a kernel
   resident across multiple steps inside one command buffer, using the software
   global barrier from the barrier probe (diaries 0047-0056).

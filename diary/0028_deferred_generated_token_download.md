# 0028: Deferred Generated-Token Download — Removing Per-Step CPU Token Readback

## Goal

Eliminate the per-step CPU download of the generated token (`argmax_result`)
on the env-gated deferred-download path, by accumulating tokens on-device
during the decode loop and downloading all generated tokens in one batch
after the loop.

Before this entry, every decode step performed a synchronous 4-byte CPU
download of `argmax_result` to read the generated token ID. This download
was the only way the result escaped the GPU — the harness needed each
token to populate `result.generated_tokens` and `tokens`. The download is
fast (4 bytes), but it serializes each step: the loop cannot advance to
the next iteration until the 4-byte readback completes through the PCIe
round-trip.

After this entry, when `SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1` and
`SPOCK_GPU_DEFER_TOKEN_DOWNLOAD=1` are both set (and the runtime is not
in verbose, debug_dump, or diagnose_decode_drift mode):

```
Loop body (per decode step):
  GPU: argmax writes token_id to argmax_result (device-local, 4 bytes)
  GPU: vkCmdCopyBuffer argmax_result[0..4) → gen_tokens[step*4 .. step*4+4)
  (no CPU download of argmax_result)
  CPU: step advances without waiting for token readback

After the loop:
  CPU: download(gen_tokens, num_generated * 4) → vector<uint32_t>
  CPU: populate result.generated_tokens and tokens from the batch
```

The per-step CPU download is replaced by a device-local `vkCmdCopyBuffer`
(4 bytes, same-queue, no host round-trip) and a single post-loop batch
download of the entire `gen_tokens` buffer.

The change is opt-in and defaults to per-step download. The existing
diagnostic paths (verbose, debug_dump, diagnose_decode_drift) always use
per-step download because they need per-token state at each step.

## Inference Concepts

### The per-step readback bottleneck

Each autoregressive decode step produces one token. In the original loop:

```
for each decode step:
  GPU: run all 24 layers → LM head → argmax → token_id in argmax_result
  CPU: wait for GPU → download 4 bytes from argmax_result → next_token
  CPU: tokens.push_back(next_token)
  CPU: result.generated_tokens.push_back(next_token)
```

The 4-byte download itself is negligible bandwidth (~0.1% of PCIe Gen4 x16).
The cost is the round-trip: submitting a one-shot command buffer, waiting on
a fence, reading 4 bytes over PCIe, and advancing the loop. For short decode
runs the cumulative host-side overhead of N individual fence-wait + download
operations is small but structural — it forces each step to complete before
the next step's command buffer can be submitted.

### What the deferred path changes

After this entry (with both env gates active):

```
for each decode step:
  GPU: run all 24 layers → LM head → argmax → token_id in argmax_result
  GPU (same cmd buffer): barrier + vkCmdCopyBuffer → gen_tokens[step*4..step*4+4)
  CPU: wait for GPU → advance to next step (no per-step download)
  (no per-step token push — that moves to post-loop)

after loop:
  CPU: batch download gen_tokens (num_generated * 4 bytes)
  CPU: for each t in gen_host: tokens.push_back(t); result.generated_tokens.push_back(t)
```

The per-step `vkCmdCopyBuffer` is a same-queue transfer with no host
involvement — it is recorded in the existing command buffer before
`vkEndCommandBuffer`. A `VkBufferMemoryBarrier` transitions `argmax_result`
from `VK_ACCESS_SHADER_WRITE_BIT` to `VK_ACCESS_TRANSFER_READ_BIT` before
the copy, then the copy writes into `gen_tokens` at offset
`decode_step * 4`.

The per-step CPU download is conditionally skipped: when
`defer_token_download` is true, `next_token` is not downloaded and
`tokens`/`result.generated_tokens` are not pushed per-step. Instead,
the post-loop batch download populates both vectors from the device-local
`gen_tokens` buffer.

### What this does NOT change

- **This is NOT full GPU offload and NOT the megakernel.** The host still
  orchestrates per-step command recording, submission, and fence wait.
  The decode loop is still a serial per-step host-mediated sequence. The
  argmax download is eliminated; the step-by-step submit-wait loop is not.
- **Default behavior is unchanged.** The per-step download path remains the
  default. The deferred path activates only when both env gates are set and
  `verbose`, `debug_dump`, and `diagnose_decode_drift` are inactive.
- **The device-resident token gate (`SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1`) is a
  prerequisite.** The deferred download path builds on diary 0027: the
  token_id in `argmax_result` must already be the source of the next
  embedding lookup. If the CPU were still re-injecting the token via push
  constant, the per-step download would still be required for that
  re-injection. With device-resident embedding (diary 0027), the per-step
  download only serves output/parity, so it can be deferred.
- **No change to the embedding, attention, DeltaNet, MLP, LM head, argmax,
  or any other decode-loop component.** Only the post-argmax readback,
  push-to-results, and the post-loop batch download are affected.
- **No performance speedup claim.** The per-step download of 4 bytes is
  fast. The value of this change is structural: it removes a data dependency
  that would block future loop restructuring (e.g., overlapping the next
  step's submission with the prior step's download, or submitting multiple
  steps without host waits).

## Implementation Work Completed

### Local device buffer: `gen_tokens`

A `VulkanDevice::Buffer gen_tokens` is declared as a local variable in
`vk_session.cpp ::decode()`, allocated before the decode loop and destroyed
after the post-loop batch download. The buffer is device-local, sized
`max_new_tokens * 4` bytes (one `uint32_t` per generated token).

The allocation is guarded by `max_new_tokens > 0` to avoid creating a
zero-sized Vulkan buffer. This guard was necessary because `vkCreateBuffer`
with size 0 produces `VK_ERROR_INITIALIZATION_FAILED` on the RADV driver.
Previously, the code never reached Vulkan buffer allocation with
`max_new_tokens == 0` because the loop body would execute zero iterations
and skip all GPU work. But the deferred path allocates the buffer
_pre-loop_, so the zero-token case must be handled. The guard solves this:

```cpp
VulkanDevice::Buffer gen_tokens{};
if (defer_token_download && max_new_tokens > 0) {
  gen_tokens = dev_.create_device_local_buffer(max_new_tokens * 4);
}
```

When `max_new_tokens == 0`, `gen_tokens` remains a default-constructed
(empty/invalid) buffer. Since the loop body never executes, no copy or
download touches it. The post-loop section checks `num_generated > 0`
before downloading from `gen_tokens`, and `destroy_buffer` on a
default-constructed buffer is a no-op (the `Buffer` destructor also
handles this case).

### Gate logic (lines ~1737–1749)

```cpp
const bool defer_token_download = device_resident_token &&
    !verbose && !debug_dump && !diagnose_decode_drift &&
    []() {
      const char* e = std::getenv("SPOCK_GPU_DEFER_TOKEN_DOWNLOAD");
      return e && e[0] == '1' && e[1] == '\0';
    }();
```

The gate requires:
- `device_resident_token == true` (i.e., `SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1`)
- `SPOCK_GPU_DEFER_TOKEN_DOWNLOAD=1`
- `verbose == false`
- `debug_dump == false`
- `diagnose_decode_drift == false`

The diagnostic-mode exclusions are necessary because all three diagnostic
paths read `next_token` (the per-step downloaded value) during the loop
body:
- `verbose` prints the token at each step.
- `debug_dump` dumps per-step logits and requires the token for the
  dump-tokens path.
- `diagnose_decode_drift` captures drift state that depends on the
  per-step token value.

When any diagnostic flag is active, the gate evaluates to false and the
existing per-step download path is used — exactly as before this change.

### Per-step copy (lines ~2588–2606)

After the argmax dispatch in each decode step's command buffer, when
`defer_token_download` is true:

```cpp
if (defer_token_download) {
  VkBufferMemoryBarrier argmax_copy_barrier{...};
  argmax_copy_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  argmax_copy_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  argmax_copy_barrier.buffer = B.argmax_result.buffer;
  argmax_copy_barrier.offset = 0;
  argmax_copy_barrier.size = 4;
  vkCmdPipelineBarrier(cmd,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
      0, 0, nullptr, 1, &argmax_copy_barrier, 0, nullptr);
  VkBufferCopy token_copy{};
  token_copy.srcOffset = 0;
  token_copy.dstOffset = static_cast<VkDeviceSize>(decode_step) * 4;
  token_copy.size = 4;
  vkCmdCopyBuffer(cmd, B.argmax_result.buffer, gen_tokens.buffer, 1, &token_copy);
}
```

The barrier ensures the argmax shader has finished writing `argmax_result`
before the copy reads it. The copy writes to `gen_tokens` at offset
`decode_step * 4`, where `decode_step` is the zero-based decode index
(first decode step → offset 0, second → offset 4, etc.). This preserves
step order in the buffer.

The copy and barrier are recorded into the existing per-step command
buffer, so they add no additional submission overhead — they execute
in the same queue submission as the layer/argmax dispatches.

### Conditional per-step download and result push (lines ~2618–2620, ~2728–2731)

Two conditional blocks gate the per-step download and per-step result
population on `!defer_token_download`:

```cpp
// Per-step download:
uint32_t next_token = 0;
if (!defer_token_download) {
  dev_.download_from_device(B.argmax_result, &next_token, 4);
}
```

```cpp
// Per-step result push:
if (!defer_token_download) {
  tokens.push_back(next_token);
  result.generated_tokens.push_back(next_token);
}
```

When `defer_token_download` is true, `next_token` remains 0 and the
per-step vectors are not populated. The token values will be read from
the `gen_tokens` buffer post-loop.

Note: `next_token` is still declared and zero-initialized (it may be read
by diagnostic code paths even when `defer_token_download` is true, though
the gate forces `defer_token_download = false` for `verbose`,
`debug_dump`, and `diagnose_decode_drift`).

### Post-loop batch download (lines ~2740–2752)

After the decode loop completes, if `defer_token_download` was active:

```cpp
if (defer_token_download) {
  uint32_t num_generated = total_steps - decode_start;
  if (num_generated > 0) {
    std::vector<uint32_t> gen_host(num_generated);
    dev_.download_from_device(gen_tokens, gen_host.data(), num_generated * 4);
    for (uint32_t t : gen_host) {
      tokens.push_back(t);
      result.generated_tokens.push_back(t);
    }
  }
  dev_.destroy_buffer(gen_tokens);
}
```

`num_generated` is the number of decode steps that actually produced tokens
(accounts for the prefill steps that are part of `total_steps` but not
decode-generated). The batch download reads all `num_generated * 4` bytes
from the device-local `gen_tokens` buffer into a host vector, then pushes
them in order into `tokens` and `result.generated_tokens`.

The `num_generated > 0` guard protects against the zero-token case where
`gen_tokens` may still be a default-constructed buffer (allocation guarded
by `max_new_tokens > 0`).

After the batch download, `gen_tokens` is destroyed via
`dev_.destroy_buffer(gen_tokens)`. Since `gen_tokens` is a local variable,
the scope-bound `~Buffer()` would also handle cleanup, but explicit
destruction is consistent with the codebase pattern for managed buffers.

### Files modified

The change touches a single file:

- `src/runtime/vk_session.cpp`: ~60 lines of new code (gate, buffer allocation,
  per-step copy barrier+dispatch, conditional download/result push, post-loop
  batch download, buffer destruction). No new shaders, no new pipelines, no
  new descriptor sets, no changes to session state or header.

This is a purely CPU-side structural change. The GPU work is:
- One `vkCmdPipelineBarrier` (transition from compute write to transfer read)
- One `vkCmdCopyBuffer` (4 bytes, argmax_result → gen_tokens at step offset)

No new shader, SPIR-V, or pipeline compilation is needed.

## Verification

All commands run on the target RADV RX 6750 XT (NAVI22) hardware.

### Build and whitespace

```sh
source ~/.zshrc && cmake -S . -B build && cmake --build build -j
```

Passed cleanly. No compilation or linking errors. No new shaders to compile.

```sh
git diff --check
```

No whitespace errors.

### Zero-token parity (max-new-tokens 0)

```sh
SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1 SPOCK_GPU_DEFER_TOKEN_DOWNLOAD=1 \
  python3 tests/run_vk_decode_parity.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --reference tests/data/reference_tokens.jsonl \
  --ids short_correctness_001 \
  --max-new-tokens 0
```

```
{"status":"ok","checked":1,"failures":[]}
```

The `max_new_tokens > 0` guard on `gen_tokens` allocation prevents a
zero-sized Vulkan buffer from being created. The deferred path correctly
skips both the buffer allocation and the post-loop batch download when
no tokens are generated. The result matches the reference (no generated
tokens). Previously, without the guard, the code path would attempt
`create_device_local_buffer(0)` and fail on RADV.

### Short generation parity (max-new-tokens 16)

```sh
SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1 SPOCK_GPU_DEFER_TOKEN_DOWNLOAD=1 \
  python3 tests/run_vk_decode_parity.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --reference tests/data/reference_tokens.jsonl \
  --ids short_correctness_001 \
  --max-new-tokens 16
```

```
{"status":"ok","checked":1,"failures":[]}
```

The deferred download path produces the correct 16 generated tokens for
`short_correctness_001`. The batch download after the loop retrieves all
16 tokens from the device-local `gen_tokens` buffer and populates
`result.generated_tokens` and `tokens` identically to the per-step path.

### Combined with GPU chunk-prefill gates

```sh
SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1 SPOCK_GPU_DEFER_TOKEN_DOWNLOAD=1 \
  SPOCK_GPU_CHUNK_PREFILL=1 SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1 \
  SPOCK_GPU_CHUNK_PREFILL_TILED=1 \
  python3 tests/run_vk_decode_parity.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --reference tests/data/reference_tokens.jsonl \
  --ids short_correctness_001 \
  --max-new-tokens 16
```

```
{"status":"ok","checked":1,"failures":[]}
```

The deferred download composes correctly with all GPU chunk-prefill gates
active simultaneously.

### Combined gate on longer prompts

```sh
SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1 SPOCK_GPU_DEFER_TOKEN_DOWNLOAD=1 \
  SPOCK_GPU_CHUNK_PREFILL=1 SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1 \
  SPOCK_GPU_CHUNK_PREFILL_TILED=1 \
  python3 tests/run_vk_decode_parity.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --reference tests/data/reference_tokens.jsonl \
  --ids mixed_correctness_023,pp520_046 \
  --max-new-tokens 4
```

```
{"status":"ok","checked":2,"failures":[]}
```

The combined path with deferred download passes on longer prompts
`mixed_correctness_023` and `pp520_046` through 4 generated tokens each.

### Combined with chunk-prefill compare (fallback path preserved)

```sh
SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1 SPOCK_GPU_DEFER_TOKEN_DOWNLOAD=1 \
  SPOCK_GPU_CHUNK_PREFILL=1 SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1 \
  SPOCK_GPU_CHUNK_PREFILL_TILED=1 SPOCK_GPU_CHUNK_PREFILL_COMPARE=1 \
  python3 tests/run_vk_decode_parity.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --reference tests/data/reference_tokens.jsonl \
  --ids short_correctness_001 \
  --max-new-tokens 1
```

```
{"status":"ok","checked":1,"failures":[]}
```

The chunk-prefill compare diagnostic path is orthogonal to the deferred
download path. When `SPOCK_GPU_CHUNK_PREFILL_COMPARE=1` is set, the
chunk-prefill path uses its host-visible fallback as before (diary 0025),
while the deferred download gate independently governs whether tokens are
downloaded per-step or batched post-loop — provided `verbose`,
`debug_dump`, and `diagnose_decode_drift` are inactive. The compare flag
does not disable deferred download (it is not in the flag set that gates
`defer_token_download`), so the deferred path remains active and produces
correct output.

### CTest GPU-collect suite (3/3 passed)

```sh
SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1 SPOCK_GPU_DEFER_TOKEN_DOWNLOAD=1 \
  ctest --test-dir build --output-on-failure -R spock_vk_decode_gpu_collect_chunk_prefill
```

| Test | Status | Time |
|------|--------|------|
| `spock_vk_decode_gpu_collect_chunk_prefill_short` | Passed | 114.98 sec |
| `spock_vk_decode_gpu_collect_chunk_prefill_tiled` | Passed | 9.01 sec |
| `spock_vk_decode_gpu_collect_chunk_prefill_short_baseline` | Passed | 5.09 sec |

All three CTest tests pass with both env gates active. Times are within
noise of the diary 0027 runs. No performance speedup is claimed — the
change does not restructure the submit-wait loop or reduce the number of
host round-trips per token.

### Default path parity (env vars not set)

```sh
python3 tests/run_vk_decode_parity.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --reference tests/data/reference_tokens.jsonl \
  --ids short_correctness_001 \
  --max-new-tokens 16
```

```
{"status":"ok","checked":1,"failures":[]}
```

The default path (per-step download, unchanged) continues to produce
correct output. The new gate has no effect when the env vars are not set.

## Known Limitations

1. **Per-step submit-wait loop is unchanged.** The host still submits one
   command buffer per step and waits on a fence. The deferred download does
   not batch across steps or overlap submission with download. Removing
   per-step download is a structural prerequisite for future loop
   restructuring, not a performance optimization in itself.

2. **Diagnostic paths use per-step download.** When `verbose`, `debug_dump`,
   or `diagnose_decode_drift` is active, the deferred path is disabled and
   the per-step download is always used. This is correct — the diagnostics
   need per-token values at each step — but it means the deferred path
   cannot be tested in diagnostic mode.

3. **Requires `SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1`.** The deferred download
   gate depends on the device-resident token gate (diary 0027). Without
   device-resident embedding, the per-step download is needed both for
   output/parity and for the next step's embedding push constant. The
   deferred path cannot function without the prerequisite gate.

4. **Coverage still limited.** Verified on `short_correctness_001` (16
   tokens), `mixed_correctness_023`, and `pp520_046` (4 tokens each).
   Broader P0 coverage and 512+ token prompts are still pending, as in
   prior entries.

5. **Post-loop batch download is a single large transfer.** For very long
   decode runs (e.g., thousands of tokens), a single batch download of
   `num_generated * 4` bytes may introduce a multi-millisecond transfer
   that delays the first visible output. This is acceptable for the current
   batch-1 correctness-focused harness; streaming use cases would need a
   different strategy (e.g., periodic partial batch downloads).

6. **No performance speedup is claimed.** The per-step download of 4 bytes
   is negligible relative to the layer compute. The deferred path does not
   change the number of command buffer submissions, fence waits, or the
   overall step latency. Its value is structural — it prepares the loop
   for future work that overlaps or defers host-side operations.

## Next Work

### Near-term: Default decision for deferred token download

The deferred path is now verified correct on the tested prompts. The
default remains per-step download. A decision to default the deferred path
would require:
- Broader P0 coverage (all 48 prompts).
- Verification on longer sequences (>512 tokens).
- Confirmation that no diagnostic use case is broken by the switch.
- A decision on whether `SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1` should also
  be defaulted (the deferred path depends on it).

### Near-term: Prompt coverage expansion

Run the deferred download path (alone and combined with GPU chunk-prefill
gates) on additional P0 prompts and longer sequences. The change is
structurally simple (copy + batch download), but empirical verification
across the prompt corpus is needed before considering default.

### Medium-term: Decouple per-step submission from per-step download

With the per-step download eliminated on the deferred path, the decode
loop could be restructured to submit step N+1's command buffer before
waiting on step N's fence. This is not currently done — the loop is still
serial submit-wait per step — but the data dependency is removed. A future
change could:
- Record all N command buffers upfront or submit asynchronously.
- Use timeline semaphores to chain steps without host waits.
- Overlap the post-loop batch download with other work.

### Medium-term: Toward the fused megakernel

- The deferred download eliminates one more CPU-GPU synchronization point
  in the decode loop. The remaining host-mediated points before full GPU
  offload are:
  - Per-step command recording and submission (one submit per step).
  - Per-step fence wait (serializes the loop).
  - Per-layer dispatch orchestration (host iterates layers).
  - Diagnostic readbacks (host downloads per-layer state when enabled).
- Each of these must be addressed before the loop can become a single
  persistent GPU dispatch. The deferred download is a step toward removing
  the per-step readback; the remaining steps require deeper loop
  restructuring.

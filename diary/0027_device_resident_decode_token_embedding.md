# 0027: Device-Resident Decode Token Embedding — Removing CPU Token Re-Injection

## Goal

Eliminate CPU token-value re-injection as the source of the next decode
step's embedding lookup on the env-gated device-resident token path.

Before this entry, every decode step followed this sequence for the embedding:

```
GPU: argmax writes token_id to device-local argmax_result buffer
CPU: download(argmax_result) → uint32_t next_token
CPU: set push_constant = next_token
GPU: embedding_lookup reads token_id from push constant, looks up embedding row from weight table
```

The CPU download of `argmax_result` was always required — it is the *output* of the decode step and must be returned to the test harness for parity checking. But using that downloaded CPU token value as the *source* of the next step's embedding lookup introduced a host round-trip into the per-step data dependency chain: the next embedding dispatch could not begin until the CPU had read, processed, and re-injected the token.

After this entry, when `SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1`:

```
GPU: argmax writes token_id to device-local argmax_result buffer
CPU: download(argmax_result) → uint32_t next_token  (still happens for output/parity)
GPU: next step embedding_lookup_from_buffer reads token_id directly from argmax_result
```

The embedding shader reads `token_id` from `bufs_->argmax_result` — the
same device-local buffer the argmax shader just wrote to — rather than
from a CPU-provided push constant. The CPU download still occurs for output
and parity. In the current serial decode loop, that download still happens
before the next loop iteration; this entry removes the data dependency on
the downloaded value, not the host wait or download scheduling.

## Inference Concepts

### The per-step data chain

Each autoregressive decode step produces a token and then uses that token as the input for the next step's embedding lookup:

```
step N: argmax → token_id[N]
         ↓
step N+1: embedding_lookup(token_id[N]) → hidden_vector
```

Before this entry, the token_id[N] → embedding_lookup link crossed a CPU boundary:

```
argmax_result (device) → CPU download → push constant → shader
```

This round-trip is not expensive relative to the full step time (a 4-byte
download is negligible), but its significance is structural: it makes the
next embedding input come from a CPU-observed value.

After this entry on the gated path, the link is device-resident:

```
argmax_result (device) → embedding shader reads via storage buffer
```

The CPU download still happens (for parity/output), but it is a *read of
the output*, not a *source for the next input*. The current loop still
performs that read before advancing to the next iteration; future loop
restructuring would be needed to overlap or defer it.

### What this changes about the data flow

Before (all paths):

```
GPU: argmax writes token_id → argmax_result (device-local, 4 bytes)
CPU: vkMapMemory / download → next_token (uint32_t)
CPU: vkCmdPushConstants(cmd, ..., 4, &next_token)
GPU: embedding_lookup reads token_id from push_constants
```

After (device-resident token path, SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1):

```
GPU: argmax writes token_id → argmax_result (device-local, 4 bytes)
GPU: (still on device) embedding_lookup_from_buffer reads token_id from argmax_result binding
CPU: downloads argmax_result for output/parity (still synchronous in current loop)
```

### What this does NOT change

- **The CPU still downloads `argmax_result` each decode step.** The token output must be returned for parity checking and external consumption. This entry does not remove that download or the host wait before the next loop iteration — it removes the downloaded token value as a *source* of the next step's embedding.
- **This is NOT full GPU offload and NOT the megakernel.** The host still orchestrates per-step command recording, submission, and fence wait. The decode loop is still per-layer host-mediated. The argmax/logit computation is still on GPU but directed by host dispatches. The token output still reaches the CPU for every step.
- **The existing `embedding_lookup.comp` shader is unchanged.** The old path (push-constant token_id) is preserved as the default. The new path is env-gated and uses a separate shader `embedding_lookup_from_buffer.comp`.
- **No change to the LM head, argmax, logits, or any other decode-loop component.** Only the embedding lookup dispatch is affected. The token embedding weight table is shared between both paths (binding 1 in both shaders).

## Implementation Work Completed

### New shader: `shaders/embedding_lookup_from_buffer.comp`

A new compute shader was written at `shaders/embedding_lookup_from_buffer.comp`. It is structurally identical to the existing `shaders/embedding_lookup.comp` with one difference: the token_id is read from a storage buffer binding instead of a push constant.

Existing shader (push-constant token):

```glsl
layout(push_constant) uniform Params {
    uint token_id;
} params;

void main() {
    uint base = params.token_id * HIDDEN_SIZE;
    ...
}
```

New shader (buffer-resident token):

```glsl
layout(set = 0, binding = 0, std430) readonly buffer TokenBuffer {
    uint token_id;
} token_buf;

void main() {
    uint tok = token_buf.token_id;
    uint base = tok * HIDDEN_SIZE;
    ...
}
```

The binding layout for `embedding_lookup_from_buffer`:
- Binding 0: `bufs_->argmax_result` (device-local, 4 bytes, written by argmax shader)
- Binding 1: token embedding weight table (shared with the existing `embedding_lookup.comp`)
- Binding 2: activation buffer (same output target as the existing `embedding_lookup.comp`)

The shader uses set=0 to match `pipeline_layout_3` (the three-descriptor pipeline layout). Workgroup size is 64 invocations, same as the existing shader, processing `1024 / 64 = 16` elements per invocation.

### Session initialization: `src/runtime/vk_session.cpp` and `src/runtime/vk_session.hpp`

Four new session resources are added in the constructor:

1. **Shader module** (`pipes_->embedding_from_buffer_module`): loaded from `embedding_lookup_from_buffer.comp.spv` in the shader init section alongside the existing `embedding_module`.
2. **Pipeline** (`pipes_->embedding_from_buffer`): created from the new module using `pipeline_layout_3` (the three-descriptor pipeline layout already used by RMSNorm, matvec, and other shaders). This is the same layout used by the new shader's three bindings.
3. **Descriptor set** (`dsets_->embedding_from_buffer`): allocated from `ds_layout_3` via `alloc3()`.
4. **Descriptor bindings**: populated in the session init descriptor-wiring section:
   - Binding 0 → `bufs_->argmax_result` (the device-local 4-byte token buffer)
   - Binding 1 → token embedding weight table (offset, nbytes)
   - Binding 2 → activation buffer `a` (the same buffer the existing embedding writes to)

The destructor destroys the pipeline and shader module in the same teardown section where the other pipelines and modules are destroyed.

### Decode-loop dispatch: `src/runtime/vk_session.cpp ::decode()`

Two changes in the decode function:

**1. Environment gate (lines ~1727–1736)**

A `device_resident_token` boolean is derived from the environment:

```cpp
const bool device_resident_token = []() {
  const char* e = std::getenv("SPOCK_GPU_DEVICE_RESIDENT_TOKEN");
  return e && e[0] == '1' && e[1] == '\0';
}();
if (device_resident_token && !tokens.empty()) {
  upload_raw(dev_, B.argmax_result, &tokens.back(), 4);
}
```

The `upload_raw` call seeds `argmax_result` with the last prompt token before the decode loop starts. This is necessary because the first decode step's embedding must read a valid token_id from `argmax_result`, and the argmax shader has not yet run for that prompt token (it was handled during prefill, or the argmax is about to produce step 0's output).

**2. Conditional dispatch (lines ~1818–1827 in `!skip_layers` branch)**

Inside the embedding lookup block of the decode loop, the pipeline and descriptor set are chosen based on the gate:

```cpp
if (device_resident_token) {
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.embedding_from_buffer);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
      P.pipeline_layout_3, 0, 1, &D.embedding_from_buffer, 0, nullptr);
} else {
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.embedding);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
      P.pipeline_layout_2, 0, 1, &D.embedding, 0, nullptr);
  uint32_t push_token = current_token;
  vkCmdPushConstants(cmd, P.pipeline_layout_2, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &push_token);
}
vkCmdDispatch(cmd, 1, 1, 1);
```

When `device_resident_token` is true:
- The new `embedding_from_buffer` pipeline is bound (uses `pipeline_layout_3`, no push constants).
- The descriptor set `D.embedding_from_buffer` is bound (binding 0 = argmax_result, binding 1 = weights, binding 2 = activation).
- No push constant is set — the token_id is already in `argmax_result` from the previous step's argmax (or from the seed at loop start).

When `device_resident_token` is false:
- The existing `embedding` pipeline is bound (uses `pipeline_layout_2`, with push constants).
- The existing descriptor set `D.embedding` is bound.
- `current_token` is pushed as a push constant — this is the CPU-provided token, exactly as before.

### No other files modified

The change touches 3 files: `CMakeLists.txt` (1 line for the new shader source), `src/runtime/vk_session.cpp` (33 insertions, 5 deletions for init + teardown + dispatch logic), and `src/runtime/vk_session.hpp` (3 lines for pipeline, module, and descriptor set declarations).

### Gate condition

The device-resident token path activates under a single env gate:

- `SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1`

When not set, the runtime uses the existing push-constant embedding path — identical to the behavior before this change.

The gate is independent of the GPU chunk-prefill gates (`SPOCK_GPU_CHUNK_PREFILL`, `SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT`, `SPOCK_GPU_CHUNK_PREFILL_TILED`). It can be combined with them.

## Verification

All commands run on the target RADV RX 6750 XT (NAVI22) hardware. Verification was performed by the orchestrator after the patch was applied in the working tree.

### Build

```sh
source ~/.zshrc && cmake -S . -B build && cmake --build build -j
```

Passed cleanly. No compilation or linking errors. The new shader compiles to SPIR-V alongside the existing shaders.

### git diff --check

```sh
source ~/.zshrc && git diff --check
```

No whitespace errors.

### Default path parity (device-resident token NOT active)

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

The default path (push-constant token, unchanged) continues to produce correct output.

### Device-resident token path parity

```sh
SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1 \
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

The device-resident token path produces the correct decoded output for `short_correctness_001` through 16 generated tokens. The embedding reads `token_id` directly from device-local `argmax_result` instead of from a CPU push constant.

### Combined gate: device-resident token + GPU chunk-prefill all gates

```sh
SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1 \
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

The device-resident token path composes correctly with the GPU chunk-prefill path. All gates active simultaneously produce correct output.

### Combined gate on longer prompts

```sh
SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1 \
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

The combined path passes on longer prompts `mixed_correctness_023` and `pp520_046` through 4 generated tokens each.

### Combined gate with chunk-prefill compare (fallback path preserved)

```sh
SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1 \
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

The combine gate with chunk-prefill compare active also passes. The device-resident token path does not interfere with the chunk-prefill compare diagnostic. The chunk-prefill compare uses the host-visible fallback path (diary 0025) which is orthogonal to the embedding dispatch.

### CTest regression: GPU-collect suite (3/3 passed)

```sh
SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1 \
  ctest --test-dir build --output-on-failure -R spock_vk_decode_gpu_collect_chunk_prefill
```

| Test | Status | Time |
|------|--------|------|
| `spock_vk_decode_gpu_collect_chunk_prefill_short` | Passed | 114.92 sec |
| `spock_vk_decode_gpu_collect_chunk_prefill_tiled` | Passed | 8.97 sec |
| `spock_vk_decode_gpu_collect_chunk_prefill_short_baseline` | Passed | 5.10 sec |

All three CTest tests pass with `SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1` active simultaneously with their respective env gates. Times are within noise of the diary 0026 runs (114.89s, 8.97s, 5.08s). No performance claim is made — the change does not optimize the hot path; it restructures a data dependency.

## Known Limitations

1. **CPU download of argmax_result still occurs.** The token output must be returned for parity and external consumption. This entry removes the downloaded CPU value as the *source* of the next embedding dispatch, not the download itself or the current host wait. The CPU still reads 4 bytes from device memory per step.

2. **Does not make full GPU offload complete.** This is a narrow change to one data dependency in the decode loop. The remaining CPU/host pieces (unchanged since diary 0025):

   - **Per-layer orchestration and submission.** The host still iterates layers, records command buffers, binds descriptors, submits, and waits.
   - **Decode loop orchestration.** Each step requires host intervention to set up the embedding, layer loop, LM head, and argmax dispatches.
   - **Decode argmax/logit orchestration.** The argmax shader is dispatched from host code; the result is still downloaded for output.
   - **Diagnostic/fallback paths.** Readback-heavy diagnostic paths remain.
   - **Broader megakernel fusion and persistent dispatch.** The decode loop remains a per-step host-mediated sequence.

3. **Env-gated, not default.** The existing push-constant embedding path is preserved as the default. The new path activates only with `SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1`. The default must remain unchanged until the device-resident path has broader coverage and a decision is made to default it.

4. **Seeding dependency.** The `upload_raw` call that seeds `argmax_result` with the last prompt token before the decode loop is a host-side data write. This is a one-time cost at loop start (not per-step), but it means the path is not fully device-resident for the first decode step's embedding. A future improvement could make argmax run on the prefill output's final hidden state to produce the initial token without CPU seeding.

5. **Narrow scope.** This change affects only the embedding lookup in the decode loop. The attention, DeltaNet, MLP, LM head, and argmax dispatches are unchanged. The chrounce embedding lookup is a small component (~0.1 ms dispatch out of a multi-second step at current sizes). Performance characterization is deferred.

6. **Coverage still limited.** Verified on `short_correctness_001` (16 tokens), `mixed_correctness_023`, and `pp520_046` (4 tokens each). Broader P0 coverage and 512+ token prompts are still pending, as in prior entries.

## Next Work

### Near-term: Consistent env-gate conventions

The device-resident token gate (`SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1`) follows a different naming convention than the chunk-prefill gates (`SPOCK_GPU_CHUNK_PREFILL`, `SPOCK_GPU_CHUNK_PREFILL_TILED`). Consider standardizing gate naming across the codebase as the number of env gates grows.

### Near-term: Prompt coverage expansion

Run the device-resident token path (alone and combined with GPU chunk-prefill gates) on additional P0 prompts and longer sequences (>512 tokens). The change is structurally straightforward (buffer read vs. push constant), but empirical verification across the prompt corpus is needed before considering default.

### Medium-term: Decouple argmax download from embedding dispatch

The CPU download of `argmax_result` currently happens synchronously after
the per-step GPU dispatches complete, and the serial loop still waits for
that download before starting the next iteration. With the device-resident
token path active, the next embedding no longer needs the downloaded CPU
value, so a future loop restructure could defer the download until after
the next step's embedding dispatch starts, or overlap it with other async
work. The current decode-loop structure does not expose this overlap — it
would require restructuring the loop to submit step N+1's embedding before
waiting for step N's argmax download.

### Medium-term: Toward the fused megakernel

- Remove the seeding `upload_raw` by having the prefill's final argmax write directly to `argmax_result`.
- Eliminate per-step host submission by fusing all layers and the embedding/argmax into a single persistent dispatch, where the embedding reads its input from the previous workgroup's argmax output without host intervention.
- This entry is a tiny step in that direction — it demonstrates that a decode-loop input (the token_id for embedding) can be sourced from device memory rather than CPU injection. The remaining host-mediated data dependencies in the decode loop are: per-layer dispatch control flow, barrier placement, and the argmax download.

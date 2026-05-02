# 0031: Pre-bound RoPE Descriptors — Removing the Per-Step RoPE Descriptor Mutation Blocker

## Goal

Remove the RoPE/per-layer-gated covered path descriptor blocker by
moving the RoPE frequency offset from a descriptor binding
(per-step offset into `B.rope_freq`) into a push constant. The RoPE
descriptors `D.rope` and `D.rope_k` are pre-bound once at session
construction time to the full `rope_freq` table (`MAX_SEQ * ROTARY_DIM * 4`
bytes each), and the per-step position is communicated solely through
the push constant `freq_offset = seq_pos * ROTARY_DIM`.

This removes the RoPE descriptor mutation blocker for the per-layer
stable descriptor set path (diary 0029): under
`SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1`, the per-step RoPE mutation
was the only remaining RoPE `update_descriptor_set` call on the covered
path. The 24 covered descriptor sets plus the 2 RoPE sets are now
pre-bound, but intra-DeltaNet sub-step descriptors (8 sets) still
mutate per dispatch on the decode path.

## Inference Concepts

### RoPE and the per-step offset problem

Rotary Position Embedding (RoPE) applies a frequency-dependent rotation
to the first `rotary_dim` elements of each head vector. The rotation
angles depend on the absolute sequence position `seq_pos`: during
prefill `seq_pos = t` (the prefill-token index), during decode
`seq_pos = step` (the generation step). For a precomputed `rope_freq`
table of shape `[MAX_SEQ, rotary_dim]` with interleaved cos/sin pairs,
position `t` selects slice `t * rotary_dim` through
`(t + 1) * rotary_dim`.

Before this change, the per-step offset was encoded by updating the
descriptor binding at binding 1 (`B.rope_freq`) with a dynamically
changing offset: `seq_pos * ROTARY_DIM * 4` bytes. This required one
`vkUpdateDescriptorSets` call per step for each of `D.rope` and
`D.rope_k`, even though the buffer handle never changed — only the
byte offset into the same buffer.

After this change, the descriptor binds the *entire* `rope_freq` table
at offset 0, and the shader reads `freq_buf.cos_sin[params.freq_offset + lid * 2u]`
where `freq_offset` comes from push constant field `uint freq_offset`.

### What this changes about the descriptor model

| Aspect | Before | After |
|--------|--------|-------|
| RoPE descriptor binding 1 | Per-step offset into `B.rope_freq` | Full table at offset 0, `MAX_SEQ * ROTARY_DIM * 4` bytes |
| Position mechanism | `vkUpdateDescriptorSet(D.rope, binding 1, …, seq_pos * ROTARY_DIM * 4, ROTARY_DIM * 4)` | `vkCmdPushConstants(cmd, …, freq_offset = seq_pos * ROTARY_DIM)` |
| Push constant struct | `{u32 num_heads, u32 head_dim, u32 rotary_dim}` (12 bytes) | `{u32 num_heads, u32 head_dim, u32 rotary_dim, u32 freq_offset}` (16 bytes) |
| Per-step descriptor mutation | 2 `update_descriptor_set` calls (one `D.rope`, one `D.rope_k`) | Zero |
| Descriptor bindings | 3 bindings (in_buf, freq, out_buf) | 3 bindings (unchanged) — freq binding now spans full table |

## Implementation Work Completed

### Shader: `shaders/rope_apply.comp`

Three changes were made:

1. **Push constant struct gains `freq_offset`:**

```glsl
layout(push_constant) uniform Params {
    uint num_heads;
    uint head_dim;
    uint rotary_dim;
    uint freq_offset;  // seq_pos * rotary_dim
} params;
```

Push constant total size grows from 12 bytes to 16 bytes.

2. **Freq buffer comment and mental model updated:**

```glsl
// Before: Freq: [rotary_dim] float32 cos/sin interleaved [cos0,sin0,...]
// After:  Freq: [MAX_SEQ * rotary_dim] float32 cos/sin interleaved [cos0,sin0,...];
//         position selected via push constant freq_offset.
```

The buffer declaration is unchanged (`float cos_sin[]`) — only the
documentation reflects that it now contains the full table rather
than a single position's slice.

3. **Read path uses `freq_offset`:**

```glsl
// Before:
float cos_a = freq_buf.cos_sin[lid * 2u];
float sin_a = freq_buf.cos_sin[lid * 2u + 1u];

// After:
float cos_a = freq_buf.cos_sin[params.freq_offset + lid * 2u];
float sin_a = freq_buf.cos_sin[params.freq_offset + lid * 2u + 1u];
```

### C++ host: `src/runtime/vk_session.cpp`

All RoPE descriptor mutation calls are removed. The pre-binding is done
once at session construction:

**Constructor — pre-bind full table:**

```cpp
// Pre-bind RoPE descriptors: full rope_freq table, Q/K buffers
dev_.update_descriptor_set(dsets_->rope, 0, bufs_->q);
dev_.update_descriptor_set(dsets_->rope, 1, bufs_->rope_freq, 0,
                           MAX_SEQ * ROTARY_DIM * 4);
dev_.update_descriptor_set(dsets_->rope, 2, bufs_->q);
dev_.update_descriptor_set(dsets_->rope_k, 0, bufs_->k);
dev_.update_descriptor_set(dsets_->rope_k, 1, bufs_->rope_freq, 0,
                           MAX_SEQ * ROTARY_DIM * 4);
dev_.update_descriptor_set(dsets_->rope_k, 2, bufs_->k);
```

This binds the full `rope_freq` table at offset 0 with size
`MAX_SEQ * ROTARY_DIM * 4` bytes. The Q/K buffer bindings (binding 0
and binding 2) are also pre-bound — they are the same `B.q`/`B.k`
buffers every time.

**Removed in `layer_major_prefill()`:**

The per-position descriptor mutation block that was inside the per-layer
attention dispatch loop is deleted. In its place, the push constant at
the dispatch site now includes `freq_offset`:

```cpp
// Before (3 fields, 12 bytes):
struct { uint32_t num_heads; uint32_t head_dim; uint32_t rotary_dim; } rope_q_push = { Q_HEADS, HEAD_DIM, ROTARY_DIM };
vkCmdPushConstants(cmd, P.pipeline_layout_32, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, &rope_q_push);

// After (4 fields, 16 bytes):
struct { uint32_t num_heads; uint32_t head_dim; uint32_t rotary_dim; uint32_t freq_offset; } rope_q_push = { Q_HEADS, HEAD_DIM, ROTARY_DIM, t * ROTARY_DIM };
vkCmdPushConstants(cmd, P.pipeline_layout_32, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, &rope_q_push);
```

The same transformation is applied to the `rope_k_push` struct.

**Removed in `decode()` — two mutation blocks:**

Both blocks are removed:
1. The per-layer mutation block (inside `if (!per_layer_sets_enabled_)`):
   the RoPE descriptor updates for `D.rope` and `D.rope_k` that updated
   offsets per layer (redundantly — they used the same `seq_pos` for
   every layer in a given step) are deleted.
2. The per-step mutation block under the per-layer gate
   (`if (per_layer_sets_enabled_ && is_attn)`): the special-case RoPE
   mutation for the gated path is deleted. This was the last per-step
   RoPE `update_descriptor_set` on the covered path (the 24 covered
   sets plus 2 RoPE sets are pre-bound; intra-DeltaNet sub-step
   descriptors still mutate per dispatch).

The decode push constant now passes `step * ROTARY_DIM` as `freq_offset`
instead of a fixed `ROTARY_DIM`.

**Removed in `correct_last_token_hidden()`:**

The same mutation block was present in the `correct_last_token_hidden()`
function for attention layers during prefill's last-token correction
pass. It is deleted, and the push constant now passes
`seq_pos * ROTARY_DIM` as `freq_offset`.

**Comment updated in the per-layer sets section:**

```
// Before: Rope descriptors (D.rope, D.rope_k) are still mutated once per step.
// After:  RoPE descriptors (D.rope, D.rope_k) are pre-bound once at construction time;
```

### What does NOT change

- **No pipeline layout change.** The shader uses `pipeline_layout_32`
  throughout. The push constant range already covers the shader stage.
  The extra 4 bytes (12→16) fit within the existing push constant range.
- **No descriptor pool sizing change.** The two RoPE descriptor sets were
  already allocated. No additional sets are allocated or freed.
- **No buffer reallocation.** `B.rope_freq` is unchanged —
  `MAX_SEQ * ROTARY_DIM * 4` bytes. The only change is that the
  descriptor now references the full buffer rather than a byte slice.
- **No RoPE semantics change.** The rotation applied is identical. The
  mathematical formula, the cos/sin table layout, and the per-head
  rotation pattern are unchanged.
- **No per-layer stable descriptor set interaction.** The RoPE descriptors
  are session-level (shared across all layers), not per-layer sets. The
  per-layer gate (`SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1`) is orthogonal.
- **This is NOT full GPU offload, NOT single-submit, NOT persistent
  dispatch, and NOT the megakernel.** This is a narrow change that
  removes a specific descriptor mutation blocker for future
  command-buffer pre-recording. The host still orchestrates every step
  (per-layer iteration, submission, fence wait). Argmax, logit
  computation, diagnostic readbacks, and fallback paths remain
  host-mediated.

## Verification

All commands run on the target RADV RX 6750 XT (NAVI22) hardware.

### Whitespace and build

```sh
git diff --check
```
No whitespace errors.

```sh
cmake --build build -j
```
Passed cleanly. The shader recompilation and C++ recompilation complete
without warnings.

### Baseline decode parity (default path, no env vars)

```sh
python3 tests/run_vk_decode_parity.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --reference tests/data/reference_tokens.jsonl \
  --ids short_correctness_001 \
  --max-new-tokens 16
```

```json
{"status":"ok","checked":1,"failures":[]}
```

### Per-layer stable descriptor sets gate

```sh
SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1 \
  python3 tests/run_vk_decode_parity.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --reference tests/data/reference_tokens.jsonl \
  --ids short_correctness_001 \
  --max-new-tokens 16
```

```json
{"status":"ok","checked":1,"failures":[]}
```

The per-layer stable descriptor set path no longer has the special-case
per-step RoPE mutation block. Previously (diary 0029), the gated path
still had `if (per_layer_sets_enabled_ && is_attn)` updating RoPE
descriptors per step. This is now removed — RoPE bindings are fully
pre-bound.

### Combined gates (device-resident token, deferred download, GPU chunk-prefill suite)

```sh
SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1 \
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

```json
{"status":"ok","checked":1,"failures":[]}
```

### Longer prompts with combined gates

```sh
SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1 \
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

```json
{"status":"ok","checked":2,"failures":[]}
```

### CTest GPU-collect suite (3/3 passed)

```sh
SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1 \
  SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1 SPOCK_GPU_DEFER_TOKEN_DOWNLOAD=1 \
  ctest --test-dir build --output-on-failure -R spock_vk_decode_gpu_collect_chunk_prefill
```

| Test | Status | Time |
|------|--------|------|
| `spock_vk_decode_gpu_collect_chunk_prefill_short` | Passed | 101.67 sec |
| `spock_vk_decode_gpu_collect_chunk_prefill_tiled` | Passed | 16.82 sec |
| `spock_vk_decode_gpu_collect_chunk_prefill_short_baseline` | Passed | 8.93 sec |

All 3/3 CTest tests pass. Total suite time 127.42s.

## Known Limitations

1. **This does NOT make full GPU offload complete.** The host still:
   - Iterates per-layer dispatches in the decode loop.
   - Submits command buffers and waits on fences per step.
   - Observes/downloads generated-token outputs, logits, and diagnostic data for external output, parity checking, and fallback/diagnostic paths.
   - Orchestrates prefill (layer-major loop).

2. **This does NOT make single-submit complete.** Remaining blockers for
   single-submit recording:
   - Intra-DeltaNet sub-step descriptors (`dn_split_q`, `dn_split_kv`,
     `dn_l2_q`, `dn_l2_k`, `dn_recurrent`, `dn_norm_gate`,
     `dn_out_proj`, `dn_compute_g_beta`) still mutate per dispatch
     on the decode path. Diary 0030 documents that a naive pre-binding
     attempt for these failed with decode-state corruption at step 1.
   - Host-side per-layer/per-step submissions and fence waits remain.
   - Fallback/diagnostic readbacks remain.

3. **This does NOT remove the per-step push constant update.** The RoPE
   position is now communicated via `vkCmdPushConstants` (per dispatch)
   rather than `vkUpdateDescriptorSet` (per step). Both are per-step
   work. The win is descriptor immutability (no per-step
   `vkUpdateDescriptorSet` call for RoPE), not completed pre-recording.
   Push constants are recorded into command buffers and cannot be
   changed at submission time without re-recording. A single-submit
   strategy would need to either move the step position into a
   GPU-readable state (e.g., a storage buffer read by the shader) or
   adopt a command-buffer strategy that supports per-step parameter
   updates.

4. **Push constant is now 16 bytes instead of 12.** This is within the
   guaranteed push constant range of all Vulkan 1.0+ devices (128 bytes),
   and the existing `pipeline_layout_32` push constant range already
   covers this. No layout change was needed.

## Relationship to Diary 0029

Diary 0029 (per-layer stable descriptor sets) identified RoPE descriptor
mutation as a remaining blocker under `SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1`.
The `decode()` function had a special-case block:

```cpp
if (per_layer_sets_enabled_ && is_attn) {
  dev_.update_descriptor_set(D.rope, …);
  dev_.update_descriptor_set(D.rope_k, …);
}
```

This entry removes that special-case block. Under the gate, no RoPE
`update_descriptor_set` calls remain on the decode path. The 24 covered
descriptor sets plus the 2 RoPE sets are pre-bound at session
construction time — all 26 together are descriptor-mutation-free on
the covered path. Only the 8 intra-DeltaNet sub-step descriptors
remain uncovered and still mutate per dispatch on the decode path.

## Files Changed

```
shaders/rope_apply.comp       — Push constant gains freq_offset; freq read path uses it
src/runtime/vk_session.cpp    — RoPE pre-bound at construction; all per-step mutation blocks removed
```

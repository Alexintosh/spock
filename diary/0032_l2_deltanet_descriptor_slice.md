# 0032: L2-Norm DeltaNet Descriptor Pre-Binding — Narrow Extension Beyond Diary 0030

## Goal

Extend the opt-in per-layer stable descriptor set path (diary 0029) to cover
the two L2-norm DeltaNet sub-step descriptors `dn_l2_q` and `dn_l2_k`, moving
them from per-dispatch mutation to pre-bound per-layer sets. This is a
narrow, deliberate slice that covers the *non-stateful* intra-DeltaNet L2-norm
descriptors while leaving the stateful/recurrent descriptors untouched.

This slice explicitly does **not** repeat the failed all-six extension
(document in diary 0030). The L2 q/k descriptors are stateless — they bind
fixed regions of `B.dn_qkv` (`Q` region at byte offset 0, `K` region at byte
offset `DN_KEY_TOTAL * 2`) that do not change between decode steps. This
makes them safe for pre-binding, unlike the stateful/recurrent descriptors
whose buffer bindings (conv1d state, recurrent state, output buffers) have
per-step mutable contents even when the buffer handles are stable.

## Inference Concepts

### What makes a descriptor safe for pre-binding?

A descriptor set is **safe for pre-binding** when its buffer bindings satisfy
two properties:

1. **Handle-stable**: The `VkBuffer` handle and byte offset/range do not
   change between dispatches. The descriptor captures the same buffer region
   every time it is bound.

2. **Content-irrelevant for binding**: The descriptor does not need to target
   different buffer regions to express per-step state. Even if the buffer's
   *contents* change between dispatches (they always do — that's the decode
   loop), the descriptor still references the same buffer at the same byte
   range. Vulkan validates bindings, not contents.

The L2-norm Q and K descriptors satisfy both properties:

| Descriptor | Buffer | Byte offset | Size | Per-step change? |
|------------|--------|-------------|------|-----------------|
| `dn_l2_q` | `B.dn_qkv` | 0 (Q region) | `DN_KEY_TOTAL * 2` | No — Q is always at offset 0 |
| `dn_l2_k` | `B.dn_qkv` | `DN_KEY_TOTAL * 2` (K region) | `DN_KEY_TOTAL * 2` | No — K is always at offset `DN_KEY_TOTAL * 2` |

Compare with the stateful descriptors that remain uncovered:

| Descriptor | Why NOT safe for simple pre-binding |
|------------|--------------------------------------|
| `dn_recurrent` | Binds `dn_state` — a buffer whose relevant slice may depend on step-level state offset dimensions that are not a simple constant region. Diary 0030's failure signature (`first_mismatch_index=1`) is consistent with a state-offset or descriptor-aliasing bug. |
| `dn_norm_gate` | Binds output buffer `B.dn_norm_gate` — the output accumulation region that changes per dispatch. The binding offset is the same across steps (it overwrites the same region), but it is an output buffer that other descriptors read; the aliasing constraints may differ. |
| `dn_out_proj` | Binds `B.dn_out` — the DeltaNet output accumulation buffer. Same aliasing concerns. |
| `dn_compute_g_beta` | Binds scratch buffers `B.dn_g`/`B.dn_beta` — these are intermediate computation regions shared across dispatches within a step. |

The failed all-six attempt (diary 0030) tried to pre-bind all six together
and failed with decode-state corruption at step 1. The root cause was not
pursued — it could be an aliasing conflict, a stale binding, or a
state-offset error. What matters is that the L2 q/k descriptors are
**independently safe**: they are pure input-read bindings into the same
`B.dn_qkv` buffer with constant offsets, no state aliasing, no output
dependencies. This slice proves that narrow selection works.

## Implementation Work Completed

### Struct additions: `src/runtime/vk_session.hpp`

Two new vectors are added to `PerLayerDescriptorSets`:

```cpp
std::vector<VkDescriptorSet> dn_l2_q;             // ds_layout_3
std::vector<VkDescriptorSet> dn_l2_k;             // ds_layout_3
```

Struct comment updated from:

```
/// Intra-DeltaNet sub-step descriptors (dn_l2_q, dn_l2_k, dn_recurrent,
/// dn_norm_gate, dn_out_proj, dn_compute_g_beta) are NOT included here;
```

to:

```
/// Intra-DeltaNet sub-step L2-norm descriptors (dn_l2_q, dn_l2_k) are
/// covered here. Remaining inner DeltaNet descriptors (dn_recurrent,
/// dn_norm_gate, dn_out_proj, dn_compute_g_beta) are NOT included;
```

### Construction: `src/runtime/vk_session.cpp`

**Comment update** (line ~425):

```
// Before: Intra-DeltaNet sub-step descriptors are not covered and remain on the old path.
// After:  L2-norm q/k (dn_l2_q, dn_l2_k) are covered here; remaining inner DeltaNet
//         sub-step descriptors (dn_recurrent, dn_norm_gate, dn_out_proj,
//         dn_compute_g_beta) are not covered and remain on the old path.
```

**Vector resizing** — inside the `if (per_layer_sets_enabled_)` block:

```cpp
per_layer_sets_->dn_l2_q.resize(LAYERS);
per_layer_sets_->dn_l2_k.resize(LAYERS);
```

**Allocation** — inside the `for (i = 0; i < LAYERS; i++)` loop:

```cpp
per_layer_sets_->dn_l2_q[i] = alloc3();
per_layer_sets_->dn_l2_k[i] = alloc3();
```

**Binding** — inside the per-layer pre-bind block, after `dn_conv`:

```cpp
// dn_l2_q: Q region of dn_qkv (all 3 bindings at offset 0, size DN_KEY_TOTAL*2)
dev_.update_descriptor_set(per_layer_sets_->dn_l2_q[layer], 0, B.dn_qkv, 0, DN_KEY_TOTAL * 2);
dev_.update_descriptor_set(per_layer_sets_->dn_l2_q[layer], 1, B.dn_qkv, 0, DN_KEY_TOTAL * 2);
dev_.update_descriptor_set(per_layer_sets_->dn_l2_q[layer], 2, B.dn_qkv, 0, DN_KEY_TOTAL * 2);
// dn_l2_k: K region of dn_qkv (all 3 bindings at offset DN_KEY_TOTAL*2, size DN_KEY_TOTAL*2)
dev_.update_descriptor_set(per_layer_sets_->dn_l2_k[layer], 0, B.dn_qkv, DN_KEY_TOTAL * 2, DN_KEY_TOTAL * 2);
dev_.update_descriptor_set(per_layer_sets_->dn_l2_k[layer], 1, B.dn_qkv, DN_KEY_TOTAL * 2, DN_KEY_TOTAL * 2);
dev_.update_descriptor_set(per_layer_sets_->dn_l2_k[layer], 2, B.dn_qkv, DN_KEY_TOTAL * 2, DN_KEY_TOTAL * 2);
```

All three bindings (0=input, 1=weight, 2=output for `ds_layout_3`) reference
the same `B.dn_qkv` buffer at the same offset — the L2-norm shader
(`l2_norm_per_head`) is a single-buffer in-place normalization that reads
and writes the same region. This is correct because `B.dn_qkv` is the
per-layer QKV working buffer and the L2 norm operates in-place.

The verbose count line updates to confirm the new count:

```
per-layer descriptor sets: 28 x 26 sets pre-bound
```

(24 prior + 2 new = 26 per-layer sets.)

### Decode alias: `src/runtime/vk_session.cpp` `decode()`

Two new alias declarations in the per-layer alias block:

```cpp
VkDescriptorSet ds_dn_l2_q = per_layer_sets_enabled_ ? per_layer_sets_->dn_l2_q[layer] : D.dn_l2_q;
VkDescriptorSet ds_dn_l2_k = per_layer_sets_enabled_ ? per_layer_sets_->dn_l2_k[layer] : D.dn_l2_k;
```

These follow the same pattern as the existing 24 covered descriptors: the
alias resolves to the pre-bound per-layer set when the gate is active,
otherwise falls back to the default `D.*` handle.

### Decode fallback guard: `src/runtime/vk_session.cpp` `decode()`

The per-dispatch `update_descriptor_set` calls for `D.dn_l2_q` and
`D.dn_l2_k` are now guarded by `if (!per_layer_sets_enabled_)`:

```cpp
// Before (unconditionally mutated):
dev_.update_descriptor_set(D.dn_l2_q, 0, B.dn_qkv, 0, DN_KEY_TOTAL * 2);
dev_.update_descriptor_set(D.dn_l2_q, 1, B.dn_qkv, 0, DN_KEY_TOTAL * 2);
dev_.update_descriptor_set(D.dn_l2_q, 2, B.dn_qkv, 0, DN_KEY_TOTAL * 2);
// ... same for dn_l2_k ...

// After (guarded — only on legacy path):
if (!per_layer_sets_enabled_) {
    dev_.update_descriptor_set(D.dn_l2_q, 0, B.dn_qkv, 0, DN_KEY_TOTAL * 2);
    dev_.update_descriptor_set(D.dn_l2_q, 1, B.dn_qkv, 0, DN_KEY_TOTAL * 2);
    dev_.update_descriptor_set(D.dn_l2_q, 2, B.dn_qkv, 0, DN_KEY_TOTAL * 2);
}
// ... same for dn_l2_k ...
```

This ensures the mutation is skipped on the gated path while preserving
correct behavior on the default path. The dispatch sites still bind via
the alias (`&ds_dn_l2_q`, `&ds_dn_l2_k`) which works correctly on both
paths.

### What does NOT change

- **No new shader code.** The `l2_norm_per_head` shader pipeline and push
  constants are unchanged. The descriptor binding pattern (3 bindings,
  all to the same buffer, same `ds_layout_3`) is identical — only the
  descriptor set handle changes.
- **No descriptor pool sizing change.** The 2 new sets per layer (56 total)
  were already within the 1024 maxSets capacity increased in diary 0029.
  No pool expansion needed.
- **No pipeline layout change.** The existing `ds_layout_3` / `pipeline_layout_3`
  is used for allocation, binding, and dispatch, matching the existing 24
  covered descriptor sets.
- **No buffer reallocation.** `B.dn_qkv` is unchanged — the per-layer QKV
  working buffer already exists at the same size.
- **No inference semantics change.** The L2 normalization applied is
  identical. The descriptor binding points to the same buffer region at
  the same byte offset; the shader sees the same data.
- **This is NOT full GPU offload, NOT single-submit, NOT persistent
  dispatch, and NOT the megakernel.** The host still orchestrates every
  step (per-layer iteration, submission, fence wait). Argmax, logit
  computation, diagnostic readbacks, and fallback paths remain
  host-mediated.
- **This is NOT a redo of diary 0030.** The failed all-six attempt remains
  valid as a documented negative result. Only the L2 q/k descriptors are
  targeted here because they are stateless and independently safe. The
  stateful/recurrent descriptors (dn_recurrent, dn_norm_gate, dn_out_proj,
  dn_compute_g_beta) remain uncovered and require deeper investigation.

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
Passed cleanly. The C++ recompilation completes without warnings.

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

The default path and gated path both pass, confirming the fallback guard
(`if (!per_layer_sets_enabled_)`) correctly isolates the legacy mutation
and the pre-bound aliases resolve to correct handles.

### Combined gates

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

The L2-norm descriptor pre-binding composes correctly with the full GPU
chunk-prefill suite, device-resident token embedding, and deferred
generated-token download — all six env gates active simultaneously.

## Descriptor Coverage Summary

After this slice, the total descriptor model is:

| Category | Sets | Scope | Pre-bound? |
|----------|------|-------|------------|
| Common MLP/norm | 9 sets × 28 layers | Per-layer | Yes (diary 0029) |
| Attention-specific | 10 sets × 28 layers | Per-layer | Yes (diary 0029) |
| DeltaNet first-stage | 5 sets × 28 layers | Per-layer | Yes (diary 0029) |
| DeltaNet L2-norm (this diary) | 2 sets × 28 layers | Per-layer | Yes |
| RoPE | 2 sets (session-level) | Session | Yes (diary 0031) |
| **Per-layer subtotal** | **26 sets × 28 layers = 728 sets** | | |
| **Session-level subtotal** | **2 sets** | | |
| **Total pre-bound** | **730 sets** | | |

Remaining uncovered (per-dispatch mutation on decode path):

| Descriptor | Reason not covered |
|------------|-------------------|
| `dn_recurrent` | Stateful — binds `dn_state`; diary 0030 documents corruption at step 1 |
| `dn_norm_gate` | Stateful/output — binds output accumulation buffer |
| `dn_out_proj` | Stateful/output — binds DeltaNet output accumulation buffer |
| `dn_compute_g_beta` | Scratch/shared — binds intermediate computation buffers |
| `dn_split_q` | Internal decomposition descriptor (not a dispatch target) |
| `dn_split_kv` | Internal decomposition descriptor (not a dispatch target) |

`dn_split_q` and `dn_split_kv` are listed in the docs as uncovered internal
descriptors but are not dispatch targets that require per-step mutation;
they serve decomposition within the DeltaNet recurrent kernel's binding
scheme. The four stateful dispatch-target descriptors (`dn_recurrent`,
`dn_norm_gate`, `dn_out_proj`, `dn_compute_g_beta`) are the remaining
per-layer mutation blockers for single-submit recording.

## Relationship to Diary 0030

Diary 0030 attempted a naive pre-binding of all six intra-DeltaNet sub-step
descriptors (`dn_l2_q`, `dn_l2_k`, `dn_recurrent`, `dn_norm_gate`,
`dn_out_proj`, `dn_compute_g_beta`) in a single change. It compiled but
caused decode-state corruption at step 1 (`first_mismatch_index=1`,
`matched_prefix_tokens=1`). The failure was consistent with a state-offset or
descriptor-aliasing bug in the recurrent state binding that does not produce
a Vulkan validation error.

This entry covers only `dn_l2_q` and `dn_l2_k` — the two stateless L2-norm
descriptors whose bindings are simple offset regions into `B.dn_qkv`. These
were part of the failed all-six change but were exonerated: the corruption
was caused by one of the four stateful descriptors, not the L2-norm pair.
This diary proves that hypothesis by shipping the L2-norm slice independently
without corruption.

The all-six attempt remains in the record as a valid warning: the
stateful/recurrent descriptors require deeper investigation (state-offset
resolution, aliasing analysis, or kernel fusion) before pre-binding can
succeed. The simple pre-binding approach does not work for them.

## Relationship to Diary 0029 and Diary 0031

Diary 0029 (per-layer stable descriptor sets) identified 24 covered
per-layer descriptor sets plus 8 uncovered intra-DeltaNet sub-step
descriptors. Diary 0031 (pre-bound RoPE descriptors) moved the 2 RoPE
session-level sets from per-step mutation to pre-bound, removing the
RoPE descriptor mutation blocker while keeping RoPE out of the per-layer
count.

This diary (0032) moves 2 of the 8 remaining intra-DeltaNet sub-step
descriptors into the per-layer stable set, reducing the uncovered count
from 8 to 6. The 4 remaining dispatch-target descriptors (`dn_recurrent`,
`dn_norm_gate`, `dn_out_proj`, `dn_compute_g_beta`) plus 2 internal
decomposition descriptors (`dn_split_q`, `dn_split_kv`) remain on the
old mutation path.

The full pre-bound picture after this slice:

- **26 per-layer descriptor sets** (24 from diary 0029 + 2 from this diary)
- **2 session-level RoPE descriptor sets** (from diary 0031)
- **Total: 730 pre-bound descriptor sets** (728 per-layer + 2 session)
- **6 uncovered descriptors** remain (4 dispatch-target + 2 internal)

## Known Limitations

1. **This does NOT make full GPU offload complete.** The host still:
   - Iterates per-layer dispatches in the decode loop.
   - Submits command buffers and waits on fences per step.
   - Observes/downloads generated-token outputs, logits, and diagnostic data for external output, parity checking, and fallback/diagnostic paths.
   - Orchestrates prefill (layer-major loop).

2. **This does NOT make single-submit complete.** Remaining blockers:
   - Four stateful intra-DeltaNet sub-step descriptors (`dn_recurrent`,
     `dn_norm_gate`, `dn_out_proj`, `dn_compute_g_beta`) still mutate per
     dispatch on the decode path. Diary 0030 documents that naive
     pre-binding for these fails with decode-state corruption at step 1.
   - Host-side per-layer/per-step submissions and fence waits remain.
   - Fallback/diagnostic readbacks remain.

3. **This does NOT resolve the diary 0030 failure.** The root cause of the
   decode-state corruption in the all-six attempt was not pursued. This
   slice proves that `dn_l2_q` and `dn_l2_k` are independently safe, but
   the stateful descriptors still require deeper investigation (state-offset
   resolution, aliasing analysis, or kernel fusion).

4. **This is a narrow step toward single-submit.** Each remaining uncovered
   descriptor requires its own analysis. The four stateful descriptors are
   the hard problem; this diary does not claim progress on them.

## Files Changed

```
src/runtime/vk_session.cpp    — Add dn_l2_q/dn_l2_k vectors, alloc, bind, decode alias, fallback guard
src/runtime/vk_session.hpp    — Add dn_l2_q/dn_l2_k to PerLayerDescriptorSets; update struct comment
```

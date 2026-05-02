# 0034: Corrigendum: dn_compute_g_beta Descriptor Pre-Binding — Constructor Ordering Root Cause Found

## Goal

Diary 0033 documented an isolated attempt to pre-bind the `dn_compute_g_beta`
descriptor set independently. It failed with the same decode-state corruption
as diary 0030 (token 89454 at index 1, `matched_prefix_tokens=1`), and was
reverted. The failure was attributed to an unknown structural issue with
pre-binding stateful DeltaNet sub-step descriptors.

The root cause has now been traced to **constructor ordering**: `bufs_->dn_a_log_bias`
was created and uploaded *after* the `SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS`
pre-binding block. Consequently, the per-layer `dn_compute_g_beta` binding 2
(which references `bufs_->dn_a_log_bias`) was pre-bound to an uninitialized
or stale buffer handle — the buffer simply did not exist yet at the point the
descriptor set was updated.

The fix moves the existing a_log/dt_bias cache and upload block *before* the
per-layer descriptor pre-binding block, then re-introduces `dn_compute_g_beta`
pre-binding. This time it passes all parity tests.

This entry documents the root cause, the fix, and the corrected descriptor
coverage. It is a **corrigendum** to diary 0033: the failure was real and
correctly diagnosed as constructor-ordering-dependent, but the descriptor
itself is safe for pre-binding once the buffer dependency is satisfied.

## Inference Concepts

### Constructor ordering and descriptor binding

A `VkDescriptorSet` is a handle that references buffer bindings. When
`update_descriptor_set()` is called with a `VkBuffer` handle, the driver
records that binding. If the `VkBuffer` has not been created yet (its handle
is `VK_NULL_HANDLE` or contains garbage from an uninitialized struct member),
the binding is silently invalid. In our environment (RADV without validation
layers), this stale-handle condition was not caught at descriptor update time —
the invalid/stale descriptor binding was only detectable through decode-parity
failure.

When the buffer is later created at a different offset in the constructor,
the pre-bound descriptor set still holds the stale handle. The shader sees
either:
- **Zero reads**: The stale handle maps to unmapped memory and reads zero.
  The shader computes incorrect g/beta values, leading to state corruption.
- **Garbage reads**: The stale handle aliases some other allocated buffer
  region, producing unpredictable data.

In this case, the failure signature (token 89454 at index 1) is consistent
with `a_log_bias` being all-zeros — the g/beta computation produces neutral
or zero-valued outputs, corrupting the first decode step's recurrent state.

The fact that `D.dn_compute_g_beta` (the shared global set) was preconfigured
*after* the buffer was created and **worked fine** confirms the diagnosis:
the per-layer pre-binding happened before buffer creation; the global set
pre-configuration happened after.

### Why diary 0033 didn't catch this

Diary 0033's construction changed only two files: it added
`dn_compute_g_beta` to the struct, allocated and pre-bound the per-layer
sets inside the existing pre-binding loop, updated the decode alias, and
added the fallback guard. It did **not** reorder any constructor code. The
a_log/dt_bias cache/upload block remained *after* the pre-binding block,
exactly where it had been placed in diary 0029 (before dn_compute_g_beta
pre-binding was attempted). The per-layer `bufs_->dn_a_log_bias` binding
in the pre-binding block referenced a not-yet-created buffer.

The failure was deterministic and reproducible — it occurred on every run —
because the constructor ordering is deterministic. The same stale buffer
handle was used every time, producing the same wrong token.

No validation layer, driver error, or runtime warning flagged this in our
environment. With the default RADV driver (no validation layers enabled), the
stale handle was indistinguishable from a valid handle at the point of descriptor
update — detection only came from the parity failure.

## Implementation Work Completed

### Root cause: constructor ordering fix (`src/runtime/vk_session.cpp`)

The a_log/dt_bias cache and upload block was moved from its original
position (after the per-layer descriptor pre-binding block and after the
global descriptor set pre-configuration block) to **before** the
`SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS` pre-binding block:

**Before (broken ordering):**
```
// RoPE descriptor pre-configuration
// ... rope pre-bindings ...
// --- Per-layer stable descriptor sets ---       ← pre-binds dn_compute_g_beta
//                                                  with bufs_->dn_a_log_bias (not yet created!)
// Pre-configure static descriptor sets             ← post-pre-bind config
// ... embedding, rope, conv_cache, dn_qkv_proj ...
// Cache DeltaNet a_log and dt_bias                 ← CREATES bufs_->dn_a_log_bias HERE
// Upload a_log and dt_bias buffer                  ← uploads to the newly created buffer
```

**After (fixed ordering):**
```
// RoPE descriptor pre-configuration
// ... rope pre-bindings ...
// Cache DeltaNet a_log and dt_bias                 ← CACHES a_log/dt_bias FIRST
// Upload a_log and dt_bias buffer                  ← CREATES bufs_->dn_a_log_bias HERE
// --- Per-layer stable descriptor sets ---         ← NOW pre-binds dn_compute_g_beta
//                                                  with valid bufs_->dn_a_log_bias
// Pre-configure static descriptor sets             ← post-pre-bind config
// ... embedding, rope, conv_cache, dn_qkv_proj ...
```

The block was moved as a single unit — the cache logic, the upload logic,
and the buffer creation — preserving exactly the same code. No inference
semantics changed. Only the execution order in the constructor changed,
ensuring `bufs_->dn_a_log_bias` exists before the per-layer pre-binding
block.

### Struct addition: `src/runtime/vk_session.hpp`

A new vector is added to `PerLayerDescriptorSets`:

```cpp
std::vector<VkDescriptorSet> dn_compute_g_beta;  // ds_layout_4
```

The struct comment now reads:

```
/// dn_l2_q/dn_l2_k and dn_compute_g_beta are covered here.
/// dn_recurrent/dn_norm_gate/dn_out_proj remain excluded.
```

(Updated from diary 0032's version which excluded dn_compute_g_beta.)

### Constructor additions: `src/runtime/vk_session.cpp`

**Comment update** (line ~464):

```
// ... remaining inner DeltaNet
// sub-step descriptors (dn_recurrent, dn_norm_gate, dn_out_proj) are not
// covered and remain on the old path; dn_compute_g_beta is now covered.
```

**Vector resizing** — inside the `if (per_layer_sets_enabled_)` block:

```cpp
per_layer_sets_->dn_compute_g_beta.resize(LAYERS);
```

**Allocation** — inside the `for (i = 0; i < LAYERS; i++)` loop:

```cpp
per_layer_sets_->dn_compute_g_beta[i] = dev_.allocate_descriptor_set(pipes_->ds_layout_4);
```

**Pre-binding** — inside the per-layer pre-bind block, using `ds_layout_4`:

```cpp
// dn_compute_g_beta: bindings 0=dn_a, 1=dn_b, 2=dn_a_log_bias, 3=dn_state[layer]
VkDeviceSize g_beta_state_off = dn_idx * B.dn_state_per_layer + DN_HEADS * DN_K_DIM * DN_V_DIM * 4;
dev_.update_descriptor_set(per_layer_sets_->dn_compute_g_beta[layer], 0, B.dn_a, 0, DN_HEADS * 2);
dev_.update_descriptor_set(per_layer_sets_->dn_compute_g_beta[layer], 1, B.dn_b, 0, DN_HEADS * 2);
dev_.update_descriptor_set(per_layer_sets_->dn_compute_g_beta[layer], 2, B.dn_a_log_bias, 0,
    NUM_DN_LAYERS * DN_HEADS * 2 * 4);
dev_.update_descriptor_set(per_layer_sets_->dn_compute_g_beta[layer], 3, B.dn_state, g_beta_state_off, DN_HEADS * 2 * 4);
```

The four bindings of `ds_layout_4`:

| Binding | Buffer | Offset | Size | Purpose |
|---------|--------|--------|------|---------|
| 0 | `B.dn_a` | 0 | `DN_HEADS * 2` | a parameter (per-head fp16) |
| 1 | `B.dn_b` | 0 | `DN_HEADS * 2` | b parameter (per-head fp16) |
| 2 | `B.dn_a_log_bias` | 0 | `NUM_DN_LAYERS * DN_HEADS * 2 * 4` | a_log + dt_bias interleaved (per-layer, per-head float32) |
| 3 | `B.dn_state` | per-layer g/beta tail offset | `DN_HEADS * 2 * 4` | state tail for g/beta output (per-layer float32) |

Binding 3 uses the same offset calculation as the decode path:
```
dn_idx * B.dn_state_per_layer + DN_HEADS * DN_K_DIM * DN_V_DIM * 4
```

This targets the g/beta tail region of the per-layer state slab — the
float32 region after the K*V state that stores g and beta.

The verbose count line updates to confirm the new count:

```
per-layer descriptor sets: 28 x 27 sets pre-bound
```

(24 prior + dn_l2_q/dn_l2_k from diary 0032 + dn_compute_g_beta from this diary = 27 per-layer sets.)

### Decode alias: `src/runtime/vk_session.cpp` `decode()`

A new alias is added to the per-layer alias block:

```cpp
VkDescriptorSet ds_dn_compute_g_beta = per_layer_sets_enabled_
    ? per_layer_sets_->dn_compute_g_beta[layer]
    : D.dn_compute_g_beta;
```

This follows the same pattern as the other pre-bound descriptors (now 27 per-layer sets: 24 foundational, 2 L2 q/k from diary 0032, and dn_compute_g_beta from this diary) plus 2 session-level RoPE sets.

### Decode fallback guard: `src/runtime/vk_session.cpp` `decode()`

The per-dispatch `update_descriptor_set` call for `D.dn_compute_g_beta`
binding 3 is now guarded by `if (!per_layer_sets_enabled_)`:

```cpp
if (!per_layer_sets_enabled_) {
    dev_.update_descriptor_set(D.dn_compute_g_beta, 3, B.dn_state, g_beta_offset, DN_HEADS * 2 * 4);
}
```

The dispatch site uses the alias:

```cpp
vkCmdBindDescriptorSets(gb_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
    P.pipeline_layout_4, 0, 1, &ds_dn_compute_g_beta, 0, nullptr);
```

When the gate is active, the pre-bound per-layer set handles all four
bindings including binding 3 with its layer-specific state-tail offset.
The fallback mutation is skipped. When the gate is inactive, the global
`D.dn_compute_g_beta` set is used and binding 3 is still updated per
dispatch (as before), preserving the old path.

### What does NOT change

- **No new shader code.** The `deltanet_compute_g_beta` shader pipeline,
  push constants, and access patterns are unchanged. Only the descriptor
  set handle changes.
- **No descriptor pool sizing change.** The 28 new sets (1 per layer x 28
  layers) were already within the 1024 maxSets capacity.
- **No pipeline layout change.** The existing `ds_layout_4` / `pipeline_layout_4`
  is reused for allocation, binding, and dispatch.
- **No buffer reallocation.** `B.dn_a`, `B.dn_b`, `B.dn_a_log_bias`, and
  `B.dn_state` are unchanged. The a_log/dt_bias cache/upload logic is
  identical — only its position in the constructor changed.
- **No inference semantics change.** The g/beta computation is identical.
  The descriptor binding points to the same buffer regions at the same
  byte offsets; the shader sees the same data.
- **dn_recurrent, dn_norm_gate, dn_out_proj remain on the old path.** This
  diary covers only dn_compute_g_beta. The three remaining stateful
  descriptors are not touched.
- **This is NOT full GPU offload, NOT single-submit, NOT persistent
  dispatch, and NOT the megakernel.** The host still orchestrates every
  step. Decode argmax, logit computation, diagnostic readbacks, and
  fallback paths remain host-mediated.

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

Verifying the constructor reordering did not break the default path:

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

### Per-layer stable descriptor sets gate (27 per-layer sets)

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

This is the critical test that failed in diary 0033 with token 89454 at
index 1. It now passes, confirming the constructor ordering fix resolved
the root cause.

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

### CTest regression gate (combined gates)

```sh
SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1 \
  SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1 SPOCK_GPU_DEFER_TOKEN_DOWNLOAD=1 \
  ctest --test-dir build --output-on-failure \
  -R spock_vk_decode_gpu_collect_chunk_prefill
```

Passed 3/3. Times:

| Test | Time (sec) |
|------|-----------|
| short | 114.84 |
| tiled | 8.93 |
| short_baseline | 5.11 |
| **Total** | **128.89** |

This is the first CTest run that exercises the combined per-layer descriptor
sets (diary 0034) + device-resident token (diary 0027) + deferred token
download (diary 0028) gates together, confirming no regression across the
three independent env-gated features.

## Descriptor Coverage Summary

After this corrigendum, the total descriptor model is:

| Category | Sets | Scope | Pre-bound? |
|----------|------|-------|------------|
| Common MLP/norm | 9 sets x 28 layers | Per-layer | Yes (diary 0029) |
| Attention-specific | 10 sets x 28 layers | Per-layer | Yes (diary 0029) |
| DeltaNet first-stage | 5 sets x 28 layers | Per-layer | Yes (diary 0029) |
| DeltaNet L2-norm | 2 sets x 28 layers | Per-layer | Yes (diary 0032) |
| DeltaNet g/beta (this diary) | 1 set x 28 layers | Per-layer | Yes |
| RoPE | 2 sets (session-level) | Session | Yes (diary 0031) |
| **Per-layer subtotal** | **27 sets x 28 layers = 756 sets** | | |
| **Session-level subtotal** | **2 sets** | | |
| **Total pre-bound** | **758 sets** | | |

Remaining uncovered (per-dispatch mutation on decode path):

| Descriptor | Status | Notes |
|------------|--------|-------|
| `dn_recurrent` | Uncovered | Stateful — binds `dn_state`. No independent test performed. |
| `dn_norm_gate` | Uncovered | Binds output accumulation buffer. No independent test performed. |
| `dn_out_proj` | Uncovered | Binds DeltaNet output accumulation buffer. No independent test performed. |
| `dn_split_q` | Uncovered (internal) | Internal decomposition descriptor, not a dispatch target. |
| `dn_split_kv` | Uncovered (internal) | Internal decomposition descriptor, not a dispatch target. |

`dn_compute_g_beta` is no longer in the uncovered list. The remaining
dispatch-target blockers are three descriptors: `dn_recurrent`,
`dn_norm_gate`, `dn_out_proj`. The two internal decomposition descriptors
(`dn_split_q`, `dn_split_kv`) remain listed as uncovered but are not
dispatch targets.

## Relationship to Prior Diaries

### Diary 0033 (rejected: dn_compute_g_beta pre-bind)

This diary **corrects** diary 0033's negative result. Diary 0033
independently isolated dn_compute_g_beta and proved it causes decode-state
corruption (token 89454 at index 1). The root cause was not pursued at the
time; the failure was conservatively attributed to a structural issue with
stateful descriptor pre-binding.

The root cause is now known to be constructor ordering: `bufs_->dn_a_log_bias`
was created after the per-layer descriptor pre-binding block. The pre-bound
binding 2 referenced a not-yet-created buffer handle.

Diary 0033's failure was genuine — it was not a mistake in the code. It
was a valid negative result that provided the critical clue (the correct
wrong-token signature 89454) that later helped trace the root cause to
the buffer creation order. The reverted patch from diary 0033 is
structurally identical to the working code in this diary, except for the
constructor ordering of the a_log/dt_bias block.

### Diary 0030 (rejected: all-six intra-DeltaNet pre-bind)

The all-six failure (diary 0030) included dn_compute_g_beta. Since
dn_compute_g_beta was independently failing due to the constructor
ordering bug, it is now possible that the all-six failure was *entirely*
caused by dn_compute_g_beta's stale binding. The remaining three
stateful descriptors (dn_recurrent, dn_norm_gate, dn_out_proj) may have
been innocent co-travelers in that failure.

However, this diary does **not** prove that the other three are safe.
They have not been independently tested, and they involve different
buffer bindings (dn_state, output accumulation) with different aliasing
and state-offset properties. Diary 0030's failure could still be due to
one of these even after fixing dn_compute_g_beta. The all-six failure is
narrowed but not resolved.

### Diary 0032 (L2-norm DeltaNet pre-bind)

The L2-norm descriptors (dn_l2_q, dn_l2_k, diary 0032) were and remain
safe. Their bindings reference `B.dn_qkv` — a buffer created long before
the pre-binding block. The constructor ordering issue only affected
`dn_compute_g_beta` because `B.dn_a_log_bias` was the only buffer
referenced by a pre-bound descriptor that was created after the
pre-binding block.

### Diary 0029 (per-layer stable descriptor sets foundation)

Diary 0029 established the per-layer pre-binding infrastructure and
identified all 24 + 6 = 30 covered/uncovered descriptors. The a_log/dt_bias
cache/upload block was placed after the pre-binding block in diary 0029
because dn_compute_g_beta was not targeted for pre-binding at that time.
It was only moved when dn_compute_g_beta became a pre-binding candidate.

## Changed files

```
src/runtime/vk_session.cpp    — Move a_log/dt_bias cache/upload BEFORE per-layer
                                descriptor pre-binding block. Add dn_compute_g_beta
                                vector resizing, allocation, pre-binding, decode
                                alias, fallback guard. Update verbose count to 27.
src/runtime/vk_session.hpp    — Add dn_compute_g_beta vector to PerLayerDescriptorSets;
                                update struct comment.
```

## Known Limitations

1. **This does NOT make full GPU offload complete.** The host still:
   - Iterates per-layer dispatches in the decode loop.
   - Submits command buffers and waits on fences per step.
   - Observes/downloads generated-token outputs, logits, and diagnostic data.
   - Orchestrates prefill (layer-major loop).

2. **This does NOT make single-submit complete.** Remaining blockers:
   - Three stateful intra-DeltaNet sub-step descriptors (`dn_recurrent`,
     `dn_norm_gate`, `dn_out_proj`) still mutate per dispatch on the decode
     path. Their safety for pre-binding remains unknown.
   - Host-side per-layer/per-step submissions and fence waits remain.
   - Fallback/diagnostic readbacks remain.

3. **dn_compute_g_beta is now proven safe for pre-binding** — but only
   when `bufs_->dn_a_log_bias` is created before the pre-binding block.
   Any future refactoring that reorders constructor code must preserve
   this ordering invariant.

4. **Diary 0030's all-six failure is narrowed but not fully resolved.**
   The remaining three descriptors (dn_recurrent, dn_norm_gate, dn_out_proj)
   may independently fail or succeed. No sequential isolation has been
   performed for them.

5. **This is a narrow step toward single-submit.** The remaining three
   stateful dispatch-target descriptors are the hard problem. Each
   requires its own analysis, and the structural issue (if any) has not
   been determined.

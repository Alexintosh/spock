# 0035: dn_recurrent Descriptor Pre-Binding — Final Dispatch-Target Blocker Narrowed

## Goal

Extend the opt-in per-layer stable descriptor set path to cover the
`dn_recurrent` descriptor — the DeltaNet recurrent kernel dispatch target
that binds the per-layer `dn_state` buffer. This moves the 28th per-layer
descriptor set from per-dispatch mutation to pre-bound, reducing the
remaining dispatch-target blockers from three to two (`dn_norm_gate`,
`dn_out_proj`).

This diary independently isolates `dn_recurrent` from the other stateful
descriptors that failed in diary 0030's all-six attempt. Unlike the
constructor-ordering issue that caused diary 0033's failure for
`dn_compute_g_beta` (corrected in diary 0034), `dn_recurrent` involves
different buffer binding semantics — it directly binds the recurrent
state buffer (`B.dn_state`) with a per-layer offset, not a scratch or
intermediate buffer. Its independent success narrows the remaining
corruption hypothesis from diaries 0030/0033.

## Inference Concepts

### What dn_recurrent's descriptor bindings represent

The `ds_layout_3` descriptor set for `dn_recurrent` has three bindings:

| Binding | Buffer | Offset | Size | Purpose |
|---------|--------|--------|------|---------|
| 0 | `B.dn_qkv` | 0 (Q region) | `DN_KEY_TOTAL * 2` | Q input — the normalized query vector (fp16) |
| 1 | `B.dn_qkv` | `DN_KEY_TOTAL * 2` | `(DN_KEY_TOTAL + DN_VAL_TOTAL) * 2` | KV input — the key and value vectors (fp16, contiguous region) |
| 2 | `B.dn_state` | Per-layer recurrent state slab | `dn_state_per_layer` | Recurrent state — the per-head K*V outer-product accumulator (fp32) |

Binding 2 is the critical one: it points to a per-layer slice of the
multi-layer `B.dn_state` buffer. Each DeltaNet layer has its own state
slab (`dn_state_per_layer` bytes), and the pre-bound set captures this
layer-specific offset at construction time. The binding is **handle-and-offset
stable** — `B.dn_state` does not change between decode steps, and the per-layer
offset is a constant derived from the DN layer index within the 28-layer
schedule.

### Why this was thought risky (diary 0030 context)

Diary 0030's all-six pre-binding failure was initially attributed to *any* of
the four stateful/recurrent descriptors (dn_recurrent, dn_norm_gate,
dn_out_proj, dn_compute_g_beta). Diary 0034 proved that `dn_compute_g_beta`
failed solely due to constructor ordering — `B.dn_a_log_bias` was created
after the pre-binding block. That removed one variable from diary 0030's
corruption equation.

`dn_recurrent` remained suspect because:
- It binds `B.dn_state`, which has a per-layer offset into a multi-layer
  buffer — if the offset calculation were wrong, the shader would read
  stale state from another layer.
- `dn_state` is written by the `deltanet_recurrent.comp` shader — the
  output is consumed by subsequent sub-steps (dn_norm_gate, dn_out_proj)
  in the same decode loop iteration. If the descriptor aliased the wrong
  region, the corruption would be immediate.

This diary proves that `dn_recurrent`'s offset calculation is correct and
the descriptor is safe for pre-binding.

## Implementation Work Completed

### Struct additions: `src/runtime/vk_session.hpp`

`PerLayerDescriptorSets` now includes a `dn_recurrent` vector (added in
this slice). The struct comment is updated to reflect the new coverage:

```
/// dn_l2_q/dn_l2_k, dn_compute_g_beta, and dn_recurrent are covered here.
/// dn_norm_gate and dn_out_proj remain excluded.
```

### Construction: `src/runtime/vk_session.cpp`

**Comment update** (line ~467):

The comment block is updated to include `dn_recurrent` in the covered
list:
```
// L2-norm q/k (dn_l2_q, dn_l2_k) and dn_recurrent are covered here;
// remaining inner DeltaNet sub-step descriptors (dn_norm_gate, dn_out_proj) are not
// covered and remain on the old path; dn_compute_g_beta is now covered.
```

**Vector resizing** — `per_layer_sets_->dn_recurrent.resize(LAYERS)` is
added in the allocation loop.

**Allocation** — `per_layer_sets_->dn_recurrent[i] = alloc3()` is added
in the per-layer allocation loop.

**Pre-binding** — The pre-binding block now contains a `dn_recurrent`
section inside `if (per_layer_sets_enabled_)`:
```
// dn_recurrent: bindings 0=Q(dn_qkv), 1=KV(dn_qkv), 2=dn_state[layer]
VkDeviceSize rec_state_off = dn_idx * B.dn_state_per_layer;
dev_.update_descriptor_set(per_layer_sets_->dn_recurrent[layer], 0, B.dn_qkv, 0, DN_KEY_TOTAL * 2);
dev_.update_descriptor_set(per_layer_sets_->dn_recurrent[layer], 1, B.dn_qkv, DN_KEY_TOTAL * 2,
    (DN_KEY_TOTAL + DN_VAL_TOTAL) * 2);
dev_.update_descriptor_set(per_layer_sets_->dn_recurrent[layer], 2, B.dn_state, rec_state_off);
```

**Verbose count update** — The verbose line at construction now reports:
```
per-layer descriptor sets: 28 x 28 sets pre-bound
```

(Updated from 27 in diary 0034, adding dn_recurrent.)

### Decode alias: `src/runtime/vk_session.cpp` `decode()`

The per-layer alias for `dn_recurrent` is added:

```cpp
VkDescriptorSet ds_dn_recurrent = per_layer_sets_enabled_
    ? per_layer_sets_->dn_recurrent[layer]
    : D.dn_recurrent;
```

### Decode fallback guard: `src/runtime/vk_session.cpp` `decode()`

The per-dispatch `update_descriptor_set` calls for `D.dn_recurrent` are
guarded by `if (!per_layer_sets_enabled_)`:

```cpp
if (!per_layer_sets_enabled_) {
    dev_.update_descriptor_set(D.dn_recurrent, 2, B.dn_state, rec_state_off);
}
```

The dispatch site uses the alias:

```cpp
vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
    P.pipeline_layout_32, 0, 1, &ds_dn_recurrent, 0, nullptr);
```

When the per-layer set is active, the pre-bound per-layer set handles all three
bindings including binding 2 with its layer-specific state offset. The
fallback mutation is skipped.

### What does NOT change

- **No new shader code.** The `deltanet_recurrent.comp` shader pipeline,
  push constants, and access patterns are unchanged. The descriptor
  binding pattern (3 bindings: Q, KV, dn_state) is identical — only the
  descriptor set handle changes.
- **No descriptor pool sizing change.** The existing capacity (1024 maxSets,
  4096 storage buffer slots) from diary 0029 is sufficient.
- **No pipeline layout change.** Descriptor allocation uses `ds_layout_3`
  and dispatch uses `pipeline_layout_32`.
- **No buffer reallocation.** `B.dn_qkv`, `B.dn_state`, and `dn_state_per_layer`
  are unchanged. The per-layer state offset calculation is identical to the
  legacy path.
- **No inference semantics change.** The recurrent kernel computation is
  identical. The descriptor binding points to the same buffer regions at
  the same byte offsets; the shader sees the same data.
- **dn_norm_gate and dn_out_proj remain on the old path.** This diary
  covers only dn_recurrent. Two remaining dispatch-target descriptors
  are not touched.
- **This is NOT full GPU offload, NOT single-submit, NOT persistent
  dispatch, and NOT the megakernel.** The host still orchestrates every
  step (per-layer iteration, submission, fence wait). Argmax, logit
  computation, diagnostic readbacks, and fallback paths remain
  host-mediated.

## Verification

All commands run on the target RADV RX 6750 XT (NAVI22) hardware. Test
evidence provided by the implementation agent.

### Whitespace and build

```sh
git diff --check
```

No whitespace errors.

```sh
cmake --build build -j
```

Passed cleanly. C++ recompilation completes without warnings.

### Baseline decode parity (default path, no env vars)

Verifying the default path is not regressed:

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

### Per-layer stable descriptor sets gate (28 per-layer sets)

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

This is the critical test: the full gated path with 28 pre-bound per-layer
sets, including the newly covered dn_recurrent descriptor. Passes parity.

### Multiple prompts with per-layer gate

```sh
SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1 \
  python3 tests/run_vk_decode_parity.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --reference tests/data/reference_tokens.jsonl \
  --ids short_correctness_001,mixed_correctness_023,pp520_046 \
  --max-new-tokens 4
```

```json
{"status":"ok","checked":3,"failures":[]}
```

All three prompts pass at `--max-new-tokens 4`, covering diverse prompt
lengths and structures.

### Chunk prefill regression coverage (per-layer gate + device-resident token)

Runs the chunk-prefill CTest suite with all three relevant env vars active:

- `SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1` — the per-layer descriptor gate
  this diary adds dn_recurrent coverage to.
- `SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1` — keeps the full token buffer on
  device during prefill, exercising the prefill path's token binding logic.
- `SPOCK_GPU_DEFER_TOKEN_DOWNLOAD=1` — avoids a synchronous token download
  at the end of prefill, exercising the deferred download path that
  interacts with descriptor lifetime.

This combination is relevant because the `dn_recurrent` alias/guard in
`decode()` is mirrored in the prefill path (`layer_major_prefill()`).
Prefill iterates layers just like decode, and the same `ds_dn_recurrent`
alias logic applies. A regression in the dn_recurrent pre-binding —
such as an incorrect per-layer offset into `B.dn_state`, or a descriptor
set handle leaking across layers — would manifest here.

```sh
SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1 \
  SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1 \
  SPOCK_GPU_DEFER_TOKEN_DOWNLOAD=1 \
  ctest --test-dir build --output-on-failure \
    -R spock_vk_decode_gpu_collect_chunk_prefill
```

**Results:** 3/3 tests passed.

| Test | Time |
|------|------|
| `spock_vk_decode_gpu_collect_chunk_prefill_short` | 114.96 s |
| `spock_vk_decode_gpu_collect_chunk_prefill_tiled` | 8.98 s |
| `spock_vk_decode_gpu_collect_chunk_prefill_short_baseline` | 5.11 s |
| **Total real time** | **129.05 s** |

All three CTest subtests pass. The 114-second `_short` variant exercises the
full chunked-prefill pipeline (token embedding, layer-major iteration,
prefill kernel launches) with per-layer descriptor sets active. The `_tiled`
and `_short_baseline` variants cover alternative prefill paths. No crashes,
Vulkan validation errors, or token mismatches.

## Descriptor Coverage Summary

After this diary, the total descriptor model is:

| Category | Sets | Scope | Pre-bound? |
|----------|------|-------|------------|
| Common MLP/norm | 9 sets x 28 layers | Per-layer | Yes (diary 0029) |
| Attention-specific | 10 sets x 28 layers | Per-layer | Yes (diary 0029) |
| DeltaNet first-stage | 5 sets x 28 layers | Per-layer | Yes (diary 0029) |
| DeltaNet L2-norm | 2 sets x 28 layers | Per-layer | Yes (diary 0032) |
| DeltaNet g/beta | 1 set x 28 layers | Per-layer | Yes (diary 0034) |
| DeltaNet recurrent (this diary) | 1 set x 28 layers | Per-layer | Yes |
| RoPE | 2 sets (session-level) | Session | Yes (diary 0031) |
| **Per-layer subtotal** | **28 sets x 28 layers = 784 sets** | | |
| **Session-level subtotal** | **2 sets** | | |
| **Total pre-bound** | **786 sets** | | |

Remaining uncovered (per-dispatch mutation on decode path):

| Descriptor | Status | Notes |
|------------|--------|-------|
| `dn_norm_gate` | Uncovered (dispatch target) | Binds output accumulation buffer. No independent test performed. |
| `dn_out_proj` | Uncovered (dispatch target) | Binds DeltaNet output accumulation buffer. No independent test performed. |
| `dn_split_q` | Uncovered (internal) | Internal decomposition descriptor, not a dispatch target. |
| `dn_split_kv` | Uncovered (internal) | Internal decomposition descriptor, not a dispatch target. |

`dn_recurrent` is no longer in the uncovered list. The remaining
dispatch-target blockers are two descriptors: `dn_norm_gate` and
`dn_out_proj`. The two internal decomposition descriptors (`dn_split_q`,
`dn_split_kv`) remain listed as uncovered but are not dispatch targets.

## Relationship to Prior Diaries

### Diary 0030 (rejected: all-six intra-DeltaNet pre-bind)

Diary 0030's all-six failure included dn_recurrent alongside the other five
intra-DeltaNet sub-step descriptors. The all-six failure was grouped —
without individual isolation, any of the six could have been the cause.

With `dn_compute_g_beta` independently corrected (diary 0034) and now
`dn_recurrent` independently verified, the all-six failure is further
narrowed. The remaining two dispatch-target descriptors (`dn_norm_gate`,
`dn_out_proj`) are the untested subset from diary 0030's failure set.

It is now possible that diary 0030's failure was caused by *either*
`dn_compute_g_beta` (constructor ordering, proven in diary 0034) *or*
`dn_norm_gate`/`dn_out_proj`, or a combination. `dn_recurrent` was
definitively an innocent co-traveler in that failure. `dn_l2_q` and
`dn_l2_k` were already proven innocent in diary 0032.

### Diary 0033/0034 (dn_compute_g_beta isolation and correction)

Diary 0033 independently isolated `dn_compute_g_beta` and proved it
fails with the same corruption signature. Diary 0034 traced the root
cause to constructor ordering (`B.dn_a_log_bias` created after pre-binding
block) and corrected it. With both `dn_compute_g_beta` and `dn_recurrent`
now independently verified, the set of "proven safe for pre-binding among
the original six" grows to four descriptors (dn_l2_q, dn_l2_k, dn_compute_g_beta,
dn_recurrent).

### Diary 0032 (L2-norm DeltaNet pre-bind)

The L2-norm descriptors (dn_l2_q, dn_l2_k) were and remain safe, as proven
in diary 0032. This diary does not change their status.

## Files Changed

```
src/runtime/vk_session.cpp    — Add dn_recurrent vector resize, per-layer
                                allocation, pre-binding block,
                                layer_major_prefill alias/guard, decode
                                alias/guard, and update verbose count from
                                27 to 28.
src/runtime/vk_session.hpp    — Add dn_recurrent vector declaration and
                                update struct comments.
```

## Known Limitations

1. **This does NOT make full GPU offload complete.** The host still:
   - Iterates per-layer dispatches in the decode loop.
   - Submits command buffers and waits on fences per step.
   - Observes/downloads generated-token outputs, logits, and diagnostic data
     for external output, parity checking, and fallback/diagnostic paths.
   - Orchestrates prefill (layer-major loop).

2. **This does NOT make single-submit complete.** Remaining blockers:
   - Two stateful intra-DeltaNet sub-step descriptors (`dn_norm_gate`,
     `dn_out_proj`) still mutate per dispatch on the decode path. They are
     the last dispatch-target descriptors not yet proven safe for pre-binding.
   - Host-side per-layer/per-step submissions and fence waits remain.
   - Fallback/diagnostic readbacks remain.

3. **Diary 0030's all-six failure is further narrowed but not fully resolved.**
   With four of six descriptors now independently verified (`dn_l2_q`,
   `dn_l2_k`, `dn_compute_g_beta`, `dn_recurrent`), the remaining
   untested subset is `dn_norm_gate` and `dn_out_proj`. Either or both
   may independently fail.

4. **This is a narrow step toward single-submit.** Two remaining stateful
   dispatch-target descriptors are the hard problem. Each requires its own
   independent isolation test.

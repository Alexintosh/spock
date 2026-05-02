# 0037: dn_out_proj Descriptor Pre-Binding — Final Dispatch-Target Blocker Eliminated

## Goal

Extend the opt-in per-layer stable descriptor set path to cover the
`dn_out_proj` descriptor — the last DeltaNet dispatch-target kernel that
applies the delta_out_proj weight projection to produce the final per-head
output after the norm/gate step. This moves the 30th per-layer descriptor
set from per-dispatch mutation to pre-bound, eliminating the last tracked
dispatch-target descriptor mutation under `SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1`.

This diary independently isolates `dn_out_proj` — the sole remaining
untested descriptor from diary 0030's all-six failure set. All six
intra-DeltaNet sub-step descriptors (`dn_l2_q`, `dn_l2_k`,
`dn_compute_g_beta`, `dn_recurrent`, `dn_norm_gate`, `dn_out_proj`) are
now independently verified safe for pre-binding. The decode path no
longer has any dispatch-target descriptor that must mutate per dispatch
when the per-layer descriptor gate is active.

## Inference Concepts

### What dn_out_proj's descriptor bindings represent

The `ds_layout_3` descriptor set for `dn_out_proj` has three bindings:

| Binding | Buffer | Offset | Size | Purpose |
|---------|--------|--------|------|---------|
| 0 | `B.weights` | Per-layer `delta_out_proj` weight offset | Per-layer `delta_out_proj` weight size | Weight buffer — the output projection weight row for this layer (fp16) |
| 1 | `B.dn_qkv` | `(DN_KEY_TOTAL + DN_KEY_TOTAL) * 2` (V section) | `DN_VAL_TOTAL * 2` | IO buffer — the per-head V-space output from the norm/gate step (fp16); read then overwritten with projected output |
| 2 | `B.act_b` | 0 | full buffer | Output buffer — the per-layer activation result written by `dn_out_proj` |

Binding 0 is the per-layer weight slice. The weight lookup uses the
artifact's `delta_out_proj` role:
`artifact_.find_by_role("layer.${layer}.delta_out_proj")` — the same
lookup pattern used by all other weight-bound per-layer descriptors.

Binding 1 is the V section of the shared `B.dn_qkv` buffer — the same
region used by `dn_norm_gate` and `dn_recurrent`. The offset
`(DN_KEY_TOTAL + DN_KEY_TOTAL) * 2` skips the Q and K regions to reach
the V-space accumulator. The range `DN_VAL_TOTAL * 2` covers all heads'
V-space values in fp16. The `dn_out_proj` kernel reads the gated V-space
output (written by `dn_norm_gate`) and projects it through the per-layer
`delta_out_proj` weight, writing the result to `B.act_b` via binding 2.

Binding 2 is the full `B.act_b` activation buffer — no per-layer offset.
This is a fixed, session-level buffer reference identical to how
`dn_out_proj` binds it on the legacy mutation path.

### Why this was the final untested descriptor

After diaries 0032–0036, five of the six intra-DeltaNet sub-step
descriptors had been independently verified safe for pre-binding:

- `dn_l2_q` (diary 0032)
- `dn_l2_k` (diary 0032)
- `dn_compute_g_beta` (diary 0034, after constructor-ordering correction)
- `dn_recurrent` (diary 0035)
- `dn_norm_gate` (diary 0036)

`dn_out_proj` was the sole remaining untested descriptor from diary
0030's all-six failure. Its independent isolation was the final open
question from that failure analysis. With all six descriptors now
individually verified, diary 0030's grouped failure can be attributed
to the combined effect of the constructor-ordering bug in
`dn_compute_g_beta` (diary 0034) — with the remaining descriptors
being proven safe individually after being grouped in diary 0030 failure — they happened to be entangled in the same
batch.

## Implementation Work Completed

### Struct additions: `src/runtime/vk_session.hpp`

`PerLayerDescriptorSets` now includes a `dn_out_proj` vector. The struct
comment is updated to reflect full coverage of all DeltaNet dispatch-target
sub-step descriptors:

```
/// dn_l2_q/dn_l2_k, dn_compute_g_beta, dn_recurrent, dn_norm_gate, and dn_out_proj
/// are covered here.
```

The `dn_out_proj` vector declaration is added to the struct, allocated
from `ds_layout_3` (same pool slot as the other pre-bound per-layer sets):

```cpp
std::vector<VkDescriptorSet> dn_out_proj;         // ds_layout_3
```

### Construction: `src/runtime/vk_session.cpp`

**Comment update** (line ~461):

The comment block is updated to include `dn_out_proj` in the covered list
and to note that all DeltaNet dispatch-target sub-step descriptors are now
covered:

```
// L2-norm q/k (dn_l2_q, dn_l2_k), dn_recurrent, dn_norm_gate, and dn_out_proj are covered here;
// All DeltaNet dispatch-target sub-step descriptors are covered under the per-layer descriptor gate;
// dn_split_q and dn_split_kv remain internal decomposition descriptors, not part of this per-dispatch-target list.
```

**Vector resizing** — `per_layer_sets_->dn_out_proj.resize(LAYERS)` is
added in the allocation loop.

**Allocation** — `per_layer_sets_->dn_out_proj[i] = alloc3()` is added
in the per-layer allocation loop.

**Weight lookup** — The `dn_out_w` lookup is added alongside the
existing weight assertions:

```cpp
auto dn_out_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".delta_out_proj");
assert(dn_qkv_w && dn_z_w && dn_a_w && dn_b_w && dn_conv_w && dn_norm_w && dn_out_w);
```

**Pre-binding** — The pre-binding block now contains a `dn_out_proj`
section:

```cpp
// dn_out_proj: bindings 0=weight(delta_out_proj), 1=dn_qkv(V section), 2=act_b
dev_.update_descriptor_set(per_layer_sets_->dn_out_proj[layer], 0, B.weights, dn_out_w->offset, dn_out_w->nbytes);
dev_.update_descriptor_set(per_layer_sets_->dn_out_proj[layer], 1, B.dn_qkv, (DN_KEY_TOTAL + DN_KEY_TOTAL) * 2, DN_VAL_TOTAL * 2);
dev_.update_descriptor_set(per_layer_sets_->dn_out_proj[layer], 2, B.act_b);
```

**Verbose count update** — The verbose line at construction now reports:

```
per-layer descriptor sets: 28 x 30 sets pre-bound
```

(Updated from 29 in diary 0036, adding dn_out_proj.)

### Prefill alias: `src/runtime/vk_session.cpp` `layer_major_prefill()`

The per-layer alias for `dn_out_proj` is added:

```cpp
VkDescriptorSet ds_dn_out_proj = per_layer_sets_enabled_
    ? per_layer_sets_->dn_out_proj[layer]
    : D.dn_out_proj;
```

### Prefill guard: `src/runtime/vk_session.cpp` `layer_major_prefill()`

The per-dispatch `update_descriptor_set` calls for `D.dn_out_proj` are
guarded by `if (!per_layer_sets_enabled_)`:

```cpp
if (!per_layer_sets_enabled_) {
    dev_.update_descriptor_set(D.dn_out_proj, 0, B.weights, dn_out_w->offset, dn_out_w->nbytes);
    dev_.update_descriptor_set(D.dn_out_proj, 1, B.dn_qkv, (DN_KEY_TOTAL + DN_KEY_TOTAL) * 2, DN_VAL_TOTAL * 2);
    dev_.update_descriptor_set(D.dn_out_proj, 2, B.act_b);
}
```

The dispatch site passes the alias:

```cpp
vkCmdBindPipeline(rec_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.deltanet_out_proj);
vkCmdBindDescriptorSets(rec_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
    P.pipeline_layout_3, 0, 1, &ds_dn_out_proj, 0, nullptr);
```

### Decode alias + guard: `src/runtime/vk_session.cpp` `decode()`

The same alias and guarded mutation pattern is applied in the decode loop.
One dispatch site uses the `ds_dn_out_proj` alias: the **single-step
decode submit**. It skips the mutation and selects the pre-bound per-layer
set when the gate is active.

### correct_last_token_hidden alias + guard: `src/runtime/vk_session.cpp` `correct_last_token_hidden()`

A third dispatch site for `dn_out_proj` is inside `correct_last_token_hidden()`:
the **grouped out_proj/residual/MLP submit**. This site follows the same
alias and guarded mutation pattern. It skips the mutation and selects the
pre-bound per-layer set when the gate is active.

### What does NOT change

- **No new shader code.** The `deltanet_out_proj.comp` shader pipeline,
  push constants, and access patterns are unchanged. The descriptor
  binding pattern (3 bindings: delta_out_proj weight, dn_qkv V section,
  act_b output) is identical — only the descriptor set handle changes.
- **No descriptor pool sizing change.** The existing capacity (1024 maxSets,
  4096 storage buffer slots) from diary 0029 is sufficient.
- **No pipeline layout change.** Descriptor allocation uses `ds_layout_3`
  and dispatch uses `pipeline_layout_3`.
- **No buffer reallocation.** `B.weights`, `B.dn_qkv`, `B.act_b`, and all
  weight regions are unchanged. The per-layer weight offset/size
  calculation is identical to the legacy path.
- **No inference semantics change.** The output projection kernel
  computation is identical. The descriptor binding points to the same
  buffer regions at the same byte offsets; the shader sees the same data.
- **dn_split_q and dn_split_kv remain uncovered.** These are internal
  decomposition descriptors (not dispatch targets) and are not part of
  the per-dispatch-target descriptor work. They are listed for completeness
  but are not blockers.
- **This is NOT full GPU offload, NOT single-submit, NOT persistent
  dispatch, and NOT the megakernel.** The host still orchestrates every
  step (per-layer iteration, submission, fence wait). Argmax, logit
  computation, diagnostic readbacks, and fallback paths remain
  host-mediated.

## Verification

All commands run on the target RADV RX 6750 XT (NAVI22) hardware.

### Whitespace and build

```
git diff --check
```

No whitespace errors.

```
cmake --build build -j
```

Passed cleanly. C++ recompilation completes without warnings.

### Baseline decode parity (default path, no env vars)

Verifying the default path is not regressed:

```
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

### Per-layer descriptor sets gate (30 per-layer sets)

```
SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1 \
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

This is the critical test: the full gated path with 30 pre-bound per-layer
sets, including the newly covered dn_out_proj descriptor. Passes parity.

### Combined gate suite (per-layer sets + GPU chunk-prefill + device-resident token + deferred download)

Six env gates exercised simultaneously:

- `SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1` — per-layer descriptor pre-binding
  with the newly covered dn_out_proj.
- `SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1` — device-resident decode token embedding.
- `SPOCK_GPU_DEFER_TOKEN_DOWNLOAD=1` — deferred batch token download.
- `SPOCK_GPU_CHUNK_PREFILL=1` — GPU chunk-prefill path.
- `SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1` — GPU-collected chunk inputs.
- `SPOCK_GPU_CHUNK_PREFILL_TILED=1` — tiled single-dispatch shader.

Single-prompt decode parity:

```
SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1 \
SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1 \
SPOCK_GPU_DEFER_TOKEN_DOWNLOAD=1 \
SPOCK_GPU_CHUNK_PREFILL=1 \
SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1 \
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

Multi-prompt decode parity (two diverse prompts):

```
SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1 \
SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1 \
SPOCK_GPU_DEFER_TOKEN_DOWNLOAD=1 \
SPOCK_GPU_CHUNK_PREFILL=1 \
SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1 \
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

### CTest regression suite (per-layer sets + device-resident token + deferred download)

Combined gate CTest for the GPU-collected chunk-prefill path:

```
SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1 \
SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1 \
SPOCK_GPU_DEFER_TOKEN_DOWNLOAD=1 \
  ctest --test-dir build --output-on-failure \
    -R spock_vk_decode_gpu_collect_chunk_prefill
```

**Results:** 3/3 tests passed.

| Test | Time |
|------|------|
| `spock_vk_decode_gpu_collect_chunk_prefill_short` | 114.89 s |
| `spock_vk_decode_gpu_collect_chunk_prefill_tiled` | 8.96 s |
| `spock_vk_decode_gpu_collect_chunk_prefill_short_baseline` | 5.10 s |
| **Total real time** | **128.96 s** |

All three CTest subtests pass. The 114-second `_short` variant exercises
the full chunked-prefill pipeline with per-layer descriptor sets active.
The `_tiled` and `_short_baseline` variants cover alternative prefill
paths. No crashes, Vulkan validation errors, or token mismatches.

## Descriptor Coverage Summary

After this diary, the total descriptor model is:

| Category | Sets | Scope | Pre-bound? |
|----------|------|-------|------------|
| Common MLP/norm | 9 sets x 28 layers | Per-layer | Yes (diary 0029) |
| Attention-specific | 10 sets x 28 layers | Per-layer | Yes (diary 0029) |
| DeltaNet first-stage | 5 sets x 28 layers | Per-layer | Yes (diary 0029) |
| DeltaNet L2-norm | 2 sets x 28 layers | Per-layer | Yes (diary 0032) |
| DeltaNet g/beta | 1 set x 28 layers | Per-layer | Yes (diary 0034) |
| DeltaNet recurrent | 1 set x 28 layers | Per-layer | Yes (diary 0035) |
| DeltaNet norm_gate | 1 set x 28 layers | Per-layer | Yes (diary 0036) |
| DeltaNet out_proj (this diary) | 1 set x 28 layers | Per-layer | Yes |
| RoPE | 2 sets (session-level) | Session | Yes (diary 0031) |
| **Per-layer subtotal** | **30 sets x 28 layers = 840 sets** | | |
| **Session-level subtotal** | **2 sets** | | |
| **Total pre-bound** | **842 sets** | | |

Remaining uncovered (per-dispatch mutation on decode path):

| Descriptor | Status | Notes |
|------------|--------|-------|
| `dn_split_q` | Uncovered (internal) | Internal decomposition descriptor, not a dispatch target. |
| `dn_split_kv` | Uncovered (internal) | Internal decomposition descriptor, not a dispatch target. |

`dn_out_proj` is no longer in the uncovered list. With this diary,
**all decode dispatch-target descriptor mutations tracked by the per-layer
descriptor work are eliminated** under `SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1`.
The two remaining uncovered descriptors (`dn_split_q`, `dn_split_kv`) are
internal decomposition descriptors — not dispatch targets — and have never
been part of this work's scope.

This is a milestone: the per-layer stable descriptor set path can now be
said to cover every descriptor that is bound per dispatch on the decode
path. The descriptor-mutation blocker for single-submit recording is
removed for all dispatch-target descriptors.

## Relationship to Prior Diaries

### Diary 0030 (rejected: all-six intra-DeltaNet pre-bind)

Diary 0030's all-six failure included `dn_out_proj` alongside the other
five intra-DeltaNet sub-step descriptors. At the time, any of the six
could have been the cause — the failure was grouped and no individual
isolation had been performed.

With `dn_compute_g_beta` corrected (diary 0034, constructor-ordering bug),
`dn_recurrent` independently verified (diary 0035), `dn_norm_gate`
independently verified (diary 0036), and now `dn_out_proj` independently
verified, diary 0030's all-six descriptor-mutation blocker is empirically retired for the six tracked dispatch-target descriptors. Each descriptor has now been independently pre-bound and verified; dn_compute_g_beta had a proven constructor-ordering root cause, while the other descriptors were proven safe individually:

- `dn_l2_q` — independently proven safe (diary 0032)
- `dn_l2_k` — independently proven safe (diary 0032)
- `dn_compute_g_beta` — root cause identified as constructor-ordering bug
  (diary 0033/0034); independently proven safe after correction (diary 0034)
- `dn_recurrent` — independently proven safe (diary 0035)
- `dn_norm_gate` — independently proven safe (diary 0036)
- `dn_out_proj` — independently proven safe (this diary)

Diary 0030's grouped failure is empirically retired: dn_compute_g_beta had a proven constructor-ordering root cause (diary 0034), while the other five descriptors were proven safe individually across diaries 0032, 0035, 0036, and 0037.

### Diary 0036 (dn_norm_gate pre-bind)

Diary 0036 described `dn_norm_gate` as the "penultimate dispatch-target
blocker" and noted `dn_out_proj` as "the last untested descriptor from
diary 0030's all-six failure set." This diary confirms that classification
and completes the sequence. The pre-binding pattern for `dn_out_proj`
follows the same structural approach as `dn_norm_gate` — per-layer weight
slice from `B.weights`, V-section offset into `B.dn_qkv`, alias/guard in
both decode and prefill paths — with the difference that its binding 2
references `B.act_b` (full buffer) rather than `B.dn_z`.

## Files Changed

```
src/runtime/vk_session.cpp    — Add dn_out_proj vector resize, per-layer
                                allocation, weight lookup, pre-binding
                                block, layer_major_prefill alias/guard,
                                decode alias/guard, correct_last_token_hidden
                                alias/guard, and update verbose count from
                                29 to 30.
src/runtime/vk_session.hpp    — Add dn_out_proj vector declaration and
                                update struct comments.
```

## Known Limitations

1. **This does NOT make full GPU offload complete.** The host still:
   - Iterates per-layer dispatches in the decode loop.
   - Submits command buffers and waits on fences per step.
   - Observes/downloads generated-token outputs, logits, and diagnostic
     data for external output, parity checking, and fallback/diagnostic
     paths.
   - Orchestrates prefill (layer-major loop).

2. **This does NOT make single-submit complete.** While all dispatch-target
   descriptor mutations are eliminated, remaining blockers include:
   - Host-side per-layer/per-step submissions and fence waits remain.
   - Fallback/diagnostic readbacks remain.
   - Step-varying parameters (e.g., position, scale factors) still require
     per-dispatch push constants or GPU-readable state updates.

3. **This does NOT make persistent dispatch or the megakernel.** No fusion
   of dispatches, no cross-workgroup coordination, no persistent kernel.

4. **dn_split_q and dn_split_kv remain on the legacy mutation path.** These
   are internal decomposition descriptors (not dispatch targets) and are
   not part of the per-dispatch-target descriptor work. They are listed
   for completeness but are not blockers for single-submit recording.

5. **Diary 0030's all-six descriptor-mutation blocker is empirically retired for the six tracked dispatch-target descriptors.**
   dn_compute_g_beta had a proven constructor-ordering root cause (diary 0034),
   while the other descriptors were proven safe individually.
   This does NOT imply full GPU offload, single-submit, persistent dispatch,
   or megakernel completion.

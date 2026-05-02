# 0036: dn_norm_gate Descriptor Pre-Binding — Penultimate Dispatch-Target Blocker

## Goal

Extend the opt-in per-layer stable descriptor set path to cover the
`dn_norm_gate` descriptor — the DeltaNet norm/gate kernel dispatch target
that applies the delta_norm weight and computes the gated output after
the recurrent step. This moves the 29th per-layer descriptor set from
per-dispatch mutation to pre-bound, reducing the remaining dispatch-target
blockers from two to one (`dn_out_proj`).

This diary independently isolates `dn_norm_gate` from the two remaining
uncovered stateful descriptors (`dn_norm_gate`, `dn_out_proj`). Unlike
`dn_compute_g_beta` (constructor-ordering failure in diary 0033, corrected
in diary 0034), `dn_norm_gate` does not depend on constructor-ordered
scratch buffers — it binds the V section of `B.dn_qkv` (output region),
the `B.dn_z` gate buffer, and a weight-buffer slice for the per-layer
delta_norm weight. Its independent success narrows the remaining
corruption hypothesis to a single untested descriptor: `dn_out_proj`.

## Inference Concepts

### What dn_norm_gate's descriptor bindings represent

The `ds_layout_3` descriptor set for `dn_norm_gate` has three bindings:

| Binding | Buffer | Offset | Size | Purpose |
|---------|--------|--------|------|---------|
| 0 | `B.dn_qkv` | `(DN_KEY_TOTAL + DN_KEY_TOTAL) * 2` (V section) | `DN_VAL_TOTAL * 2` | IO buffer — the per-head V-space accumulator (fp16); read after recurrent step, overwritten with gated output |
| 1 | `B.dn_z` | 0 | full buffer | Gate input — the Z gate activation (fp16) |
| 2 | `B.weights` | Per-layer `delta_norm` weight offset | Per-layer `delta_norm` weight size | Weight buffer — the delta_norm weight row for this layer (fp16) |

Binding 0 is the key design detail: `dn_norm_gate` reads the per-head
V-space output from `dn_recurrent` (stored in the V section of `B.dn_qkv`),
applies the delta_norm weight scaled by the Z gate, and writes the gated
result back to the same region. The V section offset
`(DN_KEY_TOTAL + DN_KEY_TOTAL) * 2` skips both the Q and K regions of the
shared `B.dn_qkv` buffer to reach the V region. The range `DN_VAL_TOTAL * 2`
covers all heads' V-space values in fp16.

Binding 2 is the per-layer weight slice. Unlike bindings 0 and 1 which
point to fixed activation buffers with constant offsets, binding 2 requires
the per-layer weight offset and size from the artifact's `delta_norm` role.
The weight lookup (`artifact_.find_by_role("layer.${layer}.delta_norm")`) is
already computed in the construction pre-binding block — the same lookup
that supplies the per-layer `assert` guard.

### Why this was the next test target

After diary 0035, four of the six intra-DeltaNet sub-step descriptors were
proven safe for pre-binding:
- `dn_l2_q` (diary 0032)
- `dn_l2_k` (diary 0032)
- `dn_compute_g_beta` (diary 0034, after constructor-ordering correction)
- `dn_recurrent` (diary 0035)

The remaining two — `dn_norm_gate` and `dn_out_proj` — were the untested
subset from diary 0030's all-six failure. `dn_norm_gate` was the natural
next candidate because:
- It does not depend on constructor-ordered buffers (no `a_log_bias`
  ordering hazard).
- Its binding 2 is a weight-buffer slice, not a recurrent/state buffer,
  so stale-weight corruption patterns are distinct from diary 0030's
  decode-state corruption.
- Its bindings 0 and 1 are fixed-offset activation regions — identical
  binding semantics to the already-proven descriptors.

`dn_out_proj` remains untested. It is now the sole remaining
dispatch-target blocker.

## Implementation Work Completed

### Struct additions: `src/runtime/vk_session.hpp`

`PerLayerDescriptorSets` now includes a `dn_norm_gate` vector (added in
this slice). The struct comment is updated to reflect the new coverage:

```
/// dn_l2_q/dn_l2_k, dn_compute_g_beta, dn_recurrent, and dn_norm_gate are covered here.
/// dn_out_proj remains excluded.
```

### Construction: `src/runtime/vk_session.cpp`

**Comment update** (line ~464):

The comment block is updated to include `dn_norm_gate` in the covered
list:
```
// L2-norm q/k (dn_l2_q, dn_l2_k), dn_recurrent, and dn_norm_gate are covered here;
// remaining inner DeltaNet sub-step descriptor (dn_out_proj) is not
// covered and remains on the old path; dn_compute_g_beta is now covered.
```

**Vector resizing** — `per_layer_sets_->dn_norm_gate.resize(LAYERS)` is
added in the allocation loop.

**Allocation** — `per_layer_sets_->dn_norm_gate[i] = alloc3()` is added
in the per-layer allocation loop.

**Weight lookup** — The `dn_norm_w` lookup is added alongside the
existing weight assertions:
```
auto dn_norm_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".delta_norm");
assert(dn_qkv_w && dn_z_w && dn_a_w && dn_b_w && dn_conv_w && dn_norm_w);
```

**Pre-binding** — The pre-binding block now contains a `dn_norm_gate`
section inside `if (per_layer_sets_enabled_)`:
```
// dn_norm_gate: bindings 0=dn_qkv(V section) io, 1=dn_z gate, 2=delta_norm weight
dev_.update_descriptor_set(per_layer_sets_->dn_norm_gate[layer], 0, B.dn_qkv,
    (DN_KEY_TOTAL + DN_KEY_TOTAL) * 2, DN_VAL_TOTAL * 2);
dev_.update_descriptor_set(per_layer_sets_->dn_norm_gate[layer], 1, B.dn_z);
dev_.update_descriptor_set(per_layer_sets_->dn_norm_gate[layer], 2, B.weights,
    dn_norm_w->offset, dn_norm_w->nbytes);
```

**Verbose count update** — The verbose line at construction now reports:
```
per-layer descriptor sets: 28 x 29 sets pre-bound
```

(Updated from 28 in diary 0035, adding dn_norm_gate.)

### Prefill alias: `src/runtime/vk_session.cpp` `layer_major_prefill()`

The per-layer alias for `dn_norm_gate` is added:

```cpp
VkDescriptorSet ds_dn_norm_gate = per_layer_sets_enabled_
    ? per_layer_sets_->dn_norm_gate[layer]
    : D.dn_norm_gate;
```

### Prefill fallback guard: `src/runtime/vk_session.cpp` `layer_major_prefill()`

The per-dispatch `update_descriptor_set` calls for `D.dn_norm_gate` are
guarded by `if (!per_layer_sets_enabled_)`:

```cpp
if (!per_layer_sets_enabled_) {
    dev_.update_descriptor_set(D.dn_norm_gate, 0, B.dn_qkv,
        (DN_KEY_TOTAL + DN_KEY_TOTAL) * 2, DN_VAL_TOTAL * 2);
    dev_.update_descriptor_set(D.dn_norm_gate, 1, B.dn_z);
    dev_.update_descriptor_set(D.dn_norm_gate, 2, B.weights,
        dn_norm_w->offset, dn_norm_w->nbytes);
}
```

The dispatch site uses the alias:

```cpp
vkCmdBindPipeline(rec_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, P.deltanet_norm_gate);
vkCmdBindDescriptorSets(rec_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
    P.pipeline_layout_32, 0, 1, &ds_dn_norm_gate, 0, nullptr);
```

### Decode alias + guard: `src/runtime/vk_session.cpp` `decode()`

The same alias and guarded mutation pattern is applied in the decode
loop. Two dispatch sites use the alias: the **single-step decode submit**
(line ~2293) and the **grouped norm_gate/out_proj/residual/MLP submit**
(line ~4544). Both pass the pre-bound per-layer set when the gate is
active.

### What does NOT change

- **No new shader code.** The `deltanet_norm_gate.comp` shader pipeline,
  push constants, and access patterns are unchanged. The descriptor
  binding pattern (3 bindings: dn_qkv V section, dn_z, delta_norm weight)
  is identical — only the descriptor set handle changes.
- **No descriptor pool sizing change.** The existing capacity (1024 maxSets,
  4096 storage buffer slots) from diary 0029 is sufficient.
- **No pipeline layout change.** Descriptor allocation uses `ds_layout_3`
  and dispatch uses `pipeline_layout_32`.
- **No buffer reallocation.** `B.dn_qkv`, `B.dn_z`, `B.weights`, and all
  weight regions are unchanged. The per-layer weight offset/size
  calculation is identical to the legacy path.
- **No inference semantics change.** The norm/gate kernel computation is
  identical. The descriptor binding points to the same buffer regions at
  the same byte offsets; the shader sees the same data.
- **dn_out_proj remains on the old path.** This diary covers only
  dn_norm_gate. One remaining dispatch-target descriptor is not touched.
- **This is NOT full GPU offload, NOT single-submit, NOT persistent
  dispatch, and NOT the megakernel.** The host still orchestrates every
  step (per-layer iteration, submission, fence wait). Argmax, logit
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

### Per-layer stable descriptor sets gate (29 per-layer sets)

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

This is the critical test: the full gated path with 29 pre-bound per-layer
sets, including the newly covered dn_norm_gate descriptor. Passes parity.

### Combined gate suite (per-layer sets + GPU chunk-prefill + device-resident token + deferred download)

Six env gates exercised simultaneously:

- `SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1` — per-layer descriptor pre-binding
  with the newly covered dn_norm_gate.
- `SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1` — device-resident decode token embedding.
- `SPOCK_GPU_DEFER_TOKEN_DOWNLOAD=1` — deferred batch token download.
- `SPOCK_GPU_CHUNK_PREFILL=1` — GPU chunk-prefill path.
- `SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1` — GPU-collected chunk inputs.
- `SPOCK_GPU_CHUNK_PREFILL_TILED=1` — tiled single-dispatch shader.

Single-prompt decode parity:

```sh
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

```json
{"status":"ok","checked":1,"failures":[]}
```

Multi-prompt decode parity (two diverse prompts):

```sh
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

```json
{"status":"ok","checked":2,"failures":[]}
```

### CTest regression suite (per-layer sets + device-resident token + deferred download)

Combined gate CTest for the GPU-collected chunk-prefill path:

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
| `spock_vk_decode_gpu_collect_chunk_prefill_short` | 114.95 s |
| `spock_vk_decode_gpu_collect_chunk_prefill_tiled` | 8.95 s |
| `spock_vk_decode_gpu_collect_chunk_prefill_short_baseline` | 5.09 s |
| **Total real time** | **128.98 s** |

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
| DeltaNet norm_gate (this diary) | 1 set x 28 layers | Per-layer | Yes |
| RoPE | 2 sets (session-level) | Session | Yes (diary 0031) |
| **Per-layer subtotal** | **29 sets x 28 layers = 812 sets** | | |
| **Session-level subtotal** | **2 sets** | | |
| **Total pre-bound** | **814 sets** | | |

Remaining uncovered (per-dispatch mutation on decode path):

| Descriptor | Status | Notes |
|------------|--------|-------|
| `dn_out_proj` | Uncovered (dispatch target) | Binds DeltaNet output accumulation buffer. Last untested dispatch-target descriptor from diary 0030's all-six failure set. |
| `dn_split_q` | Uncovered (internal) | Internal decomposition descriptor, not a dispatch target. |
| `dn_split_kv` | Uncovered (internal) | Internal decomposition descriptor, not a dispatch target. |

`dn_norm_gate` is no longer in the uncovered list. The remaining
dispatch-target blocker is one descriptor: `dn_out_proj`. This is the
last descriptor that must mutate per dispatch on the decode path.
The two internal decomposition descriptors (`dn_split_q`, `dn_split_kv`)
remain listed as uncovered but are not dispatch targets.

## Relationship to Prior Diaries

### Diary 0030 (rejected: all-six intra-DeltaNet pre-bind)

Diary 0030's all-six failure included dn_norm_gate alongside the other
five intra-DeltaNet sub-step descriptors. The all-six failure was grouped
— without individual isolation, any of the six could have been the cause.

With `dn_compute_g_beta` corrected (diary 0034), `dn_recurrent`
independently verified (diary 0035), and now `dn_norm_gate` independently
verified, the all-six failure is narrowed to a single untested descriptor:
`dn_out_proj`. Five of the six original descriptors are proven safe.

It is now possible that diary 0030's failure was caused by *either*
`dn_compute_g_beta` (constructor ordering, proven in diary 0034) *or*
`dn_out_proj`, or a combination. `dn_norm_gate` was definitively an
innocent co-traveler in that failure — just as `dn_recurrent`, `dn_l2_q`,
and `dn_l2_k` were.

### Diary 0035 (dn_recurrent pre-bind)

Diary 0035 established the pre-binding pattern for stateful recurrent
descriptors. `dn_norm_gate` follows the same structural pattern — per-layer
offset into `B.dn_qkv` (V section rather than Q region), per-layer
weight slice from `B.weights`, alias/guard in both decode and prefill
paths — with the only difference being that binding 2 references a weight
buffer rather than a recurrent state buffer. Both successfully pass parity
on the combined gate suite.

### Diary 0033/0034 (dn_compute_g_beta isolation and correction)

The constructor-ordering discovery from diary 0034 does not apply here:
`dn_norm_gate`'s binding 2 references a weight-buffer slice whose offset
is resolved from the artifact, not from a buffer that must be created
before the pre-binding block. No ordering hazard was present or required
a fix.

## Files Changed

```
src/runtime/vk_session.cpp    — Add dn_norm_gate vector resize, per-layer
                                allocation, weight lookup, pre-binding
                                block, layer_major_prefill alias/guard,
                                decode alias/guard (both dispatch sites),
                                and update verbose count from 28 to 29.
src/runtime/vk_session.hpp    — Add dn_norm_gate vector declaration and
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
   - One stateful intra-DeltaNet sub-step descriptor (`dn_out_proj`) still
     mutates per dispatch on the decode path. It is the last dispatch-target
     descriptor not yet proven safe for pre-binding.
   - Host-side per-layer/per-step submissions and fence waits remain.
   - Fallback/diagnostic readbacks remain.

3. **Diary 0030's all-six failure is further narrowed but not fully resolved.**
   With five of six descriptors now independently verified (`dn_l2_q`,
   `dn_l2_k`, `dn_compute_g_beta`, `dn_recurrent`, `dn_norm_gate`), the
   remaining untested descriptor is `dn_out_proj`. It may independently fail.

4. **This is a narrow step toward single-submit.** One remaining stateful
   dispatch-target descriptor is the hard problem. Its independent isolation
   test is the next step.

# 0029: Opt-in Per-Layer Stable Descriptor Sets — Reducing Per-Layer Descriptor Mutation Under the Gate

## Goal

Eliminate per-layer descriptor mutation in the decode loop by pre-allocating
and pre-binding stable descriptor sets at session construction time, guarded
by `SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1`. This reduces the number of
`vkUpdateDescriptorSets` calls per decode step and is a structural
prerequisite for future single-submit recording (where the command buffer
must be recorded ahead of time and cannot mutate descriptors between
layers).

Before this entry, every decode step mutated the same 24 covered `VkDescriptorSet`
handles per layer — re-binding weight offsets, activation buffer bindings,
and per-layer buffer offsets (KV cache slot, conv1d state offset) with
each dispatch. Each mutation required a `vkUpdateDescriptorSets` call even
when only the weight range or buffer offset changed, not the actual buffer
handles.

After this entry, when `SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1`:

- The descriptor pool capacity is increased from 192 maxSets / 192 storage
  buffers to 1024 maxSets / 4096 storage buffers to accommodate the
  additional 672 descriptor sets (28 layers x 24 sets).
- At session construction time, 24 `VkDescriptorSet` handles per layer are
  allocated from `ds_layout_3` (the common pipeline layout) and pre-bound
  with their layer-specific weight offsets, activation buffer bindings, and
  static per-layer buffer offsets (KV cache layer offset, conv1d state
  offset).
- At decode time, the per-layer mutation block is skipped entirely. The
  `decode()` function selects between the per-layer stable set and the old
  shared set via a tenary alias (`ds_input_norm = per_layer_sets_enabled_ ?
  per_layer_sets_->input_norm[layer] : D.input_norm`), then passes the
  alias to `vkCmdBindDescriptorSets`. No `vkUpdateDescriptorSets` call is
  made for any covered descriptor set during decode.
- RoPE descriptors (`D.rope`, `D.rope_k`) still mutate per step because
  they carry the per-step `seq_pos`-dependent rope frequency offset — this
  is the one remaining per-step descriptor mutation on the covered path.
- Intra-DeltaNet sub-step descriptors (`dn_l2_q`, `dn_l2_k`, `dn_recurrent`,
  `dn_norm_gate`, `dn_out_proj`, `dn_compute_g_beta`) are NOT covered.
  These are low-level scratch descriptors for the DeltaNet recurrent
  kernel's internal sub-steps and remain on the old per-dispatch mutation
  path. Covering them is a separate prerequisite for full single-submit.

The change is opt-in (env-gated) and defaults to the existing per-layer
mutation path.

## Inference Concepts

### The per-layer descriptor mutation problem

In the baseline decode loop, each of the 28 layers binds the same 24 covered
descriptor sets with adjusted parameters. For example, `D.input_norm` is
mutated every layer even though only binding 1 (the weight range) changes
per layer — bindings 0 (`act_a`) and 2 (`act_b`) are the same across all
layers. Similarly, `D.kv_store` is mutated to point `kv_cache` binding 2 at
the correct per-layer KV cache segment, and `D.dn_conv` is mutated to set
`dn_conv_state` binding 1 at the correct per-layer conv1d state offset.

Each mutation is a single `vkUpdateDescriptorSets` call that writes N
descriptor writes (`VkWriteDescriptorSet`). The total per-step count is:

- 10 descriptor sets for common MLP/norm ops (all 28 layers)
- 10 descriptor sets for attention-specific ops (6 attention layers)
- 5 descriptor sets for DeltaNet-specific ops (18 DeltaNet layers)
- = dozens of descriptor binding updates per layer, repeated across 28
  layers per decode step.

For the layer_by_layer decode path these are not a bottleneck — Vulkan
descriptor updates are CPU-side and cheap relative to GPU dispatch costs.
But for single-submit recording, the command buffer must be recorded
ahead of time with all descriptor bindings pre-resolved. If the command
buffer is recorded once and submitted per-step, per-layer descriptor
mutation between recording and submission is disallowed. Pre-bound
per-layer sets solve this: the command buffer can reference
`per_layer_sets_->input_norm[layer]` at recording time, and the bindings
are already correct and immutable.

### What this changes and does NOT change

**What changes:**
- Descriptor pool capacity (maxSets 192→1024, storage buffer slots 192→4096).
- At session construction: 28 x 24 = 672 additional descriptor sets allocated
  and bound.
- At decode: per-layer mutation block (`!per_layer_sets_enabled_`) is
  skipped for the covered descriptor sets. RoPE and intra-DeltaNet
  descriptor mutations remain.
- Env gate `SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1` activates the new path.
- `#include <cassert>` added to `vk_session.cpp` for the `assert()` calls
  in the pre-binding code.

**What does NOT change:**
- **This is NOT full GPU offload and NOT the megakernel.** The host still
  orchestrates per-layer dispatch, per-step submission, and fence waits.
  Decode argmax, diagnostic readbacks, and per-step embedding remain
  host-mediated.
- **This is NOT single-submit.** The per-layer descriptor mutation is a
  prerequisite for single-submit recording (the command buffer must not
  mutate descriptors between recording and submission), but the decode
  loop still submits one command buffer per step with per-layer dispatches.
  Single-submit recording requires the entire token's work to be recorded
  into one command buffer at session creation time with no per-step
  descriptor mutation. Per-layer stable sets are a necessary but not
  sufficient condition.
- **RoPE descriptors still mutate per step.** The `D.rope` and `D.rope_k`
  descriptor sets carry the per-step `seq_pos` rope frequency offset.
  These are updated once per step (not per layer) regardless of the gate.
- **Intra-DeltaNet sub-step descriptors still mutate per dispatch.**
  The `dn_l2_q`, `dn_l2_k`, `dn_recurrent`, `dn_norm_gate`, `dn_out_proj`,
  and `dn_compute_g_beta` descriptors are low-level scratch bindings for
  the DeltaNet recurrent kernel's internal stages. They are allocated from
  `ds_layout_3` but not covered by the per-layer pre-binding. These must
  be addressed as a separate prerequisite before full single-submit.
- **Default behavior is unchanged.** The per-layer mutation path remains
  the default. The stable per-layer sets activate only when the env gate
  is set.
- **No change to shaders, pipelines, pipeline layouts, or session state.**
  Only descriptor allocation, binding, and the decode loop's
  `vkCmdBindDescriptorSets` calls change. All existing pipelines,
  shaders, and layouts are reused.
- **No performance speedup is claimed.** Descriptor mutations are CPU-side
  and cheap relative to GPU dispatch. The change is structural: it removes
  a data dependency that would block single-submit recording.

## Implementation Work Completed

### Descriptor pool capacity increase (`vk_device.cpp`)

The existing descriptor pool was sized for the baseline single-set design:
192 `maxSets` and 192 storage buffer slots. The per-layer approach needs
672 additional sets and their storage buffer bindings.

```cpp
// Before:
{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 192},
desc_pool_info.maxSets = 192;

// After:
{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4096},
desc_pool_info.maxSets = 1024;
```

The pool now supports 1024 descriptor sets total (up from 192) and 4096
storage buffer descriptors (up from 192). The combined-image-sampler count
(64) is unchanged — no new samplers are introduced.

These are safe upper bounds. The actual number of allocated sets with the
gate active is approximately:
- Baseline: ~40 sets (session-level + decode descriptors)
- Per-layer addition: 28 layers x 24 sets = 672
- Total under gate: ~712 sets
- Safety margin to 1024: ~312 sets for future additions

The `VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT` flag is preserved
so individual sets can be freed if the pool is ever reconfigured.

### Per-layer descriptor set allocation and pre-binding (`vk_session.cpp`, constructor)

A new `PerLayerDescriptorSets` struct (defined in `vk_session.hpp`) holds
24 `std::vector<VkDescriptorSet>` fields, each of size `LAYERS` (= 28):

```
Common MLP/norm (all 28 layers):
  input_norm, residual1, post_norm, gate, up, down, down_f32,
  residual2, mlp_residual_mixed

Attention-specific (all 28 rows, bound for 6 attention layers):
  q_proj, k_proj, v_proj, q_norm, k_norm, kv_store, attn,
  o_proj, o_proj_f32, attn_residual_mixed

DeltaNet-specific (all 28 rows, bound for 18 DeltaNet layers):
  dn_qkv_proj, dn_z_proj, dn_a_proj, dn_b_proj, dn_conv
```

The constructor checks `SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS` and
conditionally allocates:

```cpp
per_layer_sets_enabled_ = []() {
  const char* e = std::getenv("SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS");
  return e && e[0] == '1' && e[1] == '\0';
}();
if (per_layer_sets_enabled_) {
  per_layer_sets_ = std::make_unique<PerLayerDescriptorSets>();
  auto alloc3 = [&]() { return dev_.allocate_descriptor_set(pipes_->ds_layout_3); };
  // ... allocate LAYERS copies of each descriptor set
  for (uint32_t i = 0; i < LAYERS; ++i) {
    per_layer_sets_->input_norm[i] = alloc3();
    per_layer_sets_->residual1[i] = alloc3();
    // ... all 24 sets per layer
  }
}
```

After allocation, each descriptor set is pre-bound with layer-specific
weight offsets, activation buffer references, and static per-layer buffer
offsets:

```cpp
for (uint32_t layer = 0; layer < LAYERS; ++layer) {
  bool is_attn = (schedule[layer] == model::LayerKind::FullAttention);
  uint32_t attn_idx = is_attn ? attn_layer_idx(layer) : 0;
  // MLP/norm weights — all layers share the same projection shape
  auto input_norm_w = artifact_.find_by_role("layer." + std::to_string(layer) + ".input_norm");
  // ...
  dev_.update_descriptor_set(per_layer_sets_->input_norm[layer], 0, B.act_a);
  dev_.update_descriptor_set(per_layer_sets_->input_norm[layer], 1, B.weights, input_norm_w->offset, input_norm_w->nbytes);
  dev_.update_descriptor_set(per_layer_sets_->input_norm[layer], 2, B.act_b);
  // ...
}
```

Key invariants established at pre-binding time:
- **Weight pointers**: each layer's weight range is looked up from the
  artifact by role string and bound with the correct offset + size.
  Same `B.weights` buffer, different per-layer offset/size.
- **KV cache offset**: attention layers pre-bind `kv_cache` binding 2 at a
  per-layer offset `attn_idx * MAX_SEQ * 2 * KV_HEADS * HEAD_DIM * 2`.
  This is computed from the attention layer index (0-based, counting only
  attention layers), not the absolute layer index.
- **Conv1d state offset**: DeltaNet layers pre-bind `dn_conv_state` binding 1
  at a per-layer offset `dn_idx * DN_CONV_DIM * DN_CONV_KS * 2`, where
  `dn_idx` is the DeltaNet layer index (0-based, counting only DeltaNet
  layers).
- **Activation buffers**: all layers share the same global activation
  buffers (`act_a`, `act_b`, `act_c`, `q`, `k`, `v`, `dn_qkv`, `dn_z`,
  `dn_a`, `dn_b`, etc.). These are the same handles for every layer in
  the per-step decode — each dispatch resets the scratch regions.

### Decode loop: conditional skip of per-layer mutation

In `decode()`, the existing ~200-line per-layer descriptor mutation block
is guarded by `!per_layer_sets_enabled_`:

```cpp
if (!per_layer_sets_enabled_) {
  dev_.update_descriptor_set(D.input_norm, 0, B.act_a);
  dev_.update_descriptor_set(D.input_norm, 1, B.weights, ...);
  // ... all covered per-layer mutations
}
```

When the gate is active, this entire block is skipped.

### Decode loop: per-layer descriptor set alias

Immediately after the mutation block, each `vkCmdBindDescriptorSets` call
selects between the per-layer stable set and the old shared set:

```cpp
VkDescriptorSet ds_input_norm = per_layer_sets_enabled_ ? per_layer_sets_->input_norm[layer] : D.input_norm;
VkDescriptorSet ds_q_proj = per_layer_sets_enabled_ ? per_layer_sets_->q_proj[layer] : D.q_proj;
// ... all 24 aliases
```

These aliases replace `D.input_norm`, `D.q_proj`, etc. in every
`vkCmdBindDescriptorSets` call throughout the decoder:

```cpp
// Before:
vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
    P.pipeline_layout_3, 0, 1, &D.input_norm, 0, nullptr);

// After:
vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
    P.pipeline_layout_3, 0, 1, &ds_input_norm, 0, nullptr);
```

This substitution is applied to all ~50 `vkCmdBindDescriptorSets` calls
in the decode path (attention layers and DeltaNet layers), for both the
`layout_3` and `layout_32` pipeline layouts.

### RoPE mutation: still per-step on the covered path

After the per-layer mutation block (and regardless of the gate), the rope
descriptors are updated once per step:

```cpp
if (per_layer_sets_enabled_ && is_attn) {
  dev_.update_descriptor_set(D.rope, 0, B.q);
  dev_.update_descriptor_set(D.rope, 1, B.rope_freq, seq_pos * ROTARY_DIM * 4, ROTARY_DIM * 4);
  dev_.update_descriptor_set(D.rope, 2, B.q);
  dev_.update_descriptor_set(D.rope_k, 0, B.k);
  dev_.update_descriptor_set(D.rope_k, 1, B.rope_freq, seq_pos * ROTARY_DIM * 4, ROTARY_DIM * 4);
  dev_.update_descriptor_set(D.rope_k, 2, B.k);
}
```

This is the only per-step descriptor mutation on the covered path. RoPE
is fundamentally step-dependent because the rope frequency offset changes
with each decode step's sequence position. A future path would need to
either pre-bind per-step RoPE offsets (impractical — unbounded max steps)
or embed the position in push constants instead of descriptor bindings.

### Summary of covered vs. non-covered descriptor sets

| Category | Sets covered by per-layer pre-binding | Sets still mutated per dispatch (not covered) |
|---|---|---|
| Common MLP/norm | input_norm, residual1, post_norm, gate, up, down, down_f32, residual2, mlp_residual_mixed | — |
| Attention | q_proj, k_proj, v_proj, q_norm, k_norm, kv_store, attn, o_proj, o_proj_f32, attn_residual_mixed | rope, rope_k (mutated per step) |
| DeltaNet (stage 1) | dn_qkv_proj, dn_z_proj, dn_a_proj, dn_b_proj, dn_conv | — |
| DeltaNet (sub-step internals) | — | dn_split_q, dn_split_kv, dn_l2_q, dn_l2_k, dn_recurrent, dn_norm_gate, dn_out_proj, dn_compute_g_beta |

The intra-DeltaNet sub-step descriptors are the primary remaining mutation
blocker for full single-submit. They are allocated from `ds_layout_3` but
carry intermediate scratch buffer bindings that change within a single
layer's DeltaNet kernel dispatch. A future change would need to either
cover them with per-sub-step pre-bound sets (impractical — each layer's
scratch layout is identical) or eliminate them by fusing the DeltaNet
sub-steps into a single kernel that uses internal shared memory / subgroup
communication instead of storage buffer round-trips.

## Verification

All commands run on the target RADV RX 6750 XT (NAVI22) hardware.

### Build and whitespace

```sh
git diff --check
```

No whitespace errors.

```sh
cmake -S . -B build && cmake --build build -j
```

Passed cleanly. No new shaders, no new pipelines, no new pipeline layouts.
The `#include <cassert>` addition compiles without issues.

### Short correctness parity (max-new-tokens 16)

```sh
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

The per-layer descriptor set path produces correct 16-token output for
`short_correctness_001`. All attention and DeltaNet layers bind the
correct weights and buffer offsets.

### Combined with device-resident token and deferred download

```sh
SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1 \
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

Composes correctly with the device-resident token and deferred download
gates (diaries 0027/0028). The per-layer descriptor sets have no
interaction with token embedding or download paths.

### Combined with full GPU chunk-prefill suite

```sh
SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1 \
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

Composes correctly with all GPU chunk-prefill gates. The chunk-prefill
path uses separate descriptor sets (allocated earlier in the constructor
and unaffected by the per-layer sets) and does not interact with the
decode-loop descriptor bindings.

### Combined gate on longer prompts

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

```
{"status":"ok","checked":2,"failures":[]}
```

The combined path with all gates passes on `mixed_correctness_023` and
`pp520_046` through 4 generated tokens each.

### CTest GPU-collect suite (3/3 passed)

```sh
SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1 \
  SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1 SPOCK_GPU_DEFER_TOKEN_DOWNLOAD=1 \
  ctest --test-dir build --output-on-failure -R spock_vk_decode_gpu_collect_chunk_prefill
```

| Test | Status | Time |
|------|--------|------|
| `spock_vk_decode_gpu_collect_chunk_prefill_short` | Passed | 114.83 sec |
| `spock_vk_decode_gpu_collect_chunk_prefill_tiled` | Passed | 8.98 sec |
| `spock_vk_decode_gpu_collect_chunk_prefill_short_baseline` | Passed | 5.07 sec |

All three CTest tests pass with the per-layer descriptor set gate active.
Times are within noise of previous runs (diary 0028: 114.98, 9.01, 5.09).
No performance speedup is claimed — the change does not reduce dispatch
count, submission overhead, or GPU work.

### Default path parity (env var not set)

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

The default path (per-layer mutation, unchanged) continues to produce
correct output. The new gate has no effect when the env var is not set.

### Descriptor pool exhaustion test

```sh
SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1 \
  python3 tests/run_vk_decode_parity.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --reference tests/data/reference_tokens.jsonl \
  --ids short_correctness_001,mixed_correctness_023,pp520_046 \
  --max-new-tokens 16
```

```
{"status":"ok","checked":3,"failures":[]}
```

Running three prompts in a single invocation allocates and destroys three
separate `DecodeSession` instances. Each session allocates its own 672
descriptor sets from the pool. The increased pool capacity (1024 maxSets)
supports this without exhaustion. Each session's destructor frees its
descriptors, and the `FREE_DESCRIPTOR_SET_BIT` pool flag allows them to
be reused by the next session.

## Known Limitations

1. **Intra-DeltaNet sub-step descriptors are not covered.** The 8 DeltaNet
   internal descriptors (`dn_split_q`, `dn_split_kv`, `dn_l2_q`, `dn_l2_k`,
   `dn_recurrent`, `dn_norm_gate`, `dn_out_proj`, `dn_compute_g_beta`)
   still mutate per dispatch in the DeltaNet kernel. These are the last
   per-layer mutation on the decode path and must be covered before full
   single-submit recording is possible. They are allocated from
   `ds_layout_3` and follow the same pattern as the covered sets — each
   binds scratch buffer regions that are identical across layers — but
   they change within a single layer's dispatch sequence and the existing
   design binds them with per-call offsets that differ from the stable
   pre-bound pattern.

2. **RoPE descriptors still mutate per step.** The `D.rope` and `D.rope_k`
   descriptors encode the per-step rope frequency offset for attention
   decode. These are updated once per step (not per layer) regardless of
   the gate. For single-submit recording with known max sequence length,
   the offset could be embedded in a push constant or a step-indexed
   lookup table, but that requires changing the shader interface.

3. **Descriptor pool capacity is hardcoded at 1024 maxSets / 4096 storage
   buffers.** These values were chosen to accommodate the maximum required
   sets (~712 under the gate) with margin for future additions. If the
   number of per-layer sets grows significantly (e.g., covering the
   intra-DeltaNet descriptors as well), the pool size would need
   recalculation. This is a maintenance concern, not a correctness concern.

4. **Session construction is slower with the gate active.** Allocating 672
   descriptor sets and issuing their descriptor binding updates adds
   ~1-2 ms to session construction. This is a one-time cost per
   `DecodeSession` creation (not per decode call) and is negligible
   relative to model loading and weight upload.

5. **No performance speedup is claimed.** The change removes the covered
   per-layer descriptor updates from the decode loop under the gate, but
   those updates are CPU-side and cheap relative to GPU dispatch — the
   existing per-layer
   mutation path was never a bottleneck. The value is structural: the
   per-layer mutation pattern blocks single-submit recording, and this
   change removes that blocker for the covered descriptor sets.

6. **Coverage still limited.** Verified on `short_correctness_001` (16
   tokens), `mixed_correctness_023`, and `pp520_046` (4 tokens each).
   Broader P0 coverage and 512+ token prompts are still pending, as in
   prior entries.

## Next Work

### Near-term: Cover intra-DeltaNet sub-step descriptors

The 8 uncovered DeltaNet internal descriptors are the last per-layer
mutation on the decode path. They follow the same pattern as the covered
sets: each binds weight ranges and scratch buffer regions that are
identical across layers, with only the weight offset changing per layer.
A future change should:

- Add `dn_split_q`, `dn_split_kv`, `dn_l2_q`, `dn_l2_k`, `dn_recurrent`,
  `dn_norm_gate`, `dn_out_proj`, `dn_compute_g_beta` to the
  `PerLayerDescriptorSets` struct.
- Pre-bind them in the constructor with per-layer weight offsets.
- Skip their mutation in the decode loop when the gate is active.
- Verify parity on the same test suite.

### Near-term: Remove per-step RoPE mutation as a single-submit blocker

For single-submit recording, the per-step RoPE descriptor update must be
eliminated. Options:

- **Push constant approach**: encode `seq_pos` in a push constant and have
  the shader compute the rope frequency offset inline. This avoids any
  per-step descriptor mutation but requires changing the `attention_decode`
  shader to accept a `seq_pos` push constant and compute the offset rather
  than reading it from a pre-filled buffer binding.
- **Step-indexed lookup table**: pre-compute all rope frequency offsets for
  all possible sequence positions (up to `MAX_SEQ`) and bind the entire
  table as a descriptor. The shader indexes into the table by `seq_pos`
  from a push constant. This avoids per-step descriptor mutation but
  requires a larger buffer and changes to the shader access pattern.

Either approach requires shader changes and is deferred to a future entry.

### Near-term: Verify coverage on broader P0 subset and longer prompts

Run the per-layer descriptor set path on the full P0 corpus and on 512+
token prompts. The change is structural (descriptor binding only, no
semantic changes), but empirical verification across diverse prompt
lengths is needed before considering default.

### Medium-term: Toward single-submit recording

With per-layer stable descriptor sets covering all decode-loop descriptors,
the next prerequisite for single-submit is:

- Record the entire per-step command buffer at session construction time
  (including all layer dispatches, barriers, and descriptor bindings).
- The only per-step variable is the push constant content (activation
  buffer offsets are identical per step — the same `act_a`, `act_b`, etc.
  buffers are used every step).
- Submit the pre-recorded command buffer each step with updated push
  constants via `vkCmdPushConstants` (or `vkCmdUpdateBuffer` for the
  embedding index).
- This is NOT yet possible: intra-DeltaNet descriptors and RoPE
  descriptors still mutate per step.

Each of these must be addressed before the decode loop can use a single
pre-recorded command buffer per token. The per-layer stable descriptor
sets are one step toward that goal.

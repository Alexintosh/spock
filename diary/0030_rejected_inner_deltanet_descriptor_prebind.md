# 0030: Rejected: Intra-DeltaNet Sub-Step Descriptor Pre-Binding — Decode-State Corruption Under the Gate

## Goal

Extend `SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1` from 24 covered per-layer
descriptor sets to 30, covering the intra-DeltaNet sub-step descriptors
that remained on the old per-dispatch mutation path in diary 0029. This
was one structural prerequisite for single-submit recording; RoPE
mutation, host-side submission orchestration, and fallback readbacks
remain as additional blockers.

Covering these descriptors would mean the decode loop under the gate
performs zero per-layer descriptor mutations — every layer's bindings
would be pre-resolved at session construction time, and the old mutation
block would be entirely dead code under the gate.

## The Attempt

### New descriptor sets added (6)

The following six descriptor sets were added to the `PerLayerDescriptorSets`
struct (in `vk_session.hpp`) and pre-bound in the constructor (in
`vk_session.cpp`):

- `dn_l2_q` — L2-normalized Q input to the DeltaNet recurrent kernel
- `dn_l2_k` — L2-normalized K input to the DeltaNet recurrent kernel
- `dn_recurrent` — recurrent state bindings for the DeltaNet step kernel
- `dn_norm_gate` — norm-and-gate bindings for DeltaNet output path
- `dn_out_proj` — output projection bindings for DeltaNet
- `dn_compute_g_beta` — g/beta computation bindings for DeltaNet

These were allocated from `ds_layout_3` and pre-bound with:
- **Weight offsets**: looked up from the artifact by per-layer role string,
  same pattern as the existing 24 covered sets.
- **Activation buffer references**: the same global scratch buffers used
  in the old mutation path (`B.dn_qkv`, `B.dn_state`, `B.dn_z`,
  `B.dn_a`, `B.dn_b`, `B.act_b`).
- **Static per-layer buffer offsets**: identical to the offsets computed
  in the old per-dispatch mutation block.

The corresponding ~80 lines of `vkUpdateDescriptorSets` calls in the
decode loop's mutation block were guarded by `!per_layer_sets_enabled_`.

### What did NOT change

- **No shader changes.** The descriptor layout `ds_layout_3` was reused.
  Pipeline bindings, buffer layouts, and shader access patterns are
  identical. The only difference is when (and how many times) the
  descriptor bindings are written.
- **No pipeline layout changes.** All existing `ds_layout_3` and
  `ds_layout_32` layouts are unchanged.
- **No new buffers or weight artifacts.** Only descriptor allocation and
  binding timing changed.
- **No change to RoPE mutation.** RoPE descriptors still update per step
  (step-dependent rope frequency offset).

## Result: Parity Failure — Decode-State Corruption

## Verification

Verification for this attempt was intentionally strict: the broad descriptor
prebinding change had to preserve decode parity against the accepted baseline,
and it did not. The build succeeded, but the targeted parity command below
showed decode-state corruption rather than harmless performance noise. Because
the failure affected model output, the correct result was to revert the change
and record the negative slice instead of narrowing it in place.

### Build

The patch compiled cleanly. No warnings, no linker errors, no Vulkan
validation layer issues reported during session construction or decode.

### Parity command

```sh
SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1 \
  python3 tests/run_vk_decode_parity.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --reference tests/data/reference_tokens.jsonl \
  --ids short_correctness_001 \
  --max-new-tokens 16
```

### Failure signature

```json
{
  "status": "mismatch",
  "checked": 1,
  "failures": [
    {
      "id": "short_correctness_001",
      "first_mismatch_index": 1,
      "matched_prefix_tokens": 1,
      "expected_prefix": [271, 248068, 271, 248069],
      "actual_prefix": [271, 89454, 4384, 6813]
    }
  ]
}
```

- **First token (index 0) matches**: token 271 is common. This means the
  prefill output is correct and the first decode step produces the same
  argmax.
- **Second token (index 1) diverges**: expected 248068, got 89454. The
  entire subsequent sequence is different.
- **matched_prefix_tokens = 1**: only the initial prefill token matches.
  The decode loop produces no correct tokens beyond step 0.

### Interpretation: recurrent state mutation or descriptor aliasing

The failure pattern — correct prefill, correct first decode step, then
complete divergence — is characteristic of a state corruption that
accumulates over decode steps. Possible root causes:

1. **Recurrent state offset mismatch.** The `dn_recurrent` descriptor
   binds `B.dn_state` at a per-layer offset. If the pre-bound offset
   does not match the offset the `deltanet_recurrent.comp` shader
   actually writes at decode time, the state for layer N would overlap
   with or miss the correct segment. This would corrupt state silently
   because Vulkan does not bounds-check storage buffer accesses — the
   shader reads/writes the wrong bytes and produces plausible-looking
   but wrong output for subsequent steps.

2. **Descriptor aliasing / binding collision.** If a pre-bound descriptor
   set references a buffer region that is also referenced by another
   binding (e.g., a scratch buffer that changes meaning between sub-steps
   of the DeltaNet kernel), the pre-bound reference may not reflect the
   runtime-valid region. The intra-DeltaNet sub-step descriptors
   (`dn_norm_gate`, `dn_out_proj`, `dn_compute_g_beta`) are particularly
   suspect because they bind overlapping scratch regions that are reused
   across sub-steps — a pre-bound descriptor referencing a region that
   was valid at construction time may alias a region that was overwritten
   by an earlier sub-step.

3. **Missing barrier or memory dependency.** The old per-dispatch mutation
   path implicitly serialized descriptor updates relative to dispatches
   on the CPU timeline. Pre-binding removes those CPU-timeline sequencing
   points. If the shader relies on a host-side memory barrier happening
   between descriptor update and dispatch (which it should not, but a
   subtle timing dependency cannot be ruled out without deeper
   investigation), removing the mutation could change the observable
   behavior.

Root cause was not resolved. Analysis was not pursued further because the
fix would require a deeper rework of how the DeltaNet recurrent kernel's
sub-step descriptors map to buffer regions. The attempted approach — naive
1:1 pre-binding of the same descriptors with the same per-layer offsets —
is insufficient for these six descriptors.

## Revert

The patch was reverted from the working tree:

```sh
git restore src/runtime/vk_session.cpp src/runtime/vk_session.hpp
```

### Post-revert verification

```sh
git diff --check
```

No whitespace errors.

```sh
cmake --build build -j
```

Build passed cleanly (incremental, only the two restored files recompiled).

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

The pre-existing 24-set per-layer descriptor path (diary 0029) is
unaffected. Parity passes after revert.

## Risk Assessment

This is not a safe docs-only caveat — the failure is a **decode-state
corruption** that does not produce a Vulkan error, crash, or validation
layer warning. It silently produces wrong output. Specific risks:

- **State-offset errors are invisible to Vulkan.** Storage buffer access
  out of bounds is not trapped by the driver on any Vulkan implementation
  the author is aware of. A mis-bound `dn_recurrent` descriptor silently
  reads/writes the wrong memory. The failure would only be detected by
  output-level parity comparison — exactly what the test caught.
- **Descriptor aliasing is subtle and shader-specific.** The
  `dn_compute_g_beta`, `dn_out_proj`, and `dn_norm_gate` descriptors
  bind scratch buffers that are reused across sub-steps of the same
  DeltaNet kernel. Pre-binding them at construction time with the
  "correct" offsets may not match the state of those buffers at the
  point in the shader execution where they are bound at runtime. The
  old mutable path updated the descriptors immediately before each
  dispatch, ensuring the bindings reflected the current buffer state.
- **Mis-binding one descriptor corrupts the entire decode loop.**
  Because DeltaNet state is recurrent (each layer's output feeds the
  next layer, and each step's state feeds the next step), a single
  mis-bound descriptor for one layer at one step corrupts all subsequent
  layers and steps. The failure at step 1 (not step 0) is consistent
  with a state corruption that takes one decode step to manifest.

## Current Accepted Scope

After revert, the accepted scope is exactly diary 0029:

- **24 covered per-layer descriptor sets** (MLP/norm, attention, first-stage
  DeltaNet). These are pre-bound and stable under the gate.
- **8 intra-DeltaNet sub-step descriptors NOT covered**: `dn_split_q`,
  `dn_split_kv` (uncovered from diary 0029, not part of this attempt),
  `dn_l2_q`, `dn_l2_k`, `dn_recurrent`, `dn_norm_gate`, `dn_out_proj`,
  `dn_compute_g_beta` (the six targeted by this attempt). All eight
  remain on the old per-dispatch mutation path under the gate.
- **RoPE descriptors NOT covered** and still mutate per step.

Any future attempt to cover the intra-DeltaNet descriptors must
either:
- Investigate and fix the state-offset or descriptor-aliasing issue
  identified above, or
- Eliminate the descriptors by fusing the DeltaNet sub-steps into a
  single kernel that uses internal shared memory / subgroup
  communication instead of storage buffer round-trips, making the
  pre-binding unnecessary.

## What This Does NOT Change

- **This is NOT full GPU offload and NOT the megakernel.** The host still
  orchestrates per-layer dispatch, per-step submission, and fence waits.
  Decode argmax, diagnostic readbacks, and per-step embedding remain
  host-mediated.
- **This is NOT single-submit.** Descriptor elimination is only one
  prerequisite; RoPE mutation, host-side submission orchestration, and
  fallback readbacks remain as additional blockers. This negative result
  does not change the single-submit feasibility — it only documents that
  the naive pre-binding approach for intra-DeltaNet descriptors is
  incorrect.
- **The default path is unchanged.** The per-layer mutation path remains
  the default. The 24-set per-layer gate is opt-in and produces correct
  output. All intra-DeltaNet descriptors mutate per dispatch on both the
  default and gated paths.

## Documentation Note

This diary documents a rejected approach. The diary itself and any
updates to cross-referencing docs (NEXT_STEPS.md, runtime_strategy.md)
are commit candidates. No code changes accompany this revision.

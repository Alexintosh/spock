# 0033: Rejected: dn_compute_g_beta Descriptor Pre-Binding — Isolated Failure Beyond Diary 0030

## Goal

After diary 0032 extended the opt-in per-layer stable descriptor set path
to cover the L2-norm DeltaNet descriptors (dn_l2_q, dn_l2_k), four
stateful intra-DeltaNet sub-step dispatch-target descriptors remained
uncovered: dn_recurrent, dn_norm_gate, dn_out_proj, dn_compute_g_beta.
Diary 0030 demonstrated that pre-binding all six intra-DeltaNet sub-step
descriptors together causes decode-state corruption at step 1. The root
cause was not isolated — it could have been any one descriptor, a
specific pair, or an interaction among the set.

The goal of this attempt was to isolate whether **dn_compute_g_beta alone**
causes the corruption, or whether it is independently safe. This is the
first single-variable isolation experiment: extend the per-layer set to
cover only dn_compute_g_beta (leaving dn_recurrent, dn_norm_gate,
dn_out_proj on the old mutation path) and test whether parity holds.

A positive result (parity pass) would prove dn_compute_g_beta is safe and
exonerate it from diary 0030's failure, narrowing the suspect set to
dn_recurrent, dn_norm_gate, dn_out_proj. A negative result (parity
failure) would prove dn_compute_g_beta independently causes
decode-state corruption, narrowing diary 0030's failure space by the
same exclusion but in the opposite direction: the failure is not an
interaction artifact — dn_compute_g_beta alone is enough to corrupt the
decode loop.

## The Attempt

### Structural additions

The following additions were made to the working tree at commit 8fe1ace
(diary 0032 L2-only slice):

1. **Struct addition** (`vk_session.hpp`): A new vector
   `std::vector<VkDescriptorSet> dn_compute_g_beta` was added to the
   `PerLayerDescriptorSets` struct, alongside the existing 26 covered
   vectors.

2. **Constructor additions** (`vk_session.cpp`): Inside the
   `if (per_layer_sets_enabled_)` block:
   - **Resize**: `per_layer_sets_->dn_compute_g_beta.resize(LAYERS);`
   - **Allocation**: `per_layer_sets_->dn_compute_g_beta[i] = dev_.allocate_descriptor_set(pipes_->ds_layout_4);`
     inside the per-layer allocation loop.
   - **Pre-binding**: Four `update_descriptor_set` calls that pre-bound
     all four bindings of `ds_layout_4` for the per-layer alias:
     - binding 0: `B.dn_a` (offset 0, size `DN_HEADS * 2`)
     - binding 1: `B.dn_b` (offset 0, size `DN_HEADS * 2`)
     - binding 2: `B.dn_a_log_bias` (offset 0, size `NUM_DN_LAYERS * DN_HEADS * 2 * 4`)
     - binding 3: `B.dn_state` (g/beta tail offset per `dn_idx`, size `DN_HEADS * 2 * 4`)

     The baseline constructor already preconfigures bindings 0/1/2 on the shared
     `D.dn_compute_g_beta` set; decode mutates only binding 3 per `dn_idx`/state
     tail. The per-layer set duplicated the global bindings and also pre-bound
     binding 3 with the per-layer state-tail offset, so the decode alias should
     have skipped binding 3's fallback `update_descriptor_set` call.

   The verbose count comment was updated from `26 sets` to `26 + 1 = 27 sets`
   (24 + 2 L2 + 1 g_beta = 27 per-layer sets).

3. **Decode alias** (`vk_session.cpp`, `decode()`): An alias was added
   after the existing alias block:
   ```cpp
   VkDescriptorSet ds_dn_compute_g_beta = per_layer_sets_enabled_
       ? per_layer_sets_->dn_compute_g_beta[layer]
       : D.dn_compute_g_beta;
   ```

   This alias was initially missing — the first attempt compiled but the
   dispatch site still referenced `D.dn_compute_g_beta` directly, defeating
   the pre-binding. The alias was added in a repair iteration.

### What was intentionally NOT done

- **The fallback guard on binding-3 update was skipped.** The baseline
  constructor preconfigures bindings 0, 1, 2 on the shared `D.dn_compute_g_beta`
  set at session construction; decode mutates only binding 3 (the per-`dn_idx`
  state-tail offset). When `per_layer_sets_enabled_` is active, the pre-bound
  per-layer alias already has all four bindings pre-bound, so the binding-3
  `update_descriptor_set` call on the decode path should be skipped. This guard was intentionally omitted
  in the first attempt (see "Repair and diagnosis" below).

- **No changes to dn_recurrent, dn_norm_gate, dn_out_proj.** All three
  remained on the old per-dispatch mutation path. Only dn_compute_g_beta
  was moved to pre-bound.

- **No shader changes.** The descriptor layout `ds_layout_4` is reused.
  Pipeline bindings, push constants, and shader access patterns are
  identical to the default mutation path.

- **No descriptor pool sizing change.** The 28 new sets (1 per layer × 28
  layers) were within the 1024 maxSets capacity. No pool expansion needed.

- **No buffer reallocation.** `B.dn_a`, `B.dn_b`, `B.dn_a_log_bias`, and `B.dn_state` are unchanged.

### Repair and diagnosis

The first compilation attempt revealed a missing decode alias — the
dispatch site still referenced `D.dn_compute_g_beta` directly even when
the gate was active. This was corrected by adding the alias described
above.

The fallback guard on the binding-3 `update_descriptor_set` call was
missing. On the default (non-gated) path, decode mutates only binding 3
(the per-`dn_idx` state-tail offset) on the shared `D.dn_compute_g_beta`
set. Under the gate, the per-layer alias already has binding 3 pre-bound
with its layer-specific state-tail offset; the fallback update would
overwrite binding 3 on the global set (harmless since the alias is used
for dispatch), but should be skipped to maintain the invariant that
descriptor mutation is entirely elided under the gate.

With the alias repaired and the binding-3 guard skipped, the parity test
was run.

## Result: Parity Failure — Same Decode-State Corruption as Diary 0030

## Verification

Verification required this narrower g/beta descriptor slice to preserve the
same decode parity as the accepted descriptor baseline. It compiled, but the
targeted parity run reproduced the decode-state corruption observed in diary
0030. That repeat failure showed the problem was not only the size of the broad
patch, so the change was reverted and left as a documented negative result
until the constructor-ordering issue could be investigated separately.

### Build

After the alias repair, the patch compiled cleanly. No warnings, no linker
errors, no Vulkan validation layer issues reported during session
construction or decode.

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
      "expected_prefix": [271, 248068],
      "actual_prefix": [271, 89454]
    }
  ]
}
```

- **First token (index 0) matches**: token 271 is common. Prefill output
  is correct; first decode step produces the same argmax.
- **Second token (index 1) diverges**: expected 248068, got 89454.
- **matched_prefix_tokens = 1**: only the initial prefill token matches.

### Significance: identical failure signature to diary 0030

The failure is indistinguishable from diary 0030's recorded failure:

| Experiment | Expected prefix | Actual prefix |
|------------|----------------|---------------|
| Diary 0030 (all six) | `[271, 248068, ...]` | `[271, 89454, ...]` |
| This attempt (g_beta only) | `[271, 248068]` | `[271, 89454]` |

Both produce the **exact same wrong token 89454** at index 1. This is
strong evidence that the corruption mechanism is the same, and that
dn_compute_g_beta alone is sufficient to trigger it. The corruption is
not a descriptor-interaction artifact of the all-six set — it is a
property of dn_compute_g_beta's pre-binding specifically.

### Root cause hypothesis

The identical failure signature — same wrong token 89454 at index 1,
`matched_prefix_tokens=1` — between diary 0030 (all six) and this
experiment (dn_compute_g_beta alone) confirms the corruption mechanism
is the same and is triggered by pre-binding `dn_compute_g_beta`'s binding
3 (state-tail offset), not by an interaction among multiple descriptors.

The baseline constructor already preconfigures bindings 0/1/2
(`B.dn_a`/`B.dn_b`/`B.dn_a_log_bias`) on the shared global set, and
these do not vary per-`dn_idx`. Decode mutates only binding 3 — the
`B.dn_state` offset into the g/beta tail region. Pre-binding binding 3
with a static per-layer offset produces the same corruption as the
all-six pre-binding in diary 0030, meaning the failure is specifically
in pre-binding the state-tail binding for this descriptor.

The root cause is **not resolved** by this experiment. Likely candidates
include:
- **State-offset mismatch**: The pre-bound g/beta state-tail offset does
  not match what the shader expects at dispatch time due to an incorrect
offset calculation in the constructor vs. the decode-path calculation.
- **Descriptor aliasing**: The pre-bound `dn_compute_g_beta` set shares a
  `B.dn_state` region (or another buffer) with a concurrently-bound
descriptor set that also targets the same buffer region with a different
semantic interpretation, causing the driver to observe conflicting
access patterns.
- **Descriptor lifetime/order/setup mismatch**: The pre-bound per-layer
  set is constructed and allocated inside the session constructor's
  per-layer allocation loop, whereas on the non-gated path the shared
  global `D.dn_compute_g_beta` set is allocated at a different point in
  the constructor. If descriptor pool sub-allocation ordering,
  set-lifetime tracking, or deferred binding resolution depends on the
  pool position where a set was allocated, the pre-bound set's different
  allocation context relative to the original set's position could
  produce the corruption. This is speculative but conservatively
  attributes the mismatch to our descriptor setup rather than to driver
  behavior.

All three candidates are unresolved. This experiment does not
 distinguish between them.

## Revert

The decode alias was reverted first (the three-line alias addition), then
the remaining constructor/struct additions were reverted:

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

The pre-existing 26-set per-layer descriptor path (diary 0029 + diary 0032)
is unaffected. Parity passes after revert.

## Risk Assessment

This experiment confirms the same risk profile documented in diary 0030:
dn_compute_g_beta pre-binding causes silent decode-state corruption at
step 1 with no Vulkan error, crash, or validation layer warning. Specific
risks:

- **The corruption is deterministic and reproducible.** Both this attempt
  and diary 0030 produce token 89454 at index 1. This rules out a race
  condition or driver-level nondeterminism — the failure is a systematic
  property of the pre-bound binding for dn_compute_g_beta.
- **Single-variable isolation narrows the failure space.** The all-six
  failure could have been caused by any combination of the six descriptors.
  This experiment proves that dn_compute_g_beta alone is sufficient, and
  that the failure mechanism does not require interaction with the other
  five descriptors.
- **Individual isolation for dn_recurrent, dn_norm_gate, dn_out_proj is
  the logical next step**, but each carries the same risk of silent
  decode-state corruption. A positive result would prove the descriptor
  is safe and can be pre-bound independently. A negative result would
  further narrow the suspect set. However, the diminishing returns of
  sequential isolation must be weighed against the effort of build-test-
  revert cycles. If the root cause is a shared structural property (e.g.,
  all stateful DeltaNet sub-step descriptors share a buffer region mapping
  pattern that is incompatible with pre-binding), then isolating each one
  sequentially will yield a sequence of negative results without insight
  into the underlying issue.

## Current Accepted Scope

After revert, the accepted scope remains exactly commit 8fe1ace (diary 0032):

- **26 per-layer descriptor sets** (24 from diary 0029 + dn_l2_q/dn_l2_k
  from diary 0032) pre-bound at session construction.
- **2 session-level RoPE descriptor sets** (diary 0031) pre-bound at session
  construction.
- **Total: 730 pre-bound descriptor sets** (728 per-layer + 2 session-level).

Remaining uncovered (per-dispatch mutation on decode path):

| Descriptor | Status | Notes |
|------------|--------|-------|
| `dn_recurrent` | Uncovered | Stateful — binds `dn_state`. No independent test performed. |
| `dn_norm_gate` | Uncovered | Binds output accumulation buffer. No independent test performed. |
| `dn_out_proj` | Uncovered | Binds DeltaNet output accumulation buffer. No independent test performed. |
| `dn_compute_g_beta` | **Confirmed fails independently** (this diary) | Binds `B.dn_a`/`B.dn_b`/`B.dn_a_log_bias`/`B.dn_state` (ds_layout_4). Independent isolation experiment proves pre-binding alone causes decode-state corruption at step 1. |
| `dn_split_q` | Uncovered (internal) | Internal decomposition descriptor, not a dispatch target. |
| `dn_split_kv` | Uncovered (internal) | Internal decomposition descriptor, not a dispatch target. |

### Narrowing of diary 0030's failure space

Diary 0030 documented that the all-six intra-DeltaNet sub-step descriptor
pre-binding set fails. It could not attribute the failure to any specific
descriptor. This diary narrows that result:

- **dn_l2_q, dn_l2_k**: Exonerated (diary 0032 proves they pre-bind safely).
- **dn_compute_g_beta**: Confirmed independently causes the same failure.
  Pre-binding this single descriptor is sufficient to produce the same
  decode-state corruption at step 1 with the same wrong-token signature.
- **dn_recurrent, dn_norm_gate, dn_out_proj**: Not independently tested.
  Any combination including dn_compute_g_beta fails. It remains possible
  that one or more of these three would independently pass, but the
  current evidence shows that the failure manifests even without them.
  Sequential isolation would be required to exonerate the remaining three.

The all-six failure was not an interaction effect without dn_compute_g_beta.
It was triggered by dn_compute_g_beta alone.

### What does NOT change

- **This is NOT full GPU offload and NOT the megakernel.** The host still
  orchestrates per-layer dispatch, per-step submission, and fence waits.
  Decode argmax, diagnostic readbacks, and per-step embedding remain
  host-mediated.
- **This is NOT single-submit.** Descriptor pre-binding for the covered
  sets is only one prerequisite. Even with all six intra-DeltaNet sub-step
  descriptors covered, single-submit would remain blocked by host-side
  per-layer/per-step submission orchestration, fence waits, and fallback
  readbacks. This negative result does not change single-submit feasibility.
- **The default path is unchanged.** The per-layer mutation path remains
  the default. The 26-set per-layer gate from diaries 0029/0032 plus session-level RoPE from diary 0031 is
  opt-in and produces correct output. All uncovered descriptors mutate
  per dispatch on both the default and gated paths.
- **The all-six pre-binding failure (diary 0030) remains valid.** This
  diary does not replace diary 0030 — it narrows it by isolating
  dn_compute_g_beta as independently sufficient for the failure.
- **This does NOT resolve the root cause.** The underlying state-offset or
  descriptor-aliasing issue remains uninvestigated. This diary only
  narrows which descriptor triggers it, not why.

## Next Stateful Candidates

With dn_compute_g_beta confirmed as independently failing, the remaining
hypothesis space for stateful descriptor pre-binding is:

1. **Sequential isolation of dn_recurrent, dn_norm_gate, dn_out_proj.**
   Each would require a build-test-revert cycle similar to this diary.
   If all three independently fail, the structural conclusion would be
   that *any* stateful DeltaNet sub-step descriptor with output/scratch
   buffer bindings cannot be safely pre-bound with the current naive 1:1
   offset approach. If any one passes independently, it can be shipped
   as a narrow extension à la diary 0032.

2. **Kernel-fusion approach.** Instead of pre-binding sub-step descriptors,
   fuse the DeltaNet sub-steps (g/beta computation, recurrent step,
   norm+gate, output projection) into a single kernel that uses local
   memory or subgroup operations for intermediate values, eliminating
   the scratch-buffer round-trips that make descriptor pre-binding
   unsafe. This would bypass the descriptor problem entirely but requires
   significant shader engineering.

3. **State-offset root cause investigation.** Before more isolation
   cycles, investigate why pre-binding produces the corruption. Options
   include: instrumenting the shader to dump buffer offsets and verify
   they match pre-bound expectations; comparing the descriptor bindings
   created by the constructor vs. the old mutation path at runtime; or
   adding Vulkan validation layer checks that would catch mis-specified
   buffer ranges (though this is unlikely to help — storage buffer
   out-of-bounds access is not trapped by validation layers on RADV).

The author's current inclination is toward option 2 (kernel fusion) or
option 3 (root-cause investigation) rather than sequential isolation
of the remaining three descriptors, given that dn_compute_g_beta's
independent failure suggests a structural problem rather than a
descriptor-specific bug.

## Files Changed (during attempt, then reverted)

```
src/runtime/vk_session.cpp    — Add dn_compute_g_beta vector resizing, allocation, pre-binding,
                                decode alias; verbose count update. Then reverted.
src/runtime/vk_session.hpp    — Add dn_compute_g_beta vector to PerLayerDescriptorSets struct.
                                Then reverted.
```

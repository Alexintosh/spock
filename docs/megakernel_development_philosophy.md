# Megakernel Development Philosophy

This document is the operating philosophy for the Vulkan-native megakernel
track. It exists to keep the project honest about the actual target:
`Qwen/Qwen3.5-0.8B` decode on `AMD Radeon RX 6750 XT (RADV NAVI22)`, using a
Vulkan persistent-dispatch path where the hot decode loop remains GPU-resident.

For the phase-by-phase rationale that ties each prerequisite to the final
archived inference target, see `docs/megakernel_phase_rationale.md`.
For the current execution map, including the proof state, next gates, test
references, and risk register for the RX 6750 XT target, see
`docs/rx6750xt_megakernel_execution_map.md`.

The central rule is that every fused step must be backed by a smaller,
reproducible gate that explains failures. The project can move slowly, but it
must not move blindly.

## End Goal

The target is not a generic Vulkan backend. The target is a Vulkan-native,
RX 6750 XT-oriented inference path for `Qwen/Qwen3.5-0.8B` that can run basic
test inference with the hot decode path resident on the GPU.

The strongest version of that target is a persistent decode megakernel:

- model state stays GPU-resident through the decode loop;
- layer work, residual updates, token-mixer state, MLP, final norm, LM head, and
  token selection are fused as far as Vulkan and RADV allow;
- CPU involvement is reduced to setup, launch, and final result collection;
- correctness is archived with reproducible tests and artifacts.

If full persistent dispatch is not robust or not faster on this driver/GPU
stack, the honest fallback is the strongest single-submit or fused Vulkan path.
The project should not claim megakernel parity unless persistent dispatch is
actually correct, stable, and worth its complexity.

## Definition Of The Target Path

The target path is a decode-first, batch-1, model-specific engine:

- one pinned model family and artifact layout;
- one known GPU and RADV driver stack;
- fp16 weights and activations, with fp32 where the architecture needs state or
  accumulation stability;
- a persistent Vulkan dispatch for the decode hot path if the software barrier
  remains viable under real work;
- no host readback/upload bridge between layer sub-blocks in the hot path;
- final proof as archived basic test inference, not just isolated probe output.

The project is allowed to keep diagnostic and fallback paths. Those paths are
valuable. They are not the target path unless they are the path used by the
archived inference proof.

## Non-Goals For This Track

The megakernel track should not spend effort on generality until the specific
target works:

- no multi-model abstraction;
- no multi-GPU support;
- no batching or serving scheduler;
- no cross-vendor portability promise;
- no quantization work unless it directly supports the target proof;
- no premature performance claims from synthetic probes.

These are not dismissed forever. They are deferred because each one expands the
state space before the single model/single GPU path has proven correctness.

## Why Not Start With The Megakernel

A megakernel hides bugs. Once normalization, attention or DeltaNet, MLP,
residual updates, LM head, token selection, barriers, and scratch reuse are
fused into one shader, a wrong token gives very little information about where
the error entered.

The project therefore builds the pieces in an order that preserves
observability:

- first prove artifact loading and parity baselines;
- then prove ordinary Vulkan decode correctness;
- then remove CPU round trips from known paths;
- then prove persistent synchronization separately;
- then add real fp16/fp32 compute to persistent probes;
- then feed those probes real model weights and real captured activations;
- only then compose a full layer-shaped persistent kernel.

This is a correctness strategy, not a lack of ambition. The megakernel is the
goal, but it should be assembled from validated pieces rather than debugged as
one opaque blob.

There is a second reason: Vulkan does not provide a CUDA-style cooperative grid
barrier. The project's persistent path depends on a software global barrier
implemented with coherent storage buffers and bounded spinning. If that
primitive fails only after we have fused the whole model, the failure mode will
look like random model drift. Proving synchronization separately keeps
correctness failures local.

## Dependency Ladder

The current ladder is:

1. Artifact and parity freeze.
   Freeze model artifacts, tensor layouts, reference tokens, and what "correct"
   means for greedy decode.

2. Ordinary Vulkan decode.
   Bring up an explicit, debuggable Vulkan path with all model pieces wired,
   even if it uses many dispatches and host orchestration.

3. GPU-resident handoffs.
   Remove host readback/upload bridges where correctness is already understood:
   prefill collection, chunk-prefill output handoff, generated-token handling,
   descriptor stability, and command-buffer grouping.

4. Software global barrier viability.
   Vulkan has no native cross-workgroup barrier inside a dispatch. The
   persistent path depends on a bounded software barrier that does not deadlock,
   lose visibility, or trigger device loss on RADV.

5. Persistent skeletons.
   Exercise the barrier with decode-shaped work, first with synthetic payloads,
   then with real fp16/fp32 dot products.

6. Real model weights.
   Load repacked model tensors into persistent probes and mirror shader
   reduction order exactly enough to separate layout bugs from math differences.

7. Real activation handoffs.
   Feed persistent probes values captured from the real decode pipeline, so the
   probe validates production-shaped data rather than only synthetic patterns.

8. Layer-shaped persistent kernel.
   Add RMSNorm, token mixer, MLP, residual stream updates, scratch reuse, and
   per-layer state transitions in one persistent layer probe.

9. Full 24-layer persistent decode.
   Run the layer loop under persistent dispatch with bounded barrier use and
   stable GPU-resident state.

10. LM head and token loop.
    Keep final norm, logits, argmax or sampling, and next-token handoff on the
    GPU-resident path.

11. Archived basic test inference.
    Store the command, artifacts, expected output, environment, and test result
    that prove the target path can generate correctly end to end.

This order is intentional. RMSNorm depends on the same residual-stream contract
the MLP consumes. MLP handoff depends on model-weight layout and captured
activation extraction. Layer composition depends on barrier viability and
scratch lifetime discipline. Multi-layer decode depends on per-layer state
transitions being correct. LM-head/token selection should come late because a
wrong token is the least diagnostic failure signal in the system.

## How The Pieces Fit Together

The project has three classes of work that must converge:

1. **Reference and artifact work.**
   This freezes what the model is, how tensors are named, what precision is
   stored, and what output counts as correct. Without this, every GPU mismatch
   could be blamed on an artifact or reference ambiguity.

2. **Conventional Vulkan runtime work.**
   This proves the model can run through explicit dispatches with enough
   observability to capture hidden states, compare components, and isolate
   layer drift. It is the diagnostic spine of the project.

3. **Persistent probe work.**
   This turns known-correct layer pieces into single-dispatch, barrier-staged
   compute. It starts with synchronization, then projection math, then MLP,
   then RMSNorm, then layer-shaped composition, then the full decode loop.

The conventional runtime is not wasted work. It is the source of real captured
activations and the reference behavior for persistent probes. The probes are
not throwaway toys either. Each one validates a specific contract that the final
megakernel will rely on.

## Current Position

As of diary 0120, the project has not reached the Vulkan-native megakernel.
The current persistent path is a validated sub-block track:

- the software global barrier has survived synthetic and decode-shaped probes;
- persistent fp16/fp32 projection skeletons can use real repacked model weights;
- `vk_persistent_mlp_probe` runs gate, up, SiLU-gated activation, down, and
  optional residual update in one dispatch;
- captured fp16 MLP handoffs are gated by per-row fp16 output comparison;
- layer selection exists for the persistent MLP probe;
- the layer-20 CPU-vs-GPU precision boundary case is handled by an explicit,
  opt-in fp16 ULP tolerance policy;
- pre-MLP RMSNorm is integrated into the persistent MLP probe and validated with
  real weights at model width;
- captured runtime `mlp_normed_fp16` matches persistent Stage 0 RMSNorm output
  exactly for layer 0, step 1;
- captured runtime gate/up/product/down boundaries are mapped: raw gate
  projection max 1 ULP, raw up projection max 2 ULP, activation product max
  2 ULP, and down output max 2 ULP;
- captured runtime post-residual MLP output is bounded by explicit max and
  population gates rather than silently accepted;
- layer 20 adds representative mid-network RMSNorm+MLP population evidence;
- runtime `mixer_output_fp16` capture plus `vk_residual_add_probe` closes the
  first token-mixer residual equation exactly:
  `input_hidden + mixer_output -> mixer_residual`;
- `vk_matvec_probe` proves the layer-0 DeltaNet output projection exactly:
  `dn_gated_fp16 + layer.0.delta_out_proj -> dn_out_fp16` with zero
  mismatches, closing the matvec handoff from gated activation to mixer
  output at full model width.
- `vk_deltanet_norm_gate_probe` proves the preceding layer-0 norm-gate
  equation exactly:
  `dn_core_fp16 + dn_z_fp16 + layer.0.delta_norm -> dn_gated_fp16`.
- `vk_matvec_probe` also proves the layer-0 z projection exactly:
  `dn_input_norm_fp16 + layer.0.delta_in_proj_z -> dn_z_fp16`.
- runtime `dn_qkv_raw_fp16` capture plus `vk_matvec_probe` prove the layer-0
  raw qkv projection exactly:
  `dn_input_norm_fp16 + layer.0.delta_in_proj_qkv -> dn_qkv_raw_fp16`.
- runtime `dn_a_fp16` and `dn_b_fp16` captures plus `vk_matvec_probe` prove the
  layer-0 A/B projections exactly.
- runtime `dn_g_bits`/`dn_beta_bits` plus `vk_deltanet_g_beta_probe` prove the
  layer-0 g/beta scalar computation exactly.
- runtime pre-recurrent state capture plus `vk_deltanet_recurrent_probe` prove the
  layer-0 recurrent core output exactly:
  `q/k/v + g/beta + state_pre -> dn_core_fp16` with zero mismatches.
- `vk_deltanet_mixer_probe` proves the complete layer-0 DeltaNet mixer pipeline
  produces exact fp16-bit-identical output and residual when all eleven stages
  run in a single Vulkan submit:
  `dn_input_norm -> qkv/z/a/b projections -> conv1d -> L2 q/k -> g/beta ->
   recurrent -> norm-gate -> out_proj -> residual_add` with zero mismatches
  on both mixer_output and mixer_residual (diary 0113).

- `vk_persistent_layer0_probe` establishes the first layer-shaped persistent
  scaffold with `local_size_x=128` and 82 workgroups for the post-mixer tail:
  `mixer_residual -> post_norm RMSNorm -> MLP gate/up -> SiLU product -> down
  -> residual add -> post_mlp` (diary 0114).
- `vk_persistent_layer0_probe --mode projections` gates the stateless DeltaNet
  projection prefix `dn_input_norm -> qkv_raw, z, a, b` at 128 lanes and 82
  workgroups. Exact comparison fails on 9 rows at 1 ULP (reduction-order boundary);
  `--projection-fp16-ulp-tolerance 1` passes (diary 0116).
- `vk_persistent_layer0_probe --mode conv-l2` gates the next persistent DeltaNet
  boundary: `dn_qkv_raw_fp16 + conv_state_pre + delta_conv -> conv-mutated q/k/v`
  with one software global barrier between conv mutation and q/k normalization.
  Q, K, and V all pass exact fp16 comparison at 128 lanes and 82 workgroups
  (diary 0117).
- `vk_persistent_layer0_probe --mode g-beta` gates the DeltaNet scalar branch
  from captured `dn_a_fp16`/`dn_b_fp16` plus repacked `delta_a_log`/`dt_bias`
  into exact g/beta fp32 bit patterns with zero mismatches (diary 0118).
- `vk_persistent_layer0_probe --mode recurrent` gates the recurrent core inside
  `persistent_layer0_probe.comp` from captured q/k/v, exact g/beta bits, and
  captured pre-update fp32 state. The output matches `dn_core_fp16` exactly with
  zero fp16 ULP drift and no software global barrier (diary 0119).
- `vk_persistent_layer0_probe --mode mixer-tail` gates norm-gate, output
  projection, and first residual add inside `persistent_layer0_probe.comp`.
  The persistent output projection has two exact mismatches at max 1 fp16 ULP
  from 128-lane reduction order, the ULP-1 gate passes, and the final residual
  handoff is exact (diary 0120).
This is meaningful progress toward the target. The full DeltaNet mixer for
layer 0 is now closed at both the unit-gate and end-to-end composed levels.
Every sub-block from `dn_input_norm_fp16` through `mixer_residual_fp16` has
independent exact gates and the composed probe confirms they chain correctly.
The first layer-shaped persistent scaffold is validated:
`vk_persistent_layer0_probe` runs the post-mixer tail at 128 lanes with the same
bounded precision policy as the 64-lane MLP probe. The remaining target pieces
are still large: DeltaNet mixer integration into the persistent layer shader,
attention-layer coverage, bounded multi-layer persistent decode, 24-layer
persistent decode, final norm, LM head, token selection, and archived
end-to-end inference.

The DeltaNet backward-validation ladder is complete for layer 0, both as
individual unit gates (diaries 0099-0112) and as a composed end-to-end probe
(diary 0113).

## Why RMSNorm + MLP Came First

The persistent MLP probe was the right first center of gravity because it
exercised the same kinds of constraints that the full megakernel will face:

- multiple dependent compute stages inside one dispatch;
- fp16 storage with fp32 accumulation;
- real model-weight loading;
- scratch buffers with strict producer/consumer lifetimes;
- residual-stream semantics;
- software global barriers between stages;
- CPU-vs-GPU precision boundaries that must be described, not hidden.

Adding RMSNorm before MLP was not a detour. In the real layer pipeline, the MLP
does not consume an arbitrary vector. It consumes a normalized residual stream.
The final layer-shaped persistent probe needs this sequence:

```
raw residual -> RMSNorm -> mixer/MLP input -> MLP -> residual update
```

Testing MLP without RMSNorm proved the matvec and activation stages. Testing
RMSNorm inside the same persistent dispatch started validating the actual layer
boundary. That gave the project a bounded downstream consumer for token-mixer
work: once the first residual add produces `mixer_residual_fp16`, the existing
RMSNorm+MLP gates can explain the rest of the layer.

## Why DeltaNet Is The Current Focus

The current implementation focus is DeltaNet, not because the MLP path is
finished forever, but because the layer-shaped persistent probe cannot be
honest until the first residual add is produced by real token-mixer compute.

The attention/DeltaNet side has more moving state than the MLP:

- attention layers touch KV cache and RoPE state;
- DeltaNet layers touch recurrent state, short convolution state, and chunk or
  recurrent update rules;
- both paths feed the residual stream consumed by the MLP path;
- both have more opportunities for state aliasing and descriptor offset bugs.

For that reason, the project did not start persistent layer composition with
the token mixer. The MLP/RMSNorm path was narrower, but it validated barriers,
scratch staging, real weights, and residual semantics. Now that those contracts
exist, DeltaNet can be integrated against a known downstream consumer instead
of being debugged together with the MLP.

The DeltaNet gates intentionally walk backward from the value consumed by the
rest of the layer:

1. `mixer_output_fp16` plus `input_hidden_fp16` must reproduce
   `mixer_residual_fp16`.
2. `dn_gated_fp16 + delta_out_proj` must reproduce `dn_out_fp16`.
3. `dn_core_fp16 + dn_z_fp16 + delta_norm` must reproduce `dn_gated_fp16`.
4. `dn_input_norm_fp16` must reproduce the stateless projections:
   qkv raw, z, a, and b.
5. `dn_a_fp16`, `dn_b_fp16`, `delta_a_log`, and `delta_dt_bias` must reproduce
   g/beta exactly.
6. raw qkv must be advanced through conv1d mutation and q/k L2 normalization.
7. q/k/v plus g/beta and recurrent state must reproduce `dn_core_fp16` (done: diary 0112).
8. All eleven stages composed in one submit must reproduce `mixer_output` and
   `mixer_residual` exactly (done: diary 0113).

This order matters because each new gate removes one possible explanation for a
future recurrent mismatch. If the recurrent probe fails after qkv, z, a/b,
g/beta, norm-gate, output projection, and residual add are all independently
closed, the failure has to live in recurrent state handling, q/k/v preparation,
state decay/update, q scaling, or output accumulation. That is a small enough
debugging surface to justify moving the work into a persistent layer probe.

The same rule applies to attention layers later. They should not be fused into
the persistent decode loop until their KV-cache, RoPE, score/softmax/value, and
output-projection contracts can explain failures locally.

## How The Pieces Fit The Final Megakernel

The final RX 6750 XT megakernel is not a single shader written in one step. It
is the convergence of four validated tracks:

1. **Observable conventional runtime.**
   This path remains the source of captured checkpoints. It tells us what the
   model actually did at a layer boundary and gives every standalone probe real
   input and expected output tensors.

2. **Exact or bounded component gates.**
   Matvec, residual add, RMSNorm, MLP, DeltaNet scalar branches, conv/L2, and
   recurrent updates each need their own runnable proof. These gates define the
   contracts the megakernel is allowed to rely on.

3. **Persistent synchronization and scratch discipline.**
   Barrier probes and persistent skeletons answer whether RADV on the RX 6750
   XT can keep enough workgroups resident and coherent for a long-running
   decode dispatch. Persistent MLP already exercises multi-stage scratch
   lifetimes; DeltaNet recurrent and layer-shaped probes must do the same for
   stateful token mixing.

4. **Layer and decode composition.**
   Only after one layer is explainable should the project widen to
   representative layers, then bounded multi-layer decode, then all 24 layers,
   then final norm, LM head, token selection, and the GPU-resident token loop.

Every piece exists to reduce uncertainty before composition. The runtime gives
captures, captures become fixtures, fixtures gate standalone kernels,
standalone kernels become persistent stages, persistent stages become a layer,
layers become decode, and decode plus LM-head/token selection becomes archived
basic inference.

## Quality Bar

The plan does not compromise code quality. It intentionally avoids two common
failure modes:

- building an impressive fused shader before the contracts are known;
- adding permissive tolerances that make tests green without explaining the
  numerical contract.

The quality bar for this track is:

- small probes must have narrow, named correctness gates;
- JSON probe output must expose enough diagnostics to explain failures;
- default behavior must remain strict unless a documented policy says
  otherwise;
- all tolerances must be opt-in, bounded, and reported;
- fixtures must represent real handoff data when a claim depends on real model
  behavior;
- documentation must say what a probe proves and what it explicitly does not
  prove;
- performance measurements must include environment, command, commit, warmups,
  timed runs, and whether timings are host or GPU-side.

This is slower than writing the final shader first, but it is a higher-quality
route to a result that can be trusted.

## Risk Register

### Software Global Barrier

The barrier is the largest architectural risk. Vulkan does not guarantee a
global synchronization primitive within one dispatch. The current barrier is an
empirical RADV/RX 6750 XT strategy. It must be stress-tested under increasingly
real workloads, not assumed valid forever.

Mitigation:

- keep bounded spin limits and failure counters;
- run synthetic, memory-payload, model-width, and real-compute probes;
- treat device loss, timeout, or nonzero failure counters as release blockers;
- preserve a single-submit fallback path as an honest alternative.

### Occupancy And Residency

Persistent dispatch only works if all resident workgroups needed by the barrier
can make progress. Workgroup count, register pressure, LDS use, scratch buffers,
and shader length can change that.

Mitigation:

- keep workgroup counts explicit in probe output;
- test the Luce-derived 82-workgroup shape and smaller shapes;
- avoid large LDS allocations until measured;
- re-test barrier behavior after adding each major stage.

### Precision Drift

The target uses fp16 storage on hardware without native bf16. CPU reference
math and GLSL math can differ at rounding boundaries.

Mitigation:

- compare the downstream fp16 contract by default;
- keep fp32 checksums as diagnostics, not always as gates;
- allow bounded ULP tolerance only when documented and explicitly requested;
- keep GPU-vs-GPU gates exact.

### State Aliasing

The final megakernel will reuse scratch buffers aggressively. A stale write or
wrong offset can appear as model drift many layers later.

Mitigation:

- introduce scratch reuse only after single-purpose scratch is correct;
- add generation counters or trace fields in probes where useful;
- validate each layer boundary with captured fixtures before multi-layer runs.

### Documentation Drift

The diary and docs are part of the correctness system. If they claim more than
the tests prove, later work will optimize the wrong thing.

Mitigation:

- diary entries must include verification and limitations;
- docs must distinguish runtime infrastructure from final megakernel progress;
- every milestone should name the next smallest gate.

## Test Strategy

The tests should follow the same ladder as the implementation:

- artifact tests validate tensor names, dtype, shape, offsets, and exact bytes;
- reference tests validate token parity and component dumps;
- conventional Vulkan tests validate layer-by-layer decode behavior;
- extraction tests validate that component dumps become raw fp16 fixtures
  without bit changes;
- barrier tests validate synchronization, timeout behavior, and generation
  counters;
- persistent skeleton tests validate fp16/fp32 projection math with synthetic
  and real weights;
- persistent MLP tests validate stage order, scratch lifetime, output equality,
  residual addition, layer selection, captured inputs, and tolerance policy;
- RMSNorm tests validate formula, weight role, fp16 output rounding, generation
  count, and residual input preservation;
- DeltaNet projection tests validate each stateless branch from captured
  normalized input before recurrent state is involved;
- DeltaNet scalar tests validate g/beta with exact fp32 bit fixtures because the
  runtime and probe run the same GPU shader;
- DeltaNet conv/L2 tests should validate qkv mutation and q/k normalization
  from the raw qkv checkpoint before the recurrent core is blamed;
- DeltaNet recurrent tests consume already-gated q/k/v and g/beta inputs
  and compare against captured `dn_core_fp16` (gate closed: diary 0112, exact);
- layer-shaped tests should validate captured pre/post checkpoints before
  multi-layer execution;
- final inference tests must archive command, commit, artifact, environment, and
  generated tokens.

For every new fused stage, the minimum acceptable test is:

1. a small synthetic smoke that is easy to debug;
2. a real-weight smoke at reduced dimensions;
3. a model-width real-weight run;
4. a captured activation fixture when the stage consumes runtime state;
5. a negative or parser test for any new option or tolerance.

## Milestone Policy

Milestones should be closed only when their verification artifact can be rerun.
A diary entry without a runnable command is a note, not an archived milestone.

The next milestones should be:

1. ~~validate the DeltaNet recurrent core against captured `dn_core_fp16` using~~
   ~~already-gated q/k/v and g/beta inputs~~ (done: diary 0112, exact gate);
2. produce the full layer-0 DeltaNet mixer output without substituting captured
   intermediate tensors after `dn_input_norm_fp16`;
3. compose a layer-shaped persistent probe with token mixer, first residual
   add, post-mixer RMSNorm, MLP, and second residual update;
4. sweep the layer-shaped probe across representative DeltaNet and attention
   layers only after layer 0 is explainable;
5. run bounded multi-layer persistent decode with captured checkpoint gates;
6. extend to all 24 layers only after smaller multi-layer runs explain
   failures locally;
7. add final norm, LM head, token selection, and device-resident next-token
   handoff;
8. archive the first basic test inference from the target path.

## What The Current Probes Prove

### Barrier Probe

`vk_barrier_probe` proves that a bounded software global barrier can coordinate
multiple workgroups on the target Vulkan stack. Decode-shaped and model-width
payload modes make the workload more relevant to the eventual megakernel.

It does not prove inference, model math, layer semantics, or full persistent
decode. It only answers whether the synchronization primitive is viable enough
to keep investing.

### Persistent Decode Skeleton

`vk_persistent_decode_skeleton` combines the barrier pattern with fp16 inputs,
fp16 weights, and fp32 accumulation. Later entries extend it to real repacked
weights, multiple roles, row-strided coverage, and model-width shapes.

It proves that persistent dispatch can carry real tensor-shaped compute. It does
not prove RMSNorm, MLP activation, token mixer behavior, residual-stream
semantics, LM head, or token generation.

### Persistent MLP Probe

`vk_persistent_mlp_probe` is the first multi-stage persistent compute probe:
gate projection, up projection, SiLU-gated activation, down projection, and
optional residual update inside one dispatch. It now covers full real layer-0
MLP dimensions with real model weights.

It proves that one important layer sub-block can run under persistent
barrier-synchronized staging. RMSNorm-before-MLP is now integrated (diary 0090).
It is still not a full layer, because attention or DeltaNet,
and final token generation are outside this probe.

### Component FP16 Extraction

`tools/extract_component_fp16.py` bridges `spock-decode --dump-step-components`
JSON into raw fp16 files that probes can ingest. It is intentionally a diagnostic
format bridge, not runtime machinery.

It proves that real pipeline captures can be moved into standalone probes
without changing the bit pattern.

### Captured FP16 Handoff Gate

Diary 0086 validates the persistent MLP probe with a real captured layer-0,
step-1 `mixer_residual` vector. The final fp16 output matches exactly across all
1024 rows (`output_mismatches == 0`) even though the fp32 diagnostic checksum
differs because GLSL `exp` and CPU `std::exp` can land on opposite sides of a
SiLU rounding boundary.

This proves the probe can accept real residual-stream values and produce exact
fp16 output for that captured case. It does not prove RMSNorm integration,
attention or DeltaNet integration, full-layer composition, or inference.

## Correctness Rules

The project uses the strictest gate that matches the actual contract:

- Use exact token parity for end-to-end greedy decode.
- Use exact byte or fp16 equality for storage-format handoffs.
- Use exact fp32 checksums only when the CPU and shader are expected to produce
  identical fp32 internals.
- Treat fp32 checksums as diagnostics when the real contract is fp16 output and
  the divergence is explained by valid CPU/GPU math-library differences.
- Do not add silent tolerances. If tolerance becomes necessary, document the
  reason, bound it, and test that it catches real regressions.
- Do not make performance claims before correctness is locked.

For the persistent MLP captured-handoff probe specifically (diary 0089), a
three-tier fp16 output comparison model applies:

1. **Exact equality (default, tolerance 0):** every output row must match
   bit-for-bit. This is the correct gate for GPU-vs-GPU comparisons and any
   case where reference and implementation use identical arithmetic.

2. **Opt-in bounded ULP tolerance (`--output-fp16-ulp-tolerance N`):** rows
   with 0 < ULP diff <= N are tolerated but still reported as
   `output_exact_mismatches`. Only rows with ULP diff > N are gate-breaking
   `output_mismatches`. This is appropriate for CPU-vs-GPU captured-handoff
   probes where GLSL `exp` and CPU `std::exp` can differ at rounding
   boundaries.

3. **Opposite-sign mismatch:** if GPU and CPU outputs have opposite signs
   (neither zero), the ULP diff is `UINT32_MAX`. No finite tolerance accepts
   this. Always gate-breaking.

GPU-vs-GPU comparisons remain exact. The tolerance is scoped to CPU-vs-GPU
captured-handoff probes only.

## Performance Rules

Performance work has to earn its place after correctness:

- Reducing dispatches is useful only if the output remains reproducible.
- Persistent dispatch is useful only if the barrier overhead and occupancy are
  better than the single-submit alternative.
- A benchmark is not a benchmark until the command, prompt set, warmups,
  timed runs, GPU timestamps or host timings, environment, and commit are
  recorded.
- A single successful timing sample is evidence, not a conclusion.

The megakernel path must beat the strongest honest baseline. If it does not,
the project should keep the simpler fused Vulkan path and say so plainly.

## Completion Framing

Overall project maturity can be much higher than megakernel completion.

Artifact ingestion, reference parity, ordinary Vulkan decode, diagnostics,
chunked command buffers, GPU-resident handoffs, barrier probes, skeletons, and
MLP probes are real progress. They reduce risk and make the final target
possible.

They are not the integrated target. The actual megakernel completion percentage
stays low until a layer-shaped persistent kernel exists, then a 24-layer
persistent path, then GPU-resident LM-head/token loop, then archived basic test
inference. This distinction prevents the project from confusing infrastructure
maturity with final-path completion.

Diaries 0094-0096 completed the layer-0 MLP scratch-boundary map under captured
RMSNorm input: RMSNorm is exact, raw gate projection is max 1 ULP, raw up
projection is max 2 ULP, activation product is max 2 ULP, down output is max
2 ULP, and post-residual output is max 87 ULP. That evidence is enough to stop
splitting this micro-probe for now. The next quality-preserving step is to
compose a layer-shaped persistent probe with captured checkpoints, not to chase
one-off exactness inside an already-bounded MLP subcomponent.

Diary 0097 tightened that foundation by measuring the post-residual population
tail: only 10 of 1024 layer-0 rows are above 16 ULP, and only 1 row is above
64 ULP. The project should preserve this kind of distribution gate when moving
from micro-probes to composed persistent layers. A generated-token pass is too
coarse to replace these local numerical contracts.

Diary 0098 adds the same discipline at layer 20. The max tail is larger
(209 ULP), but only 1 of 1024 rows is above 16 ULP. That combination is exactly
why the project tracks max and population separately: max catches isolated
tail risk, while population catches broad drift. Both contracts should travel
forward into the next persistent layer-shaped artifact.

Diary 0099 starts the token-mixer side without jumping straight into DeltaNet
complexity. Capturing `mixer_output_fp16` lets the project verify the first
residual handoff exactly: `input_hidden + mixer_output -> mixer_residual`.
That algebraic closure is the bridge from MLP-only checkpoints to a real
layer-shaped persistent probe. The next token-mixer implementation should
target the mixer output first, then reuse the residual-add gate to separate
token-mixer errors from residual handoff errors.

Diary 0101 closes the first token-mixer output projection exactly. The generic
`vk_matvec_probe` proves
`dn_gated_fp16 + layer.0.delta_out_proj -> dn_out_fp16` at full model width
with zero mismatches. This replaces the previous future-tense milestone with
completed evidence and opens the path to walk backward through DeltaNet
recurrent/norm/gate internals.

Diary 0102 then closes the norm-gate stage exactly:
`dn_core_fp16 + dn_z_fp16 + layer.0.delta_norm -> dn_gated_fp16`. The validated
downstream chain is now recurrent core to gated vector to mixer output to mixer
residual. That made the recurrent core producer the next useful DeltaNet gate,
not another downstream handoff; diary 0112 later closed that producer in the
standalone recurrent probe.

Diary 0103 closes the z-projection side exactly:
`dn_input_norm_fp16 + layer.0.delta_in_proj_z -> dn_z_fp16`. It also records
why the qkv projection cannot be gated against the current dumped q/k/v fields:
q and k have already passed through L2 normalization. The next qkv-side gate
must either add raw qkv projection captures or compose projection, split, and
q/k L2 normalization together.

Diary 0104 adds that raw qkv capture and proves the qkv projection exactly:
`dn_input_norm_fp16 + layer.0.delta_in_proj_qkv -> dn_qkv_raw_fp16`. The next
qkv-side work can now move to conv1d mutation and q/k L2 normalization with a
valid raw input checkpoint.

Diary 0105 adds raw A/B captures and proves
`dn_input_norm_fp16 + layer.0.delta_in_proj_a -> dn_a_fp16` and
`dn_input_norm_fp16 + layer.0.delta_in_proj_b -> dn_b_fp16` exactly. The
stateless projection fanout from `dn_input_norm_fp16` is now closed.

Diary 0106 adds exact g/beta bit fixtures and proves
`dn_a_fp16 + dn_b_fp16 + delta_a_log + delta_dt_bias -> dn_g/dn_beta`
bit-for-bit. The scalar branch feeding recurrent state is now closed for layer
0, step 1.
Diary 0109 closes the conv/L2 gate exactly: `dn_qkv_raw_fp16 + conv_state_pre +
delta_conv_weights -> dn_q/dn_k/dn_v_fp16` with zero mismatches. The unfused
runtime path (conv1d_step + L2-norm Q + L2-norm K) produces bit-identical output
to the captured handoff tensors for layer 0, step 1.

## Current Next Milestones

After diary 0120, the next useful milestones are:

1. ~~Validate the DeltaNet recurrent core producer against captured `dn_core_fp16`,~~
   ~~including q/k/v inputs, g/beta parameters, and recurrent state handling.~~ (done: diary 0112)
2. ~~Produce the full layer-0 DeltaNet mixer output without substituting captured~~
   ~~intermediate tensors after `dn_input_norm_fp16`.~~ (done: diary 0113, exact composed probe)
3. ~~Establish the first layer-shaped persistent scaffold with 128-lane post-mixer~~
   ~~tail execution.~~ (done: diary 0114, persistent_layer0_probe with bounded gate)
4. ~~Gate the persistent layer-0 projection prefix `dn_input_norm -> qkv_raw, z, a, b`~~
   ~~in the persistent layer shader.~~ (done: diary 0116, ULP-1 bounded gate)
5. ~~Gate persistent layer-0 conv/L2 from captured qkv_raw and conv state.~~
   (done: diary 0117, exact q/k/v gate)
6. ~~Gate persistent layer-0 g/beta from captured A/B and repacked scalar weights.~~
   (done: diary 0118, exact fp32-bit gate)
7. ~~Gate persistent layer-0 recurrent core from captured q/k/v, g/beta, and~~
   ~~pre-update recurrent state inside `persistent_layer0_probe.comp`.~~
   (done: diary 0119, exact fp16 gate)
8. ~~Gate persistent layer-0 norm-gate, output projection, and first residual~~
   ~~add inside `persistent_layer0_probe.comp`.~~
   (done: diary 0120, ULP-1 mixer-output gate with exact residual handoff)
9. Compose a full layer-shaped persistent probe that combines DeltaNet mixer
   output, first residual add, RMSNorm, MLP, and second residual update
   with captured layer-0 checkpoints.
10. Sweep the layer-shaped probe across representative layers only after
   layer 0 is explainable.
9. Run a bounded multi-layer persistent decode probe before attempting all
   24 layers.
10. Add final norm, LM head, and token selection only after layer
   composition is correct and debuggable.
11. Archive the first basic test inference from the target path with
   commands, artifacts, environment, and expected output.

The discipline is simple: every fused step must have a smaller gate that can
explain failures. That is how the project gets to a real megakernel without
turning correctness into guesswork.

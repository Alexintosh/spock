# RX 6750 XT Vulkan-Native Megakernel Execution Map

This document is the current operating map for the real target: a
Vulkan-native, model-specific decode path for `Qwen/Qwen3.5-0.8B` on
`AMD Radeon RX 6750 XT (RADV NAVI22)`, ending in archived basic test inference
from the target path.

The target is not "many Vulkan probes" and it is not a generic backend. The
target is a GPU-resident hot decode loop, preferably a persistent dispatch, that
keeps the layer state, token-mixer state, MLP work, final norm, LM head, token
selection, and next-token handoff on the Vulkan path.

The project is deliberately building that target from small gates because the
final shader is the worst place to discover the first correctness bug. A wrong
token after 24 fused layers says almost nothing about where the failure entered.
A failed local gate says which equation, tensor layout, state update, precision
policy, or barrier transition is wrong.

## Current Proof State

As of diary 0121, the project has a strong foundation but not the final
megakernel. The important distinction is:

- foundation maturity is high enough to make the target realistic;
- final-path completion has progressed: the persistent full-mixer is now composed
  but the full-layer and full-decode path are not assembled.

The closed pieces are:

- artifact ingestion and repack for the pinned Qwen 3.5 0.8B layout;
- conventional Vulkan decode with all 24 layers wired;
- fixed prompt/reference parity machinery and component dumps;
- GPU-resident handoff experiments for chunk prefill and generated-token
  handling;
- per-layer descriptor prebinding and single-submit-oriented runtime work;
- software global barrier probes, including decode-shaped and model-width
  payloads;
- persistent skeletons with fp16 inputs, fp16 weights, and fp32 accumulation;
- persistent MLP and RMSNorm+MLP gates with real model weights and captured
  activation fixtures;
- exact layer-0 DeltaNet component gates for residual add, output projection,
  norm-gate, z projection, raw qkv projection, A/B projections, g/beta bits,
  conv/L2, and recurrent core;
- an exact composed layer-0 DeltaNet mixer probe in one Vulkan submit;
- the first layer-shaped persistent scaffold for the post-mixer tail:
  `mixer_residual -> post_norm -> MLP -> post_mlp` at 128 lanes and
  82 workgroups;
- the persistent layer-0 projection-prefix gate: `dn_input_norm -> qkv_raw, z, a, b`
  at 128 lanes and 82 workgroups, with 1-ULP bounded tolerance (diary 0116);
- the persistent layer-0 conv/L2 gate:
  `qkv_raw + conv_state_pre + delta_conv -> q/k/v` at 128 lanes and 82
  workgroups, passing exact q/k/v comparison with one software global barrier
  (diary 0117).
- the persistent layer-0 g/beta gate:
  `dn_a + dn_b + delta_a_log + delta_dt_bias -> g/beta bits`, passing exact
  fp32 bit-pattern comparison with no software global barriers (diary 0118).
- the persistent layer-0 recurrent core gate:
  `dn_q + dn_k + dn_v + g/beta + recurrent_state_pre -> dn_core`, passing exact
  fp16 comparison inside `persistent_layer0_probe.comp` with no software global
  barriers (diary 0119).
- the persistent layer-0 mixer-tail gate:
  `dn_core + dn_z + delta_norm -> dn_gated -> delta_out_proj -> mixer_output`
  followed by `input_hidden + mixer_output -> mixer_residual`, passing the
  explicit ULP-1 persistent gate with exact residual handoff and two software
  global barriers (diary 0120).
- the persistent layer-0 full-mixer gate:
  `dn_input_norm -> projections -> g/beta + conv -> L2 -> recurrent -> norm-gate
  -> out_proj -> mixer_output -> mixer_residual`, passing the bounded ULP-16
  persistent gate with 6 software global barriers at 128 lanes and 82 workgroups
  (diary 0121).

The missing target pieces are:
- persistent mixer + post-mixer tail composition from `dn_input_norm` through `post_mlp`;
- full layer-0 persistent composition from `dn_input_norm` through `post_mlp`;
- representative attention-layer component gates and persistent composition;
- representative layer sweeps;
- bounded multi-layer persistent decode;
- all 24 layers in the target persistent or strongest honest fused Vulkan path;
- final RMSNorm, LM head, token selection, and GPU-resident next-token loop;
- archived basic test inference with command, commit, environment, artifacts,
  prompt, and generated output.

## Why The Pieces Are Built In This Order

The current order is a dependency graph, not a preference list.

### 1. Artifact and parity contract

Everything depends on stable bytes. The repacked weights, tensor names, dtype
policy, layer schedule, and reference tokens must be fixed before a GPU mismatch
can mean anything.

Tests and references:

- artifact manifest and tensor layout checks;
- `docs/artifact_format.md`;
- `docs/parity_contract.md`;
- frozen prompt and reference decode gates.

Why it comes first: without this, every later failure can be blamed on the
artifact instead of the kernel.

### 2. Conventional Vulkan runtime

The conventional runtime is the diagnostic spine. It is allowed to be less
fused because its job is to run the actual model and expose hidden states.

Tests and references:

- `spock-decode` parity runs;
- `tests/run_vk_decode_parity.py`;
- component dump paths used by `tools/extract_component_fp16.py`;
- `docs/runtime_strategy.md`.

Why it comes before persistent composition: persistent probes need real input
and expected output tensors. The runtime is where those tensors come from.

### 3. Component extraction and local equations

Captured component tensors are turned into raw fixtures. Then each local
equation is gated before being fused.

Examples:

- `input_hidden + mixer_output -> mixer_residual`;
- `dn_gated + delta_out_proj -> dn_out`;
- `dn_core + z + delta_norm -> dn_gated`;
- `dn_input_norm + qkv/z/a/b weights -> raw branches`;
- `a/b + biases -> g/beta`;
- `raw qkv + conv state -> q/k/v`;
- `q/k/v + g/beta + recurrent state -> dn_core`;
- `mixer_residual -> post_norm -> MLP -> post_mlp`.

Tests and references:

- `vk_matvec_probe`;
- `vk_residual_add_probe`;
- `vk_deltanet_norm_gate_probe`;
- `vk_deltanet_g_beta_probe`;
- `vk_deltanet_conv_l2_probe`;
- `vk_deltanet_recurrent_probe`;
- `vk_deltanet_mixer_probe`;
- `vk_persistent_mlp_probe`;
- `vk_persistent_layer0_probe`.

Why it comes before larger fusion: once these equations are closed, a composed
failure has a small search space. Without them, a composed failure is just drift.

### 4. Software barrier and persistent skeletons

Vulkan does not provide a CUDA-style cooperative grid barrier inside one
dispatch. The persistent path depends on a software global barrier over
coherent storage buffers and bounded spinning.

Tests and references:

- barrier probe entries 0047-0056 and 0071-0073;
- `vk_barrier_probe`;
- persistent skeleton entries 0075-0079;
- decode-shaped 82-workgroup payloads;
- generation counters, failure counters, and timeout behavior.

Why it is isolated: barrier failure can look like random model drift. It must
be tested before real model math is blamed.

### 5. Persistent MLP and RMSNorm tail

The MLP tail was the right first persistent sub-block because it has real
multi-stage compute, real weights, residual semantics, scratch lifetimes, and
RMSNorm, but less state than DeltaNet or attention.

Tests and references:

- diaries 0080-0098;
- `vk_persistent_mlp_probe`;
- layer-0 exact and bounded captured gates;
- layer-20 population gate;
- diary 0114 `vk_persistent_layer0_probe`.

Why it comes before token mixer persistence: it proves the downstream consumer
of `mixer_residual`. Once the token mixer produces `mixer_residual`, the rest of
the layer already has an explainable gate.

### 6. DeltaNet mixer gates

DeltaNet is stateful and layout-sensitive. It has projections, conv state,
normalization, scalar branches, recurrent state, output projection, and a
residual handoff. The project closed it backward from the value consumed by the
rest of the layer.

Tests and references:

- diaries 0099-0113;
- `spock_deltanet_conv_l2_probe_layer0_exact`;
- `spock_deltanet_recurrent_probe_layer0_exact`;
- `spock_deltanet_mixer_probe_layer0_exact`.

Why this order matters: the recurrent core should not be debugged until q/k/v
and g/beta are known; the full mixer should not be debugged until each
component is known; the persistent layer should not be debugged until the
single-submit mixer is known.

### 7. Layer-shaped persistent probe

The first real megakernel-shaped milestone is one persistent layer:

```text
pre-layer residual
  -> pre-mixer RMSNorm
  -> token mixer
  -> first residual add
  -> post-mixer RMSNorm
  -> MLP
  -> second residual add
```

Tests and references:

- current `vk_persistent_layer0_probe` tail gate;
- current persistent DeltaNet projection-prefix gate;
- current persistent DeltaNet conv/L2 gate;
- next full layer-0 persistent gate;
- captured layer-0 pre/post fixtures;
- barrier generation and failure counters.

Why it comes before multi-layer decode: one layer is still explainable. Multiple
layers add state continuity, scratch reuse, and layer selection on top of
already-hard compute.

### 8. Representative layers and attention

Layer 0 is necessary but not sufficient. The model has 18 DeltaNet layers and
6 attention layers. Attention brings KV cache, RoPE, GQA layout, scores,
softmax, and value accumulation.

Tests and references:

- representative DeltaNet layer sweeps after layer 0;
- attention KV-cache and RoPE fixture checks;
- attention output projection and residual handoff gates;
- mid-network MLP population gates as precedent.

Why it comes here: attention should not be fused into the target loop until it
has local tests comparable to the DeltaNet tests.

### 9. Bounded multi-layer persistent decode

After one layer and representative layers are explainable, widen to small
multi-layer runs.

Tests and references:

- 2-layer and 4-layer persistent gates;
- checkpoint comparison after selected layers;
- state-continuity checks for DeltaNet recurrent state and attention KV cache;
- scratch aliasing stress checks.

Why it comes before all 24 layers: multi-layer bugs are usually state bugs.
Small runs preserve enough checkpoints to locate them.

### 10. Full target-path inference

Only after hidden-state production is trustworthy should the project add final
RMSNorm, LM head, token selection, and the next-token handoff to the target
path.

Tests and references:

- final norm fixture checks;
- LM-head matvec checks;
- argmax or deterministic sampling checks;
- device-resident token handoff checks;
- frozen-prompt greedy token parity;
- archived basic inference record.

Why it comes last: generated tokens are the coarsest possible correctness
signal. They are excellent as a final proof and poor as the first debugger.

## Current Non-Stop Execution Plan

The immediate implementation ladder from the current state is:

1. ~~Finish the persistent layer-0 projection-prefix gate:~~
   ~~`dn_input_norm -> qkv_raw, z, a, b`.~~ (done: diary 0116)
2. ~~Add persistent conv/L2 stages using the already captured and gated qkv
   fixtures.~~ (done: diary 0117)
3. ~~Add persistent g/beta computation.~~ (done: diary 0118)
4. ~~Add persistent recurrent core.~~ (done: diary 0119)
5. ~~Add persistent norm-gate, output projection, and first residual add.~~
   (done: diary 0120)
6. Compose the existing persistent post-mixer tail in the same shader.
7. Compare full layer-0 persistent output against captured `post_mlp`.
8. Add representative DeltaNet layers only after layer 0 is explainable.
9. Build the equivalent attention-layer gates before fusing attention layers.
10. Run bounded multi-layer persistent decode with checkpoint gates.
11. Extend to all 24 layers.
12. Add final norm, LM head, token selection, and GPU-resident token handoff.
13. Archive basic test inference from the target path.

The plan should not skip from step 1 to step 10. That would trade away the
debugging evidence needed to make failures actionable.

## What To Test And How

Every new gate should answer five questions:

- Did the shader run to completion without barrier failures or timeouts?
- Did it use the expected workgroup count and generation count?
- Does it match exact fp16 or exact fp32-bit fixtures where the contract is
  exact?
- If it needs tolerance, is the tolerance explicit, bounded, reported, and
  justified by a known CPU-vs-GPU precision boundary?
- Does the CTest name make clear what is exact, what is bounded, and what is
  expected to fail?

Minimum verification for code changes:

- `cmake --build build -j`;
- direct probe command with full JSON output inspected;
- focused CTest set for the touched probes;
- `python3 tests/run_diary_check.py` when diary changes;
- `git diff --check`;
- commit only the intended files.

Minimum verification for persistent shader changes:

- direct old-mode regression command first;
- direct new-mode command second;
- structural checks: `failures == 0`, final generation matches expected barrier
  count, and `arrived == 0`;
- exact fixture comparison whenever GPU-vs-GPU arithmetic is expected to match;
- bounded comparison only when a documented CPU-vs-GPU numerical boundary
  applies;
- a CTest gate for the passing path;
- a WILL_FAIL CTest only when preserving an exact-failure boundary is useful.

Minimum verification before claiming target-path inference:

- command line and environment recorded;
- commit hash recorded;
- artifact path and checksums recorded;
- GPU and driver capability dump recorded;
- prompt and generated tokens archived;
- exact greedy parity or explicitly documented deterministic-output contract;
- no hidden host readback/upload bridge inside the claimed hot path.

## Code Quality Rules

This plan does not compromise on code quality. The quality strategy is stricter
than a direct megakernel rewrite:

- do not fuse a stage before its standalone contract exists;
- do not remove observability until the replacement has tests;
- do not introduce silent tolerances;
- do not label infrastructure as final-path completion;
- do not hide persistent barrier risk behind successful math probes;
- do not claim GPU residency if a host bridge remains in the hot path;
- keep fallback paths, but label them as fallback paths;
- commit documentation and diary updates with the behavior they describe.

The staged approach creates more small programs and tests, but that is the
point. The final megakernel should be assembled from proven contracts, not from
hope that one huge shader happens to produce the right token.

## Risk Register

### Software global barrier

The persistent path can fail if RADV does not keep all required workgroups
resident or if memory visibility is insufficient under real shader pressure.

Mitigation: keep barrier counters, bounded spin limits, decode-shaped probes,
model-width probes, and retest after every major stage increase.

### Occupancy and shader pressure

Adding DeltaNet stages can increase register pressure and scratch use enough to
change workgroup residency.

Mitigation: grow the persistent shader incrementally, keep workgroup count
explicit, and preserve the strongest single-submit fallback as an honest
alternative.

### Precision drift

The target uses fp16 storage and fp32 accumulation on hardware without native
bf16. CPU and GPU math libraries can cross fp16 rounding boundaries.

Mitigation: prefer GPU-vs-GPU exact gates, use CPU references for diagnostics,
and require bounded ULP policy when exact CPU-vs-GPU output is not the actual
contract.

### Stateful mixer bugs

DeltaNet recurrent state, conv state, and attention KV cache can be correct for
one layer and wrong across layers.

Mitigation: add checkpoint gates for representative layers and small multi-layer
runs before all-layer decode.

### Scratch aliasing

The target shader will reuse buffers aggressively. A stale write can surface
only much later.

Mitigation: start with explicit scratch regions, add aliasing only after
single-purpose scratch is gated, and keep debug checksums or trace fields where
they clarify ownership.

### Documentation drift

The project has many probes. If the docs imply a probe proves more than it
does, future work will optimize the wrong path.

Mitigation: every diary entry must state what changed, what was verified, and
what remains out of scope. Docs must separate foundation maturity from
target-path completion.

## Completion Framing

The current project is not near final inference completion, even though many
hard prerequisites are complete. The correct framing is:

- the foundation is substantial;
- the DeltaNet layer-0 single-submit contract is strong;
- the persistent post-mixer tail scaffold is useful;
- the actual persistent full-layer path is not complete;
- the 24-layer persistent decode path does not exist yet;
- final norm, LM head, token selection, and archived inference remain future
  work.

Percent estimates should therefore stay conservative until target-path code
exists. A future percentage can rise sharply when one full persistent layer
passes, then again when bounded multi-layer decode passes, and again when all
24 layers plus LM head/token loop run from the target path.

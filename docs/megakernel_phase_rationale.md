# Vulkan Megakernel Phase Rationale

This document explains why Spock is building the RX 6750 XT target in narrow
phases instead of writing the final persistent megakernel first. It is written
for the current project goal: a Vulkan-native, model-specific decode path for
`Qwen/Qwen3.5-0.8B` on `AMD Radeon RX 6750 XT (RADV NAVI22)`, ending in
archived basic test inference from the target path.

The short version is simple: the final target is a persistent GPU-resident
decode loop, but a persistent megakernel is a bad place to discover math,
layout, state, barrier, or precision bugs. Every prerequisite exists to make
one class of failure observable before it becomes hidden inside the fused path.

## Target Shape

The desired hot path is:

```text
token on GPU
  -> embedding
  -> 24 model layers
       -> pre-mixer RMSNorm
       -> DeltaNet or attention token mixer
       -> first residual update
       -> post-mixer RMSNorm
       -> MLP
       -> second residual update
  -> final RMSNorm
  -> LM head
  -> token selection
  -> next token remains GPU-resident
```

The strongest implementation is one persistent Vulkan dispatch for the decode
pass, with explicit software barriers between internal stages and no host
readback/upload bridge in the hot loop. If the persistent barrier is not robust
or not faster on this RADV stack, the honest fallback is the strongest fused
single-submit Vulkan path. That fallback can be useful, but it is not the
megakernel target.

## Why The Project Does Not Start At The Final Shader

A full fused layer or full decode megakernel collapses too many questions into
one failure signal. A wrong generated token could come from any of these:

- tensor packing or offset mistakes;
- fp16/fp32 conversion or rounding policy;
- RMSNorm formula or weight role;
- DeltaNet q/k/v preparation;
- DeltaNet convolution state mutation;
- g/beta scalar computation;
- recurrent state decay, update, q scaling, or output accumulation;
- attention RoPE, KV cache, score, softmax, or value accumulation;
- residual stream ordering;
- MLP gate/up/down math;
- scratch aliasing;
- descriptor binding lifetime;
- cross-workgroup barrier visibility;
- final norm, LM head, or argmax.

When those are fused too early, the generated token tells us only that
something is wrong. It does not say where. The current philosophy is to close
local contracts until a future megakernel mismatch has only a small number of
possible causes.

## The Phase Ladder

### 1. Artifact and parity contract

This phase freezes what model is being run and what "correct" means. It covers
the text-only model artifact, tensor roles, dtype policy, prompt corpus, decode
semantics, and benchmark metadata.

Why it comes first:

- a GPU failure cannot be diagnosed if the artifact might be wrong;
- a performance result is meaningless without a fixed prompt and decode
  contract;
- every later fixture depends on stable tensor names, offsets, and precision.

Representative tests:

- artifact manifest validation;
- packed tensor shape, dtype, offset, and checksum checks;
- text-only load-plan validation;
- fixed prompt token parity against the trusted reference.

### 2. Conventional Vulkan runtime

This phase brings up explicit Vulkan dispatches with enough observability to
run the whole model and dump component boundaries. It is allowed to be slow and
over-instrumented.

Why it comes before persistent work:

- it gives the project a real GPU execution spine;
- it can capture hidden states and component outputs from actual model runs;
- it separates model math and layout bugs from persistent synchronization bugs.

Representative tests:

- `tests/run_vk_decode_parity.py` on the frozen prompt set;
- CTest decode regression gates;
- component dump parsing and fixture extraction checks.

### 3. GPU-resident handoffs and descriptor stability

This phase removes avoidable host bridges from already-correct paths:
chunk-prefill input/output handoffs, generated-token transfer patterns,
per-layer descriptor mutation, and command-buffer grouping.

Why it matters:

- persistent decode requires stable device-local state;
- host readbacks can hide missing GPU-to-GPU barriers;
- descriptor mutation during recording is incompatible with single-submit and
  persistent command structures.

Representative tests:

- gated decode parity with GPU-collected chunk-prefill paths;
- generated-token deferred-download checks;
- descriptor prebind parity gates;
- `git diff --check` plus diary checks for every behavior change.

### 4. Software global barrier viability

Vulkan does not provide a CUDA cooperative-grid barrier inside one dispatch.
The persistent path depends on a bounded software barrier implemented with
coherent storage-buffer state and workgroups that can all make progress.

Why it is isolated:

- barrier failure can look exactly like random model drift;
- occupancy, register pressure, and memory traffic can change barrier behavior;
- a barrier that passes a toy case is not enough for a model-width decode
  kernel.

Representative tests:

- synthetic barrier probe;
- memory-payload barrier probe;
- repeated soak runs;
- decode-shaped 82-workgroup timing and failure-counter checks;
- retesting after major shader complexity increases.

### 5. Persistent skeletons with real weights

This phase proves that a persistent dispatch can do useful fp16/fp32 tensor
work with real packed model weights, not just synchronize.

Why it matters:

- it tests memory layout and reduction order under persistent staging;
- it exposes register, scratch, and workgroup-shape pressure earlier;
- it provides the mechanical pattern used by later MLP and layer probes.

Representative tests:

- synthetic dot-product probes;
- real-weight row-strided probes;
- model-width projection checks;
- captured activation handoff gates when runtime data is consumed.

### 6. RMSNorm and MLP persistent path

The MLP side was a deliberate first persistent sub-block because it is
state-light compared with DeltaNet and attention, but still exercises the
barriers, scratch lifetimes, real weights, activation functions, and residual
semantics the full layer needs.

Why this was the right early target:

- it validates multi-stage persistent compute without recurrent state;
- it proves fp16 storage with fp32 accumulation in a real sub-block;
- it creates a known downstream consumer for token-mixer residual output.

Representative tests:

- persistent MLP stage checks;
- captured RMSNorm output equality;
- gate/up/product/down boundary checks;
- explicit ULP population gates where CPU-vs-GPU math differs by valid
  rounding behavior;
- residual update checks.

### 7. DeltaNet mixer contract

This is the active focus. DeltaNet is being closed backward from the value the
rest of the layer consumes:

```text
input_hidden + mixer_output -> mixer_residual
dn_gated + delta_out_proj -> dn_out
dn_core + z + delta_norm -> dn_gated
input_norm + projection weights -> qkv raw, z, a, b
a/b + biases -> g/beta
raw qkv + conv/L2 -> q, k, v
q/k/v + g/beta + recurrent state -> dn_core
```

Why this order matters:

- output projection and residual add are stateless and easy to pin down;
- z, qkv, a, and b projection gates remove layout and matvec ambiguity;
- g/beta bit checks remove scalar-branch ambiguity;
- conv/L2 closes q/k/v preparation before recurrent state is blamed;
- the recurrent core is tested only after its inputs are deterministic files.

Representative tests:

- `vk_matvec_probe` for output projection, z, qkv, a, and b;
- `vk_deltanet_norm_gate_probe`;
- `vk_deltanet_g_beta_probe`;
- `vk_deltanet_conv_l2_probe`;
- `vk_deltanet_recurrent_probe`;
- `vk_deltanet_mixer_probe`;
- next required gate: persistent layer composition that moves the closed
  layer-0 mixer contract into the 128-lane persistent layer scaffold before
  widening to representative layers.

### 8. Attention mixer contract

Attention layers must receive the same treatment before full decode
composition. They touch different state: KV cache, RoPE offsets, GQA layout,
scores, softmax, and value accumulation.

Why it follows DeltaNet:

- the model has 18 DeltaNet layers and 6 attention layers;
- DeltaNet is currently the larger unclosed stateful path;
- attention should not be fused into the persistent loop until its local
  failures can be explained by component gates.

Representative tests:

- KV cache store/load checks;
- RoPE q/k fixture checks;
- score and softmax diagnostics;
- attention output projection and residual handoff gates;
- representative attention-layer captured fixtures.

### 9. Layer-shaped persistent probe

Once token mixer and MLP contracts exist, the project can compose one full
layer-shaped persistent probe:

```text
pre-layer residual
  -> pre-mixer RMSNorm
  -> token mixer
  -> first residual
  -> post-mixer RMSNorm
  -> MLP
  -> second residual
```

Why this is the first real megakernel-shaped milestone:

- it tests the actual layer ordering;
- it forces scratch lifetime discipline between mixer and MLP;
- it validates that local contracts survive composition;
- it is still small enough to compare with captured pre/post layer fixtures.

Representative tests:

- layer-0 captured input/output equality or bounded policy;
- representative DeltaNet and attention layer sweeps;
- scratch aliasing stress checks;
- barrier failure counters and GPU timestamps.

### 10. Bounded multi-layer persistent decode

After one layer is explainable, widen to several layers before all 24. This
phase tests per-layer state transitions, descriptor selection, scratch reuse,
and recurrent/KV cache continuity.

Why it is not skipped:

- layer-local correctness does not prove state continuity;
- bugs can appear only when a later layer consumes previous-layer drift;
- bounded runs preserve enough checkpoints to locate failure.

Representative tests:

- 2-layer and 4-layer persistent decode gates;
- representative layer-pattern gates covering DeltaNet and attention;
- checkpoint comparisons after selected layers;
- generated-token checks only after intermediate checkpoints are stable.

### 11. Full decode loop, LM head, and token selection

The final GPU-resident decode loop adds final RMSNorm, LM head, argmax or
deterministic sampling, and next-token handoff without CPU mediation in the hot
path.

Why it comes late:

- token output is the coarsest diagnostic signal;
- LM head and token selection are only meaningful after all layer outputs are
  trustworthy;
- performance claims require the complete path, not isolated kernels.

Representative tests:

- exact greedy token parity on the frozen prompt corpus;
- final norm and LM-head fixture checks;
- argmax/token handoff checks;
- device-resident next-token loop checks;
- archived basic inference command and output.

## How The Pieces Fit Together

The project is not building unrelated probes. Each probe supplies a contract
that the final megakernel will depend on:

- artifact checks make weight bytes trustworthy;
- the conventional runtime creates real checkpoints;
- extraction tools preserve checkpoint bits as fixtures;
- component probes close local equations;
- persistent skeletons prove the execution style;
- barrier probes bound the synchronization risk;
- MLP probes validate multi-stage persistent compute;
- DeltaNet and attention probes validate stateful token mixers;
- layer probes validate composition;
- multi-layer decode validates state continuity;
- LM-head/token-loop work turns correct hidden states into archived inference.

The conventional runtime remains valuable even after persistent work advances.
It is the diagnostic reference for captured states. The probes remain valuable
even after composition advances. They are the tests that explain failures when
larger kernels regress.

## Code Quality Position

This plan does not trade quality for speed. The quality position is:

- do not claim completion from infrastructure alone;
- do not hide numerical drift with broad tolerances;
- do not fuse a stage unless its standalone contract is already testable;
- do not optimize away observability before the replacement has tests;
- keep fallbacks and diagnostics, but label them as fallbacks and diagnostics;
- require every milestone to state what it proves and what it does not prove.

The result may look slower than writing one large shader first. It is more
likely to produce a correct archived inference result because the project will
know which contract failed when the first composed persistent path is wrong.

## Current Completion Framing

The project has substantial infrastructure maturity: artifact ingestion,
reference/runtime decode, component dumps, descriptor prebinding, GPU-resident
handoffs, barrier probes, persistent skeletons, MLP probes, and a growing
DeltaNet component chain.

That is not the same as megakernel completion. Completion against the actual
goal remains low until these target-path milestones exist:

1. DeltaNet recurrent core probe passes against captured `dn_core_fp16`.
2. Full layer-0 DeltaNet mixer output is produced without substituting captured
   internals after `dn_input_norm_fp16`.
3. A layer-shaped persistent probe passes against captured layer checkpoints.
4. Representative DeltaNet and attention layers pass under the same structure.
5. Bounded multi-layer persistent decode passes with checkpoint gates.
6. All 24 layers run in the persistent or strongest honest fused path.
7. Final norm, LM head, token selection, and next-token handoff run on the
   target path.
8. Basic test inference is archived with command, commit, environment,
   artifact, prompt, and generated tokens.

Until those are closed, percentage estimates should distinguish final-path
completion from the broader foundation that makes the final path achievable.

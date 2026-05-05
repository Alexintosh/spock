# Megakernel Development Philosophy

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
barrier-synchronized staging. It is still not a full layer, because RMSNorm,
attention or DeltaNet, and final token generation are outside this probe.

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

This is why the diary 0086 change is acceptable: it does not hide a wrong
output. It moves the pass/fail gate to exact equality of the fp16 values the
downstream consumer actually receives, while retaining the fp32 checksum as a
diagnostic signal.

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

## Current Next Milestones

After diary 0086, the next useful milestones are:

1. Validate captured fp16 handoff across more layers and decode steps.
2. Add RMSNorm-before-MLP to the persistent MLP probe with real norm weights.
3. Build a layer-shaped persistent probe that composes RMSNorm, mixer handoff,
   MLP, and residual update with real captured checkpoints.
4. Integrate the token-mixer side of the layer under the persistent barrier
   model.
5. Run a bounded multi-layer persistent decode probe before attempting all
   24 layers.
6. Add final norm, LM head, and token selection only after layer composition is
   correct and debuggable.
7. Archive the first basic test inference from the target path with commands,
   artifacts, environment, and expected output.

The discipline is simple: every fused step must have a smaller gate that can
explain failures. That is how the project gets to a real megakernel without
turning correctness into guesswork.

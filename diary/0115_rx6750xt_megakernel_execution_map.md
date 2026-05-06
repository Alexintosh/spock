# 0115: RX 6750 XT Megakernel Execution Map

## Goal

Write down the current project philosophy in a more concrete form, centered on
the actual goal: a Vulkan-native megakernel-style decode path for
`Qwen/Qwen3.5-0.8B` on `AMD Radeon RX 6750 XT (RADV NAVI22)`, ending in
archived basic test inference from the target path.

This entry exists because the project now has enough validated infrastructure
that it is easy to mistake foundation maturity for final-path completion. The
foundation is real: artifacts, conventional Vulkan decode, component dumps,
descriptor work, GPU-resident handoff experiments, barrier probes, persistent
skeletons, MLP/RMSNorm gates, exact DeltaNet component gates, an exact
single-submit layer-0 DeltaNet mixer, and a 128-lane persistent post-mixer tail
scaffold. But the final target is not complete until the persistent layer path,
multi-layer persistent decode, LM head, token loop, and archived inference proof
exist.

## What changed

Added `docs/rx6750xt_megakernel_execution_map.md`.

The new document records:

- the current proof state after diary 0114;
- what remains missing before the actual megakernel target is achieved;
- why the work is ordered as artifact contract, conventional runtime,
  component extraction, local equations, barrier probes, persistent skeletons,
  MLP/RMSNorm tail, DeltaNet gates, layer-shaped persistence, representative
  layers, bounded multi-layer decode, and final target-path inference;
- which probes and tests correspond to each phase;
- the immediate non-stop implementation ladder from the current state;
- the verification expectations for new code, persistent shader changes, and
  eventual archived inference;
- the code quality rules that prevent the staged approach from becoming a
  shortcut;
- the major risks: software global barrier, occupancy, precision drift,
  stateful mixer bugs, scratch aliasing, and documentation drift.

Updated:

- `README.md` to point readers to the execution map;
- `IMPLEMENTATION_PLAN.md` to replace stale diary 0101-0109 current-checkpoint
  language with the diary 0101-0114 state;
- `docs/megakernel_development_philosophy.md` to point to the new map and
  update the current-position framing from diary 0113 to diary 0114;
- `docs/megakernel_phase_rationale.md` to replace the now-completed recurrent
  "next required gate" with the current persistent layer-composition gate;
- `diary/README.md` to link this entry.

No shader, runtime, app, fixture, or CMake behavior was intentionally changed in
this entry.

## Current reasoning

The shortest honest explanation is that the final megakernel should be a
composition of already-tested contracts, not the first place those contracts are
tested. A wrong generated token after a full fused decode path is too coarse a
signal. It might come from tensor packing, RMSNorm, q/k/v preparation, DeltaNet
conv state, recurrent update, g/beta calculation, attention KV cache, residual
ordering, MLP math, scratch aliasing, barrier visibility, final norm, LM head,
or token selection.

The current staged plan keeps those failure domains local. The conventional
runtime produces real captured checkpoints. The extraction tools preserve those
checkpoints as raw fixtures. Standalone probes close local equations against
the fixtures. Barrier and persistent skeleton probes test the execution model
separately from model math. Then the project composes one layer, then several
layers, then all layers, then the final token loop.

The MLP/RMSNorm path came before full token-mixer persistence because it was the
smaller state surface while still exercising the important persistent-kernel
constraints: real weights, fp16 storage, fp32 accumulation, multi-stage scratch,
software barriers, activation math, and residual updates. That path gives the
DeltaNet work a downstream consumer. Once the token mixer computes
`mixer_residual`, the post-mixer tail already has a bounded, documented gate.

The DeltaNet component ladder came next because DeltaNet is the larger stateful
risk for this model. Layer 0 now has exact gates for the stateless projections,
scalar branch, conv/L2 handoff, recurrent core, and composed mixer output. That
means the next useful step is not another generic plan or a jump to all 24
layers. The next useful step is to move those closed contracts into the
128-lane persistent layer scaffold in the smallest order that keeps failures
diagnosable.

## Quality position

This plan does not compromise on code quality. It raises the bar by requiring
evidence at every handoff:

- exact gates where the GPU-vs-GPU contract should be exact;
- explicit bounded ULP policy only where CPU-vs-GPU math boundaries justify it;
- JSON diagnostics that expose mismatch counts, max ULP, generation counters,
  failure counters, and tolerance behavior;
- CTest names that distinguish exact gates, bounded gates, and expected exact
  failures;
- diary entries that say what a probe proves and what it does not prove;
- no claim of GPU residency or megakernel completion while host bridges remain
  in the claimed hot path.

The cost is more probes and more documentation. That is acceptable because the
alternative is debugging an opaque shader with only token drift as evidence.

## Verification

Documentation-only change. Verification performed or required for this entry:

- `python3 tests/run_diary_check.py`
- `git diff --check`

No build or CTest behavior is expected to change from these documentation edits.
The existing incomplete shader edit from the prior implementation attempt is
not part of this diary entry and should not be committed with it.

## Remaining scope

This entry does not complete any implementation milestone. It is intentionally
only a planning and documentation checkpoint.

Remaining target-path work includes:

- finish the persistent layer-0 projection-prefix gate;
- add persistent conv/L2, g/beta, recurrent, norm-gate, output projection, and
  first residual add;
- compose full persistent layer-0 output through the existing post-mixer tail;
- add representative DeltaNet and attention layer coverage;
- run bounded multi-layer persistent decode;
- extend to all 24 layers;
- add final norm, LM head, token selection, and GPU-resident next-token handoff;
- archive the first basic test inference from the target path.

The completion estimate should remain conservative until those integrated
target-path pieces exist. The project has strong prerequisites, but the final
Vulkan-native megakernel target is not achieved by prerequisites alone.

# 0111: Megakernel Phase Rationale

## Goal

Write down the current development philosophy in a form that directly answers
why the project is building prerequisites before composing the Vulkan-native
megakernel.

The target is unchanged: `Qwen/Qwen3.5-0.8B` decode on `AMD Radeon RX 6750 XT
(RADV NAVI22)`, with the hot decode path GPU-resident and persistent-dispatch
if the software global barrier remains correct, stable, and worthwhile.

## What changed

Added `docs/megakernel_phase_rationale.md`.

The new document explains:

- the intended target-path shape from GPU-resident token through all 24 layers,
  final norm, LM head, token selection, and next-token handoff;
- why a full fused shader is the wrong first debugging surface;
- the phase ladder from artifact contract through archived basic inference;
- what each phase proves and why it must come before the next phase;
- how conventional runtime captures, fixtures, component probes, barrier
  probes, persistent skeletons, MLP probes, DeltaNet probes, layer composition,
  multi-layer decode, and LM-head/token-loop work fit together;
- why this plan is a code-quality strategy rather than a compromise.

Updated `docs/megakernel_development_philosophy.md` and
`IMPLEMENTATION_PLAN.md` to point to the dedicated rationale document.

## Current philosophy

The project should continue to close contracts in the smallest useful order:

1. make model bytes and parity criteria unambiguous;
2. use the conventional Vulkan runtime as the observable diagnostic spine;
3. turn real runtime checkpoints into fixtures;
4. prove local equations with exact or explicitly bounded gates;
5. prove the persistent execution model separately from model math;
6. compose one explainable layer before widening to multiple layers;
7. add final norm, LM head, token selection, and next-token handoff only after
   hidden-state production is trustworthy;
8. archive basic test inference only when the target path itself runs.

This keeps the final megakernel goal intact while preventing the project from
confusing infrastructure maturity with final-path completion.

## Why the phase order is intentional

The final target combines several independently risky systems: a real model
artifact, fp16/fp32 numerical contracts, stateful DeltaNet recurrence,
attention KV-cache behavior, two residual updates per layer, software
cross-workgroup synchronization, scratch reuse, final logits, and a
GPU-resident token loop. A generated token is the final observable product of
all of those systems, but it is a poor diagnostic. If the project jumps
directly to a full fused shader and gets a wrong token, the failure could live
almost anywhere.

The current order keeps failures local. The conventional Vulkan runtime exists
because it can expose real hidden states from the actual decode path. Component
extraction turns those hidden states into raw fixtures without changing the bit
patterns. Standalone probes then test single equations against those fixtures:
matvec handoffs, residual addition, RMSNorm, MLP, DeltaNet scalar branches,
conv/L2 preparation, and next the recurrent core. Persistent skeletons and
barrier probes answer a different question: whether the RX 6750 XT/RADV stack
can sustain the synchronization and workgroup residency pattern required by a
long-running decode dispatch.

Those tracks meet only after each has a contract. The layer-shaped persistent
probe should not be the first place where q/k/v preparation, g/beta layout,
recurrent state mutation, residual ordering, post-mixer RMSNorm, MLP scratch,
and barrier progress are tested together. It should be the first place where
already-tested pieces are forced to coexist. That is the distinction between
composition and guesswork.

This also explains the low megakernel completion estimate. The project has a
lot of useful infrastructure, and that infrastructure lowers risk. But the
goal is not "many probes exist." The goal is archived basic inference from the
target path. Until a layer-shaped persistent path, bounded multi-layer decode,
all-layer decode, LM head, token selection, and GPU-resident next-token loop
exist, the final-path percentage should remain conservative.

## Verification

Documentation-only change. Required verification:

- `python3 tests/run_diary_check.py`
- `git diff --check`

Broader build and focused CTest verification are still required for the
recurrent fixture boundary in diary 0110.

## Remaining scope

- Not recurrent-core parity.
- Not layer-shaped persistent decode.
- Not 24-layer persistent decode.
- Not final norm, LM head, token selection, or archived inference.

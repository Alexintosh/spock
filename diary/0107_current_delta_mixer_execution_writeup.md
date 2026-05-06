# 0107: Current DeltaNet Mixer Execution Writeup

## Goal

Update the project documentation so the current execution philosophy is written
down in the same terms as the actual target: a Vulkan-native persistent
megakernel for `Qwen/Qwen3.5-0.8B` on the `AMD Radeon RX 6750 XT (RADV
NAVI22)`.

This is a documentation entry. It does not add a shader, runtime gate, fixture,
or inference result. It clarifies why the project is closing small DeltaNet
contracts before attempting a fused layer or full megakernel.

## Why this was needed

`docs/megakernel_development_philosophy.md` already stated the broad strategy,
but part of it still read as if RMSNorm+MLP were the active center of gravity.
That was historically true: the persistent MLP probe gave the project a
bounded downstream consumer and proved multi-stage persistent dispatch with
real weights, scratch lifetimes, residual semantics, and explicit tolerance
policy.

After diaries 0099-0106, the active center of gravity moved to DeltaNet:

- diary 0099 proved the mixer residual-add handoff exactly;
- diary 0101 proved the DeltaNet output projection exactly;
- diary 0102 proved the norm-gate stage exactly;
- diary 0103 proved the z projection exactly;
- diary 0104 proved the raw qkv projection exactly;
- diary 0105 proved the A/B projections exactly;
- diary 0106 proved g/beta exactly by fp32 bit comparison.

The docs now reflect that sequence directly.

## Current development philosophy

The project should continue to walk backward from the value consumed by the
rest of the layer:

1. prove the residual equation after token mixer output;
2. prove the DeltaNet output projection;
3. prove norm-gate;
4. prove stateless input projections;
5. prove g/beta;
6. prove conv1d mutation and q/k L2 normalization;
7. prove recurrent core production of `dn_core_fp16`;
8. compose the full layer-0 mixer path without substituting captured
   intermediates after `dn_input_norm_fp16`;
9. compose the layer-shaped persistent probe;
10. widen to representative layers, bounded multi-layer decode, all 24 layers,
    final norm, LM head, token selection, and archived basic inference.

This is not a reduction in ambition. The target is still the persistent
Vulkan-native megakernel. The small gates exist because a fused megakernel is a
bad place to discover whether an error came from qkv preparation, recurrent
state decay, residual addition, RMSNorm, MLP projection, scratch reuse, or a
software barrier.

## Documentation changes

Updated `docs/megakernel_development_philosophy.md` to:

- rename the stale RMSNorm+MLP focus section into historical context;
- add a current DeltaNet-focus section;
- describe how observable runtime captures, component gates, persistent
  synchronization, and layer/decode composition fit together;
- add DeltaNet projection, scalar, conv/L2, and recurrent checks to the test
  strategy;
- replace stale RMSNorm-first next milestones with the current conv/L2,
  recurrent, layer-shaped, multi-layer, LM-head, and archive sequence.

Updated `IMPLEMENTATION_PLAN.md` so its top-level checkpoint points to the
current diary-0101 through diary-0106 DeltaNet closure rather than diary 0100.

## Verification

Documentation-only change. Required verification:

- `python3 tests/run_diary_check.py`
- `git diff --check`

The broader g/beta implementation verification remains covered by diary 0106.

## Remaining scope

- Not conv1d or q/k L2 parity.
- Not recurrent core parity.
- Not layer-shaped persistent decode.
- Not inference.
- Not megakernel completion.

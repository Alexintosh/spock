# 0100: Current Megakernel Execution Philosophy

## Goal

Record the current development philosophy for the real project target:
a Vulkan-native persistent megakernel path for `Qwen/Qwen3.5-0.8B` on the
`AMD Radeon RX 6750 XT (RADV NAVI22)`, ending in archived basic test inference.

This entry is documentation-only. It exists because the project now has enough
validated infrastructure to make progress look deceptively larger than the
actual megakernel completion. The distinction matters. Runtime gates, captured
fixtures, barrier probes, and persistent micro-probes are necessary, but they
are not the final target.

## Current stance

The current plan is still the original Vulkan-native megakernel plan. It is not
a pivot to a generic backend, and it is not an attempt to compete with
llama.cpp by becoming a broad framework.

The target path remains:

- model-specific;
- batch-1 decode-first;
- tuned for the local RX 6750 XT/RADV stack;
- GPU-resident through the hot decode path;
- persistent-dispatch if the software global barrier remains reliable under
  real work;
- archived with a reproducible basic inference command and expected output.

The fallback remains honest: if persistent dispatch is not correct, stable, or
faster on this stack, the strongest valid result is a fused single-submit
Vulkan path, not a falsely labeled megakernel.

## Why the project builds narrow gates first

A fused megakernel is a poor debugging environment. If token generation fails
after all layers, residual updates, token mixer state, RMSNorm, MLP, final norm,
LM head, token selection, scratch reuse, and software barriers are fused, the
wrong token does not identify the bug.

The project therefore builds the megakernel from contracts:

1. artifact layout and reference parity define what correct means;
2. the conventional Vulkan runtime provides observable layer checkpoints;
3. component extraction turns those checkpoints into stable fp16 fixtures;
4. residual and projection probes validate small equations with real data;
5. persistent probes validate barrier-staged compute and scratch lifetimes;
6. layer-shaped probes will compose already measured contracts;
7. multi-layer persistent decode comes only after one layer is explainable;
8. final norm, LM head, and token selection come after the layer path is stable.

This is not a compromise on code quality. It is the quality strategy. Every
tolerance must be explicit, bounded, reported, and justified by a numerical
contract. Every new fused stage should still have a smaller test that can
explain a failure.

## What the recent gates prove

Diaries 0094-0098 mapped the MLP side under captured RMSNorm input:

- RMSNorm output is exact for layer 0, step 1;
- raw gate projection is within 1 fp16 ULP;
- raw up projection is within 2 fp16 ULP;
- activation product is within 2 fp16 ULP;
- down projection is within 2 fp16 ULP;
- post-residual output has sparse bounded tails, tracked by max and population
  gates;
- layer 20 confirms the same population-gate discipline on a mid-network layer.

Diary 0099 opened the token-mixer side at the smallest useful boundary:

```
input_hidden_fp16 + mixer_output_fp16 -> mixer_residual_fp16
```

The Vulkan residual-add probe reproduces that equation exactly for layer 0,
step 1. That does not prove DeltaNet computation. It proves the handoff that a
future DeltaNet mixer probe must satisfy.

## Why the next gate should be DeltaNet output projection

The next narrow token-mixer computation gate should be:

```
dn_gated_fp16 + layer.0.delta_out_proj -> dn_out_fp16
```

This is the right next step because it is the last projection before the mixer
result enters the residual stream. If this matvec is correct, the existing
residual-add probe can immediately validate:

```
input_hidden_fp16 + dn_out_fp16 -> mixer_residual_fp16
```

That gives the token-mixer track the same debugging shape that the MLP track
now has: prove the consumed output first, then walk backward through the more
stateful internals. For DeltaNet, the harder pieces are recurrent state,
short-convolution state, gate/norm behavior, and scratch aliasing. Those should
be integrated after the output contract is pinned down.

## What remains before the goal

The project is still early relative to the final megakernel target. The missing
work is not cosmetic:

- DeltaNet token-mixer computation parity under captured checkpoints;
- attention token-mixer parity for the attention layers;
- layer-shaped persistent execution with both residual updates;
- scratch lifetime and state-transition discipline across real layers;
- bounded multi-layer persistent decode;
- full 24-layer persistent decode;
- GPU-resident final norm, LM head, token selection, and next-token loop;
- archived basic test inference from the target path.

The practical completion estimate for the Vulkan-native megakernel target
should therefore remain low even though project maturity is much higher. A
reasonable current estimate is around 18-22% of the final target, depending on
whether the estimate counts only integrated target-path code or also the
diagnostic infrastructure that makes that target reachable.

## Documentation changes

This entry updates `docs/megakernel_development_philosophy.md` to make the
post-0099 execution plan explicit and to remove stale "after diary 0090"
framing. The current plan is now written down as:

1. prove the DeltaNet output projection gate;
2. reuse the residual-add gate downstream;
3. walk backward through token-mixer internals;
4. compose a layer-shaped persistent probe;
5. widen to multi-layer and then full decode;
6. archive basic inference only after the target path itself runs.

## Verification

Documentation-only change. The required verification is:

- `python3 tests/run_diary_check.py`
- `git diff --check`

No runtime behavior, shader, fixture, or CTest expectation changes in this
entry.

## Remaining scope

- Not a new Vulkan probe.
- Not DeltaNet output projection parity.
- Not layer-shaped persistent decode.
- Not inference.
- Not megakernel completion.

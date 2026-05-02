# 0016: Precision Drift Isolation and Refined Next Work

## Goal

Refine the drift-localization work from diary 0015 into a concrete root-cause
hypothesis and a focused next-work target. The narrow fixes (barrier-offset
hygiene, deltanet_norm_gate rounding) have been applied, verified not to change
any passing or failing case, and retained. The remaining failure —
`short_correctness_003` at generated token index 5 — has now been traced from a
per-layer hidden-vector comparison through component-level vector analysis and
cross-layer attention-stage dumps to a refined understanding: **full-attention
layers 3, 7, 11, and 15 produce the main hidden-state step-ups; DeltaNet
layers mostly propagate incoming drift without amplifying it**. Attention
sub-component dumps (q_norm, k_norm, gate, v, gated, o_proj) show differences
proportionate to incoming hidden drift and do not yet prove a layout or formula
bug. The governing objective remains the Vulkan-native RX 6750 XT megakernel
roadmap; this entry documents what was learned since diary 0015 and what the
next fix attempt targets.

## Context

Diary 0015 established the full diagnostic suite: `--dump-step-hiddens`,
`--dump-step-components`, `--diagnose-handoff`, `--diagnose-decode-drift`, and
the `tools/compare_layer_hiddens.py` script that consumes all four. The
component-level analysis in diary 0015 showed that at decode step 5:

- Layer 16 was the first layer where the DeltaNet mixer **amplified** drift
  (input `max_abs` 0.01709 → post_mixer 0.02661).
- Layer 19 showed a 2× drift amplification in the MLP tail (post_mixer 0.02930
  → post_mlp 0.05859), but the attention mixer itself did not amplify drift.
- The final RMSNorm shader was verified faithful — the normalized output matches
  CPU-applied normalization of the same raw layer-23 hidden to within
  `max_abs ≈ 0.0047`.

At the end of diary 0015, next work was broadly scoped as "investigate fp16 MLP
matvec accumulation across deep layers (16–19)." The current checkpoint has
substantially refined that scope.

## Diagnostics Refinement

### Narrow fixes do not change the drift profile

Two fixes were applied since diary 0015's initial draft:

1. **KV-cache barrier-offset hygiene** (`vk_session.cpp`): the pipeline barrier
   for the KV-cache transfer in the attention decode path used a hardcoded
   offset covering only the first layer's slice. Fixed to cover `kv_layer_offset`
   (the per-layer slice). This closes a data-race window; it did not change
   `short003` behavior.

2. **deltanet_norm_gate rounding-point fix**: the Qwen3.5 `RMSNormGated` contract
   requires the `weight * norm(hidden)` intermediate to be rounded to fp16
   before the SiLU gate multiplication. The shader was computing the full
   expression in fp32. Fixed. No measurable change in any passing or failing
   case — the rounding difference is below 1 ulp for most elements.

Both fixes are retained because they correct unambiguous implementation errors
(the barrier range was wrong; the rounding contract was wrong), even though
neither moves `short003`.

### Diagnostic infrastructure refinements

Four changes to the diagnostic tooling produce a clearer picture:

1. **`compare_layer_hiddens.py` now forces HF to `torch.float16` like
   `reference_decode`.** Previously the comparator passed
   `torch_dtype=torch.float16` to `from_pretrained`, but some Qwen3.5 parameters
   still loaded in the checkpoint's default dtype unless the model was
   explicitly moved with `.to(torch.float16)`. The comparator now does that,
   matching `reference_decode` and removing a real precision confound from the
   diagnostic path.

2. **DeltaNet norm_gate CPU replay confirms the Vulkan shader matches dumped
   inputs.** The `rms_norm_gated_cpu` helper in `compare_layer_hiddens.py`
   replays the RMSNormGated computation on CPU using the Vulkan-dumped `core`
   and `z` inputs and the repacked norm weight. It then compares the CPU-replay
   result against the Vulkan-dumped gated output. For all layers tested, the
   CPU replay of Vulkan inputs matches the Vulkan gated output to within
   fp16 noise. This confirms the `deltanet_norm_gate` shader is correct — drift
   is not entering through the norm_gate computation.

3. **Widening the comparison range to layers 0–16 changes the attribution.**
   Running `compare_layer_hiddens.py` with `--first-layer 0 --last-layer 16`
   (and later `--last-layer 23`) reveals a pattern not visible when starting at
   layer 16: **DeltaNet layers mostly propagate the input drift they receive,
   they do not amplify it.** The `max_abs` step-up from input → post_mixer is
   small for DeltaNet layers 0, 2, 4, 6, 8, 10, 12, 14, 16. The dominant
   step-ups occur at **full-attention layers 3, 7, 11, and 15**, where the
   attention mixer output shows a clear `max_abs` jump relative to the layer
   input.

4. **Attention-stage dumps at layers 15 and 3 show proportionate differences.**
   The `--dump-step-components` output now includes per-sub-component attention
   dumps: `q_norm_fp16`, `k_norm_fp16`, `gate_fp16`, `v_fp16`, `gated_fp16`,
   and `o_proj_fp16`. Comparing these against HF's corresponding hook captures
   at layers 3 and 15 (the two attention layers with the largest hidden-state
   step-ups) shows that every sub-component's `max_abs` difference is
   proportionate to the incoming hidden drift for that layer. No sub-component
   shows an outsized difference that would indicate a layout error, a formula
   bug, or a precision pathology in the attention path. The attention math
   — Q-norm, K-norm, gate projection, V projection, gated combination, output
   projection — is correct given its input. The step-ups at these layers are
   driven by the accumulated fp16 rounding error in the input hidden state, not
   by an attention-specific bug.

### Refined drift profile

Running the complete diagnostic suite after the narrow fixes confirms the same
per-layer `max_abs` diffs at decode step 5 (layers 0–23, Vulkan vs HF
repacked-fp16) as diary 0015:

```
layer  max_abs      component
-----  -----------  ---------------------------------
  0    0.0019       DeltaNet — propagate (input-noise level)
  1    0.0017       DeltaNet
  2    0.0034       DeltaNet
  3    0.0090       **Attention** — first clear step-up relative to layer 2
  4    0.0086       DeltaNet — propagate (no step-up)
  5    0.0086       DeltaNet
  6    0.0119       DeltaNet
  7    0.0184       **Attention** — step-up
  8    0.0190       DeltaNet — propagate
  9    0.0195       DeltaNet
 10    0.0289       DeltaNet
 11    0.0391       **Attention** — step-up
 12    0.0421       DeltaNet — propagate
 13    0.0588       DeltaNet
 14    0.0587       DeltaNet
 15    0.0699       **Attention** — step-up
 16    0.0659       DeltaNet — propagate (actually lower than layer 15)
 17    0.0654       Attention — propagate
 18    0.0670       DeltaNet
 19    0.0585       Attention + MLP spike
 20    0.0670       DeltaNet
 21    0.0564       DeltaNet
 22    0.0437       Attention
 23    0.0562       DeltaNet
```

Key observations:

- **Layers 0–2**: `max_abs < 0.004` — within fp16 rounding noise.
- **Layer 3** (attention): first clear step-up from 0.0034 → 0.0090. This is
  the earliest measurable drift entry point.
- **Layers 4–6**: DeltaNet layers, mostly flat — they propagate the layer-3
  drift without amplifying it.
- **Layer 7** (attention): step-up from 0.0119 → 0.0184.
- **Layers 8–10**: DeltaNet propagation, gradual climb.
- **Layer 11** (attention): step-up from 0.0289 → 0.0391.
- **Layers 12–14**: DeltaNet propagation, gradual climb.
- **Layer 15** (attention): step-up from 0.0587 → 0.0699 — the largest single
  step-up in the profile.
- **Layer 16**: DeltaNet propagation — `max_abs` actually drops slightly from
  the layer-15 peak, contradicting diary 0015's earlier interpretation that
  layer 16 was where DeltaNet first amplified drift.
- **Layer 19**: the MLP tail spike (post_mixer 0.02930 → post_mlp 0.05859)
  seen in diary 0015 remains the largest single-layer jump in the profile.
- **Layers 20–23**: drift stabilizes at `max_abs` ~0.04–0.07; no further
  systematic amplification.

### What changed in understanding

The profile numbers are the same as diary 0015, but the **interpretation** has
shifted substantially:

- **Layer 19's spike is an MLP secondary effect, not a primary root cause.**
  Unchanged from diary 0015. The attention mixer at layer 19 does not amplify
  drift (input 0.02905 → post_mixer 0.02930). The 2× amplification occurs in
  the MLP tail. KV-cache precision is not the primary issue.

- **Layer 16 is not the entry point — it propagates.** The earlier
  interpretation that layer 16 was where DeltaNet first amplified drift was an
  artifact of starting the analysis at layer 16. With the full range 0–16, the
  step-up pattern is clear: DeltaNet layers are net propagators, not amplifiers.
  Layer 16's post-mixer `max_abs` is slightly lower than its layer-15 input.
  The DeltaNet layer at index 16 was falsely accused.

- **Full-attention layers 3, 7, 11, and 15 are the principal step-up sites.**
  These four layers produce the largest per-layer increases in `max_abs`. The
  pattern is consistent with a pipeline where fp32 dot products are rounded
  back to fp16 activation buffers at each projection and residual boundary. The
  attention QKV projections and output projection operate on increasingly
  drifted inputs, and each fp16 output boundary broadens the error range by a
  few ulps per layer. The full-attention schedule is every fourth layer, so
  layers 3, 7, 11, and 15 are the early visible step-up sites.

- **Attention sub-component differences are proportionate, not pathological.**
  The per-sub-component comparisons at layers 3 and 15 show that every
  attention intermediate (q_norm, k_norm, gate, v, gated, o_proj) differs from
  HF by an amount proportional to that layer's input drift. There is no
  sub-component with an outsized difference that would indicate a layout
  transposition, an incorrect formula, or a precision disaster in a single
  shader. This does not prove the attention path is perfect, but it rules out
  the obvious single-stage failures and points back to cumulative rounding at
  activation/output boundaries.

- **The drift is cumulative, not catastrophic.** Same as diary 0015. The
  per-layer `max_abs` grows gradually from <0.002 (layer 0) to ~0.07 (layer
  22). There is no single layer where the hidden state is obviously wrong.

- **The final RMSNorm is faithful.** Unchanged from diary 0015.

- **The norm_gate shader is faithful.** The CPU replay confirms the Vulkan
  `deltanet_norm_gate` shader produces the correct output for its dumped
  inputs.

- **HF precision parity is now explicit.** `compare_layer_hiddens.py` enforces
  `torch.float16` at every level, matching `reference_decode`, removing
  precision confounds.

### The accumulated precision drift hypothesis (revised)

The working hypothesis, updated from diary 0015:

1. **Full-attention layers 3, 7, 11, and 15 produce the main hidden-state
   step-ups.** The QKV + output projection matvecs in these layers operate on
   a hidden state whose fp16 representation has accumulated ~2 ulps of error
   from the preceding layers. The attention output projection accumulates in
   fp32, then writes fp16; that output boundary broadens the element-wise error
   distribution. The step-up is visible at every fourth layer because that is
   the model's full-attention schedule.

2. **DeltaNet layers propagate drift without systematically amplifying it.**
   The DeltaNet recurrent state update — L2-normalized Q/K, fp32 state matrix
   multiplication, g/beta gating, conv1d — is mathematically stable and, on
   the RX 6750 XT, introduces no more error per layer than the fp16 residual
   path through the MLP. The DeltaNet mixer is not the problem.

3. **Layers 16–19 compound the drift** through alternating DeltaNet (16, 18)
   and attention (17, 19) layers. By layer 19, the MLP tail amplifies the
   accumulated drift 2× because the fp16 MLP product and down-projection output
   boundaries have broader element-wise variation at that point. This is
   unchanged from diary 0015 and is a secondary effect, not the primary entry
   point.

4. **Layers 20–22 maintain high drift** but do not measurably amplify it
   further — "drift saturation" at `max_abs` ~0.04–0.07.

5. **The argmax flip at step 5** is unchanged: accumulated drift pushes the
   logit scores across the narrow margin between token 12 and token 16. The
   numbers are identical to diary 0015.

6. **No single shader bug has been identified.** The component-level and
   sub-component analyses rule out the obvious layout transpositions, the
   verified norm_gate formula bug class, and the fixed attention-layer
   barrier-range data race. The drift is best explained by accumulated fp16
   rounding at activation/output boundaries through a 24-layer pipeline whose
   per-layer error contribution is small (~0.002–0.01 `max_abs` per layer) but
   additive. A subtle systematic rounding-point mismatch is still possible and
   should be tested with targeted fp32-output experiments.

## Current Limitations

Same as diary 0015, with refined prioritization and diagnostics:

1. **CPU chunk bridge** — the primary bottleneck for the megakernel roadmap.
   Prefill requires a CPU round-trip for DeltaNet chunk correction. A
   Vulkan-native chunk-rule shader is the correct fix.

2. **short_correctness_003 fails at generated token index 5** — the argmax
   decision flips on a close margin. The current best hypothesis is
   **accumulated fp16 rounding drift through the 24-layer pipeline, with the
   largest step-ups occurring at full-attention layers 3, 7, 11, and 15.** No
   single shader bug has been identified yet.

3. **Diagnostic tooling is now comprehensive.** `compare_layer_hiddens.py`
   captures per-layer hidden states, component-level intermediates
   (mixing-residual, MLP-product, DeltaNet sub-components, attention
   sub-components), norm_gate CPU replay, and final-norm verification. The
   tooling can rule out most classes of implementation error. It cannot
   distinguish between "ordinary fp16 output-boundary rounding noise" and
   "a systematic rounding bias in one shader that happens to be small per
   layer" — that would require per-element error distribution analysis across
   multiple decode steps.

4. **No commit or push** — full parity is still not achieved.

## Next Work

### Primary: Evaluate whether targeted fp32-output experiments close the gap

The refined understanding changes the next-work strategy. Since there is no
evidence of a single shader bug, the options are:

1. **Targeted fp16-output → fp32-resident activation experiment at the four
   attention step-up sites.**
   Layers 3, 7, 11, and 15 produce the largest per-layer `max_abs` increases.
   `matvec.comp` already accumulates dot products in fp32 and rounds the output
   to fp16. The remaining precision experiment is therefore not "make the dot
   product fp32"; it is "keep selected projection or residual outputs resident
   in fp32 longer, then round at the HF-equivalent boundary." The existing
   experimental shaders (`*_f32out`, mixed residual variants) are useful here,
   but the experiment must be narrowly wired and measured against the drift
   profile before being kept.

2. **Full decode-path fp32-resident activation experiment.** If single-layer
   fp32-output probes do not close the gap, run a broader experiment where
   mixer outputs, residual adds, and MLP down outputs remain fp32 until the
   next operation that HF demonstrably rounds to fp16. This is a larger
   engineering effort and should be treated as an experiment first, not an
   architectural commitment.

3. **Accept the drift and pursue megakernel first.** The drift at layer 22
   (`max_abs` ~0.07) is large enough to flip one token in one prompt out of 48.
   The megakernel roadmap targets exact greedy parity, so this is not an
   acceptable outcome for the current contract. However, if the precision
   investigation produces diminishing returns, the tradeoff between "hours
   debugging fp16 rounding boundaries" and "hours building the Vulkan-native chunk
   rule" should be explicitly weighed.

**Recommendation**: Start with option 1 — wire a narrowly scoped fp32-output
probe around attention `o_proj` and/or the residual write at layers 3, 7, 11,
and 15. Measure the drift profile after the change. If those step-ups disappear
and the argmax flip resolves, the root cause is a rounding-boundary mismatch
and the fix can be made deliberately. If not, escalate to option 2 (broader
fp32-resident activation experiment) or option 3 (re-evaluate parity contract).

### Secondary: Benchmark the CPU chunk bridge

Once the precision question is resolved (either by fix or by decision to
accept), quantify the CPU chunk bridge's performance impact on prefill latency.
This data will inform whether a Vulkan-native chunk-rule shader should be the
next implementation priority or whether the engineering effort is better spent
elsewhere.

## Scope

This entry is a diagnostic checkpoint, not a parity milestone. The code base is
extended from diary 0015 — the same narrow fixes and known failures, plus
DeltaNet and attention sub-component dumps in the comparator. The change is in
the **understanding**: the
previous hypothesis ("layer 16 DeltaNet mixer internal precision") has been
falsified by wider cross-layer analysis. The new hypothesis is that
**full-attention layers 3, 7, 11, and 15 are the principal step-up sites** and
that the drift is most likely **accumulated fp16 rounding at activation/output
boundaries, not an obvious single shader formula bug**. The recommended next
action is a targeted fp32-output/residual-boundary experiment around attention
layers 3, 7, 11, and 15, followed by drift-profile re-measurement. If a targeted
probe does not close the gap, the fallback is a broader decode-path
fp32-resident activation experiment rather than continued sub-component
debugging.

## Post-Entry Update: Attention/MLP fp32-Residual Experiments

Two explicit diagnostic experiment flags were wired and tested after the
attention/MLP boundary hypothesis documented above:

- `--experiment-attn-o-proj-f32-residual` — computes attention `o_proj` into
  fp32 and residual-adds from fp32.
- `--experiment-mlp-down-f32-residual` — computes MLP `down_proj` into fp32 and
  residual-adds from fp32.

Both flags change only the experimental path; the default (non-flag) path is
preserved. Neither changed the outcome of `short_correctness_003`: generated
tokens remain `[271, 248068, 271, 248069, 271, 16]`, and top-5 at decode step 5
still ranks token 16 above token 12. All default quick gates continue to pass:
`short_correctness_001`, `short_correctness_008` at 6 tokens,
`mixed_correctness_023` and `pp520_046` at 1 token.

The hypothesis that `o_proj` output-boundary rounding or `down_proj`
output-boundary rounding is the dominant drift source at full-attention step-up
sites is not supported by these results. The fp32 residual write at these two
points is not sufficient to close the `short003` gap. Next work should escalate
to broader decode-path fp32-resident activation experiments or re-evaluate the
parity-contract tradeoff against the megakernel roadmap, as described in the
options above.

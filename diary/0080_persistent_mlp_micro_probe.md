# 0080: Persistent MLP Micro-Probe -- Gate, Up, SiLU-Gated Activation, Down

## Goal

Exercise a complete MLP-shaped micro-probe inside the persistent decode skeleton: gate projection, up projection, element-wise SiLU(gate) * up activation, and down projection. Previous entries (0075 through 0079) validated persistent dispatch correctness for individual projection rows and row-strided coverage, but each projection ran in isolation. This entry chains four projections into the gated-MLP pattern that Qwen 3.5 actually uses, so the persistent barrier coordination is tested across multiple dependent stages rather than a single repeated matvec.

## Background

The Qwen 3.5 MLP block computes: `output = down(SiLU(gate(x)) * up(x))`. The gate and up projections each map a hidden-size input to an intermediate-size vector. The element-wise SiLU activation and multiply happen on those intermediate values. The down projection maps intermediate-size back to hidden-size. For Qwen 3.5 0.8B, hidden=1024 and intermediate=3584.

The persistent megakernel cannot execute these projections as separate host-submitted dispatches. Each projection must happen inside the same persistent dispatch, coordinated by software global barriers so that gate and up outputs are fully materialized before the activation step reads them, and the activation output is fully materialized before down reads it. This micro-probe is the first test of that intra-dispatch multi-stage dependency pattern.

The probe deliberately uses reduced dimensions (hidden=128, intermediate=16, output-rows=8, workgroups=8) for its default synthetic run, because the goal is to validate the staged compute-and-barrier pattern at a scale where every value can be independently verified. The `--repack-dir` option then loads real fp16 weights at whatever dimensions the actual model artifact specifies, proving the pattern also works with real data.

## Design

The app `vk_persistent_mlp_probe` uses shader `persistent_mlp_probe.comp`, which runs a single-dispatch persistent micro-probe with the following stages:

1. **Gate and up projections.** Each workgroup computes its assigned rows of both matvecs: `gate_scratch = gate_weight * input` and `up_scratch = up_weight * input`.
2. **First software global barrier.** All workgroups wait until every gate/up scratch row is written.
3. **SiLU(gate) * up activation.** Each workgroup applies the element-wise gated activation: `SiLU(gate_scratch[row]) * up_scratch[row]` and writes the result back into gate_scratch.
4. **Second software global barrier.** All workgroups wait until every activated scratch row is written.
5. **Down projection.** Each workgroup computes its assigned rows of the down matvec: output = down_weight * activated_gate_scratch.

The shader bindings include: control buffer (arrived counter, generation counter, failures), gate scratch buffer, up scratch buffer, input vector, gate weight matrix, up weight matrix, down weight matrix, and output buffer. Two software global barriers are issued, producing an expected generation=2.

The host generates deterministic synthetic fp16 values for input and all three weight matrices by default, and computes expected output checksums using a CPU reference that mirrors the shader's fp32 reduction order and fp16 scratch conversions. When `--repack-dir` is supplied, the host loads real fp16 weight artifacts (`layer.0.mlp_gate`, `layer.0.mlp_up`, `layer.0.mlp_down`) from the repacked model manifest and uses deterministic synthetic input.

## Verification

### Synthetic direct run

Default parameters: hidden=128, intermediate=16, output-rows=8, workgroups=8.

```
build/vk_persistent_mlp_probe --hidden 128 --intermediate 16 --output-rows 8 --workgroups 8
```

Result: status ok, checksum 371183224, expected_checksum 371183224, failures 0, generation 2.

The checksum matches exactly, confirming the full gate/up -> barrier -> SiLU*up -> barrier -> down chain produces bit-identical output to the CPU reference at default scale.

### Real weight direct run

Loading real repacked Qwen 3.5 0.8B MLP weights with synthetic input:

```
build/vk_persistent_mlp_probe \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --hidden 128 --intermediate 16 --output-rows 8 --workgroups 8
```

Result: status ok, checksum 1616650692, expected_checksum 1616650692, failures 0, generation 2.

Real fp16 weights from `layer.0.mlp_gate`, `layer.0.mlp_up`, and `layer.0.mlp_down` pass through the same four-stage pipeline with exact checksum agreement. The weights have different numerical distributions from the synthetic defaults (different checksum values confirm this), proving the shader is not accidentally ignoring the weight data.

### CTest

The focused CTest command passed 9/9:

```
ctest --test-dir build -R "spock_persistent_(decode_skeleton|mlp_probe)|spock_diary_check" --output-on-failure
```

This covers both the existing persistent decode skeleton CTest gates (from diaries 0075 through 0079) and the new persistent MLP micro-probe gate. The MLP probe tests confirm the multi-stage barrier pattern remains correct alongside the previously validated single-projection persistent skeleton.

## Interpretation

This result is structurally different from the single-projection probes in diaries 0075 through 0079. Those entries proved that a persistent dispatch can compute one projection correctly with row-strided coverage. This entry proves that a persistent dispatch can chain multiple dependent compute stages with inter-stage barriers and maintain correctness across the dependency chain.

The two-barrier pattern (generation=2) is the minimum viable MLP dependency graph: gate and up must both finish before activation, and activation must finish before down. A full megakernel would need many more barriers (one per layer boundary at minimum), but this probe validates that the fundamental pattern of "compute dependent producers, barrier, compute activation, barrier, consume activation" works inside a single persistent dispatch on this RADV stack.

The fact that both synthetic and real-weight runs produce generation=2 with zero failures is important. The generation counter confirms that both barriers completed successfully (each barrier increments generation by 1, and all workgroups observed the final generation). Zero failures means no workgroup observed stale data from a prior stage, which would manifest as a checksum mismatch. This is direct evidence that the software global barrier correctly synchronizes dependent compute stages within a single dispatch.

The checksum values themselves tell a useful story. The synthetic run checksum (371183224) and the real-weight run checksum (1616650692) are substantially different, confirming the shader uses distinct weight data in each case. If the shader were accidentally using a zeroed or stale weight buffer, both runs would produce the same checksum or fail validation.

The default dimensions (hidden=128, intermediate=16, output-rows=8, workgroups=8) are intentionally small. At this scale, the GPU work per stage is tiny and the barriers dominate execution time. This is not a performance measurement; it is a correctness gate for the dependency pattern. Scaling to model-width hidden=1024 and intermediate=3584 at full row coverage should follow the same structural pattern, since diary 0079 already proved that row-strided persistent projections work at model width. The value of this entry is proving the multi-stage barrier coordination, not the dimensional scaling.

## What This Is

- **A multi-stage persistent MLP micro-probe.** The persistent dispatch chains gate, up, activation, and down projections with software global barriers between dependent stages.
- **Correctness at two weight configurations.** Both synthetic and real repacked Qwen 3.5 fp16 weights produce exact checksum agreement with CPU reference.
- **Dependency-pattern validation.** The two-barrier (generation=2) result proves inter-stage barriers correctly synchronize dependent compute within a single persistent Vulkan dispatch.
- **A natural extension of the persistent skeleton series.** Entries 0075 through 0079 built single-projection persistent correctness; this entry adds multi-stage dependencies.

## What This Is Not

- **Not inference.** There is no real activation vector from a model step, no residual connection, no layer norm, and no token generation. The input is synthetic (deterministic fp16 values), not an actual hidden-state vector from a preceding attention or DeltaNet block.
- **Not full MLP coverage.** The default dimensions are hidden=128, intermediate=16, output-rows=8. This does not cover all 3584 intermediate rows or the full 1024 hidden width at default scale. Real-weight runs use actual weight shapes but still synthetic input.
- **Not real activations.** The SiLU(gate) * up activation operates on projection outputs from synthetic (or real-weight, synthetic-input) data, not from actual hidden-state activations flowing through the model.
- **Not residual or layer semantics.** The probe does not add the output back to a residual stream, does not apply RMSNorm before or after, and does not represent a complete transformer MLP layer.
- **Not the megakernel.** This is still a standalone probe exercising one narrow aspect of the eventual persistent decode path. It does not run attention, DeltaNet, KV cache updates, or the full 24-layer schedule.
- **Not a performance benchmark.** The small default dimensions and two-barrier pattern mean barrier overhead dominates. No timing claims should be drawn from this probe.

## Next Work

1. Scale the MLP micro-probe to model-width dimensions (hidden=1024, intermediate=3584) with full or partial row coverage and real weights, to confirm the dependency pattern holds at decode-relevant scale.
2. Add a residual-add stage to the probe: input + down(activation), validating the residual stream update pattern inside persistent dispatch.
3. Begin layer-shaped persistent decode: chain attention/DeltaNet stages with MLP stages and inter-layer barriers, approaching the full megakernel structure.

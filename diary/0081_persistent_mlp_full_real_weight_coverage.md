# 0081: Persistent MLP Micro-Probe -- Full Real-Weight MLP Coverage

## Goal

Scale the persistent MLP micro-probe from diary 0080 to the full real layer.0 MLP projection sizes: hidden=1024, intermediate=3584, output_rows=1024, workgroups=82. Diary 0080 proved the multi-stage persistent dependency pattern at small dimensions. This entry proves the same gate/up, SiLU-gated activation, and down-projection chain over the full real fp16 MLP weight slices from the repacked Qwen 3.5 0.8B artifact.

## Background

The MLP block is one of the largest compute pieces in the decode path. For Qwen 3.5 0.8B, the relevant layer.0 tensors are:

- `layer.0.mlp_gate`: `[3584,1024]`
- `layer.0.mlp_up`: `[3584,1024]`
- `layer.0.mlp_down`: `[1024,3584]`

Previous persistent skeleton entries built up to this in stages. Diaries 0075-0077 proved persistent fp16/fp32 projection correctness and real multi-weight loading. Diaries 0078-0079 proved row-strided coverage at model width. Diary 0080 chained the MLP dependency graph in one persistent dispatch, but only at a tiny default scale for its CTest gate. The missing proof was whether the same shader and CPU reference remain exact at full MLP weight dimensions.

This entry keeps the same synthetic deterministic input vector as diary 0080. That is intentional: the purpose is to validate full real-weight MLP compute and persistent barrier coordination, not model-state fidelity. Real hidden activations will come later when this probe is connected to a layer-shaped decode path.

## Verification

### Full real-weight direct run

```
build/vk_persistent_mlp_probe \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --hidden 1024 --intermediate 3584 --output-rows 1024 --workgroups 82
```

Result:

- status: ok
- failures: 0
- arrived: 0
- generation: 2
- expected_generation: 2
- checksum: 2160240877
- expected_checksum: 2160240877

The exact checksum match confirms the shader and CPU reference agree across the full gate/up/down MLP weight slices. The generation value confirms that both software global barriers completed. `arrived=0` confirms the barrier counter returned to its quiescent state after dispatch.

### Prefix scale-up runs

Before the full run, several real-weight prefix sizes were checked:

- hidden=1024, intermediate=128, output_rows=64, workgroups=82: checksum 472311119 matched expected_checksum 472311119.
- hidden=1024, intermediate=256, output_rows=128, workgroups=82: checksum 2696672940 matched expected_checksum 2696672940.
- hidden=1024, intermediate=512, output_rows=256, workgroups=82: checksum 3733362356 matched expected_checksum 3733362356.
- hidden=1024, intermediate=1024, output_rows=512, workgroups=82: checksum 2918281365 matched expected_checksum 2918281365.

These intermediate checks reduce the chance that the full-dimension result is hiding a size-specific issue. The same app, shader, weight loader, and CPU reference passed every prefix before reaching full MLP dimensions.

## CTest

A new gate `spock_persistent_mlp_probe_full_real_weight_smoke` runs:

```
vk_persistent_mlp_probe \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --hidden 1024 --intermediate 3584 --output-rows 1024 --workgroups 82
```

This makes full real-weight MLP coverage part of the persistent probe regression set. It is still a smoke gate rather than a benchmark; it checks correctness of the full real-weight MLP computation with deterministic input.

## Interpretation

This is the strongest persistent-compute result so far for the MLP path. The probe now covers every row and column of the layer.0 MLP gate, up, and down matrices used by the repacked artifact. It also does so inside one persistent dispatch with two software global barriers. That matters because the eventual megakernel needs exactly this kind of in-kernel dependency sequencing: produce intermediate data, synchronize all resident workgroups, transform intermediate data, synchronize again, then consume it in a later projection.

The result does not prove performance. It does prove that the current shader and host reference can survive the numerical range and memory footprint of a full real MLP block. That is a necessary condition before we connect this work to actual hidden states or residual/layer semantics.

The full-dimension checksum, 2160240877, is not meaningful by itself outside this artifact and input pattern. Its value is useful because it is deterministic, gated, and produced by both CPU and GPU references. Any future change to row assignment, fp16 conversion, scratch layout, or down-projection indexing should preserve that checksum unless the change intentionally modifies the numerical contract.

## What This Is

- **Full real-weight MLP coverage.** The probe uses full layer.0 `mlp_gate`, `mlp_up`, and `mlp_down` tensor dimensions from the repacked artifact.
- **Single persistent dispatch.** Gate/up, activation, and down projection execute inside one Vulkan dispatch with two software global barriers.
- **A correctness gate.** The CTest protects full real-weight MLP parity against future changes.

## What This Is Not

- **Not inference.** The input vector is synthetic, not a real hidden state produced by embedding, attention, DeltaNet, or prior layers.
- **Not a complete transformer layer.** There is no RMSNorm, residual add, attention/DeltaNet block, or layer schedule.
- **Not the megakernel.** This is still a standalone probe.
- **Not a throughput result.** No timing or tokens/sec claim is attached to this entry.

## Next Work

1. Add a residual-update variant: output = input + MLP(input), still with synthetic input.
2. Add an RMSNorm-before-MLP variant using real `layer.0.input_norm` or post-attention norm weights.
3. Feed the MLP probe with a real activation vector from an existing decode/reference path.
4. Begin composing the MLP probe with the DeltaNet/attention side of the layer-shaped persistent path.

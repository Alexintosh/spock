# 0082: Persistent MLP Micro-Probe -- Residual Update

## Goal

Extend `vk_persistent_mlp_probe` with an optional residual update so the probe can validate `output = input + down(SiLU(gate(input)) * up(input))` for covered output rows. Diary 0081 proved the full real layer.0 MLP weight path without residual semantics. This entry adds the first residual-stream update pattern inside the persistent MLP probe.

## Background

Transformer layers do not consume an MLP output in isolation. The output is added back to a residual stream. A persistent megakernel therefore needs to preserve not only projection correctness but also the in-dispatch update pattern that writes a transformed vector back into the model state. This is smaller than a complete transformer layer, but it is a necessary step toward layer semantics.

The probe still uses deterministic synthetic input. In residual mode, the down-projection output row is accumulated in fp32, then `input[row]` is added before checksum and output storage. The flag is intentionally optional (`--residual`) so the non-residual checksums from diaries 0080 and 0081 remain stable regression points.

## Implementation

The app now accepts:

```
--residual
```

When enabled, the host validates `output_rows <= hidden`, because each output row adds `input_vec[row]`. The shader push constants now include `residual_enabled`. Stage 3 remains the down projection stage, but lane 0 applies the residual add before storing the output and before adding to the checksum:

```
float total = down_dot;
if (residual_enabled != 0) {
    total += float(input_vec[row]);
}
checksum += floatBitsToUint(total);
```

The CPU reference mirrors this exactly. Non-residual mode remains the default and preserves the existing CTest checksums.

## Verification

### Non-residual default regression

```
build/vk_persistent_mlp_probe --hidden 128 --intermediate 16 --output-rows 8 --workgroups 8
```

Result: status ok, checksum 371183224, expected_checksum 371183224, failures 0, arrived 0, generation 2.

This confirms the residual flag did not perturb the default probe.

### Synthetic residual run

```
build/vk_persistent_mlp_probe --hidden 128 --intermediate 16 --output-rows 8 --workgroups 8 --residual
```

Result: status ok, residual true, checksum 374853240, expected_checksum 374853240, failures 0, arrived 0, generation 2.

The checksum differs from the non-residual default, as expected, and matches the CPU reference exactly.

### Full real-weight residual run

```
build/vk_persistent_mlp_probe \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --hidden 1024 --intermediate 3584 --output-rows 1024 --workgroups 82 --residual
```

Result: status ok, residual true, checksum 3327711045, expected_checksum 3327711045, failures 0, arrived 0, generation 2.

This is the strongest MLP-shaped persistent probe so far: full real layer.0 MLP weights, full hidden-width output rows, deterministic input, residual add, and exact checksum agreement.

### Validation error

```
build/vk_persistent_mlp_probe --hidden 8 --intermediate 16 --output-rows 16 --workgroups 8 --residual
```

Result: exit 2 with message `--residual requires --output-rows <= --hidden`.

### CTest

A new gate `spock_persistent_mlp_probe_full_real_weight_residual_smoke` covers the full real-weight residual path:

```
vk_persistent_mlp_probe \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --hidden 1024 --intermediate 3584 --output-rows 1024 --workgroups 82 --residual
```

## Interpretation

This entry moves the MLP probe one step closer to layer semantics. The output is no longer just a standalone projection result; it can be folded back into an input-like vector with the same row indexing that a residual stream would use. That is still not a real transformer layer, but it validates a pattern the megakernel will need repeatedly: compute a block result, synchronize dependencies, and update the persistent state vector.

The residual mode is deliberately simple. It adds `input[row]`, not a real hidden state produced by prior layer work. It also does not include RMSNorm, attention/DeltaNet, or a post-MLP residual handoff. The point is to isolate the residual arithmetic and checksum contract before combining it with more model semantics.

The full real-weight residual checksum, 3327711045, is now a regression sentinel. If a future change alters fp16 conversion, down-projection indexing, residual ordering, or push-constant layout, this checksum should catch it.

## What This Is

- **Residual update inside the persistent MLP probe.** The shader can now add input rows to the down-projection output before checksum/output storage.
- **Full real-weight residual gate.** The new CTest protects full layer.0 MLP weights plus residual update.
- **A layer-semantics stepping stone.** This validates one piece of the residual stream update pattern needed by persistent decode.

## What This Is Not

- **Not inference.** Input is still synthetic and no token is generated.
- **Not a full transformer layer.** No RMSNorm, attention/DeltaNet, or real hidden-state handoff.
- **Not the megakernel.** This remains a standalone probe.
- **Not a performance claim.** No throughput or timing is reported.

## Next Work

1. Add RMSNorm-before-MLP with real norm weights and synthetic input.
2. Feed the MLP residual probe with an actual activation vector from the existing decode/reference path.
3. Compose MLP residual semantics with the recurrent/attention side of a layer-shaped persistent probe.

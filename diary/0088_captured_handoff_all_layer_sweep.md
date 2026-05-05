# 0088: Captured FP16 Handoff Sweep -- Layer 20 Boundary Case

## Goal

Sweep the layer-selectable persistent MLP probe across all 24 layers using the same real step-1 `mixer_residual` capture, and identify whether the fp16 output equality gate holds beyond the checked-in layer 0 and layer 1 fixtures.

## Background

Diary 0087 made `vk_persistent_mlp_probe` layer-selectable via `--layer N` and added a checked-in layer 1 captured handoff gate. That proved the captured-handoff workflow is not hardcoded to layer 0, but it still covered only two of the model's 24 layers.

Before adding more permanent fixtures, a run-only sweep used the existing `/tmp/spock_components1_stderr.txt` decode-step-1 component dump. For each layer, `tools/extract_component_fp16.py` extracted `mixer_residual_fp16` to a temporary raw fp16 file, and `vk_persistent_mlp_probe --layer N` ran with full real dimensions, real weights, and residual update.

## Sweep Result

The sweep covered all 24 layers:

- 23 layers returned `status: ok` with `output_mismatches == 0`.
- 18 layers also had exact fp32 diagnostic checksum agreement.
- 5 layers had checksum-only diagnostic differences while still matching every final fp16 output row.
- 1 layer, layer 20, failed exact fp16 output equality.

Layer 20 command:

```
vk_persistent_mlp_probe --layer 20 \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --hidden 1024 --intermediate 3584 --output-rows 1024 --workgroups 82 \
  --residual \
  --input-fp16-file /tmp/layer20_step1_mixer_residual.fp16
```

Layer 20 result:

- status: fail
- checksum: 3796215163
- expected_checksum: 3796224648
- output_mismatches: 2
- first_mismatch_row: 657

## Verification

The sweep was diagnostic and run-only. It used the existing `/tmp/spock_components1_stderr.txt` decode-step-1 component dump, extracted each layer's `mixer_residual_fp16` field with `tools/extract_component_fp16.py`, and invoked `vk_persistent_mlp_probe --layer N` for each layer.

The final source tree was restored after temporary instrumentation. Re-running the layer 20 command with the normal binary still exits nonzero with `output_mismatches: 2`, confirming that no tolerance gate or diagnostic-only source change was retained.

## Layer 20 Diagnostic

Temporary instrumentation identified the exact mismatch:

| Output row | GPU fp16 | CPU fp16 | Difference |
|---:|---:|---:|---:|
| 657 | `0x92AC` | `0x92AD` | 1 fp16 ULP |
| 954 | `0x1F68` | `0x1F69` | 1 fp16 ULP |

Stage evidence:

- Up scratch: 0 mismatches across all 3584 intermediate rows.
- Activation scratch: 1 mismatch across all 3584 intermediate rows.
- Activation mismatch row: 1874.
- Activation GPU fp16: `0x1D39`.
- Activation CPU fp16: `0x1D38`.

The single activation difference propagates through the layer 20 down projection. For 1022 output rows it is absorbed by final fp16 rounding. For rows 657 and 954 the final fp32 totals sit close enough to a fp16 rounding boundary that the perturbation flips the final stored fp16 value by 1 ULP.

## Interpretation

This is not a weight-layout issue, extraction issue, barrier failure, or up-projection mismatch. The first divergence appears at the SiLU-gated activation stage, after exact up-projection agreement. It is the same class of GPU/CPU arithmetic boundary seen in diary 0086, but layer 20 is the first captured case where the 1-ULP activation difference survives into final fp16 output for two rows.

The current exact fp16 output gate should remain strict. Layer 20 is a real failure under that contract. Accepting it requires an explicit future precision policy, not a silent gate change.

## What This Is

- **A run-only all-layer captured-handoff sweep** using real step-1 `mixer_residual` captures.
- **Evidence that 23 of 24 layers pass exact fp16 output equality** for this capture.
- **A concrete layer 20 boundary case**: one activation ULP causes two final fp16 output ULP differences.

## What This Is Not

- **Not a committed layer 20 fixture gate.** The layer 20 input remains temporary under `/tmp`.
- **Not a tolerance decision.** Exact fp16 output equality remains the current gate.
- **Not inference.** No RMSNorm integration, no token mixer composition, no LM head, no token generation.
- **Not the megakernel.** This is still standalone persistent MLP validation.

## Next Work

1. Decide the precision contract for persistent MLP captured handoff before adding all-layer permanent gates:
   - keep exact fp16 output equality and treat layer 20 as a blocker;
   - make GPU-produced intermediates the reference for downstream stage checks;
   - or define a documented, bounded fp16 ULP tolerance with regression tests.
2. Prefer avoiding silent tolerance. If tolerance is adopted, it needs explicit JSON fields and tests that still fail for larger drift.
3. Add a diagnostic mode only if it preserves the exact pass/fail gate and remains useful for future layer-shaped probes.
4. Continue toward RMSNorm-before-MLP only after the layer 20 precision policy is settled.

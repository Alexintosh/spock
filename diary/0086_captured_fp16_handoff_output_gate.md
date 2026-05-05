# 0086: Captured FP16 Handoff Probe -- Per-Row FP16 Output Equality Gate

## Goal

Validate the persistent MLP probe against a real hidden-state capture from the decode pipeline, and establish per-row fp16 output equality as the authoritative pass/fail gate rather than fp32 aggregate checksum comparison.

## Background

Diaries 0084 and 0085 built the tooling to extract real intermediate activations from the decode pipeline into raw fp16 files. The `vk_persistent_mlp_probe --input-fp16-file` path can now ingest any captured vector. Diary 0080--0083 validated the persistent MLP probe's multi-stage compute with synthetic input, real weights, residual update, embedding input, and file input -- all using fp32 aggregate checksum agreement as the correctness gate.

The fp32 checksum gate is appropriate when the CPU reference and GPU shader compute identical fp32 intermediate values. But the persistent MLP probe computes SiLU activation (`x * sigmoid(x)`) using `exp()` in GLSL, while the CPU reference uses `std::exp()`. These can produce different fp32 values at the rounding boundary for certain inputs. When those differences land exactly on a fp16 rounding boundary, the GPU and CPU can disagree by 1 fp16 ULP at an intermediate stage, even though the final fp16 output row is identical after down projection.

This entry tests the probe against a real capture -- layer 0, step 1, `mixer_residual` -- and introduces per-row fp16 output equality as the authoritative correctness gate, demoting the fp32 checksum to a diagnostic field.

## Implementation

### Test fixture: `tests/data/layer0_step1_mixer_residual_1024.fp16`

A 2048-byte raw little-endian fp16 file containing 1024 fp16 values extracted from the real decode pipeline's layer 0 step 1 `mixer_residual` component. This is the post-DeltaNet residual stream that feeds into the MLP block.

### Probe behavior change

`vk_persistent_mlp_probe` now reports:

- `output_mismatches` -- count of output rows where the GPU fp16 value does not exactly match the CPU reference fp16 value. This is the authoritative pass/fail gate.
- `checksum` and `expected_checksum` -- fp32 aggregate checksums retained as diagnostic fields. A checksum difference does not cause test failure when `output_mismatches == 0`.

### CTest: `spock_persistent_mlp_probe_captured_fp16_handoff`

A new CTest entry exercises the full real-weight persistent MLP probe with the captured fp16 input:

```
vk_persistent_mlp_probe --hidden 1024 --intermediate 3584 --output-rows 1024 --workgroups 82 \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --input-fp16-file tests/data/layer0_step1_mixer_residual_1024.fp16 \
  --residual
```

The test asserts `output_mismatches == 0` (all 1024 output rows match exactly in fp16).

## Verification

### Captured real-input run

```
vk_persistent_mlp_probe --hidden 1024 --intermediate 3584 --output-rows 1024 --workgroups 82 \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --input-fp16-file tests/data/layer0_step1_mixer_residual_1024.fp16 \
  --residual
```

Result:

- output_mismatches: 0
- checksum: 67820897
- expected_checksum: 67824746
- failures: 0
- arrived: 0
- generation: 2

All 1024 final fp16 output rows match exactly. The checksum difference (67820897 vs 67824746) is a known diagnostic artifact from the GLSL `exp` vs `std::exp` rounding boundary in SiLU.

### Root cause of checksum divergence

Stage 2 (activation: SiLU(gate) * up) had a 1-fp16-ULP activation difference at intermediate row 3180: GPU produced fp16 value `0x832E` while CPU produced `0x832D`. This single intermediate difference propagated through the down projection into the fp32 checksum, producing the observed aggregate divergence. However, the down projection's fp16 output rounding absorbed the 1-ULP difference, so all 1024 final output rows match exactly.

This confirms that fp32 checksum comparison is an overly strict correctness gate for a pipeline that produces fp16 output. Per-row fp16 equality is the correct gate because it tests the actual output representation.

### CTest suite

Full CTest suite passes with the new test added.

## Interpretation

This entry resolves a precision question that was always latent but not observable with synthetic inputs: the persistent MLP probe's CPU and GPU SiLU implementations can disagree at the fp16 rounding boundary, but those disagreements are absorbed by the down projection's fp16 output rounding. The fp32 checksum is sensitive to intermediate-stage rounding differences that do not survive to the output. Per-row fp16 equality is the correct gate because it tests what the downstream consumer actually sees.

The captured handoff test is the strongest validation of the persistent MLP probe to date: real layer 0 weights, real hidden-state input from the decode pipeline, real residual update, and exact fp16 output agreement across all 1024 rows. The probe is now validated end-to-end against a real intermediate activation from the production pipeline.

## What This Is

- **Per-row fp16 output equality as the authoritative correctness gate** for the persistent MLP probe.
- **Validation against a real hidden-state capture** from the decode pipeline (layer 0, step 1, mixer_residual).
- **Understanding of the GLSL `exp` vs `std::exp` checksum divergence** as a benign diagnostic artifact absorbed by down-projection fp16 rounding.

## What This Is Not

- **Not inference.** No token generation, no RMSNorm, no attention/DeltaNet, no LM head.
- **Not RMSNorm integration.** The captured input is a raw mixer_residual/residual-stream value; the probe does not apply RMSNorm. RMSNorm-before-MLP remains future work.
- **Not attention or DeltaNet.** Only the MLP side of one layer.
- **Not the megakernel.** This is a standalone probe with captured real data.
- **Not a performance claim.** No throughput or timing is involved.

## Next Work

1. Extend the captured-handoff probe to additional layers and steps to confirm the fp16 output equality gate holds beyond layer 0 step 1.
2. Add RMSNorm-before-MLP with real norm weights so the probe can accept pre-RMSNorm input.
3. Compose the MLP probe with the token-mixer side of a layer-shaped persistent probe.

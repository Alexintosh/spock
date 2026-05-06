# 0089: FP16 ULP Tolerance Policy for Captured Handoff Probes

## Goal

Introduce an explicit, opt-in fp16 ULP tolerance for the persistent MLP captured-handoff probe, while keeping exact fp16 output equality as the default behavior. This resolves the layer 20 boundary case from diary 0088 without silently tolerating any difference.

## Background

Diary 0088 swept all 24 layers and found layer 20 is the only hard failure: two output rows differ by exactly 1 fp16 ULP from the CPU reference. The root cause is a single activation-stage SiLU `exp` rounding difference between GLSL and `std::exp` that crosses final fp16 rounding boundaries for rows 657 and 954. The other 23 layers pass exact fp16 output equality.

The exact default must remain strict because:
- GPU-vs-GPU comparisons (shader vs shader) produce identical results.
- Only CPU-vs-GPU `exp` divergence causes these 1-ULP differences.
- Silently tolerating drift hides regressions.

## Three-Tier Policy

The probe now implements a three-tier output comparison model:

1. **Tier 0 (default): exact fp16 equality, tolerance 0.** Every output row must match bit-for-bit. This is the gate for GPU-vs-GPU comparisons and for any case where the reference and implementation use identical arithmetic. Layer 20 fails this tier by design.

2. **Tier 1: opt-in bounded ULP tolerance via `--output-fp16-ulp-tolerance N`.** Rows with 0 < ULP diff <= N are counted as `output_within_tolerance` and do not cause test failure. Rows with ULP diff > N are counted as `output_mismatches` and remain gate-breaking. This is the appropriate tier for CPU-vs-GPU captured-handoff probes where the `exp` implementation differs.

3. **Tier 2: opposite-sign mismatch.** If GPU and CPU output rows have opposite signs (and neither is zero), the ULP diff is reported as `UINT32_MAX`. No finite tolerance can accept such a difference. These always count as gate-breaking mismatches.

The policy is scoped to CPU-vs-GPU captured-handoff probes only. GPU-vs-GPU comparisons remain exact. The option does not affect structural checks (barrier generation, arrival count, failure count).

## Implementation

### FP16 ULP diff helper

Added `fp16_ulp_diff(uint16_t a, uint16_t b)` to the anonymous namespace in `apps/vk_persistent_mlp_probe.cpp`:

- Returns 0 for identical values.
- Returns 0 for signed-zero comparisons (+0 vs -0).
- Returns `UINT32_MAX` for opposite-sign non-zero values.
- Returns the absolute difference of raw integer representations for same-sign values (correct because fp16 integer ordering matches magnitude ordering within a sign).

### CLI option

`--output-fp16-ulp-tolerance N` (default 0) accepts only nonnegative integers. Rejects:
- Negative values (sign characters).
- Non-numeric strings.
- Partial strings like `1abc`.
- Empty strings.

All rejections produce a JSON error object with exit code 2, matching the `--layer` parsing pattern.

### JSON output

The probe now reports five output-comparison fields:

- `output_exact_mismatches`: rows where GPU fp16 != expected fp16 (any nonzero ULP diff).
- `output_within_tolerance`: rows where 0 < ULP diff <= tolerance.
- `output_mismatches`: rows where ULP diff > tolerance (gate-breaking count).
- `max_fp16_ulp_diff`: maximum ULP diff observed over all output rows.
- `output_fp16_ulp_tolerance`: the configured tolerance value.

With default tolerance 0: `output_mismatches == output_exact_mismatches` and `output_within_tolerance == 0`.

### Status gate

`status: "ok"` only if structural checks pass AND `output_mismatches == 0`. The checksum is a diagnostic and never a gate.

## Test Fixtures

### Layer 20 fixture

`tests/data/layer20_step1_mixer_residual_1024.fp16`: 2048 bytes (1024 fp16 values) copied from the `/tmp` extraction used in diary 0088. Same step-1 `mixer_residual` capture as layer 0 and layer 1 fixtures.

### Paired CTest gates

1. `spock_persistent_mlp_probe_layer20_captured_fp16_handoff_exact_fails` (WILL_FAIL TRUE): proves the default exact gate still rejects layer 20. Output: `output_exact_mismatches: 2, output_mismatches: 2, max_fp16_ulp_diff: 1`.

2. `spock_persistent_mlp_probe_layer20_captured_fp16_handoff_ulp1`: proves explicit tolerance 1 accepts layer 20. Output: `output_exact_mismatches: 2, output_within_tolerance: 2, output_mismatches: 0, max_fp16_ulp_diff: 1`.

3. Three invalid-tolerance WILL_FAIL gates (`_invalid_negative`, `_invalid_string`, `_invalid_partial`): prove strict parsing rejects `-1`, `abc`, and `1abc`.

## Verification

- Build passes.
- Layer 20 exact command exits 1 with `output_mismatches: 2, max_fp16_ulp_diff: 1`.
- Layer 20 with `--output-fp16-ulp-tolerance 1` exits 0 with `output_exact_mismatches: 2, output_within_tolerance: 2, output_mismatches: 0`.
- Layer 0 and layer 1 captured handoff tests still pass with all new fields.
- Invalid tolerance `1abc` exits 2 with JSON error.
- All existing CTest gates pass.

## What This Is

- **A scoped precision policy** for captured-handoff probes only.
- **Paired exact-fail/tolerance-pass CTest gates** proving both modes work correctly.
- **Explicit opt-in**: the default remains exact fp16 equality.

## What This Is Not

- **Not a global tolerance.** This applies only to `vk_persistent_mlp_probe` output comparison.
- **Not inference.** No RMSNorm integration, no token mixer composition, no LM head, no token generation.
- **Not the megakernel.** This is still standalone persistent MLP validation.
- **Not checksum gating.** Checksums remain diagnostic fields.

## Next Work

1. Add RMSNorm-before-MLP to the persistent probe with real norm weights.
2. Build a layer-shaped persistent probe composing RMSNorm, mixer handoff, MLP, and residual update.
3. Consider whether the ULP tolerance should apply to per-stage scratch comparisons if stage diagnostics are added later.

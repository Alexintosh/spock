# 0090: Pre-MLP RMSNorm Smoke Probe

## Goal

Add an optional `--pre-mlp-rmsnorm` flag to `vk_persistent_mlp_probe` that applies RMSNorm to the input vector before gate/up projections, matching the runtime layer pipeline where `post_norm` normalizes the residual stream before the MLP block.

## Formula

```
out[i] = fp16(input[i] * rsqrt(mean(input^2) + 1e-6) * (1 + weight[i]))
```

Where:
- `input[i]` is the raw fp16 input vector element
- `mean(input^2)` is the mean of squared input values computed in fp32
- `weight[i]` is the `layer.N.post_norm` fp16 weight element
- The weight role is `(1 + weight[i])`, matching the Qwen RMSNorm convention
- Output is fp16-rounded, matching the runtime's behavior before the MLP matvec

## Design

### Shader Stage 0

When `pre_mlp_rmsnorm` is set in push constants, the shader executes a Stage 0 RMSNorm before the existing Stage 1 (gate/up projections):

1. Each workgroup independently computes the full sum-of-squares over `input_vec[hidden]` using lane-strided fp32 accumulation and tree-reduction.
2. All workgroups get the same `inv_rms = inversesqrt(mean_sq + 1e-6)` since they read the same input.
3. All resident invocations stripe over `norm_output[hidden]`, writing each element exactly once as `fp16(input[c] * inv_rms * (1 + weight_norm[c]))`.
4. A global barrier ensures Stage 0 completes before Stage 1 reads `norm_output`.

### New bindings

- Binding 8: `NormOutput` buffer (`float16_t[hidden]`) — written by Stage 0, read by Stage 1
- Binding 9: `WeightNorm` buffer (`float16_t[hidden]`) — `layer.N.post_norm` weights

### Push constants

Grew from 5 to 6 uint32s: added `pre_mlp_rmsnorm` flag.

### Barrier count

Without RMSNorm: 2 global barriers (generation=2).
With RMSNorm: 3 global barriers (generation=3), one additional between Stage 0 and Stage 1.

### Input preservation

The raw `input_vec` is never modified. When `--residual` is also enabled, Stage 3 reads the original input for residual addition while Stage 1 uses the normalized input for gate/up projections. This matches the runtime behavior where RMSNorm output feeds the MLP but the residual bypass uses the pre-norm hidden state.

## CPU Reference

The CPU reference mirrors the shader:

1. Compute `sum_sq = sum(fp32(input[c])^2)` for c in [0, hidden)
2. `mean_sq = sum_sq / hidden`
3. `inv_rms = 1.0 / sqrt(mean_sq + 1e-6)`
4. For each c: `normalized[c] = fp16(fp32(input[c]) * inv_rms * (1 + fp32(weight[c])))`
5. Use `normalized` for gate/up dots, raw `input` for residual addition

## Weight loading

Added `load_weight_vector()` for rank-1 fp16 tensors. `post_norm` is shape [hidden] (1024 in the model, 128 in the smoke test), dtype fp16. The function validates rank-1, dtype, and extracts the first `hidden` elements.

## Smoke test

`spock_persistent_mlp_probe_pre_mlp_rmsnorm_smoke`:
- `--repack-dir artifacts/spock-text-repack-qwen35-0p8b`
- `--hidden 128 --intermediate 16 --output-rows 8 --workgroups 8 --pre-mlp-rmsnorm`
- Layer 0 real weights, real post_norm
- Result: `status: ok`, `generation: 3`, `output_exact_mismatches: 0`, `max_fp16_ulp_diff: 0`

Direct JSON:
```json
{
  "hidden": 128,
  "intermediate_count": 16,
  "output_rows": 8,
  "workgroups": 8,
  "real_weight": true,
  "layer": 0,
  "pre_mlp_rmsnorm": true,
  "repack_dir": "artifacts/spock-text-repack-qwen35-0p8b",
  "status": "ok",
  "failures": 0,
  "arrived": 0,
  "generation": 3,
  "expected_generation": 3,
  "checksum": 1315416920,
  "expected_checksum": 1315416920,
  "output_exact_mismatches": 0,
  "output_within_tolerance": 0,
  "output_mismatches": 0,
  "max_fp16_ulp_diff": 0,
  "output_fp16_ulp_tolerance": 0
}
```

## Verification

- All 20 focused tests pass (persistent MLP probes, diary check, and `extract_component_fp16`)
- `git diff --check` clean
- Default behavior and checksums unchanged for non-RMSNorm tests
- ULP tolerance semantics unchanged
- A full model-width RMSNorm+residual direct run passes exact fp16 output equality:
  - `--hidden 1024 --intermediate 3584 --output-rows 1024 --workgroups 82 --pre-mlp-rmsnorm --residual`
  - `status: ok`, `generation: 3`, `output_exact_mismatches: 0`, `max_fp16_ulp_diff: 0`

## Numerical boundary found during verification

The full model-width RMSNorm+MLP path without residual is sensitive to the
RMSNorm reciprocal-square-root implementation difference between CPU and GPU:

- `--pre-mlp-rmsnorm` without `--residual` at hidden=1024 produces 186 exact
  output mismatches against the CPU reference, with `max_fp16_ulp_diff: 33`.
- The same run passes only with an explicit tolerance of 33 ULP.
- The layer-shaped path uses residual addition, and that relevant path is exact
  at model width.

This is not being promoted to a correctness claim for captured runtime RMSNorm.
The next stage still needs captured pre/post checkpoints from the real decode
pipeline.

## Remaining scope

- Not full captured RMSNorm validation (no captured post-attention residual + RMSNorm end-to-end fixture yet)
- Narrow smoke is hidden=128; the relevant model-width residual path is covered
  by a separate full gate
- Not all layers swept
- Not inference, not attention/DeltaNet, not LM head, not megakernel

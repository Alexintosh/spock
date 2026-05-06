# 0093: Captured RMSNorm Activation-Product Boundary

## Goal

Compare the persistent probe's Stage 2 activated MLP product against the
runtime's captured `mlp_product_fp16` for layer 0, step 1.

Diaries 0091 and 0092 narrowed the captured RMSNorm+MLP mismatch:

- final post-MLP residual output: 314 mismatches, max 87 ULP
- down-projection output before residual: 17 mismatches, max 2 ULP

This entry moves one stage earlier. The question is whether the SiLU-gated
activation product entering the down projection is already different, or whether
the down projection itself introduces the first runtime-vs-persistent
difference.

## Implementation

`vk_persistent_mlp_probe` now accepts:

```
--expected-mlp-product-fp16-file PATH
```

The file is a raw little-endian fp16 vector with at least `intermediate_count`
values. When provided, the app downloads `gate_scratch_buf` after dispatch.
After Stage 2, that buffer contains:

```
fp16(silu(gate_dot) * up_dot)
```

The comparison emits:

- `expected_mlp_product_fp16_file`
- `mlp_product_exact_mismatches`
- `mlp_product_within_tolerance`
- `mlp_product_mismatches`
- `mlp_product_max_fp16_ulp_diff`
- optional `first_mlp_product_mismatch_row`

If an expected MLP product file is provided, `mlp_product_mismatches` also
participates in final `status`. The same `--output-fp16-ulp-tolerance` flag is
used for this intermediate comparison so the current tolerance policy remains
single-sourced and explicit.

Default behavior is unchanged when the option is absent.

## Fixture

Added:

```
tests/data/layer0_step1_mlp_product_3584.fp16
```

Extraction command:

```
python3 tools/extract_component_fp16.py \
  --input /tmp/spock_components1_stderr.txt \
  --layer 0 \
  --field mlp_product_fp16 \
  --output /tmp/layer0_step1_mlp_product_3584.fp16
```

This is the same runtime component dump used for the layer-0
`mixer_residual`, `post_mlp`, and `down_output` fixtures.

## Direct command

Exact comparison:

```
build/vk_persistent_mlp_probe \
  --layer 0 \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --hidden 1024 \
  --intermediate 3584 \
  --output-rows 1024 \
  --workgroups 82 \
  --pre-mlp-rmsnorm \
  --input-fp16-file tests/data/layer0_step1_mixer_residual_1024.fp16 \
  --expected-mlp-product-fp16-file tests/data/layer0_step1_mlp_product_3584.fp16
```

Result:

```json
{
  "status": "fail",
  "generation": 3,
  "expected_generation": 3,
  "mlp_product_exact_mismatches": 10,
  "mlp_product_mismatches": 10,
  "mlp_product_max_fp16_ulp_diff": 2,
  "first_mlp_product_mismatch_row": 53,
  "output_mismatches": 0
}
```

With explicit tolerance:

```
--output-fp16-ulp-tolerance 2
```

Result:

```json
{
  "status": "ok",
  "mlp_product_exact_mismatches": 10,
  "mlp_product_within_tolerance": 10,
  "mlp_product_mismatches": 0,
  "mlp_product_max_fp16_ulp_diff": 2,
  "output_mismatches": 0,
  "output_fp16_ulp_tolerance": 2
}
```

## CTest gates

Two gates encode this boundary:

- `spock_persistent_mlp_probe_layer0_captured_rmsnorm_mlp_product_exact_fails`
  is marked `WILL_FAIL`.
- `spock_persistent_mlp_probe_layer0_captured_rmsnorm_mlp_product_ulp2` must
  pass.

## Interpretation

The activation product is already slightly different from the runtime capture,
but the difference is small:

- activation product: 10 rows, max 2 ULP
- down output: 17 rows, max 2 ULP
- post-MLP residual output: 314 rows, max 87 ULP

This strongly suggests the broad final-output spread is not caused by a major
layout, barrier, or weight-loading error in the persistent MLP probe. A small
number of fp16 activation differences enter before down projection; those
differences remain small through down projection; then residual addition changes
the fp16 ULP distance more dramatically for many final rows.

The remaining unexplained source is earlier than Stage 2. The next useful probe
is Stage 0/Stage 1 localization:

- compare persistent RMSNorm output against a runtime post-norm capture;
- if no runtime post-norm capture exists, add one to `--dump-step-components`;
- compare gate/up scratch before SiLU if needed.

That is the next step before claiming exact captured RMSNorm+MLP handoff or
building the layer-shaped persistent probe.

## Verification

- `cmake --build build -j`
- focused persistent MLP CTest set
- `git diff --check`
- `python3 tests/run_diary_check.py`
- direct JSON parse of exact and ULP-2 activation-product commands

## Remaining scope

- Not exact activation-product parity.
- Not RMSNorm output comparison.
- Not gate/up scratch comparison.
- Not token mixer integration.
- Not layer-shaped persistent decode.
- Not inference or megakernel completion.

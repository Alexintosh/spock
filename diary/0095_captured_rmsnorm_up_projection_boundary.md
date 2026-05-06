# 0095: Captured RMSNorm Up-Projection Boundary

## Goal

Compare the persistent probe's Stage 1 up-projection scratch against the
runtime's captured `mlp_up_fp16` for layer 0, step 1.

Diary 0094 proved that the persistent Stage 0 RMSNorm output matches the
runtime `mlp_normed_fp16` capture exactly. Diary 0093 showed the activation
product differs on 10 intermediate rows, max 2 ULP. This entry asks whether the
up projection already differs before SiLU/product computation.

## Runtime dump extension

`spock-decode --dump-step-components` now emits:

```
mlp_up_fp16
mlp_up_norm
```

The runtime captures `B.mlp_up` after `up_matvec(act_a) -> mlp_up_buf` and its
barrier. The value survives the later `silu_gate` dispatch, so no staging buffer
is needed; the diagnostic path downloads `B.mlp_up` after the layer command
completes, using the same style as existing `mlp_product_fp16` capture.

## Persistent probe extension

`vk_persistent_mlp_probe` now accepts:

```
--expected-up-scratch-fp16-file PATH
```

The file is a raw little-endian fp16 vector with at least `intermediate_count`
values. When present, the probe downloads `up_scratch_buf` after dispatch and
compares it with the same ULP accounting used by the other captured-boundary
checks:

- `up_scratch_exact_mismatches`
- `up_scratch_within_tolerance`
- `up_scratch_mismatches`
- `up_scratch_max_fp16_ulp_diff`
- optional `first_up_scratch_mismatch_row`

The comparison participates in final `status` only when the expected file is
provided. Default behavior is unchanged.

## Fixture

Added:

```
tests/data/layer0_step1_mlp_up_3584.fp16
```

The fixture came from a fresh step-1 component dump with prompt tokens
`151644 872 198`, the same prompt used by the existing layer-0 captured
fixtures:

```
build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --tokens /tmp/spock_step1_tokens.txt \
  --max-new-tokens 2 \
  --dump-step-components 1 \
  > /tmp/spock_components1_up_stdout.txt \
  2> /tmp/spock_components1_up_stderr.txt

python3 tools/extract_component_fp16.py \
  --input /tmp/spock_components1_up_stderr.txt \
  --layer 0 \
  --field mlp_up_fp16 \
  --output /tmp/layer0_step1_mlp_up_3584.fp16
```

The fresh dump's `mixer_residual_fp16` extraction was byte-compared against
`tests/data/layer0_step1_mixer_residual_1024.fp16` and matched exactly. The
generated token sequence remained `[410, 149852]`.

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
  --expected-up-scratch-fp16-file tests/data/layer0_step1_mlp_up_3584.fp16
```

Result:

```json
{
  "status": "fail",
  "generation": 3,
  "expected_generation": 3,
  "up_scratch_exact_mismatches": 5,
  "up_scratch_mismatches": 5,
  "up_scratch_max_fp16_ulp_diff": 2,
  "first_up_scratch_mismatch_row": 53,
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
  "up_scratch_exact_mismatches": 5,
  "up_scratch_within_tolerance": 5,
  "up_scratch_mismatches": 0,
  "up_scratch_max_fp16_ulp_diff": 2,
  "output_mismatches": 0,
  "output_fp16_ulp_tolerance": 2
}
```

## CTest gates

Two gates encode the boundary:

- `spock_persistent_mlp_probe_layer0_captured_rmsnorm_up_exact_fails` is marked
  `WILL_FAIL`.
- `spock_persistent_mlp_probe_layer0_captured_rmsnorm_up_ulp2` must pass.

## Interpretation

The up projection already differs from the runtime on 5 intermediate rows, max
2 ULP. Combined with the exact RMSNorm output from diary 0094, this localizes
the first observed difference to projection math after RMSNorm, not to
normalization.

The current layer-0 boundary stack is now:

- RMSNorm output: exact
- up projection: 5 rows, max 2 ULP
- activation product: 10 rows, max 2 ULP
- down output: 17 rows, max 2 ULP
- post-MLP residual output: 314 rows, max 87 ULP

This is coherent propagation: a small projection difference enters first, stays
small through activation and down projection, then becomes a wider fp16 ULP
spread after residual addition.

The remaining missing half of Stage 1 is the gate projection. The persistent
shader overwrites `gate_scratch` with the activated product in Stage 2, so an
exact gate-projection comparison will require either a preserved pre-activation
gate buffer or a stop/dump mode before Stage 2. That should be the next narrow
diagnostic if we want to fully explain the activation-product boundary.

## Verification

- `cmake --build build -j`
- focused persistent MLP CTest set
- `git diff --check`
- `python3 tests/run_diary_check.py`
- direct JSON parse of exact and ULP-2 up-scratch commands

## Remaining scope

- Not gate-projection scratch comparison.
- Not exact activation-product parity.
- Not exact down-output or post-MLP parity.
- Not token mixer integration.
- Not layer-shaped persistent decode.
- Not inference or megakernel completion.

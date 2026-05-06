# 0094: Captured RMSNorm Output Exact Gate

## Goal

Add a captured runtime comparison for the persistent probe's Stage 0 RMSNorm
output, and determine whether the remaining layer-0 RMSNorm+MLP mismatch starts
inside RMSNorm or later in gate/up/activation.

Diary 0093 showed that the activated MLP product differs from runtime
`mlp_product_fp16` on 10 intermediate rows, with a maximum fp16 ULP distance of
2. That left two plausible causes:

- the persistent RMSNorm output already differs slightly from the runtime
  post-norm vector;
- RMSNorm is exact, and the first difference appears in gate/up projection,
  SiLU, or activation-product rounding.

This entry resolves that question for layer 0, step 1.

## Runtime dump extension

`spock-decode --dump-step-components` now includes:

```
mlp_normed_fp16
mlp_normed_norm
```

This vector is captured immediately after:

```
post_norm(act_c) -> act_a
```

and before:

```
gate_matvec(act_a)
up_matvec(act_a)
```

The capture uses a host-visible staging buffer recorded into the same layer
command buffer. This is necessary because `act_a` is reused later in the layer:
first as MLP input, then as the post-MLP residual output. A post-submit download
of `B.act_a` would be too late.

The component dump path is diagnostic-only and already disables the single-submit
fast path. The staging copy therefore does not affect the normal runtime path or
performance claims.

## Persistent probe extension

`vk_persistent_mlp_probe` now accepts:

```
--expected-norm-output-fp16-file PATH
```

The file is a raw little-endian fp16 vector with at least `hidden` values. The
option requires `--pre-mlp-rmsnorm`, because `norm_output` is only written by
shader Stage 0 in that mode.

When provided, the app downloads `norm_output_buf` after dispatch and reports:

- `expected_norm_output_fp16_file`
- `norm_output_exact_mismatches`
- `norm_output_within_tolerance`
- `norm_output_mismatches`
- `norm_output_max_fp16_ulp_diff`
- optional `first_norm_output_mismatch_row`

`norm_output_mismatches` participates in final `status` when the file is
provided. Default behavior is unchanged when the option is absent.

## Fixture

Added:

```
tests/data/layer0_step1_mlp_normed_1024.fp16
```

The fixture came from a fresh component dump after adding `mlp_normed_fp16`:

```
printf '151644 872 198\n' > /tmp/spock_step1_tokens.txt
build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --tokens /tmp/spock_step1_tokens.txt \
  --max-new-tokens 2 \
  --dump-step-components 1 \
  > /tmp/spock_components1_normed_stdout.txt \
  2> /tmp/spock_components1_normed_stderr.txt

python3 tools/extract_component_fp16.py \
  --input /tmp/spock_components1_normed_stderr.txt \
  --layer 0 \
  --field mlp_normed_fp16 \
  --output /tmp/layer0_step1_mlp_normed_1024.fp16
```

The same fresh dump's `mixer_residual_fp16` extraction was byte-compared against
the existing checked-in `layer0_step1_mixer_residual_1024.fp16` fixture and
matched exactly. The generated token sequence also matched the previous step-1
dump: `[410, 149852]`.

## Direct command

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
  --expected-norm-output-fp16-file tests/data/layer0_step1_mlp_normed_1024.fp16
```

Result:

```json
{
  "status": "ok",
  "generation": 3,
  "expected_generation": 3,
  "norm_output_exact_mismatches": 0,
  "norm_output_within_tolerance": 0,
  "norm_output_mismatches": 0,
  "norm_output_max_fp16_ulp_diff": 0,
  "output_mismatches": 0
}
```

## CTest gates

Two gates were added:

- `spock_persistent_mlp_probe_layer0_captured_rmsnorm_normed_exact`
- `spock_persistent_mlp_probe_expected_norm_requires_rmsnorm`

The first proves exact captured RMSNorm output equality for layer 0, step 1.
The second is a negative gate proving the expected-norm file option is rejected
unless `--pre-mlp-rmsnorm` is enabled.

## Interpretation

The persistent Stage 0 RMSNorm output is bit-exact against the runtime capture.
That is a stronger result than the activation and down-output comparisons:

- RMSNorm output: exact
- activation product: 10 rows differ, max 2 ULP
- down output: 17 rows differ, max 2 ULP
- post-MLP residual output: 314 rows differ, max 87 ULP

The first observed mismatch is therefore not RMSNorm. It appears after RMSNorm,
most likely in gate/up matvec reduction order or SiLU activation rounding. This
is useful because RMSNorm can now be treated as a validated component in the
layer-0 captured handoff path.

The next narrow diagnostic is to expose and compare gate and up projection
scratch values independently. If gate/up are exact, the mismatch starts in
SiLU/product rounding. If gate/up differ by 1-2 ULP, the activation-product
boundary is explained by matvec accumulation order.

## Verification

- `cmake --build build -j`
- focused persistent MLP CTest set
- `git diff --check`
- `python3 tests/run_diary_check.py`
- direct JSON parse of the exact RMSNorm-output command

## Remaining scope

- Not gate/up scratch comparison.
- Not exact activation-product parity.
- Not exact down-output or post-MLP parity.
- Not token mixer integration.
- Not layer-shaped persistent decode.
- Not inference or megakernel completion.

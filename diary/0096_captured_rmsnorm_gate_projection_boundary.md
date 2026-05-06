# 0096: Captured RMSNorm Gate-Projection Boundary

## Goal

Compare the persistent probe's raw Stage 1 gate projection against the
runtime's captured `mlp_gate_fp16` for layer 0, step 1.

Diary 0094 proved that RMSNorm output is exact. Diary 0095 showed the up
projection differs from the runtime on 5 intermediate rows, max 2 ULP. This
entry closes the other half of Stage 1: the gate projection that feeds SiLU.

## Runtime dump extension

`spock-decode --dump-step-components` now emits:

```
mlp_gate_fp16
mlp_gate_norm
```

The runtime captures `B.mlp_gate` after `gate_matvec(act_a) -> mlp_gate_buf`.
`B.mlp_gate` is an input to `silu_gate`; it is not overwritten by that dispatch,
so the diagnostic path can download it after the layer command completes. This
matches the existing `mlp_up_fp16` and `mlp_product_fp16` dump style.

## Persistent probe extension

`vk_persistent_mlp_probe` now accepts:

```
--expected-gate-scratch-fp16-file PATH
```

The persistent shader still runs the full Stage 0/1/2/3 pipeline. To avoid
changing the structural path under test, Stage 1 writes the raw gate dot to both
the existing `gate_scratch` buffer and a new preserved diagnostic buffer:

```
binding 10: raw_gate_scratch
```

Stage 2 continues to overwrite `gate_scratch` with `silu(gate) * up`, so the
existing activation-product comparison remains unchanged. The new comparison
downloads `raw_gate_scratch` after dispatch and reports:

- `gate_scratch_exact_mismatches`
- `gate_scratch_within_tolerance`
- `gate_scratch_mismatches`
- `gate_scratch_max_fp16_ulp_diff`
- optional `first_gate_scratch_mismatch_row`

The comparison participates in final `status` only when the expected file is
provided. Default behavior is unchanged.

## Fixture

Added:

```
tests/data/layer0_step1_mlp_gate_3584.fp16
```

The fixture came from a fresh step-1 component dump with prompt tokens
`151644 872 198`:

```
build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --tokens /tmp/spock_step1_tokens.txt \
  --max-new-tokens 2 \
  --dump-step-components 1 \
  > /tmp/spock_components1_gate_stdout.txt \
  2> /tmp/spock_components1_gate_stderr.txt

python3 tools/extract_component_fp16.py \
  --input /tmp/spock_components1_gate_stderr.txt \
  --layer 0 \
  --field mlp_gate_fp16 \
  --output /tmp/layer0_step1_mlp_gate_3584.fp16
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
  --expected-gate-scratch-fp16-file tests/data/layer0_step1_mlp_gate_3584.fp16
```

Result:

```json
{
  "status": "fail",
  "generation": 3,
  "expected_generation": 3,
  "gate_scratch_exact_mismatches": 7,
  "gate_scratch_mismatches": 7,
  "gate_scratch_max_fp16_ulp_diff": 1,
  "first_gate_scratch_mismatch_row": 278,
  "output_mismatches": 0
}
```

With explicit tolerance:

```
--output-fp16-ulp-tolerance 1
```

Result:

```json
{
  "status": "ok",
  "gate_scratch_exact_mismatches": 7,
  "gate_scratch_within_tolerance": 7,
  "gate_scratch_mismatches": 0,
  "gate_scratch_max_fp16_ulp_diff": 1,
  "output_mismatches": 0,
  "output_fp16_ulp_tolerance": 1
}
```

## CTest gates

Two gates encode the boundary:

- `spock_persistent_mlp_probe_layer0_captured_rmsnorm_gate_exact_fails` is
  marked `WILL_FAIL`.
- `spock_persistent_mlp_probe_layer0_captured_rmsnorm_gate_ulp1` must pass.

## Interpretation

The raw gate projection is tightly bounded but not exact: 7 intermediate rows
differ by exactly 1 fp16 ULP. Combined with diary 0095, the Stage 1 picture is:

- gate projection: 7 rows, max 1 ULP
- up projection: 5 rows, max 2 ULP

Since RMSNorm output is exact, these differences enter at the projection dot
products. The activation product then has 10 rows, max 2 ULP, which is
consistent with both projection inputs being nearly identical but not
bit-exact.

The current layer-0 boundary stack is now:

- RMSNorm output: exact
- gate projection: 7 rows, max 1 ULP
- up projection: 5 rows, max 2 ULP
- activation product: 10 rows, max 2 ULP
- down output: 17 rows, max 2 ULP
- post-MLP residual output: 314 rows, max 87 ULP

This completes the narrow MLP-internal boundary map for layer 0. The next useful
step is not another scratch split inside this micro-probe; it is a layer-shaped
persistent probe that composes RMSNorm, captured mixer handoff, MLP, and
residual update with the same checkpoint discipline.

## Verification

- `cmake --build build -j`
- direct exact raw-gate JSON command
- direct ULP-1 raw-gate JSON command
- fresh dump token sequence check
- fresh `mixer_residual_fp16` byte-compare against existing fixture

The focused CTest suite still needs to be run after adding the CTest entries.

## Remaining scope

- Not exact activation-product parity.
- Not exact down-output or post-MLP parity.
- Not token mixer integration.
- Not layer-shaped persistent decode.
- Not inference or megakernel completion.

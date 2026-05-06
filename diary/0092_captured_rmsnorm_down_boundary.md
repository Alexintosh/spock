# 0092: Captured RMSNorm+Down-Projection Boundary

## Goal

Narrow the captured RMSNorm+MLP mismatch from diary 0091 by comparing the
persistent probe's down-projection output before residual addition against the
runtime's captured `down_output_fp16`.

Diary 0091 showed that persistent RMSNorm+MLP+residual differs from runtime
`post_mlp_fp16` on layer 0, step 1:

- 314 exact fp16 output mismatches
- max fp16 ULP diff 87
- all structural barrier checks pass

That result did not identify whether the mismatch entered during RMSNorm,
gate/up projection, activation, down projection, or residual addition.

## Fixture

Added:

```
tests/data/layer0_step1_down_output_1024.fp16
```

Extraction command:

```
python3 tools/extract_component_fp16.py \
  --input /tmp/spock_components1_stderr.txt \
  --layer 0 \
  --field down_output_fp16 \
  --output /tmp/layer0_step1_down_output_1024.fp16
```

This uses the same component dump as the existing layer-0 `mixer_residual` and
`post_mlp` fixtures.

## Direct command

Run the persistent probe without `--residual` so its output is the down
projection result:

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
  --expected-output-fp16-file tests/data/layer0_step1_down_output_1024.fp16
```

Exact result:

```json
{
  "status": "fail",
  "generation": 3,
  "expected_generation": 3,
  "checksum": 455177825,
  "expected_checksum": 455177825,
  "output_exact_mismatches": 17,
  "output_mismatches": 17,
  "max_fp16_ulp_diff": 2,
  "output_fp16_ulp_tolerance": 0,
  "first_mismatch_row": 31
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
  "output_exact_mismatches": 17,
  "output_within_tolerance": 17,
  "output_mismatches": 0,
  "max_fp16_ulp_diff": 2,
  "output_fp16_ulp_tolerance": 2
}
```

## CTest gates

Two gates encode the boundary:

- `spock_persistent_mlp_probe_layer0_captured_rmsnorm_down_exact_fails` is
  marked `WILL_FAIL`.
- `spock_persistent_mlp_probe_layer0_captured_rmsnorm_down_ulp2` must pass.

## Interpretation

The down-projection output is much closer to the runtime capture than the final
post-MLP residual output:

- down output: 17 mismatches, max 2 ULP
- post-MLP residual output: 314 mismatches, max 87 ULP

This suggests the main amplification happens at or after residual addition, not
through widespread MLP compute divergence. A small down-output difference can
produce a larger fp16 ULP distance after adding the residual stream, especially
near cancellation or small-magnitude values.

This distinction matters for the megakernel path. If the persistent probe had
shown hundreds of down-output mismatches with a large ULP spread, the next work
would need to revisit RMSNorm, gate/up projection, SiLU, down projection layout,
or barrier ordering before composing more of the layer. Instead, the
down-projection comparison says the persistent path is already very close at the
last MLP compute boundary. The remaining post-MLP mismatch is therefore a more
specific residual-stream question: how the runtime and persistent path round the
down output, combine it with the captured mixer residual, and round the final
fp16 hidden state.

The current probe also keeps the internal CPU checksum exact for the same
command:

```
"checksum": 455177825,
"expected_checksum": 455177825
```

That means the persistent shader and its host reference agree for the standalone
probe's own arithmetic contract. The mismatch is not between the persistent
shader and its verifier. It is between the persistent target-path arithmetic and
the existing runtime's ordinary dispatch arithmetic. This is exactly why the
external expected-output gate is useful: it separates "the persistent probe is
self-consistent" from "the persistent probe matches the current runtime at a
captured layer boundary."

The 17 differing down-output rows should be treated as diagnostic handles. The
next pass should inspect whether those rows correspond to a small set of
activation-scratch differences, down-matvec accumulation-order differences, or
RMSNorm output differences. The runtime component dump already contains
`mlp_product_fp16`, so the next narrow gate can compare the persistent Stage 2
activated scratch against that fixture before adding any new layer-shaped
shader. If the activation product is exact, the difference enters in the
down-projection matvec. If it is also bounded, the project can decide whether to
align reduction order or carry an explicit target-path tolerance while the
megakernel is still being assembled.

This entry therefore advances the project by reducing the search space. It does
not make the final path correct, and it does not justify broad tolerances. The
point is to keep failures small enough that the next probe can answer a concrete
question.

The next smallest useful diagnostic is to compare the persistent activation
scratch against runtime `mlp_product_fp16`. If activation is exact or tightly
bounded, then the remaining mismatch is isolated to down-projection rounding
and residual-add rounding.

## Verification

- `cmake --build build -j`
- focused persistent MLP CTest set: 26/26 passed
- `git diff --check`
- `python3 tests/run_diary_check.py`
- direct JSON parse of exact and ULP-2 down-output commands

## Remaining scope

- Not exact runtime-vs-persistent down-output parity.
- Not activation scratch comparison.
- Not RMSNorm stage comparison.
- Not token mixer integration.
- Not layer-shaped persistent decode.
- Not inference or megakernel completion.

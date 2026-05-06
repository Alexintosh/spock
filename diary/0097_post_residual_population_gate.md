# 0097: Post-Residual Population Gate

## Goal

Strengthen the captured RMSNorm+MLP residual gate so it checks both the worst
case and the shape of the error distribution.

The previous layer-0 post-residual gate used `--output-fp16-ulp-tolerance 87`.
That bounded the maximum observed fp16 ULP difference, but a max-only gate can
miss a regression where many more rows drift while staying under the same tail
bound. Before composing more persistent layer work, the probe should know
whether the residual error is concentrated or broad.

## Probe extension

`vk_persistent_mlp_probe` now always emits output ULP population buckets:

- `output_ulp_le_1`
- `output_ulp_le_2`
- `output_ulp_le_4`
- `output_ulp_le_8`
- `output_ulp_le_16`
- `output_ulp_le_32`
- `output_ulp_le_64`
- `output_ulp_gt_64`

It also accepts an opt-in population gate:

```
--output-fp16-population-ulp-threshold N
--output-fp16-max-rows-above-population-threshold N
```

The first option reports how many output rows have ULP difference above the
threshold. The second option makes that count part of final `status`. The max
rows option requires the threshold option so the gate is explicit.

This is intentionally in the existing probe, not a new shader or executable.
No GPU path changes are needed; this is stricter accounting around an already
validated captured boundary.

## Layer-0 post-residual distribution

Command:

```
build/vk_persistent_mlp_probe \
  --layer 0 \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --hidden 1024 \
  --intermediate 3584 \
  --output-rows 1024 \
  --workgroups 82 \
  --pre-mlp-rmsnorm \
  --residual \
  --input-fp16-file tests/data/layer0_step1_mixer_residual_1024.fp16 \
  --expected-output-fp16-file tests/data/layer0_step1_post_mlp_1024.fp16 \
  --output-fp16-ulp-tolerance 87 \
  --output-fp16-population-ulp-threshold 16
```

Observed distribution:

```json
{
  "status": "ok",
  "output_exact_mismatches": 314,
  "output_within_tolerance": 314,
  "output_mismatches": 0,
  "max_fp16_ulp_diff": 87,
  "output_ulp_le_1": 955,
  "output_ulp_le_2": 982,
  "output_ulp_le_4": 1005,
  "output_ulp_le_8": 1013,
  "output_ulp_le_16": 1014,
  "output_ulp_le_32": 1019,
  "output_ulp_le_64": 1023,
  "output_ulp_gt_64": 1,
  "output_rows_above_population_ulp_threshold": 10
}
```

Only 10 of 1024 rows are above 16 ULP, and only 1 row is above 64 ULP. That
supports the current interpretation: the post-residual max-87 case is a narrow
tail, not a broad distribution shift.

## CTest gates

Two tests encode the population contract:

- `spock_persistent_mlp_probe_layer0_captured_rmsnorm_mlp_population_ulp16`
  requires at most 10 rows above 16 ULP and must pass.
- `spock_persistent_mlp_probe_layer0_captured_rmsnorm_mlp_population_ulp16_max9_fails`
  is marked `WILL_FAIL` to prove the population gate is active.

## Interpretation

This does not make the MLP residual output exact. It makes the current
non-exactness more actionable. The layer-0 MLP slice now has:

- exact RMSNorm output
- bounded raw projection differences
- bounded activation and down-output differences
- a bounded post-residual max error
- a bounded post-residual population tail

That is the right quality bar before composing a larger persistent layer probe.
The next implementation step should build on this gate, not weaken it by only
checking generated tokens or a max-only tolerance.

The practical reason is that the eventual megakernel will remove many host-side
inspection points. Once multiple stages are fused behind persistent barriers,
debugging a token mismatch becomes much harder unless each smaller boundary has
already recorded both its maximum error and its distribution shape. A max-only
gate tells us the largest local damage. A population gate tells us whether the
damage is sparse enough to treat as rounding-tail behavior or broad enough to
suspect a structural ordering, descriptor, or arithmetic mismatch. That
distinction matters before the project starts carrying this slice across
multiple layers.

## Verification

- `cmake --build build -j`
- focused persistent MLP CTest set, 36/36 passed
- direct JSON parse of the population-threshold command

## Remaining scope

- Not exact post-residual parity.
- Not representative-layer population sweep.
- Not token mixer integration.
- Not layer-shaped persistent decode.
- Not inference or megakernel completion.

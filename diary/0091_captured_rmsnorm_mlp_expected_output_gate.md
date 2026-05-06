# 0091: Captured RMSNorm+MLP Expected-Output Gate

## Goal

Add an external fp16 expected-output comparison path to
`vk_persistent_mlp_probe` so captured runtime outputs can become the probe's
authoritative comparison target.

Diary 0090 proved that the persistent probe can run RMSNorm before MLP and that
the model-width RMSNorm+residual path matches its internal CPU reference. That
still did not compare against the real runtime's `post_mlp_fp16` capture. This
entry adds the missing handoff gate.

## Implementation

`vk_persistent_mlp_probe` now accepts:

```
--expected-output-fp16-file PATH
```

The file is a raw little-endian fp16 vector. It must contain at least
`output_rows` values. When provided, the probe uses that vector for:

- `output_exact_mismatches`
- `output_within_tolerance`
- `output_mismatches`
- `max_fp16_ulp_diff`
- `first_mismatch_row`
- final `status`

The existing internal CPU reference is still computed. Its aggregate
`expected_checksum` remains a diagnostic field. The external file only changes
the output-vector comparison target.

The default path is unchanged when `--expected-output-fp16-file` is absent.

## Captured fixture

Added:

```
tests/data/layer0_step1_post_mlp_1024.fp16
```

This fixture was extracted from the existing step-1 component dump:

```
python3 tools/extract_component_fp16.py \
  --input /tmp/spock_components1_stderr.txt \
  --layer 0 \
  --field post_mlp_fp16 \
  --output /tmp/layer0_step1_post_mlp_1024.fp16
```

The same dump's `mixer_residual_fp16` extraction was compared against the
checked-in `tests/data/layer0_step1_mixer_residual_1024.fp16`, and the bytes
matched exactly. That confirms the new `post_mlp` fixture is paired with the
existing layer-0 input fixture.

## Direct command

Exact comparison against the captured runtime output:

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
  --expected-output-fp16-file tests/data/layer0_step1_post_mlp_1024.fp16
```

Result:

```json
{
  "status": "fail",
  "generation": 3,
  "expected_generation": 3,
  "checksum": 3274229726,
  "expected_checksum": 3274229726,
  "output_exact_mismatches": 314,
  "output_mismatches": 314,
  "max_fp16_ulp_diff": 87,
  "output_fp16_ulp_tolerance": 0,
  "first_mismatch_row": 0
}
```

With explicit tolerance:

```
--output-fp16-ulp-tolerance 87
```

Result:

```json
{
  "status": "ok",
  "output_exact_mismatches": 314,
  "output_within_tolerance": 314,
  "output_mismatches": 0,
  "max_fp16_ulp_diff": 87,
  "output_fp16_ulp_tolerance": 87
}
```

## CTest gates

Two gates encode the current state:

- `spock_persistent_mlp_probe_layer0_captured_rmsnorm_mlp_exact_fails` is
  marked `WILL_FAIL`. It proves the default exact GPU-vs-GPU captured runtime
  comparison is not yet exact.
- `spock_persistent_mlp_probe_layer0_captured_rmsnorm_mlp_ulp87` passes with an
  explicit 87-ULP tolerance. This bounds the current difference and prevents the
  captured handoff from drifting silently beyond the measured boundary.

Two additional negative gates cover the new file input path:

- `spock_persistent_mlp_probe_expected_output_missing`
- `spock_persistent_mlp_probe_expected_output_too_small`

This is deliberately not documented as a completed exact RMSNorm+MLP captured
handoff. It is a measured boundary.

## Interpretation

The exact failure is not a barrier failure: `failures == 0`, `arrived == 0`,
and `generation == expected_generation == 3`.

The persistent probe still matches its own internal CPU reference checksum for
this command. The mismatch is against the runtime's captured GPU output. The
likely source is that the standalone persistent probe and the existing runtime
RMSNorm/MLP path do not use identical reduction structure and math-library
choices. The runtime RMSNorm shader uses a single 256-invocation workgroup and
`inversesqrt`; the persistent probe uses the persistent 64-lane staged shape.
The MLP stages also differ from the runtime's ordinary dispatch path.

The next correctness step is to narrow where the 314 row differences enter:

- compare captured runtime post-norm input against persistent Stage 0
  `norm_output`;
- compare gate/up scratch against captured or instrumented runtime equivalents;
- only then decide whether the target should align the persistent reduction
  order to runtime, accept a bounded policy, or replace the runtime comparison
  target with a target-path GPU reference.

## Verification

- `cmake --build build -j`
- focused persistent MLP CTest set: 24/24 passed
- `git diff --check`
- `python3 tests/run_diary_check.py`
- direct JSON parse of the exact and ULP-87 commands

## Remaining scope

- Not exact captured RMSNorm+MLP parity.
- Not token mixer integration.
- Not a layer-shaped persistent probe.
- Not 24-layer persistent decode.
- Not final norm, LM head, token selection, or inference.
- Not the Vulkan-native megakernel.

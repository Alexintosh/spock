# 0103: DeltaNet Z-Projection Matvec Gate

## Goal

Validate another stateless upstream DeltaNet boundary using the generic
`vk_matvec_probe` from diary 0101:

```
dn_input_norm_fp16 + layer.0.delta_in_proj_z -> dn_z_fp16
```

Diary 0102 proved that captured `dn_z_fp16` participates exactly in the
norm-gate equation:

```
dn_core_fp16 + dn_z_fp16 + layer.0.delta_norm -> dn_gated_fp16
```

This entry walks backward on the z-gate side only. It does not attempt to prove
q/k/v projection yet, because the currently dumped `dn_q_fp16` and `dn_k_fp16`
fields are post-L2-normalized values rather than raw `delta_in_proj_qkv`
outputs.

## Implementation

Added a fixture:

```
tests/data/layer0_step1_dn_input_norm_1024.fp16
```

It was extracted from `/tmp/spock_components1_mixer_stderr.txt` field
`dn_input_norm_fp16` with `tools/extract_component_fp16.py`.

No new app was needed. The existing `vk_matvec_probe` can already load an fp16
rank-2 weight by role and compare fp16 output against a captured fixture. The
new CTest gate is:

```
spock_matvec_probe_layer0_delta_z_proj_exact
```

It runs:

```
vk_matvec_probe \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --weight-role layer.0.delta_in_proj_z \
  --in-dim 1024 \
  --out-dim 2048 \
  --input-fp16-file tests/data/layer0_step1_dn_input_norm_1024.fp16 \
  --expected-output-fp16-file tests/data/layer0_step1_dn_z_2048.fp16
```

## Why z is the right next stateless gate

The z projection is independent of recurrent state. It consumes the normalized
DeltaNet input vector and produces the gate vector used later by the norm-gate
stage. That makes it a clean matvec boundary: one input vector, one weight
matrix, one output vector, and no hidden state mutation.

This matters for the larger megakernel path because the downstream chain now
has an exact gate at every stateless handoff after `dn_z_fp16` appears:

```
dn_input_norm -> dn_z
dn_core + dn_z + delta_norm -> dn_gated
dn_gated + delta_out_proj -> dn_out
input_hidden + dn_out -> mixer_residual
```

Only `dn_core` still depends on the recurrent state update. By proving z
separately, the eventual recurrent-core probe will not have to debug z
projection, norm-gate, output projection, and residual addition at the same
time. A mismatch in `dn_core` can stay local to q/k/v, g/beta, and state
handling.

The z gate is also useful because it exercises the same generic matvec probe
against a second DeltaNet projection shape: `[2048, 1024]` instead of diary
0101's `[1024, 2048]` output projection. That broadens confidence in the
weight-loader and shader handoff without introducing a new abstraction.

## Result

The direct command passes exact fp16 equality:

```json
{
  "in_dim": 1024,
  "out_dim": 2048,
  "weight_role": "layer.0.delta_in_proj_z",
  "status": "ok",
  "output_exact_mismatches": 0,
  "output_mismatches": 0,
  "max_fp16_ulp_diff": 0,
  "output_fp16_ulp_tolerance": 0
}
```

All 2048 z-gate projection outputs match bit-for-bit.

## Rejected qkv gate

I also tested the tempting qkv projection comparison:

```
dn_input_norm_fp16 + layer.0.delta_in_proj_qkv -> dn_q_fp16/dn_k_fp16/dn_v_fp16
```

That is not a valid gate with current fixtures. The direct comparison against a
temporary concatenation of dumped `dn_q_fp16`, `dn_k_fp16`, and `dn_v_fp16`
failed immediately and broadly. The reason is architectural, not necessarily a
projection bug: the dumped q/k fields are captured after L2 normalization, while
the merged qkv projection emits raw q/k/v before the split and normalization
steps. A correct qkv projection gate needs either raw qkv projection captures or
a composed probe that includes split plus q/k L2 normalization.

This is exactly why these small gates are useful. They prevent the project from
turning an invalid expected fixture into a false regression.

The rejected check is not wasted. It records a real fixture contract problem:
the dump field names `dn_q_fp16` and `dn_k_fp16` are semantically downstream of
the raw qkv projection. Future work should either add explicit raw fields, such
as `dn_q_raw_fp16` and `dn_k_raw_fp16`, or build a probe that intentionally
matches the runtime sequence:

```
delta_in_proj_qkv -> split q/k/v -> q L2 norm -> k L2 norm
```

That composed gate would be valid because its expected output would match the
meaning of the existing dumped q/k/v fields. Until then, promoting qkv matvec
comparison would lower code quality by hiding a bad test contract behind a
large mismatch count.

## Verification

- direct z-projection `vk_matvec_probe` command passes exact
- focused CTest coverage includes the z projection, norm-gate, DeltaNet output
  projection, residual-add, component extraction, and diary checks
- `python3 tests/run_diary_check.py`
- `git diff --check`

## Remaining scope

- Not raw qkv projection parity.
- Not q/k L2-normalization parity.
- Not DeltaNet recurrent core parity.
- Not layer-shaped persistent decode.
- Not inference or megakernel completion.

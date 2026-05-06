# 0105: DeltaNet A/B Projection Gates

## Goal

Add raw A/B projection captures and prove the two small DeltaNet scalar
projection gates:

```
dn_input_norm_fp16 + layer.0.delta_in_proj_a -> dn_a_fp16
dn_input_norm_fp16 + layer.0.delta_in_proj_b -> dn_b_fp16
```

These projections feed g/beta computation. They are stateless matvecs, so they
should be isolated before building a g/beta or recurrent-core probe.

## Runtime dump extension

`spock-decode --dump-step-components` now emits:

```
dn_a_fp16
dn_b_fp16
```

Both fields contain 16 fp16 values for layer 0, step 1, one per DeltaNet head.
The runtime downloads `B.dn_a` and `B.dn_b` after the projection command buffer
has completed and before g/beta computation consumes those buffers. Unlike qkv,
these buffers are not overwritten by conv or L2 normalization, so no staging
copy is required.

## Fixtures

Added:

```
tests/data/layer0_step1_dn_a_16.fp16
tests/data/layer0_step1_dn_b_16.fp16
```

They were extracted from a fresh deterministic component dump generated after
the runtime fields were added:

```
build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --tokens /tmp/spock_step1_tokens.txt \
  --max-new-tokens 2 \
  --dump-step-components 1 \
  > /tmp/spock_components1_ab_stdout.txt \
  2> /tmp/spock_components1_ab_stderr.txt
```

## Why A/B are separate gates

The A and B projections are tiny compared with qkv and z, but they control a
stateful part of DeltaNet. The runtime uses A together with `delta_a_log` and
`delta_dt_bias` to compute `g`, and it uses B to compute `beta`. Those scalars
then control decay and update strength inside the recurrent state rule. A small
projection error here can therefore change every value in the recurrent output,
even though the projection itself has only 16 fp16 outputs.

That is why this entry keeps A/B as plain matvec gates instead of jumping
straight to g/beta. If a later g/beta probe fails, the A/B inputs are already
known to be exact, and the investigation can focus on softplus, sigmoid,
fp32/fp16 conversion, `delta_a_log`, `delta_dt_bias`, or descriptor binding.

The runtime dump fields also make the diagnostic vocabulary clearer. Before
this entry, the dump exposed `dn_g` and `dn_beta` but not their projected
inputs. That forced any g/beta investigation to infer whether the problem was
in projection or scalar nonlinear math. With `dn_a_fp16` and `dn_b_fp16`, those
two contracts are now separable.

This is another example of the project's "walk backward" strategy. The
downstream norm-gate and output projection are already closed. Now the stateless
projection fanout from `dn_input_norm_fp16` is closed too. The remaining hard
work is stateful: conv/L2 mutation, g/beta scalar math, and recurrent state.

## CTest gates

Added:

```
spock_matvec_probe_layer0_delta_a_proj_exact
spock_matvec_probe_layer0_delta_b_proj_exact
```

The direct commands use `vk_matvec_probe` with:

- input: `tests/data/layer0_step1_dn_input_norm_1024.fp16`
- A weight: `layer.0.delta_in_proj_a`, shape `[16, 1024]`
- B weight: `layer.0.delta_in_proj_b`, shape `[16, 1024]`
- outputs: the new 16-value raw A/B fixtures

## Result

Both direct commands pass exact fp16 equality:

```json
{
  "out_dim": 16,
  "status": "ok",
  "output_exact_mismatches": 0,
  "output_mismatches": 0,
  "max_fp16_ulp_diff": 0,
  "output_fp16_ulp_tolerance": 0
}
```

## Interpretation

The stateless projection fanout from `dn_input_norm_fp16` is now covered:

```
dn_input_norm -> dn_qkv_raw
dn_input_norm -> dn_z
dn_input_norm -> dn_a
dn_input_norm -> dn_b
```

This closes the raw projection side of the layer-0 DeltaNet block. The next
small gate should use `dn_a_fp16`, `dn_b_fp16`, `layer.0.delta_a_log`, and
`layer.0.delta_dt_bias` to validate g/beta computation against the existing
`dn_g` and `dn_beta` dump fields.

This is still not recurrent-core parity. The recurrent output depends on q/k/v
after conv/L2, g/beta, and the recurrent state tensor. But the projection inputs
to those later stages are no longer ambiguous.

## Verification

- `cmake --build build -j`
- fresh `spock-decode --dump-step-components 1` after adding `dn_a_fp16` and
  `dn_b_fp16`
- direct A and B `vk_matvec_probe` commands pass exact
- focused CTest coverage includes A/B, qkv, z, output projection, norm-gate,
  residual-add, component extraction, and diary checks
- `python3 tests/run_diary_check.py`
- `git diff --check`

## Remaining scope

- Not g/beta computation parity.
- Not conv1d or q/k L2-normalization parity.
- Not recurrent core parity.
- Not layer-shaped persistent decode.
- Not inference or megakernel completion.

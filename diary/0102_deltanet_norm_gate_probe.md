# 0102: DeltaNet Norm-Gate Probe

## Goal

Validate the DeltaNet norm+gate boundary that feeds the already proven output
projection:

```
dn_core_fp16 + dn_z_fp16 + layer.0.delta_norm -> dn_gated_fp16
```

Diary 0101 proved:

```
dn_gated_fp16 + layer.0.delta_out_proj -> dn_out_fp16
```

This entry walks one stage backward. It proves that the captured recurrent core
and z gate produce the captured gated vector through the same Vulkan
`deltanet_norm_gate.comp` shader used by the runtime.

## Probe design

Added:

```
vk_deltanet_norm_gate_probe
```

The probe uses `shaders/deltanet_norm_gate.comp` unchanged. The descriptor and
push-constant contract mirrors runtime decode:

- binding 0: fp16 core vector as in-place IO/output
- binding 1: fp16 z gate vector
- binding 2: fp32 `delta_norm` weight
- push constants: `num_heads`, `head_dim`, `epsilon_bits`, `output_offset`
- dispatch: one workgroup per head

The command-line interface is:

```
--repack-dir DIR
--weight-role ROLE
--core-fp16-file PATH
--gate-fp16-file PATH
--expected-output-fp16-file PATH
--num-heads N
--head-dim N
--output-fp16-ulp-tolerance N
```

Default comparison is exact fp16 equality. The JSON output reports the same
fields as the matvec and residual probes:

- `status`
- `output_exact_mismatches`
- `output_within_tolerance`
- `output_mismatches`
- `max_fp16_ulp_diff`
- `output_fp16_ulp_tolerance`
- optional `first_mismatch_row`

## Inference context

For DeltaNet layers, `dn_core_fp16` is the per-head value-space result after
the recurrent update has consumed the current q/k/v vectors and state. That
core is not yet the value that the output projection consumes. The model first
applies a per-head RMSNorm with `layer.N.delta_norm`, rounds the weighted
normalized value back through fp16, and then multiplies by the SiLU-transformed
z gate. The result is `dn_gated_fp16`, a 2048-value vector arranged as
16 heads x 128 value dimensions.

This stage is worth isolating because it combines the three numerical risks
that often hide inside a larger fused token mixer: per-head reductions,
fp32-to-fp16 rounding at an architecture-specific point, and nonlinear gate
math. A full layer-shaped persistent probe should not be the first place those
risks are observed. With this gate, any future mismatch in `dn_gated_fp16` can
be separated from both the downstream output projection and the upstream
recurrent state update.

The exact gate is intentionally GPU-vs-GPU. The expected vector was captured
from the runtime Vulkan decode path, and the probe runs the same Vulkan shader
with the same fp16 inputs and fp32 norm weight. Unlike a CPU replay, there is no
different math library or alternate reduction order to justify tolerance here.
If exact equality failed, the correct response would be to inspect descriptor
binding, offsets, push constants, and fixture provenance before accepting any
ULP window.

## Fixtures

Added:

```
tests/data/layer0_step1_dn_core_2048.fp16
tests/data/layer0_step1_dn_z_2048.fp16
```

The expected output reuses the diary 0101 fixture:

```
tests/data/layer0_step1_dn_gated_2048.fp16
```

All three were byte-checked against fresh extractions from
`/tmp/spock_components1_mixer_stderr.txt` using `tools/extract_component_fp16.py`.

## Direct command

```
build/vk_deltanet_norm_gate_probe \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --weight-role layer.0.delta_norm \
  --num-heads 16 \
  --head-dim 128 \
  --core-fp16-file tests/data/layer0_step1_dn_core_2048.fp16 \
  --gate-fp16-file tests/data/layer0_step1_dn_z_2048.fp16 \
  --expected-output-fp16-file tests/data/layer0_step1_dn_gated_2048.fp16
```

Result:

```json
{
  "num_heads": 16,
  "head_dim": 128,
  "weight_role": "layer.0.delta_norm",
  "status": "ok",
  "output_exact_mismatches": 0,
  "output_mismatches": 0,
  "max_fp16_ulp_diff": 0,
  "output_fp16_ulp_tolerance": 0
}
```

All 2048 output values match bit-for-bit.

## CTest gates

Added:

- `spock_deltanet_norm_gate_probe_help`
- `spock_deltanet_norm_gate_probe_layer0_exact`

The layer-0 gate is exact because this is a GPU-vs-runtime-GPU comparison using
captured fp16 inputs and the same Vulkan shader. No CPU math tolerance is
justified.

## Interpretation

The downstream DeltaNet boundary is now two stages deep:

```
dn_core + dn_z + delta_norm -> dn_gated
dn_gated + delta_out_proj -> dn_out
input_hidden + dn_out -> mixer_residual
```

That closes the norm-gate, output projection, and residual handoff equations
for layer 0, step 1. The next DeltaNet work should continue walking backward
into the recurrent core producer rather than jumping to a fused layer-shaped
kernel.

## Verification

- `cmake --build build -j`
- direct `vk_deltanet_norm_gate_probe` command passes exact
- fresh `dn_core_fp16`, `dn_z_fp16`, and `dn_gated_fp16` extractions byte-match
  the checked-in fixtures
- focused CTest coverage passes for norm-gate, matvec, residual-add,
  component extraction, and diary checks

## Remaining scope

- Not DeltaNet recurrent core parity.
- Not q/k/v/z projection parity.
- Not attention token-mixer parity.
- Not layer-shaped persistent decode.
- Not inference or megakernel completion.

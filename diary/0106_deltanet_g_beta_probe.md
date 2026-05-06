# 0106: DeltaNet G/Beta Probe

## Goal

Validate the scalar DeltaNet g/beta computation after the A/B projection gates:

```
dn_a_fp16 + dn_b_fp16 + delta_a_log + delta_dt_bias -> dn_g, dn_beta
```

Diary 0105 proved that `dn_a_fp16` and `dn_b_fp16` are exact projection outputs
from `dn_input_norm_fp16`. This entry proves the next scalar nonlinear stage
with exact fp32 bit comparison against the runtime Vulkan dump.

## Runtime dump extension

`spock-decode --dump-step-components` already emitted decimal `dn_g` and
`dn_beta` values. Decimal JSON is useful for humans, but it is not a safe exact
fixture for fp32 shader output. This entry adds exact bit fields:

```
dn_g_bits
dn_beta_bits
```

Each field contains 16 uint32 values, one per DeltaNet head. The values are the
bit pattern of the fp32 g/beta values downloaded from the runtime `dn_state`
tail after `deltanet_compute_g_beta.comp` runs. The decimal fields remain in
place for readability.

## Probe design

Added:

```
vk_deltanet_g_beta_probe
```

The probe runs `shaders/deltanet_compute_g_beta.comp` unchanged. Its descriptor
contract mirrors the runtime shader:

- binding 0: `dn_a_fp16`
- binding 1: `dn_b_fp16`
- binding 2: packed fp32 `[a_log, dt_bias]`
- binding 3: fp32 output buffer `[g heads][beta heads]`
- push constants: `num_heads`, `layer_index`

The app loads `layer.0.delta_a_log` as fp32 and `layer.0.delta_dt_bias` as fp16
from the repacked artifact, converts dt-bias to fp32 with the same half decoder
used by the runtime, packs the buffer as the shader expects, dispatches one
workgroup per head, and compares the downloaded fp32 bit patterns against a raw
uint32 fixture.

The current probe supports `--layer-index 0`; it is intentionally scoped to the
layer-0 captured fixture.

## Fixture

Added:

```
tests/data/layer0_step1_dn_g_beta_32.u32
```

It contains 32 little-endian uint32 values:

- first 16: `dn_g_bits`
- next 16: `dn_beta_bits`

The fixture came from a fresh deterministic component dump generated after the
runtime bit fields were added:

```
build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --tokens /tmp/spock_step1_tokens.txt \
  --max-new-tokens 2 \
  --dump-step-components 1 \
  > /tmp/spock_components1_gbeta_stdout.txt \
  2> /tmp/spock_components1_gbeta_stderr.txt
```

## Direct command

```
build/vk_deltanet_g_beta_probe \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --num-heads 16 \
  --layer-index 0 \
  --a-fp16-file tests/data/layer0_step1_dn_a_16.fp16 \
  --b-fp16-file tests/data/layer0_step1_dn_b_16.fp16 \
  --expected-g-beta-bits-file tests/data/layer0_step1_dn_g_beta_32.u32
```

Result:

```json
{
  "num_heads": 16,
  "layer_index": 0,
  "status": "ok",
  "output_exact_mismatches": 0
}
```

All 32 fp32 outputs match bit-for-bit.

## Interpretation

The scalar DeltaNet branch is now closed through g/beta:

```
dn_input_norm -> dn_a
dn_input_norm -> dn_b
dn_a + dn_b + a_log + dt_bias -> g/beta
```

This removes another ambiguity before recurrent-core work. A future recurrent
probe can assume the projected A/B inputs and scalar g/beta outputs are exact
for layer 0, step 1. Any recurrent-core mismatch should then be investigated in
q/k/v preparation, state input, state decay/update, q scaling, or output
accumulation.

The bitwise comparison is strict by design. Because the expected fixture is a
runtime GPU capture and the probe runs the same shader, there is no CPU math
library difference to excuse a mismatch. If this gate fails later, descriptor
bindings, weight packing, layer index, or shader behavior changed.

## Verification

- `cmake --build build -j`
- fresh `spock-decode --dump-step-components 1` after adding `dn_g_bits` and
  `dn_beta_bits`
- direct `vk_deltanet_g_beta_probe` command passes exact
- focused CTest coverage includes g/beta, A/B, qkv, z, output projection,
  norm-gate, residual-add, component extraction, and diary checks
- `python3 tests/run_diary_check.py`
- `git diff --check`

## Remaining scope

- Not conv1d or q/k L2-normalization parity.
- Not recurrent state input capture.
- Not recurrent core parity.
- Not layer-shaped persistent decode.
- Not inference or megakernel completion.

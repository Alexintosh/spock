# 0118: Persistent Layer-0 G/Beta Gate

## Goal

Validate that `vk_persistent_layer0_probe` can compute the DeltaNet g/beta
scalar branch inside the persistent layer-0 shader:

```text
dn_a_fp16 + dn_b_fp16 + layer.0.delta_a_log + layer.0.delta_dt_bias
  -> g/beta fp32 bit patterns
```

This is the next narrow persistent DeltaNet gate after projection-prefix
(diary 0116) and conv/L2 (diary 0117). It is not recurrent computation, not
the full mixer, not full layer composition, not inference, and not the
megakernel.

## Why G/Beta Next

The recurrent core consumes q/k/v from conv/L2 and g/beta from the scalar
branch. Conv/L2 is now closed in the persistent execution shape, so g/beta is
the remaining non-recurrent producer needed before adding recurrent state
update and core output to the persistent layer probe.

The existing non-persistent `vk_deltanet_g_beta_probe` already proved this
branch exactly for layer 0, step 1. This entry proves the same fp32 bit-pattern
contract from inside `persistent_layer0_probe.comp` without adding recurrent
state yet.

## Implementation

### Shader: `shaders/persistent_layer0_probe.comp`

- Added `mode=3` alongside tail (`mode=0`), projection-prefix (`mode=1`), and
  conv/L2 (`mode=2`).
- Mode 3 uses no software global barriers. `expected_generation == 0`.
- Inputs:
  - binding 1: `dn_a_fp16` (`16` fp16 values)
  - binding 2: `dn_b_fp16` (`16` fp16 values)
  - binding 0 extended control payload: `delta_a_log[16]`,
    `delta_dt_bias[16]`, and `g_beta_bits[32]`
- Formula mirrors `deltanet_compute_g_beta.comp`:
  - `g[h] = -exp(a_log[h]) * log(1 + exp(a[h] + dt_bias[h]))`
  - `beta[h] = 1 / (1 + exp(-b[h]))`
- Output order matches the existing exact-bit fixture contract:
  - `g_beta_bits[0..15] = g`
  - `g_beta_bits[16..31] = beta`

The control-buffer payload is a standalone probe layout, not the final
recurrent-state layout. It keeps the existing ten-binding persistent probe
structure intact while validating exact g/beta math and bit ordering.

### Host App: `apps/vk_persistent_layer0_probe.cpp`

- Added `--mode g-beta`.
- Added required inputs:
  - `--a-fp16-file`
  - `--b-fp16-file`
  - `--expected-g-beta-bits-file`
- Repack loading mirrors `apps/vk_deltanet_g_beta_probe.cpp`:
  - `layer.0.delta_a_log` must be fp32
  - `layer.0.delta_dt_bias` must be fp16 and is converted to fp32 on load
  - values are packed per head into the control payload
- JSON output reports:
  - `probe = persistent_layer0_g_beta`
  - `mode = g-beta`
  - structural counters: `failures`, `arrived`, `generation`,
    `expected_generation`
  - `g_beta_bit_mismatches`
  - first mismatch diagnostics if any mismatch occurs

### CMake

Added exact CTest gate:

- `spock_persistent_layer0_probe_g_beta_exact`

Exact bit comparison passes, so no bounded fallback gate is needed.

## Verification

### Direct Command

```sh
./build/vk_persistent_layer0_probe --mode g-beta \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --a-fp16-file tests/data/layer0_step1_dn_a_16.fp16 \
  --b-fp16-file tests/data/layer0_step1_dn_b_16.fp16 \
  --expected-g-beta-bits-file tests/data/layer0_step1_dn_g_beta_32.u32
```

Result:

- `status = ok`
- `failures = 0`
- `arrived = 0`
- `generation = 0`
- `expected_generation = 0`
- `g_beta_bit_mismatches = 0`

### Regression Coverage

The targeted regression set passes with the new exact gate included:

- `spock_persistent_layer0_probe_help`
- `spock_persistent_layer0_probe_post_mlp_exact_fails`
- `spock_persistent_layer0_probe_post_mlp_bounded`
- `spock_persistent_layer0_probe_projection_prefix_exact_fails`
- `spock_persistent_layer0_probe_projection_prefix_ulp1`
- `spock_persistent_layer0_probe_conv_l2_exact`
- `spock_persistent_layer0_probe_g_beta_exact`
- `spock_deltanet_g_beta_probe_layer0_exact`
- `spock_diary_check`

## Known Limitations

- This is a scalar-branch gate only. It does not add recurrent state update,
  norm-gate, output projection, residual add, or full mixer composition.
- The g/beta values are emitted through the standalone probe control payload,
  not through the final DeltaNet recurrent-state layout.
- Single layer (0) and one captured step only. Not a representative-layer sweep.
- Not full layer persistence, not inference, not the megakernel.

## Next Work

1. Add persistent recurrent state/core output using persistent q/k/v and g/beta
   inputs.
2. Add persistent norm-gate, output projection, and first residual add.
3. Compose the existing persistent post-mixer tail after the full mixer path.
4. Compare full layer-0 persistent output against captured `post_mlp`.

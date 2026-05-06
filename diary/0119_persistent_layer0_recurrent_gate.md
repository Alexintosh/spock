# 0119: Persistent Layer-0 Recurrent Core Gate

## Goal

Validate that `vk_persistent_layer0_probe` can run the DeltaNet recurrent core
inside `persistent_layer0_probe.comp`:

```text
dn_q_fp16 + dn_k_fp16 + dn_v_fp16 + g/beta bits + recurrent_state_pre_f32
  -> dn_core_fp16
```

This is the next narrow persistent DeltaNet gate after projection-prefix
(diary 0116), conv/L2 (diary 0117), and g/beta (diary 0118). It is not the
full mixer, not full layer composition, not inference, and not the megakernel.

## Why This Gate Matters

The recurrent core is the first persistent DeltaNet gate that consumes both
the vector branch and the scalar branch:

- q/k/v come from the conv/L2 gate;
- g/beta come from the scalar g/beta gate;
- the fp32 recurrent state carries the time-dependent DeltaNet memory.

If this step is wrong, downstream norm-gate, output projection, and residual
add gates can all look wrong while the real bug lives in state decay, state
update, q scaling, or fp16 output rounding. Closing this gate inside the
persistent shader gives the future full layer probe a clean recurrent contract.

## Implementation

### Shader: `shaders/persistent_layer0_probe.comp`

- Added `mode=4` for the recurrent core.
- Mode 4 uses one 128-lane workgroup per DeltaNet head.
- It uses no software global barrier; only workgroup-local `barrier()` calls
  are required. `expected_generation == 0`.
- Inputs:
  - binding 0: persistent control payload, `g_beta_bits[32]`, and fp32
    recurrent state stored as raw uint32 bits in `control.extra[]`
  - binding 1: `dn_q_fp16[2048]`
  - binding 2: `dn_k_fp16[2048]`
  - binding 3: `dn_v_fp16[2048]`
  - binding 6: `dn_core_fp16[2048]` output
- The math mirrors `deltanet_recurrent.comp`:
  - `q_scaled = fp16(q) * inversesqrt(128)`
  - `k` and `v` load as fp16-to-fp32
  - `state *= exp(g)`
  - `kv_mem = state^T @ k`
  - `delta = (v - kv_mem) * beta`
  - `state += k outer delta`
  - `output = state^T @ q_scaled`
  - final output is stored as fp16

### Host App: `apps/vk_persistent_layer0_probe.cpp`

- Added `--mode recurrent`.
- Added required inputs:
  - `--q-fp16-file`
  - `--k-fp16-file`
  - `--v-fp16-file`
  - `--g-beta-bits-file`
  - `--state-pre-f32-file`
  - `--expected-output-fp16-file`
- The host path dispatches `persistent_layer0_probe.comp.spv`, not the
  standalone recurrent shader. This is important: the CTest name must mean that
  the persistent shader branch is the code under test.
- The host packs the first `16 * 128 * 128` fp32 recurrent-state values into
  the persistent control payload as raw uint32 bits, and stores the captured
  g/beta fp32 bit patterns in the control payload's `g_beta_bits` section.
- JSON output reports:
  - `probe = persistent_layer0_recurrent`
  - `mode = recurrent`
  - structural counters: `failures`, `arrived`, `generation`,
    `expected_generation`
  - `output_exact_mismatches`
  - `output_max_fp16_ulp`

### CMake

Added exact CTest gate:

- `spock_persistent_layer0_probe_recurrent_exact`

Exact comparison passes, so no bounded fallback gate is needed.

## Verification

### Direct Command

```sh
./build/vk_persistent_layer0_probe --mode recurrent \
  --q-fp16-file tests/data/layer0_step1_dn_q_2048.fp16 \
  --k-fp16-file tests/data/layer0_step1_dn_k_2048.fp16 \
  --v-fp16-file tests/data/layer0_step1_dn_v_2048.fp16 \
  --g-beta-bits-file tests/data/layer0_step1_dn_g_beta_32.u32 \
  --state-pre-f32-file tests/data/layer0_step1_dn_recurrent_state_pre_262176.f32 \
  --expected-output-fp16-file tests/data/layer0_step1_dn_core_2048.fp16
```

Result:

- `status = ok`
- `failures = 0`
- `arrived = 0`
- `generation = 0`
- `expected_generation = 0`
- `output_exact_mismatches = 0`
- `output_max_fp16_ulp = 0`

### Regression Coverage

The targeted regression set includes the new exact recurrent gate alongside
the earlier persistent gates:

- `spock_persistent_layer0_probe_help`
- `spock_persistent_layer0_probe_post_mlp_exact_fails`
- `spock_persistent_layer0_probe_post_mlp_bounded`
- `spock_persistent_layer0_probe_projection_prefix_exact_fails`
- `spock_persistent_layer0_probe_projection_prefix_ulp1`
- `spock_persistent_layer0_probe_conv_l2_exact`
- `spock_persistent_layer0_probe_g_beta_exact`
- `spock_persistent_layer0_probe_recurrent_exact`
- `spock_deltanet_recurrent_probe_layer0_exact`
- `spock_diary_check`

## Known Limitations

- This mode currently consumes captured q/k/v and captured g/beta rather than
  composing projection-prefix, conv/L2, g/beta, and recurrent in one persistent
  pass.
- The recurrent state is carried in a probe-specific control payload. The final
  megakernel will need a cleaner state-buffer layout for all layers and decode
  steps.
- Single layer (0) and one captured step only. Not a representative-layer sweep.
- Not full mixer composition, not full layer persistence, not inference, not
  the megakernel.

## Next Work

1. Add persistent norm-gate, output projection, and first residual add.
2. Compose persistent projection-prefix, conv/L2, g/beta, recurrent, norm-gate,
   output projection, and residual add into one layer-0 mixer gate.
3. Feed the composed mixer output into the existing persistent post-mixer tail.
4. Compare full layer-0 persistent output against captured `post_mlp`.

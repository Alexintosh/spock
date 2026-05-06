# 0112: DeltaNet Recurrent Core Probe

## Goal

Prove the layer-0 DeltaNet recurrent core (`deltanet_recurrent.comp`) produces
exact fp16-bit-identical output to the captured runtime handoff tensor when
given the same q, k, v, g/beta, and pre-recurrent state inputs. This closes
gate 7 from the DeltaNet backward-validation ladder:

```
q/k/v (2048 each) + g/beta (32 u32) + state_pre (262176 f32) --> [deltanet_recurrent] --> dn_core (2048 fp16)
```

## Implementation

New probe app: `apps/vk_deltanet_recurrent_probe.cpp`.

The probe loads six fixtures:

- `layer0_step1_dn_q_2048.fp16` — L2-normalized query vectors (16 heads × 128 dim)
- `layer0_step1_dn_k_2048.fp16` — L2-normalized key vectors (16 heads × 128 dim)
- `layer0_step1_dn_v_2048.fp16` — value vectors (16 heads × 128 dim)
- `layer0_step1_dn_g_beta_32.u32` — g/beta fp32 bit patterns (16 g + 16 beta)
- `layer0_step1_dn_recurrent_state_pre_262176.f32` — pre-recurrent state matrix + g/beta tail
- `layer0_step1_dn_core_2048.fp16` — expected recurrent core output

The shader implements the gated delta rule for batch-1 decode:

1. Decay state by `exp(g)`
2. Compute `kv_mem = state^T @ k`
3. Compute `delta = (v - kv_mem) * beta`
4. Update state: `state += k ⊗ delta`
5. Produce output: `output = state^T @ q`

### Buffer layout

The shader expects three storage buffer bindings:

- Binding 0: Q (readonly fp16), `num_heads * k_dim` values
- Binding 1: KV+out (fp16), K section (`num_heads * k_dim`) then V section (`num_heads * v_dim`).
  Output overwrites the V section.
- Binding 2: State (fp32), matrix (`num_heads * k_dim * v_dim`) then g tail (`num_heads`) then
  beta tail (`num_heads`)

Push constants: `num_heads`, `k_dim`, `v_dim`, `state_total`, `q_scale_bits`.

The shader scales Q internally by `1/sqrt(k_dim)` via the `q_scale_bits` push constant.

### State tail handling

The pre-state fixture (1,048,704 bytes = 262,176 fp32 values) already includes
the g/beta tail from the runtime capture. The probe unconditionally overwrites
the tail from `--g-beta-bits-file` so the fixture contract is explicit: the
probe's correctness depends on the independently-gated g/beta bits, not on an
implicit assumption that the pre-state fixture's tail matches.

### Dispatch

One workgroup per head (16 workgroups × 128 invocations). The full KV buffer
(K + V) is uploaded, the shader dispatches, and the V/output section is
compared against the expected `dn_core_fp16` fixture as raw fp16 bits with
zero tolerance.

CLI required args: `--q-fp16-file`, `--k-fp16-file`, `--v-fp16-file`,
`--g-beta-bits-file`, `--state-pre-f32-file`, `--expected-output-fp16-file`.
Optional numeric args: `--num-heads` (16), `--k-dim` (128), `--v-dim` (128).

JSON output includes `status`, `num_heads`, `k_dim`, `v_dim`, `output_count`,
`output_mismatches`, `max_fp16_ulp_diff`, and `first_mismatch_*` diagnostics
when any mismatch is found.

## CTest gates

- `spock_deltanet_recurrent_probe_help` — verifies --help exits cleanly
- `spock_deltanet_recurrent_probe_layer0_exact` — runs the full recurrent core
  with layer-0 fixtures and asserts exact fp16 match

## Verification

- `cmake --build build -j` compiles cleanly
- `build/vk_deltanet_recurrent_probe --help` prints usage and exits 0
- Direct run with layer-0 fixtures: `status: "ok"`, `output_mismatches: 0`,
  `max_fp16_ulp_diff: 0`
- `ctest --test-dir build -R spock_deltanet_recurrent_probe --output-on-failure`
  passes both help and exact gate
- All existing DeltaNet probes continue to pass:
  conv/L2, g/beta, norm-gate, matvec, residual-add, diary check
- `python3 tests/run_diary_check.py` passes
- `git diff --check` clean

## What this proves

If the exact gate passes, the `deltanet_recurrent.comp` shader produces
bit-identical output to the unfused runtime path for layer 0, step 1. Combined
with the already-closed gates (output projection, norm-gate, z projection, raw
qkv, A/B projections, g/beta, conv/L2, residual add), the full DeltaNet
token-mixer path from `dn_input_norm_fp16` through `mixer_residual_fp16` is
now covered by independent exact gates for layer 0.

This eliminates the recurrent core as a possible cause for any downstream
mismatch in the unfused decode path. The DeltaNet backward-validation ladder
is complete for layer 0.

## What this does not prove

- Not other-layer recurrent correctness.
- Not the fused `deltanet_recurrent_gbeta.comp` or
  `deltanet_recurrent_gbeta_norm_gate.comp` shaders.
- Not layer-shaped persistent execution.
- Not multi-layer decode.
- Not inference or megakernel completion.

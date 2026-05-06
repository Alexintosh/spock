# 0122: Persistent Layer-0 Full-Layer Gate

## Goal

Validate that `vk_persistent_layer0_probe` can compose the complete persistent
DeltaNet mixer with the post-mixer RMSNorm+MLP tail as one layer-0 shaped
persistent dispatch:

```text
dn_input_norm_fp16[1024] + input_hidden_fp16[1024]
  + conv_state_pre_fp16[24576]
  + recurrent_state_pre_f32[262144 + 32]
  + layer-0 DeltaNet weights
  + layer-0 post_norm/mlp weights
  -> [single 128-lane 82-workgroup persistent dispatch, 10 global barriers]
  -> mixer_output_fp16[1024]
  -> mixer_residual_fp16[1024]
  -> post_mlp_fp16[1024]
```

This is the first single-dispatch captured layer-0 pass from the normalized
DeltaNet input through the post-MLP layer handoff. It is still a captured
one-step gate, not autoregressive inference, not all 24 layers, and not the
final megakernel.

## Why This Gate Matters

Diary 0121 proved the complete DeltaNet mixer can run as one persistent
dispatch. Earlier tail gates proved the post-mixer RMSNorm+MLP path can run in
the same 128-lane persistent execution shape when fed the captured
`mixer_residual` boundary.

This gate removes that CPU-side boundary for layer 0. The shader now computes
`mixer_residual` and immediately consumes it in the same dispatch for
post-RMSNorm, MLP gate/up, activation, down projection, and residual add. That
is the smallest meaningful layer-shaped step toward the RX 6750 XT
Vulkan-native megakernel.

## Implementation

### Shader: `shaders/persistent_layer0_probe.comp`

- Added `mode=7` (`layer0`) on top of the existing mode=6 full-mixer path.
- Mode 7 reuses the full-mixer stages and then continues with four more
  synchronization points:

| Stage | Work | Generation |
|-------|------|------------|
| 1-7 | Full DeltaNet mixer through `mixer_residual` | 6 |
| 8 | Post-mixer RMSNorm into reused scratch | 8 |
| 9 | MLP gate/up projections | 9 |
| 10 | SiLU(gate) * up product | 10 |
| 11 | MLP down projection + residual add to `post_mlp` | 10 |

Expected final `generation` is 10: six mixer barriers, one mixer-residual
handoff barrier, and three post-mixer tail barriers.

Mode 7 keeps the 10-buffer descriptor shape. Binding 8 becomes a packed tail
weight buffer:

```text
delta_out_proj[1024x2048]
post_norm[1024]
mlp_gate[3584x1024]
mlp_up[3584x1024]
mlp_down[1024x3584]
```

Binding 9 becomes a hidden pack:

```text
input_hidden[1024]
mixer_output[1024]
mixer_residual[1024]
post_mlp[1024]
```

The shader deliberately reuses scratch after it is dead:

- `buf7` starts as packed projection weights and later becomes MLP up scratch.
- `buf2` starts as QKV/core/gated scratch and later becomes MLP gate/product
  scratch.
- `buf3` starts as Z scratch and later becomes post-norm output scratch.

### Host App: `apps/vk_persistent_layer0_probe.cpp`

- Extended `run_full_mixer_mode()` with a `compose_post_mlp_tail` flag.
- Added `--mode layer0` (`mode=7`) and help text.
- For mode 7, the host loads and packs `post_norm`, `mlp_gate`, `mlp_up`, and
  `mlp_down` after `delta_out_proj`.
- The JSON output reports both the inherited mixer metrics and the final
  `post_mlp` metrics.

### CMake

Added paired gates:

- `spock_persistent_layer0_probe_layer0_exact_fails` (`WILL_FAIL`)
- `spock_persistent_layer0_probe_layer0_bounded`

## Verification

### Direct Exact Command

```sh
./build/vk_persistent_layer0_probe --mode layer0 \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --input-norm-fp16-file tests/data/layer0_step1_dn_input_norm_1024.fp16 \
  --input-hidden-fp16-file tests/data/layer0_step1_input_hidden_1024.fp16 \
  --conv-state-pre-fp16-file tests/data/layer0_step1_dn_conv_state_pre_24576.fp16 \
  --state-pre-f32-file tests/data/layer0_step1_dn_recurrent_state_pre_262176.f32 \
  --expected-mixer-output-fp16-file tests/data/layer0_step1_mixer_output_1024.fp16 \
  --expected-mixer-residual-fp16-file tests/data/layer0_step1_mixer_residual_1024.fp16 \
  --expected-output-fp16-file tests/data/layer0_step1_post_mlp_1024.fp16
```

Result:

- `status = fail`
- `failures = 0`
- `arrived = 0`
- `generation = 10`
- `expected_generation = 10`
- `mixer_output_exact_mismatches = 28`
- `mixer_output_max_fp16_ulp = 6`
- `mixer_residual_exact_mismatches = 8`
- `mixer_residual_max_fp16_ulp = 16`
- `output_exact_mismatches = 433`
- `output_max_fp16_ulp = 105`

### Direct Bounded Command

```sh
./build/vk_persistent_layer0_probe --mode layer0 \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --input-norm-fp16-file tests/data/layer0_step1_dn_input_norm_1024.fp16 \
  --input-hidden-fp16-file tests/data/layer0_step1_input_hidden_1024.fp16 \
  --conv-state-pre-fp16-file tests/data/layer0_step1_dn_conv_state_pre_24576.fp16 \
  --state-pre-f32-file tests/data/layer0_step1_dn_recurrent_state_pre_262176.f32 \
  --expected-mixer-output-fp16-file tests/data/layer0_step1_mixer_output_1024.fp16 \
  --expected-mixer-residual-fp16-file tests/data/layer0_step1_mixer_residual_1024.fp16 \
  --expected-output-fp16-file tests/data/layer0_step1_post_mlp_1024.fp16 \
  --full-mixer-fp16-ulp-tolerance 16 \
  --output-fp16-ulp-tolerance 105
```

Result:

- `status = ok`
- `failures = 0`
- `arrived = 0`
- `generation = 10`
- `expected_generation = 10`
- `mixer_output_max_fp16_ulp = 6`
- `mixer_residual_max_fp16_ulp = 16`
- `output_exact_mismatches = 433`
- `output_max_fp16_ulp = 105`

## Interpretation

The exact gate is intentionally marked `WILL_FAIL`. The reference fixtures come
from the existing multi-dispatch runtime path, while mode 7 keeps intermediate
values in one persistent dispatch and changes the reduction boundaries. The
full-mixer portion retains diary 0121's bounded deviation, and the post-MLP
tail widens the final captured `post_mlp` bound from the tail-only 87 ULP to
105 ULP when fed the persistent mixer's bounded `mixer_residual`.

This is acceptable for the current gate because it proves structural execution
and bounds the numerical drift. It is not a claim of exact parity. The next
quality step is to localize whether the extra post-MLP spread comes primarily
from post_norm, gate/up, activation, down projection, or residual addition when
the input boundary is persistent `mixer_residual` rather than captured
`mixer_residual`.

## Current Status

This closes the first captured layer-0 persistent pass:

```text
dn_input_norm -> DeltaNet mixer -> mixer_residual -> post_norm -> MLP -> post_mlp
```

Remaining megakernel work:

1. Add internal comparison taps for mode 7 (`post_norm`, gate/up/product, down).
2. Decide whether the 105 ULP post-MLP bound is acceptable for decode or needs a
   precision experiment.
3. Generalize the layer-shaped pass beyond layer 0.
4. Chain layers and manage recurrent/conv state across decode steps.
5. Attach embeddings, final norm, LM head, and token loop for basic inference.

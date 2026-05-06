# 0108: DeltaNet Conv/L2 Fixture Boundary

## Goal

Extend the `--dump-step-components` runtime to capture `dn_conv_state_pre_fp16`,
the conv1d rolling state slice *before* `conv1d_step` mutates it. Extract fixture
boundaries for this new field alongside the existing `dn_q_fp16`, `dn_k_fp16`,
`dn_v_fp16` captures so a future conv/L2 probe can consume `dn_qkv_raw_fp16` +
`dn_conv_state_pre_fp16` and target the post-L2 `dn_q_fp16`, `dn_k_fp16`, and
`dn_v_fp16` fields.

This is a fixture boundary entry, not conv/L2 probe parity. It adds no shader,
no CTest gate, and no inference result.

## Runtime dump extension

`spock-decode --dump-step-components` now emits a new field per deltanet layer:

```
dn_conv_state_pre_fp16
```

The field is captured by staging a copy of the conv rolling state for the current
layer inside the same command buffer, immediately before `conv1d_step` is
dispatched. The copy is downloaded after the command buffer completes, so later
mutations to the conv state buffer do not affect the captured value.
The pre-conv state copy uses a shader-write → transfer-read buffer
barrier (`VkBufferMemoryBarrier`) before `vkCmdCopyBuffer` to ensure the
conv1d shader's writes are visible to the transfer copy.

The field has 24576 fp16 values per layer:

- DN_CONV_DIM * DN_CONV_KS = 6144 * 4 = 24576
- DN_CONV_DIM = DN_KEY_TOTAL * 2 + DN_VAL_TOTAL = 2048 * 2 + 2048 = 6144
- DN_CONV_KS = 4 (conv kernel size)

## Existing fields extracted alongside

The decode run also produces the existing post-conv, post-L2 fields that the
future conv/L2 probe needs to target:

- `dn_q_fp16`: 2048 values (post-conv, post-L2 normalized query)
- `dn_k_fp16`: 2048 values (post-conv, post-L2 normalized key)
- `dn_v_fp16`: 2048 values (post-conv value, no L2 normalization)

## Fixtures

Added:

```
tests/data/layer0_step1_dn_conv_state_pre_24576.fp16  (49152 bytes)
tests/data/layer0_step1_dn_q_2048.fp16               (4096 bytes)
tests/data/layer0_step1_dn_k_2048.fp16               (4096 bytes)
tests/data/layer0_step1_dn_v_2048.fp16               (4096 bytes)
```

The `dn_q`, `dn_k`, `dn_v` fixtures already existed in earlier dump runs but were
not previously extracted as standalone `.fp16` files. They are now materialized
so the conv/L2 probe can use file-based comparison without re-running the full
decode.

## Why this boundary matters

Diary 0104 proved the raw qkv projection exactly. The next unknown in the
DeltaNet decode path is the conv1d step and q/k L2 normalization. The pipeline is:

```
dn_qkv_raw_fp16 (6144) --[conv1d_step]--> post-conv qkv buffer
                                        --[q/k L2 norm]--> dn_q_fp16, dn_k_fp16
                                        --[v pass-through]--> dn_v_fp16
```

The conv1d step mutates the qkv buffer in place using a per-layer rolling
conv state. Capturing the conv state *before* mutation gives the probe both
inputs (raw qkv + conv state) and all three outputs (post-L2 q, k, v).

This isolates conv mutation and L2 normalization from the rest of the
DeltaNet pipeline. A probe failure here means the conv1d or L2 normalization
shader is wrong, not the projection, recurrent core, or output projection.

The conv state itself is a per-layer rolling buffer of shape [DN_CONV_DIM * DN_CONV_KS]. On each decode step, the conv1d kernel shifts this buffer left by one slot, inserts the new qkv vector at the final slot (ks-1), and computes a weighted sum across the kernel dimension using learned conv weights. The pre-capture therefore records the exact history input that the conv1d shader will consume, which is critical for reproducing the convolution output deterministically. Without this boundary, a conv/L2 probe would need to reconstruct the rolling state from scratch, which defeats the purpose of an isolated component test.

## Verification

- `cmake --build build -j` compiles cleanly with the new dump code
- Fresh `spock-decode --dump-step-components 1` produces all four fields:
  - `dn_conv_state_pre_fp16`: 24576 values
  - `dn_q_fp16`: 2048 values
  - `dn_k_fp16`: 2048 values
  - `dn_v_fp16`: 2048 values
- Decode result unchanged: `generated_tokens` was `[410, 149852]`
- `python3 tests/run_diary_check.py`
- `git diff --check`

## Remaining scope

- Not conv1d qkv mutation parity.
- Not q/k L2-normalization parity.
- Not recurrent core parity.
- Not layer-shaped persistent decode.
- Not inference or megakernel completion.
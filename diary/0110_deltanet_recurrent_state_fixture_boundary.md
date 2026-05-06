# 0110: DeltaNet Recurrent State Pre-Update Fixture Boundary

## Goal

Capture the raw fp32 recurrent state (matrix + g/beta tail) *before* the
deltanet_recurrent shader mutates it at decode step 1, layer 0. This provides the
exact GPU-side state that the recurrent update consumes, enabling a future
recurrent-core probe to target post-recurrent output against a known pre-state.

This is only the recurrent-state fixture boundary. It is not recurrent parity.
No shader, no CTest gate, and no inference result change.

## Sidecar capture mechanism

Two new CLI options control the capture:

- `--dump-dn-recurrent-state-pre-layer N`: select the deltanet layer (0-23)
- `--dump-dn-recurrent-state-pre-file PATH`: output path for the raw binary sidecar

Both require `--dump-step-components N` to be active. Validation is enforced
in `spock-decode` before any GPU work begins: if either option is given without
the other, or without `--dump-step-components`, the tool exits with an error.
The validation block is formatted consistently with the surrounding 2-space
indentation of other option checks in `main()`.

The capture itself happens inside the deltanet decode loop in `vk_session.cpp`,
between g/beta computation and the recurrent dispatch. After the g/beta shader
writes the state tail and a compute→transfer barrier is inserted, the sidecar
code:

1. Allocates a HOST_VISIBLE|HOST_COHERENT staging buffer of `dn_state_per_layer` bytes
2. Records `vkCmdCopyBuffer` from the device-local state buffer to staging
3. Submits and waits (synchronous)
4. Reads raw bytes directly from the mapped staging pointer via `memcpy`
5. Writes the bytes to disk

The `memcpy`-from-mapped approach avoids an unnecessary
`download_from_device` round-trip. Since the staging buffer is HOST_VISIBLE and
HOST_COHERENT, the data is available immediately after `submit_and_wait` — no
additional staging-copy or invalidation needed.

## Byte layout

Per-layer `dn_state_per_layer` bytes:

- Matrix: `DN_HEADS * DN_K_DIM * DN_V_DIM * 4` = 16 × 128 × 128 × 4 = 1,048,576 bytes
- G/beta tail: `DN_HEADS * 2 * 4` = 16 × 2 × 4 = 128 bytes
- Total: 1,048,704 bytes = 262,176 fp32 values

The matrix layout is `[head][k_dim][v_dim]` in fp32. The tail stores all g
values first, then all beta values:

```
tail[0..15]  = g[0..15]
tail[16..31] = beta[0..15]
```

No conversion is applied. The raw GPU bytes are written as-is.

## Fixture

Added:

```
tests/data/layer0_step1_dn_recurrent_state_pre_262176.f32  (1048704 bytes)
```

Generated with:
```
build/spock-decode --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --tokens /tmp/spock_step1_tokens.txt --max-new-tokens 2 \
  --dump-step-components 1 --dump-dn-recurrent-state-pre-layer 0 \
  --dump-dn-recurrent-state-pre-file tests/data/layer0_step1_dn_recurrent_state_pre_262176.f32
```

Input tokens: `[151644, 872, 198]` (BOS, special token, newline).
Generated tokens: `[410, 149852]` (unchanged from prior runs).

## Why this boundary matters

Diary 0109 isolated the conv1d + L2 normalization sub-block. The next unknown
after conv/L2 is the recurrent core: the `deltanet_recurrent` shader. For each
head it decays the state with `exp(g)`, computes `kv_mem = state^T @ k`, applies
`delta = (v - kv_mem) * beta`, updates the state with `state += k outer delta`,
and produces the output vector with `output = state^T @ q`.

To test this shader in isolation, a probe needs:
1. The pre-recurrent state (this fixture)
2. The post-L2 q, k, v vectors (existing fixtures from diary 0108)
3. The computed g and beta scalars (existing g/beta fixtures from diary 0106)

This fixture provides input 1. Combined with the existing fixtures, all inputs
to the recurrent shader are now deterministic and file-based.

## Verification

- `cmake --build build -j` compiles cleanly
- Fixture size: exactly 1048704 bytes (262176 fp32 values)
- Decode result unchanged: `generated_tokens` was `[410, 149852]`
- `python3 tests/run_diary_check.py`
- `git diff --check`

## Remaining scope

- Not recurrent-core parity.
- Not norm/gate or output projection parity.
- Not layer-shaped persistent decode.
- Not inference or megakernel completion.

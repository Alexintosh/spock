# 0041: Fused DeltaNet Conv+L2 Decode Sub-Block

## Goal

Fuse the DeltaNet decode `conv1d_step` and L2 Q/K norm into a single Vulkan
dispatch, reducing per-layer dispatch count on the merged DeltaNet decode path.
This replaces three separate dispatches (`conv1d_step`, L2 Q, and L2 K) with
one dispatch when `SPOCK_GPU_FUSED_DN_CONV_L2=1` is active. Default inference
is unchanged.

## Background

The model (Qwen 3.5 0.8B) has 24 layers total: 18 DeltaNet and 6 attention.
Each DeltaNet decode step includes a conv1d sliding-window projection followed by
L2 normalization of Q and K vectors. On the baseline path, these are separate
dispatches: `conv1d_step`, `l2_norm` for Q, `l2_norm` for K. Under the merged
DeltaNet gate (`SPOCK_GPU_MERGED_DELTANET=1`), these three dispatches were
already recorded into the same per-layer command buffer (diary 0038), eliminating
extra `submit_and_wait` calls. However, three dispatches still means three
pipeline binds and three dispatch overheads within that command buffer.

Fusing conv1d + L2 Q + L2 K into one shader means one pipeline bind and one
dispatch per DeltaNet layer for this phase, instead of three.

## Implementation Work Completed

### New shader: `shaders/deltanet_conv_l2_qk.comp`

A single compute shader that performs:

1. conv1d sliding-window projection (reading from the QKV projection output and
   conv1d state buffer, writing updated conv1d state and the conv1d output)
2. L2 normalization of the Q slice of the conv1d output
3. L2 normalization of the K slice of the conv1d output

The shader uses the `ds_dn_conv` descriptor set and `pipeline_layout_32`. It
reads and writes the same buffers as the three separate dispatches it replaces,
and is verified through token parity against the existing unfused path.

### Runtime gate: `SPOCK_GPU_FUSED_DN_CONV_L2=1`

Default-off. Active only when `SPOCK_GPU_MERGED_DELTANET=1` is already set.
When both are active, the runtime replaces the three-dispatch sequence
(conv1d_step -> L2 Q -> L2 K) in the DeltaNet decode branch with a single
dispatch of `deltanet_conv_l2_qk.comp`.

### What this is not

- **Not full GPU offload.** The host still orchestrates per-layer iteration,
  command buffer recording, submission, and fence waits.
- **Not persistent dispatch.** Each token still gets command buffer submissions
  on the host-managed path.
- **Not the megakernel.** This fuses one narrow sub-block of the DeltaNet decode
  path; the rest (recurrent step, norm/gate, out_proj, attention, MLP) remains
  as separate dispatches.
- **Not a standalone gate.** Requires `SPOCK_GPU_MERGED_DELTANET=1`. The fused
  shader replaces dispatches only within the merged command buffer path.

## Verification

All commands were run locally on the target Vulkan/RADV path.

### Whitespace and build

```sh
git diff --check
```

No whitespace errors.

```sh
cmake --build build -j
```

Passed.

### Default short parity (no env gates)

Default parity passed — confirming the gate does not alter inference when
disabled.

### Fused + merged + per-layer short parity

```sh
SPOCK_GPU_FUSED_DN_CONV_L2=1 \
SPOCK_GPU_MERGED_DELTANET=1 \
SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1 \
  python3 tests/run_vk_decode_parity.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --reference tests/data/reference_tokens_fp16.jsonl \
  --ids short_correctness_001 \
  --max-new-tokens 16
```

Parity OK.

### Fused + merged + per-layer + single-submit + device-resident + deferred

```sh
SPOCK_GPU_FUSED_DN_CONV_L2=1 \
SPOCK_GPU_SINGLE_SUBMIT=1 \
SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1 \
SPOCK_GPU_MERGED_DELTANET=1 \
SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1 \
SPOCK_GPU_DEFER_TOKEN_DOWNLOAD=1 \
  python3 tests/run_vk_decode_parity.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --reference tests/data/reference_tokens_fp16.jsonl \
  --ids short_correctness_001 \
  --max-new-tokens 16
```

Parity OK.

### Full combined gates

```sh
SPOCK_GPU_FUSED_DN_CONV_L2=1 \
SPOCK_GPU_SINGLE_SUBMIT=1 \
SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1 \
SPOCK_GPU_MERGED_DELTANET=1 \
SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1 \
SPOCK_GPU_DEFER_TOKEN_DOWNLOAD=1 \
SPOCK_GPU_CHUNK_PREFILL=1 \
SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1 \
SPOCK_GPU_CHUNK_PREFILL_TILED=1 \
  python3 tests/run_vk_decode_parity.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --reference tests/data/reference_tokens_fp16.jsonl \
  --ids mixed_correctness_023,pp520_046 \
  --max-new-tokens 4
```

Parity OK.

### CTest chunk-prefill regression

```sh
SPOCK_GPU_FUSED_DN_CONV_L2=1 \
SPOCK_GPU_MERGED_DELTANET=1 \
SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1 \
  ctest --test-dir build \
  -R "spock_vk_decode_gpu_collect_chunk_prefill_(short|tiled|short_baseline)" \
  --output-on-failure
```

3/3 passed in 128.97s.

## Known Limitations

- **Default-off.** Must be explicitly set. The baseline three-dispatch path
  remains the default.
- **Requires merged DeltaNet.** Only dispatched when
  `SPOCK_GPU_MERGED_DELTANET=1` is active.
- **Decode only.** Not used during prefill. Prefill uses the chunk-rule path,
  not the per-token conv1d+L2 decode path.
- **Narrow scope.** Fuses only the conv1d + L2 Q/K sub-block. The recurrent
  step, norm/gate, and output projection remain separate dispatches.

## Next Work

- Measure dispatch reduction on the RX 6750 XT using `SPOCK_GPU_TIMESTAMPS=1`.
- Consider fusing additional adjacent DeltaNet sub-blocks (e.g., g/beta
  computation) if performance data justifies it.
- Expand parity verification to longer prompts and broader P0 subsets.

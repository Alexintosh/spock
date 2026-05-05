# 0042: Fused DeltaNet G/Beta + Recurrent Decode Sub-Block

## Goal

Fuse the DeltaNet decode g/beta scalar computation with the recurrent update
into a single Vulkan dispatch. This removes one dispatch per DeltaNet layer on
the merged decode path by replacing `deltanet_compute_g_beta.comp` followed by
`deltanet_recurrent.comp` with `deltanet_recurrent_gbeta.comp` when
`SPOCK_GPU_FUSED_DN_GBETA_RECURRENT=1` is active.

Default inference is unchanged. The fused path is opt-in and only participates
inside the existing merged DeltaNet decode path.

## Background

After diary 0041, the DeltaNet decode path had a correct fused conv+L2
sub-block, but a quick timestamp comparison did not show a clear throughput
gain. That result suggested the project should keep moving toward larger
DeltaNet hot-block fusion instead of spending more effort on host scheduling or
very small dispatch reductions.

The next natural boundary was the pair:

1. `deltanet_compute_g_beta.comp`
2. `deltanet_recurrent.comp`

In the unfused path, `deltanet_compute_g_beta.comp` reads projected `dn_a`,
projected `dn_b`, and the packed per-layer `a_log`/`dt_bias` table, computes
one `g` and one `beta` scalar per DeltaNet head, and writes those scalars into
the tail of `dn_state`. The recurrent shader then reads the same state buffer,
including that g/beta tail, to decay the recurrent matrix and apply the gated
delta update.

That is correct, but it creates an avoidable intermediate. The recurrent
workgroup already owns one head. It can compute that head's g/beta scalars
directly from `dn_a`, `dn_b`, and `a_log`/`dt_bias`, then immediately use them
for the recurrent update. This is a better match for the eventual megakernel
direction because it removes both an extra dispatch and an intermediate
write/read dependency.

## Implementation Work Completed

### New shader: `shaders/deltanet_recurrent_gbeta.comp`

The new shader combines the scalar computation from
`deltanet_compute_g_beta.comp` with the recurrent update from
`deltanet_recurrent.comp`.

It uses one workgroup per DeltaNet head, matching the existing recurrent
shader. Each workgroup:

1. reads its head's `dn_a` and `dn_b` fp16 values
2. reads the matching fp32 `a_log` and `dt_bias` pair from the packed table
3. computes `g = -exp(a_log) * softplus(a + dt_bias)`
4. computes `beta = sigmoid(b)`
5. loads Q/K/V into shared memory
6. decays the recurrent state by `exp(g)`
7. computes `kv_mem`, `delta`, updates state, and writes the output over the V
   section exactly like the existing recurrent shader

The shader uses a new six-binding descriptor layout:

| Binding | Buffer |
|---------|--------|
| 0 | `dn_a` fp16 |
| 1 | `dn_b` fp16 |
| 2 | packed `dn_a_log_bias` fp32 |
| 3 | Q slice of `dn_qkv` |
| 4 | K/V slice of `dn_qkv`, with output overwriting V |
| 5 | current DeltaNet layer recurrent state |

The g/beta tail in `dn_state` is not used as an intermediate when this gate is
enabled.

### Runtime gate: `SPOCK_GPU_FUSED_DN_GBETA_RECURRENT=1`

The new path is default-off and requires the merged DeltaNet command-buffer path
to be active. In practice, the gate is considered only when
`SPOCK_GPU_MERGED_DELTANET=1` is active and diagnostics that disable merged
DeltaNet recording are not active.

When the gate is enabled, decode skips the standalone
`deltanet_compute_g_beta` dispatch and binds `deltanet_recurrent_gbeta` in
place of `deltanet_recurrent`. When the gate is disabled, the original g/beta
and recurrent dispatches remain unchanged.

### Descriptor and pipeline wiring

The runtime now owns:

- `ds_layout_6`
- `pipeline_layout_6_32`
- `deltanet_recurrent_gbeta`
- `deltanet_recurrent_gbeta_module`
- shared descriptor set `dn_recurrent_gbeta`
- per-layer descriptor vector `dn_recurrent_gbeta`

The per-layer descriptor set is pre-bound for DeltaNet layers so the gate
composes with `SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1` and
`SPOCK_GPU_SINGLE_SUBMIT=1`. This raises the opt-in fused per-layer descriptor
coverage from 30 sets per layer to 31 sets per layer. With the checked-in model
configuration, that is 31 x 24 = 744 per-layer sets.

## Verification

All verification was run locally against the repacked fp16 artifact.

### Whitespace and build

```sh
git diff --check
cmake --build build -j
```

Both passed. The build compiled `deltanet_recurrent_gbeta.comp.spv` and linked
`spock-decode`.

### New fused recurrent gate

```sh
SPOCK_GPU_FUSED_DN_GBETA_RECURRENT=1 \
SPOCK_GPU_MERGED_DELTANET=1 \
SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1 \
  python3 tests/run_vk_decode_parity.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --reference tests/data/reference_tokens_fp16.jsonl \
  --ids short_correctness_001 \
  --max-new-tokens 16
```

Result: parity OK, checked 1 prompt, 0 failures.

### Combined fused decode gates

```sh
SPOCK_GPU_FUSED_DN_GBETA_RECURRENT=1 \
SPOCK_GPU_FUSED_DN_CONV_L2=1 \
SPOCK_GPU_MERGED_DELTANET=1 \
SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1 \
SPOCK_GPU_SINGLE_SUBMIT=1 \
SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1 \
SPOCK_GPU_DEFER_TOKEN_DOWNLOAD=1 \
  python3 tests/run_vk_decode_parity.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --reference tests/data/reference_tokens_fp16.jsonl \
  --ids short_correctness_001 \
  --max-new-tokens 16
```

Result: parity OK, checked 1 prompt, 0 failures.

### Full combined gates on mixed prompts

```sh
SPOCK_GPU_FUSED_DN_GBETA_RECURRENT=1 \
SPOCK_GPU_FUSED_DN_CONV_L2=1 \
SPOCK_GPU_MERGED_DELTANET=1 \
SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1 \
SPOCK_GPU_SINGLE_SUBMIT=1 \
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

Result: parity OK, checked 2 prompts, 0 failures.

### CTest chunk-prefill subset

```sh
SPOCK_GPU_FUSED_DN_GBETA_RECURRENT=1 \
SPOCK_GPU_FUSED_DN_CONV_L2=1 \
SPOCK_GPU_MERGED_DELTANET=1 \
SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1 \
  ctest --test-dir build \
  -R "spock_vk_decode_gpu_collect_chunk_prefill_(short|tiled|short_baseline)" \
  --output-on-failure
```

Result: 3/3 passed in 128.81 seconds.

## Known Limitations

- **Not full GPU offload.** The host still records and submits command buffers.
- **Not persistent dispatch.** There is still no resident GPU scheduler loop.
- **Not the megakernel.** This is a second narrow DeltaNet decode fusion slice,
  not a full layer or full-model fusion.
- **Default-off.** The unfused g/beta and recurrent dispatches remain the
  default runtime path.
- **Decode only.** Prefill still uses the chunk-prefill path.
- **No standalone performance claim yet.** Correctness is established; timing
  should be measured after this and the previous fused conv+L2 gate are both in
  place.

## Next Work

The next step is to measure GPU timestamp deltas with both fused DeltaNet decode
sub-blocks enabled. If the bottleneck remains dominated by the recurrent update
itself, the next fusion boundary should probably move outward to combine
recurrent output with norm/gate or out projection, or add finer block-level GPU
timestamp instrumentation to quantify where the per-token GPU time is going.

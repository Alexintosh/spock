# 0045: Tiled LM-Head Decode Matvec

## Goal

Add a default-off tiled LM-head decode shader that parallelizes each vocabulary
row dot product across lanes. The immediate goal is to remove the biggest
coarse region exposed by diary 0044 block-level timestamps without changing
default inference behavior.

The new gate is `SPOCK_GPU_LM_HEAD_TILED=1`. It affects only the final LM-head
dispatch in `decode()`. General matvec users, diagnostics, and the default
path continue to use `matvec.comp`.

## Background

Diary 0044 added `gpu_region_us`, which made the next bottleneck visible. On
an 8-token full fused single-submit sample, the original LM-head path consumed
about 2.61 seconds of roughly 5.43 seconds of GPU decode time. That was too
large to ignore: after the DeltaNet fusion slices, the vocabulary projection
was the dominant region in the sample.

The old LM-head path reused the generic `matvec.comp` shader. That shader maps
one invocation to one output row. For ordinary hidden-size projections this is
simple and acceptable, but the LM head has one row per vocabulary token. With
the checked-in model configuration, each row dot product loops serially over
the hidden dimension inside a single invocation. The GPU receives many rows,
but each row's 1024 multiply-adds are not split across lanes. That is a poor
match for the final vocabulary projection, where the same input vector is read
against a very large row-major weight matrix.

The new shader keeps the same math boundary and output format: fp16 weights,
fp16 input, fp32 accumulation, fp16 logits. It only changes how work is divided
inside the LM-head dispatch.

## Implementation Work Completed

### New shader: `shaders/lm_head_tiled.comp`

The new shader uses the same descriptor layout as `matvec.comp`:

| Binding | Buffer |
|---------|--------|
| 0 | LM-head weights, fp16 row-major `[VOCAB, HIDDEN]` |
| 1 | input hidden vector, fp16 `[HIDDEN]` |
| 2 | output logits, fp16 `[VOCAB]` |

The push constants are unchanged:

```cpp
struct {
  uint32_t out_dim;
  uint32_t in_dim;
};
```

Each workgroup computes eight output rows. The workgroup has 64 lanes. For
each row, every lane accumulates a slice of the hidden dimension, writes its
partial sum into shared memory, and then participates in a tree reduction. Lane
0 writes the final fp16 logit for each row handled by the workgroup.

This means the LM-head dot product is no longer one serial hidden-dimension
loop per row invocation. The same total math remains, but the row dot product
has lane-level parallelism.

### Runtime gate: `SPOCK_GPU_LM_HEAD_TILED=1`

The runtime now owns:

- `lm_head_tiled`
- `lm_head_tiled_module`

The shader is built by CMake and loaded with the other decode shaders. When
the gate is active, the final LM-head section binds `P.lm_head_tiled` and
dispatches `ceil(VOCAB / 8)` workgroups. When the gate is disabled, the
runtime binds `P.matvec` and dispatches the original `ceil(VOCAB / 64)`
workgroups.

The descriptor set stays `D.lm_head`; no new descriptor layout or per-layer
descriptor coverage is needed. This keeps the slice narrow and makes the gate
compose with single-submit, deferred token download, and the fused DeltaNet
decode gates.

## Verification

All verification was run locally against the repacked fp16 artifact.

### Whitespace and build

```sh
git diff --check
cmake --build build -j
```

Both passed. The build compiled `lm_head_tiled.comp.spv` and linked
`spock-decode`.

### Default parity

```sh
python3 tests/run_vk_decode_parity.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --reference tests/data/reference_tokens_fp16.jsonl \
  --ids short_correctness_001 \
  --max-new-tokens 16
```

Result: parity OK, checked 1 prompt, 0 failures.

### Gated tiled LM-head parity

```sh
SPOCK_GPU_LM_HEAD_TILED=1 \
SPOCK_GPU_TIMESTAMPS=1 \
SPOCK_GPU_BLOCK_TIMESTAMPS=1 \
SPOCK_GPU_FUSED_DN_REC_NORM_GATE=1 \
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

### Mixed prompts with full GPU gates

```sh
SPOCK_GPU_LM_HEAD_TILED=1 \
SPOCK_GPU_FUSED_DN_REC_NORM_GATE=1 \
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
SPOCK_GPU_LM_HEAD_TILED=1 \
SPOCK_GPU_FUSED_DN_REC_NORM_GATE=1 \
SPOCK_GPU_FUSED_DN_CONV_L2=1 \
SPOCK_GPU_MERGED_DELTANET=1 \
SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1 \
  ctest --test-dir build \
  -R "spock_vk_decode_gpu_collect_chunk_prefill_(short|tiled|short_baseline)" \
  --output-on-failure
```

Result: 3/3 passed.

### Timing sample

With the full fused single-submit timestamp gates and `--max-new-tokens 8`,
the original LM-head sample reported approximately:

- `gpu_decode_us`: 5.42572e+06
- `gpu_region_us["lm_head"]`: 2.6147e+06

With `SPOCK_GPU_LM_HEAD_TILED=1`, the comparable local sample reported
approximately:

- `gpu_decode_us`: 2.31209e+06
- `gpu_region_us["lm_head"]`: 38399.3

This is a strong directional result, but it is still a local sample, not a
formal benchmark. The gate remains default-off until broader prompt coverage
and repeated timing runs are collected.

## Known Limitations

- **Default-off.** The original generic matvec path remains the default.
- **Decode LM head only.** This does not change MLP projections, attention
  projections, DeltaNet projections, or diagnostic LM-head calls.
- **Assumes the checked-in hidden dimension divides cleanly across lanes.** The
  current model has `HIDDEN=1024`, which is divisible by 64.
- **Not full GPU offload.** The host still owns decode-loop control and command
  submission.
- **Not persistent dispatch.** There is still no resident GPU scheduler loop.
- **Not the megakernel.** This is a targeted projection kernel improvement.

## Next Work

Broaden correctness and timing coverage with the tiled LM-head gate enabled.
If repeated samples remain stable, the next step is to decide whether to make
the tiled LM head part of the main fast gate stack or keep it separate while
the project moves toward persistent dispatch and larger fused layer kernels.

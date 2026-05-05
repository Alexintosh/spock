# 0046: Tiled Decode Matvec

## Goal

Extend the tiled matvec approach from diary 0045 (LM-head only) to all matvec
dispatches in the main decode path: attention projections, DeltaNet merged
projections, DeltaNet out_proj, MLP gate/up/down projections, and the final LM
fallback when the tiled LM-head gate is not active. The goal is to reduce per-
layer decode time by parallelizing hidden-dimension dot products across 64 lanes
for every weight projection, not just the vocabulary head.

The new gate is `SPOCK_GPU_MATVEC_TILED=1`, default-off. It does not change
`matvec_f32_out`, `cmd1` fallback dispatches, diagnostic paths, or the default
inference behavior.

## Background

Diary 0045 introduced `lm_head_tiled.comp`, which parallelized the LM-head
vocabulary dot product across lanes. That removed the dominant coarse region in
the timestamp profile. After that gate, the remaining decode time was spread
across the 24 layer matvec dispatches: attention Q/K/V/O projections, DeltaNet
QKV/Z/A/B projections and out_proj, and MLP gate/up/down projections. Each of
those still used the generic `matvec.comp`, which maps one invocation to one
output row with a serial hidden-dimension loop.

The per-layer projections have much smaller out_dim than the vocabulary head
(e.g. 3584 for MLP gate/up/down, 1024 for attention O/DeltaNet out projection),
so the absolute
gain per dispatch is smaller. But there are roughly 24 layers * ~10 matvec
dispatches per layer = ~240 dispatches, so the aggregate effect is meaningful.

The new shader applies the same lane-parallel dot product strategy to these
projections. Unlike the LM-head shader, which assumed a fixed row stride of 8,
the general tiled matvec needs to handle arbitrary `in_dim` values. It does this
by striding `j += 64` across the hidden dimension, so any `in_dim` works
correctly (partial final tiles are handled naturally by the loop bound).

## Implementation Work Completed

### New shader: `shaders/matvec_tiled.comp`

The new shader uses the same descriptor layout and push constants as
`matvec.comp`:

| Binding | Buffer |
|---------|--------|
| 0 | weights fp16, row-major `[out_dim, in_dim]` |
| 1 | input vector fp16 `[in_dim]` |
| 2 | output vector fp16 `[out_dim]` |

Push constants:

```cpp
struct {
  uint32_t out_dim;
  uint32_t in_dim;
};
```

Each workgroup has 64 lanes and processes `BLOCK_ROWS=8` output rows. For each
row, every lane accumulates a strided slice of the input dimension (`j += 64`),
writes its partial sum into shared memory, and participates in a tree reduction.
Lane 0 writes the final fp16 output element.

The strided `j += 64` loop means `in_dim` does not need to be a multiple of 64.
Lanes beyond the actual input dimension contribute zero to the sum, so the
reduction is correct for arbitrary `in_dim`.

Accumulation is fp32; output is fp16. This matches the numerical boundary of
`matvec.comp`.

### Runtime gate: `SPOCK_GPU_MATVEC_TILED=1`

The runtime now owns `P.matvec_tiled`, built from the new shader. When the gate
is active, `bind_matvec` and `dispatch_matvec` (the helper functions used for
all general matvec dispatches in the main decode path) select the tiled pipeline
and compute the workgroup count as `ceil(out_dim / BLOCK_ROWS)` instead of
`ceil(out_dim / 64)`.

The gate affects:

- Attention projections: Q, K, V, O
- DeltaNet merged projections: QKV, Z, A, B
- DeltaNet out_proj
- MLP projections: gate, up, down
- Final LM head fallback when `SPOCK_GPU_LM_HEAD_TILED` is not active

The gate does NOT affect:

- `matvec_f32_out` dispatches (separate fp32 output path)
- `cmd1` fallback dispatches
- Diagnostic matvec calls
- The tiled LM-head shader (`SPOCK_GPU_LM_HEAD_TILED=1` still uses
  `lm_head_tiled.comp` for the final vocabulary projection)

The descriptor sets are unchanged; no new descriptor layout or per-layer
descriptor coverage is needed. The gate composes with single-submit, deferred
token download, fused DeltaNet, device-resident token, and chunk-prefill gates.

## Verification

All verification was run locally against the repacked fp16 artifact.

### Whitespace and build

```sh
git diff --check
cmake --build build -j
```

Both passed. The build compiled `matvec_tiled.comp.spv` and linked
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

### Gated tiled matvec + tiled LM-head parity

```sh
SPOCK_GPU_MATVEC_TILED=1 \
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
SPOCK_GPU_MATVEC_TILED=1 \
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
SPOCK_GPU_MATVEC_TILED=1 \
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

With `SPOCK_GPU_MATVEC_TILED=1`, `SPOCK_GPU_LM_HEAD_TILED=1`, and the full
fused single-submit timestamp gates at `--max-new-tokens 8`, a local sample
reported approximately:

- `gpu_decode_us`: 157679
- `per_token_gpu_us`: about 5.8 ms for step 0, then 21-22 ms for subsequent
  steps
- `gpu_region_us["lm_head"]`: about 23542.9 us
- Per-layer regions: about 4.8-5.5 ms each

The previous tiled-LM-only sample (diary 0045) had `gpu_decode_us` about
2.31e+06. The matvec+tiled combination reduces total GPU decode time by roughly
an order of magnitude in this local sample.

This is directional only, not a formal benchmark. The sample size is one run
with one prompt, and the timing is sensitive to GPU thermal state, driver
scheduling, and system load.

## Known Limitations

- **Default-off.** The original generic matvec path remains the default.
  Reduction order changes under the gate (strided lane reduction vs serial
  accumulation), so bit-exact fp16 output is not guaranteed between the two
  paths. Parity is checked at the argmax level.
- **Main decode path only.** The gate does not affect `matvec_f32_out`,
  `cmd1` fallback dispatches, or diagnostic matvec calls.
- **Not full GPU offload.** The host still owns decode-loop control and command
  submission.
- **Not persistent dispatch.** There is still no resident GPU scheduler loop.
- **Not the megakernel.** This is a targeted projection kernel improvement.

## Next Work

Broader timing and correctness coverage with the tiled matvec gate. If repeated
samples remain stable, the natural next step is to decide whether to promote
the tiled matvec to default alongside or after the tiled LM head, and whether
to combine both into a single gate or keep them independently switchable for
diagnostic isolation.

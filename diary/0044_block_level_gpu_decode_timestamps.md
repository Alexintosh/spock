# 0044: Block-Level GPU Decode Timestamps

## Goal

Add opt-in block-level GPU timestamp instrumentation for decode so the project
can see where GPU time goes after the first DeltaNet fusion slices. Diary 0040
measured total GPU time for each recorded decode token. Diary 0044 keeps that
existing measurement intact and adds a narrower diagnostic gate,
`SPOCK_GPU_BLOCK_TIMESTAMPS=1`, that breaks single-submit decode work into
coarse regions.

This is not an optimization. It is a measurement tool for choosing the next
fusion or scheduling slice with evidence instead of guessing from whole-token
timings.

## Background

Diaries 0041, 0042, and 0043 fused increasingly large DeltaNet decode
sub-blocks:

1. conv1d + Q/K L2 norm
2. g/beta + recurrent update
3. g/beta + recurrent update + RMSNorm + SiLU(z) gate

The third slice passed correctness, but a quick timestamp sample was
essentially flat compared with the second fused slice. That made the next
engineering question more important than another immediate shader fusion:
which region is currently consuming the GPU time?

The existing `SPOCK_GPU_TIMESTAMPS=1` gate brackets the decode command buffer
and reports `gpu_decode_us` plus `per_token_gpu_us`. That tells us whether the
GPU-side token time moved, but it does not say whether time is now dominated by
DeltaNet layers, attention layers, the final LM head matvec, argmax, or other
work. A megakernel-oriented project needs that split before widening the fused
boundary again.

## Implementation Work Completed

### New gate: `SPOCK_GPU_BLOCK_TIMESTAMPS=1`

The new gate is default-off. It is active only when all of the following are
true:

- `SPOCK_GPU_TIMESTAMPS=1` is active
- the selected Vulkan queue supports timestamp queries
- the decode step is eligible for the single-submit path
- `SPOCK_GPU_BLOCK_TIMESTAMPS=1` is set exactly

The dependency on `SPOCK_GPU_TIMESTAMPS=1` is intentional. Block timestamps are
a refinement of the existing GPU timestamp mode, not a separate timing system.
When only `SPOCK_GPU_TIMESTAMPS=1` is set, the output remains exactly at the
diary 0040 level: `gpu_decode_us` and `per_token_gpu_us`, with no region
object.

### Query layout

Each recorded single-submit decode step uses 56 timestamp queries:

| Region | Query Pair |
|--------|------------|
| `embedding` | 0, 1 |
| `layer_0` through `layer_23` | 2 through 49 |
| `final_norm` | 50, 51 |
| `lm_head` | 52, 53 |
| `argmax` | 54, 55 |

The layer regions are deliberately coarse. A layer timestamp covers that whole
layer's recorded command sequence, including attention or DeltaNet mixer work,
residuals, and MLP work. This is enough to distinguish layer work from final
LM head cost and to compare per-layer outliers, but it is not a per-dispatch
profiler.

The query pool is sized as `max_new_tokens * 56`, but queries are recorded only
for steps that actually use the single-submit command buffer. This matters
because the decode path can still run non-single-submit steps, such as
prefill-related work or skip-layer first decode steps after chunk prefill.
The implementation guards every block timestamp write with the current
single-submit eligibility and verifies that the current decode step has an
active block timestamp range.

### Result and JSON output

`DecodeResult` now has:

```cpp
std::map<std::string, double> gpu_region_us;
```

The runtime sums each region across recorded steps and stores microseconds in
that map. `spock-decode` emits a `gpu_region_us` JSON object only when the map
is non-empty. Default output is unchanged. Timestamp-only output is unchanged.

Example region names are stable:

- `embedding`
- `layer_0` ... `layer_23`
- `final_norm`
- `lm_head`
- `argmax`

The existing `gpu_decode_us` and `per_token_gpu_us` fields are preserved and
remain the whole-command-buffer timing source.

## Verification

All verification was run locally against the repacked fp16 artifact.

### Whitespace and build

```sh
git diff --check
cmake --build build -j
```

Both passed.

### Default output remains unchanged

```sh
build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --max-new-tokens 1
```

Result: decode completed successfully and did not emit `gpu_region_us`.

### Timestamp-only output remains unchanged

```sh
SPOCK_GPU_TIMESTAMPS=1 \
  build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --max-new-tokens 1
```

Result: decode completed successfully and emitted `gpu_decode_us` and
`per_token_gpu_us`, but did not emit `gpu_region_us`.

### Block timestamp output

```sh
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
  build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --max-new-tokens 2
```

Result: decode completed successfully and emitted `gpu_region_us` with the
expected region keys. The sample showed `lm_head` as the largest measured
region, but this was a small diagnostic run, not a benchmark.

### Parity with full fused single-submit timestamp gates

```sh
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

## Known Limitations

- **Instrumentation only.** This does not change the execution schedule or
  reduce GPU work.
- **Single-submit only.** Block regions are recorded only for single-submit
  eligible decode steps.
- **Coarse regions.** Layer regions are whole-layer timings, not per-dispatch
  timings inside attention, DeltaNet, or MLP.
- **Not full GPU offload.** The host still controls the decode loop and command
  submission.
- **Not persistent dispatch.** There is still no resident GPU scheduler loop.
- **Not the megakernel.** This is a measurement gate to guide that work.
- **Default-off.** No block query pool is allocated unless both timestamp gates
  are active.

## Next Work

Use the `gpu_region_us` split to choose the next implementation slice. If the
LM head remains dominant across longer and more representative prompts, the
next target should be LM head/argmax structure or vocabulary matvec strategy
rather than another narrow DeltaNet fusion. If specific DeltaNet layers remain
dominant under the fused gates, add a second level of timing around DeltaNet
sub-dispatches or move outward toward fusing the DeltaNet output projection.

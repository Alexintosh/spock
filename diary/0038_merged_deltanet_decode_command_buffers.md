# 0038: Merged DeltaNet Decode Command Buffers — Removing Two Per-Layer Submits Under the Gate

## Goal

Add an opt-in decode path that reduces host orchestration after the descriptor
pre-binding work from diaries 0029-0037.

`SPOCK_GPU_MERGED_DELTANET=1` records two formerly separate DeltaNet command
buffer phases directly into the existing per-layer decode command buffer:

- Phase 1: DeltaNet input norm, QKV/Z/A/B projections, conv1d, and Q/K L2 norm.
- Phase 2: `dn_compute_g_beta`.

The legacy path submitted both phases separately through `submit_and_wait`.
The merged path removes those two extra submits per DeltaNet layer on the
decode fast path. It is default-off and remains conservative: it does not
attempt single-submit, persistent dispatch, a megakernel, or full GPU offload.

## Implementation Work Completed

### Gate

The new path is enabled only by:

```sh
SPOCK_GPU_MERGED_DELTANET=1
```

It is additionally disabled when intermediate decode diagnostics are active:

```cpp
const bool can_merge_deltanet = merge_deltanet_cmds &&
    dump_step_components < 0 && dump_step_hiddens < 0;
```

This preserves the old submit boundaries for diagnostic paths that may need
intermediate GPU state.

### DeltaNet phase 1 merge

In the DeltaNet decode branch, the following kernels are now recorded directly
into the per-layer `cmd` when the gate is active:

- `rmsnorm`
- `dn_qkv_proj`
- `dn_z_proj`
- `dn_a_proj`
- `dn_b_proj`
- `conv1d_step`
- `dn_l2_q`
- `dn_l2_k`

The fallback path still allocates `cmd1`, records the same sequence, ends it,
and calls `submit_and_wait(cmd1)`.

### g/beta merge

The `deltanet_compute_g_beta` dispatch is also recorded into the same per-layer
`cmd` when the gate is active. A barrier over the per-layer `dn_state` slab is
inserted after the dispatch because `deltanet_recurrent` reads that state.

The fallback path still allocates `gb_cmd`, records `deltanet_compute_g_beta`,
and calls `submit_and_wait(gb_cmd)`.

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

### Default path

```sh
python3 tests/run_vk_decode_parity.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --reference tests/data/reference_tokens.jsonl \
  --ids short_correctness_001 \
  --max-new-tokens 16
```

```json
{"status":"ok","checked":1,"failures":[]}
```

### Merged DeltaNet gate

```sh
SPOCK_GPU_MERGED_DELTANET=1 \
  python3 tests/run_vk_decode_parity.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --reference tests/data/reference_tokens.jsonl \
  --ids short_correctness_001 \
  --max-new-tokens 16
```

```json
{"status":"ok","checked":1,"failures":[]}
```

### Merged DeltaNet + per-layer descriptors

```sh
SPOCK_GPU_MERGED_DELTANET=1 \
SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1 \
  python3 tests/run_vk_decode_parity.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --reference tests/data/reference_tokens.jsonl \
  --ids short_correctness_001 \
  --max-new-tokens 16
```

```json
{"status":"ok","checked":1,"failures":[]}
```

### Combined gate suite

```sh
SPOCK_GPU_MERGED_DELTANET=1 \
SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1 \
SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1 \
SPOCK_GPU_DEFER_TOKEN_DOWNLOAD=1 \
SPOCK_GPU_CHUNK_PREFILL=1 \
SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1 \
SPOCK_GPU_CHUNK_PREFILL_TILED=1 \
  python3 tests/run_vk_decode_parity.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --reference tests/data/reference_tokens.jsonl \
  --ids short_correctness_001 \
  --max-new-tokens 16
```

```json
{"status":"ok","checked":1,"failures":[]}
```

```sh
SPOCK_GPU_MERGED_DELTANET=1 \
SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1 \
SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1 \
SPOCK_GPU_DEFER_TOKEN_DOWNLOAD=1 \
SPOCK_GPU_CHUNK_PREFILL=1 \
SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1 \
SPOCK_GPU_CHUNK_PREFILL_TILED=1 \
  python3 tests/run_vk_decode_parity.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --reference tests/data/reference_tokens.jsonl \
  --ids mixed_correctness_023,pp520_046 \
  --max-new-tokens 4
```

```json
{"status":"ok","checked":2,"failures":[]}
```

### Chunk-prefill regression

```sh
SPOCK_GPU_MERGED_DELTANET=1 \
SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1 \
SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1 \
SPOCK_GPU_DEFER_TOKEN_DOWNLOAD=1 \
  ctest --test-dir build --output-on-failure -R spock_vk_decode_gpu_collect_chunk_prefill
```

| Test | Result |
|------|--------|
| `spock_vk_decode_gpu_collect_chunk_prefill_short` | Passed, 115.03 s |
| `spock_vk_decode_gpu_collect_chunk_prefill_tiled` | Passed, 8.99 s |
| `spock_vk_decode_gpu_collect_chunk_prefill_short_baseline` | Passed, 5.08 s |
| **Total real time** | **129.10 s** |

## Known Limitations

- This is **not** full GPU offload. The host still records/submits per-layer
  work, waits on fences, observes generated-token outputs, and owns fallback
  and diagnostic paths.
- This is **not** single-submit. It removes two extra DeltaNet submits inside
  each DeltaNet layer, but the runtime still submits each layer command buffer.
- This is **not** persistent dispatch or the megakernel. No shader fusion or
  persistent scheduler was introduced.
- The gate does not alter prefill or `correct_last_token_hidden()`.
- The gate is disabled for `dump_step_components` and `dump_step_hiddens`
  diagnostics to preserve intermediate observation points.

## Next Work

The next orchestration target is broader command-buffer scheduling:

- reduce or eliminate per-layer submit/fence waits;
- decide whether the next step should be single-submit per token or a smaller
  staged command-buffer cache;
- keep step-varying parameters explicit and testable before attempting any
  persistent-dispatch or megakernel design.

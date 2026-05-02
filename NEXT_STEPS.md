# Spock Handoff: Next Steps

## Current State

The branch has a working layer-major prefill using the recurrent DeltaNet path.
Full 48-prompt parity test passes. The layer-major restructuring is complete and verified.

### What Works
- **CTest regression gate for GPU-collected chunk-prefill path** (diary 0022):
  Two new CTest tests (`spock_vk_decode_gpu_collect_chunk_prefill_short` and
  `spock_vk_decode_gpu_collect_chunk_prefill_short_baseline`) protect the
  double-gated GPU collect → GPU chunk-prefill path from accidental
  regression. Both run `short_correctness_001 --max-new-tokens 1`; the gated
  test sets `SPOCK_GPU_CHUNK_PREFILL=1` and
  `SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1`, the baseline runs with no
  env vars.
- **Session extraction** (`DecodeSession`): persistent device, pipelines, buffers, weights
- **Layer-major prefill** with recurrent DeltaNet path: all 48 reference prompts pass
- **Chunk rule primitive** (`deltanet_chunk.cpp`): unit-tested, produces correct output for
  single tokens and matches CPU recurrent simulation for multi-token sequences
- **Tiled single-dispatch chunk-prefill probe** (diary 0023): proves
  `deltanet_chunk_prefill_tiled.comp` produces correct output in a single
  `vkCmdDispatch(num_heads, ceil(v_dim/TILE_V), 1)` matching CPU reference
  within machine epsilon. Removes the major proof blocker for replacing
  per-head submits. Not yet wired into runtime.
- **All existing tests pass**: capabilities, chunk unit, reference parity (48 prompts × 16 tokens)
- **Experimental GPU chunk-prefill path** wired behind env gate `SPOCK_GPU_CHUNK_PREFILL=1`:
  passes `mixed_correctness_023` and `pp520_046` at `--max-new-tokens 1`. Not the default;
  uses conservative per-head submit workaround (slow).
- **GPU-collected chunk-prefill path** wired behind
  `SPOCK_GPU_CHUNK_PREFILL=1 SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1`:
  preserves GPU-collected Q/K/V/g/beta buffers for all DeltaNet layers in
  device-local per-layer segments and feeds them directly into
  `deltanet_chunk_prefill.comp` via `gpu_chunk_prefill_from_gpu_collect()`.
  Avoids CPU intermediate packing/upload for chunk-prefill inputs.
  Default behavior unchanged. When no diagnostic compare flag is set, the
  per-token CPU collection bridge (staging download, half_to_float conversion,
  prefill_chunks_ population) is now skipped entirely. CPU collection remains
  when either `SPOCK_GPU_COLLECT_PREFILL_COMPARE=1` or
  `SPOCK_GPU_CHUNK_PREFILL_COMPARE=1` is set.
- **GPU collect → chunk-prefill standalone pipeline probe**:
  `spock-deltanet-prefill-pipeline-probe` proves `deltanet_prefill_collect.comp`
  can populate the exact fp32 head-major buffers consumed by
  `deltanet_chunk_prefill.comp`, without CPU intermediate packing. Verified
  `compare-ok` at heads=16, seq_len=104, total_seq=128, chunk_size=64 with
  max_rel_core=8.94e-8, max_rel_state=1.19e-7, nan_count=0.
- **Runtime GPU prefill collection diagnostic** (`SPOCK_GPU_COLLECT_PREFILL_COMPARE=1`):
  Dispatches `deltanet_prefill_collect.comp` from real `DecodeSession` activation
  buffers during layer-major prefill, compares GPU-collected head-major fp32
  Q/K/V/g/beta against the CPU-collected `PrefillChunkState`. Verified exact match
  on `short_correctness_001` (seq_len=9, all 18 DeltaNet layers, max_rel=0,
  max_abs=0, nan_count=0). Diagnostic only; does not change inference output.

### What Needs Work
- **Chunk rule integration**: The fp32 chunk rule produces slightly different output than the
  fp16 recurrent path. For some prompts, this causes argmax divergence on borderline tokens.
  The chunk rule output is *more accurate* (closer to fp32 PyTorch), but the reference was
  generated with the same fp16 recurrent computation. Resolution requires either:
  (a) regenerating the reference with fp32 chunk-based computation, or
  (b) accepting the fp16 recurrent path for parity and using chunk rule only for numerical
      stability on very long prompts.

## Key Findings from Chunk Rule Investigation

1. **Chunk rule is mathematically correct**: CPU simulation confirms chunk output matches
   per-token recurrent simulation for the same Q/K/V/g/beta inputs.
2. **q_scale must NOT be double-applied**: The chunk rule applies `1/sqrt(k_dim)` internally.
   When feeding L2-normalized Q from GPU, set `use_qk_l2norm=false` and do NOT pre-multiply
   by q_scale.
3. **conv1d state is a side effect**: The conv1d kernel updates sliding window state. Running
   it twice (Phase A + Phase C) would corrupt the state. Phase C should only recompute
   input_norm + Z gate projection (no conv1d, no QKV, no A/B projections).
4. **fp16 vs fp32 divergence**: The chunk rule uses fp32 throughout, while the recurrent path
   uses fp16 Q/K/V with fp32 state accumulation. For most prompts these agree, but on
   borderline tokens the difference can flip argmax.

## Next Steps

### 1. GPU chunk-prefill path — efficiency and correctness hardening

The env-gated GPU path (`SPOCK_GPU_CHUNK_PREFILL=1`) is verified-correct
but uses a per-head submit workaround (24 layers × 16 heads = 384 submit-wait
cycles per chunk).

**Tiled single-dispatch proof completed (diary 0023).** A new experimental
shader and probe (`shaders/deltanet_chunk_prefill_tiled.comp` and
`apps/spock-deltanet-chunk-prefill-tiled-probe.cpp`) proves a single
`vkCmdDispatch(num_heads, ceil(v_dim/TILE_V), 1)` produces correct chunk-rule
output matching the CPU reference within machine epsilon. This removes the
major proof blocker for replacing per-head submits. The tiled shader is not
yet wired into runtime — this is the next integration step.

Follow-up:
- [Pending] Wire tiled shader into runtime behind a new env gate.
- [Done] GPU collection wired into session (diary 0020): session-owned
  per-layer persistent buffers, env-gated diagnostics, device-local feeds into
  `gpu_chunk_prefill()`. CPU collection bridge bypassed on the no-compare
  gated path (diary 0021): per-token staging downloads, half_to_float
  conversion, and prefill_chunks_ population are skipped when neither compare
  flag is active.
- [Done] Add formal tests for the gated path (regression, per-layer diagnostic,
  parity harness integration).
- Only then consider defaulting to GPU path.

### 2. Chunk rule for numerical stability (optional)

If longer prompts show fp16 accumulation errors, integrate the chunk rule as a fallback.
The implementation approach:
- Phase A: per-token GPU projections → download Q/K/V/g/beta to CPU
- Phase B: CPU chunk rule → upload final state + per-token attn output
- Phase C: per-token GPU: input_norm + Z proj → upload chunk attn → norm+gate + MLP

### 3. Resume megakernel roadmap

After hardware P0 is green, proceed with compute megakernel fusion per IMPLEMENTATION_PLAN.md.
The tiled single-dispatch approach in diary 0023 is a step toward the fused
megakernel design — the per-head-tile workgroup decomposition generalizes to
larger computation kernels that share read-only inputs across tiles.

## Key Files

| File | Purpose |
|------|---------|
| `src/runtime/vk_session.cpp` | Session implementation + layer-major prefill + `gpu_chunk_prefill()` |
| `src/runtime/vk_session.hpp` | Persistent decode session class |
| `src/runtime/deltanet_chunk.cpp` | Native HF-style chunk-rule primitive |
| `src/runtime/vk_device.cpp` | Deterministic RX-vs-llvmpipe device selection |
| `shaders/deltanet_recurrent.comp` | Recurrent decode/update kernel |
| `shaders/deltanet_chunk_prefill.comp` | GPU chunk-rule shader (experimental) |
| `shaders/deltanet_prefill_collect.comp` | GPU per-token prefill collection shader |
| `apps/spock-deltanet-chunk-prefill-probe.cpp` | Standalone probe (9 probe cases) |
| `apps/spock-deltanet-prefill-collect-probe.cpp` | Standalone GPU collection probe |
| `apps/spock-deltanet-prefill-pipeline-probe.cpp` | Standalone collect → chunk-prefill pipeline probe |
| `shaders/deltanet_chunk_prefill_tiled.comp` | Tiled single-dispatch chunk-prefill shader (experimental) |
| `apps/spock-deltanet-chunk-prefill-tiled-probe.cpp` | Standalone tiled chunk-prefill probe (diary 0023) |
| `tests/run_vk_decode_parity.py` | Vulkan-vs-reference parity harness |
| `tests/run_deltanet_chunk_unit.py` | Torch-vs-native chunk-rule regression test |

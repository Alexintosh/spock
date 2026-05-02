# Spock Handoff: Next Steps

## Current State

The branch has a working layer-major prefill using the recurrent DeltaNet path.
Full 48-prompt parity test passes. The layer-major restructuring is complete and verified.

The **GPU-collected + tiled no-compare fast path** (diaries 0025/0026) now
eliminates all CPU data touches for chunk-prefill compute. Diary 0025 removed
the CPU readback/re-upload bridge for chunk-prefill output (device-local
`out_buf`, GPU-to-GPU state copy, `deltanet_chunk_last_to_fp16.comp` shader,
device-local `correct_last_token_hidden()` handoff). Diary 0026 removes the
last CPU data touch — init state zero-fill — by replacing host-visible
`init_buf` + CPU `memset` with device-local `init_buf` + `vkCmdFillBuffer`.
The host still orchestrates (per-layer iteration, command recording,
submission, fence wait) but does not read or write chunk-prefill compute
data on the no-compare GPU-collected+tiled path.

### What Works
- **Opt-in device-resident decode token embedding** (diary 0027): Behind
  `SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1`, the per-step embedding lookup reads
  token_id directly from device-local `argmax_result` instead of a
  CPU push constant. This removes the CPU token value as the source for
  the next embedding. CPU still downloads `argmax_result` each decode
  step for external output/parity — this is NOT full GPU offload and NOT
  the megakernel. New shader `embedding_lookup_from_buffer.comp`.
  Verified parity on `short_correctness_001` (16 tokens),
  `mixed_correctness_023`/`pp520_046` (4 tokens), and combined with full
  GPU chunk-prefill gate suite. Still env-gated, not default.

- **GPU-resident chunk-prefill path** (diaries 0025/0026): On the no-compare
  GPU-collected+tiled path (`SPOCK_GPU_CHUNK_PREFILL=1`,
  `SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1`, `SPOCK_GPU_CHUNK_PREFILL_TILED=1`,
  no compare flag), all chunk-prefill compute data stays on-device.
  Output handoff (diary 0025): device-local `out_buf`, GPU-to-GPU `final_state`
  copy, `deltanet_chunk_last_to_fp16.comp` for on-device fp32→fp16
  last-token extraction, device-local `correct_last_token_hidden()` handoff.
  Init clear (diary 0026): device-local `init_buf` zeroed via `vkCmdFillBuffer`
  instead of CPU `memset`. Verified parity on `short_correctness_001` (16 tokens),
  `mixed_correctness_023`/`pp520_046` (4 tokens), all CTest gates pass.
  Fallback host-visible paths preserved for compare diagnostics, non-tiled
  paths, and CPU-collected chunk input paths.
- **CTest regression gate for GPU-collected chunk-prefill paths** (diaries
  0022, 0024, 0025): Three CTest tests protect the gated GPU chunk-prefill paths.
  `spock_vk_decode_gpu_collect_chunk_prefill_short` (per-head submit, diary
  0022), `spock_vk_decode_gpu_collect_chunk_prefill_tiled` (tiled dispatch,
  diary 0024), and `spock_vk_decode_gpu_collect_chunk_prefill_short_baseline`
  (no env vars). All run `short_correctness_001 --max-new-tokens 1`.
- **Session extraction** (`DecodeSession`): persistent device, pipelines, buffers, weights
- **Layer-major prefill** with recurrent DeltaNet path: all 48 reference prompts pass
- **Chunk rule primitive** (`deltanet_chunk.cpp`): unit-tested, produces correct output for
  single tokens and matches CPU recurrent simulation for multi-token sequences
- **Tiled single-dispatch chunk-prefill runtime gate** (diary 0024):
  `deltanet_chunk_prefill_tiled.comp` is now wired into the runtime behind
  `SPOCK_GPU_CHUNK_PREFILL_TILED=1` (requires `SPOCK_GPU_CHUNK_PREFILL=1`).
  Both CPU-collected and GPU-collected data paths support the tiled dispatch.
  Verified parity on `short_correctness_001` through 16 generated tokens, and
  on `mixed_correctness_023`/`pp520_046` through 4 generated tokens; diagnostics report
  `nan_count=0`.
  CTest tiled gate: 8.95 sec in the latest rerun after handoff flag cleanup.
  Still env-gated, not default.
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
  The output side of this path (diary 0025) is now also GPU-resident on the
  no-compare tiled path: chunk output goes to a device-local buffer,
  `final_state` is copied GPU-to-GPU into `dn_state`, and the last-token fp32
  `core_attn_out` slice is extracted and converted to fp16 by
  `deltanet_chunk_last_to_fp16.comp`.
- **GPU collect → chunk-prefill standalone pipeline probe**:
  `spock-deltanet-prefill-pipeline-probe` proves `deltanet_prefill_collect.comp`
  can populate the exact fp32 head-major buffers consumed by
  `deltanet_chunk_prefill.comp`, without CPU intermediate packing. Verified
  `compare-ok` at heads=16, seq_len=104, total_seq=128, chunk_size=64 with
  max_rel_core=8.94e-8, max_rel_state=1.19e-7, nan_count=0.
- **Runtime GPU prefill collection diagnostic** (`SPOCK_GPU_COLLECT_PREFILL_COMPARE=1`):
  Dispatches `deltanet_prefill_collect.comp` from real `DecodeSession` activation
  buffers during layer-major prefill, compares GPU-collected head-major fp32
  Q/K/V/g/beta against the CPU-collected `PrefillChunkState`. Verified exact match on
  `short_correctness_001` (seq_len=9, all 18 DeltaNet layers, max_rel=0,
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
and now supports a tiled single-dispatch mode
(`SPOCK_GPU_CHUNK_PREFILL_TILED=1`) that removes the per-head submit
workaround (diary 0024). Both CPU-collected and GPU-collected data paths
use the same tiled shader `deltanet_chunk_prefill_tiled.comp`.

The GPU-collected+tiled no-compare path (diary 0025) now also eliminates
the CPU readback/re-upload bridge for chunk-prefill output: device-local
chunk output buffer, GPU-to-GPU `final_state` copy, new
`deltanet_chunk_last_to_fp16.comp` shader for on-device fp32→fp16
conversion of the last-token attn slice, and device-local handoff in
`correct_last_token_hidden()`. The host no longer touches chunk-prefill
output data on this path.

Still env-gated, not default.

Follow-up:
- [Done] Wire tiled shader into runtime behind a new env gate (diary 0024).
- [Done] GPU collection wired into session (diary 0020): session-owned
  per-layer persistent buffers, env-gated diagnostics, device-local feeds into
  `gpu_chunk_prefill()`. CPU collection bridge bypassed on the no-compare
  gated path (diary 0021): per-token staging downloads, half_to_float
  conversion, and prefill_chunks_ population are skipped when neither compare
  flag is active.
- [Done] GPU-resident chunk-prefill output handoff (diary 0025): device-local
  output buffer, GPU-to-GPU state copy, `deltanet_chunk_last_to_fp16.comp`
  shader, device-local `correct_last_token_hidden()` handoff. Host no longer
  touches chunk-prefill output data on the no-compare GPU-collected+tiled path.
- [Done] Add formal tests for the gated paths: CTest suite
  `spock_vk_decode_gpu_collect_chunk_prefill` with 3 tests (per-head,
  tiled, baseline), plus parity harness and diagnostic verification.
- [Done] First prompt-coverage expansion: tiled GPU-collect path passes
  `mixed_correctness_023` and `pp520_046` at `--max-new-tokens 4`.
- [Done] First multi-token tiled decode check:
  `short_correctness_001 --max-new-tokens 16`.
- [Done] Device-local buffer usage fix: `VK_BUFFER_USAGE_TRANSFER_SRC_BIT`
  added to device-local buffers used as copy sources (diary 0025).
- [Done] Init zero staging: replaced host-visible init_buf + CPU memset with
  device-local init_buf + vkCmdFillBuffer (diary 0026). Removes the last CPU
  data touch for chunk-prefill init state on the no-compare path.
- [Pending] Expand coverage to 512+ token prompts and broader P0 subsets.
- [Pending] Multi-token decode verification on longer prompts.
- [Pending] Performance characterization: chunk size sensitivity, occupancy,
  register pressure, driver overhead of 24-dispatch-per-chunk orchestration.
- Only then consider defaulting to GPU path.

### 1a. GPU-resident handoff — remaining CPU mediation

Diaries 0025/0026 removed all CPU data bridges for chunk-prefill compute
(device-local output handoff + GPU-side init clear), but the host still
mediates the fast path in non-data ways:
- Per-layer orchestration and submission (host iterates layers, records
  command sequences, submits, waits).
- `gpu_chunk_handoff_ready_` flag is a host-visible boolean (control, not
  data, but still host-mediated).
- Diagnostic/fallback paths require host-visible readback.
- Init zero staging for fallback/compare paths still uses CPU memset.
- Decode argmax/logit computation is on CPU.
- Full megakernel fusion and persistent dispatch not started.

### 2. Chunk rule for numerical stability (optional)

If longer prompts show fp16 accumulation errors, integrate the chunk rule as a fallback.
The implementation approach:
- Phase A: per-token GPU projections → download Q/K/V/g/beta to CPU
- Phase B: CPU chunk rule → upload final state + per-token attn output
- Phase C: per-token GPU: input_norm + Z proj → upload chunk attn → norm+gate + MLP

### 3. Resume megakernel roadmap

After hardware P0 is green, proceed with compute megakernel fusion per IMPLEMENTATION_PLAN.md.
The tiled single-dispatch approach in diaries 0023/0024 is a step toward the fused
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
| `shaders/deltanet_chunk_last_to_fp16.comp` | GPU fp32→fp16 last-token attn extraction shader (diary 0025) |
| `apps/spock-deltanet-chunk-prefill-tiled-probe.cpp` | Standalone tiled chunk-prefill probe (diary 0023) |
| `tests/run_vk_decode_parity.py` | Vulkan-vs-reference parity harness |
| `tests/run_deltanet_chunk_unit.py` | Torch-vs-native chunk-rule regression test |

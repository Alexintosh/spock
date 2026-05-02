# Spock Handoff: Next Steps

## Current State

The branch has a working layer-major prefill using the recurrent DeltaNet path.
Full 48-prompt parity test passes. The layer-major restructuring is complete and verified.

### What Works
- **Session extraction** (`DecodeSession`): persistent device, pipelines, buffers, weights
- **Layer-major prefill** with recurrent DeltaNet path: all 48 reference prompts pass
- **Chunk rule primitive** (`deltanet_chunk.cpp`): unit-tested, produces correct output for
  single tokens and matches CPU recurrent simulation for multi-token sequences
- **All existing tests pass**: capabilities, chunk unit, reference parity (48 prompts × 16 tokens)
- **Experimental GPU chunk-prefill path** wired behind env gate `SPOCK_GPU_CHUNK_PREFILL=1`:
  passes `mixed_correctness_023` and `pp520_046` at `--max-new-tokens 1`. Not the default;
  uses conservative per-head submit workaround (slow).

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
cycles per chunk). Follow-up:
- Replace per-head submit with a correct efficient shader (single dispatch,
  all heads, intra-shader sync).
- Wire GPU collection into session (collect shader exists as verified probe; see
  diary 0017 Extension). Add session-owned buffers, env gate, and feed into
  gpu_chunk_prefill().
- Add formal tests for the gated path (regression, per-layer diagnostic,
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

## Key Files

| File | Purpose |
|------|---------|
| `src/runtime/vk_session.cpp` | Session implementation + layer-major prefill + `gpu_chunk_prefill()` |
| `src/runtime/vk_session.hpp` | Persistent decode session class |
| `src/runtime/deltanet_chunk.cpp` | Native HF-style chunk-rule primitive |
| `src/runtime/vk_device.cpp` | Deterministic RX-vs-llvmpipe device selection |
| `shaders/deltanet_recurrent.comp` | Recurrent decode/update kernel |
| `shaders/deltanet_chunk_prefill.comp` | GPU chunk-rule shader (experimental) |
| `apps/spock-deltanet-chunk-prefill-probe.cpp` | Standalone probe (9 probe cases) |
| `tests/run_vk_decode_parity.py` | Vulkan-vs-reference parity harness |
| `tests/run_deltanet_chunk_unit.py` | Torch-vs-native chunk-rule regression test |

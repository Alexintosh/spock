# 0013: Native DeltaNet Chunk Rule Primitive

## Goal

Implement the exact HF `torch_chunk_gated_delta_rule` math natively in C++ and
lock it down with a regression test before wiring it into Vulkan prompt
prefill.

## Context

After the RMSNorm and RoPE fixes, the remaining Vulkan-vs-reference failures no
longer looked like random shader drift. For at least one failing prompt,
Vulkan's first generated token matched HF **sequential** prefill with repacked
fp16 weights, but still differed from the frozen **chunk-prefill** reference.

That narrowed the gap to DeltaNet prompt semantics:

- decode uses the recurrent one-token rule
- prompt prefill in HF uses the chunk rule for `seq_len > 1`

Before changing `vk_decode.cpp`, the chunk rule itself needed a native,
verified implementation.

## Implementation

Added a host-side DeltaNet chunk-rule implementation:

| File | Change |
|------|--------|
| `src/runtime/deltanet_chunk.hpp` | New config/input/output types for native chunk execution |
| `src/runtime/deltanet_chunk.cpp` | C++ implementation of the HF torch chunk rule |
| `apps/spock-deltanet-chunk.cpp` | Small CLI to run the native helper against JSON fixtures |
| `tests/run_deltanet_chunk_unit.py` | Torch-vs-native regression test |
| `CMakeLists.txt` | Build the helper CLI and register the new test |

The native helper matches the important HF behavior:

1. Optional Q/K L2 norm with `eps=1e-6`
2. Query scaling by `1 / sqrt(k_dim)`
3. Chunk padding to the requested chunk size
4. Lower-triangular chunk solve used by the DeltaNet scan
5. Final recurrent-state carry across chunks

This is not wired into prompt prefill yet. It is the missing mathematical
primitive needed for that refactor.

## Verification

New regression test:

```text
python3 tests/run_deltanet_chunk_unit.py --runner build/spock-deltanet-chunk
```

Result:

```json
{
  "max_core_attn_out_diff": 1.4901161193847656e-08,
  "max_final_state_diff": 0.0
}
```

That is float32-noise agreement with the torch reference.

## Why This Matters

The previous Vulkan path only had the recurrent DeltaNet update. That was
enough for decode and enough to prove most of the layer math, but it was not
enough to reproduce the model's official prompt-prefill contract.

With the chunk rule implemented natively, the next step is no longer "figure
out the math." The next step is a runtime refactor:

- prefill becomes layer-major
- attention can still run token-sequential within that phase
- DeltaNet layers use the native chunk rule across the prompt sequence
- decode remains the existing recurrent single-token path

## Next Work

1. Refactor `src/runtime/vk_decode.cpp` to separate prompt prefill from decode.
2. Use the native chunk helper for DeltaNet prefill while keeping recurrent
   decode unchanged.
3. Re-run the full 48-prompt parity sweep against the frozen chunk-prefill
   corpus.

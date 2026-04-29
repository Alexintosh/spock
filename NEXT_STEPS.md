# Spock Handoff: Next Steps

## Current State

All 24 layers execute correctly on the RX 6750 XT with numerical parity against a Python fp16 sequential-prefill trace:

- **6 attention layers** (3,7,11,15,19,23): Fixed V-accumulation bug. Output matches Python within 0.006 after 24 layers.
- **18 DeltaNet layers** (0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22): Gated delta rule with causal conv1d, FP32 state, norm+gate. All intermediate values verified.
- **Prefill**: Sequential prefill processes prompt tokens 0..N-2, then decodes token N-1 with argmax.

**Tests**: 12/12 pass. `spock-decode` runs end-to-end.

Important correction: the repo now has an executable Vulkan-vs-reference test,
`spock_vk_decode_reference_parity`. It checks the first frozen prompt against
the trusted HF/repacked reference for 16 generated tokens. The old
`spock_p0_parity` test is only a reference-file structural check; it does not
prove Vulkan parity.

## Critical Bug Fixed

Qwen3.5 RMSNorm uses `output = norm(x) * (1 + weight)`. Vulkan was multiplying
by `weight` directly in `rms_norm.comp` and `rms_norm_per_head.comp`. That made
the first layer diverge immediately and caused token `220` loops. After fixing
the RMSNorm shaders, the first frozen prompt matches the reference:

`[271, 248068, 271, 248069, 271, 89454, 4384, 6813, 513, 16099, 1521, 781, 3300, 264, 14294, 11]`

## Critical Bug Fixed (This Session)

**Attention V-accumulation zeroed elements 4-255 for seq_len < 64**. The `attention_decode.comp` shader split both positions and elements across invocations using the same `lid`. For seq_len=1, only lid=0 entered the position loop, leaving elements 4-255 as zero. Fixed by making all invocations iterate over all positions.

## Parity Status

| Metric | Status |
|--------|--------|
| Single-token Vulkan vs Python fp16 sequential | Match (both produce token 220, top-5 logits identical) |
| Multi-token Vulkan vs Python fp16 sequential | Match (both produce token 220 for 9-token prompt) |
| Vulkan vs fp32 chunk-prefill reference | Different (token 220 vs 271) — expected due to precision and prefill method |

The reference tokens in `tests/data/reference_tokens.jsonl` were generated with fp32 weights and chunk prefill. They don't match Vulkan's fp16 + sequential prefill computation. A new reference is needed for P0 token-level parity.

## Remaining Work

### 1. Expand Vulkan-vs-reference parity coverage

The first frozen prompt passes. Expand `spock_vk_decode_reference_parity` from
`--limit 1` to more prompts as runtime allows, then to the full 48-prompt corpus
on the real RX 6750 XT.

`tools/reference_decode.py` also has a `--sequential-prefill` mode using the HF
model one token at a time, but the frozen reference currently remains the
chunk-prefill HF/repacked corpus in `tests/data/reference_tokens.jsonl`.

### 2. Investigate token-220 loop

The token-220 loop was caused by the RMSNorm multiplier bug for the first tested
prompt. Keep this item open until several prompts pass the executable parity
harness.

### 3. Performance optimization

Current: ~3 command buffer submissions per DeltaNet layer + 1 per attention layer = ~60 submits per token. Reduce by:
- Fusing DeltaNet Submit 1 + Submit 2 (move g/beta computation to GPU)
- Combining multiple layers into a single command buffer
- Reducing barrier granularity

## Architecture

- **Model**: Qwen 3.5-0.8B, 24 layers, hybrid DeltaNet + Attention
- **Attention**: Full multi-head attention with KV cache, QK-norm, mRoPE (rotary_dim=64), Q-gate, GQA (8/2)
- **DeltaNet**: Gated delta rule with FP32 state, causal conv1d (kernel=4), adaptive decay
- **Precision**: fp16 weights, fp16 activations, fp32 DeltaNet state, fp32 accumulate in matmul
- **Constants**: hidden=1024, inter=3584, vocab=248320, q_heads=8, kv_heads=2, head_dim=256, dn_heads=16, dn_k_dim=128, dn_v_dim=128

## Key Files

| File | Purpose |
|------|---------|
| `src/runtime/vk_decode.cpp` | Main decode loop |
| `shaders/attention_decode.comp` | Fixed: V accumulation for all seq_len |
| `shaders/deltanet_recurrent.comp` | FP32 state update |
| `shaders/deltanet_norm_gate.comp` | Norm+gate output |
| `tools/reference_decode.py` | Updated: fp16 weight mode |
| `diary/0011_attention_v_accumulation_bug.md` | This session's diary |

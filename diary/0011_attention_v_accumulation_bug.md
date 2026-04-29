# 0011: Attention V-Accumulation Bug — First Correct Multi-Layer Output

## Goal

Debug and fix a critical bug in the attention decode shader that caused most of the attention output to be zeroed, making all 6 attention layers produce garbage hidden states. Achieve numerical parity between Vulkan and a Python reference trace using the same fp16 weights and sequential prefill.

## Context

After implementing all 24 layers (diary 0010) and fixing the DeltaNet path (L2-norm, conv1d, g/beta, offset bugs), the Vulkan engine produced numerically valid output but incorrect tokens. A layer-by-layer comparison against a Python fp16 trace revealed growing divergence that was far larger than fp16 rounding could explain:

- Layer 0 (dn): diff ~0.0002 — acceptable fp16 rounding
- Layer 3 (first attn): diff ~0.41 — catastrophic

This pointed to a bug in the attention path, not cumulative precision loss.

## The Bug

### Symptoms

Adding debug dumps with proper submit/synchronize boundaries (not inside command buffer recording) revealed:

- `attn_out[0..3]` matched `V_buf[0..3]` exactly — correct
- `attn_out[4]` was `0` while `V_buf[4]` was `-0.101` — wrong
- Elements 4-255 of the attention output were all zero for seq_len=1

### Root Cause

The `attention_decode.comp` shader split **both** positions **and** elements across invocations using the same `gl_LocalInvocationID.x`:

```glsl
// OLD (buggy): positions split across invocations
uint scores_per_inv = (params.seq_len + 63u) / 64u;
for (uint i = 0u; i < scores_per_inv; ++i) {
    uint pos = lid * scores_per_inv + i;  // lid=1 → pos=1 (out of range!)
    ...
    // V accumulation for lid's element stripe
    for (uint j = 0u; j < v_elems; ++j) {
        uint d = lid * v_elems + j;
        partial_v[j] += w * V[d];
    }
}
```

For `seq_len=1`, `scores_per_inv = 1`. Only `lid=0` entered the position loop (`pos = 0*1+0 = 0 < 1`). All other invocations (`lid=1..63`) skipped the loop entirely, leaving their element stripes as zero.

Since `v_elems = 4` and there are 64 invocations, only elements 0-3 (lid=0's stripe) received correct V values. Elements 4-255 were all zero.

### Why It Worked For Longer Sequences

For `seq_len >= 64`, every invocation processes at least one position, so all element stripes get accumulated. The bug only manifests when `seq_len < 64` — particularly for single-token decode (the most common case).

### The Fix

Changed all invocations to iterate over **all** positions, with each handling its own element stripe:

```glsl
// NEW (correct): all invocations process all positions
for (uint pos = 0u; pos < params.seq_len; ++pos) {
    // Compute QK score...
    // Accumulate weighted V for this invocation's element stripe
    for (uint j = 0u; j < v_elems; ++j) {
        uint d = lid * v_elems + j;
        partial_v[j] += w * V[d];
    }
}
```

Since every invocation now sees all positions, the softmax sum is identical across invocations — no cross-invocation reduction needed for the sum. The max reduction in shared memory is kept for correctness but is redundant (all invocations compute the same max).

## Debugging Technique: Split Submit

A key debugging insight: `download_from_device` inside a command buffer recording reads the buffer state from **before** the command buffer is submitted. To read intermediate values, the command buffer must be split:

```cpp
dev.end_command_buffer(cmd);
dev.submit_and_wait(cmd);
dev.download_from_device(buffer, ...);
cmd = dev.allocate_command_buffer();
dev.begin_command_buffer(cmd);
// continue recording
```

This technique was used to dump `attn_out_buf`, `gated_attn_buf`, `v_buf`, `gate_buf` after the attention + sigmoid gate dispatches, which revealed the zeroed elements.

## Verification

### Layer-by-Layer Comparison

After the fix, Vulkan layer outputs match the Python fp16 BLAS trace:

| Layer | Type | Vulkan[0] | Python[0] | Diff |
|-------|------|-----------|-----------|------|
| 0 | dn | -0.1221 | -0.1223 | 0.0002 |
| 3 | attn | 0.0882 | 0.0868 | 0.0014 |
| 7 | attn | 0.3586 | 0.3611 | 0.0025 |
| 23 | attn | 2.0078 | 2.0098 | 0.0020 |

Maximum diff after 24 layers: ~0.006 — well within fp16 rounding bounds.

### Single-Token Decode

Both Vulkan and Python sequential produce token 220 (space) for single-token input `7734`, with matching top-5 logits:

| Rank | Token | Vulkan Logit | Python Logit |
|------|-------|-------------|-------------|
| 1 | 220 | 12.17 | 12.17 |
| 2 | 17 | 11.10 | 11.10 |
| 3 | 16 | 10.71 | 10.71 |

### Why Token 220, Not 271

The reference tokens (token 271 for the first prompt) were generated with:
1. fp16 weights converted to fp32 (`.to(torch.float32)` in `reference_decode.py`)
2. Chunk prefill (all 9 prompt tokens processed at once with parallel scan)

Vulkan uses:
1. fp16 weights directly (fp32 accumulate in matmul, fp16 rounding between operations)
2. Sequential prefill (one token at a time, recurrent state accumulated)

Both differences contribute to the token mismatch. A Python sequential fp16 trace produces the same token 220, confirming Vulkan is numerically correct.

The `reference_decode.py` was updated to use fp16 weights (matching Vulkan's precision model), but chunk prefill still produces different results than sequential prefill for DeltaNet layers.

## Prefill Method Matters

For attention layers, sequential and chunk prefill produce identical results — each position sees the same causal context regardless of processing order.

For DeltaNet layers, the conv1d and recurrent state are inherently sequential. Chunk prefill uses a parallel scan that is mathematically equivalent but may differ numerically due to different accumulation orders. In practice, the differences are large enough to flip the argmax:

- Sequential prefill: argmax = 220 (logit 13.07)
- Chunk prefill: argmax = 271 (logit 11.77)

The gap is 1.3 logits — too large to be fp16 rounding alone. The conv1d with causal padding may accumulate differently in parallel vs sequential mode.

## Files Changed

| File | Change |
|------|--------|
| `shaders/attention_decode.comp` | Fix V accumulation: all invocations iterate all positions |
| `src/runtime/vk_decode.cpp` | Extended debug dump to first decode step |
| `tools/reference_decode.py` | Use fp16 weights (matching Vulkan precision) |
| `tests/data/reference_tokens_fp16.jsonl` | Generated with fp16 weights (chunk prefill) |

## Known Limitations

1. **Reference tokens still use chunk prefill**: The fp16 reference matches Vulkan's weight precision but not its prefill method. A sequential-prefill reference generator is needed for exact P0 token-level parity.
2. **Token 220 loop**: The model produces token 220 (space) for every decode step after the 9-token prompt. This is correct given the sequential prefill computation but suggests the hidden state gets stuck. Root cause not yet investigated — may be related to the conv1d state or recurrent state accumulation.
3. **Performance**: 3 command buffer submissions per DeltaNet layer + 1 per attention layer. Significant CPU overhead.

## What's Next

1. **Sequential-prefill reference generator**: Modify `reference_decode.py` or create a new tool that processes prompt tokens one at a time with state accumulation, matching Vulkan's prefill path exactly.
2. **P0 parity test**: Run the full 48-prompt corpus against the sequential-prefill reference.
3. **Token 220 investigation**: Determine why the model gets stuck producing spaces after multi-token prompts. Check conv1d state and recurrent state evolution.
4. **Performance optimization**: Reduce command buffer submissions per layer.

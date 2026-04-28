# Spock Handoff: Next Steps

## Current State

All 24 layers execute on the RX 6750 XT and produce numerically valid output (no NaN):
- **6 attention layers** (3,7,11,15,19,23): Full multi-head attention with KV cache, QK-norm, mRoPE, GQA, sigmoid gate
- **18 DeltaNet layers** (0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22): Gated delta rule with causal conv1d, FP32 state, norm+gate

Output tokens are real but not yet correct. Known gaps: no prefill, no L2-norm in pipeline, placeholder g/beta values (a_log=0, dt_bias=0).

**Tests**: 11/11 pass. `spock-decode` runs end-to-end at ~2.2s for 3 tokens.

## Key Bug Fixed

**NaN from fp32/fp16 type mismatch**: `delta_norm` weights are stored as fp32 [128] in the repacked artifact, but `deltanet_norm_gate.comp` read them as `float16_t`. The garbage interpretation produced NaN through RMSNorm. Fixed by reading as `float`. This was the root cause of all-zeros output.

## Remaining DeltaNet Fixes (Ordered)

### Fix 1: Move input_norm to DeltaNet Submit 1

**Bug**: DeltaNet projections (Submit 1) read from `act_b` before `input_norm(act_a) → act_b` runs. The `input_norm` is recorded in the main layer command buffer, but Submit 1 executes first. This means projections read uninitialized `act_b` → all-zero output → NaN after norm_gate.

**Fix**: Move `input_norm` dispatch into Submit 1 for DeltaNet layers. Wrap the existing `input_norm` in `cmd` with `if (is_attn)` to skip for DeltaNet.

### Fix 2: Cache a_log and dt_bias per layer

Currently reads from disk via `read_tensor_bytes` for each DeltaNet layer at each decode step. Pre-read all 18 layers' a_log and dt_bias into CPU vectors before the decode loop.

### Fix 3: Use proper g/beta from a_log and dt_bias

Replace placeholder values (a_log=0, dt_bias=0) with cached per-layer values. The g computation: `g = -exp(a_log) * softplus(a + dt_bias)`.

### Fix 4: Wire L2-norm for Q/K

The `l2_norm_per_head.comp` shader exists and descriptor sets `dn_l2_q_ds`/`dn_l2_k_ds` are allocated. Add L2-norm dispatches after conv1d in Submit 1.

### Fix 5: Prefill loop

Restructure the decode loop to process prompt tokens before generating. Current loop processes only `tokens.back()` at each step. Prefill needs to process `tokens[0..N-2]` first.

The prefill loop was implemented but introduced a crash (segfault during the first step's Vulkan submission). Root cause not yet identified — likely related to command buffer state management with the split-submit pattern.

### Fix 6: Zero-initialize state buffers

Device-local buffers (`kv_cache_buf`, `dn_state_buf`, `dn_conv_state_buf`) have undefined content. Need to zero-initialize before first use.

## Architecture Recap

- **Model**: Qwen 3.5-0.8B, 24 layers, hybrid DeltaNet + Attention
- **Attention**: Full multi-head attention with KV cache, QK-norm, mRoPE (rotary_dim=64), Q-gate, GQA (8/2)
- **DeltaNet**: Gated delta rule with FP32 state, causal conv1d (kernel=4), adaptive decay
- **Constants**: hidden=1024, inter=3584, vocab=248320, q_heads=8, kv_heads=2, head_dim=256, dn_heads=16, dn_k_dim=128, dn_v_dim=128

## Key Files

| File | Purpose |
|------|---------|
| `src/runtime/vk_decode.cpp` | Main decode loop |
| `shaders/deltanet_norm_gate.comp` | Fixed: reads norm weight as fp32 |
| `shaders/deltanet_recurrent.comp` | Fixed: Q scale factor |
| `diary/0010_deltanet_decode_path.md` | DeltaNet implementation diary |

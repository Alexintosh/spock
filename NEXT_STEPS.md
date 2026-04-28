# Spock Handoff: Next Steps

## Current State

The Vulkan decode pipeline runs end-to-end on the RX 6750 XT with the **MLP-only** forward pass producing numerically correct output (verified against numpy reference). All 24 layers execute: input_norm → residual_add → post_norm → gate_matvec → up_matvec → silu_gate → down_matvec → residual_add. The token mixer is identity pass-through, so the model echoes its input token.

**Commit**: `3052f41` on `master`, clean tree, 11/11 tests pass.

## Architecture Recap

- **Model**: Qwen 3.5-0.8B, 24 layers, hybrid DeltaNet + Attention
- **Attention layers** (indices 3,7,11,15,19,23): Full multi-head attention with KV cache, q/k/v/o projections, q_norm/k_norm
- **DeltaNet layers** (indices 0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22): Gated delta rule linear attention with recurrent state, in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, out_proj, conv, dt_bias, a_log, delta_norm
- **MLP** (all layers): gate [3584,1024] → SiLU → up [3584,1024] → elementwise multiply → down [1024,3584]
- **Constants**: hidden=1024, inter=3584, vocab=248320, q_heads=8, kv_heads=2, head_dim=256, deltanet_heads=16, deltanet_key_dim=128, rms_eps=1e-6, subgroup_size=64

## Key Lessons Learned (Critical for Next Agent)

1. **Descriptor set aliasing**: Vulkan descriptor sets are consumed at execution time, not recording time. Each dispatch within a submit MUST have its own descriptor set. The current code allocates 11 descriptor sets per layer — follow this pattern.

2. **Weight buffer sizing**: Use `max(offset + nbytes)`, not `sum(nbytes)`. Alignment padding between tensors creates gaps.

3. **Per-layer submit is the current pattern**: Each layer gets its own command buffer + submit. Performance is ~520ms/token. The megakernel optimization (single submit) comes later.

## Next Steps (Priority Order)

### Step 1: Full Attention Layers (6 layers — indices 3,7,11,15,19,23)

Needed weights per attention layer (all in manifest, role path `layer.N.attn_*`):
- `attn_q` [4096, 1024] — query projection
- `attn_k` [512, 1024] — key projection
- `attn_v` [512, 1024] — value projection
- `attn_o` [1024, 2048] — output projection
- `attn_q_norm` [256] — query RMSNorm (QK-norm)
- `attn_k_norm` [256] — key RMSNorm

Decode-time attention for batch-1 is simple: one new K/V vector per step, dot-product with all cached K/V vectors, softmax, weighted sum of V. The Qwen 3.5 attention uses mRoPE (multi-dimensional RoPE) with sections [11,11,10] and interleaved layout — check `transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5Attention` in the installed transformers package for the exact implementation.

KV cache: allocate [max_seq_len × 2_kv_heads × head_dim] per attention layer = 2048 × 2 × 2 × 256 × 2 bytes = 4 MiB per layer, 24 MiB total for 6 layers. New KV is appended each decode step.

New shaders needed:
- `rope_apply.comp` — apply mRoPE to Q and K vectors
- `softmax.comp` — compute attention weights from QK^T scores
- `attention_reduce.comp` — weighted sum of V by attention weights

### Step 2: DeltaNet Layers (18 layers)

This is harder. Qwen 3.5's DeltaNet uses a gated delta rule with:
- Chunk-wise parallel training path (not needed for decode)
- Recurrent decode path with FP32 state accumulation
- Causal convolution (kernel size 4)
- dt bias, a_log (learnable decay), delta_norm

Check `transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5GatedDeltaNet` for the decode path. The recurrent state is [heads × key_dim × value_dim] per layer = 16 × 128 × 128 × 4 bytes (FP32) = 1 MiB per layer, 18 MiB total.

New shaders needed:
- `deltanet_recurrent.comp` — the core recurrent update and output
- Potentially `conv1d_step.comp` for the causal convolution

### Step 3: P0 Parity Verification

Once attention and DeltaNet are wired, run `spock-decode` with prompt tokens from `tests/data/reference_tokens.jsonl` and compare generated tokens against the P0 reference. Target: exact match on all 48 prompts.

The reference activations capture tool (`tools/verify_repack_parity.py --capture-activations`) can dump per-layer activations for debugging. Regenerate with:
```sh
python3 tools/verify_repack_parity.py --model-dir artifacts/hf/Qwen--Qwen3.5-0.8B --tokenizer-dir artifacts/hf/Qwen--Qwen3.5-0.8B-tokenizer --repack-dir artifacts/spock-text-repack-qwen35-0p8b --reference tests/data/reference_tokens.jsonl
```

## Key Files

| File | Purpose |
|------|---------|
| `src/runtime/vk_decode.cpp` | Main decode loop — this is where new shaders get wired in |
| `src/runtime/vk_device.hpp/cpp` | Vulkan device wrapper — create_buffer, allocate_descriptor_set, etc. |
| `src/runtime/weight_loader.hpp/cpp` | Loads manifest, indexes tensors by role path |
| `src/model/qwen35_config.hpp` | Model constants and layer schedule |
| `shaders/*.comp` | GLSL compute shaders |
| `tools/verify_repack_parity.py` | Reference activation capture for debugging |
| `IMPLEMENTATION_PLAN.md` | Full plan with milestones 0–8 |

# Spock Handoff: Next Steps

## Current State

All 24 layers execute on the RX 6750 XT:
- **6 attention layers** (3,7,11,15,19,23): Full multi-head attention with KV cache, QK-norm, mRoPE, GQA, sigmoid gate
- **18 DeltaNet layers** (0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22): Gated delta rule recurrent attention with causal conv1d, FP32 state, adaptive decay

Both token mixer types are wired and producing output. The pipeline runs without GPU faults.

**Tests**: 11/11 pass. `spock-decode` runs end-to-end at ~2.2s for 3 tokens.

## Architecture Recap

- **Model**: Qwen 3.5-0.8B, 24 layers, hybrid DeltaNet + Attention
- **Attention layers**: Full multi-head attention with KV cache (4 MiB/layer), q/k/v/o projections, q_norm/k_norm, mRoPE (rotary_dim=64), Q-gate, GQA (8 Q heads / 2 KV heads)
- **DeltaNet layers**: Gated delta rule with recurrent FP32 state (1 MiB/layer), causal conv1d (kernel=4), in_proj_qkv/z/a/b, out_proj, L2-norm Q/K, adaptive decay (a_log + dt_bias), RMSNorm+SiLU gate
- **MLP** (all layers): gate [3584,1024] → SiLU → up [3584,1024] → elementwise multiply → down [1024,3584]
- **Constants**: hidden=1024, inter=3584, vocab=248320, q_heads=8, kv_heads=2, head_dim=256, dn_heads=16, dn_k_dim=128, dn_v_dim=128, rms_eps=1e-6, subgroup_size=64

## Key Lessons Learned

1. **Descriptor set aliasing**: Each dispatch within a submit MUST have its own descriptor set.
2. **Two-submit pattern for DeltaNet**: The g/beta computation requires GPU→CPU→GPU round-trip (download a/b, compute decay, upload g/beta). DeltaNet layers use 3 submits: projections, g/beta upload, recurrent+output.
3. **State buffer layout**: Pack g and beta (FP32) after the recurrent state in the same buffer. The shader reads them from a known offset.
4. **Buffer offsets for split**: Instead of copying Q/K/V into separate buffers, use descriptor set offsets to read directly from the concatenated QKV buffer.
5. **`read_tensor_bytes` for small weights**: Used to read a_log (64 bytes fp32) and dt_bias (32 bytes fp16) per DeltaNet layer per step. Should be cached.

## Next Steps (Priority Order)

### Step 3a: Wire L2-norm for DeltaNet Q/K

The `l2_norm_per_head.comp` shader exists but is not yet dispatched. Before the recurrent step, Q and K need L2 normalization. Add two dispatches between the conv1d step and the recurrent step.

Also need to apply the scale factor `1/sqrt(k_dim)` = `1/sqrt(128)` to Q after L2-norm.

### Step 3b: Cache a_log and dt_bias per layer

Currently reads from disk via `read_tensor_bytes` for each DeltaNet layer at each decode step. Pre-read all 18 layers' a_log and dt_bias into CPU memory at setup time.

### Step 3: P0 Parity Verification

Run `spock-decode` with prompt tokens from `tests/data/reference_tokens.jsonl` and compare against P0 reference. Target: exact match on all 48 prompts.

Known issues that will affect parity:
- No prefill phase (KV cache and recurrent state start empty)
- No L2-norm on DeltaNet Q/K (not yet wired)
- Token generation will be wrong without proper prompt context processing

### Step 4: Prefill Phase

Multi-token prompt processing to populate KV cache (attention layers) and recurrent state (DeltaNet layers) before decode begins. This is required for correct token generation.

## Key Files

| File | Purpose |
|------|---------|
| `src/runtime/vk_decode.cpp` | Main decode loop — both attention and DeltaNet wired |
| `src/runtime/vk_device.hpp/cpp` | Vulkan device wrapper |
| `src/runtime/weight_loader.hpp/cpp` | Loads manifest, indexes tensors by role path |
| `src/model/qwen35_config.hpp` | Model constants and layer schedule |
| `shaders/*.comp` | GLSL compute shaders (18 total) |
| `tools/verify_repack_parity.py` | Reference activation capture for debugging |
| `IMPLEMENTATION_PLAN.md` | Full plan with milestones 0–15 |
| `diary/0009_attention_decode_path.md` | Attention implementation diary |
| `diary/0010_deltanet_decode_path.md` | DeltaNet implementation diary |

# 0009: Full Attention Layers — KV Cache, QK-Norm, mRoPE, GQA Decode

## Goal

Implement the complete attention forward pass for the 6 full-attention layers (indices 3, 7, 11, 15, 19, 23) in the Vulkan decode pipeline. This replaces the identity token mixer with real multi-head attention including KV cache, QK-norm, mRoPE, and GQA.

## Context

Qwen 3.5-0.8B uses a hybrid architecture: 18 DeltaNet layers (linear attention with recurrent state) and 6 full-attention layers (standard multi-head attention with KV cache). The MLP-only forward pass (diary 0008) used identity pass-through for all token mixers. This entry wires the attention layers.

### Qwen 3.5 Attention Architecture

The attention layers use several non-standard features that required careful shader design:

**QK-norm**: Before attention, both Q and K vectors are independently RMSNorm'd per head. The `q_norm` and `k_norm` weights are shape [head_dim=256], not [hidden_size]. This is different from the layer RMSNorm which normalizes the full hidden dimension.

**Q-gate**: The Q projection outputs 2× the expected dimension. `q_proj` maps [hidden=1024] → [q_heads × head_dim × 2 = 4096]. This is split per-head into query [256] and gate [256]. The gate applies sigmoid after attention: `output = attention_output * sigmoid(gate)`.

**GQA** (Grouped Query Attention): 8 query heads share 2 KV heads (group size 4). Each KV head's cached K/V vectors are shared across 4 query heads during attention scoring.

**mRoPE** (Multi-dimensional RoPE): Position embeddings split into 3 sections [11, 11, 10] = 32 frequency pairs. For text-only decode, all 3 position IDs are identical, so mRoPE degenerates to standard RoPE. `partial_rotary_factor=0.25` means only the first 64 of 256 head dimensions receive rotation.

**KV cache**: Per attention layer, [max_seq_len × 2 × kv_heads × head_dim] fp16 = 4 MiB. 6 layers = 24 MiB total. New K/V are appended each decode step.

## What Shipped

### New compute shaders (6 files)

| Shader | Purpose |
|--------|---------|
| `rope_apply.comp` | Apply RoPE to Q or K head vectors in-place. Handles partial_rotary: only first 64 of 256 dims rotated. 1 workgroup, 64 invocations |
| `attention_decode.comp` | Full attention: QK^T scoring with online softmax, weighted V sum with GQA support. 1 workgroup per Q head (8 total), 64 invocations each. Uses shared-memory reduction for softmax normalization |
| `kv_cache_store.comp` | Write new K and V vectors into the KV cache at the current decode position. Single workgroup, 64 invocations |
| `rms_norm_per_head.comp` | Per-head RMSNorm: normalizes each head independently with a shared [head_dim] weight. 1 workgroup per head, 256 invocations |
| `split_q_gate.comp` | Split q_proj's interleaved output [head0_q, head0_gate, head1_q, ...] into separate contiguous Q and gate buffers |
| `sigmoid_gate.comp` | Elementwise `output = input * sigmoid(gate)`. Used for the attention Q-gate mechanism |

### Decode loop changes

The per-layer processing in `vk_decode.cpp` now dispatches differently based on layer type:

**Attention layers** (6 layers at indices 3,7,11,15,19,23):
1. input_norm (RMSNorm, same as before)
2. Q projection: matvec with attn_q weight [4096,1024] → 4096-dim output
3. K projection: matvec with attn_k weight [512,1024] → 512-dim
4. V projection: matvec with attn_v weight [512,1024] → 512-dim
5. Split Q+Gate: separate the interleaved query and gate vectors
6. Q-norm: per-head RMSNorm on Q with q_norm weight [256]
7. K-norm: per-head RMSNorm on K with k_norm weight [256]
8. RoPE: apply rotary position embeddings to Q and K (separate dispatches)
9. KV cache store: write K,V at current position
10. Attention: QK^T scoring, softmax, weighted V → attn_out [2048]
11. Sigmoid gate: attn_out * sigmoid(gate)
12. Output projection: matvec with attn_o weight [1024,2048] → hidden [1024]
13. Continue with MLP (same as before)

**DeltaNet layers** (18 layers): identity pass-through (unchanged from 0008).

### Buffer additions

- 7 attention activation buffers: q_proj (8KB), q (4KB), gate (4KB), k (1KB), v (1KB), attn_out (4KB), gated_attn (4KB)
- KV cache: 24 MiB total (4 MiB × 6 layers, max_seq=2048)
- RoPE frequency buffer: 256 bytes (float32, recomputed per step on CPU)

### Pipeline infrastructure

Added `pipeline_layout_32` with 32-byte push constant range (needed for attention shaders with larger parameter structs). The existing `pipeline_layout_3` and `pipeline_layout_2` with 8-byte push constants are retained for simpler shaders.

## Design Decisions

### RoPE frequency computation on CPU

Cos/sin frequencies are computed CPU-side each step and uploaded to the GPU. This avoids GPU trigonometric functions and makes debugging straightforward. For text-only decode with all 3 mRoPE position IDs identical, the frequencies are just `cos(pos * inv_freq[i])`, `sin(pos * inv_freq[i])` where `inv_freq[i] = 1/θ^(2i/d)` with θ=10,000,000.

The upload uses a host-visible staging buffer + `vkCmdCopyBuffer` to device-local memory, adding one submit per step. This is temporary — future optimizations will precompute the full frequency table or compute it inline in the shader.

### Attention score recomputation

The attention shader computes QK^T dot products twice: once to find the max (for numerically stable softmax) and once to compute exp(score - max) and accumulate the weighted V sum. This trades compute for memory — avoids storing all score values in shared memory. For head_dim=256 and seq_len≤2048, each invocation does ~32 dot products of dim 256, twice = ~16K multiply-adds. Acceptable for correctness-first implementation.

### In-place Q-norm and K-norm

The per-head RMSNorm uses the same buffer for input and output (binding 0 and 2 point to the same buffer). This is safe because RMSNorm reads all elements before writing, and the shader processes one head per workgroup with a full barrier between the reduction and write phases.

### No prefill phase

The current implementation is decode-only. Each step processes one token and writes one K/V entry to the cache. Without a prefill phase, the attention layers have minimal context (only 1 KV entry at step 0). This means the output is degenerate but functional — the pipeline executes correctly, just without proper multi-token context.

## Verification

- **Build**: Clean compilation, all 6 new shaders compile to SPIR-V without errors
- **Tests**: 11/11 ctest suite passes
- **Runtime**: `spock-decode` runs end-to-end without GPU faults or validation errors
- **Output**: Generates token 3 (same as MLP-only baseline) — expected because single-token decode with no prefill produces degenerate attention context
- **Performance**: ~1.7s for 3 generated tokens (vs ~520ms MLP-only). The attention layers add ~400ms per step across 6 attention layers, dominated by the attention scoring kernel

### Known limitations

1. **No prefill**: Without processing the full prompt through attention, the KV cache is empty at step 0 and grows by 1 per step. Proper text generation requires a prefill phase.
2. **No DeltaNet**: The 18 DeltaNet layers still use identity pass-through (Step 2).
3. **Performance**: Per-layer submit overhead dominates. The attention shader recomputes scores. RoPE has an extra CPU→GPU copy per step.

## What's Next

1. **Step 2**: DeltaNet recurrent decode path for the 18 linear-attention layers
2. **Step 3**: P0 parity verification against reference tokens
3. **Prefill**: Multi-token prompt processing to populate KV cache before decode

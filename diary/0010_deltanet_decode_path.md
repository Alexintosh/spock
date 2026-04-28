# 0010: DeltaNet Recurrent Decode — All 24 Layers Active

## Goal

Implement the DeltaNet recurrent decode path for the 18 linear-attention layers (indices 0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22). This completes the token mixer for all 24 layers: 6 attention layers + 18 DeltaNet layers.

## Context

Qwen 3.5-0.8B's DeltaNet uses a gated delta rule — a form of linear attention with a recurrent state that is updated each decode step. Unlike standard attention (which stores all past K/V in a cache), DeltaNet maintains a fixed-size recurrent state `[heads × key_dim × value_dim]` that is incrementally updated. This makes it O(1) per step regardless of sequence length, at the cost of being an approximation.

### Key DeltaNet Concepts

**Gated Delta Rule**: The recurrent update follows:
```
state = state * exp(g)               // decay (forget)
kv_mem = state^T @ key               // read: what does state know about key direction?
delta = (value - kv_mem) * beta       // correction: what's new?
state += key ⊗ delta                  // update: write new information
output = state^T @ query              // read: retrieve from updated state
```

**Causal Conv1d**: Before the recurrent step, the QKV vectors pass through a depthwise causal conv1d (kernel_size=4) with SiLU activation. In decode mode, this is a rolling buffer update — shift left by 1, append new value, dot product with filter weights.

**L2 Normalization**: Query and Key are L2-normalized per head before the recurrent step. This is different from the attention layers which use RMSNorm for QK-norm.

**Gated Output**: After the recurrent step, the output passes through RMSNorm × SiLU(gate). The gate (z) comes from a separate projection of the input.

**Decay Computation**: `g = -exp(A_log) * softplus(a + dt_bias)` where A_log and dt_bias are learned parameters, and `a` is projected from the input at each step. This gives per-head, per-step adaptive decay rates.

## What Shipped

### New compute shaders (4 files)

| Shader | Purpose |
|--------|---------|
| `deltanet_recurrent.comp` | Core recurrent update: decay state, compute kv_mem, delta correction, outer product update, query readout. 1 workgroup per head (16), 128 invocations. FP32 state, FP16 I/O |
| `conv1d_step.comp` | Depthwise causal conv1d for decode: shift rolling buffer, apply filter, SiLU activation. 1 workgroup, 256 invocations |
| `deltanet_norm_gate.comp` | Per-head RMSNorm × SiLU(gate) on recurrent output. 1 workgroup per head, 256 invocations |
| `l2_norm_per_head.comp` | L2 normalization per head for Q and K before recurrent step. 1 workgroup per head, 256 invocations |

### Decode loop changes

DeltaNet layers use a **two-submit pattern** (unlike attention which uses a single submit):

**Submit 1** (separate command buffer): Projections + conv1d
1. `in_proj_qkv(act_b)` → [6144] (Q, K, V concatenated)
2. `in_proj_z(act_b)` → [2048] (gate)
3. `in_proj_a(act_b)` → [16] (decay input)
4. `in_proj_b(act_b)` → [16] (beta input)
5. `conv1d_step(qkv, conv_state, weights)` → updated qkv + conv_state

**CPU bridge**: Download a/b from GPU. Read a_log and dt_bias from weight artifact. Compute g/beta. Upload to state buffer.

**Submit 2** (main layer command buffer): Recurrent + output
6. `deltanet_recurrent(q, kv, state)` → output in V section of qkv buffer
7. `deltanet_norm_gate(output, z, norm_weight)` → gated output
8. `out_proj(output)` → act_b [1024]

### Buffer additions

- 7 DeltaNet activation buffers: qkv (12KB), z (4KB), a (32B), b (32B), q (4KB), kv_out (8KB)
- Recurrent state: 18 MiB total (1 MiB × 18 layers, FP32, includes g/beta tail per layer)
- Conv state: 864 KiB total (49 KiB × 18 layers, FP16 rolling buffers)

## Design Decisions

### Two-submit pattern for DeltaNet

The g/beta computation requires reading the `a` and `b` projection outputs from GPU, then computing decay/gating values on CPU using the weight artifact's `a_log` and `dt_bias` parameters. This forces a GPU→CPU→GPU round-trip in the middle of each DeltaNet layer.

Alternative considered: compute g/beta entirely on GPU using a separate shader. This would avoid the round-trip but require additional shader complexity for reading a_log/dt_bias from the weight buffer. The CPU approach is simpler and correctness-first.

Cost: each DeltaNet layer has 3 submits instead of 1 (projection submit, g/beta upload submit, main layer submit). This adds latency (~36 extra submits per decode step for 18 layers).

### State buffer layout

The recurrent state buffer is organized as:
```
[layer0_state (1 MiB)] [layer0_g_beta (128 B)] [layer1_state] [layer1_g_beta] ...
```

Each layer's g/beta values are packed immediately after its state in FP32. The shader reads them from the tail of its assigned region via `state_buf.data[state_total + head]`.

### Q/K read directly from qkv buffer

Instead of copying Q and K into separate buffers, the recurrent shader reads them directly from the qkv buffer using descriptor set offsets. Q is at offset 0 [2048 elements], K is at offset 2048 [2048 elements], V is at offset 4096 [2048 elements]. Output overwrites the V section.

## Verification

- **Build**: Clean compilation, all 4 new shaders compile to SPIR-V
- **Tests**: 11/11 ctest suite passes
- **Runtime**: `spock-decode` runs end-to-end without GPU faults, 2.2s for 3 tokens
- **Output**: Token 0 generated (vs token 3 for MLP-only and attention-only). The output is not yet correct (no prefill, L2-norm not yet applied to Q/K in the pipeline), but the computation runs through all 24 layers without errors

### Known limitations

1. **No L2-norm applied to Q/K**: The recurrent shader reads Q and K directly from the conv1d output without L2 normalization. This affects the magnitude of the dot products in the recurrent state update. The L2-norm shader exists but is not yet wired in.
2. **No prefill**: Same as attention — no prompt context processing.
3. **Performance**: 3 submits per DeltaNet layer adds significant overhead. The g/beta computation should eventually move to GPU.
4. **a_log and dt_bias read from disk**: `read_tensor_bytes` reads from the weight file on disk for each DeltaNet layer at each decode step. This should be cached.

## What's Next

1. Wire L2-norm into DeltaNet pipeline (Q and K before recurrent)
2. Cache a_log and dt_bias per layer (avoid disk reads per step)
3. Step 3: P0 parity verification against reference tokens
4. Prefill phase for proper multi-token context

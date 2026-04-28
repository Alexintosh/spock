# 0008: MLP Decode Path — First Correct GPU Forward Pass

## Goal

Wire the full per-layer MLP forward pass on the GPU. Each of the 24 layers now runs: input RMSNorm → residual add → post RMSNorm → gate projection → up projection → SiLU gating → down projection → residual add. The token mixer (attention/DeltaNet) is identity pass-through.

This is the first phase where the GPU produces numerically correct output, verified against a numpy reference.

## What Shipped

### New compute shaders

- `shaders/silu_gate.comp` — Elementwise `SiLU(gate) * up`, workgroup 256, single dispatch
- `shaders/residual_add.comp` — Elementwise `a + b`, workgroup 256, single dispatch

### Per-layer decode loop

Each layer runs 9 dispatches in a single command buffer:
1. `input_norm(hidden) → normed` (RMSNorm)
2. `[Identity token mixer]`
3. `residual_add(hidden, normed) → first_residual` 
4. `post_norm(first_residual) → normed2` (RMSNorm)
5. `gate_matvec(normed2) → gate_buf` (3584×1024 matvec)
6. `up_matvec(normed2) → up_buf` (3584×1024 matvec)
7. `silu_gate(gate_buf, up_buf) → silu_buf` (elementwise)
8. `down_matvec(silw_buf) → down_out` (1024×3584 matvec)
9. `residual_add(first_residual, down_out) → layer_output`

Buffer layout: 3 HIDDEN buffers (act_a, act_b, act_c), 3 INTER buffers (mlp_gate, mlp_up, mlp_silu). Total ~27 KB of GPU memory for activations.

## Bugs Found and Fixed

### Descriptor set aliasing

The first attempt reused descriptor sets within a single command buffer. Vulkan descriptor sets are consumed at execution time, not recording time. If you update a descriptor set and rebind it within one submit, all dispatches see the last-written state.

Symptom: all dispatches in a layer used the weights from the last `vkUpdateDescriptorSets` call, producing garbage output.

Fix: allocate one descriptor set per dispatch (11 total: input_norm, residual1, post_norm, gate, up, silu_gate, down, residual2, final_norm, lm_head, argmax, plus embedding). Pre-configure the static ones once; update per-layer ones with weight offsets before each layer's command buffer.

### Weight buffer undersized by alignment padding

`WeightArtifact::total_bytes_` was accumulated as `sum(tensor.nbytes)`. Due to 256-byte alignment padding between tensors, the actual binary file was 7488 bytes larger. The GPU buffer was allocated too small, causing the last tensors to read past the end.

Fix: `total_bytes_ = max(offset + nbytes)` instead of `sum(nbytes)`.

Symptom: `radv: GPUVM fault at address 0x800159f16000` with `PERMISSION_FAULTS: 3`.

## Verification

The MLP-only path was verified against a numpy reference:

- Token 0 (last prompt token) → produces token 0 via argmax ✓
- Token 9419 → produces token 9419 via argmax ✓
- Intermediate activations match layer-by-layer ✓

Without attention, the model echoes the input token — this is expected. The MLP contributions are non-trivial (hidden norm grows from 0.72 to ~1094 over 24 layers) but the residual structure means the input dominates.

## Performance

Single-token decode: ~2.1s for 4 tokens on RX 6750 XT (with per-layer submit). This is the `single_submit` baseline — each layer is a separate command buffer submit. The megakernel path (one submit for all layers) will be ~10-20x faster once descriptor set aliasing is resolved via push-constant-based weight offset selection or dynamic descriptor sets.

## What's Next

1. Wire full attention layers (6 of 24) with KV cache
2. Wire DeltaNet layers (18 of 24) with recurrent state
3. Compare against P0 reference tokens for exact parity

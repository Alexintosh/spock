# DeltaNet Decode Bug Progress

## Resolved Bugs

### 1. Conv1d "wrong output" — dump timing bug
The "after_conv1d" debug dump was reading dn_qkv_buf AFTER the L2-norm Q/K dispatches had already overwritten it. The conv1d was always correct. Fixed by inserting debug split between conv1d and L2-norm.

### 2. Norm_gate and output projection reading K instead of V
`dn_norm_gate_ds` and `dn_out_proj_ds` binding 1 used offset `DN_KEY_TOTAL * 2 = 4096` (K section start). Should be `(DN_KEY_TOTAL + DN_KEY_TOTAL) * 2 = 8192` (V section start). Fixed.

### 3. rms_norm_per_head.comp `(1+w)` bug
Line 72: `v * norm_factor * (1.0 + w)` → `v * norm_factor * w`. Affects attention Q/K norm at layers 3,7,11,15,19,23.

## Root Cause of Remaining Parity Failure

**fp16 precision in recurrent step, amplified by RMSNorm on near-zero first-token output.**

### Mechanism
1. First decode step: recurrent state is all zeros → output = β·(q·k)·v ≈ 1e-8 to 1e-7
2. fp16 quantization in Vulkan causes ~2x differences at this scale (e.g., 0 vs 2.16e-8)
3. RMSNorm normalizes near-zero input to unit RMS → amplifies 1e-8 input to ~1.0
4. Small fp16 differences get amplified by 1e7-1e8x
5. Gated output differs by 5-50x between Python (fp32) and Vulkan (fp16)
6. After output projection (1024×2048 weight matrix), differences compound to wrong logits

### Evidence
- Recurrent output head 0: Python [2.16e-08, -7.05e-08, ...] vs Vulkan [0, -5.96e-08, ...] — similar
- After RMSNorm+gate head 0: Python [1.13e-05, -3.37e-05, ...] vs Vulkan [0, 2.03e-06, ...] — 5-50x different
- Recurrent output head 1: Python [7.70e-06, ...] vs Vulkan [7.69e-06, ...] — matches! (larger values, less precision impact)
- Top logit: Python 264=16.0 vs Vulkan 220=8.24

### Key Insight
The problem is specific to the first decode step (zero state). After a few tokens, the recurrent state accumulates and outputs become O(1), making fp16 precision sufficient. The RMSNorm amplification of near-zero values is the critical failure mode.

## Possible Fixes

### A. Keep recurrent output in fp32 longer (recommended)
- Write recurrent output to a separate fp32 buffer instead of fp16 dn_qkv_buf
- Apply norm_gate in fp32
- Convert to fp16 only after gating
- Pros: Eliminates precision loss at the critical point
- Cons: Extra buffer, extra conversion step, norm_gate shader needs fp32 output

### B. Skip RMSNorm when input norm is below threshold
- In norm_gate shader: if `sum_sq < threshold`, skip RMSNorm (output = input directly)
- Pros: Simple change, avoids the amplification
- Cons: Changes semantics, may affect convergence for later tokens

### C. Use fp32 activations throughout DeltaNet path
- All intermediate buffers (Q, K, V, recurrent output) in fp32
- Pros: Maximum precision
- Cons: 2x memory, slower (more bandwidth)

### D. Fused recurrent+norm+gate shader
- Single shader that does recurrent step + RMSNorm + gate without intermediate fp16 write
- Pros: No intermediate precision loss
- Cons: Complex shader, harder to debug

## Current State
- All intermediate values match Python through conv1d
- Recurrent step is correct (head-to-head comparison matches Python in fp32)
- Norm_gate offset fixed (reads V instead of K)
- rms_norm_per_head fix committed
- **Attention V-accumulation bug found and fixed** (diary 0011)
- The "fp16 precision amplification" theory was wrong — the real issue was zeroed attention output
- Vulkan now matches Python fp16 sequential trace within 0.006 after 24 layers
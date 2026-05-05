# 0043: Fused DeltaNet Recurrent+Norm/Gate Decode Sub-Block

## Goal

Fuse the DeltaNet g/beta computation, recurrent update/output, RMSNorm, and
SiLU(z) gate into a single Vulkan dispatch. This removes two dispatches per
DeltaNet layer on the merged decode path by replacing the
`deltanet_recurrent_gbeta` dispatch plus the separate `deltanet_norm_gate`
dispatch with `deltanet_recurrent_gbeta_norm_gate.comp` when
`SPOCK_GPU_FUSED_DN_REC_NORM_GATE=1` is active.

Default inference is unchanged. The fused path is opt-in and only participates
inside the existing merged DeltaNet decode path.

## Background

After diary 0042, the DeltaNet decode path had two fused sub-blocks:

1. conv+L2 norm (diary 0041)
2. g/beta + recurrent (diary 0042)

The remaining gap between the recurrent output and out-projection was the
standalone `deltanet_norm_gate.comp` dispatch. In the unfused path, after the
recurrent shader writes its output to the V section of `dn_qkv`, the norm+gate
shader reads that output, computes RMSNorm, multiplies by the SiLU of the gate
input `dn_z`, and writes the result back. That is an output-to-norm
round-trip---the recurrent workgroup already holds the per-head output in
registers and shared memory before writing it out. It can apply the norm and
gate directly, skipping the intermediate write/read and the barrier between the
two dispatches.

This is the third and widest DeltaNet decode fusion slice: it absorbs g/beta
computation, the recurrent state update, the recurrent output, and the
post-recurrent norm+gate. The megakernel direction wants as much DeltaNet
per-head work as possible in one dispatch; this is the natural next boundary.

## Implementation Work Completed

### New shader: `shaders/deltanet_recurrent_gbeta_norm_gate.comp`

The new shader combines:

- g/beta scalar computation from `deltanet_compute_g_beta.comp`
- recurrent update and output from `deltanet_recurrent.comp`
- RMSNorm + SiLU(z) gate from `deltanet_norm_gate.comp`

It uses one workgroup per DeltaNet head, 128 invocations per workgroup. Each
workgroup:

1. reads its head's `dn_a` and `dn_b` fp16 values
2. reads the matching fp32 `a_log` and `dt_bias` pair from the packed table
3. computes `g = -exp(a_log) * softplus(a + dt_bias)`
4. computes `beta = sigmoid(b)`
5. loads Q/K/V into shared memory
6. decays the recurrent state by `exp(g)`
7. computes `kv_mem`, `delta`, updates state, and computes `raw_out = state^T @ query`
8. computes `norm = RMSNorm(raw_out)` over the head's v_dim elements using the
   delta_norm weight (binding 7) and the epsilon from push constants
9. computes the SiLU of the gate input `dn_z` (binding 6) and multiplies:
   `output = fp16(norm * weight) * SiLU(z)`
10. writes the gated output to the V section of the KV/output buffer

The shader uses an eight-binding descriptor layout:

| Binding | Buffer |
|---------|--------|
| 0 | `dn_a` fp16 |
| 1 | `dn_b` fp16 |
| 2 | packed `dn_a_log_bias` fp32 |
| 3 | Q slice of `dn_qkv` |
| 4 | K/V slice of `dn_qkv`, with gated output overwriting V |
| 5 | current DeltaNet layer recurrent state |
| 6 | `dn_z` fp16 gate input |
| 7 | `delta_norm` weight fp32 |

Push constants: `num_heads`, `k_dim`, `v_dim`, `state_total`, `q_scale_bits`,
`layer_idx`, `epsilon_bits` (7 uint32 = 28 bytes).

### Runtime gate: `SPOCK_GPU_FUSED_DN_REC_NORM_GATE=1`

The new path is default-off and requires the merged DeltaNet command-buffer
path to be active. The gate is considered only when
`SPOCK_GPU_MERGED_DELTANET=1` is active and diagnostics that disable merged
DeltaNet recording are not active.

When the gate is enabled:

- decode skips the standalone `deltanet_compute_g_beta` dispatch (like diary
  0042)
- decode skips the standalone `deltanet_norm_gate` dispatch (new in this
  slice)
- decode binds `deltanet_recurrent_gbeta_norm_gate` in place of both the
  recurrent and norm+gate dispatches
- the barrier between recurrent and norm+gate is eliminated

When the gate is disabled, the original g/beta, recurrent, and norm+gate
dispatches remain unchanged. When the diary 0042 fused g/beta+recurrent gate
(`SPOCK_GPU_FUSED_DN_GBETA_RECURRENT=1`) is active but this gate is not, the
0042 path applies as before.

### Descriptor and pipeline wiring

The runtime now owns:

- `ds_layout_8`
- `pipeline_layout_8_32`
- `deltanet_recurrent_gbeta_norm_gate`
- `deltanet_recurrent_gbeta_norm_gate_module`
- shared descriptor set `dn_recurrent_gbeta_norm_gate`
- per-layer descriptor vector `dn_recurrent_gbeta_norm_gate`

The per-layer descriptor set is pre-bound for DeltaNet layers so the gate
composes with `SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1` and
`SPOCK_GPU_SINGLE_SUBMIT=1`. This raises the opt-in fused per-layer descriptor
coverage from 31 sets per layer (diary 0042) to 32 sets per layer. With the
checked-in model configuration, that is 32 x 24 = 768 per-layer sets plus 2
session-level RoPE sets.

Bindings 5 (`dn_state`) and 7 (`delta_norm` weight) vary per layer and are
updated per-layer in `decode()` when per-layer sets are not active. When
per-layer sets are active, all eight bindings are pre-bound at construction
time.

### Dispatch selection priority

In the merged DeltaNet decode loop, the dispatch selection now follows:

1. If `fused_dn_rec_norm_gate`: one dispatch of the new shader, skip g/beta
   and norm+gate
2. Else if `fused_dn_gbeta_recurrent`: one dispatch of the diary 0042 shader,
   skip g/beta only
3. Else: separate g/beta + recurrent dispatches, then norm+gate dispatch

Each step in this chain strictly widens the fused boundary.

## Verification

All verification was run locally against the repacked fp16 artifact.

### Whitespace and build

```sh
git diff --check
cmake --build build -j
```

Both passed. The build compiled `deltanet_recurrent_gbeta_norm_gate.comp.spv`
and linked `spock-decode`.

### New fused recurrent+norm_gate gate

```sh
SPOCK_GPU_FUSED_DN_REC_NORM_GATE=1 \
SPOCK_GPU_MERGED_DELTANET=1 \
SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1 \
  python3 tests/run_vk_decode_parity.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --reference tests/data/reference_tokens_fp16.jsonl \
  --ids short_correctness_001 \
  --max-new-tokens 16
```

Result: parity OK, checked 1 prompt, 0 failures.

### Full fused short parity

```sh
SPOCK_GPU_FUSED_DN_REC_NORM_GATE=1 \
SPOCK_GPU_FUSED_DN_GBETA_RECURRENT=1 \
SPOCK_GPU_FUSED_DN_CONV_L2=1 \
SPOCK_GPU_MERGED_DELTANET=1 \
SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1 \
SPOCK_GPU_SINGLE_SUBMIT=1 \
SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1 \
SPOCK_GPU_DEFER_TOKEN_DOWNLOAD=1 \
  python3 tests/run_vk_decode_parity.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --reference tests/data/reference_tokens_fp16.jsonl \
  --ids short_correctness_001 \
  --max-new-tokens 16
```

Result: parity OK, checked 1 prompt, 0 failures.

### Full combined gates on mixed prompts with chunk-prefill

```sh
SPOCK_GPU_FUSED_DN_REC_NORM_GATE=1 \
SPOCK_GPU_FUSED_DN_GBETA_RECURRENT=1 \
SPOCK_GPU_FUSED_DN_CONV_L2=1 \
SPOCK_GPU_MERGED_DELTANET=1 \
SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1 \
SPOCK_GPU_SINGLE_SUBMIT=1 \
SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1 \
SPOCK_GPU_DEFER_TOKEN_DOWNLOAD=1 \
SPOCK_GPU_CHUNK_PREFILL=1 \
SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1 \
SPOCK_GPU_CHUNK_PREFILL_TILED=1 \
  python3 tests/run_vk_decode_parity.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --reference tests/data/reference_tokens_fp16.jsonl \
  --ids mixed_correctness_023,pp520_046 \
  --max-new-tokens 4
```

Result: parity OK, checked 2 prompts, 0 failures.

### CTest chunk-prefill subset

```sh
SPOCK_GPU_FUSED_DN_REC_NORM_GATE=1 \
SPOCK_GPU_FUSED_DN_GBETA_RECURRENT=1 \
SPOCK_GPU_FUSED_DN_CONV_L2=1 \
SPOCK_GPU_MERGED_DELTANET=1 \
SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1 \
  ctest --test-dir build \
  -R "spock_vk_decode_gpu_collect_chunk_prefill_(short|tiled|short_baseline)" \
  --output-on-failure
```

Result: 3/3 passed.

### GPU timestamp sample

A single GPU timestamp sample with all fused gates active recorded
`gpu_decode_us` of approximately 2.49761e+06. The prior diary 0042 two-fusion
sample was approximately 2.49461e+06. These are single samples, not a
benchmarked comparison, so no speedup or regression claim is made.

## Known Limitations

- **Not full GPU offload.** The host still records and submits command buffers.
- **Not persistent dispatch.** There is still no resident GPU scheduler loop.
- **Not the megakernel.** This is a third narrow DeltaNet decode fusion slice,
  not a full layer or full-model fusion.
- **Default-off.** The unfused g/beta, recurrent, and norm+gate dispatches
  remain the default runtime path.
- **Decode only.** Prefill still uses the chunk-prefill path.
- **No standalone performance claim.** Correctness is established. A single
  timestamp sample does not support a throughput conclusion.

## Next Work

The next step is to add finer block-level GPU timestamp instrumentation to
quantify where per-token GPU time is going under full fusion. If the
recurrent+norm+gate dispatch is no longer the bottleneck, the next fusion
boundary should move outward to combine the DeltaNet output with out projection,
or absorb the conv+L2 sub-block into a single end-to-end DeltaNet decode
dispatch.

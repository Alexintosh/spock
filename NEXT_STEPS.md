# Spock Handoff: Next Steps

## Current State

The branch has a working layer-major prefill using the recurrent DeltaNet path.
Full 48-prompt parity test passes. The layer-major restructuring is complete and verified.

The **GPU-collected + tiled no-compare fast path** (diaries 0025/0026) now
eliminates all CPU data touches for chunk-prefill compute. Diary 0025 removed
the CPU readback/re-upload bridge for chunk-prefill output (device-local
`out_buf`, GPU-to-GPU state copy, `deltanet_chunk_last_to_fp16.comp` shader,
device-local `correct_last_token_hidden()` handoff). Diary 0026 removes the
last CPU data touch — init state zero-fill — by replacing host-visible
`init_buf` + CPU `memset` with device-local `init_buf` + `vkCmdFillBuffer`.
The host still orchestrates (per-layer iteration, command recording,
submission, fence wait) but does not read or write chunk-prefill compute
data on the no-compare GPU-collected+tiled path.

### What Works
- **Opt-in device-resident decode token embedding** (diary 0027): Behind
  `SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1`, the per-step embedding lookup reads
  token_id directly from device-local `argmax_result` instead of a
  CPU push constant. This removes the CPU token value as the source for
  the next embedding. CPU still downloads `argmax_result` each decode
  step for external output/parity — this is NOT full GPU offload and NOT
  the megakernel. New shader `embedding_lookup_from_buffer.comp`.
  Verified parity on `short_correctness_001` (16 tokens),
  `mixed_correctness_023`/`pp520_046` (4 tokens), and combined with full
  GPU chunk-prefill gate suite. Still env-gated, not default.

- **Opt-in deferred generated-token download** (diary 0028): Behind
  `SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1 SPOCK_GPU_DEFER_TOKEN_DOWNLOAD=1`,
  the per-step CPU download of `argmax_result` is replaced by a
  device-local `vkCmdCopyBuffer` into a `gen_tokens` buffer and a single
  batch download after the decode loop. Disabled when
  `verbose`/`debug_dump`/`diagnose_decode_drift` is active. Guards
  `max_new_tokens > 0` to avoid zero-sized Vulkan buffer allocation;
  zero-token parity now passes. Default behavior remains per-step download;
  the gate is disabled for `verbose`, `debug_dump`, and
  `diagnose_decode_drift`. Does not restructure the submit-wait loop — no
  performance speedup claimed. Still env-gated, not default.

- **Opt-in per-layer stable descriptor sets** (diary 0029): Behind
  `SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1`, per-layer descriptor mutation
  in `decode()` is eliminated by pre-allocating and pre-binding
  layer-specific descriptor sets at session construction time. Covers
  common MLP/norm descriptors (input_norm, residual, post_norm, gate, up,
  down, MLP residual paths), attention descriptors (Q/K/V projections,
  QK-norm, KV store, attention decode, O projection, attention residual
  paths), and first-stage DeltaNet descriptors (QKV/Z/A/B projections,
  conv1d). RoPE descriptors (D.rope, D.rope_k) are now pre-bound once at
  session construction (diary 0031); per-step position is communicated via
  push constant freq_offset. Intra-DeltaNet sub-step descriptors are now
  covered for L2-norm (dn_l2_q, dn_l2_k, diary 0032) and g/beta
  computation (dn_compute_g_beta, diary 0034), the recurrent step
  (dn_recurrent, diary 0035), the norm/gate step
  (dn_norm_gate, diary 0036), and the output projection
  (dn_out_proj, diary 0037); all DeltaNet dispatch-target sub-step
  descriptors are now covered. Increases descriptor pool capacity
  from 192 to 1024 maxSets and 192 to 4096 storage buffer slots. Default
  unchanged. This is NOT full GPU offload, NOT single-submit, and NOT the
  megakernel. Prerequisite for future single-submit recording. Verified
  parity on `short_correctness_001` (16 tokens),
  `mixed_correctness_023`/`pp520_046` (4 tokens), and combined with full
  GPU chunk-prefill gate suite and device-resident token + deferred
  download gates. Still env-gated, not default. No performance speedup
  claimed.

- **Single-submit decode** (`SPOCK_GPU_SINGLE_SUBMIT=1`, diary 0039):
  opt-in decode fast path records all dispatches for a decode step
  (embedding lookup + 24 layers + final norm + LM head + argmax) into
  one Vulkan command buffer and submits once per token, reducing host
  orchestration from 26 submit/wait round-trips per decode step to 1.
  Requires `SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1` and
  `SPOCK_GPU_MERGED_DELTANET=1`. Disabled for prefill steps,
  `skip_layers` steps (first decode step after chunk prefill), and any
  diagnostic/dump/verbose mode. Default unchanged. This is NOT
  persistent dispatch and NOT the megakernel. Verified parity on
  `short_correctness_001` (16 tokens),
  `mixed_correctness_023`/`pp520_046` (4 tokens), combined with all
  existing GPU gates (per-layer descriptors, merged DeltaNet,
  device-resident token, deferred download), and chunk-prefill
  CTest 3/3. Still env-gated, not default.

- **Fused DeltaNet conv+L2 decode sub-block**
  (`SPOCK_GPU_FUSED_DN_CONV_L2=1`, diary 0041): opt-in decode shader
  fusion that replaces `conv1d_step` + L2 Q + L2 K with one dispatch in
  the merged DeltaNet path. Requires `SPOCK_GPU_MERGED_DELTANET=1`.
  The default path remains unchanged, and this is still not full GPU
  offload, not persistent dispatch, and not the megakernel. Verified with
  short parity, combined single-submit/device-resident/deferred gates,
  mixed correctness prompts, and the chunk-prefill CTest subset.

- **Fused DeltaNet g/beta + recurrent decode sub-block**
  (`SPOCK_GPU_FUSED_DN_GBETA_RECURRENT=1`, diary 0042): opt-in decode
  shader fusion that computes g/beta inside the recurrent DeltaNet update,
  replacing `deltanet_compute_g_beta` + `deltanet_recurrent` with one
  dispatch in the merged DeltaNet path. Requires
  `SPOCK_GPU_MERGED_DELTANET=1`. Default inference is unchanged; this is
  still not full GPU offload, persistent dispatch, or the megakernel.
  Verified with short parity, combined fused conv+L2/single-submit/
  device-resident/deferred gates, mixed correctness prompts, and the
  chunk-prefill CTest subset.

- **Fused DeltaNet recurrent + norm_gate decode sub-block**
  (`SPOCK_GPU_FUSED_DN_REC_NORM_GATE=1`, diary 0043): opt-in decode
  shader fusion that computes g/beta, runs the recurrent DeltaNet update,
  then applies DeltaNet RMSNorm+SiLU gating in one dispatch. This replaces
  the recurrent/g_beta path plus `deltanet_norm_gate` in the merged
  DeltaNet path. Default inference is unchanged; this is still not full GPU
  offload, persistent dispatch, or the megakernel. Correctness passes the
  same short, mixed, and chunk-prefill gates as the earlier fused slices.
  A quick timestamp sample was essentially flat versus diary 0042, so it is
  not yet a standalone performance win.

- **GPU timestamp decode instrumentation** (`SPOCK_GPU_TIMESTAMPS=1`,
  diary 0040): opt-in measurement gate that brackets the decode
  command buffer with Vulkan timestamp queries and reports
  `gpu_decode_us` and `per_token_gpu_us` in `spock-decode` JSON
  output. Always-present fields `prefill_ms`, `decode_ms`, and
  `per_token_ms` are unchanged. Measures GPU-side execution time
  for single-submit-eligible steps and the `skip_layers` LM-head-only
  first decode step after chunk prefill. This is a measurement
  instrument, NOT a performance optimization, NOT full GPU offload,
  and NOT the megakernel. Default-off; no timestamp queries are
  allocated or recorded without the env var. Verified parity on
  `short_correctness_001` with timestamps active (16 tokens).
  Locally observed: `gpu_decode_us` 403422, `per_token_gpu_us`
  [403422] for `--max-new-tokens 1`. Still env-gated, not default.

- **Block-level GPU decode timestamps** (`SPOCK_GPU_BLOCK_TIMESTAMPS=1`,
  diary 0044): opt-in refinement of `SPOCK_GPU_TIMESTAMPS=1` that records
  coarse regions inside single-submit-eligible decode command buffers:
  `embedding`, `layer_0` through `layer_23`, `final_norm`, `lm_head`, and
  `argmax`. `spock-decode` emits `gpu_region_us` only when region data is
  present. Default output and timestamp-only output are unchanged. This is
  measurement only, NOT a performance optimization, NOT full GPU offload,
  NOT persistent dispatch, and NOT the megakernel. Verified with default
  output, timestamp-only output, full fused/single-submit timestamp output,
  and `short_correctness_001` parity under the full fused timestamp gate.

- **Tiled LM-head decode matvec** (`SPOCK_GPU_LM_HEAD_TILED=1`, diary
  0045): opt-in final-LM-head-only shader that computes eight vocabulary rows
  per workgroup and reduces each row dot product across 64 lanes. It reuses
  the existing LM-head descriptor set and does not affect general matvec
  users. Default inference is unchanged. Verified with default short parity,
  gated full fused/single-submit timestamp parity, mixed prompt parity, and
  the chunk-prefill CTest subset. A local 8-token timestamp sample reduced
  `gpu_region_us["lm_head"]` from about 2.61e+06 us to about 3.84e+04 us and
  total `gpu_decode_us` from about 5.43e+06 us to about 2.31e+06 us. Treat
  this as directional until repeated benchmarks are collected.

- **Tiled decode matvec** (`SPOCK_GPU_MATVEC_TILED=1`, diary
  0046): opt-in general matvec shader that replaces `matvec.comp` for
  attention Q/K/V/O projections, DeltaNet merged QKV/Z/A/B projections,
  DeltaNet out_proj, MLP gate/up/down projections, and the final LM
  fallback when `SPOCK_GPU_LM_HEAD_TILED` is not active. Uses
  `matvec_tiled.comp` with BLOCK_ROWS=8, 64 lanes, strided j+=64 for
  arbitrary in_dim, fp32 accumulation, fp16 output. Same 3-binding layout
  and push constants as `matvec.comp`. Does not change `matvec_f32_out`,
  `cmd1` fallback, diagnostics, or default path. Verified with default
  short parity, gated full fused/single-submit/device-resident/deferred
  parity, mixed_correctness_023/pp520_046 parity, and chunk-prefill CTest
  3/3. Timing sample with matvec+tiled LM full gates at max-new-tokens 8:
  gpu_decode_us about 157679, per_token_gpu_us about 5.8ms then 21-22ms,
  lm_head about 23542.9us, layers about 4.8-5.5ms each; previous
  tiled-LM-only sample was gpu_decode_us about 2.31e+06. Directional only,
  not formal benchmark. Still env-gated, not default.

- **Persistent barrier probe** (`vk_barrier_probe`, diary 0047): Real Vulkan
  probe that stress-tests a bounded software global barrier across multiple
  workgroups. The shader (`persistent_barrier_probe.comp`) uses a control buffer
  with `arrived`/`generation`/`failures`/`checksum` atomics and a per-workgroup
  per-iteration trace buffer. Lane 0 per workgroup performs the atomic
  arrival check. The host verifies failures==0, generation==iterations,
  arrived==0, checksum, and trace_mismatches==0. CLI supports `--iterations`
  (default 10000) and `--workgroups` (default 8). Verified at 10k iterations
  with workgroup counts 8, 16, 32, 64, 82, and 128, all with failures=0 and
  trace_mismatches=0. 82 matches the Luce reference block count but this is
  only toy barrier evidence, not persistent decode and not megakernel parity.
  Limitations: no 2-layer mini-pipeline, no long soak under load, no
  occupancy/residency proof beyond tested workgroup counts, no production
  timeout recovery, not full GPU offload.

- **Merged DeltaNet decode command buffers** (`SPOCK_GPU_MERGED_DELTANET=1`,
  diary 0038): opt-in decode fast path records DeltaNet phase-1
  projections/conv/L2 and `dn_compute_g_beta` into the existing per-layer
  command buffer instead of submitting two additional command buffers per
  DeltaNet layer. Default unchanged and diagnostics with intermediate
  dump-step observation keep the old submit boundaries. This reduces host
  orchestration but is NOT full GPU offload, NOT persistent dispatch,
  and NOT the megakernel. Verified with default parity, merged parity,
  merged + per-layer descriptors, combined GPU gate parity, and
  chunk-prefill CTest 3/3. Still env-gated, not default.

- **GPU-resident chunk-prefill path** (diaries 0025/0026): On the no-compare
  GPU-collected+tiled path (`SPOCK_GPU_CHUNK_PREFILL=1`,
  `SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1`, `SPOCK_GPU_CHUNK_PREFILL_TILED=1`,
  no compare flag), all chunk-prefill compute data stays on-device.
  Output handoff (diary 0025): device-local `out_buf`, GPU-to-GPU `final_state`
  copy, `deltanet_chunk_last_to_fp16.comp` for on-device fp32→fp16
  last-token extraction, device-local `correct_last_token_hidden()` handoff.
  Init clear (diary 0026): device-local `init_buf` zeroed via `vkCmdFillBuffer`
  instead of CPU `memset`. Verified parity on `short_correctness_001` (16 tokens),
  `mixed_correctness_023`/`pp520_046` (4 tokens), all CTest gates pass.
  Fallback host-visible paths preserved for compare diagnostics, non-tiled
  paths, and CPU-collected chunk input paths.
- **CTest regression gate for GPU-collected chunk-prefill paths** (diaries
  0022, 0024, 0025): Three CTest tests protect the gated GPU chunk-prefill paths.
  `spock_vk_decode_gpu_collect_chunk_prefill_short` (per-head submit, diary
  0022), `spock_vk_decode_gpu_collect_chunk_prefill_tiled` (tiled dispatch,
  diary 0024), and `spock_vk_decode_gpu_collect_chunk_prefill_short_baseline`
  (no env vars). All run `short_correctness_001 --max-new-tokens 1`.
- **Session extraction** (`DecodeSession`): persistent device, pipelines, buffers, weights
- **Layer-major prefill** with recurrent DeltaNet path: all 48 reference prompts pass
- **Chunk rule primitive** (`deltanet_chunk.cpp`): unit-tested, produces correct output for
  single tokens and matches CPU recurrent simulation for multi-token sequences
- **Tiled single-dispatch chunk-prefill runtime gate** (diary 0024):
  `deltanet_chunk_prefill_tiled.comp` is now wired into the runtime behind
  `SPOCK_GPU_CHUNK_PREFILL_TILED=1` (requires `SPOCK_GPU_CHUNK_PREFILL=1`).
  Both CPU-collected and GPU-collected data paths support the tiled dispatch.
  Verified parity on `short_correctness_001` through 16 generated tokens, and
  on `mixed_correctness_023`/`pp520_046` through 4 generated tokens; diagnostics report
  `nan_count=0`.
  CTest tiled gate: 8.95 sec in the latest rerun after handoff flag cleanup.
  Still env-gated, not default.
- **All existing tests pass**: capabilities, chunk unit, reference parity (48 prompts × 16 tokens)
- **Experimental GPU chunk-prefill path** wired behind env gate `SPOCK_GPU_CHUNK_PREFILL=1`:
  passes `mixed_correctness_023` and `pp520_046` at `--max-new-tokens 1`. Not the default;
  uses conservative per-head submit workaround (slow).
- **GPU-collected chunk-prefill path** wired behind
  `SPOCK_GPU_CHUNK_PREFILL=1 SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1`:
  preserves GPU-collected Q/K/V/g/beta buffers for all DeltaNet layers in
  device-local per-layer segments and feeds them directly into
  `deltanet_chunk_prefill.comp` via `gpu_chunk_prefill_from_gpu_collect()`.
  Avoids CPU intermediate packing/upload for chunk-prefill inputs.
  Default behavior unchanged. When no diagnostic compare flag is set, the
  per-token CPU collection bridge (staging download, half_to_float conversion,
  prefill_chunks_ population) is now skipped entirely. CPU collection remains
  when either `SPOCK_GPU_COLLECT_PREFILL_COMPARE=1` or
  `SPOCK_GPU_CHUNK_PREFILL_COMPARE=1` is set.
  The output side of this path (diary 0025) is now also GPU-resident on the
  no-compare tiled path: chunk output goes to a device-local buffer,
  `final_state` is copied GPU-to-GPU into `dn_state`, and the last-token fp32
  `core_attn_out` slice is extracted and converted to fp16 by
  `deltanet_chunk_last_to_fp16.comp`.
- **GPU collect → chunk-prefill standalone pipeline probe**:
  `spock-deltanet-prefill-pipeline-probe` proves `deltanet_prefill_collect.comp`
  can populate the exact fp32 head-major buffers consumed by
  `deltanet_chunk_prefill.comp`, without CPU intermediate packing. Verified
  `compare-ok` at heads=16, seq_len=104, total_seq=128, chunk_size=64 with
  max_rel_core=8.94e-8, max_rel_state=1.19e-7, nan_count=0.
- **Runtime GPU prefill collection diagnostic** (`SPOCK_GPU_COLLECT_PREFILL_COMPARE=1`):
  Dispatches `deltanet_prefill_collect.comp` from real `DecodeSession` activation
  buffers during layer-major prefill, compares GPU-collected head-major fp32
  Q/K/V/g/beta against the CPU-collected `PrefillChunkState`. Verified exact match on
  `short_correctness_001` (seq_len=9, all 18 DeltaNet layers, max_rel=0,
  max_abs=0, nan_count=0). Diagnostic only; does not change inference output.

### What Needs Work
- **Chunk rule integration**: The fp32 chunk rule produces slightly different output than the
  fp16 recurrent path. For some prompts, this causes argmax divergence on borderline tokens.
  The chunk rule output is *more accurate* (closer to fp32 PyTorch), but the reference was
  generated with the same fp16 recurrent computation. Resolution requires either:
  (a) regenerating the reference with fp32 chunk-based computation, or
  (b) accepting the fp16 recurrent path for parity and using chunk rule only for numerical
      stability on very long prompts.

## Key Findings from Chunk Rule Investigation

1. **Chunk rule is mathematically correct**: CPU simulation confirms chunk output matches
   per-token recurrent simulation for the same Q/K/V/g/beta inputs.
2. **q_scale must NOT be double-applied**: The chunk rule applies `1/sqrt(k_dim)` internally.
   When feeding L2-normalized Q from GPU, set `use_qk_l2norm=false` and do NOT pre-multiply
   by q_scale.
3. **conv1d state is a side effect**: The conv1d kernel updates sliding window state. Running
   it twice (Phase A + Phase C) would corrupt the state. Phase C should only recompute
   input_norm + Z gate projection (no conv1d, no QKV, no A/B projections).
4. **fp16 vs fp32 divergence**: The chunk rule uses fp32 throughout, while the recurrent path
   uses fp16 Q/K/V with fp32 state accumulation. For most prompts these agree, but on
   borderline tokens the difference can flip argmax.

## Next Steps

### 1. GPU chunk-prefill path — efficiency and correctness hardening

The env-gated GPU path (`SPOCK_GPU_CHUNK_PREFILL=1`) is verified-correct
and now supports a tiled single-dispatch mode
(`SPOCK_GPU_CHUNK_PREFILL_TILED=1`) that removes the per-head submit
workaround (diary 0024). Both CPU-collected and GPU-collected data paths
use the same tiled shader `deltanet_chunk_prefill_tiled.comp`.

The GPU-collected+tiled no-compare path (diary 0025) now also eliminates
the CPU readback/re-upload bridge for chunk-prefill output: device-local
chunk output buffer, GPU-to-GPU `final_state` copy, new
`deltanet_chunk_last_to_fp16.comp` shader for on-device fp32→fp16
conversion of the last-token attn slice, and device-local handoff in
`correct_last_token_hidden()`. The host no longer touches chunk-prefill
output data on this path.

Still env-gated, not default.

Follow-up:
- [Done] Wire tiled shader into runtime behind a new env gate (diary 0024).
- [Done] GPU collection wired into session (diary 0020): session-owned
  per-layer persistent buffers, env-gated diagnostics, device-local feeds into
  `gpu_chunk_prefill()`. CPU collection bridge bypassed on the no-compare
  gated path (diary 0021): per-token staging downloads, half_to_float
  conversion, and prefill_chunks_ population are skipped when neither compare
  flag is active.
- [Done] GPU-resident chunk-prefill output handoff (diary 0025): device-local
  output buffer, GPU-to-GPU state copy, `deltanet_chunk_last_to_fp16.comp`
  shader, device-local `correct_last_token_hidden()` handoff. Host no longer
  touches chunk-prefill output data on the no-compare GPU-collected+tiled path.
- [Done] Add formal tests for the gated paths: CTest suite
  `spock_vk_decode_gpu_collect_chunk_prefill` with 3 tests (per-head,
  tiled, baseline), plus parity harness and diagnostic verification.
- [Done] First prompt-coverage expansion: tiled GPU-collect path passes
  `mixed_correctness_023` and `pp520_046` at `--max-new-tokens 4`.
- [Done] First multi-token tiled decode check:
  `short_correctness_001 --max-new-tokens 16`.
- [Done] Device-local buffer usage fix: `VK_BUFFER_USAGE_TRANSFER_SRC_BIT`
  added to device-local buffers used as copy sources (diary 0025).
- [Done] Init zero staging: replaced host-visible init_buf + CPU memset with
  device-local init_buf + vkCmdFillBuffer (diary 0026). Removes the last CPU
  data touch for chunk-prefill init state on the no-compare path.
- [Pending] Expand coverage to 512+ token prompts and broader P0 subsets.
- [Pending] Multi-token decode verification on longer prompts.
- [Pending] Performance characterization: chunk size sensitivity, occupancy,
  register pressure, driver overhead of 24-dispatch-per-chunk orchestration.
- Only then consider defaulting to GPU path.

### 1a. GPU-resident handoff — remaining CPU mediation

Diaries 0025/0026 removed all CPU data bridges for chunk-prefill compute
(device-local output handoff + GPU-side init clear), but the host still
mediates the fast path in non-data ways:
- Per-layer orchestration and submission (host iterates layers, records
  command sequences, submits, waits).
- `gpu_chunk_handoff_ready_` flag is a host-visible boolean (control, not
  data, but still host-mediated).
- Diagnostic/fallback paths require host-visible readback.
- Init zero staging for fallback/compare paths still uses CPU memset.
- Decode argmax/logit computation is on CPU.
- Full megakernel fusion and persistent dispatch not started.
- **Persistent layer-0 full-mixer gate** (diary 0121):
  `vk_persistent_layer0_probe --mode full-mixer` composes all five persistent
  DeltaNet sub-blocks (projection-prefix, conv/L2, g/beta, recurrent, mixer-tail)
  into a single 128-lane 82-workgroup dispatch with 6 software global barriers.
  Structural correctness verified (failures=0, generation=6). Bounded fp16 ULP
  deviation: mixer_output max 6 ULP (28 mismatches), mixer_residual max 16 ULP
  (8 mismatches). This is a reduction-order boundary from single-dispatch chaining,
  not a correctness bug. Not full layer persistence, not inference, not the megakernel.
- **Persistent layer-0 full-layer gate** (diary 0122):
  `vk_persistent_layer0_probe --mode layer0` composes the diary 0121 full-mixer
  path with the post-mixer RMSNorm+MLP tail in one persistent dispatch.
  Structural correctness verified (failures=0, arrived=0, generation=10).
  Bounded fp16 ULP deviation: mixer_output max 6, mixer_residual max 16,
  post_mlp max 105. This is the first captured layer-shaped persistent pass
  from `dn_input_norm` through `post_mlp`. Not all-layer decode, not inference,
  not the final megakernel.

### 2. Chunk rule for numerical stability (optional)

If longer prompts show fp16 accumulation errors, integrate the chunk rule as a fallback.
The implementation approach:
- Phase A: per-token GPU projections → download Q/K/V/g/beta to CPU
- Phase B: CPU chunk rule → upload final state + per-token attn output
- Phase C: per-token GPU: input_norm + Z proj → upload chunk attn → norm+gate + MLP

### 3. Resume megakernel roadmap

The persistent layer-0 full-layer gate (diary 0122) proves a captured layer-0
step can run from `dn_input_norm` through `post_mlp` as one persistent dispatch
with 10 software global barriers. This closes the first layer-shaped milestone
toward the RX 6750 XT Vulkan-native persistent megakernel. The remaining path:

1. Decide the persistent DeltaNet mixer residual precision policy.
   Diary 0124 proved the mode=7 tail implementation is correct and deterministic — the
   105/29/253/62 ULP drift in normal mode is caused by the persistent mixer residual
   being bounded-not-exact and amplified by RMSNorm/MLP, not by a tail bug. Override
   with captured fixture drops to 87/0/1/2 ULP. Diary 0125 then proves the residual
   add itself is exact relative to the actual GPU `mixer_output`: derived-vs-GPU
   residual is 0 ULP, while derived-vs-expected residual remains 16 ULP. Diary
   0126 localizes the next boundary: full-mixer `dn_gated` is only 1 ULP
   off at one element, while `mixer_output` remains max 6 ULP. Diary 0127
   recomputes the output projection from actual GPU `dn_gated`:
   derived-vs-GPU `mixer_output` is max 1 ULP, while derived-vs-expected remains
   max 6 ULP. Diary 0128 proves the same persistent full-mixer shader can run
   layer 4 with layer-specific weights and captured state: mixer_output max 7
   ULP, mixer_residual max 8 ULP, dn_gated tap max 9 ULP. Diary 0129 completes
   representative DeltaNet full-mixer coverage for layers 0, 4, 8, 12, 16, and
   20; worst observed bounds are mixer_output max 25 ULP, mixer_residual max
   32 ULP, and dn_gated tap max 15 ULP.
2. Move from representative single-layer DeltaNet gates to bounded multi-layer decode.
3. Add all 24 layers with cross-layer state management.
4. Add LM head, token selection, and archived basic inference.

After hardware P0 is green, proceed with compute megakernel fusion per IMPLEMENTATION_PLAN.md.
The tiled single-dispatch approach in diaries 0023/0024 is a step toward the fused
megakernel design — the per-head-tile workgroup decomposition generalizes to
larger computation kernels that share read-only inputs across tiles.

### 4. Per-layer stable descriptor sets — single-submit prerequisite

The per-layer stable descriptor set path (diary 0029) eliminates per-layer
descriptor mutation for 24 covered descriptor sets (MLP/norm, attention,
first-stage DeltaNet). Diary 0032 extends coverage to 26 per-layer sets,
adding the L2-norm DeltaNet descriptors (dn_l2_q, dn_l2_k). Diary 0034
extends coverage to 27 per-layer sets, adding the DeltaNet g/beta
computation descriptor (dn_compute_g_beta) after fixing a constructor
ordering bug (bufs_->dn_a_log_bias created before per-layer pre-binding
block). Diary 0035 extends coverage to 28 per-layer sets, adding the
recurrent DeltaNet descriptor (dn_recurrent). Diary 0036 extends coverage
to 29 per-layer sets, adding the DeltaNet norm/gate descriptor
(dn_norm_gate). Diary 0037 extends coverage to 30 per-layer sets, adding
the DeltaNet out_proj descriptor (dn_out_proj). All six intra-DeltaNet
sub-step dispatch-target descriptors are now pre-bound. Total pre-bound:
30 x 24 = 720 per-layer sets + 2 session-level RoPE sets = 722. Diary 0042
adds a fused g/beta+recurrent descriptor set, bringing the opt-in fused
decode path to 31 x 24 = 744 per-layer sets plus 2 session-level RoPE sets.
Diary 0043 adds a fused recurrent+norm_gate descriptor set, bringing that
path to 32 x 24 = 768 per-layer sets plus 2 session-level RoPE sets.
This is a
prerequisite for single-submit recording, where the command buffer must be
recorded ahead of time with descriptor bindings that remain valid across all
layers.

Still env-gated, not default.

**Negative result (diary 0030):** The naive all-six pre-binding attempt
(dn_l2_q, dn_l2_k, dn_recurrent, dn_norm_gate, dn_out_proj,
dn_compute_g_beta) was reverted after decode-state corruption at step 1.
Diary 0030's all-six descriptor-mutation blocker is empirically retired for
the six tracked dispatch-target descriptors because each has now been
independently pre-bound and verified; dn_compute_g_beta had a proven
constructor-ordering root cause (diary 0034), while the other descriptors
were proven safe individually. This does NOT imply full GPU offload,
single-submit, persistent dispatch, or megakernel completion.

**Corrected negative result (diary 0033/0034):** dn_compute_g_beta was
independently isolated in diary 0033 and failed with decode-state
corruption (token 89454 at index 1). The root cause was later traced to
constructor ordering: `bufs_->dn_a_log_bias` was created/uploaded *after*
the per-layer descriptor pre-binding block, so the pre-bound binding 2
referenced a not-yet-created buffer handle (diary 0034). Moving the
a_log/dt_bias cache/upload before the pre-binding block resolves the
failure. dn_compute_g_beta is now pre-bound and passes parity.
dn_recurrent is now pre-bound (diary 0035) and passes parity on the
combined gate suite. dn_norm_gate is now pre-bound (diary 0036) and passes
parity on the combined gate suite. dn_out_proj is now pre-bound (diary 0037)
and passes parity on the combined gate suite. All decode dispatch-target
descriptor mutations are eliminated under SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1.
dn_split_q and dn_split_kv remain listed as uncovered internal descriptors
(not dispatch targets).

Follow-up:
- [Done] Increase descriptor pool capacity (192→1024 maxSets, 192→4096 storage buffers).
- [Done] Pre-allocate and pre-bind 30 x 24 = 720 descriptor sets at session
  construction time (30 per-layer sets: 24 from diary 0029 + 2 L2 from diary
  0032 + 1 g/beta from diary 0034 + 1 recurrent from diary 0035 + 1 norm_gate
  from diary 0036 + 1 out_proj from diary 0037). Skip per-layer mutation block
  in decode() when gate is active.
- [Done] Verify parity on `short_correctness_001` (16 tokens),
  `mixed_correctness_023`/`pp520_046` (4 tokens), combined with GPU chunk-prefill
  gates, device-resident token, and deferred download gates.
- [Done] CTest 3/3 passes under combined gate.
- [Rejected] Naive all-six pre-binding of intra-DeltaNet sub-step
  descriptors (dn_l2_q, dn_l2_k, dn_recurrent, dn_norm_gate, dn_out_proj,
  dn_compute_g_beta) — decode-state corruption at step 1 (diary 0030).
  This descriptor-mutation blocker is empirically retired for the six tracked
  dispatch-target descriptors: dn_compute_g_beta had a proven
  constructor-ordering root cause (diary 0034), and the other descriptors are
  now independently verified safe.
- [Done] L2-norm DeltaNet descriptor pre-binding (dn_l2_q, dn_l2_k) —
  narrow slice of the all-six set that covers only the stateless L2-norm
  descriptors (diary 0032). Verified parity on combined gate suite.
  Confirms L2-norm pair was exonerated from the diary 0030 failure.
- [Done] dn_compute_g_beta pre-binding corrected via constructor ordering
  fix (diary 0034). a_log/dt_bias cache/upload moved before per-layer
  descriptor pre-binding block. Verified parity on combined gate suite.
  Confirms diary 0033's failure was constructor-ordering, not structural.
- [Done] dn_recurrent pre-binding (diary 0035): dn_recurrent is now pre-bound
  and passes parity on the combined gate suite.
- [Done] dn_norm_gate pre-binding (diary 0036): dn_norm_gate is now pre-bound
  and passes parity on the combined gate suite. Confirms dn_norm_gate was
  proven safe individually after being grouped in diary 0030's all-six failure.
- [Done] dn_out_proj pre-binding (diary 0037): dn_out_proj is now pre-bound
  and passes parity on the combined gate suite. All six intra-DeltaNet
  sub-step descriptors are now pre-bound. Eliminates the last dispatch-target
  descriptor mutation blocker.
- [Done] Pre-bound RoPE descriptors with push-constant freq_offset (diary 0031).
  RoPE is no longer a per-layer descriptor mutation blocker.
- [Done] All decode dispatch-target descriptor mutations are eliminated.
  Descriptor bindings are fully pre-resolved for dispatch-target descriptors.
  Per-step work still requires a strategy for step-varying parameters — either
  via GPU-readable state (e.g., storage buffer updated via vkCmdUpdateBuffer)
  or by adopting a command-buffer strategy that supports per-iteration
  parameter updates.
- [Done] Merged DeltaNet decode command buffers (diary 0038): under
  `SPOCK_GPU_MERGED_DELTANET=1`, phase-1 DeltaNet work and
  `dn_compute_g_beta` are recorded into the current per-layer command buffer,
  removing two extra submits per DeltaNet layer on the decode fast path.
- [Done] Single-submit decode (diary 0039): under
  `SPOCK_GPU_SINGLE_SUBMIT=1` (requires `SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1`
  and `SPOCK_GPU_MERGED_DELTANET=1`), all decode-step dispatches (embedding +
  24 layers + final norm + LM head + argmax) are recorded into one command
  buffer and submitted once per token. Reduces host orchestration from 26
  submit/wait round-trips per decode step to 1. Disabled for prefill steps,
  skip_layers steps, and any diagnostic/dump/verbose mode.
- [Done] Fused DeltaNet conv+L2 decode sub-block (diary 0041): under
  `SPOCK_GPU_FUSED_DN_CONV_L2=1` with merged DeltaNet enabled,
  `conv1d_step`, L2 Q, and L2 K are replaced by one default-off fused
  shader dispatch.
- [Done] Fused DeltaNet g/beta + recurrent decode sub-block (diary 0042):
  under `SPOCK_GPU_FUSED_DN_GBETA_RECURRENT=1` with merged DeltaNet enabled,
  g/beta scalar computation and the recurrent update/output are replaced by
  one default-off fused shader dispatch.
- [Done] Fused DeltaNet recurrent + norm_gate decode sub-block (diary 0043):
  under `SPOCK_GPU_FUSED_DN_REC_NORM_GATE=1` with merged DeltaNet enabled,
  g/beta, recurrent update/output, and DeltaNet norm_gate are replaced by one
  default-off fused shader dispatch.
- [Pending] Verify coverage on broader P0 subsets and longer prompts.

## Key Files

| File | Purpose |
|------|---------|
| `src/runtime/vk_session.cpp` | Session implementation + layer-major prefill + `gpu_chunk_prefill()` |
| `src/runtime/vk_session.hpp` | Persistent decode session class |
| `src/runtime/deltanet_chunk.cpp` | Native HF-style chunk-rule primitive |
| `src/runtime/vk_device.cpp` | Deterministic RX-vs-llvmpipe device selection |
| `shaders/deltanet_recurrent.comp` | Recurrent decode/update kernel |
| `shaders/deltanet_chunk_prefill.comp` | GPU chunk-rule shader (experimental) |
| `shaders/deltanet_prefill_collect.comp` | GPU per-token prefill collection shader |
| `apps/spock-deltanet-chunk-prefill-probe.cpp` | Standalone probe (9 probe cases) |
| `apps/spock-deltanet-prefill-collect-probe.cpp` | Standalone GPU collection probe |
| `apps/spock-deltanet-prefill-pipeline-probe.cpp` | Standalone collect → chunk-prefill pipeline probe |
| `shaders/deltanet_chunk_prefill_tiled.comp` | Tiled single-dispatch chunk-prefill shader (experimental) |
| `shaders/deltanet_chunk_last_to_fp16.comp` | GPU fp32→fp16 last-token attn extraction shader (diary 0025) |
| `apps/spock-deltanet-chunk-prefill-tiled-probe.cpp` | Standalone tiled chunk-prefill probe (diary 0023) |
| `tests/run_vk_decode_parity.py` | Vulkan-vs-reference parity harness |
| `tests/run_deltanet_chunk_unit.py` | Torch-vs-native chunk-rule regression test |

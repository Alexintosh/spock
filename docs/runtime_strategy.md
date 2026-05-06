# Vulkan Runtime Strategy

The runtime exists to make the RX 6750 XT execution path measurable, reproducible, and honest about operational parity.

## Target Runtime Modes

The runtime supports three increasingly fused modes:

- `layer_by_layer`: correctness-first path with separate dispatches for kernels or layers.
- `single_submit`: a recorded command buffer executes one token with no host mediation between layers.
- `persistent_dispatch`: one persistent Vulkan dispatch owns the decode pass.

The project should not claim full megakernel parity unless `persistent_dispatch` is correct and stable on the target RADV stack.

## Device Bring-Up

Capability detection must record:

- physical device name
- Vulkan API version
- driver name and version
- subgroup size and supported subgroup operations
- max workgroup size
- shared memory / LDS limits
- storage-buffer limits
- timestamp query support and period
- fp16 shader support
- bf16 support, expected to be unavailable for this target
- cooperative matrix support, expected to be unavailable for this target

The capability dump is benchmark metadata, not debug-only output.

## Memory Model

Required allocation classes:

- device-local weights
- device-local activations
- device-local DeltaNet recurrent state
- device-local KV cache
- host-visible staging buffers
- persistent mapped upload ring
- scratch buffers sized by pipeline mode

The hot decode path must not depend on host-visible model weights.

## DeltaNet Prefill Offload Status

Current production prefill is still not fully GPU-native. The env-gated
`SPOCK_GPU_CHUNK_PREFILL=1` path moves the DeltaNet chunk-rule computation to
Vulkan, but by default runtime Q/K/V/g/beta collection remains CPU-hosted.

**GPU-collected chunk-prefill path** is now wired behind the additional env gate
`SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1` (meaningful only with
`SPOCK_GPU_CHUNK_PREFILL=1`). When both gates are active, the runtime preserves
GPU-collected Q/K/V/g/beta buffers for all DeltaNet layers in device-local
per-layer segments and feeds them directly into `deltanet_chunk_prefill.comp`
via `gpu_chunk_prefill_from_gpu_collect()`, avoiding CPU intermediate
packing/upload for the chunk-prefill inputs. Default behavior unchanged.

Two standalone proofs and one runtime diagnostic confirm the collection
shader is correct and can be driven from real session activations:

- `spock-deltanet-prefill-collect-probe` proves per-token fp16 dn_qkv plus
  fp32 g/beta can be collected into fp32 head-major Q/K/V/g/beta buffers with
  exact CPU agreement.
- `spock-deltanet-prefill-pipeline-probe` proves those collected buffers feed
  `deltanet_chunk_prefill.comp` directly and match `run_deltanet_chunk_rule`
  at heads=16, seq_len=104, total_seq=128, chunk_size=64.
- `SPOCK_GPU_COLLECT_PREFILL_COMPARE=1` dispatches
  `deltanet_prefill_collect.comp` from real `DecodeSession` per-token
  QKV/g/beta activation buffers during layer-major prefill, downloads the
  GPU-collected buffers, and compares against the existing
  CPU-collected `PrefillChunkState`. Verified exact match on
  `short_correctness_001` (all 18 DeltaNet layers, seq_len=9, max_rel=0,
  max_abs=0, nan_count=0).

**CTest regression gate** (diary 0022) protects this gated path from
accidental regression: `spock_vk_decode_gpu_collect_chunk_prefill_short`
runs `short_correctness_001 --max-new-tokens 1` with the double-gated
env vars, and `spock_vk_decode_gpu_collect_chunk_prefill_short_baseline`
runs the same prompt with no env vars as a quick diagnostic reference.

Both tests are registered in CMakeLists.txt and execute via
`tests/run_vk_decode_parity.py`.

The CPU collection bridge is now bypassed on the no-compare gated path
(diary 0021): when `SPOCK_GPU_CHUNK_PREFILL=1` and
`SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1` are active and neither
`SPOCK_GPU_COLLECT_PREFILL_COMPARE=1` nor
`SPOCK_GPU_CHUNK_PREFILL_COMPARE=1` is set, the per-token staging downloads,
half_to_float conversion, and prefill_chunks_ population are skipped entirely.
CPU collection remains for fallback and diagnostics when either compare flag
is active.

**Tiled single-dispatch chunk-prefill gate integrated** (diary 0024). The
tiled shader (`deltanet_chunk_prefill_tiled.comp`) is now wired into the
runtime behind `SPOCK_GPU_CHUNK_PREFILL_TILED=1` (only with
`SPOCK_GPU_CHUNK_PREFILL=1`). The new gate removes the per-head submit
loop: each DeltaNet layer dispatches a single
`vkCmdDispatch(num_heads, ceil(v_dim/TILE_V), 1)` instead of 16 per-head
submits. Both the CPU-collected and GPU-collected data paths support the
tiled dispatch mode. Verified on `short_correctness_001` at
`--max-new-tokens 1` with parity OK, compare diagnostics reporting `nan_count=0`, and CTest suite
3/3 passed (tiled: 10.67 sec in diary 0024; 16.82 sec in the first diary
0025 run; 8.95 sec in the latest diary 0025 rerun after handoff flag
cleanup; not directly comparable). Still env-gated, not default. A new CTest test
`spock_vk_decode_gpu_collect_chunk_prefill_tiled` protects the
GPU-collected + tiled path from regression.

**GPU-resident chunk-prefill output handoff + init clear** (diaries 0025/0026).
On the no-compare GPU-collected+tiled path (no diagnostic compare flag
active), all chunk-prefill compute data stays on-device.

- **Output handoff** (diary 0025): The tiled shader writes to a device-local
  chunk output buffer; `final_state` is copied GPU-to-GPU via `vkCmdCopyBuffer`
  into `bufs_->dn_state`; the last-token fp32 `core_attn_out` slice is extracted
  and converted to fp16 by `deltanet_chunk_last_to_fp16.comp` into per-layer
  `dn_chunk_attn_out_` buffers; and `correct_last_token_hidden()` copies that
  fp16 slice GPU-to-GPU into the `B.dn_qkv` V region. This eliminates CPU
  readback, float_to_half conversion, and upload.
- **Init clear** (diary 0026): The chunk-prefill init state buffer is now
  device-local and zeroed via `vkCmdFillBuffer` instead of host-visible +
  CPU `memset`. This removes the last CPU data touch for chunk-prefill
  compute on the fast path.

A prerequisite buffer usage fix in `src/runtime/vk_device.cpp` added
`VK_BUFFER_USAGE_TRANSFER_SRC_BIT` to device-local buffers used as copy
sources (existing and new `vkCmdCopyBuffer` targets). Without this flag,
the new GPU-to-GPU copies and existing copies from device-local buffers
would violate Vulkan usage constraints.

The fallback host-visible paths are preserved for compare diagnostics,
non-tiled paths, and CPU-collected chunk input paths — both for `out_buf`
(CPU readback) and `init_buf` (CPU memset).

This does NOT make full GPU offload complete. Remaining CPU/host pieces
include per-layer orchestration/submission, diagnostic/fallback paths,
decode argmax/logit/diagnostic readbacks, and broader megakernel
fusion/persistent dispatch.

Still env-gated, not default.

**Opt-in device-resident decode token embedding** (diary 0027). Behind
`SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1`, the per-step embedding lookup reads
token_id directly from the device-local `argmax_result` buffer (binding 0)
instead of from a CPU push constant. The CPU still downloads
`argmax_result` each decode step for external output and parity — this is
NOT full GPU offload and NOT the megakernel. It only removes the CPU token
value as the *source* for the next step's embedding; the current serial
loop still downloads before the next iteration. A new shader
`embedding_lookup_from_buffer.comp` performs the same row lookup as the
existing `embedding_lookup.comp` but reads the index from a storage buffer
rather than a push constant. The shader uses the existing
`pipeline_layout_3` and is dispatched as a single workgroup of 64
invocations. The existing push-constant path remains the default. The gate
is independent of the GPU chunk-prefill gates and composes correctly with
all of them. Still env-gated, not default.

**Opt-in deferred generated-token download** (diary 0028). Behind
`SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1 SPOCK_GPU_DEFER_TOKEN_DOWNLOAD=1`,
the per-step CPU download of `argmax_result` is replaced by a device-local
`vkCmdCopyBuffer` (4 bytes, same-queue, no host round-trip) that writes
into a pre-allocated device-local `gen_tokens` buffer at step offset
`decode_step * 4`. After the decode loop completes, all generated tokens
are downloaded in a single batch via `download_from_device(gen_tokens,
num_generated * 4)` and pushed into `result.generated_tokens` and
`tokens`. The gate is disabled when `verbose`, `debug_dump`, or
`diagnose_decode_drift` is active, because those paths need per-token
values at each step. Guards `max_new_tokens > 0` to avoid zero-sized
Vulkan buffer allocation; zero-token parity now passes. Default behavior
remains per-step download; the gate is disabled for `verbose`,
`debug_dump`, and `diagnose_decode_drift`. Does not restructure the
submit-wait loop — no performance speedup claimed. Still env-gated, not
default.

## Descriptor Model

The baseline descriptor layout exposes:

- packed weights
- activation buffers
- DeltaNet state
- KV cache
- scratch buffers
- runtime constants

Pipeline layouts may specialize by runtime mode, but benchmark output must
identify the mode.

### Per-layer stable descriptor sets

The `layer_by_layer` decode path initially mutated a shared set of 24 (later
26, then 27, then 28) covered `VkDescriptorSet` handles for each of the 24 layers, adjusting
weight offsets and per-layer buffer offsets (KV cache slot, conv1d state)
with every dispatch. This is correct but incompatible with single-submit
recording, where the command buffer must be recorded ahead of time
and cannot mutate descriptors between recording and submission.

When `SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1` (diary 0029, extended in 0032,
0034, 0035, 0036, and 0037), the constructor pre-allocates 30 x 24 = 720 `VkDescriptorSet`
handles from pool `ds_layout_3` and `ds_layout_4` and pre-binds each with
its layer-specific weight offset, activation buffer references, and static
per-layer buffer offsets. The decode loop then skips the per-layer mutation
block and selects the pre-bound set for the current layer via alias:

Diary 0042 adds a fused g/beta+recurrent descriptor set for the fused decode
path, bringing the opt-in fused descriptor coverage to 31 x 24 = 744
per-layer sets. The original 30-set unfused coverage remains available and
default inference does not require the fused descriptor. Diary 0043 adds a
fused recurrent+norm_gate descriptor set for the larger fused decode path,
bringing that opt-in fused coverage to 32 x 24 = 768 per-layer sets plus the
two session-level RoPE descriptor sets.

```
VkDescriptorSet ds_input_norm = per_layer_sets_enabled_
    ? per_layer_sets_->input_norm[layer]
    : D.input_norm;
vkCmdBindDescriptorSets(cmd, ..., &ds_input_norm, ...);
```

Covered descriptor sets include common MLP/norm (9 sets), attention
(10 sets), first-stage DeltaNet (5 sets), L2-norm DeltaNet (2 sets),
DeltaNet g/beta computation (1 set, ds_layout_4), dn_recurrent
(1 set, diary 0035), dn_norm_gate (1 set, diary 0036), and dn_out_proj
(1 set, diary 0037). RoPE descriptors
(D.rope, D.rope_k) are pre-bound once at session construction (diary 0031);
the per-step position is communicated via push constant freq_offset.
Intra-DeltaNet sub-step L2-norm descriptors (dn_l2_q, dn_l2_k) are now
pre-bound (diary 0032) and dn_compute_g_beta is pre-bound (diary 0034).
All DeltaNet dispatch-target descriptors are now covered; the only
remaining uncovered per-layer descriptors (dn_split_q, dn_split_kv) are
internal decomposition descriptors (not dispatch targets). These two
descriptors have never been part of the per-dispatch-target descriptor
work and are not blockers for single-submit recording.

**Negative result (diary 0030):** A naive all-six pre-binding attempt
(dn_l2_q, dn_l2_k, dn_recurrent, dn_norm_gate, dn_out_proj,
dn_compute_g_beta) was reverted after decode-state corruption at step 1.
Diary 0030's all-six descriptor-mutation blocker is empirically retired for
the six tracked dispatch-target descriptors because each has now been
independently pre-bound and verified; dn_compute_g_beta had a proven
constructor-ordering root cause (diary 0034), while the other descriptors
were proven safe individually. This does NOT imply full GPU offload,
single-submit, persistent dispatch, or megakernel completion.

**Corrected negative result (diary 0033/0034):** dn_compute_g_beta was
independently isolated and failed in diary 0033 with the same corruption
signature. The root cause was traced to constructor ordering:
`bufs_->dn_a_log_bias` was created after the per-layer descriptor
pre-binding block, so the pre-bound binding 2 referenced a not-yet-created
buffer handle (diary 0034). Moving the a_log/dt_bias cache/upload before
the pre-binding block resolved the failure. dn_compute_g_beta is now
pre-bound and verified correct. dn_recurrent is now pre-bound (diary 0035)
and passes parity on the combined gate suite. dn_norm_gate is now pre-bound
(diary 0036) and passes parity on the combined gate suite. dn_out_proj is
now pre-bound (diary 0037) and passes parity on the combined gate suite.
All decode dispatch-target descriptor mutations are eliminated under
SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1.

Descriptor pool capacity was increased to accommodate the per-layer sets:
- maxSets: 192 → 1024
- storage buffer slots: 192 → 4096
- combined-image-sampler: 64 (unchanged)

The pool retains `VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT` so
sets can be destroyed per-session and the pool reused.

This is NOT single-submit. It removes the per-layer descriptor mutation
blocker for covered sets, enabling future command-buffer pre-recording.

### Merged DeltaNet Decode Command Buffers

`SPOCK_GPU_MERGED_DELTANET=1` (diary 0038) is the next opt-in host
orchestration reduction after descriptor mutation elimination. In the DeltaNet
decode branch, it records phase-1 work (input norm, QKV/Z/A/B projections,
conv1d, and Q/K L2 norm) and `dn_compute_g_beta` into the existing per-layer
command buffer instead of allocating and submitting separate `cmd1` and
`gb_cmd` command buffers.

This removes two additional `submit_and_wait` calls per DeltaNet layer on the
decode fast path. The gate is disabled for `dump_step_components` and
`dump_step_hiddens` diagnostics so intermediate observation points retain the
old submit boundaries.

This is NOT single-submit. The runtime still records and submits a command
buffer per layer and still waits on the host. It is also NOT full GPU offload,
NOT persistent dispatch, and NOT the megakernel. It is a narrow step toward
reducing host-side scheduling before attempting broader command-buffer
  pre-recording or fusion.

### Single-Submit Decode

`SPOCK_GPU_SINGLE_SUBMIT=1` (diary 0039) records all dispatches for a
decode step into one command buffer and submits once per token.

When this gate is active (requires `SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1`
and `SPOCK_GPU_MERGED_DELTANET=1`), the runtime allocates a single
`VkCommandBuffer`, records the embedding lookup, all 24 layers (attention
and DeltaNet), final RMSNorm, LM head matvec, and argmax into it, and
submits once. This reduces host orchestration from 26 `submit_and_wait`
round-trips per decode step to 1.

The gate is disabled for prefill steps, `skip_layers` steps (first decode
step after chunk prefill), and any diagnostic/dump/verbose mode.
Per-layer descriptor mutation is not needed because all descriptor sets are
pre-bound at session construction. Step-varying parameters (RoPE freq_offset,
KV cache position, push constants) are recorded inline via `vkCmdPushConstants`
at each dispatch point in the single command buffer.

This is the `single_submit` runtime mode described in the Synchronization
Strategy section. It is NOT persistent dispatch and NOT the megakernel.
Still env-gated, not default.

### Fused DeltaNet Conv+L2 Decode Sub-Block

`SPOCK_GPU_FUSED_DN_CONV_L2=1` (diary 0041) is a default-off decode-only
fusion gate for the merged DeltaNet path. When
`SPOCK_GPU_MERGED_DELTANET=1` is active, it replaces the three-dispatch
sequence `conv1d_step`, L2 Q, and L2 K with one dispatch of
`deltanet_conv_l2_qk.comp`.

The fused shader uses the existing `dn_conv` descriptor set:

- binding 0: QKV fp16 input/output
- binding 1: DeltaNet conv state fp16 input/output
- binding 2: depthwise conv weights

The Q and K slices are normalized per 128-wide head after the conv+SiLU
step; the V slice remains convolved only. This is a first fused decode
sub-block, not full GPU offload, persistent dispatch, or the megakernel.
Default inference remains on the original unfused path unless the gate is
explicitly enabled.

### Fused DeltaNet G/Beta + Recurrent Decode Sub-Block

`SPOCK_GPU_FUSED_DN_GBETA_RECURRENT=1` (diary 0042) is a default-off
decode-only fusion gate for the merged DeltaNet path. When
`SPOCK_GPU_MERGED_DELTANET=1` is active, it replaces the separate
`deltanet_compute_g_beta` and `deltanet_recurrent` dispatches with one
dispatch of `deltanet_recurrent_gbeta.comp`.

The fused shader uses a 6-binding descriptor set:

- binding 0: projected `dn_a` fp16
- binding 1: projected `dn_b` fp16
- binding 2: packed DeltaNet `a_log`/`dt_bias` fp32
- binding 3: Q slice of `dn_qkv`
- binding 4: K/V slice of `dn_qkv`, with output overwriting V
- binding 5: DeltaNet recurrent state for the current DeltaNet layer

The shader computes g and beta per head from `dn_a`, `dn_b`, and
`a_log`/`dt_bias`, then performs the same recurrent update as the existing
unfused recurrent shader. Under this gate, the g/beta tail of `dn_state` is
not used as an intermediate between those two steps. This is a second fused
decode sub-block, not full GPU offload, persistent dispatch, or the
megakernel.

### Fused DeltaNet Recurrent + Norm/Gate Decode Sub-Block

`SPOCK_GPU_FUSED_DN_REC_NORM_GATE=1` (diary 0043) is a default-off
decode-only fusion gate for the merged DeltaNet path. When
`SPOCK_GPU_MERGED_DELTANET=1` is active, it replaces the g/beta computation,
recurrent update/output, and `deltanet_norm_gate` dispatch with one dispatch
of `deltanet_recurrent_gbeta_norm_gate.comp`.

The fused shader uses an 8-binding descriptor set:

- binding 0: projected `dn_a` fp16
- binding 1: projected `dn_b` fp16
- binding 2: packed DeltaNet `a_log`/`dt_bias` fp32
- binding 3: Q slice of `dn_qkv`
- binding 4: K/V slice of `dn_qkv`, with gated output overwriting V
- binding 5: DeltaNet recurrent state for the current DeltaNet layer
- binding 6: projected `dn_z` fp16
- binding 7: DeltaNet RMSNorm weight fp32

The shader computes g and beta inline, performs the recurrent state update,
RMS-normalizes the recurrent output, applies the DeltaNet norm weight with
the same fp16 rounding boundary as the unfused path, multiplies by `SiLU(z)`,
and writes the gated result back over the V/output slice. Under this gate, the
standalone `deltanet_compute_g_beta`, fused g/beta+recurrent, and
`deltanet_norm_gate` dispatches are skipped for the fused DeltaNet path.

The first correctness sweep passes, including the combined fused gates and the
chunk-prefill CTest subset. A short timestamp sample was flat relative to the
two-fusion path, so this gate is currently a correctness/structure step, not a
claimed performance win. It is not full GPU offload, persistent dispatch, or
the megakernel.

## Synchronization Strategy

`layer_by_layer` may use command-buffer ordering and explicit barriers between operations.

`single_submit` records a full token schedule into one command buffer and submits once per token. This is the minimum operational target for "no host mediation between layers."

`persistent_dispatch` requires a proven cross-workgroup coordination strategy. If a software global barrier is used, it must have:

- a bounded progress argument
- a timeout or watchdog-safe failure mode during validation
- stress tests across repeated long decode runs
- exact-token parity against `layer_by_layer`

`vk_barrier_probe` is now a real Vulkan software-barrier probe rather than a
placeholder. Diary 0047 proved the bare barrier: configurable persistent
workgroups synchronize for `--iterations` rounds through a storage-buffer
`arrived` counter and `generation` counter, with host validation of `failures`,
`generation`, `arrived`, checksum, and per-workgroup/per-iteration trace data.

Diary 0048 extends the same probe into a two-stage mini-pipeline. Each
iteration now writes one coherent scratch slot per workgroup, runs the software
global barrier, cross-reads all scratch slots into trace/checksum, then runs a
second global barrier before scratch overwrite. The first non-coherent scratch
version reached the expected generation count but failed checksum/trace
validation at 82 workgroups x 10000 iterations; marking `scratch.values[]`
`coherent` fixed the data-visibility failure. Local runs passed 10k iterations
at workgroup counts 8, 16, 32, 64, 82, and 128 with zero failures and zero
trace mismatches.

Diary 0049 adds an opt-in `--timestamps` mode to the barrier probe. It writes
GPU timestamps around the single probe dispatch and reports `gpu_dispatch_us`,
`per_barrier_us`, and `barriers` when timestamp queries are supported. A local
82-workgroup x 10000-iteration sample measured about 113576 us GPU dispatch
time, or about 5.67878 us per software barrier, while preserving correctness.
This is a first timing hook and sample, not a final benchmark.

Diary 0050 records a longer local soak at 82 workgroups x 1000000 iterations.
Both the non-timestamped and timestamped serial runs passed with generation
2000000 and zero trace mismatches. The timestamped run measured about
1.03471e+07 us GPU dispatch time, or about 5.17354 us per software barrier.
This improves confidence in local forward progress, but it is not an under-load
soak and it still uses the toy scratch/cross-read workload.

Diary 0051 adds `--payload-iters N`, an optional deterministic per-lane ALU
payload before each scratch write. The default no-payload path preserves the
existing output shape and verification formulas. A local
82-workgroup x 10000-iteration run with `--payload-iters 64 --timestamps`
passed with zero trace mismatches and measured about 149650 us GPU dispatch
time, or about 7.48249 us per software barrier. This adds lane-level work and a
shared-memory reduction, but it is still not matvec-like memory traffic.

Diary 0052 adds `--payload-cols N`, an optional lane-strided memory payload over
deterministic uint32 input and weight buffers. The shader declares the payload
descriptors unconditionally, so the host always binds them and uses one-element
dummy buffers when the flag is absent. A local
82-workgroup x 10000-iteration run with `--payload-cols 256 --timestamps`
passed with zero trace mismatches and measured about 139090 us GPU dispatch
time, or about 6.95452 us per software barrier. Combined
`--payload-iters 64 --payload-cols 256` also passed.

Diary 0053 brackets a long-run boundary for `--payload-cols 256`: 82 workgroups
passed at 750k iterations with about 9.62s of GPU dispatch time, but 900k and
1M failed with all-zero GPU output. The 1M non-timestamped rerun printed a RADV
context-loss/hard-recovery message. This means persistent-dispatch design cannot
assume arbitrarily long single dispatches are safe on this stack once meaningful
memory payload is present.

Diary 0054 tests the immediate mitigation: five repeated bounded
100k-iteration memory-payload runs all passed, with per-barrier timing tightly
clustered around 6.45 us. This does not restore strict single-dispatch
megakernel parity, but it makes bounded persistent chunks the more defensible
next design candidate.

Diary 0055 moves repeat testing into `vk_barrier_probe` with `--repeats N`.
For repeats greater than one, the probe reuses one Vulkan device, buffers,
descriptors, and pipeline, resets per-repeat control/trace/scratch state, and
emits aggregate JSON. A local 3-repeat
`--payload-cols 256 --iterations 100000` run passed with zero aggregate trace
mismatches and per-barrier timing around 6.46 us.

Diary 0056 increases the same in-process bounded test to 10 repeats. The run
completed 2,000,000 software barriers across bounded dispatches with zero
aggregate trace mismatches and per-barrier timing clustered around 6.45-6.46 us.
This further supports bounded persistent chunks as the practical next target.

Diary 0057 adds a CTest gate for the current strongest fast decode env stack
before chunked decode orchestration work begins. The test exercises per-layer
descriptor sets, merged/fused DeltaNet gates, single-submit decode,
device-resident token input, deferred token download, tiled decode matvec, and
tiled LM head on `short_correctness_001` for four generated tokens.

Diary 0058 reserves the bounded chunked decode env gates,
`SPOCK_GPU_CHUNKED_DECODE` and `SPOCK_GPU_DECODE_CHUNK_SIZE`, in
`DecodeSession::decode()`. They are parse-only and force-disabled for now; a
CTest run with the variables set confirms current fast-path behavior is
unchanged.

Diary 0059 added a dedicated CTest for the inert scaffold contract. Diary 0060
then moves the gate to a live size-1 equivalence mode:
`chunked_decode_enabled` is true only when the chunked gate is requested with
`SPOCK_GPU_DECODE_CHUNK_SIZE=1` and all full fast-path prerequisites are
present. The `spock_vk_decode_chunked_gate_size1_fast_gate_short` CTest verifies
that this active size-1 gate preserves current output.

Diary 0061 lifts the chunk-size restriction and implements the first active
multi-token chunked decode path under `SPOCK_GPU_CHUNKED_DECODE=1` and
`SPOCK_GPU_DECODE_CHUNK_SIZE=N` (N > 1). One command buffer stays open across
up to N eligible decode steps and submits on full chunk or final partial chunk.
At this point, the first post-prefill `skip_layers` step still followed the old
single-submit path; diary 0067 later absorbs that step into the chunk. An explicit
`VkBufferMemoryBarrier` on `argmax_result` after each step's deferred token
copy ensures the next `embedding_from_buffer` read sees the coherent
next-token value and prior transfer reads finish before later argmax writes.
Timestamps remain disabled. Verified manually at chunk size 4 with
max_new_tokens 4 and 5. A size-4 full-plus-partial CTest with max_new_tokens 6
validates the boundary. Not the megakernel: the host still submits per chunk, no
persistent dispatch, no performance measurement yet.

**Submit-count instrumentation** (diary 0062) exposes `decode_submit_count` and
`chunked_decode_submit_count` in `DecodeResult` and `spock-decode` JSON output.
The counters are scoped to the main decode-loop final/chunk submissions under the
full-fast/chunked path, not every legacy diagnostic or prefill submit. Initially,
the size-4 partial CTest asserted decode_submit_count=3 and
chunked_decode_submit_count=2: one skip-layers submit, one full chunk of four
eligible steps, one final partial chunk of one eligible step. Diary 0067 updates
that current CTest expectation to 2/2 after absorbing the skip-layers step into
the chunk. This proves structural submission amortization, not wall-clock
performance.

**Size-8 multiprompt CTest gate** (diary 0063) extends chunked decode coverage to
chunk size 8 across two prompts (`short_correctness_001`, `mixed_correctness_023`)
with full 16-token reference output. The test
`spock_vk_decode_chunked_gate_size8_multiprompt_16` initially asserted
decode_submit_count=3 and chunked_decode_submit_count=2 for each prompt. Diary
0067 updates that current expectation to 2/2 after the skip-layers step is
absorbed into the chunk. This is correctness broadening, not performance proof.
Still not
persistent dispatch, not the megakernel, and not wall-clock measurement.

**Chunked decode sweep tool** (diary 0064): `tools/run_chunked_decode_sweep.py`
automates multi-chunk-size sweeps across one or more reference prompt IDs. It
invokes `spock-decode` with the full fast env stack plus
`SPOCK_GPU_CHUNKED_DECODE=1` and `SPOCK_GPU_DECODE_CHUNK_SIZE=N` for each
requested chunk size, compares generated tokens against references, and emits
JSON with `git_rev`, `env_gates`, `ids`, `chunk_sizes`, and per-run `match`,
`decode_submit_count`, `chunked_decode_submit_count`, `elapsed_ms`,
`prefill_ms`, `decode_ms`, `generated_count`. It exits nonzero on decode failure
or token mismatch. A local sweep at chunk sizes 1, 4, 8, 16 across two prompts
(`short_correctness_001`, `mixed_correctness_023`) confirmed reference parity at
all sizes. Submit counts: size 1 => decode 16, chunked 15; size 4 => decode 5,
chunked 4; size 8 => decode 3, chunked 2; size 16 => decode 2, chunked 1.
Single-run host timing captured but not benchmark proof. This is a measurement
convenience tool; it does not modify runtime, shader, or test code.

The sweep tool now supports `--warmup-runs` (default 0) and `--timed-runs` (default 1)
for controlled repeat measurement (diary 0065). Warmups run first per id/chunk_size,
must match references, and are excluded from aggregate timing. Timed runs must each match;
per-run records include `run_index`. Aggregate records per id/chunk_size include mean/min/max
for elapsed, prefill, and decode ms plus timed_runs count. Top-level JSON includes
`warmup_runs` and `timed_runs`. A local run with `--warmup-runs 1 --timed-runs 2` at chunk
sizes 8 and 16 confirmed correct submit counts and all matches true. Default warmup/timed
settings preserve one-run usage with an aggregate record. This gives controlled host-side
timing structure but remains host-side single-machine evidence, not final performance proof.

**Controlled chunk-size sweep** (diary 0066). A warmup-guarded sweep with
`--warmup-runs 1 --timed-runs 3` across chunk sizes 1, 2, 4, 8, 16 on
`short_correctness_001` at `--max-new-tokens 16` confirmed reference parity at
all sizes. Submit counts match the structural model: size 1 => decode 16,
chunked 15; size 2 => decode 9, chunked 8; size 4 => decode 5, chunked 4;
size 8 => decode 3, chunked 2; size 16 => decode 2, chunked 1. Host-side
decode_ms shows a modest monotonic decrease from 353.0 ms (chunk size 1) to
349.9 ms (chunk size 16), roughly a 0.9% reduction. Per-chunk-size variance is
under 0.7 ms across 3 timed runs. This is single-prompt, single-machine,
host-side timing evidence; not GPU timestamps, not multi-prompt, not
high-token-count, not final benchmark proof, not persistent dispatch, and
not the megakernel.

**First skip_layers step absorbed into chunk** (diary 0067). The first post-prefill
`skip_layers` decode step (final-norm + LM-head + argmax) is now recorded into the
chunked command buffer under `SPOCK_GPU_CHUNKED_DECODE=1`, eliminating the previously
separate skip-layers submit. It opens `chunk_cmd`, increments `chunk_recorded_steps`,
inserts the same argmax-result next-token barrier, and defers submit like any other
eligible step. The submit-count formula changes from `1 + ceil((N-1)/C)` to `ceil(N/C)`
with `decode_submit_count == chunked_decode_submit_count` at all chunk sizes. CTest
expectations updated: size4 partial now decode=2/chunked=2; size8 multiprompt now decode=2/
chunked=2. New size-16 single-chunk CTest asserts decode=1, chunked=1 for max_new_tokens=16.
Refreshed sweep confirms reference parity and submit counts: size1 16/16, size2 8/8, size4 4/4,
size8 2/2, size16 1/1. Host-side decode_ms means (3 runs): size1 353.09, size2 351.98,
size4 351.24, size8 350.14, size16 350.49. Size8 was best in this sample; timing is not
monotonic after size8. Every decode step is now chunked; non-chunked path unchanged.
This is structural submit-count progress, not persistent dispatch, not the megakernel,
and not wall-clock performance proof.

**GPU timestamps inside chunked command buffers** (diary 0068). The
`!gpu_timestamps` exclusion from the chunked-decode enable condition is removed.
When both `SPOCK_GPU_TIMESTAMPS=1` and `SPOCK_GPU_CHUNKED_DECODE=1` are active,
per-step start/end timestamp queries are recorded inside the chunked command buffer.
The skip-layers first decode step also records a timestamp start, so `ts_decode_steps`
has one entry per generated token. Block-level timestamps
(`SPOCK_GPU_BLOCK_TIMESTAMPS=1`) remain excluded from the chunked path via
`!gpu_block_timestamps`. `tests/run_vk_decode_parity.py` gains
`--expect-gpu-decode-us-positive`. A new CTest
`spock_vk_decode_chunked_gate_size16_timestamps_short` exercises chunk size 16,
`SPOCK_GPU_TIMESTAMPS=1`, max_new_tokens 16, asserting decode_submit_count=1,
chunked_decode_submit_count=1, and positive gpu_decode_us. A direct run produced
gpu_decode_us=347611 with 16 per_token_gpu_us values for 16 generated tokens.
Host per_token_ms remains chunk-flush-shaped; GPU per_token_gpu_us is the useful
device timing. This is basic per-token timestamp instrumentation inside chunked
command buffers, not block-level timestamps, not final benchmark proof, not
persistent dispatch, and not the megakernel.

**Chunked decode sweep GPU timestamp extension** (diary 0069). The sweep tool
`tools/run_chunked_decode_sweep.py` gains `--gpu-timestamps`, an opt-in flag
that sets `SPOCK_GPU_TIMESTAMPS=1` in the decode environment and includes GPU
timing fields from `spock-decode` JSON output in per-run and aggregate records.
Per-run records add `gpu_decode_us` and a summary of `per_token_gpu_us`
(count, mean, min, max). Aggregate records add mean/min/max for both
`gpu_decode_us` and `per_token_gpu_us_mean`. When the flag is set, the sweep
validates that `gpu_decode_us` is present and positive and that
`per_token_gpu_us` length matches `generated_count` (falling back to
`max_new_tokens` if absent), marking the per-run record `match=false` with
`gpu_timestamp_error` like other validation errors. Default behavior without the
flag is unchanged. `spock_chunked_sweep_gpu_timestamp_unit` covers the helper
logic without requiring GPU hardware. A short end-to-end run at chunk size 16,
one timed run, `short_correctness_001`, and 16 generated tokens passed with
match=true, decode/chunked submit counts 1/1, gpu_decode_us=347708, and
per_token_gpu_us_count=16.


**Controlled GPU-timestamped chunked decode sweep** (diary 0070). A warmup-guarded
sweep with `--warmup-runs 1 --timed-runs 3 --gpu-timestamps` across chunk sizes
1, 2, 4, 8, 16 on `short_correctness_001` at `--max-new-tokens 16` confirmed
reference parity at all sizes. GPU decode time is nearly flat across chunk sizes:
`gpu_decode_us` mean ranges from 348296 us (size 1) to 347333 us (size 16), a
~0.28% reduction. Per-token GPU time is nearly constant at about 21.7 ms regardless
of chunk size. Host-side `decode_ms` still improves modestly with submit-count
reduction (353.1 ms to 349.5 ms, ~1.0%), but the GPU data confirms this short run
is dominated by actual GPU work, not submit overhead. This is single-prompt,
16-token measurement evidence: not persistent dispatch, not the megakernel, and
not a throughput benchmark. Do not overclaim from these numbers.

**Barrier probe decode-shape gate** (diary 0071). `vk_barrier_probe` now accepts
`--tokens N` and `--layers N` to set `iterations = tokens * layers`. This is a
semantic wrapper around the existing persistent barrier/payload probe: it runs the
same deterministic barrier workload for a number of iterations matching the total
layer-forward count of a decode geometry. When decode-shape mode is active, JSON
output includes `tokens`, `layers`, and `decode_shape_iterations`. The existing
`--iterations` path is unchanged. A CTest gate
`spock_barrier_probe_decode_shape` exercises 16 tokens x 24 layers = 384 iterations
with 8 workgroups and 128-column memory payload. This is not real decode, not model
weights, and not the megakernel; it is a decode-shaped persistent barrier/payload
regression gate.

**Barrier probe decode-shape 82-workgroup timing** (diary 0072). A decode-shaped
run at 82 workgroups, 128 tokens x 24 layers (3072 iterations) with
`--payload-cols 128 --timestamps` passed all correctness checks:
generation 6144, checksum match, zero failures, zero trace mismatches.
Single-run per-barrier: ~7.92 us. Three-repeat run showed repeat 1 at ~8.00 us
(likely warmup/clock effect) and repeats 2–3 stabilizing near ~6.31 us, slightly
below the ~6.45 us evidence from diary 0054/0056 (which used 256 payload cols).
This improves confidence that 82-workgroup persistent barrier coordination holds at
decode-relevant iteration scales, but it is still a synthetic probe: not real decode,
not under load, not an occupancy proof for production shaders, and not megakernel parity.

**Barrier probe model-width decode-shaped payload** (diary 0073). A decode-shaped run
at the same geometry as diary 0072 (82 workgroups, 128 tokens x 24 layers, 3072 iterations)
but with `--payload-cols 1024`, matching Qwen3.5 hidden_size. All three repeats passed
correctness checks: zero failures, zero trace mismatches. Stable per-barrier timing:
repeats 2-3 at ~6.96 us/barrier vs diary 0072's ~6.31 us/barrier at cols=128.
The ~10% cost increase for 8x more memory traffic is modest and sublinear.
Still synthetic uint32 memory traffic, not real fp16/fp32 decode matvec, and not an
occupancy proof for production shaders.

This is positive viability evidence for the synchronization and data-exchange
primitive, including the Luce reference block count of 82. It is still a toy
probe: it is not persistent decode, not an under-load soak, not a repeated
barrier-overhead benchmark, not an occupancy proof for the real decode shaders,
not real fp16/fp32 decode matvec, not proof that long memory-heavy single
dispatches are safe, and not megakernel parity.

**Qwen3.5 decode-shape preset** (diary 0074). `--qwen35-decode-shape-preset` sets
tokens=128, layers=24, workgroups=82, payload_cols=1024 in one flag. User-supplied
--tokens/--layers/--workgroups/--payload-cols override the preset values. JSON output
includes `qwen35_decode_shape_preset: "active"` when the preset is used. A CTest gate
`spock_barrier_probe_qwen35_preset` exercises the full preset workload (3072 iterations,
82 workgroups, 1024 payload columns) without timestamps or repeats. This is a reproducibility
preset for the synthetic model-width probe, not real decode and not the megakernel.

**Persistent decode skeleton probe** (diary 0075). `vk_persistent_decode_skeleton` is a standalone probe that exercises the proven software-global-barrier pattern with actual fp16 input/weight buffers and fp32 accumulation, rather than the uint32 synthetic payloads of `vk_barrier_probe`. This is the first probe combining persistent dispatch with decode-shaped fp16/fp32 compute.

The shader performs per-workgroup lane-strided fp16 dot products over `hidden` columns, reduces in shared memory with fp32 accumulation, writes coherent scratch, runs two global barriers per iteration (same as `persistent_barrier_probe`), and writes per-workgroup per-iteration trace. The host generates deterministic fp16 input/weight values and validates checksum and trace against double-precision CPU reference.

CLI options: `--tokens N` (default 2), `--layers N` (default 4), `--hidden N` (default 128), `--workgroups N` (default 8), `--repeats N`, `--timestamps`, `--qwen35-preset` (tokens=128, layers=24, hidden=1024, workgroups=82). CTest gates: `spock_persistent_decode_skeleton_help`, `spock_persistent_decode_skeleton_smoke`. A Qwen3.5 preset repeat run passed with zero failures and zero trace mismatches; repeats 2-3 stabilized around 5.94 us/barrier.

This is still synthetic: not model weights, not attention/DeltaNet/KV/LM head, not production decode, not the megakernel, and not a performance benchmark. It validates that the fp16/fp32 compute + persistent barrier coordination is correct before investing in real model weight loading.

**Persistent decode skeleton real-weight probe** (diary 0076). `vk_persistent_decode_skeleton` now accepts `--repack-dir DIR` and `--weight-role ROLE` together, loading a real fp16 WeightArtifact from the repacked model manifest. The probe validates dtype fp16, rank-2 shape, and bounds (workgroups <= rows, hidden <= cols); hidden is inferred from weight cols when not explicitly supplied. Two host-side fixes were needed: the CPU expected checksum now mirrors the shader's 64-lane-strided fp32 partial-sum + tree-reduction order exactly, and fp16 subnormal values are preserved during host decoding. A CTest gate `spock_persistent_decode_skeleton_real_weight_smoke` exercises `layer.0.mlp_gate` from `artifacts/spock-text-repack-qwen35-0p8b` at hidden=128, workgroups=4. Direct runs verified exact checksum match at both hidden=128 and inferred hidden=1024. This is the first real model-weight use inside the persistent skeleton; still synthetic input, only prefix rows/cols, no layer semantics, not inference, not the megakernel.

**Persistent decode skeleton multi-role real-weight probe** (diary 0077). `vk_persistent_decode_skeleton` now accepts multiple `--weight-role ROLE` flags. When two or more roles are specified, the probe loads all roles from the WeightArtifact, validates each independently, and dispatches the persistent skeleton separately per role in a loop. Per-role results include checksum, expected_checksum, trace_mismatches, failures, and status. Single-role and synthetic modes produce identical output to diary 0076. A CTest gate `spock_persistent_decode_skeleton_multi_role_smoke` exercises `layer.0.mlp_gate` and `layer.0.mlp_up` from `artifacts/spock-text-repack-qwen35-0p8b`. This is real multi-weight skeleton validation, not inference, not layer semantics, and not the megakernel.

**Persistent decode skeleton row-strided weight coverage** (diary 0078). `vk_persistent_decode_skeleton` now accepts `--row-count N`, enabling a bounded set of resident workgroups to cover more matrix rows than there are workgroups via row-strided assignment (`row = group; row < row_count; row += workgroups`). Default `row_count == workgroups` preserves all prior checksums and behavior. A real-weight direct run (layer.0.mlp_gate, workgroups=4, row-count=16) produces checksum 3002794576 with exact agreement. A CTest gate `spock_persistent_decode_skeleton_row_count_real_weight_smoke` exercises this path. This is row-strided projection coverage, not inference, not layer semantics, and not the megakernel.

**Persistent decode skeleton model-width row-strided coverage** (diary 0079). The row-strided path now has a stronger CTest gate, `spock_persistent_decode_skeleton_row_count_model_width_smoke`, covering `layer.0.mlp_gate` at hidden=1024, workgroups=82, row-count=128. Direct checks also passed for `layer.0.mlp_gate` plus `layer.0.mlp_up` at the same geometry. This is model-width real-weight projection coverage for the persistent skeleton; still not inference, not MLP semantics, and not the megakernel.


**Persistent MLP micro-probe** (diary 0080). `vk_persistent_mlp_probe` runs a single-dispatch persistent micro-probe with shader `persistent_mlp_probe.comp`. The probe chains gate projection, up projection, SiLU(gate)*up element-wise activation, and down projection inside one persistent dispatch. Two software global barriers (generation=2) enforce inter-stage dependencies: one after gate/up materialization and one after activation materialization, before the down projection consumes activated scratch. Bindings: control, gate/up scratch, input, gate/up/down weights, output.

CLI defaults: hidden=128, intermediate=16, output-rows=8, workgroups=8. Optional `--repack-dir` loads real fp16 weights (`layer.0.mlp_gate`, `layer.0.mlp_up`, `layer.0.mlp_down`) from the repacked model manifest.

Synthetic direct run: status ok, checksum 371183224, expected_checksum 371183224, failures 0, generation 2. Real weight direct run: status ok, checksum 1616650692, expected_checksum 1616650692, failures 0, generation 2. Focused CTest passed 9/9 (persistent decode skeleton + MLP probe + diary check).

This is the first persistent probe exercising multi-stage barrier-synchronized compute dependencies rather than a single repeated projection. Still not inference, not full MLP coverage, not real activations, not residual/layer semantics, and not the megakernel.

**Persistent MLP full real-weight coverage** (diary 0081). `vk_persistent_mlp_probe` now has a full real-weight gate, `spock_persistent_mlp_probe_full_real_weight_smoke`, covering layer.0 MLP dimensions hidden=1024, intermediate=3584, output_rows=1024, workgroups=82. Direct full run: status ok, checksum 2160240877, expected_checksum 2160240877, failures 0, arrived 0, generation 2. This proves the persistent MLP probe can execute all real gate/up/down rows and columns for one layer in a single dispatch with exact CPU/GPU agreement. It remains a standalone synthetic-input probe, not inference, not residual/layer semantics, and not the megakernel.

**Persistent MLP residual update** (diary 0082). `vk_persistent_mlp_probe --residual` now validates `input + down(SiLU(gate(input))*up(input))` for covered output rows, enforcing `output_rows <= hidden`. Full real-weight residual gate `spock_persistent_mlp_probe_full_real_weight_residual_smoke` covers hidden=1024, intermediate=3584, output_rows=1024, workgroups=82 against `layer.0.mlp_gate`, `layer.0.mlp_up`, and `layer.0.mlp_down`. Direct full residual run: status ok, checksum 3327711045, expected_checksum 3327711045, failures 0, arrived 0, generation 2. This is the first residual-stream update pattern in the persistent MLP probe; still synthetic input, not RMSNorm, not attention/DeltaNet, not inference, and not the megakernel.

**Persistent MLP embedding input** (diary 0083). `vk_persistent_mlp_probe --input-token ID` loads a real token embedding row from `global.token_embedding` (shape [248320, 1024], dtype fp16) as the input vector, replacing the synthetic 1..8 pattern. The option requires `--repack-dir` and validates dtype fp16, rank 2, shape[1] >= hidden, and ID < shape[0]. A single row prefix of length `hidden` is copied into `input_data`. All existing default checksums are unchanged. A CTest gate `spock_persistent_mlp_probe_full_real_weight_embedding_input_smoke` exercises full model-width real weights plus real embedding row 0 with residual. This is real layer.0 MLP weights plus real model embedding input, but still not inference: no RMSNorm, no token mixer, no real post-attention residual stream, no attention/DeltaNet, no LM head, not megakernel.

**Persistent MLP fp16 input file** (diary 0084). `vk_persistent_mlp_probe --input-fp16-file PATH` loads the input vector from a raw little-endian fp16 file instead of the synthetic 1..8 pattern. The file must contain at least `hidden * sizeof(uint16_t)` bytes; exactly the first `hidden` values are read. The option is mutually exclusive with `--input-token` and does not require `--repack-dir`. JSON output includes `input_fp16_file` only when provided. A CTest gate `spock_persistent_mlp_probe_fp16_input_smoke` uses a checked-in fixture matching the synthetic pattern to prove checksum identity (371183224). This creates a clean handoff point for real hidden-state captures from the Vulkan decode diagnostics (e.g. RMSNorm output or residual stream snapshots) without taking on RMSNorm fusion. Still not inference, not RMSNorm, not attention/DeltaNet, not LM head, not megakernel.


**Component FP16 extract tool** (diary 0085). `tools/extract_component_fp16.py` is a standalone Python utility that converts a `spock-decode --dump-step-components` JSON field (e.g. `input_hidden_fp16`, `post_mlp_fp16`) from a specified layer into a raw little-endian fp16 file. The CTest gate `spock_extract_component_fp16` exercises 10 unit tests covering success and failure modes. This bridges the decode diagnostics pipeline to `vk_persistent_mlp_probe --input-fp16-file`, enabling offline validation of persistent MLP compute against real hidden-state captures. It is offline diagnostic tooling: not inference, not a runtime component, and not the megakernel.

**Captured fp16 handoff output gate** (diary 0086). The persistent MLP probe now uses per-row fp16 output equality (`output_mismatches == 0`) as its authoritative pass/fail gate, with fp32 aggregate `checksum`/`expected_checksum` retained as diagnostic fields. The change was motivated by a real captured-input test: the real layer 0 step 1 `mixer_residual` vector (1024 fp16 values from `tests/data/layer0_step1_mixer_residual_1024.fp16`) produces exact fp16 output agreement across all 1024 rows, but the fp32 checksum diverges (67820897 vs 67824746). The root cause is a 1-fp16-ULP difference at intermediate row 3180 in the SiLU activation stage: GLSL `exp` and `std::exp` can produce different fp32 values at the rounding boundary, and that difference lands on a fp16 rounding edge. The down-projection's fp16 output rounding absorbs the intermediate difference, so the final output is exact. Per-row fp16 equality is the correct gate because it tests what the downstream consumer actually receives, not the internal accumulation representation. A CTest gate `spock_persistent_mlp_probe_captured_fp16_handoff` protects this real-captured-input path. Still not inference, not RMSNorm, not attention/DeltaNet, not LM head, and not the megakernel.

**Layer-selectable captured fp16 handoff** (diary 0087). `vk_persistent_mlp_probe` now accepts `--layer N` (default 0) to select which layer's MLP weights to load (`layer.N.mlp_gate/up/down`). The option validates N >= 0 and rejects non-integer input with JSON error/exit 2, including partial integer strings such as `1abc`. A new captured fixture `tests/data/layer1_step1_mixer_residual_1024.fp16` provides the layer 1 step 1 mixer_residual vector. Running with `--layer 1`, full model dimensions, and residual produces output_mismatches == 0 with exact checksum agreement (2888553996), confirming the fp16 output equality gate holds for two distinct layers. CTest gates: `spock_persistent_mlp_probe_layer1_captured_fp16_handoff` plus three invalid-input WILL_FAIL gates. Still not inference, not all layers validated, not RMSNorm, not attention/DeltaNet, not LM head, and not the megakernel.

**All-layer captured handoff sweep** (diary 0088). A run-only sweep extracted step-1 `mixer_residual_fp16` for all 24 layers and invoked the layer-selectable persistent MLP probe for each. 23/24 layers passed exact fp16 output equality. Layer 20 is the only hard failure: `output_mismatches == 2`, first row 657. Temporary diagnostics found exact up-scratch agreement and one activation-stage 1-ULP difference at intermediate row 1874 (`0x1D39` GPU vs `0x1D38` CPU). That single activation difference is absorbed for 1022 output rows but flips final fp16 rounding by 1 ULP for rows 657 and 954. This is a precision-policy boundary case, not a layout or barrier failure. Diary 0089 turns that finding into an explicit opt-in tolerance policy while keeping exact fp16 output equality as the default gate.

**FP16 ULP tolerance policy** (diary 0089). `vk_persistent_mlp_probe` now supports `--output-fp16-ulp-tolerance N` (default 0, exact). The probe reports `output_exact_mismatches`, `output_within_tolerance`, `output_mismatches`, `max_fp16_ulp_diff`, and `output_fp16_ulp_tolerance`. The default remains exact fp16 equality. With explicit tolerance, CPU-vs-GPU `exp` rounding differences (1 ULP at activation stage) are tolerated but still reported as exact mismatches. A checked-in layer 20 fixture `tests/data/layer20_step1_mixer_residual_1024.fp16` enables paired CTest gates: `spock_persistent_mlp_probe_layer20_captured_fp16_handoff_exact_fails` (WILL_FAIL, proves default rejects layer 20) and `spock_persistent_mlp_probe_layer20_captured_fp16_handoff_ulp1` (proves tolerance 1 passes). Three invalid-tolerance WILL_FAIL gates exercise parsing. GPU-vs-GPU comparisons remain exact. Still not inference, not RMSNorm, not attention/DeltaNet, not LM head, not megakernel.

**Pre-MLP RMSNorm smoke probe** (diary 0090). `vk_persistent_mlp_probe --pre-mlp-rmsnorm` applies RMSNorm to the input vector before gate/up projections using the `layer.N.post_norm` weight. Formula: `out[i] = fp16(input[i] * rsqrt(mean(input^2) + 1e-6) * (1 + weight[i]))`. The raw input is preserved for residual addition. Shader Stage 0 computes the full sum-of-squares independently per workgroup (all produce the same inv_rms), stripes normalized output writes, then a global barrier separates Stage 0 from Stage 1. Two new bindings: `NormOutput` (binding 8, fp16[hidden]) and `WeightNorm` (binding 9, fp16[hidden] from `layer.N.post_norm`). Push constants grew from 5 to 6 uint32s (added `pre_mlp_rmsnorm` flag). Barrier count increases from 2 to 3 when enabled. The option requires `--repack-dir`. CTest gate `spock_persistent_mlp_probe_pre_mlp_rmsnorm_smoke` passes with exact fp16 output equality (hidden=128, intermediate=16, layer 0, real post_norm weights, generation=3). A stronger gate, `spock_persistent_mlp_probe_full_pre_mlp_rmsnorm_residual_smoke`, covers the model-width layer-shaped use (`hidden=1024`, `intermediate=3584`, `output_rows=1024`, `workgroups=82`, `--residual`) with exact fp16 output equality. Full-width non-residual RMSNorm+MLP remains a CPU-vs-GPU reciprocal-square-root sensitivity case and requires explicit tolerance; it is documented as a boundary, not promoted to a target-path claim. This is the first RMSNorm integration in the persistent probe. Not full captured validation, not all layers swept, not inference, not attention/DeltaNet, not LM head, not megakernel.

**Captured RMSNorm+MLP expected-output gate** (diary 0091). `vk_persistent_mlp_probe` now supports `--expected-output-fp16-file PATH`, a raw fp16 output vector used as the authoritative comparison target when provided. The internal CPU reference checksum remains diagnostic. A new fixture `tests/data/layer0_step1_post_mlp_1024.fp16` was extracted from the same step-1 component dump as the existing layer-0 `mixer_residual` fixture. Exact comparison of persistent RMSNorm+MLP+residual output against captured runtime `post_mlp_fp16` currently fails with 314 exact mismatches and `max_fp16_ulp_diff == 87`; the paired CTest gates encode this explicitly: exact default is `WILL_FAIL`, while `--output-fp16-ulp-tolerance 87` passes. This bounds the current runtime-vs-persistent difference but does not claim exact captured RMSNorm+MLP parity. Still not token mixer integration, not layer-shaped persistent decode, not inference, not megakernel.

**Captured RMSNorm+down-projection boundary** (diary 0092). A second captured fixture, `tests/data/layer0_step1_down_output_1024.fp16`, compares the persistent probe without `--residual` against runtime `down_output_fp16`. Exact comparison fails on only 17 rows with `max_fp16_ulp_diff == 2`; the paired CTest gates mark exact as `WILL_FAIL` and require `--output-fp16-ulp-tolerance 2` to pass. This localizes the larger `post_mlp_fp16` spread from diary 0091: the down projection itself is tightly bounded, while residual addition amplifies those small differences in final fp16 ULP space. Still not exact captured parity, not activation scratch comparison, not inference, not megakernel.

**Captured RMSNorm activation-product boundary** (diary 0093). `vk_persistent_mlp_probe` now supports `--expected-mlp-product-fp16-file PATH`, which compares the Stage 2 activated MLP product in `gate_scratch_buf` against runtime `mlp_product_fp16`. Layer 0, step 1 exact comparison fails on 10 intermediate rows with `max_fp16_ulp_diff == 2`; the ULP-2 gate passes. This places the first observed runtime-vs-persistent difference before or at activation product generation, but it is tightly bounded. The next diagnostic should compare persistent RMSNorm output or gate/up scratch against runtime captures. Still not exact captured parity, not layer-shaped persistent decode, not inference, not megakernel.

**Captured RMSNorm output exact gate** (diary 0094). `spock-decode --dump-step-components` now emits `mlp_normed_fp16`, captured immediately after `post_norm(act_c) -> act_a` and before gate/up overwrite `act_a`. `vk_persistent_mlp_probe` now supports `--expected-norm-output-fp16-file PATH`, which compares Stage 0 `norm_output` against that capture and requires `--pre-mlp-rmsnorm`. Layer 0, step 1 passes exact fp16 equality (`norm_output_mismatches == 0`). This localizes the first observed 2-ULP activation-product mismatch to after RMSNorm, likely gate/up matvec or SiLU/product rounding. Still not gate/up scratch comparison, not exact activation parity, not inference, not megakernel.

**Captured RMSNorm up-projection boundary** (diary 0095). `spock-decode --dump-step-components` now emits `mlp_up_fp16`, and `vk_persistent_mlp_probe` supports `--expected-up-scratch-fp16-file PATH`, comparing persistent Stage 1 `up_scratch_buf` against runtime up projection output. Layer 0, step 1 exact comparison fails on 5 intermediate rows with `max_fp16_ulp_diff == 2`; the ULP-2 gate passes. Since RMSNorm output is exact, this places the first observed mismatch in projection math after RMSNorm. Gate projection remains to be captured because persistent `gate_scratch` is overwritten by Stage 2. Still not exact activation parity, not layer-shaped persistent decode, not inference, not megakernel.

**Captured RMSNorm gate-projection boundary** (diary 0096). `spock-decode --dump-step-components` now emits `mlp_gate_fp16`, and `vk_persistent_mlp_probe` supports `--expected-gate-scratch-fp16-file PATH`. The persistent shader preserves raw Stage 1 gate dots in binding 10 `raw_gate_scratch` while still running the full Stage 2 overwrite of `gate_scratch`, so the existing activation-product path remains intact. Layer 0, step 1 exact comparison fails on 7 intermediate rows with `max_fp16_ulp_diff == 1`; the ULP-1 gate passes. The layer-0 MLP-internal boundary stack is now mapped: RMSNorm exact, gate projection max 1 ULP, up projection max 2 ULP, activation product max 2 ULP, down output max 2 ULP, post-residual max 87 ULP. This completes the narrow scratch-boundary map for layer 0 but is still not layer-shaped persistent decode, not inference, not megakernel.

**Post-residual population gate** (diary 0097). `vk_persistent_mlp_probe` now emits output ULP population buckets (`output_ulp_le_1` through `output_ulp_gt_64`) and supports an opt-in population status gate with `--output-fp16-population-ulp-threshold N` plus `--output-fp16-max-rows-above-population-threshold N`. The layer-0 captured RMSNorm+MLP residual gate now requires `--output-fp16-ulp-tolerance 87` and at most 10 of 1024 rows above 16 ULP; a paired WILL_FAIL test with max 9 proves the population gate is active. This shows the max-87 post-residual difference is a narrow tail, not a broad shift. Still not exact post-residual parity, not representative-layer sweep, not layer-shaped persistent decode, not inference, not megakernel.

**Layer 20 RMSNorm+MLP population gate** (diary 0098). A new fixture `tests/data/layer20_step1_post_mlp_1024.fp16` extends captured RMSNorm+MLP residual comparison to a mid-network layer using the existing layer-20 `mixer_residual` input. Exact comparison fails on 185 rows with `max_fp16_ulp_diff == 209`; the bounded gate requires `--output-fp16-ulp-tolerance 209` and at most 1 row above 16 ULP. A paired population max-0 WILL_FAIL test proves the tail gate is active. This gives representative mid-network evidence that the broad population remains tight while a sparse residual tail can be larger than layer 0. Still not all-layer sweep, not token mixer integration, not layer-shaped persistent decode, not inference, not megakernel.

**Mixer output residual-add gate** (diary 0099). `spock-decode --dump-step-components` now emits `mixer_output_fp16` and `mixer_output_norm`, staging `B.act_b` before the first residual add overwrites the layer handoff state. A new `vk_residual_add_probe` uses the existing `residual_add.comp` shader to verify `input_hidden_fp16 + mixer_output_fp16 -> mixer_residual_fp16`. Layer 0, step 1 passes exact fp16 equality against the existing `layer0_step1_mixer_residual_1024.fp16` fixture. This closes the first token-mixer handoff equation before implementing a persistent DeltaNet or attention mixer. Still not token-mixer computation parity, not full layer-shaped persistent decode, not inference, not megakernel.

**Current megakernel execution philosophy** (diary 0100). The runtime strategy is
to keep using the observable conventional Vulkan path as the source of captured
checkpoints while persistent probes absorb one contract at a time. Diary 0100
set the next token-mixer strategy: prove the DeltaNet output projection first,
reuse the residual-add gate downstream, then walk backward through recurrent
state, short-convolution state, scratch reuse, and layer-shaped persistent
execution. This is documentation of the execution plan, not a runtime feature
and not megakernel completion.

**Vulkan matvec handoff probe** (diary 0101). `vk_matvec_probe` exercises `matvec.comp` with real fp16 model weights and captured activations. The first target is layer 0 DeltaNet output projection: `dn_gated_fp16 [2048] x layer.0.delta_out_proj [1024, 2048] -> dn_out_fp16 [1024]`. Layer 0, step 1 passes exact fp16 equality (all 1024 output rows bit-for-bit). This validates the matvec shader at model width outside the decode loop and closes the DeltaNet output projection handoff equation for one layer/step. Not token-mixer computation parity, not full layer-shaped persistent decode, not inference, not megakernel.

**DeltaNet norm-gate probe** (diary 0102). `vk_deltanet_norm_gate_probe` exercises `deltanet_norm_gate.comp` with captured `dn_core_fp16`, captured `dn_z_fp16`, and real fp32 `layer.0.delta_norm` weight. Layer 0, step 1 passes exact fp16 equality for `dn_core + dn_z + delta_norm -> dn_gated` across all 2048 values. Together with diary 0101 and diary 0099, the downstream DeltaNet path is now closed from captured recurrent core through gated vector, output projection, and mixer residual add. Still not recurrent core parity, not full token-mixer parity, not layer-shaped persistent decode, not inference, not megakernel.

**DeltaNet z-projection matvec gate** (diary 0103). `vk_matvec_probe` now gates `dn_input_norm_fp16 + layer.0.delta_in_proj_z -> dn_z_fp16` for layer 0, step 1. The comparison is exact across all 2048 z-gate values. A qkv projection comparison was intentionally not promoted because the current dumped `dn_q_fp16`/`dn_k_fp16` values are post-L2-normalized, not raw projection outputs. Still not qkv/split/L2 parity, not recurrent core parity, not layer-shaped persistent decode, not inference, not megakernel.

**DeltaNet raw qkv projection gate** (diary 0104). `spock-decode --dump-step-components` now emits `dn_qkv_raw_fp16`, captured immediately after `delta_in_proj_qkv` and before conv/L2 mutation. `vk_matvec_probe` gates `dn_input_norm_fp16 + layer.0.delta_in_proj_qkv -> dn_qkv_raw_fp16` exactly for all 6144 raw q/k/v values. This corrects the invalid diary-0103 qkv comparison attempt by adding the right expected fixture boundary. Still not conv1d parity, not q/k L2 parity, not g/beta or recurrent core parity, not layer-shaped persistent decode, not inference, not megakernel.

**DeltaNet A/B projection gates** (diary 0105). `spock-decode --dump-step-components` now emits raw `dn_a_fp16` and `dn_b_fp16`. `vk_matvec_probe` gates `dn_input_norm_fp16 + layer.0.delta_in_proj_a -> dn_a_fp16` and `dn_input_norm_fp16 + layer.0.delta_in_proj_b -> dn_b_fp16` exactly for all 16 heads. The stateless DeltaNet projection fanout from `dn_input_norm_fp16` is now covered: qkv raw, z, a, and b. Still not g/beta computation, conv/L2, recurrent core, layer-shaped persistent decode, inference, or megakernel.

**DeltaNet g/beta probe** (diary 0106). `spock-decode --dump-step-components` now emits exact `dn_g_bits` and `dn_beta_bits` alongside decimal g/beta values. `vk_deltanet_g_beta_probe` runs `deltanet_compute_g_beta.comp` from captured `dn_a_fp16`, captured `dn_b_fp16`, and repacked `delta_a_log`/`delta_dt_bias`, then compares the 32 fp32 output bit patterns exactly. Layer 0, step 1 passes with zero bit mismatches. Still not conv/L2, recurrent core, layer-shaped persistent decode, inference, or megakernel.

**DeltaNet conv/L2 probe** (diaries 0108/0109). `vk_deltanet_conv_l2_probe` proves
layer-0 conv1d mutation and q/k L2 normalization produce exact fp16-bit-identical
output to the captured runtime handoff tensors. The probe loads `dn_qkv_raw_fp16`,
`dn_conv_state_pre_fp16`, and `delta_conv` weights, runs the unfused shader sequence
(`conv1d_step.comp` + `l2_norm_per_head.comp` on Q and K slices), and compares all
three output slices against captured `dn_q_fp16`, `dn_k_fp16`, `dn_v_fp16` with zero
tolerance. The pre-conv fixture was regenerated after adding a shader-write ->
transfer-read buffer barrier for the state capture. CTest gate:
`spock_deltanet_conv_l2_probe_layer0_exact`. Still not recurrent core, not
layer-shaped persistent, not inference, not megakernel.

**DeltaNet recurrent core probe** (diary 0112). `vk_deltanet_recurrent_probe` proves
the layer-0 recurrent core produces exact fp16-bit-identical output to the captured
`dn_core_fp16` handoff tensor. The probe loads q/k/v fp16 vectors, g/beta u32 bits,
the pre-recurrent fp32 state, and expected output, then dispatches
`deltanet_recurrent.comp` with one workgroup per head. The state tail is unconditionally
overwritten from the independently-gated g/beta bits file. Layer 0, step 1 passes with
`output_mismatches: 0`, `max_fp16_ulp_diff: 0`. CTest gates:
`spock_deltanet_recurrent_probe_help`, `spock_deltanet_recurrent_probe_layer0_exact`.
This closes the DeltaNet backward-validation ladder for layer 0. Still not fused shader
parity, not layer-shaped persistent, not inference, not megakernel.

**DeltaNet mixer composed probe** (diary 0113). `vk_deltanet_mixer_probe`
chains all eleven mixer stages -- QKV/Z/A/B projections, conv1d, L2 q/k norm,
g/beta computation, recurrent update, norm-gate, output projection, and residual
add -- into a single Vulkan submit with memory barriers between dependent
stages. It loads eight weight matrices from the repacked artifact and uses a
shared `qkv_buf` with sub-allocated Q/K/V sections matching the production
decode layout. Layer 0, step 1 passes with zero mismatches on both
`mixer_output` (1024 fp16) and `mixer_residual` (1024 fp16). CTest gates:
`spock_deltanet_mixer_probe_help`, `spock_deltanet_mixer_probe_layer0_exact`.
This closes the DeltaNet composed validation for layer 0: every individual
stage (diaries 0099-0112) and the full end-to-end pipeline produce exact
fp16-bit-identical output. Not multi-layer, not fused shader variants, not
persistent dispatch, not the megakernel.

**Persistent layer-0 tail probe** (diary 0114). `vk_persistent_layer0_probe`
establishes the first layer-shaped persistent scaffold with `local_size_x=128`
and 82 workgroups for the post-mixer tail:
`mixer_residual -> post_norm RMSNorm -> MLP gate/up -> SiLU product -> down
-> residual add -> post_mlp`. The 128-lane workgroup shape matches DeltaNet
recurrent's one-workgroup-per-head layout, validating that persistent execution
at this lane count is correct before adding DeltaNet mixer stages. Layer 0, step 1:
structural checks pass (generation=3, failures=0, arrived=0), GPU checksum
matches 128-lane CPU reference exactly. Output comparison against captured runtime
`post_mlp_fp16` has 314 exact mismatches with max 87 ULP (same boundary as
64-lane probe), passing bounded gate with tolerance 87, population threshold 16,
max 10 rows above threshold. CTest gates:
`spock_persistent_layer0_probe_help`,
`spock_persistent_layer0_probe_post_mlp_exact_fails` (WILL_FAIL),
`spock_persistent_layer0_probe_post_mlp_bounded`.
Not full layer persistence (no DeltaNet mixer), not inference, not megakernel.

**Persistent layer-0 projection-prefix gate** (diary 0116). `vk_persistent_layer0_probe`
now supports `--mode projections` (default `tail`). In projection mode, the persistent
shader computes the stateless DeltaNet projection fanout from `dn_input_norm_fp16`:
qkv_raw (6144), z (2048), a (16), b (16) with `local_size_x=128` and 82 workgroups.
Exact comparison against captured runtime fixtures produces 8 qkv_raw mismatches and
1 z mismatch, all at 1 ULP -- a 128-lane reduction-order boundary, not a GPU-vs-CPU
boundary. The `--projection-fp16-ulp-tolerance 1` bounded gate passes with zero failures.
A WILL_FAIL CTest preserves the exact-failure boundary. Not full mixer, not full layer,
not inference, not the megakernel.

**Persistent layer-0 conv/L2 gate** (diary 0117). `vk_persistent_layer0_probe` now
supports `--mode conv-l2`, advancing the next narrow persistent DeltaNet boundary:
captured `dn_qkv_raw_fp16` plus captured `conv_state_pre` plus repacked
`layer.0.delta_conv` run through row-strided depthwise conv1d + SiLU, one software
global barrier, head-wise L2 normalization for q and k, and v copyout. The persistent
path uses the same 128-lane execution shape and 82-workgroup residency as the other
layer-0 gates. On the RX 6750 XT validation path, q, k, and v all match the captured
runtime fixtures exactly with `generation == 1`, `arrived == 0`, and `failures == 0`.
CTest gate: `spock_persistent_layer0_probe_conv_l2_exact`. Not full mixer, not full
layer, not inference, not the megakernel.

**Persistent layer-0 g/beta gate** (diary 0118). `vk_persistent_layer0_probe`
now supports `--mode g-beta`, computing the DeltaNet scalar branch from captured
`dn_a_fp16`, captured `dn_b_fp16`, repacked `layer.0.delta_a_log`, and repacked
`layer.0.delta_dt_bias`. The shader mirrors `deltanet_compute_g_beta.comp` and
emits exact fp32 bit patterns in g-then-beta order. Layer 0, step 1 passes with
`g_beta_bit_mismatches == 0`, `generation == 0`, `arrived == 0`, and
`failures == 0`. CTest gate: `spock_persistent_layer0_probe_g_beta_exact`.
The standalone control-payload output is a probe layout, not the final recurrent
state layout. Not recurrent, not full mixer, not full layer, not inference, not
the megakernel.

## Measurement Hooks


The runtime must expose timing boundaries for:

- prefill
- one-token decode
- `tg128` decode loop
- individual major blocks during tuning
- command-buffer submission overhead
- GPU timestamp regions when supported

Every benchmark must state whether reported timing is GPU-only, host end-to-end, or both.

### GPU Timestamp Decode Instrumentation

`SPOCK_GPU_TIMESTAMPS=1` (diary 0040) is an opt-in measurement gate that
brackets the decode command buffer with Vulkan `VK_QUERY_TYPE_TIMESTAMP`
queries. When active, the `spock-decode` JSON output includes:

- `gpu_decode_us` -- total GPU decode command buffer execution time in
  microseconds
- `per_token_gpu_us` -- per-token GPU execution time array in microseconds

These are reported alongside the always-present host-side fields
`prefill_ms`, `decode_ms`, and `per_token_ms`. The timestamp fields are
absent when the gate is disabled or when queries were not recorded for a
step.

The gate measures GPU-side execution time for single-submit-eligible steps
(full embedding + 24 layers + final norm + LM head + argmax) and the
`skip_layers` LM-head-only first decode step after chunk prefill. It does
not alter inference output; parity is preserved with the gate active.

This is a measurement instrument, not a performance optimization. It is
NOT full GPU offload, NOT persistent dispatch, and NOT the megakernel.
Default-off; no timestamp queries are allocated or recorded without the
env var. Still env-gated, not default.

`SPOCK_GPU_BLOCK_TIMESTAMPS=1` (diary 0044) refines that measurement when
`SPOCK_GPU_TIMESTAMPS=1` is also active. It records coarse regions only for
single-submit-eligible decode steps and emits `gpu_region_us` from
`spock-decode` when data exists. The current regions are `embedding`,
`layer_0` through `layer_23`, `final_norm`, `lm_head`, and `argmax`. Default
output and timestamp-only output remain unchanged. This is still measurement
only, not full GPU offload, persistent dispatch, or the megakernel.

### Tiled LM-Head Decode Matvec

`SPOCK_GPU_LM_HEAD_TILED=1` (diary 0045) is a default-off final-LM-head-only
decode gate. It replaces the generic `matvec.comp` LM-head dispatch with
`lm_head_tiled.comp`, which computes eight vocabulary rows per workgroup and
reduces each row dot product across 64 lanes.

The shader reuses the existing three-binding LM-head descriptor set:

- binding 0: LM-head weights fp16, row-major `[VOCAB, HIDDEN]`
- binding 1: final hidden vector fp16
- binding 2: logits fp16

No new descriptor layout or per-layer descriptor coverage is needed. The gate
does not affect MLP projections, attention projections, DeltaNet projections,
or diagnostic LM-head calls. A local 8-token timestamp sample showed the
LM-head region dropping from about 2.61e+06 us to about 3.84e+04 us, but the
gate remains default-off pending broader timing coverage. This is not full GPU
offload, persistent dispatch, or the megakernel.


### Tiled Decode Matvec

`SPOCK_GPU_MATVEC_TILED=1` (diary 0046) is a default-off decode gate that
replaces the generic `matvec.comp` with `matvec_tiled.comp` for all general
matvec dispatches in the main decode path. The tiled shader uses BLOCK_ROWS=8,
64 lanes, and a strided `j += 64` inner loop so arbitrary `in_dim` values
work without alignment requirements. Accumulation is fp32; output is fp16.

The shader reuses the existing three-binding descriptor layout:

- binding 0: weights fp16, row-major `[out_dim, in_dim]`
- binding 1: input vector fp16 `[in_dim]`
- binding 2: output vector fp16 `[out_dim]`

The gate covers: attention Q/K/V/O projections, DeltaNet merged QKV/Z/A/B
projections, DeltaNet out_proj, MLP gate/up/down projections, and the final LM
head fallback when `SPOCK_GPU_LM_HEAD_TILED` is not active. It does not
change `matvec_f32_out`, `cmd1` fallback dispatches, diagnostic calls, or the
default path. No new descriptor layout or per-layer descriptor coverage is
needed.

A local 8-token timing sample with both `SPOCK_GPU_MATVEC_TILED=1` and
`SPOCK_GPU_LM_HEAD_TILED=1` under the full fused single-submit gate stack
reported `gpu_decode_us` about 157679 and per-layer regions about 4.8-5.5ms,
compared to the previous tiled-LM-only `gpu_decode_us` of about 2.31e+06.
Directional only, not formal benchmark. Reduction order changes under the
gate; parity is checked at the argmax level. This is not full GPU offload,
persistent dispatch, or the megakernel.

## Go / No-Go Rule

If the runtime cannot prove stable cross-workgroup synchronization on this GPU and driver, pivot to `single_submit` and benchmark that path. A stable single-submit engine is a valid outcome; an unstable persistent dispatch is not.

## Observed Device Properties

Values recorded from the local RADV stack during decode pipeline bring-up:

| Property | Expected | Observed |
| --- | --- | --- |
| Device name | AMD Radeon RX 6750 XT | AMD Radeon RX 6750 XT (RADV NAVI22) |
| Subgroup size | 32 | 64 |
| Max shared memory | 64 KiB | 64 KiB (65536 bytes) |
| Max workgroup invocations | 1024 | 1024 |
| Vulkan API version | 1.2+ | 1.4.318 |
| fp16 shader support | yes | yes |
| bf16 shader support | no | no |
| Cooperative matrix | no | no |

Note that the subgroup size is 64, not the originally assumed 32. This affects:
- matvec workgroup sizing (currently 64, which is correct)
- rms_norm shared-memory reduction array sizing (256, independent of subgroup)
- Future persistent kernel design (barrier probe needs re-evaluation with wg=64)

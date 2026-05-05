# 0018: Combined GPU Prefill Pipeline Probe

## Goal

Consolidate the verified state of the DeltaNet prefill GPU-offload work after
the standalone collection probe and chunk-prefill probe were connected into one
combined GPU pipeline probe.

The purpose of this checkpoint is narrow but important: prove that the
`deltanet_prefill_collect.comp` shader can populate the exact fp32
head-major buffers consumed by `deltanet_chunk_prefill.comp`, without a CPU
intermediate packing step between the two GPU stages.

## Current Verified State

### Proved (independently verified in this session)

| Probe | Command | Status | Metric |
|---|---|---|---|
| GPU prefill collection | `spock-deltanet-prefill-collect-probe` | `compare-ok` | max_rel=0 for all 5 tensors, nan_count=0 at heads=16, seq_len=104, k_dim=v_dim=128 |
| GPU chunk prefill (multi-head, padded, submit) | `spock-deltanet-chunk-prefill-probe --case runtime-l2-padded-submit` | `compare-ok` | max_rel_core=1.19e-7, max_rel_state=1.19e-7 across all 16 heads, nan_count=0 |
| Combined GPU collect → chunk prefill | `spock-deltanet-prefill-pipeline-probe` | `compare-ok` | max_rel_core=8.94e-8, max_rel_state=1.19e-7, nan_count=0 |

All three probes compiled and executed on the RADV RX 6750 XT (NAVI22)
target hardware, confirming the GPU shaders produce numerically correct output
against CPU reference implementations.

### Confirmed runtime behavior

| Test | GPU Env Gate | Max New Tokens | Result | Elapsed |
|---|---|---|---|---|
| `short_correctness_001` | `SPOCK_GPU_CHUNK_PREFILL=1` | 6 | matches reference | prior verified run |
| `mixed_correctness_023` | `SPOCK_GPU_CHUNK_PREFILL=1` | 1 | generated [561], matches reference | ~138s |
| `pp520_046` | `SPOCK_GPU_CHUNK_PREFILL=1` | 1 | confirmed by prior runs in diary 0017 | N/A (prompt too long for quick re-run timeout) |

The GPU chunk-prefill path has been checked against focused runtime prompts,
including a 6-token `short_correctness_001` run. Longer prompts and broader
prompt sets are too slow for routine re-verification at every checkpoint
because of the per-head submit workaround; numerical confidence comes from the
standalone probes above and the per-layer diagnostic comparison
(`SPOCK_GPU_CHUNK_PREFILL_COMPARE=1`).

### What remains NOT achieved

- **Full GPU offload.** The runtime GPU chunk-prefill still depends on
  CPU-collected Q/K/V/g/beta vectors. Only the chunk-rule B-matrix computation
  has been moved to GPU in `DecodeSession`. GPU-side collection is now proven
  both by itself and as the producer for the chunk-prefill shader, but it has
  not been wired into the runtime session.
- **Efficient per-head dispatch.** Conservative per-head submit/wait (one
  command buffer per head) is currently required for correctness with
  realistic padded multi-head data. Pipeline barriers inside a single command
  buffer are insufficient on RADV NAVI22 for this workload. This makes the
  GPU path ~100× slower than the CPU path and prohibitive for routine testing
  beyond max-new-tokens=1.
- **Precision parity.** `short_correctness_003` continues to fail at generated
  token index 5 (token 16 vs expected token 12) on both CPU and GPU decode
  paths. No single shader bug has been identified; the drift profile implicates
  accumulated fp16 rounding at activation/output boundaries across the
  24-layer pipeline, with the largest step-ups at full-attention layers 3, 7,
  11, and 15. Neither the fp32-attention-o-proj-residual nor
  fp32-MLP-down-proj-residual experiments closed the gap.
- **Vulkan-native megakernel.** The goal remains unachieved. The CPU chunk
  bridge, the precision gap, and the per-head submit inefficiency are the
  three remaining blockers. Full GPU offload requires all three to be resolved.

## Artifact Baseline (committed state)

The following artifacts are committed and constitute the checkpoint baseline:

- `shaders/deltanet_prefill_collect.comp` — GPU prefill collection/packing
  shader (proven exact: max_rel=0 vs CPU reference)
- `apps/spock-deltanet-prefill-collect-probe.cpp` — standalone probe for the
  collection shader (committed, verified heads=16, seq_len=104, k_dim=v_dim=128)
- `apps/spock-deltanet-prefill-pipeline-probe.cpp` — combined probe proving
  collect output can feed chunk prefill directly (verified heads=16,
  seq_len=104, total_seq=128, chunk_size=64)
- `shaders/deltanet_chunk_prefill.comp` — GPU chunk-prefill shader (9 probe
  cases, passes with per-head submit workaround)
- `apps/spock-deltanet-chunk-prefill-probe.cpp` — standalone chunk-prefill
  probe (9 cases, including `runtime-l2-padded-submit` at max_rel ~1.19e-7)
- `src/runtime/vk_session.cpp` — `gpu_chunk_prefill()` method with
  `SPOCK_GPU_CHUNK_PREFILL=1` env gate (committed, verified on hardware)

Git log (most recent entries):
```
97c46c7 Add GPU DeltaNet prefill collection probe
d610585 Gate GPU DeltaNet chunk prefill runtime
79e72ed Advance Vulkan DeltaNet prefill probe
e50234d Add native DeltaNet chunk-rule helper
2ede37f Fix Qwen3.5 RoPE rotation parity
```

The combined pipeline probe was added after commit `97c46c7` and is expected
to be committed as the next checkpoint.

## Combined GPU Prefill Pipeline Probe

The new probe performs the exact standalone handoff that runtime integration
needs:

1. Generate deterministic Q/K/V/g/beta inputs in the runtime-like padded
   shape: 16 heads, seq_len=104, total_seq=128, k_dim=v_dim=128, chunk_size=64.
2. L2-normalize q and k per head/token before fp16 conversion, matching the
   proven `runtime-l2-padded-submit` case.
3. For each token, fill the per-token fp16 `dn_qkv` buffer and fp32 `g_beta`
   buffer, then dispatch `deltanet_prefill_collect.comp`.
4. Bind the collected fp32 head-major Q/K/V/g/beta buffers directly as inputs
   to `deltanet_chunk_prefill.comp`.
5. Dispatch chunk prefill with the known-correct per-head separate-submit
   workaround.
6. Compare valid-token core output and final recurrent state against
   `run_deltanet_chunk_rule`.

The independent verification result was:

```json
{
  "shader": "deltanet_prefill_collect+deltanet_chunk_prefill",
  "status": "compare-ok",
  "num_heads": 16,
  "seq_len": 104,
  "total_seq": 128,
  "chunk_size": 64,
  "chunk_count": 2,
  "max_rel_core": 8.9407e-08,
  "max_rel_state": 1.19175e-07,
  "max_abs_core": 8.9407e-08,
  "max_abs_state": 2.38419e-07,
  "nan_count": 0
}
```

This does not yet make the production runtime GPU-native. It removes the main
layout-risk argument against replacing the CPU collection bridge: the producer
and consumer shaders now agree on buffer shape, precision, and indexing.

## Verification

The verified state was established by the parity runs, artifact checks, and
combined prefill-pipeline probe results recorded above. The purpose of this
checkpoint was to freeze what could be trusted before additional GPU handoff
work: the repacked artifact baseline, native chunk-rule behavior, and the first
combined collect-to-chunk probe. It deliberately did not claim full GPU offload,
persistent dispatch, or a megakernel path.

## Current Limitations

1. **CPU chunk bridge remains the default prefill path.** `run_chunk_prefill`
   → `run_deltanet_chunk_rule` on CPU. The GPU path is gated behind
   `SPOCK_GPU_CHUNK_PREFILL=1` and only moves the chunk-rule computation;
   Q/K/V/g/beta collection, rearrangement, upload, and output re-integration
   are still CPU-hosted.

2. **Per-head submit workaround is correct but prohibitively slow.**
   24 layers × 16 heads = 384 submit-wait cycles per chunk. A correct
   efficient shader design (single dispatch, all heads) is needed.

3. **Pipeline barrier insufficiency is device/driver-specific.** Only
   validated on RADV NAVI22 (RX 6750 XT). The separate-CB workaround is
   correct everywhere but slow everywhere.

4. **No formal CI tests for the GPU path.** The env-gated path is verified
   manually with specific prompts and the `SPOCK_GPU_CHUNK_PREFILL_COMPARE`
   diagnostic. There is no automated regression gate.

5. **No device-local buffers.** All GPU buffers in the chunk-prefill path are
   host-visible. Device-local memory with explicit staging is prerequisite for
   performance.

6. **Long-prompt behavior unexercised.** The probe exercises seq_len up to
   104 (2 chunks). Production prompts span many dozens of chunks.

7. **`short_correctness_003` still fails** at generated token index 5.
   The fp32-output experiments (attention o_proj residual, MLP down residual)
   did not close the gap. The working hypothesis remains accumulated fp16
   rounding at activation/output boundaries. Not addressed by this work.

8. **Precision parity is not achieved.** This checkpoint is a GPU prefill
   integration proof, not a parity milestone. The megakernel roadmap requires
   exact greedy parity.

## Next Work

### Primary: Wire GPU collection into `DecodeSession`

The next integration step is runtime wiring, preferably behind a separate
diagnostic env gate before replacing the existing path:

1. Allocate session-owned collection buffers for Q/K/V/g/beta.
2. Dispatch `deltanet_prefill_collect.comp` during layer-major prefill from
   the real per-token GPU activation buffers.
3. Compare GPU-collected buffers against the existing CPU-collected
   `prefill_chunks_` path on real activations.
4. Feed the GPU-collected buffers into `gpu_chunk_prefill()`.
5. Only after runtime correctness is stable, remove or bypass the CPU
   collection bridge on the env-gated path.

### Secondary: Precision drift

The precision investigation from diaries 0015–0017 remains unresolved. The
options, unchanged:

1. Broader decode-path fp32-resident activation experiment (beyond the
   targeted `o_proj` and `down_proj` residual probes).
2. Accept the parity tradeoff against the megakernel roadmap — revisit after
   full GPU offload is achieved.

# 0020: GPU Chunk-Prefill From GPU Collection — Avoiding CPU Intermediate Packing

## Goal

Wire the GPU-collected Q/K/V/g/beta buffers (proven numerically identical to
CPU-collected in diary 0019) directly into `gpu_chunk_prefill()` as the
chunk-prefill shader inputs, behind a new env gate. This avoids the CPU
intermediate packing and upload step for the chunk-prefill inputs on the gated
path — every buffer feeding `deltanet_chunk_prefill.comp` for a DeltaNet layer
stays in device-local memory end-to-end.

This is a runtime integration milestone, not a full GPU offload. The CPU
collection bridge and its `prefill_chunks_` data structure remain in the
default path and serve as fallback and diagnostic reference. The next
evolution — removing the CPU bridge entirely from the gated path — is
explicitly deferred.

## Implementation Work Completed

### New env gate: `SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1`

A new environment variable gate was added in `src/runtime/vk_session.cpp`.
It is meaningful only when `SPOCK_GPU_CHUNK_PREFILL=1` is also set. When both
gates are active:

1. **During each token's DeltaNet recurrent step (inside the per-layer prefill
   loop),** the collect dispatch (`deltanet_prefill_collect.comp`) writes
   Q/K/V/g/beta into persistent device-local buffers `dn_persist_q/k/v/g/beta`.
   These buffers have one aligned segment per DeltaNet layer.

2. **After prefill, inside `run_chunk_prefill()`,** the gated path calls
   `gpu_chunk_prefill_from_gpu_collect()` instead of the existing
   `gpu_chunk_prefill()` upload path. This function:
   - Takes the device-local collection buffers directly as input along with
     the layer index and prompt length.
   - Updates the chunk-prefill descriptor set to point at the per-layer
     segments of the collection buffers (each layer's data is contiguous
     within the full-prompt-size allocation).
   - Dispatches `deltanet_chunk_prefill.comp` using the standard per-head
     submit workaround (same as the existing `SPOCK_GPU_CHUNK_PREFILL=1`
     path, same correctness properties).

3. **The CPU collection bridge still runs.** The existing `prefill_chunks_`
   population is intentionally left in place for fallback and diagnostic
   comparison. The new proof is narrower: the chunk-prefill shader no longer
   receives its Q/K/V/g/beta inputs from CPU-packed/uploaded temporary buffers
   on the new gated path.

### Per-layer descriptor offsets with conservative alignment

The five persistent collection buffers (`dn_persist_q/k/v/g/beta`) are
allocated once per prefill call and can hold the current prompt length for all
DeltaNet layers. Each layer writes into a
contiguous segment of these buffers. The per-layer descriptor offset into each
buffer is computed as:

```
offset = layer_index * aligned_segment_bytes
```

with the conservative 256-byte alignment guarantee: the allocation base is
aligned to 256 bytes, and per-layer offsets are computed such that each
layer's segment starts at a position at least 256-byte aligned from the buffer
base. This avoids potential alignment constraints from any downstream
descriptor or buffer-address usage without coupling the runtime to a
particular device's minimum alignment requirement.

### Pipeline and module reuse

No new pipelines or shader modules were added. The new code path reuses:
- The existing `deltanet_prefill_collect` pipeline (from diary 0019) for the
  per-token collection dispatch.
- The existing `deltanet_chunk_prefill` pipeline (from the
  `SPOCK_GPU_CHUNK_PREFILL=1` path) for the chunk-rule shader.

The `gpu_chunk_prefill_from_gpu_collect()` function is a thin wrapper that
re-points descriptor bindings to the device-local collection buffers before
calling the same dispatch loop.

### Header changes

- `src/runtime/vk_session.hpp`: Added `gpu_chunk_prefill_from_gpu_collect()`
  method declaration and `dn_persist_q/k/v/g/beta` buffer members for lifetime
  management across `layer_major_prefill()` and `run_chunk_prefill()`.

## Inference Concepts

### Why GPU collection avoids CPU packing

The default DeltaNet prefill path works in two phases per layer:

1. **Token loop (GPU → CPU):** For each token in the prompt, a GPU dispatch
   produces per-token Q/K/V/g/beta activations. These are downloaded from
   device-local to host-visible staging, then copied to CPU-accessible memory.
   Each token's data is appended to `prefill_chunks_` in token-major order.

2. **Chunk prefill (CPU → GPU):** After all tokens are collected on the CPU,
   the host-side `run_deltanet_chunk_rule` processes the full prompt's data
   in fp32. The resulting chunk output (core + state) is uploaded back to
   device-local buffers.

   In the existing `SPOCK_GPU_CHUNK_PREFILL=1` path, the chunk-rule
   computation itself happens on GPU instead of CPU, but the
   Q/K/V/g/beta inputs are still CPU-collected in phase 1 and then re-uploaded
   to GPU in fp32 head-major layout.

The new gate eliminates the CPU round-trip for the chunk-prefill inputs.
Phase 1 still runs on GPU, but the collection shader writes directly into
device-local head-major buffers. Phase 2 reads those same device-local
buffers without any host-side intermediate copy. The only remaining CPU work
is the orchestration (dispatch submission, descriptor updates) and the
subsequent layer output reintegration.

### When the gate is meaningful

`SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1` has no effect unless
`SPOCK_GPU_CHUNK_PREFILL=1` is also set. When only
`SPOCK_GPU_CHUNK_PREFILL=1` is set (the existing gate), the runtime uses
CPU-collected Q/K/V/g/beta — the collect shader is not dispatched, and
`prefill_chunks_` is populated normally. Both gates together activate the
GPU-collected path.

This two-gate design preserves the ability to:
- Run default (all-CPU) with neither gate.
- Validate GPU chunk-prefill with CPU-collected inputs (existing gate only).
- Validate GPU-collected feeds GPU chunk-prefill (both gates).
- Compare the two GPU paths for agreement (both gates + compare flag).

### 256-byte alignment safety

Vulkan requires `VkDescriptorBufferInfo::offset` to be a multiple of the
device's `minStorageBufferOffsetAlignment`, which is typically 16–256 bytes.
The conservative 256-byte alignment guarantee means the runtime does not need
to query this limit or handle per-device alignment variance — 256 bytes is a
safe upper bound for all known Vulkan implementations. Future code could
relax this, but the cost of 256-byte alignment is negligible (at worst 255
wasted bytes per layer's segment boundary).

## Verification

All verification commands were run on the target RADV RX 6750 XT (NAVI22)
hardware.

### Build

```sh
cmake --build build -j
```

Passed cleanly (no compilation or linking errors). Only C++ runtime code
changed; shaders and probes are unchanged.

### CTest regression gate (4/4 passed)

```sh
ctest --test-dir build --output-on-failure -R "spock_capabilities|spock_deltanet_chunk_unit|spock_vk_decode_prefill_handoff_mismatch|spock_diagnose_handoff_mc023"
```

All four targeted tests pass.

### Correctness: `short_correctness_001 --max-new-tokens 1`

**CPU-collect compare diagnostic (baseline):**

```sh
SPOCK_GPU_COLLECT_PREFILL_COMPARE=1 \
  python3 tests/run_vk_decode_parity.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --reference tests/data/reference_tokens.jsonl \
  --ids short_correctness_001 \
  --max-new-tokens 1
```

```
{"status":"ok","checked":1,"failures":[]}
```

**GPU chunk-prefill from CPU collect (existing gate, confirmation):**

```sh
SPOCK_GPU_CHUNK_PREFILL=1 \
  python3 tests/run_vk_decode_parity.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --reference tests/data/reference_tokens.jsonl \
  --ids short_correctness_001 \
  --max-new-tokens 1
```

```
{"status":"ok","checked":1,"failures":[]}
```

**GPU chunk-prefill from GPU collect (new gate):**

```sh
SPOCK_GPU_CHUNK_PREFILL=1 SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1 \
  python3 tests/run_vk_decode_parity.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --reference tests/data/reference_tokens.jsonl \
  --ids short_correctness_001 \
  --max-new-tokens 1
```

```
{"status":"ok","checked":1,"failures":[]}
```

**Compare gate on the new path (GPU-collect vs CPU-collect cross-check):**

```sh
SPOCK_GPU_CHUNK_PREFILL=1 SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1 SPOCK_GPU_CHUNK_PREFILL_COMPARE=1 \
  python3 tests/run_vk_decode_parity.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --reference tests/data/reference_tokens.jsonl \
  --ids short_correctness_001 \
  --max-new-tokens 1
```

```
{"status":"ok","checked":1,"failures":[]}
```

### Verbose per-layer diagnostic

Running `spock-decode` directly with `SPOCK_GPU_CHUNK_PREFILL=1
SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1 SPOCK_GPU_CHUNK_PREFILL_COMPARE=1`
on the `short_correctness_001` prompt (seq_len=9) generated token 271 and
printed `source=gpu_collect` chunk compare lines for all 18 DeltaNet layers.
Key metrics from the observed run:

- **nan_count=0** across all layers and all tensors.
- **max_abs:** core <= `5.960464e-08`, state <= `4.768372e-07`
  in the observed run. Relative values can appear large when the CPU
  reference value is near zero, but the absolute error stays tiny.

### Standalone probe regression

All standalone probes continue to pass:

| Probe | Status | Key metric |
|---|---|---|
| `spock-deltanet-prefill-pipeline-probe` | `compare-ok` | max_rel_core=8.9407e-08, max_rel_state=1.19175e-07, nan_count=0 |
| `spock-deltanet-prefill-collect-probe` | `compare-ok` | all max_rel/max_abs=0, nan_count=0 |
| `spock-deltanet-chunk-prefill-probe --case runtime-l2-padded-submit` | `compare-ok` | max_rel_core=1.19209e-07, max_rel_state=1.19208e-07, nan_count=0 |

## Artifact Baseline (committed state)

The following changes were committed for this milestone:

- `src/runtime/vk_session.cpp` — `SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT`
  env gate, `gpu_chunk_prefill_from_gpu_collect()` implementation, per-layer
  descriptor offset computation with 256-byte alignment, direct binding of
  GPU-collected per-layer segments into `deltanet_chunk_prefill.comp`.
- `src/runtime/vk_session.hpp` — Declared `gpu_chunk_prefill_from_gpu_collect()`,
  added persistent GPU collection buffers and `persist_bufs_allocated_`.

No new shaders, probes, pipelines, or test infrastructure were added in this
step. The new code path reuses existing shaders and descriptor layouts.

## Known Limitations

1. **CPU collection bridge still exists in the code path.** The default
   (non-gated) prefill path still collects Q/K/V/g/beta to CPU and stores
   them in `prefill_chunks_`. The gated path still populates those CPU vectors
   today, but no longer uses them as the chunk shader's Q/K/V/g/beta inputs.
   The next step is to bypass that CPU collection work entirely when the new
   gate is active, which will let the prefill loop skip the per-token download
   and the `half_to_float()` conversion loop.

2. **Per-head submit inefficiency not addressed.** The chunk-prefill dispatch
   still submits one command buffer per head per layer (24 × 16 = 384
   submit-wait cycles per chunk). The collection shader is efficient (one
   dispatch with all heads), but the consumer is not. This is the dominant
   performance bottleneck and must be fixed before the gated path can be
   defaulted.

3. **Collection buffers are allocated per prefill call.** The `dn_persist_*`
   buffers are sized to the current prompt and allocated/deallocated around
   each call to `layer_major_prefill`. Hoisting to session-scope with
   resize-on-growth would reduce allocation churn across repeated decode
   calls.

4. **No automated regression test for the new gated path.** The
   `SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1` path is verified manually.
   There is no CI test that exercises the double-gated path and checks
   correctness. The standalone probes cover the shader interfaces but do not
   exercise the runtime wiring.

5. **Only verified on `short_correctness_001` (seq_len=9).** Longer prompts
   are expected to behave identically (the per-layer segments are
   layout-identical, just larger), but were not re-verified in this session
   due to the per-head submit slowdown.

6. **256-byte alignment is conservative but still a target assumption.**
   The alignment choice is conservative for the target RADV NAVI22 path and
   common Vulkan hardware. A future portability cleanup should query
   `minStorageBufferOffsetAlignment` and use the actual device limit.

## Next Work

### Primary: Remove CPU collection bridge from gated path

The current gated path no longer uses CPU-packed/uploaded Q/K/V/g/beta as
the chunk shader's inputs, but it still performs the CPU collection work and
populates `prefill_chunks_`. The next step is to bypass the CPU bridge
entirely when `SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1` is active:
short-circuit the per-token download path and let the prefill loop hand off
directly from the collect dispatch to the chunk-prefill dispatch without any
host-side data touching the collected activations.

This should be behavior-preserving if the diagnostics remain green, but it is
not just cleanup: it removes real host transfers and CPU fp16-to-fp32 packing
from the gated path.

### Secondary: Fix per-head submit inefficiency

The per-head submit workaround (384 submit-wait cycles per chunk) remains the
critical performance blocker. A correct single-dispatch multi-head chunk-prefill
design is needed before the all-GPU path can be defaulted. This is the same
next-work item from diary 0019 — it has not changed.

### Secondary: Session-scoped allocation hoist

Move the `dn_persist_*` buffer allocation from per-call to session-scoped with
lazy resize. This eliminates the allocation churn across repeated decode calls.
Low priority until the per-head submit fix is in place.

### Tertiary: Formal test harness for gated paths

Add a CI or harness test that exercises `SPOCK_GPU_CHUNK_PREFILL=1` and
`SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1` on `short_correctness_001` and
checks the output token against the reference. Also add a compare-gate
regression that asserts `nan_count=0` and `max_abs` below a threshold on
all layers.

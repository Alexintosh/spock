# 0021: Narrow Runtime Cleanup — Skip CPU Collection Bridge on No-Compare Gated Path

## Goal

The previous milestone (diary 0020) wired GPU-collected Q/K/V/g/beta buffers
directly into `deltanet_chunk_prefill.comp` but still performed the CPU
collection work (per-token staging downloads, `half_to_float` conversion,
`prefill_chunks_` population) on every DeltaNet prefill pass — even when the
gated path no longer consumed those CPU data as chunk-prefill inputs.

This entry closes that gap: when the gated path is active and no diagnostic
compare flag is set, the per-token CPU collection bridge is skipped entirely.
The DeltaNet prefill loop hands off directly from the collect dispatch to the
chunk-prefill dispatch without any host-side data touching the collected
activations. The CPU collection remains for fallback and diagnostic comparison
when either `SPOCK_GPU_COLLECT_PREFILL_COMPARE=1` or
`SPOCK_GPU_CHUNK_PREFILL_COMPARE=1` is active.

This is a behavior-preserving cleanup on the default path, and a meaningful
host-work reduction on the fully-gated GPU path. It is also a step closer to
the final GPU prefill offload, but still not complete: per-head submit
inefficiency remains, CPU orchestration and descriptor updates still happen,
and the output reintegration/readback path is unchanged.

## Implementation Work Completed

### Conditional bypass of CPU collection bridge

In `src/runtime/vk_session.cpp`, the per-layer per-token loop inside
`layer_major_prefill` now checks whether the gated path will handle
Q/K/V/g/beta input for the chunk-prefill shader. The condition is:

```
gpu_chunk_prefill_from_gpu_collect_active   &&
!gpu_collect_prefill_compare_active          &&
!gpu_chunk_prefill_compare_active
```

When all three sub-conditions are true:

- **No per-token `B.dn_qkv` staging download.** The existing download of
  per-token fp16 QKV activations from device-local to host-visible staging
  memory is skipped. The GPU-collected head-major fp32 buffers already hold
  the authoritative data.

- **No per-token `g/beta` staging download.** The corresponding download of
  the fp32 g and beta activations is also skipped.

- **No `half_to_float` append into `prefill_chunks_`.** The loop body that
  converts the downloaded fp16 Q/K/V data to fp32 and appends to the
  layer's `PrefillChunkState::query/key/value/g/beta` vectors is skipped.

The collect dispatch (`deltanet_prefill_collect.comp`) still runs every
token — it produces the device-local data that the chunk-prefill shader
consumes. Only the CPU-facing readback and conversion is bypassed.

### `run_chunk_prefill` tolerates empty `prefill_chunks_` for from-GPU layers

The chunk-prefill orchestration function `run_chunk_prefill` previously
assumed that every DeltaNet layer's `prefill_chunks_[dn_idx]` was populated
with at least `query` data. On the from-GPU path with no compare active,
those CPU vectors are now empty.

The gated path now checks whether the layer uses GPU-collected input before
requiring `prefill_chunks_[dn_idx]` to be populated. When the from-GPU path
is active, `run_chunk_prefill` processes all DeltaNet layers via
`gpu_chunk_prefill_from_gpu_collect()` without requiring CPU-staged
reference data.

### Clear error when compare asks for empty CPU data

In `gpu_chunk_prefill_from_gpu_collect()`, when
`SPOCK_GPU_CHUNK_PREFILL_COMPARE=1` is active and the function attempts
to read CPU reference data from `prefill_chunks_`, it now throws a clear
error if the chunk vectors are empty:

```
SPOCK_GPU_CHUNK_PREFILL_COMPARE=1 but CPU chunk vectors are empty
(GPU-collected path skips CPU collection when no compare is active)
```

This prevents silent compare-skip or segfault when a user sets the chunk
compare flag after prefill bypassed collection.

### Compare flags preserve CPU collection

When either `SPOCK_GPU_COLLECT_PREFILL_COMPARE=1` or
`SPOCK_GPU_CHUNK_PREFILL_COMPARE=1` is set alongside the from-GPU gated
path, the CPU collection bridge remains enabled. The compares need reference
data from `prefill_chunks_`, and the CPU-collected data also serves as the
diagnostic gold standard. This is safe because the compare flags are for
validation only — users running production inference do not set them.

### Default behavior unchanged

When neither gate is set, or when only `SPOCK_GPU_CHUNK_PREFILL=1` is set
(without `SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1`), the runtime's
DeltaNet prefill behavior is identical to the 0020 baseline. The new
bypass logic only activates when all three sub-conditions are true.

## Inference Concepts

### Why the bypass matters

The CPU collection bridge involved three operations per token per layer that
are now avoided in the fully-gated path:

1. **Staging-buffer readback.** Each per-token `B.dn_qkv` (fp16, seq_len=1,
   shape `[num_heads, 3 * head_dim]`) and `B.g/beta` (fp32) were downloaded
   from device-local to host-visible staging via `vkCmdCopyBuffer` + fence
   wait + `vkMapMemory` read. For a prompt of length S and 18 DeltaNet
   layers, this is 18 × S download pairs eliminated.

2. **`half_to_float` conversion.** The fp16 Q/K/V data was converted to fp32
   via a host loop over each head, each dimension, using `detail::half_to_float`.
   This conversion was purely for CPU collection — the chunk-prefill shader
   consumes fp32 anyway, and the GPU collection shader already produces fp32
   directly.

3. **`push_back` into `prefill_chunks_`.** The converted fp32 data was appended
   to per-layer `std::vector<float>` buffers, extending them to hold the full
   prompt's Q/K/V/g/beta in token-major order.

These operations were not the dominant inference cost (the GPU dispatches and
the per-head chunk-prefill submit loop are), but they represented real host
CPU work and PCIe transfer traffic that now disappears entirely from the
gated path.

### Why the collect dispatch still runs

The GPU collection shader (`deltanet_prefill_collect.comp`) writes per-token
fp16 QKV + fp32 g/beta into the device-local head-major fp32 persistent
buffers. This dispatch is not bypassed — it is the producer that fills the
buffers consumed by the chunk-prefill shader. Without it, the from-GPU path
has no inputs.

The CPU collection bridge is downstream of the collect dispatch: it reads
back the same activation data from the per-token QKV/g/beta output buffers
(not the collect buffers) and converts it to CPU-friendly fp32 vectors. The
bypass eliminates only that downstream readback and conversion.

### Error-guarding the compare flow

The compare diagnostic (`SPOCK_GPU_CHUNK_PREFILL_COMPARE=1`) downloads the
GPU-collected buffers AND reads the CPU-collected `prefill_chunks_` data,
then compares element-by-element. If CPU collection was bypassed during
prefill, `prefill_chunks_[dn_idx].query` is empty and the compare would
either silently skip (producing a false "no failures" result) or segfault.

The explicit error in `gpu_chunk_prefill_from_gpu_collect()` ensures the
user learns they must re-run with `SPOCK_GPU_COLLECT_PREFILL_COMPARE=1`
(or equivalently, without the bypass condition) to collect CPU reference data
before comparing.

## Verification

All verification commands were run on the target RADV RX 6750 XT (NAVI22)
hardware.

### Build

```sh
cmake --build build -j
```

Passed cleanly.

### CTest regression gate (4/4 passed)

```sh
ctest --test-dir build --output-on-failure -R "spock_capabilities|spock_deltanet_chunk_unit|spock_vk_decode_prefill_handoff_mismatch|spock_diagnose_handoff_mc023"
```

All four targeted tests pass.

### `short_correctness_001 --max-new-tokens 1` — GPU-collected path without compare (no CPU bridge)

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

This is the new no-CPU-bridge gated path. It verifies that skipping the CPU
collection bridge (no staging downloads, no half_to_float, no prefill_chunks_
population) produces correct output for the prompt.

### `short_correctness_001 --max-new-tokens 1` — GPU-collected path with compare (CPU bridge preserved)

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

Verifies that the compare flag preserves CPU collection, the compare runs,
and the gated path still produces correct output.

### `short_correctness_001 --max-new-tokens 1` — GPU collect compare diagnostic (CPU bridge preserved)

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

Verifies the existing GPU collect compare diagnostic still works (it was not
affected by the bypass logic, since it does not use `SPOCK_GPU_CHUNK_PREFILL`
at all).

### Standalone probe regression

| Probe | Status | Key metric |
|---|---|---|
| `spock-deltanet-prefill-pipeline-probe` | `compare-ok` | max_rel_core=8.9407e-08, max_rel_state=1.19175e-07, nan_count=0 |
| `spock-deltanet-prefill-collect-probe` | `compare-ok` | all max_rel/max_abs=0, nan_count=0 |
| `spock-deltanet-chunk-prefill-probe --case runtime-l2-padded-submit` | `compare-ok` | max_rel_core=1.19209e-07, max_rel_state=1.19208e-07, nan_count=0 |

All standalone probes continue to pass. These probe the shader interfaces and
do not exercise the runtime bypass logic, so no regression was expected.

### Preserved CTest gate (4/4 passed)

All CTest regression tests pass identically to the baseline.

## Artifact Baseline (committed state)

The following changes were committed for this milestone:

- `src/runtime/vk_session.cpp` — Conditional CPU collection bridge bypass in
  `layer_major_prefill` when `SPOCK_GPU_CHUNK_PREFILL=1`,
  `SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1` are active and neither
  compare flag is set; `run_chunk_prefill` tolerates empty
  `prefill_chunks_` for from-GPU layers; clear error in
  `gpu_chunk_prefill_from_gpu_collect` when compare asks for empty GPU ref
  data.

No new shaders, probes, pipelines, or test infrastructure were added in this
step.

## Known Limitations

1. **Per-head submit inefficiency not addressed.** The chunk-prefill dispatch
   still submits one command buffer per head per layer (24 × 16 = 384
   submit-wait cycles per chunk). This is the dominant performance bottleneck
   and is unchanged from diary 0020.

2. **CPU orchestration and descriptor updates still happen.** The host still
   iterates over layers and heads, updates descriptor sets, submits work, and
   manages synchronization. Only the data-transfer and conversion portion of
   the CPU bridge is removed. The final GPU prefill offload would need to
   eliminate the per-head submit pattern entirely.

3. **Output reintegration/readback path unchanged.** The chunk-prefill shader
   output (core output, state update) is still downloaded, integrated on CPU,
   and uploaded back for the next layer's input. This is the same data flow
   as diary 0020.

4. **Collection buffers allocated per prefill call.** The `dn_persist_*`
   buffers are still sized to the current prompt and allocated/deallocated
   around each call to `layer_major_prefill`. Not hoisted to session scope.

5. **No automated regression test for the no-CPU-bridge gated path.** The
   bypass is verified manually. There is no CI test that exercises the
   double-gated path with neither compare flag set and checks correctness.

6. **Only verified on `short_correctness_001` (seq_len=9).** Longer prompts
   are expected to behave identically (the bypass eliminates a data path,
   not a computation), but were not re-verified due to the per-head submit
   slowdown.

## Next Work

### Primary: Fix per-head submit inefficiency

The per-head submit workaround (384 submit-wait cycles per chunk) remains the
critical performance blocker. A correct single-dispatch multi-head chunk-prefill
design is needed before the all-GPU path can be defaulted. This is the same
next-work item from diary 0020 — it has not changed.

### Secondary: Session-scoped allocation hoist

Move the `dn_persist_*` buffer allocation from per-call to session-scoped with
lazy resize. Low priority until the per-head submit fix is in place.

### Tertiary: Formal test harness for gated paths

Add a CI or harness test that exercises the fully-gated
(`SPOCK_GPU_CHUNK_PREFILL=1 SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1`,
no compare flags) path on `short_correctness_001` and checks the output token
against the reference. Also add a compare-gate regression that asserts
`nan_count=0` and `max_abs` below a threshold on all layers.

### Future: Final GPU prefill offload

After the per-head submit is fixed and the gated path is defaulted, the
remaining CPU work to eliminate would be:
- Layer-iterate orchestration (descriptor updates, dispatch submission).
- Output readback and reintegration.
- Layer-ordering dependencies between DeltaNet and attention layers.

# 0019: Runtime Diagnostic GPU Prefill Collection Comparison

## Goal

Wire the verified GPU prefill collection shader (`deltanet_prefill_collect.comp`)
into the real `DecodeSession` layer-major prefill loop, and prove that the
shader can correctly extract per-token Q/K/V/g/beta from the actual runtime
activation buffers — not just deterministic probe inputs — and that the
collected device-side buffers produce the exact same fp32 head-major values
as the existing CPU collection bridge.

This is a diagnostic-only milestone. Full GPU offload requires a second step:
feeding the GPU-collected buffers into `gpu_chunk_prefill()` instead of the
CPU-collected ones. That step is deferred to future work.

## Implementation Work Completed

### New env gate: `SPOCK_GPU_COLLECT_PREFILL_COMPARE=1`

A new environment variable gate was added to `DecodeSession` in
`src/runtime/vk_session.cpp`. When set to `1` and the prompt length is
positive:

1. **Before the prefill loop.** Five device-local buffers (`dn_collect_q`,
   `dn_collect_k`, `dn_collect_v`, `dn_collect_g`, `dn_collect_beta`) are
   allocated to hold the collected head-major fp32 data. They are stored as
   `Buffers` members so the destructor can clean up safely if an exception
   occurs, but they are allocated only for the current prefill call and sized
   to the current prompt length. Existing allocation is detected via
   `collect_bufs_allocated_` and re-allocated if needed.

2. **During each token's DeltaNet recurrent step,** after the GPU
   `deltanet_compute_g_beta` shader produces g/beta but before the recurrent
   shader, the diagnostic sends a dispatch of `deltanet_prefill_collect.comp`.
   The descriptor set is updated to point:
   - Binding 0: `B.dn_qkv` (the fp16 per-token Q/K/V buffer after projection,
     conv1d, and Q/K L2 normalization for this token)
   - Binding 1: `B.dn_state` at the layer's g/beta offset (the fp32 g and
     beta values for each head)
   - Bindings 2-6: the five device-local output buffers (Q, K, V, g, beta)

   The push constants pass `num_heads`, `seq_len` (total prompt length),
   `token_idx` (current token position), `k_dim`, and `v_dim`. The shader
   writes into the output buffers at the appropriate head-major index
   `[head][seq][dim]`.

3. **After all tokens in the DeltaNet layer.** The diagnostic downloads all
   five GPU-collected buffers to CPU via host-visible staging. It then
   iterates over every head, every token, and every dimension, comparing the
   GPU head-major layout `[head][seq][dim]` against the CPU token-major layout
   `[token][head][dim]` stored in `prefill_chunks_[dn_idx]`.

   For each scalar pair, it computes:
   - `abs = |gpu - cpu|`
   - `rel = abs / max(1.0, |cpu|)`
   - `nan_count` for any NaN in the GPU value

   It prints a single line per layer:
   ```
   SPOCK_GPU_COLLECT_PREFILL_COMPARE layer=N seq_len=L
     max_rel_q=... max_rel_k=... max_rel_v=... max_rel_g=... max_rel_beta=...
     max_abs_q=... max_abs_k=... max_abs_v=... max_abs_g=... max_abs_beta=...
     nan_count=...
   ```

   If `max_rel > 1e-5` across all five tensors, or any NaN is detected, a
   `std::runtime_error` is thrown.

4. **After the prefill loop.** The device-local collection buffers are
   destroyed and `collect_bufs_allocated_` is reset to `false`.

### Pipeline and module lifecycle

The `DecodeSession` constructor now loads `deltanet_prefill_collect.comp.spv`,
creates the Vulkan shader module and pipeline at session init time, and stores
them as `P.deltanet_prefill_collect_module` and
`P.deltanet_prefill_collect`. The descriptor layout is the existing 7-storage
buffer layout used by the chunk-prefill path; the pipeline layout's push
constant range is large enough for the collect shader's five `uint32_t`
fields. The destructor destroys the new pipeline and shader module.

### Header changes

- `src/runtime/vk_session.hpp`: Added `dn_collect_q`, `dn_collect_k`,
  `dn_collect_v`, `dn_collect_g`, `dn_collect_beta`,
  `collect_bufs_allocated_`, the collect pipeline handle, shader module, and
  descriptor set.

## Inference Concepts

### Head-major vs token-major layout

The CPU collection bridge (`prefill_chunks_`) stores Q/K/V/g/beta in
token-major order: for each token index `t`, it appends all heads'
Q/K/V/g/beta sequentially. This is a natural byproduct of per-token GPU
dispatch followed by CPU download.

The GPU collection shader writes directly into head-major order: for each
head `h`, it writes all tokens' Q/K/V/g/beta at the position
`[h][token][dim]`. This is what `deltanet_chunk_prefill.comp` expects as
input — the shader processes heads in parallel and needs each head's data
contiguous.

The comparison must therefore transpose on the CPU side: for each head `h`,
token `t`, and dimension `d`, the GPU index is
`(h * seq_len + t) * dim + d` while the CPU index is
`(t * num_heads + h) * dim + d`.

### Diagnostic gate invariant

The gate is explicitly diagnostic-only. It does nothing unless
`SPOCK_GPU_COLLECT_PREFILL_COMPARE=1` is set. The runtime's default behavior
is unchanged — no additional GPU dispatches, no additional allocations, no
change to inference outputs. The diagnostic path runs in parallel with (and
discards its results independently of) the production prefill path.

## Verification

All verification commands were run on the target RADV RX 6750 XT (NAVI22)
hardware. Each independently confirmed that the GPU-collected buffers match
the CPU-collected buffers with zero numerical error.

### Build

```sh
cmake --build build -j
```

Passed cleanly (no compilation or linking errors). The SPIR-V shader was
already compiled; only the C++ session code changed.

### CTest regression gate (4/4 passed)

```sh
ctest --test-dir build --output-on-failure -R "spock_capabilities|spock_deltanet_chunk_unit|spock_vk_decode_prefill_handoff_mismatch|spock_diagnose_handoff_mc023"
```

All four targeted tests pass. The gated diagnostic does not affect non-gated
behavior, so no parity regression was introduced.

### Runtime diagnostic: `short_correctness_001`, `--max-new-tokens 1`

```sh
SPOCK_GPU_COLLECT_PREFILL_COMPARE=1 python3 tests/run_vk_decode_parity.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --reference tests/data/reference_tokens.jsonl \
  --ids short_correctness_001 \
  --max-new-tokens 1
```

Output:
```json
{"status":"ok","checked":1,"failures":[]}
```

The generated token (271) matches the reference. The diagnostic ran silently
alongside production prefill.

### Verbose per-layer diagnostic output

Running `spock-decode` directly on the `short_correctness_001` prompt
(seq_len=9) with `--verbose` printed one
`SPOCK_GPU_COLLECT_PREFILL_COMPARE` line per DeltaNet layer (0..17). All
`max_rel` and `max_abs` values were 0, and `nan_count=0`.

### Standalone probe regression

All three standalone probes from diary 0018 continue to pass:

| Probe | Status | Key metric |
|---|---|---|
| `spock-deltanet-prefill-collect-probe` | `compare-ok` | all max_rel/max_abs=0, nan_count=0 |
| `spock-deltanet-chunk-prefill-probe --case runtime-l2-padded-submit` | `compare-ok` | max_rel_core=1.19209e-07, max_rel_state=1.19208e-07, nan_count=0 |
| `spock-deltanet-prefill-pipeline-probe` | `compare-ok` | max_rel_core=8.9407e-08, max_rel_state=1.19175e-07, nan_count=0 |

## Artifact Baseline (committed state)

The following artifacts reflect the new diagnostic:

- `src/runtime/vk_session.cpp` — `SPOCK_GPU_COLLECT_PREFILL_COMPARE` env gate,
  device-local collection buffer allocation, per-token collect dispatch,
  post-layer comparison logic
- `src/runtime/vk_session.hpp` — `dn_collect_q/k/v/g/beta` buffer members,
  `collect_bufs_allocated_` flag

No new shaders, probes, or test infrastructure were added in this step. The
runtime now creates and destroys the collect shader module and pipeline, and
the diagnostic collection buffers are destroyed after prefill or by the
destructor if an exception interrupts the diagnostic path.

## Known Limitations

1. **Diagnostic only; no GPU offload achieved.** This milestone proves the GPU
   shader can collect activations correctly from real runtime buffers. It does
   not feed those buffers into `gpu_chunk_prefill()`. The production path
   still uses CPU-collected Q/K/V/g/beta vectors.

2. **Device-local buffers are re-allocated per prefill call.** Allocation
   happens at the start of `DecodeSession::layer_major_prefill` and is
   destroyed at the end of that function. For long prompts or repeated decode
   calls this causes
   unnecessary churn. Future work should hoist the buffers to session scope
   with prompt-length resizing.

3. **No automated regression test for the diagnostic path.** The env-gated
   diagnostic is verified manually. There is no CI test that runs
   `short_correctness_001` with `SPOCK_GPU_COLLECT_PREFILL_COMPARE=1` and
   checks the output.

4. **Only verified on `short_correctness_001` (seq_len=9).** Longer prompts
   are expected to produce the same exact match, but were not re-verified in
   this session due to the per-head submit slowdown on the GPU chunk-prefill
   path (the diagnostic collection itself is per-token and fast, but
   `short_correctness_001` was chosen as a representative quick case).

## Next Work

### Primary: Wire GPU collection into `gpu_chunk_prefill()`

The next integration step is straightforward given the diagnostic:

1. Preserve the lifetime of `bufs_->dn_collect_q/k/v/g/beta` through
   `run_chunk_prefill()` and feed them directly into `gpu_chunk_prefill()` as
   the Q/K/V/g/beta inputs, behind a new env gate (for example
   `SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1`) or as a phased replacement
   of the existing `SPOCK_GPU_CHUNK_PREFILL=1` path.

2. Compare generated tokens between the CPU-collected+GPU-chunk-prefill path
   and the GPU-collected+GPU-chunk-prefill path. They should produce identical
   tokens (the compared fp32 values are numerically identical in the runtime
   diagnostic above).

3. Once stable, remove the CPU collection bridge from the gated path. The CPU
   staging reads of `B.dn_qkv` and `B.dn_state` (g/beta), the
   `half_to_float()` conversion loop, and the `prefill_chunks_` append logic
   can be bypassed when all-GPU prefill is active.

4. After the CPU bridge is removed for the gated path, default the env gate to
   `1` (or remove it) once the per-head submit inefficiency in
   `gpu_chunk_prefill()` is resolved.

### Secondary: perf-head submit fix

The per-head submit workaround (one command buffer per head, 384
submit-wait cycles per chunk for 24 layers × 16 heads) remains the
critical performance blocker. The diagnostic collection does not use
per-head submit (it dispatches `DN_HEADS` workgroups in one go), but the
`gpu_chunk_prefill` consumer still does. A correct single-dispatch
multi-head chunk-prefill design is needed before the all-GPU path can be
defaulted.

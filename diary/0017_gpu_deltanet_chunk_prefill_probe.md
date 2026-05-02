# 0017: GPU DeltaNet Chunk-Prefill Shader — Multi-Chunk, Multi-Head Probe and Pipeline Barrier Discovery

## Goal

Record the checkpoint: the experimental GPU DeltaNet chunk-prefill shader and
standalone probe now exercise multi-chunk (chunk_count=2) and multi-head
(num_heads=16) configurations, with GPU/CPU numerical parity confirmed in all
cases. A critical finding — *an explicit compute pipeline barrier between
per-head dispatches is required for correctness* — constrains the production
integration strategy.

This is a structural step toward GPU-native chunk offload for the DeltaNet
prefill path, but it is **not** production GPU offload — the CPU bridge
(`run_deltanet_chunk_rule` → `run_chunk_prefill`) remains the active path.

The probe has advanced from:
- synthetic tiny dims (k_dim=8, v_dim=8) — diary 0016 era
- full-width single-chunk single-head (k_dim=128, v_dim=128, seq_len=4,
  chunk_size=4, num_heads=1) — previous 0017 checkpoint
- production-like chunk geometry (seq_len=64, chunk_size=64, num_heads=1)
- two-chunk prompt (seq_len=128, chunk_size=64, chunk_count=2, num_heads=1)
- multi-head with per-head repeated QKV (heads=16, repeat_per_head=true)
- multi-head with shared QKV (heads=16, repeat_per_head=false)

At each step GPU and CPU paths compute the same recurrence at full per-head
activation scale.

## Context

Diary 0016 ended with a refined understanding of precision drift (accumulated
fp16 output-boundary rounding at full-attention layers 3, 7, 11, 15) and three
next-work options: targeted fp32-output experiments, broader fp32-resident
decode, or re-evaluating the parity contract against the megakernel roadmap.

The chunk-prefill shader documented here is a separate track: the Vulkan-native
DeltaNet chunk rule (the `B` matrix computation) that would eliminate the CPU
round-trip in the prefill path. It is an early experimental probe, not yet
wired into the decode pipeline.

## Checkpoint: Experimental Chunk-Prefill Shader

### What exists

**`shaders/deltanet_chunk_prefill.comp`** — a GLSL compute shader that
implements the DeltaNet chunk-prefill `B` matrix recurrence. It accepts a
`base_head` push constant to select the starting head in the binding arrays,
enabling a single pipeline to service all heads via serial per-head dispatch.

**`apps/spock-deltanet-chunk-prefill-probe.cpp`** — a standalone application
that creates a Vulkan pipeline, allocates/binds seven host-visible storage
buffers (Q, K, V, g, beta, output, initial_state), dispatches the shader, runs
the CPU `run_deltanet_chunk_rule` on identical inputs, and prints a status
comparison. The probe now supports case selection via the `--case` flag, with
five defined cases:

| Case | seq_len | chunk_size | chunk_count | k_dim | v_dim | heads | repeat_per_head | serial_head_barrier | Description |
|------|---------|------------|-------------|-------|-------|-------|-----------------|---------------------|-------------|
| `real-width` | 4 | 4 | 1 | 128 | 128 | 1 | — | — | Verified small case (default) |
| `real-chunk` | 64 | 64 | 1 | 128 | 128 | 1 | — | — | Production-like chunk geometry |
| `two-chunks` | 128 | 64 | 2 | 128 | 128 | 1 | — | — | State carry across chunk boundary |
| `multi-head-repeat` | 64 | 64 | 1 | 128 | 128 | 16 | true | true | Per-head QKV repeated from head 0 |
| `multi-head` | 64 | 64 | 1 | 128 | 128 | 16 | false | true | Shared QKV, separate head states |

For num_heads>1 the probe dispatches the shader serially per head, incrementing
the `base_head` push constant. A `VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT` →
`VK_ACCESS_SHADER_WRITE_BIT` barrier separates each dispatch.

Each case defines its own deterministic input bounds (q_lo/q_hi, k_lo/k_hi,
v_lo/v_hi, g_lo/g_hi, beta_lo/beta_hi) so the inputs are scaled appropriately
for the sequence length — wider input ranges at small seq_len, narrower at
large seq_len, matching the statistical distribution of activations seen in
actual model layers.

#### Default case (`real-width`) — verified parity (unchanged)

```json
{
  "shader": "deltanet_chunk_prefill.comp",
  "case": "real-width",
  "status": "compare-ok",
  "max_rel_core": 2.27374e-13,
  "max_rel_state": 3.63798e-12
}
```

#### `real-chunk` case — production-like geometry passes (unchanged)

```json
{
  "shader": "deltanet_chunk_prefill.comp",
  "case": "real-chunk",
  "status": "compare-ok",
  "max_rel_core": 6.82121e-13,
  "max_rel_state": 1.81899e-11
}
```

#### `two-chunks` case — state carry across chunk boundary passes

```json
{
  "shader": "deltanet_chunk_prefill.comp",
  "case": "two-chunks",
  "total_seq": 128,
  "chunk_count": 2,
  "seq_len": 128,
  "chunk_size": 64,
  "k_dim": 128,
  "v_dim": 128,
  "num_heads": 1,
  "status": "compare-ok",
  "max_rel_core": 2.84217e-13,
  "max_rel_state": 1.45519e-11,
  "cpu_chunk_bridge_production": true
}
```

This exercises the state-carried-across-chunks path: chunk 0 produces an output
state that feeds chunk 1. The max relative errors (~3e-13 core, ~1.5e-11 state)
confirm the shader carries state across chunk boundaries correctly.

#### `multi-head-repeat` case — 16 heads, per-head repeated inputs, passes

```json
{
  "shader": "deltanet_chunk_prefill.comp",
  "case": "multi-head-repeat",
  "total_seq": 64,
  "chunk_count": 1,
  "seq_len": 64,
  "chunk_size": 64,
  "k_dim": 128,
  "v_dim": 128,
  "num_heads": 16,
  "repeat_per_head": true,
  "serial_head_barrier": true,
  "status": "compare-ok",
  "max_rel_core": 2.13163e-14,
  "max_rel_state": 1.36424e-12,
  "cpu_chunk_bridge_production": true
}
```

Each of the 16 heads derives QKV from head-0 data (same inputs, independent
states). The serial-head barrier is enabled. Max relative errors (~2e-14 core,
~1.4e-12 state) are excellent.

#### `multi-head` case — 16 heads, shared inputs, passes

```json
{
  "shader": "deltanet_chunk_prefill.comp",
  "case": "multi-head",
  "total_seq": 64,
  "chunk_count": 1,
  "seq_len": 64,
  "chunk_size": 64,
  "k_dim": 128,
  "v_dim": 128,
  "num_heads": 16,
  "repeat_per_head": false,
  "serial_head_barrier": true,
  "status": "compare-ok",
  "max_rel_core": 1.13687e-13,
  "max_rel_state": 5.45697e-12,
  "cpu_chunk_bridge_production": true
}
```

All heads share the same QKV buffer regions (independent states). Serial-head
barrier enabled. Max relative errors (~1e-13 core, ~5.5e-12 state) confirm
no cross-head interference.

### Critical finding: pipeline barrier is required for multi-head correctness

The multi-head cases did not succeed immediately.

1. **Initial attempt — no barrier, serial dispatch only:** The probe dispatched
   heads serially with `base_head` push constant changes, but without an
   explicit `VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT` →
   `VK_ACCESS_SHADER_WRITE_BIT` barrier between dispatches. Result: **multi-head
   failed** — GPU output did not match CPU.

2. **With explicit pipeline barrier:** Adding the barrier between each per-head
   dispatch resolved the mismatch. Both `multi-head-repeat` and `multi-head`
   produce `compare-ok`.

**Implication:** Even serial (non-concurrent) dispatch of the same pipeline
from the same queue cannot rely on implicit ordering when the pipeline writes
to overlapping buffer regions — or even disjoint regions without explicit
synchronization. The Vulkan specification requires explicit barriers for
visible writes across submissions, and the host-visible buffers used here do
not exempt this requirement.

For production integration this means:
- **Conservative first cut:** A serial-head loop with explicit barriers is
  correct and straightforward, matching the pattern verified here.
- **Optimization target:** A single workgroup-per-head dispatch with
  synchronization built into the shader (via subgroup or shared memory) would
  eliminate the serial loop and the barrier overhead, but requires a more
  carefully designed shader.

### What does NOT exist (not production offload)

- **No production inference GPU dispatch.** Production inference still routes
  through `vk_session.cpp`/`run_chunk_prefill`, which calls
  `run_deltanet_chunk_rule` on CPU. The GPU primitive is probed at per-head
  width and multi-head dispatch, but is not wired into the pipeline.
- No staging buffers (host-visible only, no device-local memory).
- No integration into `vk_session.cpp` or the prefill dispatch path.
- No tests, no CI gate, no benchmark.
- No W-binding — the shader computes W internally from Q and K rather than
  receiving it as input.

### Why this is progress

- Multi-chunk probing (two-chunks) confirms the state-carry path across chunk
  boundaries works correctly at full activation width.
- Multi-head probing (16 heads, with and without per-head repeat) confirms the
  serial-head dispatch pattern, push-constant offsetting, and pipeline barrier
  produce correct results with no cross-head interference.
- The barrier requirement is a concrete design constraint discovered through
  experimentation — it directly informs the production integration approach.
- All five cases pass at full per-head activation scale (k_dim=128, v_dim=128)
  with max relative errors well below 1e-4, confirming the shader logic, buffer
  wiring, descriptor layout, and multi-head dispatch scheme are structurally
  correct.

### Verification

- `cmake --build build -j` passes, producing the SPIR-V binary and linking the
  probe app.
- `./build/spock-deltanet-chunk-prefill-probe` (default `real-width`) exits 0
  with `compare-ok`.
- `./build/spock-deltanet-chunk-prefill-probe --case real-chunk` exits 0 with
  `compare-ok`.
- `./build/spock-deltanet-chunk-prefill-probe --case two-chunks` exits 0 with
  `compare-ok`.
- `./build/spock-deltanet-chunk-prefill-probe --case multi-head-repeat` exits 0
  with `compare-ok`.
- `./build/spock-deltanet-chunk-prefill-probe --case multi-head` exits 0 with
  `compare-ok`.

All max relative errors < 2e-11, all nan_count=0.

## Current Limitations

Same as diary 0016, plus the new explicit limitations from the multi-chunk and
multi-head probe work:

1. **CPU chunk bridge remains the active prefill path.** `run_chunk_prefill` →
   `run_deltanet_chunk_rule` on CPU. The probe dispatches the shader at full
   width, multi-chunk, and multi-head, but the GPU primitive is not wired into
   `vk_session.cpp` — integration is future work.

2. **short_correctness_003 fails at generated token index 5** — the argmax
   decision flips on a close margin. Unchanged from diary 0016. The working
   hypothesis remains accumulated fp16 rounding drift at activation/output
   boundaries across the 24-layer pipeline.

3. **Two-chunk geometry verified, but full prompt-length not exercised.**
   The `two-chunks` case exercises seq_len=128, chunk_size=64 (chunk_count=2).
   A production prompt may span many dozens of chunks; scratch pressure and
   numeric behavior at >2 chunk boundaries remain unexercised. Workgroup sizing
   still uses synthetic (32,1,1) rather than production-tuned dimensions.

4. **Multi-head verified via serial dispatch with explicit barrier — the
   barrier requirement is itself a constraint to optimize away.** The
   serial-head barrier loop is correct but incurs a dispatch-loop cost linear
   in num_heads. A production shader should either handle all heads in a single
   dispatch (via workgroup-per-head with intra-shader synchronization) or
   amortize the barrier across larger work-units.

5. **No performance data.** Benchmarking is premature without a production
   integration path decided and implemented.

## Next Work

### Chunk-prefill shader track — shift from probing to production shaping

Probing is complete through multi-chunk and multi-head at full activation
scale. The next phase is deciding and implementing the production integration
shape:

1. **Decide production integration approach for GPU chunk prefill.** The
   conservative path: serial-head dispatch with explicit pipeline barriers,
   matching the pattern already verified. The optimization path: a single-dispatch
   shader that handles all heads with intra-shader synchronization. Decision
   criteria: complexity, correctness risk, and performance ceiling.

2. **Implement conservative serial-head dispatch path** behind a gated runtime
   switch in `vk_session.cpp`. Wire the shader into the prefill flow as an
   experimental path gated by a probe flag or environment variable.

3. **Replace / optimize the serial-head loop** with a safer shader design once
   the integrated path is working end-to-end. The intra-shader synchronization
   approach (subgroup barriers, workgroup-local memory) would eliminate the
   per-head dispatch overhead.

4. **Add integration tests** that exercise both the GPU chunk path and the CPU
   bridge fallback, confirming the fallback remains functional and the gated
   path produces identical output.

### Precision drift track

The remaining precision investigation options from diary 0016 are unchanged.
The recommended next action is a broader decode-path fp32-resident activation
experiment, or accepting the parity tradeoff against the megakernel roadmap.

## Scope

This entry is a structural checkpoint — no parity progress, no `short003` fix,
no production pipeline integration. The probe now exercises five configurations
(`real-width`, `real-chunk`, `two-chunks`, `multi-head-repeat`, `multi-head`)
all passing GPU/CPU comparison at full per-head activation scale. The critical
barrier discovery constrains the production integration strategy: start
conservatively with the serial-head barrier loop already verified, then
optimize toward a single-dispatch design. All existing gates, diagnostics, and
decode paths are unchanged.

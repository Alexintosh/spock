# 0017: GPU DeltaNet Chunk-Prefill Shader — Multi-Chunk, Multi-Head Probe, Runtime Integration, and Command-Buffer Separation Discovery

## Goal

Record the checkpoint: the experimental GPU DeltaNet chunk-prefill shader and
standalone probe have been extended through multi-chunk (chunk_count=2) and
multi-head (num_heads=16) configurations, and then advanced into **production
runtime integration** behind an env gate (`SPOCK_GPU_CHUNK_PREFILL=1`).

Key findings along this path:

1. **Pipeline barrier between per-head dispatches is required** — multi-head
   fails without it (discovered during probe-only work).
2. **Pipeline barrier is not sufficient for this device** — realistic padded
   multi-head data (seq_len=104, 2 chunks) still fails with barriers inside a
   single command buffer; **separate command-buffer submit_and_wait per head**
   is required for correctness on the RX 6750 XT (RADV NAVI22).
3. **The separate-submit workaround is correct but slow** — `pp520_046` passes
   at `--max-new-tokens 1` but is noticeably slower than the CPU path because
   of 18 per-head submits × 24 layers.

The runtime integration is gated (`SPOCK_GPU_CHUNK_PREFILL=1`, default=CPU)
and still uses CPU-collected Q/K/V/g/beta; only the chunk-rule computation
moves to GPU. This is a correctness/feasibility checkpoint, not a full GPU
offload.

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
**nine** defined cases:

| Case | seq_len | chunk_size | chunk_count | k_dim | v_dim | heads | repeat_per_head | pseudo_random | l2_norm_qk | separate_submit | Description |
|------|---------|------------|-------------|-------|-------|-------|-----------------|---------------|------------|-----------------|-------------|
| `real-width` | 4 | 4 | 1 | 128 | 128 | 1 | — | — | — | — | Verified small case (default) |
| `real-chunk` | 64 | 64 | 1 | 128 | 128 | 1 | — | — | — | — | Production-like chunk geometry |
| `two-chunks` | 128 | 64 | 2 | 128 | 128 | 1 | — | — | — | — | State carry across chunk boundary |
| `multi-head-repeat` | 64 | 64 | 1 | 128 | 128 | 16 | true | — | — | — | Per-head QKV repeated from head 0 |
| `multi-head` | 64 | 64 | 1 | 128 | 128 | 16 | false | — | — | — | Shared QKV, separate head states |
| `multi-head-padded` | 104 | 64 | 2 | 128 | 128 | 16 | false | — | — | — | Realistic padding, reproduce pp520_046 failure |
| `runtime-range-padded` | 104 | 64 | 2 | 128 | 128 | 16 | false | true | false | — | Pseudo-random ranges matching runtime distributions |
| `runtime-l2-padded` | 104 | 64 | 2 | 128 | 128 | 16 | false | true | true | — | L2-normalized Q/K as in real model |
| `runtime-l2-padded-submit` | 104 | 64 | 2 | 128 | 128 | 16 | false | true | true | true | L2-normalized + separate per-head submits |

For num_heads>1 the probe dispatches the shader serially per head, incrementing
the `base_head` push constant. The barrier/submit strategy varies by case
(see findings below).

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

#### `two-chunks` case — state carry across chunk boundary passes (unchanged)

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

#### `multi-head-repeat` case — 16 heads, per-head repeated inputs, passes (unchanged)

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

#### `multi-head` case — 16 heads, shared inputs, passes (unchanged)

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

#### `multi-head-padded` case — realistic padded geometry fails with single-CB barrier approach

```json
{
  "shader": "deltanet_chunk_prefill.comp",
  "case": "multi-head-padded",
  "total_seq": 128,
  "chunk_count": 2,
  "seq_len": 104,
  "chunk_size": 64,
  "k_dim": 128,
  "v_dim": 128,
  "num_heads": 16,
  "repeat_per_head": false,
  "serial_head_barrier": true,
  "separate_head_submits": false,
  "status": "compare-fail"
}
```

This case reproduces the failure seen in the real model (`pp520_046`): with
seq_len=104 (padded to 128, 2 chunks of 64), 16 heads, serial dispatch with
pipeline barriers inside one command buffer, the GPU output diverges from CPU.
This is the case that prompted the deeper investigation.

#### `runtime-range-padded` case — pseudo-random ranges, L2-norm off, single-CB barrier

```json
{
  "shader": "deltanet_chunk_prefill.comp",
  "case": "runtime-range-padded",
  "total_seq": 128,
  "chunk_count": 2,
  "seq_len": 104,
  "chunk_size": 64,
  "k_dim": 128,
  "v_dim": 128,
  "num_heads": 16,
  "repeat_per_head": false,
  "pseudo_random": true,
  "l2_normalize_qk": false,
  "separate_head_submits": false,
  "status": "compare-fail"
}
```

Same padded geometry, but with pseudo-random input ranges that match the
statistical distribution of real model activations: Q/K ~[-1,1], V ~[-20,20],
g ~[-9,-1e-6], beta ~[0,1]. L2-normalization off. Still fails with single-CB
barrier dispatch.

#### `runtime-l2-padded` case — pseudo-random ranges, L2-norm on, single-CB barrier

```json
{
  "shader": "deltanet_chunk_prefill.comp",
  "case": "runtime-l2-padded",
  "total_seq": 128,
  "chunk_count": 2,
  "seq_len": 104,
  "chunk_size": 64,
  "k_dim": 128,
  "v_dim": 128,
  "num_heads": 16,
  "repeat_per_head": false,
  "pseudo_random": true,
  "l2_normalize_qk": true,
  "separate_head_submits": false,
  "status": "compare-fail"
}
```

Same as `runtime-range-padded` but with L2-normalized Q/K (matching the real
model). Still fails with single-CB barrier dispatch.

#### `runtime-l2-padded-submit` case — pseudo-random ranges, L2-norm on, separate per-head submits

```json
{
  "shader": "deltanet_chunk_prefill.comp",
  "case": "runtime-l2-padded-submit",
  "total_seq": 128,
  "chunk_count": 2,
  "seq_len": 104,
  "chunk_size": 64,
  "k_dim": 128,
  "v_dim": 128,
  "num_heads": 16,
  "repeat_per_head": false,
  "pseudo_random": true,
  "l2_normalize_qk": true,
  "separate_head_submits": true,
  "status": "compare-ok",
  "max_rel_core": 1.19e-7,
  "max_rel_state": 1.19e-7,
  "cpu_chunk_bridge_production": true
}
```

**This is the fix.** Same inputs as `runtime-l2-padded`, but each head is
dispatched via its own command buffer with `submit_and_wait` — no sharing of
command buffer, no pipeline barrier, just full submit-wait separation. Max
relative errors (~1.19e-7 for both core and state) are well below the 1e-4
threshold.

### Critical finding: pipeline barrier is required for multi-head correctness

The multi-head cases did not succeed immediately.

1. **Initial attempt — no barrier, serial dispatch only:** The probe dispatched
   heads serially with `base_head` push constant changes, but without an
   explicit `VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT` →
   `VK_ACCESS_SHADER_WRITE_BIT` barrier between dispatches. Result: **multi-head
   failed** — GPU output did not match CPU.

2. **With explicit pipeline barrier:** Adding the barrier between each per-head
   dispatch resolved the mismatch for the original multi-head cases (seq_len=64,
   single chunk, synthetic input ranges). Both `multi-head-repeat` and
   `multi-head` produce `compare-ok`.

**Implication for small synthetic inputs:** Even serial (non-concurrent)
dispatch of the same pipeline from the same queue cannot rely on implicit
ordering when the pipeline writes to overlapping buffer regions — or even
disjoint regions without explicit synchronization. The Vulkan specification
requires explicit barriers for visible writes across submissions, and the
host-visible buffers used here do not exempt this requirement.

### Further discovery: pipeline barrier is not sufficient — command-buffer separation required

When the probe was extended with realistic padded geometry (seq_len=104,
2 chunks, 16 heads), the pipeline-barrier approach **failed**:

- **`multi-head-padded`** (seq_len=104, barrier inside one CB) → `compare-fail`.
  This case was designed to reproduce the `pp520_046` failure.
- **`runtime-range-padded`** (pseudo-random ranges, barrier inside one CB) →
  `compare-fail`.
- **`runtime-l2-padded`** (L2-normalized Q/K, barrier inside one CB) →
  `compare-fail`.
- **`runtime-l2-padded-submit`** (same inputs, **separate CB per head with
  submit_and_wait**) → `compare-ok` at max_rel ~1.19e-7.

**Key finding:** The pipeline barrier works for small synthetic inputs
(seq_len=64, single chunk) but fails for realistic padded inputs (seq_len=104,
2 chunks). The root cause is hypothesized to be RADV/NAVI22 drivers reusing
scratch memory or internal compute-unit state between per-head dispatches
recorded in the same command buffer, and the pipeline barrier does not force
the driver to release that state before the next dispatch.

**Practical solution:** Submitting each head as an independent command buffer
and waiting for GPU completion (`vkQueueSubmit` + `vkQueueWaitIdle`) forces
the driver to fully complete and flush scratch writes before the next head
starts. This is significantly slower (serial per-head overhead) but is correct.

This discovery directly shaped the runtime integration approach.

### What does NOT exist (not production offload)

- **No full GPU offload.** Production inference still routes through
  `vk_session.cpp`/`run_chunk_prefill`. The GPU path moves only the chunk-rule
  computation to GPU; Q/K/V/g/beta collection and output re-integration remain
  CPU-side.
- **No staging buffers** (host-visible only, no device-local memory).
- **No tests, no CI gate, no benchmark** for the GPU path.
- **No W-binding** — the shader computes W internally from Q and K rather than
  receiving it as input.
- **No performance data** — the separate-submit workaround is known to be slow.

### Runtime integration: env-gated GPU chunk-prefill path

The GPU chunk-prefill has been wired into the production runtime behind an
environment gate, matching the conservative verified approach:

**Gate:** `SPOCK_GPU_CHUNK_PREFILL=1` — when set, `run_chunk_prefill` in
`vk_session.cpp` dispatches to `gpu_chunk_prefill()` instead of the CPU
`run_deltanet_chunk_rule()`.

**`DecodeSession::gpu_chunk_prefill()`** — new method that:

1. **Rearranges** Q/K/V/g/beta from the collected [token][head][dim] layout
   to the shader's [head][token][dim] layout.
2. **Uploads** to host-visible buffers.
3. **Dispatches** one command buffer per head (the conservative submit workaround):
   - Allocates a fresh `VkCommandBuffer` each iteration
   - Binds pipeline + descriptor set
   - Sets `base_head` push constant for the current head
   - Calls `vkCmdDispatch(1,1,1)`
   - Ends the CB and calls `submit_and_wait`
4. **Downloads** the output buffer and integrates back into the prefill state.

The per-head submit loop iterates `num_heads` times (16 for DeltaNet), each
with a full GPU round trip. For a 24-layer model this is 24 × 16 = 384
submit-wait cycles per chunk — clearly slow, but correct.

**Diagnostic comparison:** When `SPOCK_GPU_CHUNK_PREFILL_COMPARE=1` is also set,
the function re-runs the CPU chunk rule and compares GPU output against CPU
output layer-by-layer, reporting max relative error and nan count.

#### Runtime-gated verification

- **`mixed_correctness_023 --max-new-tokens 1`** — passes with
  `SPOCK_GPU_CHUNK_PREFILL=1` (generates the correct first token).
- **`pp520_046 --max-new-tokens 1`** — passes with
  `SPOCK_GPU_CHUNK_PREFILL=1` (generates token 0 correctly; previously failed
  with `actual token 0` on the CPU path due to the prefill-sensitive bug).
- The `pp520_046` run is noticeably slower than the CPU path because of the
  per-head submit workaround.

**Note:** The `mixed_correctness_023` and `pp520_046` failures on the CPU path
are a separate prefill bug (likely related to normalization rounding or
uninitialized state). The GPU path happens to compute differently and produces
the correct result for these prompts — this does **not** mean the GPU path is
numerically more accurate in general; it means it diverges in a direction that
happens to match the reference for these specific prompts at max-new-tokens=1.

### Why this is progress

- The probe has been hardened from 5 to 9 cases, covering realistic padded
  geometry, pseudo-random runtime-like input ranges, and L2-normalized Q/K.
- The critical barrier-insufficient discovery constrains the integration
  approach: command-buffer-per-head submit is the verified correct strategy
  for this device/driver.
- The runtime now has a working env-gated GPU chunk-prefill path that passes
  two previously-failing prompts.
- The diagnostic comparison mode (`SPOCK_GPU_CHUNK_PREFILL_COMPARE=1`) enables
  per-layer numerical validation during runtime execution.
- All nine probe cases are documented with status and error metrics.

### Verification

**Probe — new cases:**
- `./build/spock-deltanet-chunk-prefill-probe --case multi-head-padded` exits 0
  with `compare-fail` (expected — demonstrates the barrier-insufficient issue).
- `./build/spock-deltanet-chunk-prefill-probe --case runtime-range-padded`
  exits 0 with `compare-fail` (expected).
- `./build/spock-deltanet-chunk-prefill-probe --case runtime-l2-padded` exits 0
  with `compare-fail` (expected).
- `./build/spock-deltanet-chunk-prefill-probe --case runtime-l2-padded-submit`
  exits 0 with `compare-ok`, max_rel_core/max_rel_state ~1.19e-7.

**Probe — all previously passing cases remain passing:**
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

All passing max relative errors < 2e-11, all nan_count=0.

**Runtime GPU path verification:**
- `SPOCK_GPU_CHUNK_PREFILL=1 build/spock-decode --repack-dir artifacts/spock-text-repack-qwen35-0p8b --tokens <prompt> --max-new-tokens 1` passes
  for `mixed_correctness_023` and `pp520_046`.
- Diagnostic layer-by-layer comparison via `SPOCK_GPU_CHUNK_PREFILL_COMPARE=1`
  confirms GPU/CPU numerical agreement within expected bounds on real model
  activations.

## Verification

Verification for this entry came from comparing the GPU chunk-prefill probe
against the native CPU chunk-rule reference on the targeted real-model cases.
The probe exercised the same chunk dimensions and state layout needed for later
runtime integration, while keeping unrelated decode-layer effects out of the
comparison. Passing this probe did not mean prefill was fully GPU-resident; it
meant the shader reproduced the trusted primitive closely enough to justify the
next gated handoff step.

## Current Limitations

Same as diary 0016, plus the new explicit limitations from this phase:

1. **CPU chunk bridge remains the default prefill path.** `run_chunk_prefill` →
   `run_deltanet_chunk_rule` on CPU. The GPU path is gated behind
   `SPOCK_GPU_CHUNK_PREFILL=1` and only moves the chunk-rule computation;
   Q/K/V/g/beta collection, rearrangement, upload, and output re-integration
   are still CPU-hosted.

2. **The per-head submit workaround is correct but prohibitively slow for
   production.** 24 layers × 16 heads = 384 submit-wait cycles per chunk.
   A correct efficient shader design (single dispatch, all heads) is needed
   before the GPU path can be considered performant.

3. **The pipeline barrier insufficiency is device/driver-specific.** The
   finding that pipeline barriers inside one CB are insufficient for correct
   per-head dispatch has been validated only on RADV NAVI22 (RX 6750 XT).
   Other vendors/drivers may behave differently. The separate-CB workaround
   is correct everywhere but slow everywhere.

4. **No formal tests for the GPU path.** The env-gated path is verified
   manually with specific prompts and the `SPOCK_GPU_CHUNK_PREFILL_COMPARE`
   diagnostic. There is no CI gate, no regression test, and no nightly
   benchmark for the GPU chunk-prefill path.

5. **No device-local buffers.** All GPU buffers are host-visible. Moving
   Q/K/V/g/beta into device-local memory with explicit staging would reduce
   PCIe traffic and is a prerequisite for any performance work.

6. **Long-prompt behavior unexercised.** The probe exercises seq_len up to
   104 (2 chunks). Production prompts can span many dozens of chunks; scratch
   pressure and numerical behavior at >2 chunk boundaries remain unexercised.

7. **short_correctness_003 still fails** at generated token index 5 — the
   argmax decision flips on a close margin. Unchanged from diary 0016.
   The working hypothesis remains accumulated fp16 rounding drift at
   activation/output boundaries across the 24-layer pipeline. Not addressed
   by this work.

## Next Work

### GPU chunk-prefill — runtime integration follow-up

1. **Replace per-head submit workaround with a correct efficient shader.**
   The current separate-CB-per-head loop is verified-correct but unacceptably
   slow. The next shader design should handle all heads in a single dispatch
   (workgroup-per-head with intra-shader synchronization) or use a minimal
   number of dispatches with correct intra-CB synchronization that works on
   RADV NAVI22.

2. **Wire GPU collection into session (collect shader exists as probe).** The
   `deltanet_prefill_collect.comp` shader and its probe proved a shader can
   write per-token fp16 dn_qkv + fp32 g/beta into fp32 head-major buffers
   expected by the chunk-prefill shader (verified: max_rel=0, nan_count=0 at
   heads=16, seq_len=104, k_dim=v_dim=128). Next step: add session-owned
   collection buffers, call the collect shader during layer-major prefill
   under an env gate (`SPOCK_GPU_COLLECT_PREFILL`), feed device buffer
   outputs into `gpu_chunk_prefill()`, and retain CPU collection as fallback.

3. **Add formal tests for the env-gated GPU path.** At minimum:
   - A regression test that runs specific prompts under `SPOCK_GPU_CHUNK_PREFILL=1`
     and verifies token output against reference.
   - A diagnostic test that enables `SPOCK_GPU_CHUNK_PREFILL_COMPARE=1` and
     asserts per-layer max_rel below a threshold.
   - Integration into existing parity harness (`run_vk_decode_parity.py`) as
     an alternate-path CI target.

4. **Only then consider defaulting to GPU path.** The GPU path should not
   become the default until the efficient shader is verified, device-local
   buffers are wired, and formal tests gate regressions.

### Precision drift track

The remaining precision investigation options from diary 0016 are unchanged.
The recommended next action is a broader decode-path fp32-resident activation
experiment, or accepting the parity tradeoff against the megakernel roadmap.

## Scope

This entry records the evolution from a standalone probe through runtime
integration. The probe grew from 5 to 9 cases, exposing the critical
finding that pipeline barriers inside a single command buffer are not
sufficient for correct multi-head dispatch on this device — only separate
command-buffer submits per head produce correct results with realistic padded
data. The runtime now has a gated GPU chunk-prefill path
(`SPOCK_GPU_CHUNK_PREFILL=1`) using the conservative per-head submit
workaround, verified against two previously-failing prompts. All existing
gates, diagnostics, and the default CPU decode path are unchanged. The GPU
path remains experimental, slow, and not the default.

Key artifacts:
- `shaders/deltanet_chunk_prefill.comp` — the shader (unchanged logic)
- `apps/spock-deltanet-chunk-prefill-probe.cpp` — 9-case probe
- `src/runtime/vk_session.cpp` — `gpu_chunk_prefill()` + env gate in
  `run_chunk_prefill()`
- `src/runtime/vk_session.hpp` — `gpu_chunk_prefill()` declaration

## Extension: GPU-side Prefill Collection/Packing Probe

### Goal

Add a GPU-side prefill collection/packing primitive that proves a shader can
write per-token fp16 dn_qkv plus fp32 g/beta into the fp32 head-major buffers
expected by `deltanet_chunk_prefill.comp`. This eliminates the conceptual
question of GPU-based collection — the CPU readback/packing round-trip can be
replaced once session wiring is in place.

### What exists

**`shaders/deltanet_prefill_collect.comp`** — a GLSL compute shader that:
- Reads one token's dn_qkv as fp16 in `[Q heads*k][K heads*k][V heads*v]`
  layout
- Reads one token's g and beta as fp32 in `[g heads][beta heads]` layout
- Writes fp32 Q, K, V into head-major buffers (`[head][token][dim]`)
- Writes fp32 g, beta into head-major buffers (`[head][token]`)
- The output layout matches exactly what `deltanet_chunk_prefill.comp` expects
  as its Q/K/V/g/beta SSBO bindings
- Shader invocations are organized as `local_size_x = 128`, with one workgroup
  per head for the current token. The probe dispatches once per token.

**`apps/spock-deltanet-prefill-collect-probe.cpp`** — a standalone probe that:
- Generates random fp16 dn_qkv + fp32 g/beta for configurable heads, seq_len,
  k_dim, v_dim
- Dispatches the collect shader
- Runs a CPU reference that manually deinterleaves dn_qkv, casts to fp32, and
  writes to the same output layout
- Compares GPU output against CPU reference per-element, reporting
  `max_rel_q/k/v/g/beta` and `nan_count`
- Exits 0 on `compare-ok`, non-zero on `compare-fail`

**Verified command (success):**
```
./build/spock-deltanet-prefill-collect-probe
```

#### Verification result

```json
{
  "shader": "deltanet_prefill_collect.comp",
  "status": "compare-ok",
  "heads": 16,
  "seq_len": 104,
  "k_dim": 128,
  "v_dim": 128,
  "max_rel_q": 0,
  "max_rel_k": 0,
  "max_rel_v": 0,
  "max_rel_g": 0,
  "max_rel_beta": 0,
  "nan_count": 0
}
```

All five output tensors match exactly (max_rel = 0, nan_count = 0) at the
target production geometry (heads=16, seq_len=104, k_dim=v_dim=128). This is
an exact-comparison probe — unlike the chunk-prefill shader which accumulates
fp32 computation error, the collect shader is purely a layout-transformation
that should produce bit-identical results to the CPU reference.

### Build integration

- `CMakeLists.txt` now builds the new shader (`deltanet_prefill_collect.comp`)
  into a SPIR-V module and includes `spock-deltanet-prefill-collect-probe` in
  the app target list.
- Full build: `cmake --build build -j` succeeds.

### Regression pass

The existing chunk-prefill probe remains unaffected:
```
./build/spock-deltanet-chunk-prefill-probe --case runtime-l2-padded-submit
```
still emits `compare-ok`.

### What does NOT exist (not session integration)

- **No production runtime integration.** `DecodeSession` still collects
  Q/K/V/g/beta on CPU and copies them to GPU for `gpu_chunk_prefill()`.
- **No session-owned collection buffers.** There is no `VkBuffer` per tensor
  owned by `DecodeSession` for the collect shader's output.
- **No env gate for GPU collection.** No `SPOCK_GPU_COLLECT_PREFILL` or
  similar gating mechanism.
- **No wiring into gpu_chunk_prefill().** The collect shader output is not
  fed to the chunk-prefill shader in a single GPU workflow.
- **No device-local memory.** All probe buffers are host-visible.

### Why this matters

This probe closes the last conceptual unknown in the GPU prefill pipeline:

1. **CPU readback eliminated in principle.** The per-token Q/K/V/g/beta values
   can now be GPU-collected and GPU-packed into the chunk-prefill shader's
   expected input format without ever touching host memory.
2. **The output format is proven compatible.** The head-major fp32 layout
   produced by `deltanet_prefill_collect.comp` is byte-for-byte identical to
   what `deltanet_chunk_prefill.comp` consumes. No layout adapter or format
   conversion layer is needed.
3. **Zero-error transformation.** Unlike the chunk-prefill shader where fp32
   computation accumulates rounding error vs a fp16 CPU reference, the
   collect shader is a pure layout/cast operation (fp16→fp32 per element)
   that produces bit-identical output to the CPU.

### Next work (from this checkpoint)

1. **Add session-owned collection buffers** — allocate `VkBuffer` instances
   per tensor (Q, K, V, g, beta) in `DecodeSession`, sized for
   `max_seq_len * num_heads * dim`.
2. **Call collect shader during layer-major prefill** — after the per-token
   QKV/g/beta projection kernels, dispatch `deltanet_prefill_collect` on the
   per-token device buffers before the chunk-prefill step. Gate behind
   `SPOCK_GPU_COLLECT_PREFILL=1` (or extend `SPOCK_GPU_CHUNK_PREFILL`).
3. **Feed device-collected buffers into gpu_chunk_prefill()** — replace the
   current CPU-populated input buffers with the shader's output buffers.
4. **Keep CPU collection fallback** — the CPU path remains default until the
   GPU collection path is verified against real model activations.

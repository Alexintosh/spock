# 0024: Runtime Tiled Chunk-Prefill Gate — Integrated Single-Dispatch Path

## Goal

Wire the tiled single-dispatch `deltanet_chunk_prefill_tiled.comp` shader
(proved correct in diary 0023 on a synthetic probe case) into the actual
decode runtime behind a new env gate `SPOCK_GPU_CHUNK_PREFILL_TILED=1`.
This removes the per-head submit workaround (384 submit-wait cycles per
chunk) for chunk-prefill when the tiled gate is active, and gives both the
CPU-collected and GPU-collected chunk-prefill paths a shared single-dispatch
backend.

This is a major runtime step on the critical path to the RX 6750 XT
Vulkan-native engine. The per-head submit loop was the dominant runtime
cost of the gated GPU path; removing it brings the gated path from 99.79 sec
to 10.67 sec on the `short_correctness_001` CTest (9.4× speedup), within
1.7× of the non-gated baseline.

## Inference Concepts

### Three-tier env gate design

The runtime now has three layers of env-gated chunk-prefill behavior,
hierarchically composed:

| Gate(s) | Behavior |
|---------|----------|
| (none) | All-CPU recurrent decode (default, unchanged) |
| `SPOCK_GPU_CHUNK_PREFILL=1` | GPU chunk-prefill, per-head submit (unchanged) |
| `SPOCK_GPU_CHUNK_PREFILL=1` + `SPOCK_GPU_CHUNK_PREFILL_TILED=1` | GPU chunk-prefill, single dispatch via tiled shader |
| `SPOCK_GPU_CHUNK_PREFILL=1` + `SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1` | GPU-collected Q/K/V/g/beta, per-head submit (unchanged) |
| Same + `SPOCK_GPU_CHUNK_PREFILL_TILED=1` | GPU-collected + single-dispatch tiled |

`SPOCK_GPU_CHUNK_PREFILL_TILED=1` has no effect unless
`SPOCK_GPU_CHUNK_PREFILL=1` is also set. The TILED gate controls only the
dispatch mode — it replaces the per-head loop with a single
`vkCmdDispatch(num_heads, ceil(v_dim / TILE_V), 1)` call. It does not
change which data path (CPU-collected or GPU-collected) feeds the shader.

### What the tiled dispatch removes

Before: per layer, per head: one `vkCmdDispatch(1, 1, 1)` plus one
`vkQueueSubmit` + `vkWaitForFences` pair. For 24 DeltaNet layers × 16 heads
= 384 submit-wait cycles per chunk. Each cycle pays full driver overhead —
command-buffer recording, queue submission, fence wait. This was the
dominant cost of the gated GPU path.

After (with TILED=1): one `vkCmdDispatch(num_heads, tile_count, 1)` per
DeltaNet layer. 24 dispatches/submissions total for the 24 DeltaNet layers,
instead of 384 per-head dispatch/submission pairs. The per-head submit
workaround is completely bypassed, but the runtime still submits and waits
once per DeltaNet layer.

The tiled shader design (diary 0023) makes this possible: each workgroup
handles one head and one v-dimension tile, writes to disjoint output slices,
and needs no cross-workgroup synchronization. The workgroup decomposition
is the same as the synthetic probe, now driven from real session buffers.

### Why this is not the final megakernel

The tiled path removes CPU-mediated per-head submits, but it is **not** a
fully fused megakernel:

- **CPU orchestration remains.** The runtime still iterates layers on the
  host, binds per-layer descriptor sets, dispatches the tiled shader, then
  reads back chunk output and state for the next decode step.
- **Staging downloads/upload still happen.** Chunk output and state buffers
  transit through host-visible staging for the autoregressive decode loop.
- **No persistent dispatch.** Each chunk is recorded, submitted, waited on,
  and the results are consumed before the next decode step begins.

The tiled dispatch is a major intermediate step — it removes a correctness
blocker (per-head submit) that would have been a dead end for the
megakernel, and it validates the workgroup decomposition that a fused
megakernel would use.

## Implementation Work Completed

### New env gate: `SPOCK_GPU_CHUNK_PREFILL_TILED`

Added a new environment variable check in `vk_session.cpp`, read once at
`run_chunk_prefill()`:

```cpp
const char* tiled_env = std::getenv("SPOCK_GPU_CHUNK_PREFILL_TILED");
const bool tiled = tiled_env && tiled_env[0] == '1' && tiled_env[1] == '\0';
```

This is checked only after `SPOCK_GPU_CHUNK_PREFILL=1` has been confirmed.
If `SPOCK_GPU_CHUNK_PREFILL` is not set, the TILED gate has no effect and
the runtime continues with the default non-gated path.

### Pipeline/module loading

The tiled shader (`deltanet_chunk_prefill_tiled.comp`) is compiled to a
Vulkan pipeline at session init, alongside the existing
`deltanet_chunk_prefill.comp` pipeline. Both pipelines share:

- **Descriptor layout**: 7 storage-buffer bindings for Q, K, V, g, beta,
  output, and init_state — identical to the existing per-head submit path.
- **Push constants layout**: 40 bytes (10 × uint32) — same structure as
  the per-head submit path. The shader reads push constants at the same
  offsets.

This means no new descriptor set, no new push constant structure, and no
new binding management. The existing binding/push-constant infrastructure
is reused directly — the only difference is the shader module and the
dispatch dimensions.

### Refactoring: bool `tiled` parameter

The initial experimental patch duplicated two full runtime functions
(`gpu_chunk_prefill` and `gpu_chunk_prefill_from_gpu_collect`) for the
tiled variant — a copy-paste diff that was large and fragile. This was
refactored: both functions now accept a `bool tiled` parameter, and the
dispatch path selects at the innermost dispatch site:

```cpp
void gpu_chunk_prefill(..., bool tiled);
void gpu_chunk_prefill_from_gpu_collect(..., bool tiled);
```

The dispatch differentiation:

```cpp
if (tiled) {
    uint32_t tile_count = (v_dim + TILE_V - 1) / TILE_V;
    vkCmdDispatch(cmd, num_heads, tile_count, 1);
} else {
    for (uint32_t h = 0; h < num_heads; ++h) {
        vkCmdDispatch(cmd, 1, 1, 1);
    }
}
```

The `tiled=true` path selects `deltanet_chunk_prefill_tiled.comp`'s
pipeline; `tiled=false` continues using `deltanet_chunk_prefill.comp`.
Everything else — binding layout, push constants, buffer offsets,
descriptor updates — is identical between the two paths.

### Diagnostics enhancement

`SPOCK_GPU_CHUNK_PREFILL_COMPARE=1` now includes `dispatch=tiled` in its
stderr diagnostic line when the tiled gate is active:

- CPU-collected path: `SPOCK_GPU_CHUNK_PREFILL_COMPARE dispatch=tiled ...`
- GPU-collected path: `SPOCK_GPU_CHUNK_PREFILL_COMPARE source=gpu_collect dispatch=tiled ...`

This allows distinguishing tiled-vs-per-head diagnostic output when
comparing runs.

### No new files

The tiled shader (`shaders/deltanet_chunk_prefill_tiled.comp`) and the
standalone probe (`apps/spock-deltanet-chunk-prefill-tiled-probe.cpp`) were
created in diary 0023. The runtime integration adds a pipeline/module
loading entry and dispatch logic in existing `vk_session.cpp` files. No new
source files are required for the integration.

### CTest registration: `spock_vk_decode_gpu_collect_chunk_prefill_tiled`

A new CTest test was registered in `CMakeLists.txt`:

- **`spock_vk_decode_gpu_collect_chunk_prefill_tiled`** — runs
  `short_correctness_001 --max-new-tokens 1` with all three env gates:
  `SPOCK_GPU_CHUNK_PREFILL=1`,
  `SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1`,
  `SPOCK_GPU_CHUNK_PREFILL_TILED=1`.

This exercises the full GPU-collected + tiled dispatch path. The existing
`_short` (per-head submit) and `_baseline` (no env gates) tests remain unchanged.
The three tests form a CTest suite registered under
`-R spock_vk_decode_gpu_collect_chunk_prefill`:

| Test | Env | Dispatch | Expected runtime |
|------|-----|----------|-----------------|
| `_short` | GPU collect + GPU chunk-prefill | per-head submit | ~99 sec |
| `_tiled` | GPU collect + GPU chunk-prefill + tiled | single dispatch | ~10 sec |
| `_baseline` | (no env gates) | N/A | ~6 sec |

## Verification

All commands run on the target RADV RX 6750 XT (NAVI22) hardware.

### Build

```sh
source ~/.zshrc && cmake -S . -B build && cmake --build build -j
```

Passed cleanly. No compilation or linking errors.

### git diff --check

```sh
source ~/.zshrc && git diff --check
```

No whitespace errors.

### Standalone tiled probe (synthetic verification)

```sh
timeout 600s ./build/spock-deltanet-chunk-prefill-tiled-probe
```

Result (deltanet_chunk_prefill_tiled.comp standalone, same synthetic case
as diary 0023):

```
{"status":"compare-ok","nan_count":0,
 "max_rel_core":1.19209e-07,"max_rel_state":1.19208e-07,
 "max_abs_core":1.19209e-07,"max_abs_state":2.38419e-07}
```

The standalone probe confirms the tiled shader still produces correct output
on the synthetic case. This is a regression check against diary 0023 (the
shader itself was not modified during integration — the runtime integration
loads the same `.comp` file).

### CPU-collected + tiled parity (real decode)

```sh
SPOCK_GPU_CHUNK_PREFILL=1 SPOCK_GPU_CHUNK_PREFILL_TILED=1 \
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

The CPU-collected path (staging downloads, half_to_float, CPU-side
prefill_chunks_ population) uses the tiled shader instead of per-head
submits. The output token matches the reference.

### GPU-collected + tiled parity

```sh
SPOCK_GPU_CHUNK_PREFILL=1 SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1 \
  SPOCK_GPU_CHUNK_PREFILL_TILED=1 \
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

The GPU-collected path (device-local per-layer Q/K/V/g/beta buffers,
no CPU intermediate packing) uses the tiled shader. The output token
matches the reference.

### GPU-collected + tiled + compare diagnostic

```sh
SPOCK_GPU_CHUNK_PREFILL=1 SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1 \
  SPOCK_GPU_CHUNK_PREFILL_TILED=1 SPOCK_GPU_CHUNK_PREFILL_COMPARE=1 \
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

The direct decode output showed:

- `generated_tokens [271]`
- Per-layer diagnostic lines printed for all 18 DeltaNet layers
- Each line: `source=gpu_collect dispatch=tiled nan_count=0`
- Absolute differences: core `max_abs` up to ~5.96e-08, state `max_abs` up
  to ~4.77e-07 on this short run. Relative values can inflate near zero due
  to the existing denominator behavior in the compare logic; this is a
  property of the diagnostic comparison, not the shader output.

### CTest regression: GPU-collect suite (3/3 passed)

```sh
ctest --test-dir build --output-on-failure -R spock_vk_decode_gpu_collect_chunk_prefill
```

| Test | Status | Time |
|------|--------|------|
| `spock_vk_decode_gpu_collect_chunk_prefill_short` | Passed | 99.79 sec |
| `spock_vk_decode_gpu_collect_chunk_prefill_tiled` | Passed | 10.67 sec |
| `spock_vk_decode_gpu_collect_chunk_prefill_short_baseline` | Passed | 6.26 sec |

Key observations:

- **9.4× speedup** on the gated GPU-collect path: 99.79 sec (per-head
  submit) → 10.67 sec (tiled). This is entirely from removing the
  384 submit-wait cycles per chunk.
- **1.7× slower than baseline** (6.26 sec). The tiled path still pays extra
  dispatch, shader computation, and result readback overhead that the
  non-gated baseline does not. For a single token at seq_len=9, the GPU path is
  unlikely to beat the CPU path — the GPU advantage is expected at longer
  sequences and larger batch sizes.
- The `_short` test (per-head submit) runtime decreased from 115.14 sec
  (diary 0022) to 99.79 sec. This is ambient system variation, not a
  code change — the per-head submit code path is unmodified.

### CTest regression: Targeted gates (4/4 passed)

```sh
ctest --test-dir build --output-on-failure \
  -R "spock_capabilities|spock_deltanet_chunk_unit|spock_vk_decode_prefill_handoff_mismatch|spock_diagnose_handoff_mc023"
```

All 4/4 passed:
- `spock_capabilities` — pass
- `spock_deltanet_chunk_unit` — pass
- `spock_vk_decode_prefill_handoff_mismatch` — pass (143.53 sec)
- `spock_diagnose_handoff_mc023` — pass (7.73 sec)

Total: 7 CTest tests passed (3 GPU-collect suite + 4 targeted) plus 1
standalone probe plus the 3 parity harness runs (CPU-collected tiled,
GPU-collected tiled, GPU-collected tiled compare).

### Additional tiled runtime coverage

After the runtime gate was committed locally, two follow-up parity checks
extended coverage beyond the short CTest prompt:

```sh
SPOCK_GPU_CHUNK_PREFILL=1 SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1 \
  SPOCK_GPU_CHUNK_PREFILL_TILED=1 \
  python3 tests/run_vk_decode_parity.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --reference tests/data/reference_tokens.jsonl \
  --ids mixed_correctness_023,pp520_046 \
  --max-new-tokens 1
```

```
{"status":"ok","checked":2,"failures":[]}
```

```sh
SPOCK_GPU_CHUNK_PREFILL=1 SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1 \
  SPOCK_GPU_CHUNK_PREFILL_TILED=1 \
  python3 tests/run_vk_decode_parity.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --reference tests/data/reference_tokens.jsonl \
  --ids short_correctness_001 \
  --max-new-tokens 16
```

```
{"status":"ok","checked":1,"failures":[]}
```

This does not replace broader prompt coverage, but it raises confidence that
the tiled runtime path survives both longer prompts and multiple decode steps.

One more follow-up raised the longer-prompt checks from one generated token
to four generated tokens:

```sh
SPOCK_GPU_CHUNK_PREFILL=1 SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1 \
  SPOCK_GPU_CHUNK_PREFILL_TILED=1 \
  python3 tests/run_vk_decode_parity.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --reference tests/data/reference_tokens.jsonl \
  --ids mixed_correctness_023 \
  --max-new-tokens 4
```

```
{"status":"ok","checked":1,"failures":[]}
```

```sh
SPOCK_GPU_CHUNK_PREFILL=1 SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1 \
  SPOCK_GPU_CHUNK_PREFILL_TILED=1 \
  python3 tests/run_vk_decode_parity.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --reference tests/data/reference_tokens.jsonl \
  --ids pp520_046 \
  --max-new-tokens 4
```

```
{"status":"ok","checked":1,"failures":[]}
```

## Known Limitations

1. **Still env-gated, not default.** The tiled path is opt-in via env vars.
   Defaulting to GPU chunk-prefill — tiled or not — requires more prompt
   coverage, longer-sequence testing, and performance characterization.
   This entry does not change the default decode path.

2. **Not the final Vulkan-native megakernel.** The tiled path removes
   per-head submits but still uses CPU orchestration around each chunk:
   per-layer descriptor binding, per-layer submit/wait, staging downloads
   and uploads for chunk output and state. A true megakernel would fuse
   the entire decode pass into a single persistent dispatch.

3. **Zero initial state assumption.** The tiled shader assumes zero initial
   state for chunk prefill — matching the actual prefill usage where no
   prior state exists. The binding 6 (`init_state`) is present in the
   descriptor layout but currently unused by the tiled path. Nonzero state
   (for recompute or partial prefill scenarios) would require state loading
   in the shader.

4. **Performance not fully characterized.** The 10.67 sec CTest runtime
   (1.7× baseline) is for a single token at seq_len=9. Longer sequences,
   multi-token decode, and varying chunk sizes may shift the relative
   cost. Performance profiling should use representative workloads, not
   the minimal CTest prompt.

5. **Redundant computation exposed.** Each v-tile workgroup recomputes
   `k_cumdecay` independently (from diary 0023, limitation 3). This was a
   deliberate correctness-first choice in the shader. It does not affect
   correctness, but it limits occupancy and L1 utilization. Optimization
   is deferred until the tiled path is stable and performance-tuned.

6. **More prompt coverage needed.** The tiled path has been verified on
   `short_correctness_001` through 16 generated tokens, and on
   `mixed_correctness_023`/`pp520_046` through 4 generated tokens. Broader P0
   subsets, 512+ token prompts, and 16-token decode on longer prompts are
   still pending.

7. **Compare diagnostic relative-value inflation.** Near-zero values in
   the diagnostic comparison logic inflate relative error metrics. This is
   a property of the compare function's denominator behavior (divide-by-
   near-zero), not an actual numerical issue in the shader output. The
   `nan_count=0` and `max_abs` metrics are the authoritative indicators.

## Next Work

### Near-term: Prompt coverage expansion

Run the tiled path on additional prompts at various sequence lengths to
confirm correctness extends beyond the first expanded checks. Priority
remaining prompts from the P0 corpus:

- `mixed_correctness_025–027` and other mixed prompts.
- Longer prompts (>512 tokens) to stress test the shader and driver.

### Near-term: Multi-token decode coverage

Expand verification to multiple decode steps on longer prompts. The tiled
path removes the per-head submit cost, making multi-token gated decode
practical for CI. The first 16-token check now passes on
`short_correctness_001`; the next target is multi-token coverage on mixed
and longer prompts.

### Medium-term: Nonzero init_state

The tiled shader already has binding 6 (`init_state`) wired in the
descriptor layout. Consuming it requires:
- Loading `init_state[head * k_dim * v_dim + kd * v_dim + gv]` for the
  workgroup's v-tile slice.
- Applying the chunk-loop state update starting from that loaded state
  instead of zero initialization.

This is needed for recompute-based chunk prefill and for partial prefill
of long prompts across multiple chunk invocations.

### Medium-term: Performance characterization

Once the tiled path is validated on multiple prompts and multi-token decode,
measure:

- Wall time per token for the tiled path vs CPU baseline vs per-head submit.
- Chunk size sensitivity (chunk_size=64 is fixed; larger chunks may improve
  GPU utilization or hit timeouts).
- Occupancy and register pressure profiling on the tiled shader.
- Driver overhead of the 24-dispatch-per-chunk orchestration.

### Long-term: Toward the fused megakernel

The tiled workgroup decomposition is a building block for a fused compute
megakernel:

- Share `k_cumdecay` across tiles of the same head (precomputation dispatch
  or workgroup collaboration).
- Fuse chunk-prefill state output with Phase C (input_norm, Z gate, MLP
  compute) into a single per-layer dispatch.
- Fold the entire decode pass into a single `persistent_dispatch` that owns
  autoregressive token generation without host mediation.

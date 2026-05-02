# 0022: CTest Regression Gate for GPU Collect → GPU Chunk-Prefill Path

## Goal

Add a CTest regression gate that exercises the double-gated GPU-collect →
GPU chunk-prefill path (`SPOCK_GPU_CHUNK_PREFILL=1` +
`SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1`) on the
`short_correctness_001` prompt at `--max-new-tokens 1`, alongside a no-env
baseline test on the same prompt.

This is a pure test-infrastructure addition. It does not change any runtime
code, shaders, or build targets. The gate protects the work from diaries 0020
and 0021 from accidental regression — if a future change breaks the GPU-collect
→ chunk-prefill wiring, this test will fail before any manual re-verification
is needed.

This does not complete full GPU prefill offload. The new gate protects only the
current double-gated path while the per-head submit inefficiency (384
submit-wait cycles per chunk) remains the main blocker.

## Implementation Work Completed

### Two new CTest tests in CMakeLists.txt

Two `add_test()` entries were added to the CTest gate list in
`CMakeLists.txt` (inside the existing `if(TARGET spock-decode)` block):

1. **`spock_vk_decode_gpu_collect_chunk_prefill_short`** — the gated test.
   Runs `tests/run_vk_decode_parity.py` with:
   - `--ids short_correctness_001`
   - `--max-new-tokens 1`
   - Environment: `SPOCK_GPU_CHUNK_PREFILL=1` and
     `SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1`

   This exercises the full double-gated path: GPU collection dispatches per
   token, GPU-collected Q/K/V/g/beta in device-local per-layer segments feed
   `deltanet_chunk_prefill.comp` directly, and the CPU collection bridge is
   bypassed (no staging downloads, no half_to_float, no prefill_chunks_
   population).

2. **`spock_vk_decode_gpu_collect_chunk_prefill_short_baseline`** — the
   baseline test. Runs the same parity harness with the same arguments but
   **no** env vars. This exercises the default (all-CPU recurrent) decode
   path on the same prompt, serving as a quick reference for what a correct
   run looks like.

Both tests share the same command line:
```
python3 tests/run_vk_decode_parity.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --reference tests/data/reference_tokens.jsonl \
  --ids short_correctness_001 \
  --max-new-tokens 1
```

Only the `set_tests_properties()` env override distinguishes them: the gated
test sets the two env vars; the baseline test has no `set_tests_properties()`
entry and runs with the default (empty) environment.

### Existing targeted gates preserved

The four pre-existing targeted CTest checks used throughout this GPU-prefill
work were left untouched:

- `spock_capabilities`
- `spock_deltanet_chunk_unit`
- `spock_vk_decode_prefill_handoff_mismatch`
- `spock_diagnose_handoff_mc023`

Together with the two new tests, this gives a six-test focused GPU-prefill
validation set. The repository has additional tests outside that focused set.

## Inference Concepts

### Why a CTest gate matters

The GPU-collect → chunk-prefill path (diaries 0020 and 0021) involves wiring
across three subsystems:

- **Collect shader dispatch** (`deltanet_prefill_collect.comp`) — writes
  per-token activations into persistent device-local buffers.
- **Chunk-prefill shader dispatch** (`deltanet_chunk_prefill.comp`) — reads
  the collected buffers and computes chunk-rule output.
- **Runtime orchestration** (`gpu_chunk_prefill_from_gpu_collect()` in
  `vk_session.cpp`) — manages descriptor set bindings, per-layer offsets,
  and command-buffer submission.

A change to any of these — a shader binding renumbering, a buffer layout
change, a descriptor set reformatting — could silently break the gated path
without affecting the default (all-CPU) path. The CTest gate catches such
breakage at `ctest` time rather than during a manual verification session.

### Why a baseline test alongside the gated test

The baseline test (`_baseline`) runs the same prompt through the default
decode path with no env gates. It exists as an immediate diagnostic reference:

- If the gated test fails but the baseline passes, the problem is in the
  gated wiring. The user can focus on the gated-specific code paths without
  asking whether the prompt itself is broken.
- If both fail, the problem may be in shared infrastructure (model artifact,
  weight loading, reference data) or in the hardware/driver state.
- The baseline is fast (5.11 sec observed) compared to the gated test
  (115.14 sec observed), making it a quick first check when debugging.

### Why `short_correctness_001` at `--max-new-tokens 1`

The `short_correctness_001` prompt (seq_len=9) is the same prompt used for
manual verification in diaries 0020 and 0021. It is the shortest prompt in
the parity test suite and finishes in the shortest time — critical for a
CTest gate that will run on every build.

The `--max-new-tokens 1` flag generates exactly one token. This is sufficient
to prove the gated path is operational (the collect → chunk-prefill pipeline
runs, the output token matches the reference) without inflating test runtime
unnecessarily. Full multi-token decode gating is deferred until the per-head
submit inefficiency is resolved.

### Why the gate is still not full GPU offload

The new CTest gate protects the current gated path from regression, but the
path itself is not production-default for two reasons:

1. **Per-head submit inefficiency.** Each layer dispatches one command buffer
   per head (24 × 16 = 384 submit-wait cycles per chunk). The gated test
   takes 115.14 sec for a single token on a seq_len=9 prompt — orders of
   magnitude slower than the default CPU path.

2. **Not the default.** The gated path is opt-in via env vars. No code change
   makes it the default execution mode. Defaulting is premature until (1) is
   fixed or a correct single-dispatch multi-head design is proven.

## Verification

All commands were run on the target RADV RX 6750 XT (NAVI22) hardware.

### Build

```sh
cmake -S . -B build && cmake --build build -j
```

Passed cleanly (no compilation or linking errors). Only CMakeLists.txt
changed; no code or shaders were modified.

### New CTest tests (2/2 passed)

```sh
ctest --test-dir build --output-on-failure -R spock_vk_decode_gpu_collect_chunk_prefill_short
```

```
    Start 14: spock_vk_decode_gpu_collect_chunk_prefill_short
1/2 Test #14: spock_vk_decode_gpu_collect_chunk_prefill_short ............   Passed  115.14 sec
    Start 15: spock_vk_decode_gpu_collect_chunk_prefill_short_baseline
2/2 Test #15: spock_vk_decode_gpu_collect_chunk_prefill_short_baseline ...   Passed    5.11 sec
```

The gated test returns `{"status":"ok","checked":1,"failures":[]}`.
The baseline test returns the same result. The large runtime difference
(115.14 sec gated vs 5.11 sec baseline) is expected — the gated path's
per-head submit workaround dominates wall time.

### Preserved CTest gate (4/4 passed)

```sh
ctest --test-dir build --output-on-failure -R "spock_capabilities|spock_deltanet_chunk_unit|spock_vk_decode_prefill_handoff_mismatch|spock_diagnose_handoff_mc023"
```

All four pre-existing tests pass identically to the baseline. No regression
from the CMakeLists.txt edit.

### Focused GPU-prefill gate coverage

The two new tests and the four preserved checks all pass. This is the focused
six-test validation set for the current GPU-prefill work; it is not a claim
that every repository test was run in this checkpoint.

## Artifact Baseline (committed state)

The following changes were committed for this milestone:

- `CMakeLists.txt` — Added two `add_test()` entries for
  `spock_vk_decode_gpu_collect_chunk_prefill_short` (gated with
  `SPOCK_GPU_CHUNK_PREFILL=1;SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1`)
  and `spock_vk_decode_gpu_collect_chunk_prefill_short_baseline` (no env
  vars), both running `short_correctness_001` at `--max-new-tokens 1`.

No runtime code, shaders, probes, or test infrastructure files were created or
modified. This is a pure test-registration change.

## Known Limitations

1. **Per-head submit inefficiency is the dominant cost.** The gated test takes
   115.14 sec vs the baseline's 5.11 sec because each chunk-prefill layer
   dispatches 16 command buffers (one per head). This is not a regression
   introduced by the CTest gate — it is the same inefficiency identified in
   diaries 0020 and 0021. The gate merely makes it measurable on every build.

2. **Only one prompt, one token.** The gate runs `short_correctness_001` at
   `--max-new-tokens 1`. Longer prompts and multi-token decode are not gated.
   A failure that only manifests on longer sequences or during autoregessive
   decode would not be caught. This is acceptable for now because the per-head
   submit slowdown makes multi-token gating impractical until the submit
   pattern is fixed.

3. **No compare-flag variant in CTest.** The gate does not exercise the
   diagnostic compare flags (`SPOCK_GPU_CHUNK_PREFILL_COMPARE=1` or
   `SPOCK_GPU_COLLECT_PREFILL_COMPARE=1`). Those remain manual-diagnostic
   tools. Adding a compare-mode CTest variant would double the gated test
   count without catching hardware or numerical regressions that the
   output-token gate already covers.

4. **Requires RADV hardware to run.** Like all CTest gates that depend on
   `spock-decode`, these tests fail meaningfully on llvmpipe or
   non-RADV-Vulkan configurations. The capabilities gate
   (`spock_capabilities`) guards against running decode tests on unsupported
   hardware.

## Next Work

### Primary: Fix per-head submit inefficiency

The per-head submit workaround (384 submit-wait cycles per chunk) is the
critical performance blocker and the only reason the GPU path is not the
default. A correct single-dispatch multi-head chunk-prefill design is needed.
This is the same next-work item from diaries 0020 and 0021 — unchanged.

### Secondary: Expand CTest gate to multi-token gated decode

Once the per-head submit is fixed and the gated path runtime approaches the
baseline, expand the CTest gate to `--max-new-tokens 16` or more to catch
autoregressive-state accumulation bugs in CI.

### Tertiary: Formal compare-flag regression

If a future change touches the compare-mode data paths, add a CTest variant
with `SPOCK_GPU_CHUNK_PREFILL_COMPARE=1` that asserts `nan_count=0` and
`max_abs` below a threshold. This can wait until the per-head submit fix
makes the gated path fast enough that doubling test time is acceptable.

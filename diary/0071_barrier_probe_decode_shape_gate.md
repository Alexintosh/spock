# 0071: Barrier Probe Decode-Shape Gate

## Goal

Extend `vk_barrier_probe` with a decode-shaped iteration mode so the existing persistent barrier/payload probe can run with a workload shape matching real decode geometry (tokens x layers). This is a semantic wrapper, not a rewrite: it sets `iterations = tokens * layers` and reports the decode-shape parameters in JSON output. The existing `--iterations` path is unchanged.

## Background

The persistent barrier probe (diary 0047) stress-tests software global barriers across N workgroups for K iterations. Diaries 0051/0052 added deterministic ALU and memory payloads. Diary 0055 added in-process repeats.

The production decode path for Qwen 3.5-0.8B generates T tokens, each passing through 24 layers, for a total of T*L layer-forward steps. A decode-shaped barrier workload runs `iterations = T * L` to simulate the same total layer-forward count. This is not real decode — no model weights, no attention, no DeltaNet recurrence — but it exercises the persistent barrier primitive at the correct iteration scale with the same payload structure.

## Implementation

### CLI additions

Two new options: `--tokens N` and `--layers N`.

- When both are supplied and >0: `iterations = tokens * layers` (decode-shape mode).
- When only one is supplied: error with message requiring both or neither.
- When either is zero: error.
- When the product overflows uint32: error.
- When neither is supplied: existing `--iterations` behavior unchanged.

### JSON output

When decode-shape mode is active, the JSON output includes three additional fields after `"workgroups"`:

```json
"tokens": 16,
"layers": 24,
"decode_shape_iterations": 384
```

When decode-shape mode is inactive, these fields are absent and output is identical to before.

### CTest gate

A new CTest `spock_barrier_probe_decode_shape` runs:

```
vk_barrier_probe --tokens 16 --layers 24 --workgroups 8 --payload-cols 128
```

This exercises 384 iterations (16 tokens x 24 layers) with 8 workgroups and a memory payload. It is short and deterministic — no timestamps, no multi-repeat, no wall-clock dependence. The test passes if the probe reports status "ok" with zero failures and zero trace mismatches.

The test parameters are chosen to be:
- Small enough for fast CTest execution (~384 iterations is well within the 750k boundary from diary 0053).
- Large enough to exercise decode-relevant iteration count (16 tokens is a short decode, 24 layers matches Qwen 3.5-0.8B).
- Memory payload via `--payload-cols 128` to stress the barrier with real memory traffic at a column count that fits comfortably in device-local buffers.

## Verification

### Build

```
cmake --build build -j
```

Compiles cleanly with no warnings.

### CTest decode-shape gate

```
ctest --test-dir build -R spock_barrier_probe_decode_shape --output-on-failure
```

Passes. The probe reports status "ok", zero failures, zero trace mismatches, generation 768 (384 iterations x 2 barriers), and checksum matching the expected deterministic value.

### Existing barrier probe test

```
ctest --test-dir build -R spock_barrier_probe --output-on-failure
```

Both `spock_barrier_probe_help` and `spock_barrier_probe_decode_shape` pass. The `--help` test confirms the new options are documented.

### Error path validation

Manually verified that:
- `--tokens 16` without `--layers` produces error JSON with status "error".
- `--layers 24` without `--tokens` produces error JSON with status "error".
- `--tokens 0 --layers 24` produces error JSON.
- `--tokens 16 --layers 0` produces error JSON.

### Diary check

```
ctest --test-dir build -R spock_diary_check --output-on-failure
```

Passes.

### Whitespace check

```
git diff --check
```

Clean.

## What This Is Not

- **Not real decode.** No model weights, no attention, no DeltaNet, no KV cache, no embedding, no LM head.
- **Not the megakernel.** No persistent dispatch of actual inference work.
- **Not a performance benchmark.** No wall-clock timing claims.
- **Not a substitute for under-load soaks.** 384 iterations is small; the diary 0050 1M-iteration soak and diary 0053 boundary work remain the stress references.
- **Not occupancy proof.** 8 workgroups is less than the 82-workgroup Luce reference.

## Known Limitations

- The decode-shape mode only shapes the iteration count. The per-iteration payload is still the deterministic hash-based barrier workload, not actual matvec-like compute.
- The column count (128) is arbitrary and does not correspond to any real model dimension.
- No multi-repeat decode-shape CTest yet; the gate is a single-run smoke test.

## Next Work

1. Run decode-shape barrier probes at higher token counts (e.g., 128 tokens x 24 layers = 3072 iterations) with timestamps to compare per-barrier timing against the diary 0049/0050 baseline.
2. Exercise decode-shape at 82 workgroups to match the Luce reference block count.
3. Consider adding `--decode-shape` as a convenience flag that sets tokens/layers to the Qwen 3.5-0.8B defaults.
4. Use the decode-shaped probe as a regression gate alongside the existing full-fast decode CTest (diary 0057) to catch persistent-barrier regressions before they reach the real decode path.

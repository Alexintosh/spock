# 0074: Qwen3.5 Decode-Shape Preset

## Goal

Add a `--qwen35-decode-shape-preset` convenience flag to `vk_barrier_probe` so the exact model-width synthetic persistent barrier workload from diary 0073 (tokens=128, layers=24, workgroups=82, payload_cols=1024) can be run without manually specifying all dimensions. This is a reproducibility and ergonomics improvement, not a functional change.

This is still a synthetic barrier/payload probe using uint32 memory traffic. It is not real decode, not model weights, not fp16/fp32 matvec, and not the megakernel.

## Background

Diary 0073 established the model-width decode-shaped barrier workload at the Qwen3.5 0.8B geometry (hidden_size=1024, 24 layers, 82 workgroups). Running this requires four explicit flags:

```
vk_barrier_probe --tokens 128 --layers 24 --workgroups 82 --payload-cols 1024
```

This is the canonical geometry for model-width barrier regression testing. A single preset flag reduces invocation errors and makes the intent explicit in test commands and documentation.

## Implementation

### CLI addition

New option: `--qwen35-decode-shape-preset`.

When supplied, sets defaults for tokens=128, layers=24, workgroups=82, payload_cols=1024, and enables decode-shape mode (iterations=tokens*layers=3072). User-supplied `--tokens`, `--layers`, `--workgroups`, or `--payload-cols` override the preset values. The preset flag is not exclusive: it can be combined with any other flags (`--timestamps`, `--repeats`, etc.).

Precedence rules:
- `--tokens N` overrides the preset's tokens=128.
- `--layers N` overrides the preset's layers=24.
- `--workgroups N` overrides the preset's workgroups=82.
- `--payload-cols N` overrides the preset's payload_cols=1024.
- `--iterations N` has no interaction with the preset; decode-shape mode (tokens*layers) takes precedence over `--iterations` as before.

The implementation adds `has_workgroups` and `has_payload_cols` tracking booleans (analogous to existing `has_tokens`/`has_layers`) to detect user overrides.

### JSON output

When the preset is active, the JSON output includes `"qwen35_decode_shape_preset": "active"` immediately after `"workgroups"`. This allows test infrastructure and downstream consumers to detect preset usage in output artifacts.

### CTest gate

A new CTest `spock_barrier_probe_qwen35_preset` runs:

```
vk_barrier_probe --qwen35-decode-shape-preset
```

This exercises the full preset (3072 iterations, 82 workgroups, 1024 payload columns) without timestamps or repeats. It is short and deterministic, passing if the probe reports status "ok" with zero failures and zero trace mismatches.

## Verification

### Build

```
cmake --build build -j
```

Compiles cleanly.

### CTest preset gate

```
ctest --test-dir build -R spock_barrier_probe_qwen35_preset --output-on-failure
```

Passes.

### Existing barrier probe tests

```
ctest --test-dir build -R spock_barrier_probe --output-on-failure
```

All existing tests (`spock_barrier_probe_help`, `spock_barrier_probe_decode_shape`) continue to pass. The `--help` test confirms the new flag is documented.

### Override validation

Manually verified that `--qwen35-decode-shape-preset --workgroups 4` overrides workgroups to 4 while keeping preset defaults for tokens, layers, and payload_cols.

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

- **Not real decode.** No model weights, no attention, no DeltaNet, no KV cache.
- **Not the megakernel.** No persistent dispatch of actual inference work.
- **Not a performance benchmark.** No wall-clock timing claims.
- **Not a new workload.** The preset runs the exact same workload as diary 0073's manual invocation.

## Known Limitations

- The preset values are hardcoded for Qwen3.5 0.8B on RX 6750 XT (82 workgroups). Different GPUs or models would need different parameters.
- The CTest gate runs the full 3072-iteration, 82-workgroup, 1024-column workload. This is larger than the `spock_barrier_probe_decode_shape` gate (384 iterations, 8 workgroups, 128 columns) but well within the 750k iteration boundary from diary 0053.

## Next Work

1. Use the preset gate as a model-width regression fixture alongside the existing decode-shape gate.
2. Move toward a bounded persistent decode skeleton with real shader work.
3. Consider additional presets if other model/GPU combinations become relevant.

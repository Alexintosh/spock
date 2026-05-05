# 0075: Persistent Decode Skeleton — First fp16/fp32 Persistent Probe

## Goal

Add a standalone persistent decode skeleton probe that exercises the proven software-global-barrier pattern with actual fp16 input/weight buffers and fp32 accumulation. This is the first probe that combines persistent dispatch with decode-shaped fp16/fp32 compute rather than uint32 synthetic payloads. It is NOT production spock-decode, does not use model weights, does not implement attention/DeltaNet/KV/LM head, and is not the megakernel.

## Background

Diaries 0047-0074 built progressively stronger evidence that the software global barrier works on this RADV stack: bare barrier (0047), two-stage mini-pipeline with coherent scratch (0048), timestamps (0049), 1M-iteration soak (0050), ALU payload (0051), memory-traffic payload (0052), long-run boundary (0053), bounded repeats (0054-0056), decode-shape modes (0071-0073), and Qwen3.5 preset (0074). All payloads used uint32 arithmetic.

The next bounded step is to validate that the same barrier pattern works when the compute is decode-shaped in precision: fp16 storage, fp32 accumulation, lane-strided dot-product workload over hidden-size columns, shared-memory reduction, and deterministic checksum/trace validation. This is the precision primitive that a real persistent decode path will use.

## Implementation

### Shader: `shaders/persistent_decode_skeleton.comp`

A GLSL compute shader with the same structure as `persistent_barrier_probe.comp` but with fp16 buffers:

- **Binding 0**: Control buffer (arrived, generation, failures, checksum) — same as barrier probe.
- **Binding 1**: Trace buffer — uint32 per workgroup per iteration.
- **Binding 2**: Scratch buffer — coherent uint32 per workgroup for inter-group communication.
- **Binding 3**: Input vector — float16_t[hidden], deterministic host-generated values.
- **Binding 4**: Weight matrix — float16_t[workgroups * hidden], deterministic host-generated values.
- **Push constants**: workgroup_count, iteration_count, hidden, _pad.

Per iteration:
1. Each workgroup performs a lane-strided dot product over `hidden` columns: `float16_t * float16_t -> float` accumulate.
2. Tree reduction of per-lane fp32 partial sums in shared memory (64 lanes, stride 32..1).
3. Lane 0 writes `scratch[group] = (group+1)*(iter+1) + floatBitsToUint(total_dot)`.
4. Global barrier #1: all groups finish writing.
5. Lane 0 cross-reads all scratch slots, writes `trace[group*iterations+iter]`.
6. Global barrier #2: prevent next iteration overwrite.

Epilogue: atomicAdd per-group checksum into global control.checksum.

Requires `GL_EXT_shader_explicit_arithmetic_types_float16`.

### App: `apps/vk_persistent_decode_skeleton.cpp`

Standalone Vulkan app following `vk_barrier_probe` structure:

- **CLI options**: `--tokens N` (default 2), `--layers N` (default 4), `--hidden N` (default 128), `--workgroups N` (default 8), `--repeats N` (default 1), `--timestamps`, `--qwen35-preset` (tokens=128, layers=24, hidden=1024, workgroups=82).
- **Deterministic fp16 data**: `input_vec[c]` cycles through exact fp16 values
  1..8; `weight_mat[g*hidden+c]` uses exact fp16 products of a small group
  scale and column scale, both also in the 1..8 range. This keeps model-width
  dot products small enough that the fp32 reduction is deterministic for the
  tested geometries.
- **Expected checksum**: computed on CPU from the same fp16-rounded values.
  Per-group trace is validated against `(iter+1)*sum_g + sum_dot` where
  `sum_g = sum(g+1)` and `sum_dot = sum(floatBitsToUint(expected_dot(g)))`.
- **Validation**: zero barrier failures, expected generation = iterations*2, arrived=0, checksum match, trace correctness, repeat correctness.
- **JSON output**: tokens, layers, hidden, workgroups, iterations, status, failures, generation, expected_generation, checksum, expected_checksum, trace_mismatches. Timestamp and repeat fields when enabled.

### fp16 conversion

Host-side fp16 conversion uses explicit IEEE 754 binary16 bit manipulation (no
ARM intrinsics). Denormals flush to zero for determinism. Values are kept small
and exactly representable in fp16 to avoid depending on overflow, denormal, or
rounding edge cases.

### CTest

Two new tests:
- `spock_persistent_decode_skeleton_help` — validates `--help` output.
- `spock_persistent_decode_skeleton_smoke` — runs `--tokens 2 --layers 4 --hidden 128 --workgroups 8` (8 iterations, deterministic, short).

## Verification

### Build

```
cmake --build build -j
```

Compiles cleanly with the new shader and app.

### CTest smoke

```
ctest --test-dir build -R spock_persistent_decode_skeleton --output-on-failure
```

Both help and smoke tests pass.

The direct smoke command with timestamps:

```
build/vk_persistent_decode_skeleton \
  --tokens 2 --layers 4 --hidden 128 --workgroups 8 --timestamps
```

passed with status ok, generation 16, expected_generation 16, checksum
2229282944, expected_checksum 2229282944, zero failures, zero trace mismatches,
gpu_dispatch_us 108.64, and per_barrier_us 6.79.

### Qwen3.5 preset repeat run

```
build/vk_persistent_decode_skeleton --qwen35-preset --timestamps --repeats 3
```

passed at tokens=128, layers=24, hidden=1024, workgroups=82, iterations=3072.
All repeats reported generation 6144, checksum 1933321216 matching expected,
zero failures, and zero trace mismatches. Timing:

| Repeat | gpu_dispatch_us | per_barrier_us |
|---|---|---|
| 1 | 45875.9 | 7.46678 |
| 2 | 36486.2 | 5.93851 |
| 3 | 36485.5 | 5.93839 |

Repeats 2-3 stabilize around 5.94 us/barrier for the fp16/fp32 skeleton at
model-width geometry. This is lower than diary 0073's uint32 model-width payload
because the synthetic data and arithmetic pattern changed; it should not be
interpreted as a production decode performance estimate.

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

- **Not real decode.** No model weights, no attention, no DeltaNet, no KV cache, no LM head.
- **Not the megakernel.** No persistent dispatch of actual inference work.
- **Not a performance benchmark.** No wall-clock timing claims beyond optional GPU timestamps.
- **Not production spock-decode.** This is a standalone probe that does not alter the production decode path.
- **Not basic inference.** This is a synthetic skeleton that validates persistent fp16/fp32 compute + barrier coordination.

## Known Limitations

- The fp16 dot-product workload is synthetic: row-indexed by workgroup, not by actual model layer. Hidden dimension is configurable but the weight data is deterministic, not from a model.
- The checksum formula relies on the synthetic data staying in a range where
  fp32 reduction is deterministic for the tested geometries. This is validated
  empirically by the smoke and model-width runs.
- fp16 denormals are flushed to zero on both host and device for determinism.
- The probe does not exercise subgroup operations, cooperative matrix, or any beyond-basic compute features.

## Next Work

1. Use this probe as a regression gate for persistent fp16/fp32 barrier stability at decode-relevant scales.
2. Consider adding payload-mode options (mixed ALU + memory, variable hidden) for stress testing.
3. Progress toward a persistent decode megakernel with real model weight loading and actual layer compute.

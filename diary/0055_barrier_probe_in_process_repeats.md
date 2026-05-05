# 0055: Barrier Probe In-Process Repeats

## Goal

Move bounded repeat testing into `vk_barrier_probe` itself. Diary 0054 used a
shell loop that launched five separate probe processes, which meant every repeat
created a fresh Vulkan device, buffers, descriptors, and pipeline. That was good
enough to test bounded dispatch duration, but it did not prove that repeated
bounded chunks are stable inside one process with one Vulkan runtime setup.

The new `--repeats N` mode addresses that gap.

## Background

The memory-payload probe has two important facts:

1. Long memory-heavy single dispatches can cross a RADV context-loss boundary.
   Diary 0053 bracketed that boundary: 750k iterations passed, while 900k and
   1M failed, and one non-timestamped 1M run printed a hard-recovery message.
2. Shorter repeated memory-payload dispatches appear stable. Diary 0054 showed
   five shell-looped 100k-iteration runs all passed with tight timing.

The next question is whether bounded chunks can be repeated without tearing down
the process. A future bounded persistent-chunk runtime would not want to rebuild
the whole Vulkan context between chunks. It would reuse the device, buffers,
descriptors, and pipeline, reset per-run state, submit the next bounded dispatch,
and validate or continue.

## Implementation Work Completed

`apps/vk_barrier_probe.cpp` now accepts:

- `--repeats N` - number of in-process repeated dispatches to run. Default is
  1.

For `--repeats 1`, the output shape remains backward-compatible with the
previous single-run JSON. The help text mentions the new flag, but normal probe
output does not gain a `repeats` field unless the user requests more than one
repeat.

For `--repeats > 1`, the probe:

1. Creates one Vulkan device.
2. Allocates one set of control, trace, scratch, input, and weight buffers.
3. Creates one descriptor layout, descriptor set, pipeline layout, shader
   module, and compute pipeline.
4. For each repeat, resets control, trace, and scratch buffers to zero.
5. Records and submits one dispatch.
6. Downloads and validates that repeat independently.
7. Emits aggregate JSON with per-repeat results.

The aggregate JSON includes:

- top-level `status`
- `failures_total`
- `trace_mismatches_total`
- `expected_generation`
- `expected_checksum`
- `repeat_results[]`

Each repeat result includes generation, arrived, checksum, trace mismatches, and
timestamp fields when `--timestamps` is enabled.

The shader was not changed.

## Verification

All verification was run locally on the RX 6750 XT (RADV NAVI22).

### Build

```
cmake --build build -j
```

Passed.

### Default output compatibility

```
build/vk_barrier_probe --workgroups 8 --iterations 10
```

Passed and kept the previous single-run JSON shape:

```
status: ok
generation: 20
expected_generation: 20
checksum: 15840
expected_checksum: 15840
trace_mismatches: 0
```

### Invalid repeat count

```
build/vk_barrier_probe --repeats 0
```

Returns JSON error:

```
status: error
message: --repeats must be greater than zero
```

### In-process bounded memory payload repeats

```
build/vk_barrier_probe --workgroups 82 --iterations 100000 --payload-cols 256 --timestamps --repeats 3
```

Passed:

```
repeats: 3
status: ok
failures_total: 0
trace_mismatches_total: 0
expected_generation: 200000
expected_checksum: 3434599968
```

Per-repeat timing:

```
repeat=1  gpu_dispatch_us=1.29166e+06  per_barrier_us=6.45828
repeat=2  gpu_dispatch_us=1.29105e+06  per_barrier_us=6.45527
repeat=3  gpu_dispatch_us=1.29155e+06  per_barrier_us=6.45774
```

Each repeat produced generation 200000, checksum 3434599968, and
trace_mismatches 0.

`spock_barrier_probe_help` also passed after the help text update.

## Interpretation

This strengthens the bounded-chunk hypothesis from diary 0054. Repeated short
memory-payload dispatches can run in one process while reusing the same Vulkan
objects, as long as per-repeat control/trace/scratch state is reset. That is
closer to the shape of a bounded persistent-chunk runtime than a shell loop.

This still does not make the project a persistent decode engine. The probe is
synthetic, the payload is uint32 memory traffic, and the host still records and
submits each bounded chunk. The result says the safe direction is becoming
clearer: avoid one unbounded memory-heavy dispatch on RADV, and design the next
GPU-owned execution path around bounded intervals.

## Known Limitations

- **Only three in-process repeats were verified in this entry.** This is not a
  full failure-rate study.
- **State reset is host-driven.** The probe uses host uploads to zero the
  control, trace, and scratch buffers between repeats.
- **Still synthetic payload.** There is no fp16/fp32 model matvec or real
  recurrent/KV state.
- **No integration with decode.** This is still a standalone Milestone 11
  viability probe.

## Next Work

1. Use `--repeats` for longer bounded-run statistics without process restart
   noise.
2. Decide whether the next implementation step should be a real decode-shaped
   bounded persistent chunk skeleton or a higher-fidelity fp16/fp32 payload
   probe.
3. Keep the long-dispatch context-loss boundary from diary 0053 as a hard
   design constraint.

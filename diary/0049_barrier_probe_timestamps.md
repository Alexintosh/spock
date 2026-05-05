# 0049: Barrier Probe Timestamp Measurement

## Goal

Add a first GPU-side timing hook to `vk_barrier_probe` so the Milestone 11
software-barrier spike can measure barrier overhead instead of only reporting
correctness. The measurement is intentionally opt-in and keeps the default JSON
shape unchanged.

This is a measurement hook for the toy probe. It is NOT persistent decode, NOT
a megakernel, and NOT a final benchmark.

## Background

Diaries 0047 and 0048 proved correctness for the bare barrier and the
two-stage coherent scratch mini-pipeline at 8, 16, 32, 64, 82, and 128
workgroups x 10000 iterations. `IMPLEMENTATION_PLAN.md` still requires
acceptable barrier overhead before the project can treat persistent dispatch as
viable. The previous probe output did not quantify that overhead.

The new timestamp path measures GPU time between two timestamp writes around
the single `vkCmdDispatch`. That scope excludes host command-buffer recording,
queue submission, fence wait, and result downloads. It is therefore a
GPU-dispatch timing sample, not host end-to-end latency.

## Implementation Work Completed

`apps/vk_barrier_probe.cpp` now accepts:

- `--timestamps` - records GPU timestamp queries around the probe dispatch when
  the selected queue reports timestamp support.

When the flag is absent, the JSON output remains unchanged. When the flag is
present, the probe adds:

- `timestamp_valid` - whether timestamp measurement succeeded.
- `gpu_dispatch_us` - elapsed GPU time between the two timestamp writes, or
  `null` when unavailable.
- `per_barrier_us` - `gpu_dispatch_us / expected_generation`, or `null` when
  unavailable.
- `barriers` - the expected generation count, equal to two barriers per
  iteration in the current mini-pipeline.

The implementation uses the existing `VulkanDevice` timestamp helpers:
`create_timestamp_query_pool`, `reset_query_pool`, `get_timestamp_results`, and
`destroy_query_pool`, plus `dev.capabilities().timestamp_period`.

## Verification

All verification was run locally on the RX 6750 XT (RADV NAVI22).

### Build

```
cmake --build build -j
```

Passed.

### Help CTest

```
ctest --test-dir build -R "spock_barrier_probe_help" --output-on-failure
```

Passed.

### Backward-compatible JSON shape

```
build/vk_barrier_probe --workgroups 8 --iterations 10
```

Passed with the original fields only: no timestamp fields are emitted unless
`--timestamps` is requested.

### 82-workgroup timestamp sample

```
build/vk_barrier_probe --workgroups 82 --iterations 10000 --timestamps
```

Result:

```
status: ok
generation: 20000
expected_generation: 20000
failures: 0
trace_mismatches: 0
timestamp_valid: true
gpu_dispatch_us: 113576
per_barrier_us: 5.67878
barriers: 20000
```

This sample keeps the two-stage coherent probe correct while giving a first
order-of-magnitude cost for the software global barrier on this driver/GPU
stack.

## Known Limitations

- **One timing sample only.** This is not a benchmark suite and should not be
  treated as a stable performance number.
- **Toy workload only.** The work between barriers is one scratch write and one
  cross-read sum, not real decode matvec work.
- **GPU timestamp scope only.** The measurement excludes host recording,
  submission, fence wait, and readback.
- **No long soak.** The 10k-iteration sample does not replace a long stability
  run under system load.
- **No occupancy proof.** The probe does not reflect register pressure,
  shared-memory usage, or residency limits of the real decode shaders.

## Next Work

1. Add a repeated timing mode or external harness to gather distributions at
   multiple workgroup and iteration counts.
2. Run a longer soak under system load with timestamps enabled.
3. Replace the trivial staged work with lightweight matvec-like work so the
   barrier cost can be interpreted alongside realistic occupancy pressure.

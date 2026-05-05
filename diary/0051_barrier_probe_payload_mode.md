# 0051: Barrier Probe Payload Mode

## Goal

Add an optional payload mode to `vk_barrier_probe` so the persistent
barrier/data-exchange probe can run deterministic per-lane ALU work before each
scratch write. This moves the probe one step beyond the minimal toy workload
from diaries 0048-0050 while preserving the default verified path.

This is still a synthetic probe. It is NOT persistent decode, NOT a megakernel,
and NOT real matvec work.

## Background

The coherent two-stage barrier probe proved that workgroups can write scratch,
globally synchronize, cross-read all scratch slots, and globally synchronize
again before overwrite. The 1M local soak showed that this pattern can run 2M
software barriers at 82 workgroups on the local RX 6750 XT stack. But the work
between barriers remained extremely small: lane 0 wrote one integer, lane 0
cross-read N integers, and the other lanes mostly waited at local barriers.

That is useful for isolating the software global barrier, but it does not model
the pressure of a persistent decode kernel. A decode kernel will have many live
values, ALU work across lanes, and more complicated scheduling pressure before
it reaches a cross-workgroup synchronization point. The new payload mode is a
controlled intermediate step: it makes all 64 lanes do deterministic uint32
work, then reduces those lane results into the workgroup's scratch value before
the existing global barrier.

## Implementation Work Completed

`vk_barrier_probe` now accepts:

- `--payload-iters N` - number of deterministic per-lane ALU iterations to run
  before each Stage A scratch write. The default is 0.

The shader push constants were extended from two uint32 values to three:

1. `workgroup_count`
2. `iteration_count`
3. `payload_iters`

When `payload_iters == 0`, the shader preserves the existing Stage A value:

```
scratch[group] = (group + 1) * (iter + 1)
```

When `payload_iters > 0`, every lane computes a deterministic uint32 payload
from its group and lane ids. The work is independent of iteration so the host
can compute the expected payload contribution once per group. The shader stores
lane results in shared memory, synchronizes locally, and lane 0 reduces the 64
lane values into `payload_sum`. Stage A then writes:

```
scratch[group] = (group + 1) * (iter + 1) + payload_sum
```

The existing global protocol is unchanged after that point:

1. Barrier after all workgroups write scratch.
2. Lane 0 of each workgroup reads all scratch slots and writes trace/checksum.
3. Barrier before the next iteration overwrites scratch.

The host mirrors the same uint32 payload formula in `apps/vk_barrier_probe.cpp`
and validates the expected trace/checksum with wraparound semantics. The JSON
output includes `payload_iters` only when a non-zero payload is requested, so
the default no-payload output shape stays compatible with previous runs.

## Verification

All verification was run locally on the RX 6750 XT (RADV NAVI22).

### Build

```
cmake --build build -j
```

Passed.

### Default shape and correctness

```
build/vk_barrier_probe --workgroups 8 --iterations 10
```

Passed with the previous default JSON fields only:

```
status: ok
generation: 20
expected_generation: 20
checksum: 15840
expected_checksum: 15840
trace_mismatches: 0
```

The 10k default sweep at workgroup counts 8, 16, 32, 64, 82, and 128 also
passed after the push-constant and shader changes, preserving the diary 0048
correctness envelope.

### Payload run

```
build/vk_barrier_probe --workgroups 82 --iterations 10000 --payload-iters 64 --timestamps
```

Passed:

```
status: ok
payload_iters: 64
generation: 20000
expected_generation: 20000
failures: 0
checksum: 2238896592
expected_checksum: 2238896592
trace_mismatches: 0
timestamp_valid: true
gpu_dispatch_us: 149650
per_barrier_us: 7.48249
barriers: 20000
```

For comparison, the trivial no-payload 82-workgroup 1M timestamped run in
diary 0050 measured about 5.17 us per barrier at large iteration count, and the
10k sample in diary 0049 measured about 5.68 us. The payload sample is slower,
as expected, because every lane now does deterministic ALU work and a shared
memory reduction before Stage A writes scratch.

## Known Limitations

- **Still not matvec-like memory access.** The payload is ALU-heavy and
  deterministic, but it does not stream weights or activations.
- **Still low-fidelity occupancy pressure.** It uses one shared array and a
  simple lane reduction; real decode kernels will have different register,
  LDS, and memory behavior.
- **No long payload soak yet.** The verified payload run is 10k iterations, not
  the 1M local soak used for the no-payload path.
- **Not integrated with inference.** This is a standalone probe used to guide
  the persistent-dispatch viability decision.

## Next Work

1. Run longer payload soaks at 82 workgroups to see whether the added lane work
   changes stability or forward progress.
2. Replace or complement the ALU payload with lightweight matvec-like memory
   traffic.
3. Measure payload mode across several `payload_iters` values to build a small
   overhead curve.

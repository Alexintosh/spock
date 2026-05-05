# 0053: Memory Payload Soak Boundary

## Goal

Bracket the long-run stability limit of the new `--payload-cols 256`
memory-traffic barrier probe at 82 workgroups. The purpose is to separate
"barrier protocol is wrong" from "long single dispatch becomes unsafe on this
driver/GPU stack when enough memory payload is added."

This is a negative-result diary entry. It does not advance persistent decode,
but it directly constrains how aggressive a future persistent dispatch can be.

## Background

Diary 0052 added a lane-strided memory payload to the two-stage coherent
barrier probe. The 10k run passed and gave a first timing sample around
6.95 us per software barrier. That proved the host and shader formulas agreed,
but it did not prove that the heavier payload can run for the same 1M
iterations as the no-payload probe from diary 0050.

This distinction matters. A persistent megakernel-like dispatch could run long
enough to trip driver timeout, context-loss, or GPU recovery behavior even when
the software barrier itself is logically correct. The project should treat that
as a first-class viability risk. A single massive dispatch that resets the GPU
is not a production path, even if shorter dispatches pass exact trace/checksum
validation.

## Verification

All runs used:

```
workgroups = 82
payload_cols = 256
```

and were executed locally on the RX 6750 XT (RADV NAVI22).

### Passing runs

The following timestamped runs passed with zero shader failures, expected
generation counts, matching checksums, and `trace_mismatches == 0`:

```
iterations=100000  generation=200000   gpu_dispatch_us=1.29125e+06  per_barrier_us=6.45623
iterations=250000  generation=500000   gpu_dispatch_us=3.21271e+06  per_barrier_us=6.42543
iterations=500000  generation=1000000  gpu_dispatch_us=6.42003e+06  per_barrier_us=6.42003
iterations=750000  generation=1500000  gpu_dispatch_us=9.61898e+06  per_barrier_us=6.41265
```

The per-barrier timing stabilizes near 6.4 us for this payload once the run is
large enough that fixed overhead is negligible.

### Failing runs

The 1M timestamped memory-payload run failed with an all-zero GPU-output
signature:

```
iterations: 1000000
generation: 0
expected_generation: 2000000
checksum: 0
trace_mismatches: 82000000
timestamp_valid: false
```

The 1M non-timestamped rerun also failed and RADV printed:

```
radv/amdgpu: The CS has been cancelled because the context is lost. This context is guilty of a hard recovery.
```

The 900k timestamped run also failed with the same all-zero output pattern:

```
iterations: 900000
generation: 0
expected_generation: 1800000
checksum: 0
trace_mismatches: 73800000
timestamp_valid: false
```

Because 750k passes and 900k/1M fail, the current practical boundary for this
payload appears to sit somewhere between 750k and 900k iterations on this local
stack. In GPU timestamp terms, the last passing run was about 9.62 seconds of
GPU dispatch time. The failing 900k run would be expected to exceed 11 seconds
if it followed the passing timing curve.

## Interpretation

This does not look like a trace/checksum formula bug. When the failing runs
return, the GPU-written control buffer is still all zeros: generation never
advances, checksum stays zero, and trace remains untouched. The non-timestamped
1M run also produced an explicit RADV context-loss/hard-recovery message. That
points to long-dispatch watchdog or GPU recovery behavior, not an in-shader
barrier mismatch.

The no-payload probe previously completed 1M iterations successfully, including
2M software barriers. The memory-payload version failing near the expected
10-12 second dispatch range means payload intensity and dispatch duration both
matter. A future persistent decode path should avoid assuming an arbitrarily
long single dispatch is acceptable on RADV. It likely needs bounded work
chunks, a watchdog-aware design, or a persistent loop structure that can yield
at safe intervals.

## Known Limitations

- **Not a precise threshold.** Only 750k, 900k, and 1M were tested around the
  boundary. The actual cutoff may vary with thermals, compositor load, driver
  state, or concurrent GPU work.
- **One payload shape.** The result applies to `payload_cols=256` and 82
  workgroups, not every memory payload size.
- **No dmesg capture.** The RADV stderr message was captured, but kernel logs
  were not archived.
- **Still synthetic.** The payload is uint32 memory traffic, not model fp16/fp32
  matvec.

## Next Work

1. Treat long single-dispatch runtime as a Milestone 11 risk, not a solved
   item.
2. Test smaller bounded chunks under repeated host submissions to see whether
   the barrier remains stable without crossing the context-loss boundary.
3. Before a real persistent decode attempt, design the scheduler around bounded
   work intervals or an explicit escape hatch rather than one unbounded GPU
   dispatch.

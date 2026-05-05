# 0050: Barrier Probe 1M-Iteration Soak

## Goal

Run the two-stage coherent persistent barrier probe past the initial 10k stress
sweep and record whether it survives a 1M-iteration local soak at the Luce
reference workgroup count of 82.

This is still a toy-probe soak. It is NOT persistent decode, NOT a megakernel,
and NOT an under-load production stability proof.

## Background

Diary 0048 proved the two-stage coherent scratch mini-pipeline for 10k
iterations across workgroup counts 8, 16, 32, 64, 82, and 128. Diary 0049 added
`--timestamps` and measured about 5.68 us per software barrier at 82 workgroups
x 10000 iterations. Milestone 11 still requires longer stress testing before
the software global barrier can be treated as a viable persistent-dispatch
primitive.

The 1M run executes 2M global barriers because the current mini-pipeline uses
two barriers per iteration: one after scratch writes and one after the
cross-read trace/checksum stage.

This matters because a software global barrier can pass short tests while still
being unusable for a persistent decode loop. The risk is not only deadlock. A
long-running dispatch can expose driver watchdog behavior, queue forward-progress
limits, timestamp-query edge cases, or subtle memory visibility failures that
show up only after many generations. The 1M soak is therefore a higher bar than
the 10k sweep, but it is still intentionally narrow: one process, one GPU, one
workgroup count, and a toy shader with very low register and LDS pressure.

## Verification

All verification was run locally on the RX 6750 XT (RADV NAVI22).

### Intermediate threshold runs

The following timestamped 82-workgroup runs passed with zero failures and zero
trace mismatches:

```
iterations=100000  generation=200000   per_barrier_us=5.20073
iterations=250000  generation=500000   per_barrier_us=5.18845
iterations=500000  generation=1000000  per_barrier_us=5.18332
iterations=750000  generation=1500000  per_barrier_us=5.17408
```

### 1M no-timestamp soak

```
build/vk_barrier_probe --workgroups 82 --iterations 1000000
```

Passed:

```
status: ok
failures: 0
generation: 2000000
expected_generation: 2000000
arrived: 0
checksum: 1631008448
expected_checksum: 1631008448
trace_mismatches: 0
```

### 1M timestamped soak

```
build/vk_barrier_probe --workgroups 82 --iterations 1000000 --timestamps
```

Passed:

```
status: ok
failures: 0
generation: 2000000
expected_generation: 2000000
arrived: 0
checksum: 1631008448
expected_checksum: 1631008448
trace_mismatches: 0
timestamp_valid: true
gpu_dispatch_us: 1.03471e+07
per_barrier_us: 5.17354
barriers: 2000000
```

An earlier timestamped 1M attempt returned a JSON failure with all GPU-written
buffers still zero. It did not reproduce: the non-timestamped and timestamped
serial reruns both passed. Treat that first failure as a cautionary transient
until repeated evidence shows otherwise.

## What This Proves

- The coherent two-stage barrier/data-exchange protocol can complete 2M
  software global barriers at 82 workgroups on this local RADV/NAVI22 stack.
- Per-barrier GPU time stabilizes near 5.17 us in this toy workload at larger
  iteration counts.
- The timestamp hook remains compatible with the long local run when rerun
  serially.

## Known Limitations

- **Not under system load.** This was a local serial soak, not a stress run while
  other GPU work competes for resources.
- **Toy workload only.** The staged work is still scratch write plus cross-read
  sum, not matvec-like decode work.
- **No occupancy proof.** The shader is too small to represent register pressure
  or residency limits of the real decode kernels.
- **Transient first failure unresolved.** One 1M timestamp attempt produced an
  all-zero failure and then did not reproduce. Future soak work should repeat
  long runs and record distributions/failure rate.

## Next Work

1. Add repeated-run support or a shell harness for long-run failure-rate
   measurement.
2. Repeat the 1M soak while the system is under GPU/CPU load.
3. Replace the trivial staged work with lightweight matvec-like work to test
   the barrier under more realistic occupancy and memory-pressure conditions.

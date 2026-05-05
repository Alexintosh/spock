# 0048: Persistent Barrier Mini-Pipeline

## Goal

Extend the `vk_barrier_probe` from a bare barrier test (diary 0047) to a
two-stage per-iteration mini-pipeline: write scratch, barrier, cross-read
scratch into trace and checksum, barrier, overwrite. This exercises the
software global barrier as a synchronization point between two phases of
real GPU work, not just arrival ordering. It is the minimal fidelity
upgrade needed before the barrier pattern can be trusted for decode-stage
coordination.

This is still a toy probe. It is NOT persistent decode, NOT a megakernel,
and NOT production code.

## Background

Diary 0047 proved that a bounded software global barrier works at up to 128
workgroups for 10k iterations on the RX 6750 XT. But that probe only tested
whether workgroups could agree on a generation counter -- it did not test
whether data written before a barrier is visible to all workgroups after the
barrier. Cross-workgroup data visibility through coherent memory is a
stronger claim than arrival synchronization alone.

The megakernel roadmap (Milestone 11 in `IMPLEMENTATION_PLAN.md`) requires
barriers between decode stages that share intermediate results. Before
building that, the barrier must be proven safe for multi-phase data
exchange, not just counter increments.

## Implementation Work Completed

### Scratch storage buffer

`apps/vk_barrier_probe.cpp` now allocates a third GPU buffer: **scratch**
(binding 2), sized to one 32-bit slot per workgroup. The host validation
after GPU completion checks the same five invariants as diary 0047
(failures, generation, arrived, checksum, trace_mismatches), now against
the two-barrier-per-iteration protocol.

### Shader: two-stage barrier pipeline

`shaders/persistent_barrier_probe.comp` now runs two global barriers per
iteration:

**Stage 1 - write:** Lane 0 of each workgroup writes
`(group + 1) * (iter + 1)` into `scratch[group]`. This is the producer
phase: each group owns one scratch slot for the current iteration.

**Barrier 1:** Global barrier. After this, every workgroup can read every
other workgroup's scratch slot.

**Stage 2 - cross-read:** Lane 0 of each workgroup reads all scratch slots,
sums them, writes the sum into its trace entry, and adds the sum to its
local checksum. The host verifies that every trace entry equals
`sum(1..workgroups) * (iter + 1)` and that the final checksum matches the
same formula accumulated by every workgroup.

**Barrier 2:** Global barrier. After this, all workgroups may safely
overwrite scratch slots in the next iteration without risking cross-read
races from the current iteration.

### Key fix: `coherent` qualifier on scratch

The initial implementation passed all invariants at 8-64 workgroups but
failed at 82x10000: `generation` and `failures` were correct (the barrier
mechanics worked), but `checksum` and `trace_mismatches` indicated stale
or partially visible scratch values. The root cause was that
`scratch.values[]` was declared as a plain `buffer` block without the
`coherent` memory qualifier.

Without `coherent`, the shader could complete the arrival/generation
protocol while still reading stale or not-yet-visible scratch data. Marking
the scratch member `coherent` makes the storage-buffer accesses participate
in the memory ordering established by the shader's buffer memory barriers,
so cross-reads after Barrier 1 observe the writes from Stage 1.

This was the only code change required. The barrier logic, trace logic, and
host validation were unchanged from the failing version.

## Verification

All verification was run locally on the RX 6750 XT (RADV NAVI22).

### Build

```
cmake --build build -j
```

Clean build, no warnings.

### Stress sweep at 10k iterations

```
build/vk_barrier_probe --workgroups 8   --iterations 10000
build/vk_barrier_probe --workgroups 16  --iterations 10000
build/vk_barrier_probe --workgroups 32  --iterations 10000
build/vk_barrier_probe --workgroups 64  --iterations 10000
build/vk_barrier_probe --workgroups 82  --iterations 10000
build/vk_barrier_probe --workgroups 128 --iterations 10000
```

All six passed:
- status: ok
- failures: 0
- generation: 20000 (expected 20000 -- two barriers per iteration x 10000)
- trace_mismatches: 0

82 workgroups matches the Luce reference block count on this GPU. The
82x10000 failure in the non-coherent version is now resolved.

### CTest

The existing `spock_barrier_probe_help` smoke test continues to pass.

## Known Limitations

- **Still a toy probe.** The "work" in each stage is trivial (write one
  uint, cross-read N uints). Real decode stages would run matvec
  operations with significantly higher register and shared-memory pressure.
- **Not persistent decode.** The probe runs once and exits; there is no
  resident scheduler loop.
- **Not a megakernel.** This is a viability experiment for the barrier +
  data-exchange pattern, not a claim of megakernel parity.
- **No real decode matvec work.** The compute stages are not representative
  of decode-stage arithmetic, memory access patterns, or occupancy
  characteristics.
- **No long soak or overhead measurement.** 10k iterations is a stress
  sweep, not a soak. Production decode would run millions of iterations.
  Barrier overhead (arrive-spin-resume latency) has not been measured.

## Next Work

1. Run a longer soak (1M+ iterations) under system load to probe for
   driver/GPU timeout or forward-progress failure.
2. Measure barrier overhead: how many microseconds does the
   arrive-spin-resume cycle cost at various workgroup counts?
3. Replace the trivial scratch write/cross-read with lightweight matvec-like
   work to test occupancy and register pressure under the barrier.
4. Document residency limits for the real decode shaders (not the toy
   probe) to determine the maximum usable workgroup count for a persistent
   decode megakernel.

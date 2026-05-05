# 0047: Persistent Barrier Probe

## Goal

Upgrade `vk_barrier_probe` from a placeholder CLI to a real Vulkan compute
probe that stress-tests a bounded software global barrier across multiple
workgroups on the RX 6750 XT. The probe must prove that a spin-based
generation counter can coordinate concurrent workgroups without host
mediation, within fixed iteration limits, and with full host-side
verification of correctness.

This is a prerequisite viability experiment for the persistent-dispatch
runtime mode described in `docs/runtime_strategy.md`. It is NOT a
megakernel, NOT persistent decode, and NOT production code.

## Background

The megakernel roadmap (Milestone 11 in `IMPLEMENTATION_PLAN.md`) requires
a cross-workgroup coordination strategy. On GPUs without hardware
barriers spanning multiple workgroups, the standard technique is a software
global barrier: each workgroup arrives at a shared location in a control
buffer, the last arrival flips a generation counter, and all workgroups
spin until they see the new generation. This probe tests whether that
pattern works reliably on the target RADV NAVI22 stack.

The previous `vk_barrier_probe` was a placeholder that loaded a trivial
shader and exited. The previous `persistent_barrier_probe.comp` was a
bring-up shader that wrote deterministic values to an output buffer. Both
have been replaced.

## Implementation Work Completed

### CLI: `vk_barrier_probe`

The probe now accepts two flags:

- `--iterations N` — number of barrier iterations per workgroup (default
  10000)
- `--workgroups N` — dispatch workgroup count (default 8)

It creates a Vulkan device, loads
`persistent_barrier_probe.comp.spv`, allocates a control buffer and a trace
buffer, dispatches the requested number of workgroups, and waits for
completion. After the GPU finishes, the host downloads both buffers and
verifies:

1. `failures == 0` — no workgroup hit the spin limit without seeing the
   expected generation.
2. `generation == iterations` — the barrier advanced through every
   iteration.
3. `arrived == 0` — no workgroup is stuck mid-barrier.
4. `checksum` matches the expected value — every workgroup wrote a correct
   trace entry per iteration.
5. `trace_mismatches == 0` — no workgroup recorded a generation that
   differed from the expected value at that iteration.

### Shader: `persistent_barrier_probe.comp`

The shader implements a bounded software global barrier with two GPU-side
buffers:

**Control buffer** (binding 0): holds `arrived`, `generation`, `failures`,
and `checksum` as atomic counters.

**Trace buffer** (binding 1): per-workgroup, per-iteration trace slot
recording a deterministic `(group + 1) * (iteration + 1)` value. Used for
host-side correctness verification.

Per iteration, each workgroup:

1. Writes its deterministic trace value into its trace slot.
2. Atomically increments `arrived`.
3. Lane 0 per workgroup checks whether it is the last arrival. If so, it
   resets `arrived` to 0, then increments `generation`. If not, it spins
   until `generation` advances beyond the value captured before arrival.
4. The spin has a fixed limit. If exceeded, `failures` is incremented and
   the workgroup returns early.

The barrier is bounded: every workgroup must arrive or the last-arrival
detection stalls. This is intentional -- the probe exists to prove that
all dispatched workgroups do arrive under the tested conditions.

### Host verification

After GPU completion, the host reads back both buffers and checks the five
invariants listed above. The probe exits with a status message and a
non-zero exit code if any invariant is violated.

## Verification

All verification was run locally on the RX 6750 XT (RADV NAVI22).

### Build

```
git diff --check
cmake --build build -j
```

Both passed.

### Smoke tests

```
build/vk_barrier_probe --workgroups 1 --iterations 10
build/vk_barrier_probe --workgroups 4 --iterations 100
```

Both passed: failures=0, generation matches iterations, arrived=0,
checksum correct, trace_mismatches=0.

### Stress sweep at 10k iterations

```
build/vk_barrier_probe --workgroups 8   --iterations 10000
build/vk_barrier_probe --workgroups 16  --iterations 10000
build/vk_barrier_probe --workgroups 32  --iterations 10000
build/vk_barrier_probe --workgroups 64  --iterations 10000
build/vk_barrier_probe --workgroups 82  --iterations 10000
build/vk_barrier_probe --workgroups 128 --iterations 10000
```

All six passed with failures=0 and trace_mismatches=0.

82 workgroups matches the Luce reference block count on this GPU. This is
notable but does not constitute persistent decode or megakernel parity --
it is only evidence that the software barrier pattern survives at that
workgroup count for 10k iterations in a toy probe.

### CTest

The existing `spock_barrier_probe_help` smoke test continues to pass.

## Known Limitations

- **No 2-layer mini-pipeline.** The probe tests a bare barrier, not a
  computation-communication pipeline. Milestone 11 requires repeating this
  with a 2-layer mini-pipeline before the barrier pattern can be trusted
  for decode.
- **No long soak under load.** The 10k-iteration sweep is a stress test,
  not a soak test. Production decode would run millions of iterations.
  Longer soak tests under system load are needed.
- **No occupancy or residency proof beyond tested workgroup counts.** The
  probe verifies correctness at specific workgroup counts but does not
  measure how many workgroups can be resident simultaneously. Occupancy
  depends on register pressure and shared-memory usage of the real decode
  shaders, not this toy probe.
- **No production timeout recovery.** The probe uses a fixed spin limit.
  Production code would need a watchdog or fallback to host-mediated
  synchronization.
- **Not full GPU offload.** The host still dispatches and waits.
- **Not persistent decode.** The probe runs once and exits; there is no
  resident scheduler loop.
- **Not the megakernel.** This is a viability experiment for one
  synchronization primitive, not a claim of megakernel parity.

## Next Work

1. Extend the probe to a 2-layer mini-pipeline (per Milestone 11 in
   `IMPLEMENTATION_PLAN.md`): two compute stages separated by the same
   barrier, with real matvec-like work in each stage.
2. Run a longer soak (1M+ iterations) under system load to probe for
   driver/GPU timeout or forward-progress failure.
3. Measure barrier overhead: how many microseconds does the
   arrive-spin-resume cycle cost at various workgroup counts?
4. Document residency limits for the real decode shaders (not the toy
   probe) to determine the maximum usable workgroup count for a persistent
   decode megakernel.

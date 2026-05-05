# 0056: In-Process 10-Repeat Memory Payload Run

## Goal

Use the new `vk_barrier_probe --repeats` mode from diary 0055 to collect a
larger in-process bounded-run sample. The previous in-process verification used
three repeats. This entry raises that to ten repeats while keeping each dispatch
well below the long-dispatch boundary found in diary 0053.

This is still a probe result, not decode integration.

## Background

The project has learned a sharp constraint from the persistent barrier spike:
the software barrier and coherent data exchange are correct across large local
runs, but memory-heavy single dispatches can cross a RADV context-loss boundary.
The current safe direction is therefore bounded GPU work chunks rather than one
unbounded memory-heavy resident dispatch.

Diary 0055 made bounded repeats measurable inside one process. That matters
because it removes process startup and Vulkan object recreation from the repeat
loop. This 10-repeat run uses the same device, buffers, descriptor set, shader,
and pipeline, resetting only the per-repeat control, trace, and scratch state.

## Verification

Command:

```
build/vk_barrier_probe --workgroups 82 --iterations 100000 --payload-cols 256 --timestamps --repeats 10
```

Result summary:

```
status: ok
repeats: 10
failures_total: 0
trace_mismatches_total: 0
expected_generation: 200000
expected_checksum: 3434599968
```

Every repeat produced:

- `status: ok`
- `generation: 200000`
- `arrived: 0`
- `checksum: 3434599968`
- `trace_mismatches: 0`
- `timestamp_valid: true`

Per-repeat timing:

```
repeat=1   gpu_dispatch_us=1.29154e+06  per_barrier_us=6.45769
repeat=2   gpu_dispatch_us=1.29215e+06  per_barrier_us=6.46075
repeat=3   gpu_dispatch_us=1.29084e+06  per_barrier_us=6.45422
repeat=4   gpu_dispatch_us=1.29071e+06  per_barrier_us=6.45356
repeat=5   gpu_dispatch_us=1.29165e+06  per_barrier_us=6.45825
repeat=6   gpu_dispatch_us=1.29116e+06  per_barrier_us=6.45578
repeat=7   gpu_dispatch_us=1.29187e+06  per_barrier_us=6.45934
repeat=8   gpu_dispatch_us=1.29086e+06  per_barrier_us=6.45431
repeat=9   gpu_dispatch_us=1.29251e+06  per_barrier_us=6.46256
repeat=10  gpu_dispatch_us=1.29127e+06  per_barrier_us=6.45634
```

Total across the run:

- 1,000,000 iterations across all repeats.
- 2,000,000 software global barriers across all repeats.
- Zero shader-reported failures.
- Zero aggregate trace mismatches.

## Interpretation

This strengthens the bounded-chunk mitigation. The failed 900k/1M
single-dispatch memory-payload runs from diary 0053 had the same order of total
barrier count as this 10-repeat run, but the shape is different: each bounded
repeat runs around 1.29 seconds of GPU dispatch time instead of pushing one
dispatch above the roughly 10-second danger zone. The bounded shape passed
cleanly.

The result does not prove production decode will be stable. It does show that
the next implementation should not chase an unbounded single-dispatch
megakernel first. A watchdog-aware bounded persistent-chunk design is better
aligned with the evidence.

## Known Limitations

- **One 10-repeat run.** This is stronger than three repeats, but still not a
  statistical failure-rate study.
- **Synthetic memory payload.** The work is uint32 lane-strided memory traffic,
  not fp16/fp32 model projection.
- **Host reset between chunks.** The probe still resets control, trace, and
  scratch buffers from the host between repeats.
- **No decode state.** There is no KV cache, DeltaNet state, token loop, or
  model weight layout in this probe.

## Next Work

1. Treat bounded chunks as the default design assumption for the next persistent
   runtime experiment.
2. Decide whether to add a decode-shaped fp16/fp32 payload probe or move to a
   bounded persistent decode skeleton.
3. Keep single-dispatch megakernel claims off the table unless later evidence
   shows memory-heavy dispatch duration can stay safely below RADV recovery
   limits.

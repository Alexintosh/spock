# 0054: Bounded Memory Payload Repeats

## Goal

Check whether repeated bounded memory-payload dispatches remain stable after the
long single-dispatch boundary found in diary 0053. The question is practical:
if a 900k-iteration or 1M-iteration memory-payload dispatch can lose the RADV
context, can the same total style of work be split into shorter verified
dispatches without immediate correctness or timing instability?

This is not a production scheduler. It is a probe result that informs the next
persistent-dispatch design choice.

## Background

Diary 0053 showed that the `--payload-cols 256` probe passes up to 750k
iterations at 82 workgroups, but fails at 900k and 1M iterations. The 1M
non-timestamped run printed a RADV context-loss/hard-recovery message. That is
an architectural warning: a Vulkan-native persistent path cannot assume one
arbitrarily long memory-heavy dispatch is safe on this driver/GPU stack.

The immediate mitigation hypothesis is bounded dispatch duration. Instead of
trying to make a single dispatch run until it crosses the watchdog/recovery
range, run shorter chunks that remain well below the observed boundary. This
does not give strict Luce-style one-dispatch megakernel parity, but it may be
the honest route if RADV rejects long memory-heavy single dispatches.

## Verification

All runs used:

```
build/vk_barrier_probe --workgroups 82 --iterations 100000 --payload-cols 256 --timestamps
```

The command was executed five times in sequence. Every run passed with:

- `status: ok`
- `failures: 0`
- `generation: 200000`
- `expected_generation: 200000`
- `trace_mismatches: 0`
- matching checksum and expected checksum
- `timestamp_valid: true`

Timing results:

```
repeat=1  gpu_dispatch_us=1.29073e+06  per_barrier_us=6.45367
repeat=2  gpu_dispatch_us=1.29086e+06  per_barrier_us=6.45432
repeat=3  gpu_dispatch_us=1.29078e+06  per_barrier_us=6.45392
repeat=4  gpu_dispatch_us=1.29069e+06  per_barrier_us=6.45344
repeat=5  gpu_dispatch_us=1.29144e+06  per_barrier_us=6.45722
```

The repeated bounded runs are stable and tightly clustered. The total number of
barriers across the five runs is 1,000,000, matching the generation count of a
single 500k-iteration two-barrier run, but split across five command
submissions. This supports the idea that the failure in diary 0053 is tied to
long single-dispatch duration or context recovery behavior, not a basic
payload checksum formula or barrier data-exchange error.

## Interpretation

This result is useful but it changes the shape of the target. A strict
single-dispatch megakernel remains risky because the memory-payload probe has
already crossed a context-loss boundary. A bounded persistent chunk strategy
looks more plausible: each chunk can keep weights, state, and intermediate
buffers on the GPU, but the host would submit bounded GPU work at safe
intervals instead of one unbounded resident dispatch.

That may still be valuable. The largest existing performance waste is not
merely "there is a host submit"; it is frequent host orchestration and
fine-grained dispatch structure. Bounded chunks could still collapse many
layer-level operations into larger GPU-owned intervals while avoiding RADV's
long-dispatch recovery path. The project should be explicit that this is
different from full persistent megakernel parity.

## Known Limitations

- **Only five repeats.** This is a small sample, not a failure-rate study.
- **Host process reinitialization remains.** The shell loop launches the probe
  process five times rather than reusing one runtime object or command path.
- **Still synthetic.** The payload is uint32 memory traffic, not real model
  fp16/fp32 matvec.
- **Not under load.** The test did not run with competing GPU workloads.

## Next Work

1. Add an in-process repeat mode or harness if we need cleaner repeated-run
   statistics without process startup noise.
2. Treat bounded persistent chunks as the leading design candidate unless later
   fp16/fp32 probes show long single dispatches can safely stay below the
   recovery boundary.
3. Move the next probe toward real decode-shaped state and weight access while
   keeping dispatch duration bounded.

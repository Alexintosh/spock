# 0072: Barrier Probe Decode-Shape 82-Workgroup Timing

## Goal

Run the decode-shaped barrier probe from diary 0071 at the full Luce reference geometry — 82 workgroups, 128 tokens x 24 layers (3072 iterations) — with GPU timestamps and in-process repeats to collect timing evidence for the persistent barrier primitive at decode-relevant scale.

This is still a synthetic barrier/payload probe. It is not real decode, not model weights, and not the megakernel.

## Background

Diary 0071 added `--tokens N --layers N` to `vk_barrier_probe`, setting `iterations = tokens * layers`. The initial CTest gate ran 16 tokens x 24 layers with 8 workgroups. Diary 0054/0056 established that bounded 82-workgroup memory-payload runs produce stable per-barrier timing around 6.45 us.

The open question from diary 0071's next-work list: does the decode-shaped geometry (128 tokens x 24 layers = 3072 iterations) at 82 workgroups produce correct results and stable timing?

## Commands

### Single run

```
build/vk_barrier_probe --tokens 128 --layers 24 --workgroups 82 \
  --payload-cols 128 --timestamps
```

Output saved to `/tmp/spock_barrier_decode_shape_128x24_wg82.json`.

### Three-repeat run

```
build/vk_barrier_probe --tokens 128 --layers 24 --workgroups 82 \
  --payload-cols 128 --timestamps --repeats 3
```

Output saved to `/tmp/spock_barrier_decode_shape_128x24_wg82_repeats3.json`.

## Results

### Single run

| Field | Value |
|---|---|
| status | ok |
| iterations | 3072 |
| workgroups | 82 |
| tokens | 128 |
| layers | 24 |
| decode_shape_iterations | 3072 |
| payload_cols | 128 |
| failures | 0 |
| generation | 6144 |
| expected_generation | 6144 |
| arrived | 0 |
| checksum | 4047549440 |
| expected_checksum | 4047549440 |
| trace_mismatches | 0 |
| timestamp_valid | true |
| gpu_dispatch_us | 48654.2 |
| per_barrier_us | 7.91898 |
| barriers | 6144 |

Correctness verified: generation, checksum, and trace all match expected values. 6144 barriers is 3072 iterations x 2 barriers per iteration (write barrier + cross-read barrier from diary 0048).

### Three-repeat run

| Repeat | gpu_dispatch_us | per_barrier_us | failures | trace_mismatches |
|---|---|---|---|---|
| 1 | 49159.7 | 8.00126 | 0 | 0 |
| 2 | 38806.2 | 6.31612 | 0 | 0 |
| 3 | 38798.7 | 6.31489 | 0 | 0 |

All three repeats passed correctness checks. Per-barrier timing:
- Repeat 1: ~8.00 us (higher)
- Repeats 2–3: ~6.31–6.32 us (stable, close together)

## Interpretation

**Correctness:** The 82-workgroup, 128-token x 24-layer decode geometry passed all correctness checks in both single and repeated runs. This is the first decode-shaped run at the full Luce reference workgroup count.

**Warmup effect:** Repeat 1 is measurably slower (~8.0 us/barrier) than repeats 2–3 (~6.3 us/barrier). This is consistent with a warmup/clock ramp/cache effect: the first dispatch in a repeated sequence pays some one-time cost. The same pattern appeared in earlier probe runs (diaries 0054, 0056) where early repeats were slightly slower before stabilizing.

**Stable timing:** Repeats 2–3 converge to ~6.31 us per barrier. This is slightly lower than the ~6.45 us/barrier measured in diary 0054/0056 (100k iterations, 256 payload cols) and well within the range of expected variation given the different payload-column count (128 vs 256) and iteration count (3072 vs 100000). The direction is plausible: less memory traffic per iteration should produce slightly faster barrier rounds.

**What this improves:** This run increases confidence that 82-workgroup persistent barrier coordination holds at decode-relevant iteration scales (128 tokens x 24 layers). It also strengthens the bounded-chunk design direction from diary 0054/0056: the barrier primitive appears stable and well-bounded at this geometry.

**What this does not prove:**
- Not proof of full persistent decode correctness. No model weights, no attention, no DeltaNet recurrence, no KV cache.
- Not a performance benchmark for real decode. The payload is deterministic uint32 memory traffic, not fp16/fp32 matvec.
- Not an under-load soak. No concurrent GPU workload, no system stress.
- Not an occupancy proof for real decode shaders. The probe workgroup has a different register/shared-memory footprint than production decode shaders would.
- Not a timing benchmark. The sample size is small (one single run + one 3-repeat run).

## Verification

### Single run

```
build/vk_barrier_probe --tokens 128 --layers 24 --workgroups 82 \
  --payload-cols 128 --timestamps
```

Status ok, generation 6144 matches expected_generation 6144, checksum 4047549440 matches
expected_checksum 4047549440, zero failures, zero trace mismatches, timestamp_valid true.

### Three-repeat run

```
build/vk_barrier_probe --tokens 128 --layers 24 --workgroups 82 \
  --payload-cols 128 --timestamps --repeats 3
```

All three repeats report status ok, zero failures, zero trace mismatches.
Per-repeat timing recorded above in Results.

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

## Known Limitations

- The payload column count (128) is arbitrary and does not correspond to a real model dimension.
- 3072 iterations is moderate — well below the diary 0050 1M-iteration soak and the diary 0053 boundary work. It represents only a 128-token decode, not sustained generation.
- The warmup effect in repeat 1 is observed but not systematically characterized. The cause (shader cache, GPU clock ramp, memory controller warmup) is not isolated.
- No concurrent GPU load. Real decode would share the GPU with display and potentially other work.
- This probe uses 82 workgroups but the workgroup size and resource usage differ from what production decode shaders would require.

## Next Work

1. Consider whether a longer decode-shaped soak (e.g., 512 or 1024 tokens x 24 layers) at 82 workgroups would be informative, or whether the diary 0050/0056 evidence already covers the iteration-count question.
2. Use the decode-shaped probe as a regression gate alongside the existing full-fast decode CTest (diary 0057) to catch persistent-barrier regressions.
3. Move toward a bounded persistent decode skeleton that exercises the barrier primitive with real decode shader work (fp16/fp32 matvec, actual model dimensions) rather than synthetic payloads.
4. Keep bounded persistent chunks as the default design assumption per the diary 0054/0056 evidence. Do not pursue unbounded single-dispatch megakernel designs without new evidence that memory-heavy dispatch duration can stay safely below RADV recovery limits.

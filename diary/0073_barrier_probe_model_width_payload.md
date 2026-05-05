# 0073: Barrier Probe Model-Width Decode-Shaped Payload

## Goal

Run the decode-shaped barrier probe at the same geometry as diary 0072 (82 workgroups, 128 tokens x 24 layers) but with `--payload-cols 1024`, matching the Qwen3.5 0.8B hidden_size. This tests whether the persistent barrier/payload primitive stays correct and stable when each barrier round moves model-width columns of memory traffic rather than the 128-column synthetic baseline from diary 0072.

This is still a synthetic barrier/payload probe using uint32 memory traffic. It is not real decode, not model weights, not fp16/fp32 matvec, and not the megakernel.

## Background

Diary 0072 ran 128 tokens x 24 layers at 82 workgroups with `--payload-cols 128`. Repeats 2-3 stabilized at ~6.31 us/barrier. The Qwen3.5 0.8B model has hidden_size=1024 and layer_count=24. A payload_cols=1024 run exercises memory traffic at the same column count a real decode step would touch per projection, though the data is still synthetic uint32, not fp16 weights.

The open question: does model-width memory traffic (8x the column count) remain correct under the persistent barrier, and how does per-barrier cost change?

## Command

```
build/vk_barrier_probe --tokens 128 --layers 24 --workgroups 82 \
  --payload-cols 1024 --timestamps --repeats 3 \
  > /tmp/spock_barrier_decode_shape_128x24_wg82_cols1024_repeats3.json
```

## Results

| Field | Value |
|---|---|
| status | ok |
| iterations | 3072 |
| workgroups | 82 |
| tokens | 128 |
| layers | 24 |
| decode_shape_iterations | 3072 |
| payload_cols | 1024 |
| expected_generation/barriers | 6144 |

### Per-repeat detail

| Repeat | status | failures | trace_mismatches | gpu_dispatch_us | per_barrier_us |
|---|---|---|---|---|---|
| 1 | ok | 0 | 0 | 52971.2 | 8.62162 |
| 2 | ok | 0 | 0 | 42769.6 | 6.96120 |
| 3 | ok | 0 | 0 | 42760.7 | 6.95975 |

All three repeats passed correctness checks: zero failures, zero trace mismatches.

## Comparison to Diary 0072

Same geometry (82 wg, 128 tokens x 24 layers, 3072 iterations), payload columns differ:

| Metric | 0072 (cols=128) | 0073 (cols=1024) | Delta |
|---|---|---|---|
| Repeat 1 per_barrier_us | 8.00 | 8.62 | +0.62 (+7.8%) |
| Repeat 2 per_barrier_us | 6.316 | 6.961 | +0.645 (+10.2%) |
| Repeat 3 per_barrier_us | 6.315 | 6.960 | +0.645 (+10.2%) |

8x the column count produces roughly 10% higher per-barrier cost in the stable repeats. This is modest: an 8x increase in memory traffic per barrier round adds only ~0.65 us/barrier.

## Interpretation

**Correctness:** Model-width (1024-column) payload at 82 workgroups and decode-shaped iteration count produces zero failures and zero trace mismatches across all three repeats. The persistent barrier primitive remains correct at this memory-traffic scale.

**Cost scaling:** Per-barrier cost rises from ~6.31 us to ~6.96 us when moving from 128 to 1024 payload columns. The increase is modest and consistent with the probe spending slightly more time per barrier round on the larger memory footprint. The scaling is sublinear in column count, suggesting the barrier synchronization overhead dominates over the memory-traffic component.

**Warmup pattern:** Repeat 1 is ~24% slower than repeats 2-3, consistent with the warmup/clock-ramp effect seen in all prior repeat runs (diaries 0054, 0056, 0072).

**Viability implication:** Model-width synthetic traffic at 82 workgroups costs ~6.96 us/barrier in steady state. For a 128-token decode at 24 layers (6144 barriers), this is ~42.8 ms of barrier overhead. This is bounded and does not grow unbounded with token count in a chunked-persistent design (each chunk has its own barrier budget). It supports the bounded persistent chunk design direction but remains a lower bound: real fp16/fp32 decode shaders would have higher register and shared-memory pressure, potentially reducing occupancy and increasing barrier cost.

**What this does not prove:**
- Not proof of real decode correctness. No model weights, no attention, no DeltaNet recurrence, no KV cache.
- Not a performance benchmark for real decode. The payload is deterministic uint32 memory traffic, not fp16/fp32 matvec. Real decode shaders would have different register/shared-memory footprints and occupancy.
- Not an under-load soak. No concurrent GPU workload, no system stress.
- Not an occupancy proof for production decode shaders. The probe workgroup has a different resource profile than production shaders would.
- Not a timing benchmark. The sample size is small (one 3-repeat run).
- Not megakernel parity.

## Verification

```
build/vk_barrier_probe --tokens 128 --layers 24 --workgroups 82 \
  --payload-cols 1024 --timestamps --repeats 3 \
  > /tmp/spock_barrier_decode_shape_128x24_wg82_cols1024_repeats3.json
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

- The payload data is synthetic uint32, not fp16/fp32 weights or activations. Memory traffic patterns differ from real decode.
- 1024 columns matches hidden_size but real decode touches multiple buffers per layer (Q/K/V projections, DeltaNet projections, MLP projections). The total memory traffic per barrier round in production would be higher.
- 3072 iterations (128 tokens x 24 layers) is moderate. It represents a single decode chunk, not sustained generation across many chunks.
- The warmup effect in repeat 1 is observed but not systematically characterized.
- No concurrent GPU load. Real decode would share the GPU with display and potentially other work.
- The probe workgroup resource footprint differs from production decode shaders.

## Next Work

1. Use the model-width probe (cols=1024) as an additional regression gate alongside the existing decode-shape gate to catch persistent-barrier regressions at model-relevant memory traffic.
2. Move toward a bounded persistent decode skeleton that exercises the barrier primitive with real decode shader work (fp16/fp32 matvec, actual model dimensions) rather than synthetic payloads.
3. Consider whether a longer model-width soak (more tokens or more repeats) would add confidence beyond what diaries 0050/0053/0056 already provide for iteration stability.
4. Keep bounded persistent chunks as the default design assumption. Model-width synthetic traffic at ~6.96 us/barrier is viable but real decode shader occupancy remains the open question.

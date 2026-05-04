# 0040: GPU Timestamp Decode Instrumentation — Opt-In Measurement Gate

## Goal

Add Vulkan timestamp query support to the decode pipeline so that GPU-side
command-buffer execution time is measurable without altering inference behavior.
This is a measurement instrument, not a performance optimization and not full
GPU offload.

`SPOCK_GPU_TIMESTAMPS=1` enables Vulkan timestamp queries on the decode command
buffer. When active, `spock-decode` JSON output gains `gpu_decode_us` and
`per_token_gpu_us` fields alongside the always-present `prefill_ms`, `decode_ms`,
and `per_token_ms` wall-clock fields. The gate is default-off; when disabled,
no timestamp queries are allocated or recorded, and the JSON output contains
only host-side timing.

## Implementation Work Completed

### Vulkan timestamp query support in VulkanDevice

Vulkan timestamp query pool creation and query support are now part of the
device capability surface. The device reports `timestampPeriod` and whether
timestamp queries are supported. When `SPOCK_GPU_TIMESTAMPS=1` is set, the
session allocates a `VkQueryPool` with `VK_QUERY_TYPE_TIMESTAMP` at creation
time and resets it before each decode step.

### JSON output extensions

The `spock-decode` JSON output always includes:

- `prefill_ms` — host-measured prefill wall-clock time
- `decode_ms` — host-measured decode loop wall-clock time
- `per_token_ms` — per-token host wall-clock time array

When `SPOCK_GPU_TIMESTAMPS=1` and timestamps are recorded for a step, the JSON
additionally includes:

- `gpu_decode_us` — total GPU decode command buffer execution time in
  microseconds (sum of per-step GPU times)
- `per_token_gpu_us` — per-token GPU execution time array in microseconds

These fields are absent when the gate is disabled or when timestamp recording
is skipped for a particular step (e.g., prefill steps, steps where queries
were not recorded).

### What this measures

The timestamp queries bracket the decode command buffer for steps where
single-submit is eligible. This includes the full decode command buffer
(embedding + 28 layers + final norm + LM head + argmax) on single-submit
steps. It also covers the `skip_layers` LM-head-only first decode step after
chunk prefill, which bypasses the attention/DeltaNet layers and runs only
the final norm + LM head + argmax.

The GPU timestamp represents command-buffer execution time on the device,
which excludes host-side orchestration, fence waits, and CPU-side processing.
It is strictly a measurement instrument for understanding the GPU-side cost
of decode steps.

### What this is not

- **Not a performance optimization.** No dispatches are added, removed, or
  reordered. Inference output is bitwise identical whether the gate is active
  or not.
- **Not full GPU offload.** The host still orchestrates every step, submits
  command buffers, waits on fences, and reads back results.
- **Not persistent dispatch or the megakernel.** Each token still gets its
  own command buffer submission on the single-submit path.
- **Not a substitute for wall-clock timing.** GPU timestamps measure device
  execution only; they do not capture host overhead, driver submission cost,
  or queue scheduling latency. Both `per_token_ms` (host) and
  `per_token_gpu_us` (device) should be reported together when comparing
  paths.

## Verification

All commands were run locally on the target Vulkan/RADV path.

### Whitespace and build

```sh
git diff --check
```

No whitespace errors.

```sh
cmake --build build -j
```

Passed.

### Direct decode with timestamps

```sh
SPOCK_GPU_TIMESTAMPS=1 \
SPOCK_GPU_SINGLE_SUBMIT=1 \
SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1 \
SPOCK_GPU_MERGED_DELTANET=1 \
  python3 tests/run_vk_decode_parity.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --reference tests/data/reference_tokens.jsonl \
  --ids short_correctness_001 \
  --max-new-tokens 1
```

Locally observed output (primary evidence):

- `generated_tokens`: [271]
- `gpu_decode_us`: 403422
- `per_token_gpu_us`: [403422]
- `status`: ok

The timestamp values confirm the query pool is allocated, timestamps are
recorded for the decode command buffer, and the results are retrieved and
reported correctly in the JSON output.

### Parity with timestamps active

```sh
SPOCK_GPU_TIMESTAMPS=1 \
SPOCK_GPU_SINGLE_SUBMIT=1 \
SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1 \
SPOCK_GPU_MERGED_DELTANET=1 \
  python3 tests/run_vk_decode_parity.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --reference tests/data/reference_tokens.jsonl \
  --ids short_correctness_001 \
  --max-new-tokens 16
```

```json
{"status":"ok","checked":1,"failures":[]}
```

Parity is preserved with timestamps active — confirming the gate does not
alter inference output.

OMP also reported that baseline short parity and timestamp parity passed;
only the locally observed numbers above are included as primary evidence.

## Known Limitations

- **Default-off.** `SPOCK_GPU_TIMESTAMPS=1` must be explicitly set. No
  timestamp queries are allocated or executed without it.
- **Single-submit dependency.** Timestamp queries bracket the decode command
  buffer. On the single-submit path (`SPOCK_GPU_SINGLE_SUBMIT=1`), the entire
  decode step is one command buffer, giving a clean GPU-time measurement. On
  the per-layer-submit path, timestamps would bracket only individual command
  buffers, not the full decode step; the current implementation targets
  single-submit-eligible steps.
- **Measurement overhead.** Timestamp queries add a small amount of command
  buffer recording and query-pool management. This does not change inference
  output but may affect wall-clock timing slightly. Do not compare
  `per_token_ms` with timestamps on vs. timestamps off and attribute the
  difference to a speedup or slowdown from the instrumentation itself.
- **No per-dispatch breakdown.** Timestamps bracket the full command buffer,
  not individual dispatches within it. For per-layer or per-dispatch timing,
  a different instrumentation strategy would be needed.
- **GPU timestamp resolution.** The reported microsecond values are converted
  using `timestampPeriod` from the physical device properties. The resolution
  and accuracy depend on the driver and hardware.

## Next Work

- Use `gpu_decode_us` measurements to characterize the GPU-side cost of
  decode under single-submit vs. other paths.
- Consider per-dispatch timestamp regions if finer-grained measurement is
  needed for specific layers or operations.
- Expand measurement to prefill steps if needed for prefill/decode
  cost comparison.

# 0052: Barrier Probe Memory-Traffic Payload

## Goal

Extend `vk_barrier_probe` with an optional memory-traffic payload so the
persistent barrier probe can exercise lane-strided buffer reads and a
workgroup reduction before each cross-workgroup synchronization point.

This is a closer synthetic proxy for matvec pressure than the ALU-only payload
from diary 0051, but it is still NOT real decode matvec, NOT persistent decode,
and NOT a megakernel.

## Background

The previous payload mode made every lane perform deterministic uint32 ALU work
and reduce those lane results into the Stage A scratch value. That was useful
because it prevented the probe from being almost entirely lane-0 work, but it
still did not exercise the memory behavior that matters for decode. Real decode
projection kernels stream weights and activations, perform reductions, and then
feed intermediate values into later stages. A persistent dispatch candidate must
keep the software global barrier correct while workgroups perform meaningful
memory traffic before the barrier.

The new mode adds two readonly storage buffers: an input vector and a
workgroup-major weight matrix. Each workgroup computes a deterministic uint32
dot-like payload by having all 64 lanes read columns in a lane-strided pattern,
accumulate partial sums, write those partials to shared memory, and let lane 0
reduce them. That reduced value is added to the existing Stage A scratch value.
The global barrier and cross-read protocol remain unchanged.

## Implementation Work Completed

`vk_barrier_probe` now accepts:

- `--payload-cols N` - enables lane-strided memory payload over N columns.

When `payload_cols == 0`, the host still binds one-element dummy input/weight
buffers because the shader declares bindings 3 and 4 unconditionally. This
keeps the Vulkan descriptor contract explicit and avoids relying on the driver
to tolerate omitted inactive descriptors. The default JSON output remains
unchanged when no payload flags are used.

When `payload_cols > 0`, the host allocates and uploads:

- `input_vec[payload_cols]`
- `weight_mat[workgroups * payload_cols]`

Both are filled with deterministic uint32 hash functions. The shader computes:

```
dot_payload[group] = sum_c input_vec[c] * weight_mat[group, c]
```

with uint32 wraparound. Each lane handles columns `lane, lane + 64, ...`, then
the workgroup reduces the lane partials through shared memory. Stage A writes:

```
scratch[group] =
    (group + 1) * (iter + 1)
  + optional_alu_payload[group]
  + optional_memory_payload[group]
```

The host mirrors the same formulas and includes the payload contribution in the
expected trace and checksum. This preserves the existing correctness invariant:
every workgroup should cross-read the same sum of all scratch values at every
iteration.

## Verification

All verification was run locally on the RX 6750 XT (RADV NAVI22).

### Build

```
cmake --build build -j
```

Passed.

### Default regression

```
build/vk_barrier_probe --workgroups 8 --iterations 10
```

Passed with the old no-payload JSON shape:

```
status: ok
generation: 20
expected_generation: 20
checksum: 15840
expected_checksum: 15840
trace_mismatches: 0
```

The full no-payload 10k sweep also passed at workgroup counts 8, 16, 32, 64,
82, and 128, preserving the diary 0048 correctness envelope after adding the
new descriptor bindings and push constant.

### Memory payload

```
build/vk_barrier_probe --workgroups 82 --iterations 10000 --payload-cols 256 --timestamps
```

Passed:

```
payload_cols: 256
status: ok
generation: 20000
expected_generation: 20000
failures: 0
checksum: 313802960
expected_checksum: 313802960
trace_mismatches: 0
timestamp_valid: true
gpu_dispatch_us: 139090
per_barrier_us: 6.95452
barriers: 20000
```

### Combined ALU + memory payload

```
build/vk_barrier_probe --workgroups 82 --iterations 10000 --payload-iters 64 --payload-cols 256 --timestamps
```

Passed:

```
payload_iters: 64
payload_cols: 256
status: ok
generation: 20000
expected_generation: 20000
failures: 0
checksum: 3206214256
expected_checksum: 3206214256
trace_mismatches: 0
timestamp_valid: true
gpu_dispatch_us: 144735
per_barrier_us: 7.23677
barriers: 20000
```

## Known Limitations

- **Integer proxy, not fp16/fp32 matvec.** The payload is uint32 arithmetic and
  deterministic hashing, not model weights, activations, or floating-point
  accumulation.
- **Small memory footprint.** `payload_cols=256` is enough to exercise
  lane-strided buffer reads, but it is far smaller than real projection
  matrices.
- **Still standalone.** The probe does not share code or descriptors with the
  inference decode kernels.
- **No long payload soak yet.** The memory payload was verified at 10k
  iterations, not 1M or under load.

## Next Work

1. Run longer memory-payload soaks at 82 workgroups.
2. Sweep `payload_cols` to understand how added memory traffic changes
   per-barrier time and stability.
3. Decide whether the next probe should use fp16/fp32 payloads or move directly
   toward a persistent decode skeleton with real activation/weight layout.

# 0014: RADV Device Selection and Prefill-Sensitive Hardware Gate

## Goal

Re-align the active branch with the actual project scope:

- the target device is the RX 6750 XT on RADV
- correctness must be measured on that hardware path
- capability reporting must describe the same device the runtime actually uses

Also add a fast executable gate for the remaining prompt-prefill failures so
future fixes do not depend on a slow full-corpus sweep.

## Context

The repo had already moved back onto the real GPU. Local capability queries now
report:

- `AMD Radeon RX 6750 XT (RADV NAVI22)`
- subgroup size `64`
- shared memory `65536`

But the runtime and the capability probe were still selecting devices through
different code paths:

- `spock-decode` used `VulkanDevice::initialize()`
- `vk-capabilities` used `VulkanContext::query_default_device()`

That was a metadata integrity bug. On a machine exposing both RADV and
`llvmpipe`, the runtime could run on one device while the probe reported
another.

## Implementation

### 1. Deterministic runtime device selection

Updated `src/runtime/vk_device.cpp` so runtime selection no longer means
"first compute-capable device wins."

The selector now:

- filters to compute-capable devices
- strongly prefers discrete GPUs over CPU Vulkan
- prefers AMD vendor `0x1002`
- gives explicit weight to the known target strings `6750 XT` and `NAVI22`
- zeroes the score for `llvmpipe`

This keeps the branch aligned with the original project target without
pretending it is a generic multi-device scheduler.

### 2. Unified capability reporting

Updated `src/runtime/vk_context.*` so `vk-capabilities` now goes through the
same `VulkanDevice` path as `spock-decode`.

That means:

- capability JSON
- runtime execution
- benchmark metadata

all describe the same selected Vulkan device.

### 3. Fast prefill-sensitive executable gate

Extended `tests/run_vk_decode_parity.py` with `--ids`, then added a focused
CTest case:

- `spock_vk_decode_prefill_handoff_mismatch`

It checks:

- `mixed_correctness_023`
- `pp520_046`

for `1` generated token and currently uses `--expect-mismatch`.

This is intentionally temporary. Right now it proves the bug is still present
on hardware. Once the prefill refactor lands, the test should be inverted into
a positive parity gate.

## Hardware Validation

### Capability path

`./build/vk-capabilities` now reports:

```json
{
  "device_name": "AMD Radeon RX 6750 XT (RADV NAVI22)",
  "subgroup_size": 64,
  "max_shared_memory_bytes": 65536,
  "max_workgroup_invocations": 1024
}
```

### Current executable parity state on hardware

- 8-prompt / 16-token executable parity gate passes on RADV.
- The known prefill-sensitive failures reproduce on RADV:
  - `mixed_correctness_023`
  - `mixed_correctness_025`
  - `mixed_correctness_026`
  - `mixed_correctness_027`
  - `pp520_046`

That confirms the remaining gap is still the prompt-prefill contract, not a
software-Vulkan artifact.

## Verification

The checkpoint was validated with:

```sh
cmake --build build -j
./build/vk-capabilities
ctest --test-dir build --output-on-failure -R "spock_capabilities|spock_deltanet_chunk_unit|spock_vk_decode_prefill_handoff_mismatch"
python3 tests/run_vk_decode_parity.py --decode build/spock-decode --repack-dir artifacts/spock-text-repack-qwen35-0p8b --reference tests/data/reference_tokens.jsonl --limit 8 --max-new-tokens 16
```

The full 48-prompt sweep was not promoted to the default test path because it
is still too slow for rapid iteration and the remaining failures are already
captured by the targeted reproducer set.

## Why This Matters

This checkpoint does not solve the remaining `P0` failures. It does something
more basic and necessary:

1. It makes the hardware target explicit in code.
2. It makes benchmark/capability metadata honest.
3. It creates a fast executable reproducer for the still-broken prefill path.

That is the right setup before the larger `vk_decode.cpp` refactor that will
separate prompt prefill from recurrent decode and make the runtime reusable for
future benchmarking work.

## Next Work

1. Extract a reusable decode/session context from `vk_decode.cpp`.
2. Split prompt prefill from decode generation inside the same runtime.
3. Use the existing host-side DeltaNet chunk primitive only as the narrow
   prefill bridge, not as a second end-to-end runtime.
4. Flip `spock_vk_decode_prefill_handoff_mismatch` into a positive parity test.

## Scope Boundary

This entry is a tactical checkpoint inside the larger project scope. The
governing objective remains the Vulkan-native RX 6750 XT megakernel roadmap in
`IMPLEMENTATION_PLAN.md`, not a correctness-only hybrid runtime. The work here
exists to keep the hardware target honest and to make the next refactor
measurable.

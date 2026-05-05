# 0059: Chunked Scaffold CTest Gate

## Goal

Add a dedicated regression test proving that the reserved chunked-decode gates
from diary 0058 are currently inert. The runtime now parses
`SPOCK_GPU_CHUNKED_DECODE` and `SPOCK_GPU_DECODE_CHUNK_SIZE`, but
`chunked_decode_enabled` is force-disabled. A CTest should protect that
contract until the real implementation is ready.

This is a test-only entry. It does not implement chunked decode.

## Background

The chunked-decode scaffold exists because the next real inference-path change
will touch the most delicate part of `DecodeSession::decode()`: host-side
command-buffer recording around token-to-token dependencies. The project needs
the env names reserved, but it must not silently expose a half-working feature.

The full fast gate test from diary 0057 protects the current strongest fast path
without the chunked variables. This new test uses the same fast path and adds:

```
SPOCK_GPU_CHUNKED_DECODE=1
SPOCK_GPU_DECODE_CHUNK_SIZE=4
```

Because the scaffold is force-disabled, the result should remain identical to
the current full fast gate behavior.

This kind of negative assertion is useful during staged implementation. Once an
environment variable exists, developers and scripts may start setting it. If the
variable accidentally changes behavior before the feature is complete, failures
can look like model drift or Vulkan synchronization bugs. The dedicated CTest
makes the intended current contract explicit: the names are reserved, parsed,
and harmless. A future commit that enables the feature must update this test or
add a sibling test that checks the new active behavior.

## Implementation Work Completed

`CMakeLists.txt` now registers:

```
spock_vk_decode_chunked_scaffold_fast_gate_short
```

The command matches `spock_vk_decode_full_fast_gate_short`:

```
tests/run_vk_decode_parity.py
  --decode spock-decode
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b
  --reference tests/data/reference_tokens.jsonl
  --ids short_correctness_001
  --max-new-tokens 4
```

The environment includes the full fast-gate stack plus the two chunked scaffold
variables. The test comment explicitly says the scaffold is parse-only and
force-disabled.

## Verification

All verification was run locally on the RX 6750 XT (RADV NAVI22).

### Build

```
cmake --build build -j
```

Passed.

### New CTest

```
ctest --test-dir build -R "spock_vk_decode_chunked_scaffold_fast_gate_short" --output-on-failure
```

Passed:

```
1/1 Test #17: spock_vk_decode_chunked_scaffold_fast_gate_short ...   Passed    4.87 sec
```

## Interpretation

This test is intentionally narrow. It exists so future commits can safely turn
`chunked_decode_enabled` from `false` into a real condition and immediately see
whether behavior changes under the same env stack. Until that happens, the test
guards against accidental activation or partial code paths that would make the
scaffold observable.

The test also documents the expected chunk-boundary value for the first real
implementation attempt: chunk size 4. That size is small enough to stay far
below the RADV long-dispatch boundary found in the barrier probe, while still
large enough to test multi-token command-buffer amortization once enabled.

The test is paired with `spock_vk_decode_full_fast_gate_short` rather than
replacing it. Keeping both tests is intentional: one says the current full fast
stack works without chunked variables; the other says adding the scaffold
variables is inert. When chunked decode becomes real, the second test can become
the first active chunk-size-4 parity gate, while the first remains the baseline
for one-submit-per-token behavior.

## Known Limitations

- **No feature coverage yet.** The test proves inertness, not chunked decode.
- **Short fixture only.** Four generated tokens is enough to make chunk size 4
  meaningful later, but not enough for broad correctness.
- **No performance assertion.** The scaffold should not change timing in any
  meaningful way.

## Next Work

1. Refactor the single-token single-submit recording path so a chunked path can
   call it repeatedly inside one command buffer.
2. Enable `SPOCK_GPU_CHUNKED_DECODE` first for chunk size 1 equivalence.
3. Extend this CTest or add a sibling test once chunk size 4 is functional.

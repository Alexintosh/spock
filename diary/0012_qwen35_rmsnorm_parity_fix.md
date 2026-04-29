# 0012: Qwen3.5 RMSNorm Parity Fix

## Goal

Turn the token-220 failure into an executable correctness bug, identify the
first divergent operation, and fix the Vulkan decode path until it matches the
trusted HF/repacked reference for at least the first frozen prompt.

## Context

After the attention and DeltaNet paths were wired, `spock-decode` ran end to
end but generated token `220` where the frozen reference expected `271`. The
existing `spock_p0_parity` test only validated the structure of
`reference_tokens.jsonl`; it did not execute Vulkan or compare generated tokens.

This made the repo look healthier than it was. The first step was to add an
executable parity harness that runs `spock-decode` on prompt token IDs and
compares generated token IDs against the frozen reference.

## Root Cause

The first single-token test `[7734]` removed prefill, KV-cache, and recurrent
state complexity. HF with repacked weights generated token `264`, while Vulkan
generated token `220`.

Layer-by-layer comparison found immediate divergence after layer-0 input
RMSNorm:

| Value | HF fp16 | Vulkan before fix |
|-------|---------|-------------------|
| input norm[0] | `0.7788` | `0.3894` |
| input norm[1] | `-0.6738` | `-0.1534` |
| input norm[2] | `-1.2949` | `-0.4514` |

The artifact weights were correct. The shader formula was wrong.

Qwen3.5 uses:

```text
output = rms_norm(x) * (1 + weight)
```

The Vulkan shaders used:

```text
output = rms_norm(x) * weight
```

That is correct for some RMSNorm variants, but not for Qwen3.5's regular
RMSNorm. The gated DeltaNet norm remains different: HF applies the gated norm
with the weight directly, so that shader was not changed.

## Implementation

Updated:

| File | Change |
|------|--------|
| `shaders/rms_norm.comp` | Multiply by `1 + weight` |
| `shaders/rms_norm_per_head.comp` | Multiply attention Q/K norm by `1 + weight` |
| `tests/run_vk_decode_parity.py` | New executable Vulkan-vs-reference parity harness |
| `CMakeLists.txt` | Added `spock_vk_decode_reference_parity` CTest gate |
| `apps/spock-decode.cpp` | Reject non-token token files and fail honestly for unimplemented text tokenization |
| `tools/reference_decode.py` | Force fp16 model dtype for repacked weights and add `--sequential-prefill` |
| `src/runtime/vk_decode.cpp` | Add debug top-5 logit reporting |

## Verification

After the fix:

- Single token `[7734]` generates `264`, matching HF/repacked reference.
- First frozen prompt generates the full 16-token reference sequence:

```text
[271, 248068, 271, 248069, 271, 89454, 4384, 6813, 513, 16099, 1521, 781, 3300, 264, 14294, 11]
```

- `ctest --test-dir build --output-on-failure` passes `12/12`.

## Follow-Up Fix: RoPE Pairing

After the RMSNorm fix, the first two prompts passed but
`short_correctness_003` still flipped at generated token index 5:

| Source | Token | Logit |
|--------|-------|-------|
| HF reference | `12` | `19.375` |
| HF runner-up | `16` | `19.34375` |
| Vulkan winner | `16` | `19.3438` |
| Vulkan runner-up | `12` | `19.3281` |

That small margin made the failure look like generic fp16 drift, but the next
inspection found a concrete architectural mismatch in RoPE.

HF Qwen3.5 applies `rotate_half` over the rotary slice:

```text
(0, rotary_dim/2), (1, rotary_dim/2 + 1), ...
```

The Vulkan `rope_apply.comp` shader rotated adjacent pairs:

```text
(0, 1), (2, 3), ...
```

This does not affect position 0, so the single-token and early checks could
pass while later prompt/decode positions accumulated attention error. Updating
the shader to pair split halves cleared the first eight-prompt parity sweep.

## Current Verification

- `ctest --test-dir build --output-on-failure` passes.
- `spock_vk_decode_reference_parity` now checks the first eight frozen prompts,
  16 generated tokens each.
- Manual sweep:

```text
python3 tests/run_vk_decode_parity.py --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --reference tests/data/reference_tokens.jsonl \
  --limit 8 --max-new-tokens 16
```

returns:

```json
{"status": "ok", "checked": 8, "failures": []}
```

## Next Work

1. Run and fix the full 48-prompt parity sweep.
2. Increase `spock_vk_decode_reference_parity` only as runtime stays practical.
3. Move on to honest RX 6750 XT validation once the real GPU is exposed.

# Spock Handoff: Next Steps

## Current State

The layer-by-layer Vulkan decode path is materially correct but not yet aligned
with the model's official prompt-prefill contract.

- **Build**: `cmake --build build -j` passes.
- **Tests**: the project test suite was previously green at `12/12`; a new
  native DeltaNet chunk-rule unit test has now been added.
- **Executable parity gate**: `spock_vk_decode_reference_parity` checks the
  first 8 frozen prompts for 16 generated tokens each.
- **Current frozen sweep**: 43/48 prompts match the frozen HF/repacked corpus.

Important environment note: current Vulkan execution is still on **llvmpipe**
in this session, not on the RX 6750 XT. The code path is Vulkan compute, but
it is software-backed until RADV is exposed.

## What Is Correct

These pieces are now in good shape:

- **RMSNorm semantics**: fixed to `norm(x) * (1 + weight)` for Qwen3.5
- **RoPE pairing**: fixed to split-half `rotate_half` semantics
- **Attention decode path**: V accumulation bug fixed for short sequences
- **DeltaNet recurrent decode path**: recurrent single-token update matches the
  HF recurrent rule closely
- **Native DeltaNet chunk primitive**: host-side C++ implementation now matches
  the HF torch chunk rule to float32 noise

## Real Remaining Gap

The remaining parity failures are not pointing at another obvious shader bug.
They are pointing at **prompt prefill semantics**.

HF Qwen3.5 does this:

- `seq_len > 1` prompt prefill: **chunk gated delta rule**
- `seq_len == 1` decode with cache: **recurrent gated delta rule**

Current Vulkan runtime does this:

- prompt prefill: **recurrent one-token updates**
- decode: **recurrent one-token updates**

That explains why Vulkan can match HF sequential prefill for a failing prompt
while still disagreeing with the frozen chunk-prefill corpus.

## Immediate Next Work

### 1. Refactor prompt prefill in `vk_decode.cpp`

This is the real correctness task now.

Target shape:

- separate **prefill** from **decode**
- make prefill **layer-major**
- keep attention token-sequential within prefill
- run DeltaNet prompt tokens through the new native chunk helper
- keep decode on the existing recurrent single-token path

### 2. Re-run the full 48-prompt frozen parity sweep

After chunk-prefill integration, rerun:

```text
python3 tests/run_vk_decode_parity.py --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --reference tests/data/reference_tokens.jsonl \
  --limit 48 --max-new-tokens 16
```

The goal is to clear the remaining 5 prompts, not to widen the test gate first.

### 3. Only then move to real GPU validation

Once parity is solid:

- expose RADV / RX 6750 XT instead of llvmpipe
- rerun parity on the real device
- then start honest `tg128` / `pp520` performance work

## Performance Work That Should Wait

Do not spend time optimizing around the current prompt-prefill mismatch.

These are valid later items:

- move `g/beta` computation off the CPU
- reduce command-buffer submit count
- collapse barriers and combine layers
- benchmark against the best local generic Vulkan baseline

But they are downstream of correctness.

## Key Files

| File | Purpose |
|------|---------|
| `src/runtime/vk_decode.cpp` | Main runtime; needs the prompt-prefill refactor |
| `src/runtime/deltanet_chunk.cpp` | Native HF-style chunk-rule primitive |
| `shaders/deltanet_recurrent.comp` | Recurrent decode/update kernel |
| `shaders/attention_decode.comp` | Attention decode kernel |
| `tests/run_vk_decode_parity.py` | Executable Vulkan-vs-reference parity harness |
| `tests/run_deltanet_chunk_unit.py` | Torch-vs-native chunk-rule regression test |
| `diary/0013_native_deltanet_chunk_rule.md` | This session's diary |

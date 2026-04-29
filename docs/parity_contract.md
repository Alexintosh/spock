# Vulkan Megakernel Parity Contract

This contract freezes what "parity" means for the RX 6750 XT Vulkan megakernel plan. It is intentionally model-specific and hardware-specific.

## Scope

- Model: `Qwen/Qwen3.5-0.8B`
- Target GPU: `AMD Radeon RX 6750 XT (RADV NAVI22)`
- API: Vulkan compute
- Batch size: `1`
- Primary path: decode
- Secondary path: prefill after decode parity
- Production precision: `fp16` weights and activations with `fp32` DeltaNet state
- Maximum sequence length for v1: `2048`

## Fixed Inputs

The prompt corpus lives at `tests/data/prompts.jsonl`.

Required prompt classes:

- `short_correctness`: short prompts for fast exact-token checks.
- `pp520`: prompts intended to tokenize near the prefill benchmark length.
- `tg128`: prompts used to generate exactly 128 decode tokens.
- `mixed_correctness`: varied deterministic prompts covering instruction, code, math, formatting, and edge punctuation.

Prompt text is part of the contract. Changing a prompt requires a new corpus version and invalidates prior parity results.

## Decode Semantics

- Tokenizer revision must match the pinned model artifact revision.
- BOS and EOS handling must be explicit in benchmark output.
- Default correctness mode is greedy decode.
- Sampling tests, when added, must use a named deterministic sampler and fixed seed.
- Decode length for throughput gate `tg128` is exactly `128` generated tokens unless EOS occurs earlier; early EOS must be reported and does not satisfy throughput gates.
- Correctness comparisons are token-id based, not text-string based.

## Measurement Protocol

All benchmark runs must record:

- host CPU and OS
- GPU name
- Vulkan API version
- driver name and version
- model revision
- artifact format version
- precision mode
- prompt corpus version
- warmup count
- timed run count
- GPU timestamp availability
- synchronization points

Default benchmark protocol:

- Warmup runs: `5`
- Timed runs: `20`
- Report median, mean, min, max, and standard deviation.
- Use GPU timestamps for kernel timing when available.
- Include end-to-end host timing for user-visible throughput.
- Synchronize only at declared measurement boundaries.

## Scoreboard

| Level | Name | Decode Gate | Prefill Gate | Correctness Gate |
| --- | --- | --- | --- | --- |
| `P0` | Correctness parity | N/A | N/A | Exact greedy-token parity on the fixed corpus |
| `P1` | Baseline parity | `tg128 >= 1.0x` local generic Vulkan baseline | `pp520 >= 1.0x` local generic Vulkan baseline | P0 satisfied |
| `P2` | Relative parity | `tg128 >= 1.25x` local generic Vulkan baseline | `pp520 >= 1.75x` local generic Vulkan baseline | P0 satisfied |
| `P3` | Luce-style parity | `tg128 >= 1.45x` local generic Vulkan baseline | `pp520 >= 2.5x` local generic Vulkan baseline | P0 satisfied |

## Operational Parity Claims

The implementation may claim one of these mutually exclusive runtime modes:

- `layer_by_layer`: separate Vulkan dispatches per operation or layer.
- `single_submit`: one GPU submission per token with no host mediation between layers.
- `persistent_dispatch`: one persistent dispatch for the decode pass.

Full megakernel parity requires `persistent_dispatch`. If cross-workgroup synchronization is not reliable on this GPU and driver stack, the strongest honest claim is `single_submit`.

## Pass Criteria

A run passes `P0` only if every prompt in `tests/data/prompts.jsonl` produces the exact expected token-id sequence against the trusted CPU reference.

The executable Vulkan parity harness is `tests/run_vk_decode_parity.py`. The
CTest gate currently checks the first eight frozen prompts for 16 generated
tokens each. This is a regression guard, not a full `P0` claim. Full `P0`
requires running the harness over all 48 frozen prompts and all expected
generated tokens.

A run passes a performance level only if:

- P0 has passed for the same model revision and artifact.
- The local generic Vulkan baseline in `bench/baseline_rx6750xt.json` is populated.
- The same prompt corpus and decode settings are used.
- Throughput ratios meet or exceed the scoreboard threshold.
- The benchmark output includes enough metadata to reproduce the result.

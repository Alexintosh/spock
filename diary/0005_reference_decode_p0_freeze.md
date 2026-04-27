# 0005: Reference Decode And P0 Token Freeze

## Goal

The fifth phase establishes the P0 correctness baseline by running the full prompt corpus through the trusted HuggingFace transformers reference and freezing the exact greedy token sequences. This is the contract that every future Spock decode path must match.

## Why A Reference Decode Matters

The implementation plan defines four parity tiers. P0 is correctness parity: exact greedy-token agreement on fixed prompts. Before writing Vulkan kernels, the project needs to know exactly what tokens the model should produce for each test prompt.

Without a frozen reference, there is no way to distinguish between a correct Vulkan path and one that happens to produce plausible-looking text with wrong token IDs. Token-level correctness is harder to achieve than text-level similarity, but it is the only honest standard for a specialized engine.

## Approach

Rather than reimplementing the full Qwen 3.5 forward pass from scratch in numpy, this phase uses the HuggingFace transformers model directly with PyTorch. The model has a complex hybrid architecture with DeltaNet linear attention and full attention layers. The DeltaNet block uses gated delta rule recurrence with chunk (prefill) and recurrent (decode) paths, causal convolution, and FP32 recurrent state. Reimplementing all of that correctly in numpy before even starting Vulkan would be a large detour.

Using the upstream implementation as the reference gives us:

- A trusted source of truth for token IDs.
- Deterministic output under float32 CPU execution.
- A tool we can run at any time to regenerate or extend the reference.

## Tokenizer

The Qwen 3.5 tokenizer is tiktoken-based. It was downloaded from HuggingFace and saved to the local artifact cache. Key details:

- Vocabulary size: 248,320 tokens
- EOS token ID: 248,046 (`<|im_end|>`)
- No BOS token in the standard configuration; the model treats the first prompt token as the start.

## Reference Decode Tool

The tool is `tools/reference_decode.py`. It:

1. Loads the model from the local HuggingFace cache in float32.
2. Iterates over every prompt in `tests/data/prompts.jsonl`.
3. Generates `max_new_tokens` tokens greedily (no sampling).
4. Records prompt token IDs, generated token IDs, token counts, and per-prompt timing.
5. Writes one JSON line per prompt to the output file.

Typical performance on this machine: approximately 3-7 seconds per prompt depending on input length, running single-threaded CPU float32.

## Frozen Reference

The frozen reference is at `tests/data/reference_tokens.jsonl`. It contains 48 entries:

- 8 short correctness prompts.
- 24 mixed correctness prompts.
- 8 tg128 prompts (longer inputs for decode benchmarks).
- 8 pp520 prompts (long inputs for prefill benchmarks).

Each entry generates 16 tokens. No prompt triggers early EOS. Total generated tokens: 768.

The reference was verified for determinism by running the same prompt twice and confirming identical token sequences.

## P0 Parity Test

The test is `tests/run_p0_parity.py`. It checks:

- Structural integrity of the reference file.
- Minimum entry count (at least 32 prompts, matching the parity contract).
- Non-empty prompt and generated token arrays.

This test does not run inference itself. It validates the reference artifact that future decode paths will be compared against. When a Vulkan decode path exists, a separate comparison test will load the Vulkan output and check it against this reference token by token.

## What This Enables

With P0 tokens frozen, the next milestones can proceed:

- Milestone 4 (Vulkan runtime bring-up) can use real weights from `text_weights.bin`.
- Milestone 5 (layer-by-layer Vulkan decode) can compare its output against these exact tokens.
- Any intermediate tensor dump from the Vulkan path can be compared against the PyTorch model's intermediate activations.

## What Is Still Needed For Real Inference

The reference is correct but it runs on PyTorch, not on the Spock weight pipeline. To prove the full artifact pipeline is sound, the next step is either:

1. Load the repacked `text_weights.bin` in Python, reconstruct the model, and verify it produces the same tokens.
2. Or implement the forward pass directly in C++ using the repacked weights.

Option 1 is faster and proves the weight pipeline end-to-end without writing Vulkan kernels.

## Verification

Generate reference tokens:

```sh
python3 tools/reference_decode.py \
    --model-dir artifacts/hf/Qwen--Qwen3.5-0.8B \
    --tokenizer-dir artifacts/hf/Qwen--Qwen3.5-0.8B-tokenizer \
    --prompts tests/data/prompts.jsonl \
    --output tests/data/reference_tokens.jsonl \
    --max-new-tokens 16
```

Verify P0 parity:

```sh
python3 tests/run_p0_parity.py \
    --reference tests/data/reference_tokens.jsonl \
    --check-count 32
```

Full build verification:

```sh
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

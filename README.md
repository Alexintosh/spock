# Spock

Spock is a Vulkan-native inference engine scaffold specialized for `Qwen/Qwen3.5-0.8B` on an RX 6750 XT class RADV stack.

The current implementation freezes the parity contract, model constants, artifact format, build system, shader compilation, CLI surface, and P0 reference tokens. The actual optimized decode kernels remain milestone work.

## Build

```sh
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## CLIs

```sh
./build/spock-check --prompts tests/data/prompts.jsonl
./build/spock-bench --mode tg128 --json
./build/spock-bench --mode pp520 --csv --output /tmp/spock-pp520.csv
./build/vk-capabilities
./build/vk_barrier_probe --iterations 10000
python3 tools/convert_qwen35_0p8b.py --offline --output /tmp/spock-artifact --force
python3 tools/validate_artifact.py /tmp/spock-artifact --json
python3 tools/convert_qwen35_0p8b.py --safetensors-scan --input /path/to/Qwen3.5-0.8B --output /tmp/spock-real-artifact --force
python3 tests/run_actual_model_smoke.py --model-dir /path/to/Qwen3.5-0.8B --converter tools/convert_qwen35_0p8b.py --validator tools/validate_artifact.py
python3 tools/plan_text_artifact.py /tmp/spock-real-artifact --output /tmp/spock-real-artifact/text_plan.json
python3 tools/repack_text_weights.py /tmp/spock-real-artifact/text_plan.json --output-dir /tmp/spock-text-repack --force
python3 tools/validate_text_repack.py /tmp/spock-text-repack --check-hashes --json
python3 tools/export_repack_tasks.py /tmp/spock-real-artifact/text_plan.json --output /tmp/spock-real-artifact/text_repack_tasks.tsv
./build/spock-repack-text --tasks /tmp/spock-real-artifact/text_repack_tasks.tsv --source-root /tmp/spock-real-artifact --output-dir /tmp/spock-text-repack-native
python3 tools/reference_decode.py --model-dir artifacts/hf/Qwen--Qwen3.5-0.8B --tokenizer-dir artifacts/hf/Qwen--Qwen3.5-0.8B-tokenizer --repack-dir artifacts/spock-text-repack-qwen35-0p8b --prompts tests/data/prompts.jsonl --output tests/data/reference_tokens.jsonl --max-new-tokens 16
python3 tools/verify_repack_parity.py --model-dir artifacts/hf/Qwen--Qwen3.5-0.8B --tokenizer-dir artifacts/hf/Qwen--Qwen3.5-0.8B-tokenizer --repack-dir artifacts/spock-text-repack-qwen35-0p8b --reference tests/data/reference_tokens.jsonl
python3 tests/run_p0_parity.py --reference tests/data/reference_tokens.jsonl --check-count 32
## Status

- `P0` contract and corpus are defined.
- P0 reference tokens are frozen for all 48 prompts (768 generated tokens) using repacked FP16 weights.
- CMake builds all C++ CLIs and shader SPIR-V outputs.
- Vulkan capability discovery works on the local RADV device.
- Artifact dry-run conversion, text plan, and weight repacking are implemented.
- Weight pipeline verified end-to-end: repacked FP16 weights produce exact P0 parity (48/48 prompts).
- Reference decode uses the trusted HuggingFace transformers model for deterministic greedy output.
- Per-layer activation capture available for Vulkan kernel debugging.
- Vulkan decode kernels and barrier probe remain pending implementation.
- Engineering diary entries live in `diary/` and explain each phase for programmers new to LLM inference.

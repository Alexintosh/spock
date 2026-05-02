# Spock

Spock is a Vulkan-native inference engine specialized for `Qwen/Qwen3.5-0.8B`
on `AMD Radeon RX 6750 XT (RADV NAVI22)`.

The governing scope is the megakernel roadmap in
`IMPLEMENTATION_PLAN.md`: exact greedy-token parity first, then reusable
decode/session infrastructure, then single-submit and persistent decode work on
the RX 6750 XT.

## Current Status

- Runtime device selection now deterministically prefers the RX 6750 XT over
  `llvmpipe`, and `vk-capabilities` reports the same selected device that
  `spock-decode` actually uses.
- `spock-decode` runs end-to-end on the real RADV device with all 24 Qwen
  3.5-0.8B layers wired, including attention and DeltaNet decode paths.
- The executable hardware parity gate checks the first 8 frozen prompts for 16
  generated tokens each and currently passes on RADV.
- Full frozen-corpus `P0` is not complete. The remaining known
  prompt-prefill-sensitive failures are:
  - `mixed_correctness_023`
  - `mixed_correctness_025`
  - `mixed_correctness_026`
  - `mixed_correctness_027`
  - `pp520_046`
- `spock-bench` is still a placeholder CLI. It is useful for output-shape and
  interface work only, not for throughput claims.

## Build

```sh
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Capability And Parity

```sh
./build/vk-capabilities
ctest --test-dir build --output-on-failure -R spock_vk_decode_reference_parity
ctest --test-dir build --output-on-failure -R spock_vk_decode_prefill_handoff_mismatch
python3 tests/run_vk_decode_parity.py --decode build/spock-decode --repack-dir artifacts/spock-text-repack-qwen35-0p8b --reference tests/data/reference_tokens.jsonl --limit 8 --max-new-tokens 16
python3 tests/run_vk_decode_parity.py --decode build/spock-decode --repack-dir artifacts/spock-text-repack-qwen35-0p8b --reference tests/data/reference_tokens.jsonl --ids mixed_correctness_023,mixed_correctness_025,mixed_correctness_026,mixed_correctness_027,pp520_046 --max-new-tokens 16
```

The focused `spock_vk_decode_prefill_handoff_mismatch` test is temporary. It is
an expected-mismatch reproducer for the remaining prompt-prefill bug and should
be inverted into a positive parity test once that work lands.

## Core CLIs

```sh
./build/spock-check --prompts tests/data/prompts.jsonl
./build/spock-decode --repack-dir artifacts/spock-text-repack-qwen35-0p8b --tokens /tmp/prompt.tokens --max-new-tokens 16 --verbose
./build/spock-bench --mode tg128 --json
./build/spock-bench --mode pp520 --csv --output /tmp/spock-pp520.csv
./build/vk-capabilities
./build/vk_barrier_probe --iterations 10000
```

## Artifact Tools

```sh
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
```

## Repo Landmarks

- `IMPLEMENTATION_PLAN.md`: authoritative roadmap for the RX 6750 XT
  megakernel target.
- `src/runtime/vk_decode.cpp`: current end-to-end runtime and the next major
  refactor point.
- `src/runtime/deltanet_chunk.cpp`: native host-side chunk-rule primitive used
  to validate DeltaNet prompt-prefill semantics.
- `tests/run_vk_decode_parity.py`: executable Vulkan-vs-reference parity
  harness.
- `NEXT_STEPS.md`: current handoff and critical-path notes.
- `diary/`: engineering diary entries explaining each implementation phase.

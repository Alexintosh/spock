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

**Experimental GPU chunk-prefill path** available behind `SPOCK_GPU_CHUNK_PREFILL=1`.
Passes `mixed_correctness_023` and `pp520_046` at `--max-new-tokens 1` (conservative
per-head-submit workaround; not the default).

**GPU-collected chunk-prefill path** available behind
`SPOCK_GPU_CHUNK_PREFILL=1 SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1`.
Preserves GPU-collected Q/K/V/g/beta buffers for all DeltaNet layers in
device-local per-layer segments and feeds them directly into
`deltanet_chunk_prefill.comp` via `gpu_chunk_prefill_from_gpu_collect()`.
This avoids CPU intermediate packing/upload for the chunk-prefill inputs.
Default behavior unchanged. When no diagnostic compare flag is active, the
per-token CPU collection bridge (staging download, half_to_float conversion,
prefill_chunks_ population) is now skipped entirely — the gated path hands
off directly from collect dispatch to chunk-prefill dispatch with no
host-side data touching the collected activations. CPU collection remains
when either `SPOCK_GPU_COLLECT_PREFILL_COMPARE=1` or
`SPOCK_GPU_CHUNK_PREFILL_COMPARE=1` is set.

**CTest regression gate for GPU-collected chunk-prefill path** added in diary 0022.
Two new CTest tests protect the double-gated path:
- `spock_vk_decode_gpu_collect_chunk_prefill_short` — runs
  `short_correctness_001 --max-new-tokens 1` with
  `SPOCK_GPU_CHUNK_PREFILL=1` and `SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1`.
- `spock_vk_decode_gpu_collect_chunk_prefill_short_baseline` — runs the same
  prompt with no env vars as a quick diagnostic reference.

**Runtime GPU prefill collection diagnostic** available behind
`SPOCK_GPU_COLLECT_PREFILL_COMPARE=1`. During layer-major DeltaNet prefill,
dispatches `deltanet_prefill_collect.comp` from the real per-token activation
buffers, downloads the GPU-collected fp32 head-major Q/K/V/g/beta, and compares
against the CPU-collected `PrefillChunkState`. Verified exact match (max_rel=0,
max_abs=0, nan_count=0) across all 18 DeltaNet layers on
`short_correctness_001` (seq_len=9, token 271). Diagnostic only — does not change
inference output or default behavior.

**GPU prefill-collection probe** (`spock-deltanet-prefill-collect-probe`) proves a
shader can write per-token fp16 dn_qkv + fp32 g/beta into fp32 head-major buffers;
verified exact match (max_rel=0, nan_count=0) at heads=16, seq_len=104, k_dim=v_dim=128.

**Combined GPU prefill pipeline probe** (`spock-deltanet-prefill-pipeline-probe`)
proves the collection shader output can feed `deltanet_chunk_prefill.comp`
directly without CPU intermediate packing. Verified `compare-ok` at heads=16,
seq_len=104, total_seq=128, chunk_size=64 with max_rel_core=8.94e-8,
max_rel_state=1.19e-7, nan_count=0.

**Tiled single-dispatch chunk-prefill probe** (diary 0023) proves a single
`vkCmdDispatch(num_heads, ceil(v_dim/TILE_V), 1)` can replace the per-head
submit workaround. The experimental shader
(`deltanet_chunk_prefill_tiled.comp`) matches the CPU chunk-rule reference
within machine epsilon. Not yet wired into runtime — integration behind an
env gate is the next step.
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
- `shaders/deltanet_chunk_prefill_tiled.comp`: experimental tiled single-dispatch
  chunk-prefill shader (diary 0023).

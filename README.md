# Spock
<div style="display:flex; align-items: center; margin 0 auto; ">
  <img src="https://github.com/Alexintosh/spock/blob/5b3195753fcb7590a87d5fc22ddfe7154259090a/logo.jpg" width="400"/>
  <i><h3>Live long and Infer!</h3></i>
</div>

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
Supports two dispatch modes: per-head submit (default, slow) or tiled single-dispatch
with `SPOCK_GPU_CHUNK_PREFILL_TILED=1`. The existing gated path passes
`mixed_correctness_023` and `pp520_046` at `--max-new-tokens 1`; the tiled
runtime gate now also passes those prompts at `--max-new-tokens 4` and
`short_correctness_001` at `--max-new-tokens 16`. Not the default.

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

**GPU-resident chunk-prefill output handoff** (diary 0025). On the
no-compare GPU-collected+tiled path (`SPOCK_GPU_CHUNK_PREFILL=1`,
`SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1`,
`SPOCK_GPU_CHUNK_PREFILL_TILED=1`, no compare flag), chunk-prefill output
no longer transits through host-visible memory. The tiled shader writes to
a device-local buffer; `final_state` is copied GPU-to-GPU into
`bufs_->dn_state`; a new shader `deltanet_chunk_last_to_fp16.comp` extracts
the last-token fp32 `core_attn_out` slice and converts to fp16 on-device;
`correct_last_token_hidden()` copies that fp16 slice GPU-to-GPU into the
`B.dn_qkv` V region. The CPU readback/upload bridge for chunk-prefill
output is eliminated on this path. Fallback host-visible path preserved for
compare diagnostics. Does NOT make full GPU offload complete (per-layer
host orchestration remains, plus decode argmax, diagnostic readbacks).
Still env-gated, not default.

**GPU-side chunk init clear** (diary 0026). On the same no-compare
GPU-collected+tiled path, the chunk-prefill init state buffer (`init_buf`)
is now device-local and zeroed via `vkCmdFillBuffer` instead of
host-visible + CPU `memset`. This removes the last CPU data touch for
chunk-prefill compute on the fast path. The host still orchestrates
(layer iteration, command recording, submission, fence wait). Fallback
paths (compare, non-tiled, CPU-collected) still use host-visible init_buf
+ CPU memset. Verified parity on `short_correctness_001` (16 tokens),
all CTest gates pass. Still env-gated, not default.

**Opt-in device-resident decode token embedding** (diary 0027). Behind
`SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1`, the per-step embedding lookup reads
token_id directly from the device-local `argmax_result` buffer instead of
from a CPU-supplied push constant. The CPU still downloads `argmax_result`
each decode step for external output and parity — this is not full GPU
offload and not the megakernel. It only removes the CPU token value as
the *source* for the next step's embedding. The current serial loop still
downloads before the next iteration. New shader `embedding_lookup_from_buffer.comp`.
Verified parity on `short_correctness_001` (16 tokens),
`mixed_correctness_023`/`pp520_046` (4 tokens), and combined with
full GPU chunk-prefill gate suite. Still env-gated, not default.

**Opt-in deferred generated-token download** (diary 0028). Behind
`SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1 SPOCK_GPU_DEFER_TOKEN_DOWNLOAD=1`,
the per-step CPU download of `argmax_result` is replaced by a device-local
`vkCmdCopyBuffer` into a `gen_tokens` buffer and a single batch download
after the decode loop. Disabled for `verbose`/`debug_dump`/`diagnose_decode_drift`.
Guards `max_new_tokens > 0` to avoid zero-sized Vulkan buffer allocation;
zero-token parity passes. Default behavior remains per-step download; the
gate is also disabled for `verbose`, `debug_dump`, and
`diagnose_decode_drift`. Does not restructure the submit-wait loop — no
performance speedup claimed. Still env-gated, not default.

**Opt-in per-layer stable descriptor sets** (diary 0029). Behind
`SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1`, per-layer descriptor mutation
in `decode()` is eliminated by pre-allocating and pre-binding layer-specific
descriptor sets at session construction time. Covers common MLP/norm
descriptors (input_norm, residual, post_norm, gate, up, down, MLP residual
paths), attention descriptors (Q/K/V projections, QK-norm, KV store,
attention decode, O projection, attention residual paths), and first-stage
DeltaNet descriptors (QKV/Z/A/B projections, conv1d). RoPE descriptors
still mutate per step (step-dependent rope frequency offset). Intra-DeltaNet
sub-step descriptors (dn_l2_q, dn_l2_k, dn_recurrent, dn_norm_gate,
dn_out_proj, dn_compute_g_beta) are NOT covered and remain on the old
path. Increases descriptor pool capacity from 192 to 1024 maxSets and
192 to 4096 storage buffer slots. Default behavior unchanged — env-gated,
not default. This is NOT full GPU offload, NOT single-submit, and NOT the
megakernel. Reduces per-layer descriptor mutation under the gate and is a
prerequisite for future single-submit recording. No performance speedup
claimed.

**Runtime tiled chunk-prefill gate** (diary 0024). The tiled single-dispatch
shader is wired into the runtime behind `SPOCK_GPU_CHUNK_PREFILL_TILED=1`.
Each DeltaNet layer issues one `vkCmdDispatch(num_heads, ceil(v_dim/16), 1)`
instead of 16 per-head submits. CTest tiled gate runs in 16.82 sec in the
earlier diary 0025 run and 8.95 sec in the latest rerun after handoff flag
cleanup. Still env-gated, not default.
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
./build/vk_barrier_probe --iterations 10000 --workgroups 8
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

<<<<<<< HEAD
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
- `shaders/deltanet_chunk_prefill_tiled.comp`: tiled single-dispatch
  chunk-prefill shader, wired into runtime (diary 0024).
=======
- `P0` contract and corpus are defined.
- P0 reference tokens are frozen for all 48 prompts (768 generated tokens) using repacked FP16 weights.
- CMake builds all C++ CLIs and shader SPIR-V outputs.
- Vulkan capability discovery works on the local RADV device.
- Artifact dry-run conversion, text plan, and weight repacking are implemented.
- Weight pipeline verified end-to-end: repacked FP16 weights produce exact P0 parity (48/48 prompts).
- Reference decode uses the trusted HuggingFace transformers model for deterministic greedy output.
- Vulkan decode pipeline runs end-to-end on the RX 6750 XT.
- 6 compute shaders compiled and dispatched: embedding lookup, RMSNorm, matvec, argmax, silu_gate, residual_add.
- Full weight upload to GPU (320 tensors, 1435 MiB).
- Observed subgroup size: 64 (not the originally assumed 32).
- Full MLP forward pass wired and verified correct (matches numpy reference).
- Token mixer (attention/DeltaNet) is identity pass-through — model echoes input without context.
- Engineering diary entries live in `diary/` and explain each phase for programmers new to LLM inference.
>>>>>>> da7f97ce0f59dc8d3d4db924ef5548ddfab77c00

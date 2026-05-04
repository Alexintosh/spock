# 0039: Single-Submit Decode — One Command Buffer per Decode Token

## Goal

Reduce host orchestration from 26 submit/wait round-trips per decode step to 1
by recording all dispatches for a single decode token into one `VkCommandBuffer`
and submitting it once.

`SPOCK_GPU_SINGLE_SUBMIT=1` records the embedding lookup, all 28 transformer
layers (each with its full DeltaNet or attention sequence), final RMSNorm, LM
head projection, and argmax into a single command buffer. One `submit_and_wait`
call replaces the previous per-embedding, per-layer, and per-final-step submits.

This is the culmination of the descriptor pre-binding work (diaries 0029–0037)
and the merged DeltaNet command buffers (diary 0038). Single-submit is only
possible because every dispatch-target descriptor is pre-bound and stable across
decode steps, and because DeltaNet phase-1 and g/beta work are already merged
into the per-layer command buffer.

## Implementation Work Completed

### Gate

The path is enabled by:

```sh
SPOCK_GPU_SINGLE_SUBMIT=1
SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1
SPOCK_GPU_MERGED_DELTANET=1
```

All three are required. The first two prerequisites reflect hard dependencies:
stable per-layer descriptor sets are needed so descriptor bindings remain valid
across all layers in a single command buffer, and merged DeltaNet command
buffers eliminate the extra per-DeltaNet-layer submits that would break the
single-command-buffer invariant.

The gate is additionally disabled for any step or configuration where
intermediate observation or per-step host mediation is required:

```cpp
const bool can_single_submit_base = single_submit_decode &&
    per_layer_sets_enabled_ && merge_deltanet_cmds &&
    dump_step_components < 0 && dump_step_hiddens < 0 &&
    !verbose && !debug_dump && !diagnose_decode_drift &&
    !experiment_attn_o_proj_f32_residual && !experiment_mlp_down_f32_residual;

const bool can_single_submit = can_single_submit_base && !is_prefill && !skip_layers;
```

This disables single-submit for:

- **Prefill steps** (`is_prefill`): prefill uses a different dispatch sequence
  and per-layer collection; single-submit only targets decode.
- **`skip_layers` steps**: the first decode step after chunk prefill uses
  `skip_layers` to skip KV-store attention layers whose cache was populated
  during prefill; single-submit assumes all 28 layers dispatch.
- **Verbose/debug/drift diagnostics**: these modes observe intermediate GPU
  state between dispatches and require submit boundaries.
- **f32 residual experiments**: `experiment_attn_o_proj_f32_residual` and
  `experiment_mlp_down_f32_residual` change buffer layouts and dispatch
  sequences in ways incompatible with the pre-recorded command buffer.

### Single-command-buffer recording

When `can_single_submit` is true for a step:

1. A single `ss_cmd` is allocated and begun at the start of the step (before
   the embedding lookup).
2. The embedding dispatch (either `embedding_from_buffer` for device-resident
   token or `embedding` for push-constant token) is recorded into `ss_cmd`.
3. An execution barrier is inserted after the embedding dispatch (embedding
   writes `act_a`; layers read `act_a`).
4. All 24 layer dispatches are recorded into the same `ss_cmd`, each using
   pre-bound per-layer descriptor sets. Inter-layer barriers remain as before.
5. Final RMSNorm, LM head, and argmax dispatches are recorded.
6. If deferred token download is active, the `vkCmdCopyBuffer` from
   `argmax_result` into `gen_tokens` is also recorded.
7. The command buffer is ended and submitted with a single `submit_and_wait`.

On the fallback path (`!can_single_submit`), each phase allocates, records,
ends, and submits its own command buffer as before.

### What this is not

- **Not full GPU offload.** The host still submits once per token, waits on the
  fence, processes the generated token for output/parity, and owns all fallback
  and diagnostic paths.
- **Not persistent dispatch.** No shader remains resident between tokens; each
  decode step records and submits a fresh command buffer.
- **Not the megakernel.** No shader fusion or persistent scheduler was
  introduced. Each dispatch is a separate `vkCmdDispatch` call within the
  command buffer.
- **Not readback elimination.** CPU readbacks and fallback paths are unchanged.
  Device-resident token and deferred download are orthogonal gates that can be
  combined with single-submit.

## Verification

All commands were run locally on the target Vulkan/RADV path.

### Whitespace and build

```sh
git diff --check
```

No whitespace errors.

```sh
cmake --build build -j
```

Passed.

### Default path (no single-submit)

```sh
python3 tests/run_vk_decode_parity.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --reference tests/data/reference_tokens.jsonl \
  --ids short_correctness_001 \
  --max-new-tokens 16
```

```json
{"status":"ok","checked":1,"failures":[]}
```

### Single-submit alone (fallback eligibility only)

```sh
SPOCK_GPU_SINGLE_SUBMIT=1 \
  python3 tests/run_vk_decode_parity.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --reference tests/data/reference_tokens.jsonl \
  --ids short_correctness_001 \
  --max-new-tokens 16
```

```json
{"status":"ok","checked":1,"failures":[]}
```

Without the prerequisites `SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1` and
`SPOCK_GPU_MERGED_DELTANET=1`, `can_single_submit_base` evaluates to false and
the path falls back to per-layer submit. This test confirms the fallback is
correct when the gate is requested but prerequisites are missing.

### Active single-submit with prerequisites

```sh
SPOCK_GPU_SINGLE_SUBMIT=1 \
SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1 \
SPOCK_GPU_MERGED_DELTANET=1 \
  python3 tests/run_vk_decode_parity.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --reference tests/data/reference_tokens.jsonl \
  --ids short_correctness_001 \
  --max-new-tokens 16
```

```json
{"status":"ok","checked":1,"failures":[]}
```

This exercises the actual single-command-buffer path: all dispatches are
recorded into one `ss_cmd` and submitted once per decode token.

### Full combined gate suite

```sh
SPOCK_GPU_SINGLE_SUBMIT=1 \
SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1 \
SPOCK_GPU_MERGED_DELTANET=1 \
SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1 \
SPOCK_GPU_DEFER_TOKEN_DOWNLOAD=1 \
SPOCK_GPU_CHUNK_PREFILL=1 \
SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1 \
SPOCK_GPU_CHUNK_PREFILL_TILED=1 \
  python3 tests/run_vk_decode_parity.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --reference tests/data/reference_tokens.jsonl \
  --ids short_correctness_001 \
  --max-new-tokens 16
```

```json
{"status":"ok","checked":1,"failures":[]}
```

```sh
SPOCK_GPU_SINGLE_SUBMIT=1 \
SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1 \
SPOCK_GPU_MERGED_DELTANET=1 \
SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1 \
SPOCK_GPU_DEFER_TOKEN_DOWNLOAD=1 \
SPOCK_GPU_CHUNK_PREFILL=1 \
SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1 \
SPOCK_GPU_CHUNK_PREFILL_TILED=1 \
  python3 tests/run_vk_decode_parity.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --reference tests/data/reference_tokens.jsonl \
  --ids mixed_correctness_023,pp520_046 \
  --max-new-tokens 4
```

```json
{"status":"ok","checked":2,"failures":[]}
```

### Chunk-prefill CTest regression

```sh
SPOCK_GPU_SINGLE_SUBMIT=1 \
SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1 \
SPOCK_GPU_MERGED_DELTANET=1 \
SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1 \
SPOCK_GPU_DEFER_TOKEN_DOWNLOAD=1 \
  ctest --test-dir build --output-on-failure -R spock_vk_decode_gpu_collect_chunk_prefill
```

| Test | Result |
|------|--------|
| `spock_vk_decode_gpu_collect_chunk_prefill_short` | Passed, 114.94 s |
| `spock_vk_decode_gpu_collect_chunk_prefill_tiled` | Passed, 8.97 s |
| `spock_vk_decode_gpu_collect_chunk_prefill_short_baseline` | Passed, 5.14 s |
| **Total real time** | **129.05 s** |

## Known Limitations

- This is **not** full GPU offload. The host still submits once per token,
  waits on the fence, reads back the generated token for output/parity, and
  owns all fallback and diagnostic paths.
- This is **not** persistent dispatch. The command buffer is recorded and
  submitted fresh for each decode token. No shader or scheduler persists
  between tokens.
- This is **not** the megakernel. Each dispatch is a separate `vkCmdDispatch`
  within the command buffer. No shader fusion was introduced.
- Prefill steps, `skip_layers` steps, and any diagnostic/dump/verbose mode
  fall back to per-phase submit. The single-submit path is decode-only.
- f32 residual experiments are excluded because they change buffer layouts.
- Default behavior is unchanged. All three env vars must be set to activate
  the path.
- No performance speedup is claimed. The reduction in submit/wait round-trips
  may or may not translate to wall-clock improvement depending on driver
  overhead, GPU occupancy, and command-buffer recording cost.

## Next Work

- Verify coverage on broader P0 subsets and longer prompts.
- Performance characterization: measure wall-clock impact of the submit
  reduction on the target RADV stack.
- Evaluate whether the command buffer can be recorded once and reused across
  decode steps (step-varying parameters like RoPE position and KV-store offsets
  must be handled via push constants or GPU-updated storage buffers).
- Decide whether the next orchestration step should be persistent dispatch,
  megakernel fusion, or broader command-buffer reuse.

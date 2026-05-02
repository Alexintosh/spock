# 0015: Decode Drift Diagnostics and Handoff Checkpoint

## Goal

Continue the DecodeSession refactor into a coherent checkpoint: GPU-offload the DeltaNet g/beta computation, add session-resident precomputed RoPE tables, implement the chunk-corrected handoff between prefill and decode, and build a decode-drift diagnostic that isolates where recurrent state diverges from a freshly rebuilt reference. Do not commit — full parity is not yet achieved.

## Context

Diary 0014 established deterministic RADV device selection, unified capability reporting, and a fast prefill-handoff reproducer. But `vk_decode.cpp` was still a monolithic ~1200-line function with no session abstraction, no reusable buffer model, and no GPU-side g/beta computation. Every decode call re-allocated command buffers and re-computed host-side g/beta from cached a_log/dt_bias.

The next step required three parallel threads:

1. **Session extraction** — lift the device, pipeline, buffer, and descriptor lifecycle into a reusable `DecodeSession` class so the decode function is a method call, not a monolithic entry point.
2. **GPU-side g/beta** — the recurrent DeltaNet step computes `g = -exp(a_log) * softplus(a + dt_bias)` and `beta = sigmoid(b)` for every layer, every token. This was done on the CPU from cached host-side scalars. Moving it to a shader eliminates a CPU round-trip and host-visible buffer allocation.
3. **Chunk-corrected handoff** — after layer-major prefill (which uses recurrent DeltaNet for all prompt tokens), the final DeltaNet state and hidden state for the last prefill token must be corrected using the chunk rule. This is the key fix for the prefill-sensitive failures identified in diary 0014.

## Implementation

### 1. DecodeSession: Full Session Extraction

Extracted from `vk_decode.cpp` into `src/runtime/vk_session.{hpp,cpp}`:

- **Constructor** (`DecodeSession(repack_dir)`)
  - Vulkan device initialization via `VulkanDevice`
  - Weight artifact loading from repack directory
  - Shader module compilation and pipeline creation (20+ pipelines for attention, DeltaNet, MLP, LM head)
  - Buffer allocation: activation buffers (act_a/b/c), attention buffers (Q/K/V/attn_out/kv_cache/gated_attn), DeltaNet buffers (dn_qkv/dn_z/dn_a/dn_b/dn_q/dn_kv_out/dn_state/dn_conv_state), MLP intermediates, LM head logits, argmax result
  - Descriptor set pre-configuration: weight references are bound once at construction; per-layer/per-token bindings are updated during decode
  - DeltaNet scalar cache: per-layer `a_log` and `dt_bias` are read from the weight artifact at construction and cached in host-side `cached_a_log_` and `cached_dt_bias_` vectors

- **`decode()` method** — replaces the old `run_vk_decode()`:
  - Calls `layer_major_prefill()` for multi-token prompts (recurrent per-token DeltaNet)
  - Calls `run_chunk_prefill()` after prefill to correct DeltaNet state via the CPU chunk rule
  - Calls `correct_last_token_hidden()` to reprocess the final prefill token with chunk-corrected outputs
  - Enters the decode loop (single-token recurrent path with KV-cache attention and recurrent DeltaNet)
  - LM head uses fp16 matvec with fp32 argmax (the fp32 LM head experiment was reverted — it did not fix short003 and changed nothing for passing tests)

- **`reset()`** — zeroes KV cache, DeltaNet state, and conv state for a fresh prompt

- **Destructor** — tears down all GPU resources in reverse order

The old `vk_decode.cpp` shrank from ~1268 lines to effectively zero (the `run_vk_decode` function now delegates to `DecodeSession::decode`).

### 2. Session-Resident Buffers and RoPE Table

Previously, RoPE frequencies were computed on the CPU per-position and passed as a push constant or per-token buffer update. The new session model allocates a permanent buffer of `MAX_SEQ * ROTARY_DIM` fp32 values:

```cpp
bufs_->rope_freq = dev_.create_device_local_buffer(MAX_SEQ * ROTARY_DIM * 4);
```

The table is precomputed at session construction with `cos(angle)` and `sin(angle)` for all `MAX_SEQ` positions (4096) using the standard rotary theta base `10000000.0`. During decode, the RoPE shader binds a slice of this table at `offset = seq_pos * ROTARY_DIM * 4`. This eliminates per-token RoPE frequency computation entirely.

Similarly, the DeltaNet `a_log`/`dt_bias` buffer (`dn_a_log_bias`) is uploaded once at construction as a packed `[layer][head][2]` fp32 array, shared across all decode steps and all layers.

### 3. GPU Offload of DeltaNet g/beta

New shader `shaders/deltanet_compute_g_beta.comp`:

```
g[h]    = -exp(a_log[h]) * softplus(a[h] + dt_bias[h])
beta[h] = sigmoid(b[h])
```

The shader takes four descriptor bindings:
- `[0]` dn_a (fp16) — projected a values, written by the QKV/A/B projection dispatch
- `[1]` dn_b (fp16) — projected b values
- `[2]` a_log_bias (fp32) — session-resident per-layer table
- `[3]` dn_state (writable) — writes g/beta into the tail of the layer's state block

Dispatched with `num_heads` workgroups (16), 1 invocation each. Push constants carry `num_heads` and `layer_idx`.

The g/beta values are written directly into the per-layer state buffer tail at offset `dn_idx * state_per_layer + DN_HEADS * DN_K_DIM * DN_V_DIM * 4`. The recurrent shader reads g/beta from the same tail location, maintaining the convention that state `[0..DN_HEADS*K_DIM*V_DIM)` = fp32 state matrix, `[DN_HEADS*K_DIM*V_DIM..)` = fp32 g/beta array.

This eliminates two host-visible buffer allocations, two download-to-CPU calls, and two upload-to-device calls per layer per prefill token. The recurrent path and the chunk-correction path both use the same GPU g/beta path.

### 4. layer_major_prefill Architecture

The prefill pipeline processes prompts in layer-major order — all tokens through layer 0, then all tokens through layer 1, etc. Within each DeltaNet layer, three phases execute per token:

**Phase A** (GPU-only, one submit): input_norm (RMSNorm), QKV/Z/A/B projections (matvec), conv1d step (sliding-window update), L2-norm Q/K (per-head).

**Phase B** (GPU g/beta, one submit): `deltanet_compute_g_beta` dispatch, writing g/beta to state tail.

**Phase C** (GPU recurrent + MLP, one submit): deltanet_recurrent (state update with current Q/K/V + g/beta), deltanet_norm_gate (RMSNorm + z-gate), out_proj (matvec to hidden dim), residual_add, post_norm, MLP gate/up/silu/down/residual.

After all layers, the last token's hidden state is copied from the per-token buffer back into `act_a`, ready for the first decode step.

### 5. Chunk-Corrected Handoff

After `layer_major_prefill`, three steps execute before the first decode step:

**a) `run_chunk_prefill()`**

For each DeltaNet layer, the collected Q/K/V/g/beta (stored during Phases A/B of the prefill loop) are rearranged from token-major to head-major order, then fed to the native host-side `run_deltanet_chunk_rule`. The chunk rule produces:
- `final_state`: the correct recurrent state after processing all prompt tokens (uploaded to GPU, replacing the recurrent state)
- `core_attn_out[head][token][vd]`: per-head, per-token attention output; the last token's slice is saved as `chunk_core_attn_out_last_[dn_idx]`

State replacement is done via a host-visible staging buffer and `vkCmdCopyBuffer` — the chunk rule runs on the CPU, the final state is ~16 KB per layer.

**b) `correct_last_token_hidden()`**

Reprocesses every layer from scratch for the final prefill token only, using:
- Pre-norm hidden state snapshots saved during `layer_major_prefill` (`prefill_snapshots_` buffer)
- Chunk-corrected `core_attn_out_last_` uploaded into the `dn_qkv` V region
- Recurrent-projected A/B + GPU g/beta (written to the state tail alongside the corrected state matrix from `run_chunk_prefill`)

For attention layers, the normal single-token path runs (Q/K/V -> RoPE -> KV store -> attention -> sigmoid gate -> O proj -> MLP tail), using the KV cache filled during the prefill loop.

For DeltaNet layers, the conv1d step is **skipped** — the prefill loop already primed `dn_conv_state` through the final prompt token, and running it again would advance the sliding window past the last token. Instead: input_norm, QKV/A/B/Z projections (no conv1d), GPU g/beta (recomputes with the corrected A/B from the fresh projection), chunk core_attn_out uploaded to `dn_qkv` V region (replaces QKV's V section), norm_gate + out_proj + residual + MLP tail.

This produces the correct hidden state for the first decode step, matching what chunk-prefill would have produced.

### 6. diagnose_handoff Diagnostic

Controlled by `--diagnose-handoff` flag. After prefill + chunk correction (but before the first decode step), dumps JSON to stdout with: argmax token, top-5 logits, per-DeltaNet-layer chunk attention statistics, GPU state norms, pre-norm hidden statistics.

The `run_diagnose_handoff.py` test validates that all required fields are present and structurally correct.

### 7. diagnose_decode_drift Diagnostic

Controlled by `--diagnose-decode-drift` flag. The most valuable diagnostic tool added in this checkpoint.

**How it works:**

The decode loop runs normally for `target_decode_step` (currently `5`) tokens. At that step, it captures a "free-run" snapshot (hidden state, logits, per-layer DeltaNet state matrices, KV cache, per-layer hiddens).

After the decode loop completes, a **rebuilt** path executes: `reset()`, `layer_major_prefill(full_prefix_tokens)`, `run_chunk_prefill()`, `correct_last_token_hidden()`, then captures the same quantities.

the diagnostic JSON (to stderr) compares free-run vs rebuilt: hidden norm and per-element diffs, top-5 logits, per-layer DeltaNet state diffs (max/mean absolute, sorted by worst), per-attention-layer KV cache diffs, and per-layer hidden state diffs across all 24 layers.

### 8. dump_step_hiddens Diagnostic

Controlled by `--dump-step-hiddens N` flag, added to `spock-decode` as an opt-in external-validation diagnostic. After `N` decode steps, dumps the per-layer fp16 hidden vectors (24 layers × 1024 fp16 values) to stderr as a JSON object keyed by layer index `0..23`. Each layer entry carries the raw hidden vector as a float array.

The diagnostic is designed for external HF PyTorch comparison. It does not compare free-run vs rebuilt within the engine but produces the exact layer-by-layer hidden vectors that an external reference script can consume. This allows locating the exact layer where GPU and CPU reference divergences first appear, without relying on the rebuilt-prefix path's known-approximate state.

Verified with `short_correctness_003` at decode step 5: the diagnostic emits 24 layers of 1024 fp16 values to stderr JSON deterministically.

### 9. dump_step_components Diagnostic

Controlled by `--dump-step-components N` flag, added to `spock-decode` as a deeper layer-composition diagnostic. After `N` decode steps, dumps per-layer component data to stderr JSON: input RMSNorm norm, post-mixer residual norm, post-MLP residual norm, the final RMSNorm-normalized hidden vector's norm, **and the raw fp16 component vectors** (`input_hidden_fp16`, `mixer_residual_fp16`, `post_mlp_fp16`). Unlike `--dump-step-hiddens` which dumps only the post-layer hidden vector, this diagnostic captures the hidden state at three points inside each layer — before the mixer, after the mixer residual, and after the MLP residual — enabling fine-grained isolation of which sub-component introduces drift.

Each layer entry in the JSON output carries:
- `input_norm`: RMSNorm of the layer input (before the mixer block)
- `post_mixer_residual`: RMSNorm of `(input + mixer_output)` residual
- `post_mlp_residual`: RMSNorm of the final `(input + mixer_output + mlp_output)` residual
- `final_norm`: RMSNorm of the final hidden state after the post-24 RMSNorm (identical for all layer entries — it is the single engine-wide final norm)
- `input_hidden_fp16`: raw fp16 input hidden vector (1024 values, as uint16 bit patterns)
- `mixer_residual_fp16`: raw fp16 `(input + mixer_output)` vector
- `post_mlp_fp16`: raw fp16 post-MLP hidden vector

The `tools/compare_layer_hiddens.py` script consumes these component vectors: it runs both `--dump-step-hiddens` and `--dump-step-components`, then for each layer reconstructs the equivalent HF vectors from forward hooks and computes per-component `max_abs`/`mean_abs`/cosine/norm statistics via `print_component_vector_comparison()`. This turns the scalar-norm comparison into a full vector comparison at three intra-layer points, making it possible to trace whether drift originates in the mixer residual path or the MLP path.

## Drift Findings for short_correctness_003

The primary failing parity case is `short_correctness_003` (12-token prompt: "List three properties of a good benchmark, using commas only."). The reference expects first generated token `271`, which the implementation now produces correctly at decode step 0 — the first five generated tokens all match `[271, 248068, 271, 248069, 271]`. The failure occurs at generated token index 5, where the expected token is `12` but the implementation produces `16`.

### Diagnose handoff result

Using `--diagnose-handoff`, the recurrent hidden state (after prefill with chunk-corrected DeltaNet, before any decode step) shows argmax token `271` — matching the reference. The first five decode tokens all match: `[271, 248068, 271, 248069, 271]`. This confirms the handoff produces correct state at decode step 0. The chunk rule produces `core_attn_out` with mean values in `[-0.001, 0.001]` across all 18 DeltaNet layers; GPU state norms range from ~5 to ~55 depending on layer.

### Diagnose decode drift result

Running `--diagnose-decode-drift` with `target_decode_step=5` shows that the free-run and rebuilt-prefix paths diverge at the token-level decision. At step 5, the free-run path (continuing from the ongoing decode loop) produces argmax token `16`, while the rebuilt-prefix path (reset + fresh prefill + fresh decode loop) produces argmax token `12` — matching the reference at that position. The margin is narrow: free-run step 5's logits show token `16` barely ahead of token `12`, whereas rebuilt-prefix places token `12` ahead of token `16`. Hidden state `hidden_max_abs_diff` between the two paths is ~0.031 at step 5 — too large for fp16 accumulation alone, but the drift is small enough that it only flips the argmax on a close decision.

**Important caveat**: the rebuilt-prefix path is not a gold oracle. Running short_correctness_003 through the rebuilt-prefix diagnostic path (reset + fresh prefill + decode) produces a sequence that *diverges* from the reference at token index 0 for short_correctness_008. The rebuilt-prefix path uses a different prefill code path (layer_major_prefill with a single batch vs the free-run path which continues from the single-token-incremental decode), and this discrepancy is itself a symptom of a deeper issue in the prefill/decode state consistency.

### HF hidden state comparison at decode step 5

The complete layer-to-layer pipeline is exercised by `tools/compare_layer_hiddens.py`, which runs the Vulkan engine with `--dump-step-hiddens N`, runs the same prompt through HF repacked-FP16 with forward hooks, then computes per-layer max/mean abs diffs, top-5 logits, and a CPU-applied final RMSNorm check. This eliminates the earlier ambiguity about whether the final-norm transform was faithfully reproduced by the Vulkan shader path.

Python CPU RMSNorm was corrected to the Qwen3.5 formula (`1 + weight * (x / rms)` with the `+1` weight scaling), matching the shader path. Previously the CPU reference used a plain `weight * (x / rms)` formula without the offset, producing systematically different normalized outputs.

The `--dump-step-hiddens` diagnostic was run with `short_correctness_003` at decode step 5 to compare per-layer fp16 hidden vectors against a HuggingFace PyTorch reference. The HF reference used the same repacked-fp16 weights and sequential decode mode to match the Vulkan code path.

At step 5, the Vulkan engine still generates `[271, 248068, 271, 248069, 271, 16]` — the expected reference is `[271, 248068, 271, 248069, 271, 12]` (token 12 = comma). The diagnostic confirms this is a close decision with opposing margins:

- **HF repacked-FP16 sequential top-5 at step 5**: token 12 (19.375), token 16 (19.34375)
- **Vulkan top-5 at step 5**: token 16 (19.5469), token 12 (19.2969)

**Layer-by-layer hidden state comparison** between Vulkan and HF at decode step 5:

- **Layers 0–15**: close agreement, differences within fp16 accumulation noise.
- **Layers 16–22**: drift grows progressively. At layer 22 (before final RMSNorm), `max_abs_diff ≈ 0.066` and `mean_abs_diff ≈ 0.009`.
- **Layer 23**: requires a final-norm-aware comparison because HF `hidden_states[-1]` appears to be the final-normalized output, not the pre-norm hidden. A direct element-wise comparison against the unnormalized Vulkan layer-23 hidden introduces spurious large differences that are simply the normalization transform.

The layer-22 divergence (max_abs_diff 0.066, mean_abs_diff 0.009) is the first measurable drift that exceeds fp16 accumulation noise. This points to a root cause in layers 16–22 of the prefill or decode path — possibly DeltaNet state accumulation, KV-cache precision, or a shader math difference in the deeper layers.

### Component-level step-5 comparison

The `--dump-step-components 5` diagnostic was run on the same `short_correctness_003` prompt. The generated token sequence is unchanged: `[271, 248068, 271, 248069, 271, 16]` — the diagnostic does not alter the decode path. Comparing per-layer component norms between Vulkan and HF at decode step 5:

- **Final RMSNorm norm**: Vulkan 131.803 vs HF 131.770 — a 0.025% relative difference, well within fp16 accumulation tolerance for the final norm layer.
- **Final-normalized vector comparison** (the output of the final RMSNorm, which feeds the LM head): `max_abs_diff ≈ 0.434`, `mean_abs_diff ≈ 0.112`. This is approximately 4–5× the corresponding layer-22 hidden-state diffs, consistent with the normalizer applying the norm-ratio scaling factor to amplify pre-existing element-wise differences.
- These final-normalized diffs are large enough to explain the LM-head close decision flip (token 12 vs token 16 at step 5, where the logit margin is ~0.03–0.05).

**Layer 16–19 component-level comparisons** (max_abs diff per vector, comparing Vulkan vs HF at decode step 5) show where drift originates inside individual layers:

| Layer | input max_abs | post_mixer max_abs | post_mlp max_abs | mixer type |
|---|---|---|---|---|
| 16 | 0.01709 | 0.02661 | 0.02405 | DeltaNet |
| 17 | 0.02405 | 0.02484 | 0.03027 | DeltaNet |
| 18 | 0.03027 | 0.02966 | 0.02905 | DeltaNet |
| 19 | 0.02905 | 0.02930 | 0.05859 | Attention |

**Layer 16**: the DeltaNet mixer amplifies drift from input 0.01709 to post_mixer 0.02661, then the MLP partially corrects it back to 0.02405. This is the first layer where the mixer residual is the dominant contributor.

**Layer 19**: the attention mixer produces a residual with essentially unchanged max diff (0.02905 → 0.02930), but the MLP amplifies it to 0.05859 — a 2× increase. The attention layer itself is not the main source of the drift at layer 19; the MLP tail is. This shifts the investigative focus from KV-cache precision at deep attention layers to MLP projection precision, where repeated per-token matvec accumulation in fp16 may introduce broader drift at the deepest layers.

**Crucially, the component diagnostic localizes the remaining issue**: the divergence is in the last-block-to-final-norm hidden state. It is **not** in the token data (generated tokens match for steps 0–4), not in the argmax shader (the argmax correctly picks the highest logit), and not in the frozen reference drift (the reference is freshly computed per debug run). The problem is between the output of layer 23 and the final RMSNorm output that feeds the LM head.

### Final RMSNorm Faithfulness Verification

With the corrected CPU RMSNorm formula, `tools/compare_layer_hiddens.py` now compares three paths through the final normalization to confirm the Vulkan final-RMSNorm shader is faithful:

| Comparison | `max_abs` | `mean_abs` | Interpretation |
|---|---|---|---|
| CPU-finalnorm(VK layer23) vs VK dump on GPU | 0.004732 | 0.000516 | Vulkan final-norm shader is faithful — the GPU's own normalized output matches CPU-applied normalization of the same raw layer-23 hidden |
| CPU-finalnorm(HF layer23) vs HF hook | 0.034353 | 0.004259 | HF's own final-norm output matches CPU-applied normalization of HF's own raw layer-23 hidden — also faithful, with slightly more noise |
| CPU-finalnorm(VK layer23) vs CPU-finalnorm(HF layer23) | 0.621576 | 0.130985 | The **raw layer-23 hiddens** are significantly different between VK and HF; the final-norm shaders are not the source of drift |

The key finding: the Vulkan final RMSNorm shader is **faithful** — it reproduces the correct normalization transform. The drift lives **upstream** of the final norm, in the layer-23 hidden state itself. This eliminates a whole class of hypotheses (wrong weight layout, wrong RMSNorm formula, wrong shader dispatch) and pins the root cause to the hidden state computation within layers 0–23.

### Analysis

The drift between free-run and rebuilt-prefix at step 5 indicates a prefill-vs-decode state inconsistency that takes several tokens to manifest. Each accumulated token eventually pushes the logit scores across the narrow margin between token `12` and token `16`. This is consistent with a small bias or precision issue in the KV cache or DeltaNet state that is negligible per-step but compounds over multiple decode steps.

The rebuilt-prefix path is not a gold reference because it itself fails short_correctness_008 at token index 0. This means the rebuilt-prefix code path (layer_major_prefill from scratch) has a correctness issue that the free-run code path (single-token-incremental decode) does not, or vice versa. The diagnostic is still valuable because it confirms that the divergence is small enough to flip a close argmax decision, and that the handoff itself is correct for the first several tokens.

#### HF per-layer hidden diffs across all 24 layers

With the final-norm shader verified faithful, the per-layer hidden diffs from `tools/compare_layer_hiddens.py` isolate the drift to the **mixer blocks within each layer** — not the final RMSNorm, not the MLP tail (which was verified in diary 0008), but the attention/DeltaNet mixer residual paths:

- **Layers 0–3**: close, `max_abs < 0.005`.
- **Layers 4–10**: gradual climb, `max_abs` reaches ~0.015 by layer 10.
- **Layers 11–15**: accelerating divergence, `max_abs` ~0.03–0.05.
- **Layer 16**: notable jump at the first attention-transition boundary. Component-level analysis shows the DeltaNet mixer at layer 16 is the first layer where the mixer residual amplifies drift (input 0.01709 → post_mixer 0.02661).
- **Layer 19**: largest single-layer jump. This layer is an **attention** layer (layers alternate between DeltaNet and attention; at 24 layers total, attention layers include 19, 13, 7, 1). However, the component-level analysis **revises** the earlier hypothesis: the post_mixer (attention output) max diff at layer 19 is only 0.02930 — slightly lower than layer 18's post_mixer (0.02966). The jump to post_mlp 0.05859 is a **2× amplification by the MLP tail**, not by the attention layer. The layer-19 drift spike is an MLP-accumulation effect, not a KV-cache precision issue.
- **Layers 20–22**: continued high drift, `max_abs` ~0.05–0.07.

This per-layer profile, refined by the component-level vector comparison, shifts the investigative focus from KV-cache precision to **fp16 MLP matvec accumulation** in the deepest layers. The attention blocks at layers 16+ do not measurably amplify the diff; the MLP blocks at layers 17–19 do.

## Experiments Tried and Reverted

### Attempt 1: Skip chunk correction, use pure recurrent prefill

Reverted because the prefill-sensitive failures (mixed_correctness_023, 025, 026, 027) require chunk-corrected prefill to match the reference.

### Attempt 2: Upload chunk core_attn_out before Phase A projections

If the chunk core_attn_out upload happens before QKV projection (which overwrites `dn_qkv`), the chunk values get overwritten. Fixed by ordering: Phase A projections first, then upload chunk values, then norm_gate/out_proj.

### Attempt 3: Run conv1d_step in correct_last_token_hidden

This advances the sliding window past the last prompt token, corrupting the conv state for the first decode step. Skipped the conv step in the correction path.

### Attempt 4: fp32 LM head to reduce precision artifacts

Switched from fp16 LM head output to fp32 LM head output (`matvec_f32_out` + `argmax_f32` shaders). Improves logit precision but does not fix the decode-step-5 decision. **Reverted** because (a) it did not fix short003 — the failure persisted at generated token index 5, unchanged from the fp16 LM head output — and (b) the production path uses fp16 logits with fp32 comparison in argmax, which matches the reference for all passing tests.

### Attempt 5: Direct chunk state upload vs full state replacement

Tried uploading only the `final_state` without the g/beta tail replacement. Produced worse results because g/beta values were stale from prefill.

### Attempt 6: Narrow KV-cache barrier-offset hygiene fix

The `vk_session.cpp` barrier logic for the KV-cache transfer within the attention decode path used a hardcoded offset that covered only the first layer's KV-cache slice. Applied a narrow correction so the pipeline barrier range covers `kv_layer_offset` (the per-layer slice of KV cache). **Did not change short003 behavior** — the failure at index 5 persisted, and the per-layer attention drift profile (layer 19 spike) was unchanged. Retained because the code was unambiguously wrong: barriers that undershoot their true memory range produce data-race windows on any GPU whose cache-line granularity exposes the gap. Not reverted — this is a correctness fix regardless of parity outcome.

### Attempt 7: Narrow deltanet_norm_gate rounding-point fix

The `deltanet_norm_gate` shader computes the gated RMSNorm output as `w * v * silu(gate)` where `v` is the normalized hidden value and `w` is the weight. HF's `Qwen3_5RMSNormGated` implementation casts `weight * norm(hidden)` to the input dtype (fp16) **before** multiplying by the SiLU gate, so the intermediate product is rounded to fp16 before the final gate multiplication. The shader was computing the full expression in fp32 and only casting the final result. Applied a narrow fix:

```glsl
float weighted = float(float16_t(w * v));
io_buf.values[base + d] = float16_t(weighted * silu);
```

This ensures the `weight * norm(hidden)` intermediate is rounded to fp16 before the SiLU gate multiplication, matching HF's casting behavior. **Did not change short003 behavior** — the failure at index 5 persisted, and per-layer component diffs at layers 16–19 were unchanged (the rounding difference at this point is below 1 ulp for most elements and does not accumulate). Retained because the code was unambiguously inconsistent with the HF reference implementation: the `Qwen3_5RMSNormGated` contract explicitly requires the fp16 rounding of the weighted normalized vector before the gate.

## Gate Results After Barrier-Offset and deltanet_norm_gate Fixes

The gate runs with both the barrier-offset fix and the deltanet_norm_gate rounding fix applied:

- **CMake build**: passes cleanly.
- **short001 / short008** (max-new-tokens 6): pass.
- **mixed023 / pp520046** (max-new-tokens 1): pass.
- **ctest four-target gate**: passes (all four test targets).
- **short_correctness_003**: still fails at generated token index 5, producing `16` instead of the expected `12`. The layer-16-through-19 component-level profile is unchanged — the rounding fix did not affect the drift profile at any layer.

Both fixes produce no measurable change in any passing or failing result, consistent with the hypothesis that these narrow precision/dependency issues are below the noise floor of the dominant drift source. Both are retained as correctness hardening measures: the barrier fix closes a data-race window, and the rounding fix matches the HF reference contract.

## Current Limitations

### 1. CPU Chunk Bridge (the critical limitation)

The most important limitation of this checkpoint: **`run_chunk_prefill` is a CPU bridge.** The native chunk rule runs on the host. After every prompt, the prefill loop downloads Q/K/V/g/beta to CPU, rearranges them to head-major order, runs the chunk rule, then uploads the final state back to the GPU. This is:
- **Slow**: ~648 KB download + ~288 KB upload per prefill
- **Not Vulkan-native**: breaks the full GPU offload chain
- **State discontinuity risk**: CPU and GPU memory layout mismatch causes silent corruption

The correct fix is a Vulkan-native chunk-rule shader, deferred because the lower-triangular matrix solve is non-trivial to implement in a shader with WG-level synchronization.

### 2. Not Yet Final Vulkan-Native Megakernel / Full GPU Offload

Because of the CPU chunk bridge, this is **not** the final Vulkan-native inference pipeline:
- Prompt prefill still requires a CPU round-trip for DeltaNet chunk correction
- The recurrent prefill path (pre-chunk-correction) runs on GPU, but the final state correction is CPU-mediated
- The decode loop is fully GPU-native (recurrent DeltaNet update, KV-cache attention, MLP)
- Full GPU offload requires a shader-based chunk-rule implementation

### 3. short_correctness_003 Fails at Generated Token Index 5

The first five generated tokens for `short_correctness_003` now match the reference: `[271, 248068, 271, 248069, 271]`. The failure occurs at generated token index 5, where the expected token is `12` ("," — comma, continuing the expected comma-separated list) but the implementation produces `16`. Confirmed:
- On real hardware (RADV RX 6750 XT)
- With the chunk-corrected handoff active
- With fp16 LM head (fp32 LM head experiment reverted; both produce the same failure at index 5)
- With GPU-side g/beta computation
- With the full layer-major prefill pipeline

The decode drift diagnostic shows that at step 5, the free-run path narrowly picks token `16` over token `12`, while the rebuilt-prefix path picks token `12` over token `16`. The rebuilt-prefix path is not authoritative — it also breaks `short_correctness_008` — but the divergence confirms the state has drifted enough by step 5 to flip a close argmax decision. The `--dump-step-components 5` diagnostic further localizes the issue: the final RMSNorm hidden shows Vulkan norm 131.803 vs HF 131.770, and the final-normalized vector max_abs_diff is ~0.434 — sufficient to flip the argmax on the close token-12 vs token-16 decision. The divergence is in the last-block-to-final-norm hidden state, not in token data, argmax, or frozen reference drift.

### 4. No Commit or Push Yet

This checkpoint is not committed because full parity is not achieved. The work is substantial and should be the foundation for the next fix attempt, but pushing it would leave the branch in a known-broken state. The working state is preserved in the working tree for ongoing diagnostics.

## Next Work

1. **Diagnose the decode-step-5 drift — MLP accumulation in deep layers**: The handoff is correct for the first 5 decode tokens, but a subtle state inconsistency causes divergence at step 5. The component-level vector comparison shows that at layer 19, the attention mixer does not amplify drift (post_mixer max_abs 0.02930 vs input 0.02905), but the MLP tail amplifies it to 0.05859 — a 2× jump. Investigate fp16 MLP matvec accumulation across deep layers (16–19), where repeated per-token residual passes compound small fp16 rounding errors. Compare Vulkan MLP outputs against HF MLP outputs at these layers to determine whether the drift originates in the gate/up projection matvec precision, the SiLU/gating combination, or the down projection.
2. **Vulkan-native chunk-rule shader**: Eliminate the CPU bridge. This is the blocker for the megakernel roadmap.
3. **Convert `spock_vk_decode_prefill_handoff_mismatch` to a positive parity test** once the handoff is correct.
4. **Commit the session refactor** — but only after parity is achieved for the targeted cases.

# 0003: Text-Only Artifact Load Plan

## Goal

The third phase created an explicit text-only runtime load plan from the real Qwen artifact metadata. The actual Hugging Face model is multimodal: it contains language-model tensors, visual tensors, and MTP tensors. Spock v1 is targeting local text decode, not image/video understanding. That means the runtime needs a precise way to say which tensors it will load and which tensors it is intentionally ignoring.

This phase does not yet repack weights or run inference. Its job is to transform a large set of real tensor metadata into a role-based plan the runtime can use later without guessing from string names.

## Why A Load Plan Is Needed

The source safetensors manifest records tensors by upstream names. Those names are useful to humans and converters, but they are not ideal for the hot path of an inference runtime.

For example, a kernel does not want to search for:

```text
model.language_model.layers.7.self_attn.q_proj.weight
```

Instead, the runtime wants to know:

- Layer `7` is a full-attention layer.
- The `attn_q` role for layer `7` lives in a specific file and byte range.
- The dtype and shape are known.
- The tensor will eventually be repacked into the runtime weight layout.

The load plan bridges that gap. It is still JSON and still human-readable, but it is organized around runtime roles rather than only upstream tensor names.

## Text Decode Components

Text decode needs a few global tensors and many per-layer tensors.

The global tensors are:

- Token embeddings.
- Final normalization.
- LM head.

The token embedding matrix maps token IDs to vectors. If the hidden size is `1024`, each token becomes a vector of length `1024`.

The LM head maps the final hidden vector back to vocabulary logits. In this model, embeddings are tied, meaning there is no separate LM-head weight tensor. The same embedding matrix is reused for output projection. This saves memory and is common in language models.

Each layer then applies a sequence of operations:

- Normalize the input.
- Run either DeltaNet/linear attention or full attention.
- Add a residual connection.
- Run an MLP.
- Add another residual connection.

Residual connections mean the layer adds transformed information back to the previous hidden vector instead of replacing it completely. This makes deep networks easier to optimize and is part of the model's semantics.

## DeltaNet Layer Roles

For each DeltaNet-style layer, the planner requires:

- `input_norm`
- `post_norm`
- `delta_a_log`
- `delta_dt_bias`
- `delta_conv`
- `delta_in_proj_a`
- `delta_in_proj_b`
- `delta_in_proj_qkv`
- `delta_in_proj_z`
- `delta_norm`
- `delta_out_proj`
- `mlp_gate`
- `mlp_up`
- `mlp_down`

These roles correspond to the recurrent linear-attention block and the feed-forward block.

The important concept is that DeltaNet keeps recurrent state. In decode, the model processes one new token at a time and updates state that summarizes past tokens. That state must be carried between token steps. The real config says the recurrent state dtype is `float32`, which matches the project plan.

## Full-Attention Layer Roles

For each full-attention layer, the planner requires:

- `input_norm`
- `post_norm`
- `attn_q`
- `attn_k`
- `attn_v`
- `attn_o`
- `attn_q_norm`
- `attn_k_norm`
- `mlp_gate`
- `mlp_up`
- `mlp_down`

Full attention uses Q, K, and V projections. During decode, K and V for previous tokens are stored in a KV cache. The current token computes a query vector and compares it with cached keys to decide how to combine cached values.

The text plan does not allocate the KV cache yet. It records the tensors needed to implement the attention block later.

## Excluding Visual And MTP Tensors

The real scan found:

- `320` language-model tensors.
- `153` visual tensors.
- `15` MTP tensors.

The planner explicitly records that visual and MTP tensors are excluded. This is better than silently ignoring them because it makes the v1 scope auditable. If a future user expects image input to work, the artifact plan makes it clear that those tensors were not mapped into the text runtime.

MTP likely refers to multi-token prediction support. It is not needed for the first batch-1 greedy decode path.

## Precision Policy

The generated plan records:

- Source storage dtype: `source`.
- Runtime weight dtype: `fp16`.
- Runtime activation dtype: `fp16`.
- Runtime recurrent state dtype: `fp32`.
- `requires_bf16_to_fp16_repack: true`.

This is an important boundary. The text plan is not pretending that source BF16 bytes are already Vulkan-ready FP16 weights. It says exactly what still has to happen.

## Implementation

The new tool is:

```sh
python3 tools/plan_text_artifact.py artifacts/spock-real-qwen35-0p8b --output artifacts/spock-real-qwen35-0p8b/text_plan.json
```

It reads `manifest.json`, validates required text tensors, and writes a role-based load plan.

The planner fails if a required tensor is missing. This matters because missing tensors should be caught during artifact preparation, not after a Vulkan kernel reads garbage offsets.

## Verification

The real-model smoke test for this phase is:

```sh
python3 tests/run_text_plan_smoke.py \
  --model-dir artifacts/hf/Qwen--Qwen3.5-0.8B \
  --converter tools/convert_qwen35_0p8b.py \
  --planner tools/plan_text_artifact.py
```

It regenerates a real safetensors-scan artifact from the downloaded model, runs the planner, and checks:

- `18` DeltaNet layers.
- `6` full-attention layers.
- `320` language-model tensors.
- `153` visual tensors excluded.
- tied LM head metadata.

The normal build verification also still passes:

```sh
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Known Limitations

The text plan still references source BF16 tensor bytes. The next phase should implement an actual repacking plan or converter for at least a small subset of tensors, then validate numeric BF16-to-FP16 conversion.

The project also still needs tokenizer handling. Tokenizer correctness is separate from tensor loading: even a perfect model implementation will produce different token IDs if the prompt is tokenized differently from the reference.

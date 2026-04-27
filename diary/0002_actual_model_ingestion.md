# 0002: Actual Qwen 3.5 0.8B Artifact Ingestion

## Goal

The second phase tested Spock against the actual `Qwen/Qwen3.5-0.8B` model files from Hugging Face. This is a major step beyond a scaffold because model plans can be wrong. The real files are the source of truth for layer names, tensor dtypes, tensor shapes, vocabulary size, and configuration details.

The target revision used in this phase was:

```text
2fc06364715b967f1860aea9cf38778875588b17
```

The downloaded files were:

- `config.json`
- `model.safetensors.index.json`
- `model.safetensors-00001-of-00001.safetensors`

The safetensors weight file is about `1.7 GiB`, so the local `artifacts/` directory is ignored by git.

## What Safetensors Is

LLM weights are large arrays of numbers. A tensor has:

- A name, such as `model.language_model.layers.0.mlp.up_proj.weight`.
- A shape, such as `[3584, 1024]`.
- A dtype, such as BF16 or F32.
- A byte range inside a file.

Safetensors is a file format that stores tensor metadata in a JSON header followed by raw tensor bytes. The format is useful for this project because the header can be read without importing PyTorch. That means Spock can inspect the real model even in a minimal Python environment.

The local Python environment had no `torch`, `transformers`, `safetensors`, `huggingface_hub`, or `numpy`. Instead of installing a full ML stack, the converter was extended to parse safetensors headers directly using the Python standard library.

## Why Header Scanning Is Useful

Header scanning does not perform numerical conversion. It does not read BF16 values into arrays, run matrix multiplication, or evaluate the model. It does, however, prove several important things:

- The model files are present and readable.
- The manifest can reference actual tensor byte ranges.
- Tensor names, shapes, dtypes, and offsets can be recorded.
- File hashes can be checked.
- The model config can be compared with the implementation plan.

This is a practical intermediate step. It gives the runtime real metadata before the project has implemented BF16-to-FP16 conversion or weight-backed inference.

## Converter Changes

`tools/convert_qwen35_0p8b.py` now has a mode:

```sh
python3 tools/convert_qwen35_0p8b.py \
  --safetensors-scan \
  --input artifacts/hf/Qwen--Qwen3.5-0.8B \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --output artifacts/spock-real-qwen35-0p8b \
  --force
```

This mode:

- Finds `.safetensors` files in the model directory.
- Reads the safetensors JSON header.
- Hard-links or copies the original weight file into the Spock artifact directory.
- Records each tensor's file, offset, byte length, dtype, shape, and source name.
- Reads `config.json` when present.
- Derives the layer schedule from `text_config.layer_types`.
- Produces a `tensor_summary`.

The artifact uses `storage_dtype: source`, because it keeps the original model bytes. That is intentionally different from the future production path, which should repack weights to FP16 for this AMD Vulkan target.

## Validator Changes

`tools/validate_artifact.py` now accepts:

- `source` storage dtype.
- `bf16`, `f16`, and `f32` tensor dtypes.
- `tensor_summary` consistency checks.

The real artifact was validated with hashes:

```sh
python3 tools/validate_artifact.py artifacts/spock-real-qwen35-0p8b --check-hashes --json
```

This verifies that the manifest's recorded SHA-256 matches the actual weight file.

## Actual Model Findings

The real model scan produced:

- `488` tensor entries.
- `452` BF16 tensors.
- `36` F32 tensors.
- `320` language model tensors.
- `153` visual tensors.
- `15` MTP tensors.

The weight file SHA-256 is:

```text
04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696
```

The real config confirmed the text stack:

- `24` text layers.
- Hidden size `1024`.
- Intermediate size `3584`.
- Full attention interval `4`.
- Layer pattern: `linear_attention, linear_attention, linear_attention, full_attention`, repeated.
- Full attention heads `8`.
- KV heads `2`.
- Head dimension `256`.
- Linear attention key heads `16`.
- Linear attention value heads `16`.
- Linear key/value dimensions `128`.
- Linear convolution kernel `4`.
- Recurrent state dtype `float32`.

This validates the core architecture assumptions in the implementation plan.

## Important Surprise: The Model Is Multimodal

The Hugging Face model is not just a text-only artifact. It includes:

- Language model tensors.
- Visual tensors.
- MTP tensors.

The config includes `vision_config` and image/video token IDs. The implementation plan focuses on local text decode, so Spock must either:

- Ignore visual tensors for the text-only runtime while preserving artifact validity.
- Or produce a text-only derived artifact later.

For now, the scan records all tensors and counts groups by name prefix. This avoids silently throwing away data before the loader policy is explicit.

## Important Precision Fact: BF16 Source Weights

The official model is BF16-heavy. BF16 means bfloat16, a 16-bit floating-point format with an 8-bit exponent and 7-bit mantissa. It has a range similar to FP32 but less precision.

The target RX 6750 XT Vulkan stack does not expose the native BF16 path assumed by some CUDA implementations. The plan's production path is therefore:

- FP16 weights.
- FP16 activations.
- FP32 critical accumulations and recurrent state.

The current real artifact keeps source BF16 bytes. It does not convert them to FP16. That is correct for this phase because conversion must be implemented carefully. BF16-to-FP16 conversion can change values, overflow in edge cases, and affect token parity. It needs its own validation gate.

## Why This Still Counts As Actual-Model Progress

Even without numerical inference, this phase closes several risks:

- The exact revision is pinned.
- The real file list is known.
- The real architecture matches the plan.
- The real tensor count and dtype distribution are known.
- The artifact system can represent real model bytes.
- Hash validation works.

Without this phase, the project might start writing kernels for tensor names or shapes that do not exist in the real model.

## Actual-Model Smoke Test

A new script verifies the real-model scan:

```sh
python3 tests/run_actual_model_smoke.py \
  --model-dir artifacts/hf/Qwen--Qwen3.5-0.8B \
  --converter tools/convert_qwen35_0p8b.py \
  --validator tools/validate_artifact.py
```

It runs the converter, runs the validator with hashes, loads the generated manifest, and checks that the scan found the expected scale of tensors and BF16 weights.

The successful summary was:

```json
{
  "dtype_counts": {
    "bf16": 452,
    "f32": 36
  },
  "language_model_tensors": 320,
  "mtp_tensors": 15,
  "total": 488,
  "visual_tensors": 153
}
```

## Verification

This phase was verified with:

```sh
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
python3 tests/run_actual_model_smoke.py \
  --model-dir artifacts/hf/Qwen--Qwen3.5-0.8B \
  --converter tools/convert_qwen35_0p8b.py \
  --validator tools/validate_artifact.py
```

The CTest suite passed, and the actual model smoke test passed.

## Known Limitations

The runtime still does not execute the model. The actual-model artifact is metadata-valid and hash-valid, but decode still requires:

- BF16-to-FP16 repacking or direct BF16 handling.
- A weight loader that maps manifest tensors to runtime buffers.
- A real CPU reference implementation.
- Tokenizer handling.
- Layer-by-layer Vulkan kernels.
- Correct KV cache and DeltaNet recurrent-state updates.

The next technically useful phase is probably a text-only artifact planner: map the `model.language_model.*` tensors into the exact buffers the runtime will need, and explicitly exclude or defer visual/MTP tensors for v1 text decode.

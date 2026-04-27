# 0004: Text Weight Repack

## Goal

The fourth phase converts the text-only load plan into a packed runtime weight artifact. The previous phase proved which real tensors belong to text decode. This phase takes the next step: read those real tensor byte ranges and write a new contiguous `text_weights.bin` file designed for the runtime.

This is the first phase that transforms actual model weight bytes. It still does not run inference, but it creates the kind of buffer a Vulkan runtime will eventually upload to GPU memory.

## Why Repacking Exists

Hugging Face model files are designed for framework loaders. They preserve upstream tensor names and source dtypes, and they are general enough for many runtimes. A specialized Vulkan inference engine wants a different shape:

- A small number of large contiguous buffers.
- Explicit offsets for every tensor role.
- Predictable alignment.
- Dtypes chosen for the target GPU.
- No runtime string lookup in the hot path.

Vulkan shaders operate on buffers and byte offsets. They do not know about Python dictionaries, safetensors headers, or PyTorch modules. Repacking turns source model files into a runtime-friendly layout.

## BF16 And FP16

The official Qwen artifact is mostly BF16. BF16 and FP16 are both 16-bit floating-point formats, but they allocate bits differently.

BF16 keeps the same exponent size as FP32. This gives it a wide numeric range but relatively few mantissa bits. FP16 has a smaller exponent and more mantissa bits. It can represent more detail near ordinary values, but it has a smaller range.

The RX 6750 XT Vulkan target does not expose native BF16 support in the expected stack. The plan therefore uses:

- FP16 for most weights.
- FP16 for activations.
- FP32 for critical recurrent state and selected sensitive tensors.

The repacker converts BF16 source tensors to IEEE FP16. It preserves selected F32 tensors as FP32.

## Which F32 Tensors Are Preserved

The text load plan showed two F32 role classes:

- `delta_a_log`
- `delta_norm`

These are part of the DeltaNet/linear-attention machinery. The implementation preserves them as FP32 instead of forcing them into FP16. This follows the project's precision strategy: use FP16 where it is expected to be efficient, but keep numerically sensitive recurrent-related values in FP32 until parity data says it is safe to lower precision.

The current policy is intentionally conservative. A later tuning phase can experiment with converting more tensors to FP16, but correctness should come first.

## Streaming Conversion

The repacker is implemented in `tools/repack_text_weights.py`. It uses only Python's standard library.

For each tensor role in the text plan, it:

- Opens the source safetensors file.
- Seeks to the tensor byte range.
- Converts BF16 chunks to FP16 when needed.
- Copies F32 chunks unchanged when the role is preserved.
- Writes the result to `text_weights.bin`.
- Records the output offset, size, dtype, shape, and source metadata.

The conversion is streaming. It does not load the full model into memory. This matters because model files are large, and future models may be much larger.

The Python repacker is primarily a reference implementation. It is easy to read and useful for small tests, but the real model exposed a performance problem: converting the full text artifact in pure Python is too slow.

The production bring-up path therefore gained a native helper:

```sh
python3 tools/export_repack_tasks.py artifacts/spock-real-qwen35-0p8b/text_plan.json --output artifacts/spock-real-qwen35-0p8b/text_repack_tasks.tsv
./build/spock-repack-text --tasks artifacts/spock-real-qwen35-0p8b/text_repack_tasks.tsv --source-root artifacts/spock-real-qwen35-0p8b --output-dir artifacts/spock-text-repack-qwen35-0p8b
```

The task exporter flattens the JSON plan into a simple TSV file. The C++ repacker reads that file, streams source tensor ranges, converts BF16 to FP16, and writes the packed output. This avoids adding a JSON dependency to the C++ scaffold while still giving the real-model path native performance.

## Alignment

Every packed tensor starts at an aligned byte offset. The current default alignment is `256` bytes.

Alignment is not just neatness. GPU memory accesses are fastest when data is arranged in predictable boundaries. Future Vulkan kernels can rely on these offsets when issuing vectorized loads or subgroup-cooperative loads.

Padding bytes between tensors have no semantic meaning. Kernels must not read them as model values.

## Output Files

The repacker writes:

- `text_weights.bin`
- `text_repack_manifest.json`

The manifest records:

- packed file SHA-256
- tensor role paths
- source tensor names
- source offsets and dtypes
- output offsets and dtypes
- tensor shapes
- summary counts

The validator `tools/validate_text_repack.py` checks the structure, ranges, alignment, dtype policy, and optionally the packed file hash.

## Why This Still Is Not Inference

Repacked weights are necessary but not sufficient for inference.

To generate real tokens, Spock still needs:

- Tokenizer support to convert prompts into token IDs.
- A CPU reference that implements the real Qwen math.
- A runtime loader that uploads `text_weights.bin` into Vulkan buffers.
- Kernels for RMSNorm, projections, DeltaNet recurrence, attention, MLP, and LM-head argmax.
- State allocation for KV cache and DeltaNet recurrence.

This phase gives those later components a concrete binary input format.

## Verification

The fast synthetic repack test is:

```sh
python3 tests/run_text_repack_unit.py \
  --repacker tools/repack_text_weights.py \
  --validator tools/validate_text_repack.py
```

It checks BF16-to-FP16 conversion for simple values, F32 preservation, alignment, and manifest validation.

The real-model repack command is:

```sh
python3 tools/repack_text_weights.py \
  artifacts/spock-real-qwen35-0p8b/text_plan.json \
  --output-dir artifacts/spock-text-repack-qwen35-0p8b \
  --force

python3 tools/validate_text_repack.py \
  artifacts/spock-text-repack-qwen35-0p8b \
  --check-hashes \
  --json
```

The real native repack produced:

```json
{
  "fp16_tensors": 284,
  "fp32_tensors": 36,
  "size_bytes": 1504798720,
  "tensor_count": 320
}
```

The normal project verification remains:

```sh
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Known Limitations

The standard-library Python converter is correct enough for bring-up, but it is not optimized. The native helper solves the immediate real-model conversion problem, but the project should still consolidate the Python and C++ paths so the reference and production converters share test vectors and dtype policy definitions.

The current repack layout is also simple. It preserves role order and alignment but does not yet fuse QKV, fuse MLP gate/up, reorder matrices for subgroup-friendly access, or pretranspose weights for Vulkan kernels. Those are later layout-optimization milestones.

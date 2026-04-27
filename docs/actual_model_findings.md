# Actual Model Findings

Validated against `Qwen/Qwen3.5-0.8B` revision `2fc06364715b967f1860aea9cf38778875588b17`.

The downloaded Hugging Face artifact contains:

- `config.json`
- `model.safetensors.index.json`
- `model.safetensors-00001-of-00001.safetensors`

Real safetensors scan result:

- Tensor entries: `488`
- Tensor dtypes: `452` BF16 tensors and `36` F32 tensors
- Weight file SHA-256: `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`
- Weight file size: `1746942600` bytes

The text stack matches the intended hybrid schedule:

- `24` text layers
- Layer pattern: `linear_attention, linear_attention, linear_attention, full_attention`, repeated
- Hidden size: `1024`
- Intermediate size: `3584`
- Full attention heads: `8`
- KV heads: `2`
- Head dim: `256`
- Linear attention key/value heads: `16`
- Linear attention key/value dim: `128`
- Linear conv kernel: `4`

Important implementation consequence:

The official artifact is multimodal and BF16-heavy. On this RADV target, the production path still needs a real BF16-to-FP16 repacking step before optimized Vulkan decode. The current `--safetensors-scan` path proves that Spock can ingest and validate the actual model bytes and tensor metadata without PyTorch, but it intentionally does not convert BF16 tensor values.

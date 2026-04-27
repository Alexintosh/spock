# Packed Model Artifact Format

This document defines the offline Vulkan-friendly artifact contract for `Qwen/Qwen3.5-0.8B`. The format replaces runtime pointer packing with explicit offsets into aligned buffers.

## Goals

- Preserve effective weights from the pinned upstream model.
- Load with no tensor-name inference in the hot path.
- Use explicit offsets instead of host pointers.
- Support `fp16` production weights and optional `fp32` shadow tensors for parity checks.
- Keep layout versioned so benchmark results identify exactly what was run.

## Artifact Set

An artifact directory contains:

- `manifest.json`: metadata, versions, tensor index, checksums, and layout.
- `weights.bin`: packed production tensor data.
- `shadow_fp32.bin`: optional fp32 tensor data for reference validation.
- `tokenizer/`: tokenizer files copied from the pinned model revision.

## Manifest Contract

Required manifest fields:

```json
{
  "artifact_format_version": 1,
  "model_id": "Qwen/Qwen3.5-0.8B",
  "model_revision": "<pinned-revision>",
  "converter": {
    "name": "convert_qwen35_0p8b.py",
    "version": "<converter-version>"
  },
  "precision": {
    "weights": "fp16",
    "activations": "fp16",
    "recurrent_state": "fp32"
  },
  "alignment_bytes": 256,
  "buffers": [],
  "tensors": [],
  "layers": [],
  "checksums": {}
}
```

## Buffer Records

Each buffer record must include:

- `name`
- `path`
- `size_bytes`
- `alignment_bytes`
- `sha256`

## Tensor Records

Each tensor record must include:

- `name`: canonical upstream tensor name or derived fused tensor name.
- `buffer`: owning buffer name.
- `offset_bytes`: byte offset from the start of the buffer.
- `size_bytes`: packed byte size.
- `dtype`: stored dtype.
- `shape`: logical shape before packing.
- `layout`: logical memory layout such as `row_major`, `fused_qkv`, or `fused_gate_up`.
- `source_tensors`: upstream tensor names used to produce this tensor.
- `transform`: conversion operation, for example `identity`, `transpose`, `concat`, or `interleave`.
- `sha256`: checksum of the packed tensor byte range.

Offsets must be aligned to `alignment_bytes`. Padding bytes are not semantically meaningful and must not be read by kernels.

## Layer Records

Each layer record must include all offsets needed by the runtime without name lookup:

- layer index
- norm tensor offsets
- attention QKV projection offsets
- attention output projection offsets
- MLP gate/up fused offsets
- MLP down projection offsets
- DeltaNet projection offsets
- recurrent-state layout metadata
- scratch-size requirements

## Text-Only Load Plan

The official `Qwen/Qwen3.5-0.8B` artifact is multimodal. Spock v1 targets text decode only, so conversion is split into two phases:

- A source artifact manifest that records all real tensors.
- A text-only load plan that maps `model.language_model.*` tensors into runtime roles and explicitly excludes `model.visual.*` and `mtp.*` tensors.

Generate the text plan with:

```sh
python3 tools/plan_text_artifact.py artifacts/spock-real-qwen35-0p8b --output artifacts/spock-real-qwen35-0p8b/text_plan.json
```

The text plan includes:

- token embedding tensor
- tied LM head metadata
- final RMSNorm tensor
- per-layer role mappings for DeltaNet and full-attention layers
- excluded tensor counts
- precision policy indicating that BF16 source weights still require FP16 repacking

## Text Repack Artifact

After generating a text plan, create a runtime-oriented text weight buffer with:

```sh
python3 tools/repack_text_weights.py artifacts/spock-real-qwen35-0p8b/text_plan.json --output-dir artifacts/spock-text-repack-qwen35-0p8b --force
python3 tools/validate_text_repack.py artifacts/spock-text-repack-qwen35-0p8b --check-hashes --json
```

For the real model, prefer the native repacker:

```sh
python3 tools/export_repack_tasks.py artifacts/spock-real-qwen35-0p8b/text_plan.json --output artifacts/spock-real-qwen35-0p8b/text_repack_tasks.tsv
./build/spock-repack-text --tasks artifacts/spock-real-qwen35-0p8b/text_repack_tasks.tsv --source-root artifacts/spock-real-qwen35-0p8b --output-dir artifacts/spock-text-repack-qwen35-0p8b
python3 tools/validate_text_repack.py artifacts/spock-text-repack-qwen35-0p8b --json
```

The repack artifact contains:

- `text_weights.bin`: aligned packed runtime tensor bytes
- `text_repack_manifest.json`: role-to-offset metadata

The current precision policy converts BF16 source tensors to FP16 and preserves `delta_a_log` and `delta_norm` tensors as FP32.

## Parity Checks

Artifact conversion passes only if:

- Every required upstream tensor is accounted for exactly once or intentionally fused.
- Every fused tensor lists its source tensors and transform.
- Sample tensor values match the upstream model after inverse transform.
- Checksums are stable across repeated conversion with the same inputs.
- Loading the artifact reconstructs the same effective weights used by the CPU reference.

## Versioning

Any change to tensor order, alignment, dtype, fusion, transform, or layer metadata increments `artifact_format_version`. Benchmark files must record this version.

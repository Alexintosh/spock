# 0084: Persistent MLP Micro-Probe -- External FP16 Input File

## Goal

Extend `vk_persistent_mlp_probe` so its input vector can come from an external raw fp16 vector file. This creates a clean handoff point for real hidden-state captures from the existing Vulkan decode diagnostics, without taking on RMSNorm fusion yet.

## Background

The persistent MLP probe (diaries 0080-0083) validated gate/up/SiLU-activation/down projection, residual update, and real embedding input. Two input modes exist: the default synthetic 1..8 pattern and `--input-token ID` which loads from the repacked artifact's embedding table.

Neither mode allows arbitrary fp16 vectors to be injected. The decode diagnostics already produce per-step hidden-state tensors (RMSNorm output, residual stream, attention/DeltaNet output) as fp16 buffers. A raw fp16 file input lets those captures be fed directly into the MLP probe for isolated correctness testing, bridging the gap between the full decode pipeline and the persistent MLP probe without composing a complete layer.

## Implementation

The app now accepts:

```
--input-fp16-file PATH
```

When `--input-fp16-file PATH` is provided:

1. The file is opened in binary mode. It must contain at least `hidden * sizeof(uint16_t)` bytes.
2. Exactly the first `hidden` fp16 values are read into `input_data`.
3. Sets `use_fp16_file_input = true` for JSON output.

Mutual exclusivity:

- `--input-fp16-file` and `--input-token` cannot both be provided. If both are set, the app returns a JSON error with exit code 2.
- `--input-fp16-file` does not require `--repack-dir`. It works with both synthetic and real weights.

The file format is raw little-endian IEEE 754 binary16 (`float16`), no header, no padding, no alignment requirement. Values are read sequentially into the input vector.

A test fixture `tests/data/mlp_input_pattern_128.fp16` contains 128 fp16 values matching the existing synthetic pattern `(c % 8) + 1` for `c in 0..127`. This proves that loading the same values through the file path produces identical results to the in-process synthetic generation.

## Verification

### Non-regression: synthetic default (unchanged)

```
build/vk_persistent_mlp_probe --hidden 128 --intermediate 16 --output-rows 8 --workgroups 8
```

Result: status ok, checksum 371183224, expected_checksum 371183224, failures 0, arrived 0, generation 2.

No `input_fp16_file` field appears in JSON when the option is not provided.

### FP16 file input matching synthetic pattern

```
build/vk_persistent_mlp_probe --hidden 128 --intermediate 16 --output-rows 8 --workgroups 8 --input-fp16-file tests/data/mlp_input_pattern_128.fp16
```

Result:

```json
{
  "hidden": 128,
  "intermediate_count": 16,
  "output_rows": 8,
  "workgroups": 8,
  "real_weight": false,
  "input_fp16_file": "tests/data/mlp_input_pattern_128.fp16",
  "status": "ok",
  "failures": 0,
  "arrived": 0,
  "generation": 2,
  "expected_generation": 2,
  "checksum": 371183224,
  "expected_checksum": 371183224
}
```

Checksum 371183224 matches the synthetic default exactly, confirming the file path produces bit-identical input.

### Mutual exclusivity error

```
build/vk_persistent_mlp_probe --hidden 128 --intermediate 16 --output-rows 8 --workgroups 8 --input-fp16-file tests/data/mlp_input_pattern_128.fp16 --input-token 0
```

Result: exit 2 with message `--input-fp16-file and --input-token are mutually exclusive`.

### Validation error: file too small

```
build/vk_persistent_mlp_probe --hidden 128 --intermediate 16 --output-rows 8 --workgroups 8 --input-fp16-file /dev/null
```

Result: exit 2 with message `--input-fp16-file too small: need 128 x 2 = 256 bytes, got 0`.

### Validation error: file not found

```
build/vk_persistent_mlp_probe --hidden 128 --intermediate 16 --output-rows 8 --workgroups 8 --input-fp16-file nonexistent.fp16
```

Result: exit 2 with message `cannot open --input-fp16-file: nonexistent.fp16`.

### CTest

A new gate `spock_persistent_mlp_probe_fp16_input_smoke` exercises the fp16 file path:

```
vk_persistent_mlp_probe --hidden 128 --intermediate 16 --output-rows 8 --workgroups 8 --input-fp16-file tests/data/mlp_input_pattern_128.fp16
```

Full CTest suite: 13/13 passed (decode skeleton 6, MLP probe 6, diary check 1).

## Interpretation

This entry adds a third input mode to the MLP probe: raw fp16 file. Unlike `--input-token` (which requires the full repacked artifact), this mode needs only a flat binary file of the right size. This is the minimal bridge between the decode diagnostics pipeline and the persistent MLP probe.

The format is intentionally minimal: no header, no metadata, no alignment. Just `hidden * 2` bytes of raw fp16. This makes it trivial to produce from any tool that can dump a GPU buffer to disk. The Vulkan decode diagnostics already have the machinery to download per-step hidden states; writing them to a file and pointing the MLP probe at that file is a one-line pipeline.

The fixture demonstrates that the fp16 file path is not a separate compute path -- it feeds the same `input_data` buffer that the synthetic generator or the embedding loader would fill. The checksum identity proves this.

## What This Is

- **External fp16 input for the persistent MLP probe.** The input vector can now come from any raw fp16 file of sufficient length.
- **Clean handoff point for real hidden-state captures.** The Vulkan decode diagnostics can dump per-step hidden states to fp16 files and feed them into the MLP probe.
- **No dependency on the repacked artifact.** Unlike `--input-token`, this mode works with synthetic weights alone.

## What This Is Not

- **Not inference.** No RMSNorm, no token mixer, no attention/DeltaNet, no LM head.
- **Not RMSNorm fusion.** The hidden-state file would typically come from after RMSNorm in the real model, but the probe does not apply RMSNorm itself.
- **Not a full transformer layer.** Only the MLP side of one layer.
- **Not the megakernel.** This remains a standalone probe.
- **Not a performance claim.** No throughput or timing is reported.

## Next Work

1. Use `--input-fp16-file` with real hidden-state captures from the decode diagnostics to validate MLP computation against known intermediate values.
2. Add RMSNorm-before-MLP with real norm weights, so the probe can accept pre-RMSNorm input and apply the normalization internally.
3. Compose the MLP probe with the recurrent/attention side of a layer-shaped persistent probe.

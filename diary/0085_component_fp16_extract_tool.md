# 0085: Component FP16 Extract Tool

## Goal

Add a reproducible Python handoff utility that converts the existing `spock-decode --dump-step-components` stderr JSON into a raw little-endian fp16 vector file compatible with `vk_persistent_mlp_probe --input-fp16-file`. This bridges the gap between the decode diagnostics pipeline (which already dumps per-layer hidden-state intermediates as uint16 fp16 arrays in JSON) and the persistent MLP probe's raw binary input mode introduced in diary 0084.

## Background

Diary 0084 added `--input-fp16-file PATH` to `vk_persistent_mlp_probe`, enabling arbitrary raw fp16 vectors as probe input. The decode pipeline's `--dump-step-components` flag produces a `layer_components` JSON on stderr containing per-layer fp16 arrays: `input_hidden_fp16`, `mixer_residual_fp16`, `post_mlp_fp16`, `mlp_product_fp16`, `down_output_fp16`, and DeltaNet-specific fields like `dn_input_norm_fp16`, `dn_q_fp16`, etc.

These two systems currently have a format gap: the JSON stores fp16 values as arrays of unsigned 16-bit integers, while the probe expects a raw binary file of little-endian uint16 bytes. Converting manually is error-prone and not reproducible. A dedicated tool closes this gap.

## Implementation

### Tool: `tools/extract_component_fp16.py`

CLI:

```
python3 tools/extract_component_fp16.py --input dump.json --layer N --field FIELD --output out.fp16
```

Behavior:

1. Reads the JSON file specified by `--input`.
2. Navigates to `layers[N]` in the parsed JSON.
3. Extracts the named field (e.g. `input_hidden_fp16`, `post_mlp_fp16`).
4. Validates every value is an integer in [0, 65535].
5. Writes all values as raw little-endian uint16 to `--output`.

Error handling:

- Missing input file: clear error with path.
- Invalid JSON: reports the parse error.
- Missing `layers` key: reports the structural issue.
- Layer index out of range: reports valid range.
- Missing field in the selected layer: reports available context.
- Non-list field: rejects with type information.
- Non-integer or out-of-range values: reports the index and offending value.

The tool is intentionally minimal: no numpy dependency, no fp16 interpretation, no metadata in the output. It does a structural extraction from one serialization format (JSON integers) to another (raw LE bytes), preserving the exact bit pattern the decode pipeline already computed.

### Tests: `tests/run_extract_component_fp16_unit.py`

Ten tests covering success and failure modes:

Success cases:
1. `test_success_exact_bytes` -- extracts layer 0 `input_hidden_fp16`, verifies byte-for-byte match against `struct.pack`.
2. `test_success_layer1_field` -- extracts layer 1 `post_mlp_fp16`, verifies exact bytes.

Failure cases:
3. `test_failure_missing_file` -- nonexistent input.
4. `test_failure_invalid_json` -- malformed JSON.
5. `test_failure_missing_layers_key` -- JSON without `layers`.
6. `test_failure_layer_out_of_range` -- layer index 99 on a 2-layer dump.
7. `test_failure_missing_field` -- nonexistent field name.
8. `test_failure_non_list_field` -- field is a float (like `input_norm`), not an array.
9. `test_failure_out_of_range_value` -- value 70000 exceeds uint16.
10. `test_failure_non_integer_value` -- value 3.14 in the array.

All tests use `tempfile.TemporaryDirectory` with synthetic JSON fixtures. No GPU, no model artifacts, no network access.

### CTest

A new CTest entry `spock_extract_component_fp16` runs the unit test script.

## Verification

```
$ python3 tests/run_extract_component_fp16_unit.py
PASS: test_success_exact_bytes
PASS: test_success_layer1_field
PASS: test_failure_missing_file
PASS: test_failure_invalid_json
PASS: test_failure_missing_layers_key
PASS: test_failure_layer_out_of_range
PASS: test_failure_missing_field
PASS: test_failure_non_list_field
PASS: test_failure_out_of_range_value
PASS: test_failure_non_integer_value

all 10 tests passed
```

CTest suite passes with the new test added to the existing persistent probe and diary check gates.

## Interpretation

This tool is a bridge, not a compute step. It converts between two representations that already exist in the project: JSON arrays of uint16 integers (from decode diagnostics) and raw little-endian binary (from the MLP probe's file input). The conversion is lossless and bit-preserving.

The tool enables a concrete workflow:

1. Run `spock-decode --dump-step-components 0 --max-new-tokens 1 ...` to produce `layer_components` JSON on stderr.
2. Redirect stderr, extract the JSON.
3. Run `python3 tools/extract_component_fp16.py --input dump.json --layer 5 --field input_hidden_fp16 --output layer5_input.fp16`.
4. Feed `layer5_input.fp16` to `vk_persistent_mlp_probe --input-fp16-file layer5_input.fp16 ...`.

This workflow is manual by design. It exists to validate MLP computation against real intermediate values before composing layers or attempting the megakernel. Automating the pipeline is unnecessary until the probe proves useful for debugging.

## What This Is

- **A reproducible format bridge** from decode diagnostics JSON to the MLP probe's raw fp16 input mode.
- **A focused, tested Python utility** with no external dependencies.
- **A diagnostic workflow enabler** for validating persistent MLP compute against real hidden-state captures.

## What This Is Not

- **Not inference.** No token generation, no RMSNorm, no attention/DeltaNet, no LM head.
- **Not the megakernel.** This is a standalone offline tool, not a runtime component.
- **Not a performance claim.** No throughput or timing is involved.
- **Not a pipeline automation.** Each step runs manually; the tool does not invoke `spock-decode` or `vk_persistent_mlp_probe`.

## Next Work

1. Use the tool with real decode captures to validate persistent MLP probe output against known intermediate values.
2. Extend the probe with RMSNorm-before-MLP to accept pre-RMSNorm input and apply normalization internally.
3. Compose the MLP probe with the recurrent/attention side of a layer-shaped persistent probe.

#!/usr/bin/env python3
"""Extract a uint16 fp16 field from a dump-step-components JSON and write raw LE bytes.

Usage:
    python3 tools/extract_component_fp16.py --input dump.json --layer N --field FIELD --output out.fp16

Input: JSON produced by ``spock-decode --dump-step-components``.  The JSON
contains a ``layers`` array whose entries hold fields like
``input_hidden_fp16``, ``mixer_residual_fp16``, ``post_mlp_fp16``,
``mlp_product_fp16``, ``down_output_fp16``, etc.  Each field value is an
array of unsigned 16-bit integers representing raw fp16 bits.

The selected field values are validated to be integers in [0, 65535] and
written as little-endian uint16 to the output file.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract a fp16 field from a dump-step-components JSON into a raw LE binary file."
    )
    parser.add_argument("--input", required=True, type=Path, help="Path to the layer_components JSON file.")
    parser.add_argument("--layer", required=True, type=int, help="Zero-based layer index.")
    parser.add_argument("--field", required=True, help="Name of the fp16 array field to extract.")
    parser.add_argument("--output", required=True, type=Path, help="Output raw little-endian fp16 file.")
    args = parser.parse_args(argv)

    # --input must exist
    if not args.input.is_file():
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        return 1

    # Parse JSON
    try:
        data = json.loads(args.input.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"error: invalid JSON in {args.input}: {exc}", file=sys.stderr)
        return 1

    # Must contain a layers array
    if "layers" not in data:
        print("error: JSON missing top-level 'layers' key", file=sys.stderr)
        return 1
    layers = data["layers"]
    if not isinstance(layers, list):
        print("error: 'layers' is not an array", file=sys.stderr)
        return 1

    # Layer bounds
    if args.layer < 0 or args.layer >= len(layers):
        print(
            f"error: layer index {args.layer} out of range (0..{len(layers) - 1})",
            file=sys.stderr,
        )
        return 1

    layer_entry = layers[args.layer]

    # Field must exist
    if args.field not in layer_entry:
        print(
            f"error: field '{args.field}' not found in layer {args.layer}",
            file=sys.stderr,
        )
        return 1

    values = layer_entry[args.field]

    # Field must be a list
    if not isinstance(values, list):
        print(
            f"error: field '{args.field}' is not an array (got {type(values).__name__})",
            file=sys.stderr,
        )
        return 1

    # Validate all values
    for i, v in enumerate(values):
        if not isinstance(v, int):
            print(
                f"error: field '{args.field}'[{i}] is not an integer (got {type(v).__name__})",
                file=sys.stderr,
            )
            return 1
        if v < 0 or v > 65535:
            print(
                f"error: field '{args.field}'[{i}] = {v} is out of uint16 range [0, 65535]",
                file=sys.stderr,
            )
            return 1

    # Write raw little-endian uint16
    payload = struct.pack(f"<{len(values)}H", *values)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)

    print(f"wrote {len(values)} fp16 values ({len(payload)} bytes) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

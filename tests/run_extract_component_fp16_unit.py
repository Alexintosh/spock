#!/usr/bin/env python3
"""Unit tests for tools/extract_component_fp16.py.

Uses temp files and a tiny synthetic layer_components JSON.
Tests success and several failure modes.
"""
from __future__ import annotations

import json
import os
import struct
import tempfile
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "tools" / "extract_component_fp16.py"


def run_tool(args: list[str]) -> tuple[int, str, str]:
    """Run the tool as a subprocess and return (returncode, stdout, stderr).."""
    import subprocess

    result = subprocess.run(
        ["python3", str(TOOL)] + args,
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout, result.stderr


def make_synthetic_json(tmp: Path) -> Path:
    """Create a minimal layer_components JSON with 2 layers and small arrays."""
    data = {
        "diagnostic": "layer_components",
        "decode_step": 0,
        "layers": [
            {
                "layer": 0,
                "input_norm": 1.0,
                "mixer_norm": 1.0,
                "mlp_norm": 1.0,
                "mlp_product_norm": 1.0,
                "down_output_norm": 1.0,
                "input_hidden_fp16": [0, 1024, 32768, 65535, 1],
                "mixer_residual_fp16": [100, 200, 300],
            },
            {
                "layer": 1,
                "input_norm": 2.0,
                "mixer_norm": 2.0,
                "mlp_norm": 2.0,
                "mlp_product_norm": 2.0,
                "down_output_norm": 2.0,
                "input_hidden_fp16": [5, 10, 15, 20],
                "post_mlp_fp16": [4660, 4661, 4662],
            },
        ],
    }
    p = tmp / "dump.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_success_exact_bytes() -> None:
    """Success case: extract field and verify exact bytes written."""
    with tempfile.TemporaryDirectory(prefix="spock-extract-unit-") as tmp:
        tmpdir = Path(tmp)
        json_path = make_synthetic_json(tmpdir)
        out_path = tmpdir / "out.fp16"

        rc, stdout, stderr = run_tool([
            "--input", str(json_path),
            "--layer", "0",
            "--field", "input_hidden_fp16",
            "--output", str(out_path),
        ])

        assert rc == 0, f"tool failed: rc={rc}\nstdout={stdout}\nstderr={stderr}"
        assert out_path.is_file(), "output file not created"

        # Expected: [0, 1024, 32768, 65535, 1] as little-endian uint16
        expected = struct.pack("<5H", 0, 1024, 32768, 65535, 1)
        actual = out_path.read_bytes()
        assert actual == expected, f"byte mismatch: expected {expected.hex()}, got {actual.hex()}"
        print("PASS: test_success_exact_bytes")


def test_success_layer1_field() -> None:
    """Success case: extract from layer 1, different field."""
    with tempfile.TemporaryDirectory(prefix="spock-extract-unit-") as tmp:
        tmpdir = Path(tmp)
        json_path = make_synthetic_json(tmpdir)
        out_path = tmpdir / "layer1_post_mlp.fp16"

        rc, stdout, stderr = run_tool([
            "--input", str(json_path),
            "--layer", "1",
            "--field", "post_mlp_fp16",
            "--output", str(out_path),
        ])

        assert rc == 0, f"tool failed: rc={rc}\nstdout={stdout}\nstderr={stderr}"
        expected = struct.pack("<3H", 4660, 4661, 4662)
        actual = out_path.read_bytes()
        assert actual == expected, f"byte mismatch: expected {expected.hex()}, got {actual.hex()}"
        print("PASS: test_success_layer1_field")


def test_failure_missing_file() -> None:
    """Failure case: input file does not exist."""
    with tempfile.TemporaryDirectory(prefix="spock-extract-unit-") as tmp:
        out_path = Path(tmp) / "out.fp16"
        rc, stdout, stderr = run_tool([
            "--input", str(Path(tmp) / "nonexistent.json"),
            "--layer", "0",
            "--field", "input_hidden_fp16",
            "--output", str(out_path),
        ])
        assert rc != 0, "expected nonzero exit for missing file"
        assert "not found" in stderr.lower(), f"expected 'not found' in stderr: {stderr}"
        print("PASS: test_failure_missing_file")


def test_failure_invalid_json() -> None:
    """Failure case: input is not valid JSON."""
    with tempfile.TemporaryDirectory(prefix="spock-extract-unit-") as tmp:
        tmpdir = Path(tmp)
        bad = tmpdir / "bad.json"
        bad.write_text("this is not json {{{{", encoding="utf-8")
        out_path = tmpdir / "out.fp16"

        rc, stdout, stderr = run_tool([
            "--input", str(bad),
            "--layer", "0",
            "--field", "input_hidden_fp16",
            "--output", str(out_path),
        ])
        assert rc != 0, "expected nonzero exit for invalid JSON"
        assert "invalid json" in stderr.lower(), f"expected 'invalid json' in stderr: {stderr}"
        print("PASS: test_failure_invalid_json")


def test_failure_missing_layers_key() -> None:
    """Failure case: JSON has no 'layers' key."""
    with tempfile.TemporaryDirectory(prefix="spock-extract-unit-") as tmp:
        tmpdir = Path(tmp)
        j = tmpdir / "no_layers.json"
        j.write_text(json.dumps({"diagnostic": "layer_components"}), encoding="utf-8")
        out_path = tmpdir / "out.fp16"

        rc, stdout, stderr = run_tool([
            "--input", str(j),
            "--layer", "0",
            "--field", "input_hidden_fp16",
            "--output", str(out_path),
        ])
        assert rc != 0, "expected nonzero exit for missing 'layers'"
        assert "missing" in stderr.lower() and "layers" in stderr.lower(), f"stderr: {stderr}"
        print("PASS: test_failure_missing_layers_key")


def test_failure_layer_out_of_range() -> None:
    """Failure case: layer index beyond array bounds."""
    with tempfile.TemporaryDirectory(prefix="spock-extract-unit-") as tmp:
        tmpdir = Path(tmp)
        json_path = make_synthetic_json(tmpdir)
        out_path = tmpdir / "out.fp16"

        rc, stdout, stderr = run_tool([
            "--input", str(json_path),
            "--layer", "99",
            "--field", "input_hidden_fp16",
            "--output", str(out_path),
        ])
        assert rc != 0, "expected nonzero exit for layer out of range"
        assert "out of range" in stderr.lower(), f"stderr: {stderr}"
        print("PASS: test_failure_layer_out_of_range")


def test_failure_missing_field() -> None:
    """Failure case: requested field does not exist in the layer entry."""
    with tempfile.TemporaryDirectory(prefix="spock-extract-unit-") as tmp:
        tmpdir = Path(tmp)
        json_path = make_synthetic_json(tmpdir)
        out_path = tmpdir / "out.fp16"

        rc, stdout, stderr = run_tool([
            "--input", str(json_path),
            "--layer", "0",
            "--field", "nonexistent_field",
            "--output", str(out_path),
        ])
        assert rc != 0, "expected nonzero exit for missing field"
        assert "not found" in stderr.lower(), f"stderr: {stderr}"
        print("PASS: test_failure_missing_field")


def test_failure_non_list_field() -> None:
    """Failure case: field exists but is not a list."""
    with tempfile.TemporaryDirectory(prefix="spock-extract-unit-") as tmp:
        tmpdir = Path(tmp)
        json_path = make_synthetic_json(tmpdir)
        out_path = tmpdir / "out.fp16"

        # input_norm is a float, not a list
        rc, stdout, stderr = run_tool([
            "--input", str(json_path),
            "--layer", "0",
            "--field", "input_norm",
            "--output", str(out_path),
        ])
        assert rc != 0, "expected nonzero exit for non-list field"
        assert "not an array" in stderr.lower(), f"stderr: {stderr}"
        print("PASS: test_failure_non_list_field")


def test_failure_out_of_range_value() -> None:
    """Failure case: a value in the array is out of uint16 range."""
    with tempfile.TemporaryDirectory(prefix="spock-extract-unit-") as tmp:
        tmpdir = Path(tmp)
        data = {
            "layers": [
                {"layer": 0, "bad_field": [0, 100, 70000]},
            ],
        }
        j = tmpdir / "bad_values.json"
        j.write_text(json.dumps(data), encoding="utf-8")
        out_path = tmpdir / "out.fp16"

        rc, stdout, stderr = run_tool([
            "--input", str(j),
            "--layer", "0",
            "--field", "bad_field",
            "--output", str(out_path),
        ])
        assert rc != 0, "expected nonzero exit for out-of-range value"
        assert "out of" in stderr.lower() and "range" in stderr.lower(), f"stderr: {stderr}"
        print("PASS: test_failure_out_of_range_value")


def test_failure_non_integer_value() -> None:
    """Failure case: a value in the array is not an integer."""
    with tempfile.TemporaryDirectory(prefix="spock-extract-unit-") as tmp:
        tmpdir = Path(tmp)
        data = {
            "layers": [
                {"layer": 0, "float_field": [0, 3.14, 100]},
            ],
        }
        j = tmpdir / "float_vals.json"
        j.write_text(json.dumps(data), encoding="utf-8")
        out_path = tmpdir / "out.fp16"

        rc, stdout, stderr = run_tool([
            "--input", str(j),
            "--layer", "0",
            "--field", "float_field",
            "--output", str(out_path),
        ])
        assert rc != 0, "expected nonzero exit for non-integer value"
        assert "not an integer" in stderr.lower(), f"stderr: {stderr}"
        print("PASS: test_failure_non_integer_value")


def main() -> int:
    tests = [
        test_success_exact_bytes,
        test_success_layer1_field,
        test_failure_missing_file,
        test_failure_invalid_json,
        test_failure_missing_layers_key,
        test_failure_layer_out_of_range,
        test_failure_missing_field,
        test_failure_non_list_field,
        test_failure_out_of_range_value,
        test_failure_non_integer_value,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as exc:
            print(f"FAIL: {t.__name__}: {exc}")
            failed += 1
        except Exception as exc:
            print(f"ERROR: {t.__name__}: {exc}")
            failed += 1

    if failed:
        print(f"\n{failed}/{len(tests)} tests FAILED")
        return 1
    print(f"\nall {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

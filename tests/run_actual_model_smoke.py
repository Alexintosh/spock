#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--converter", required=True, type=Path)
    parser.add_argument("--validator", required=True, type=Path)
    args = parser.parse_args()

    if not args.model_dir.exists():
        raise SystemExit(f"model directory not found: {args.model_dir}")

    with tempfile.TemporaryDirectory(prefix="spock-real-model-") as tmp:
        artifact = Path(tmp) / "artifact"
        convert = subprocess.run(
            [
                "python3",
                str(args.converter),
                "--safetensors-scan",
                "--input",
                str(args.model_dir),
                "--output",
                str(artifact),
                "--force",
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if convert.returncode:
            print(convert.stdout)
            print(convert.stderr)
            return convert.returncode

        validate = subprocess.run(
            ["python3", str(args.validator), str(artifact), "--check-hashes", "--json"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if validate.returncode:
            print(validate.stdout)
            print(validate.stderr)
            return validate.returncode

        manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
        summary = manifest.get("tensor_summary", {})
        if summary.get("total", 0) < 400:
            raise SystemExit("actual model scan found too few tensors")
        if summary.get("dtype_counts", {}).get("bf16", 0) < 400:
            raise SystemExit("actual model scan did not find expected BF16 tensors")
        print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

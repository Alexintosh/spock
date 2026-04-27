#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--converter", required=True, type=Path)
    parser.add_argument("--validator", required=True, type=Path)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="spock-artifact-") as tmp:
        artifact = Path(tmp) / "artifact"
        convert = subprocess.run(
            ["python3", str(args.converter), "--offline", "--output", str(artifact), "--force"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if convert.returncode != 0:
            print(convert.stdout)
            print(convert.stderr)
            return convert.returncode

        validate = subprocess.run(
            ["python3", str(args.validator), str(artifact), "--json"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if validate.returncode != 0:
            print(validate.stdout)
            print(validate.stderr)
            return validate.returncode
        print(validate.stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

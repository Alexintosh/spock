#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    args = parser.parse_args()

    prompt_count = 0
    ids = set()
    for line_no, line in enumerate(args.prompts.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if "id" not in item or "text" not in item:
            raise SystemExit(f"{args.prompts}:{line_no}: missing id or text")
        ids.add(item["id"])
        prompt_count += 1

    if prompt_count < 32:
        raise SystemExit("prompt corpus must contain at least 32 prompts")
    if len(ids) != prompt_count:
        raise SystemExit("prompt ids must be unique")

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    for key in ("target", "model", "baselines"):
        if key not in baseline:
            raise SystemExit(f"baseline missing key: {key}")

    print(f"json fixtures valid: prompts={prompt_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Export a text load plan to TSV tasks for the native repacker."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


F32_PRESERVE_ROLES = {"delta_a_log", "delta_norm"}


def emit_task(writer: csv.writer, role_path: str, ref: dict, output_dtype: str) -> None:
    writer.writerow(
        [
            role_path,
            ref["name"],
            ref["file"],
            ref["offset"],
            ref["nbytes"],
            str(ref["dtype"]).lower(),
            output_dtype,
            json.dumps(ref["shape"], separators=(",", ":")),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["role_path", "name", "file", "offset", "nbytes", "source_dtype", "output_dtype", "shape"])
        emit_task(writer, "global.token_embedding", plan["global_tensors"]["token_embedding"], "fp16")
        emit_task(writer, "global.final_norm", plan["global_tensors"]["final_norm"], "fp16")
        for layer in plan["layers"]:
            index = layer["index"]
            for role in sorted(layer["roles"]):
                output_dtype = "fp32" if role in F32_PRESERVE_ROLES else "fp16"
                emit_task(writer, f"layer.{index}.{role}", layer["roles"][role], output_dtype)
    print(f"wrote tasks: {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
    parser.add_argument("--planner", required=True, type=Path)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="spock-text-plan-") as tmp:
        artifact = Path(tmp) / "artifact"
        plan_path = Path(tmp) / "text_plan.json"
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

        plan = subprocess.run(
            ["python3", str(args.planner), str(artifact), "--output", str(plan_path)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if plan.returncode:
            print(plan.stdout)
            print(plan.stderr)
            return plan.returncode

        data = json.loads(plan_path.read_text(encoding="utf-8"))
        summary = data["summary"]
        if summary["layer_kind_counts"] != {"attention": 6, "deltanet": 18}:
            raise SystemExit(f"unexpected layer counts: {summary['layer_kind_counts']}")
        if summary["text_tensors"] != 320:
            raise SystemExit(f"unexpected text tensor count: {summary['text_tensors']}")
        if summary["visual_tensors_excluded"] != 153:
            raise SystemExit(f"unexpected visual tensor count: {summary['visual_tensors_excluded']}")
        if data["global_tensors"]["lm_head"]["tied_to"] != "token_embedding":
            raise SystemExit("expected tied LM head")
        print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run spock-decode against frozen reference token IDs.

This is the real Vulkan decode parity harness. Unlike run_p0_parity.py, it
executes the decode CLI and compares generated token IDs.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def load_entries(path, limit):
    entries = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
            if limit and len(entries) >= limit:
                break
    return entries


def filter_entries(entries, ids):
    if not ids:
        return entries
    wanted = {item for item in ids if item}
    filtered = [entry for entry in entries if entry.get("id") in wanted]
    return filtered


def parse_decode_json(stdout):
    start = stdout.find("{")
    if start < 0:
        raise ValueError("decode output did not contain JSON")
    return json.loads(stdout[start:])


def first_mismatch_index(expected, actual):
    for i, (e, a) in enumerate(zip(expected, actual)):
        if e != a:
            return i
    if len(expected) != len(actual):
        return min(len(expected), len(actual))
    return None


def main():
    parser = argparse.ArgumentParser(description="Vulkan decode parity check")
    parser.add_argument("--decode", required=True, help="Path to spock-decode executable")
    parser.add_argument("--repack-dir", required=True, help="Path to repacked text weights")
    parser.add_argument("--reference", required=True, help="Path to reference_tokens.jsonl")
    parser.add_argument("--limit", type=int, default=1, help="Number of prompts to check")
    parser.add_argument("--max-new-tokens", type=int, default=1, help="Generated tokens per prompt")
    parser.add_argument(
        "--ids",
        default="",
        help="Comma-separated prompt IDs to check from the reference file",
    )
    parser.add_argument(
        "--expect-mismatch",
        action="store_true",
        help="Return success only if at least one checked prompt mismatches",
    )
    args = parser.parse_args()

    ids = [item.strip() for item in args.ids.split(",")] if args.ids else []
    load_limit = 0 if ids else args.limit
    entries = load_entries(args.reference, load_limit)
    entries = filter_entries(entries, ids)
    if not entries:
        print("FAIL: no reference entries loaded", file=sys.stderr)
        return 1

    failures = []
    checked = []
    with tempfile.TemporaryDirectory(prefix="spock-vk-parity-") as tmpdir:
        token_path = Path(tmpdir) / "prompt.tokens"
        for entry in entries:
            expected = entry["generated_ids"][: args.max_new_tokens]
            token_path.write_text(
                " ".join(str(t) for t in entry["prompt_ids"]) + "\n",
                encoding="utf-8",
            )
            proc = subprocess.run(
                [
                    args.decode,
                    "--repack-dir",
                    args.repack_dir,
                    "--tokens",
                    str(token_path),
                    "--max-new-tokens",
                    str(args.max_new_tokens),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if proc.returncode != 0:
                failures.append({
                    "id": entry.get("id"),
                    "error": "decode failed",
                    "returncode": proc.returncode,
                    "stderr": proc.stderr.strip(),
                })
                continue
            actual = parse_decode_json(proc.stdout)["generated_tokens"]
            record = {
                "id": entry.get("id"),
                "expected": expected,
                "actual": actual,
                "match": actual == expected,
            }
            mismatch_index = first_mismatch_index(expected, actual)
            if mismatch_index is not None:
                record["first_mismatch_index"] = mismatch_index
                record["matched_prefix_tokens"] = mismatch_index
            checked.append(record)
            if not record["match"]:
                failures.append(record)

    summary = {
        "status": "mismatch" if failures else "ok",
        "checked": len(checked),
        "failures": failures[:8],
    }
    print(json.dumps(summary, indent=2))

    if args.expect_mismatch:
        return 0 if failures else 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

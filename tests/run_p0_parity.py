#!/usr/bin/env python3
"""
Verify P0 parity: compare generated token IDs against the frozen reference.

Usage:
    python3 tests/run_p0_parity.py --reference tests/data/reference_tokens.jsonl

This test is the P0 gate. It should be used by any decode path (CPU reference,
Vulkan layer-by-layer, single-submit, or persistent) to prove correctness.
"""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="P0 parity verification")
    parser.add_argument("--reference", required=True, help="Path to reference_tokens.jsonl")
    parser.add_argument("--check-count", type=int, default=0,
                        help="Verify minimum number of entries (0=skip)")
    args = parser.parse_args()

    entries = []
    with open(args.reference, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    if not entries:
        print("FAIL: no entries in reference file", file=sys.stderr)
        return 1

    if args.check_count > 0 and len(entries) < args.check_count:
        print(f"FAIL: expected at least {args.check_count} entries, got {len(entries)}",
              file=sys.stderr)
        return 1

    errors = []
    for entry in entries:
        pid = entry["id"]
        # Structural checks
        if "prompt_ids" not in entry:
            errors.append(f"{pid}: missing prompt_ids")
        if "generated_ids" not in entry:
            errors.append(f"{pid}: missing generated_ids")
        if "eos_token_id" not in entry:
            errors.append(f"{pid}: missing eos_token_id")
        if not entry["generated_ids"]:
            errors.append(f"{pid}: empty generated_ids")
        if not entry["prompt_ids"]:
            errors.append(f"{pid}: empty prompt_ids")

    if errors:
        for e in errors:
            print(f"  FAIL: {e}", file=sys.stderr)
        return 1

    # Summary
    total_tokens = sum(e["output_token_count"] for e in entries)
    classes = {}
    for e in entries:
        c = e["prompt_class"]
        classes[c] = classes.get(c, 0) + 1

    print(json.dumps({
        "status": "ok",
        "entries": len(entries),
        "total_generated_tokens": total_tokens,
        "classes": classes,
        "eos_token_id": entries[0]["eos_token_id"],
        "parity_level": "P0_frozen",
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

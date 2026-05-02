#!/usr/bin/env python3
"""Validate spock-decode --diagnose-handoff output for a specific prompt.

Verifies that the diagnostic JSON contains:
- "diagnostic": "handoff_state"
- "recurrent_hidden_argmax_token" and top-5 logits
- Per-DeltaNet-layer chunk core_attn_out statistics
- GPU state norms and pre-norm hidden statistics
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def load_entry(path, target_id):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("id") == target_id:
                return entry
    return None


def main():
    parser = argparse.ArgumentParser(description="Validate diagnose-handoff output")
    parser.add_argument("--decode", required=True, help="Path to spock-decode")
    parser.add_argument("--repack-dir", required=True, help="Path to repacked weights")
    parser.add_argument("--reference", required=True, help="Path to reference_tokens.jsonl")
    parser.add_argument("--id", required=True, help="Prompt ID to diagnose")
    args = parser.parse_args()

    entry = load_entry(args.reference, args.id)
    if not entry:
        print(f"FAIL: prompt ID '{args.id}' not found in reference", file=sys.stderr)
        return 1

    expected_first_token = entry["generated_ids"][0]
    prompt_tokens_str = " ".join(str(t) for t in entry["prompt_ids"]) + "\n"

    with tempfile.TemporaryDirectory(prefix="spock-diagnose-") as tmpdir:
        token_path = Path(tmpdir) / "prompt.tokens"
        token_path.write_text(prompt_tokens_str, encoding="utf-8")

        proc = subprocess.run(
            [args.decode, "--repack-dir", args.repack_dir,
             "--tokens", str(token_path),
             "--max-new-tokens", "1",
             "--diagnose-handoff"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        if proc.returncode != 0:
            print(f"FAIL: spock-decode exit code {proc.returncode}", file=sys.stderr)
            print(f"stderr: {proc.stderr}", file=sys.stderr)
            return 1

        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            print(f"FAIL: invalid JSON output: {e}", file=sys.stderr)
            return 1

    # Required top-level fields
    errors = []
    if data.get("diagnostic") != "handoff_state":
        errors.append("missing or wrong 'diagnostic' field")
    if "prompt_len" not in data:
        errors.append("missing 'prompt_len'")
    if "recurrent_hidden_argmax_token" not in data:
        errors.append("missing 'recurrent_hidden_argmax_token'")
    if "recurrent_hidden_top5_logits" not in data:
        errors.append("missing 'recurrent_hidden_top5_logits'")

    recurrent_token = data.get("recurrent_hidden_argmax_token")

    # top-5 logits must contain the recurrent token
    top5 = data.get("recurrent_hidden_top5_logits", [])
    top5_tokens = {item["token"] for item in top5}
    if recurrent_token not in top5_tokens:
        errors.append(f"recurrent token {recurrent_token} not found in top-5 logits")

    # DeltaNet layers array must have 18 entries
    layers = data.get("deltanet_layers", [])
    if len(layers) != 18:
        errors.append(f"expected 18 DeltaNet layers, got {len(layers)}")

    for i, layer in enumerate(layers):
        for field in ["dn_idx", "model_layer", "chunk_attn_out_count",
                       "chunk_attn_out_mean", "chunk_attn_out_std",
                       "gpu_state_norm", "pre_norm_hidden_norm"]:
            if field not in layer:
                errors.append(f"layer {i}: missing '{field}'")
        heads = layer.get("heads", [])
        if len(heads) != 16:
            errors.append(f"layer {i}: expected 16 heads, got {len(heads)}")

    if errors:
        for e in errors:
            print(f"  FAIL: {e}", file=sys.stderr)
        return 1

    # Summary output for CTest log
    print(json.dumps({
        "status": "ok",
        "id": args.id,
        "prompt_len": data["prompt_len"],
        "recurrent_hidden_argmax_token": recurrent_token,
        "expected_first_token": expected_first_token,
        "match": recurrent_token == expected_first_token,
        "top5_tokens": [item["token"] for item in top5],
        "deltanet_layer_count": len(layers),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

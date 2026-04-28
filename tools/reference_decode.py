#!/usr/bin/env python3
"""
Reference decode tool for P0 parity.

Uses the trusted HuggingFace transformers implementation to generate
deterministic greedy token sequences for each prompt in the corpus.
Outputs a JSONL file with exact token IDs that the Spock engine must match.

Can optionally inject repacked FP16 weights instead of the original BF16,
producing the reference that matches the production precision path.

Usage:
    # Reference from original BF16 weights:
    python3 tools/reference_decode.py \
        --model-dir artifacts/hf/Qwen--Qwen3.5-0.8B \
        --tokenizer-dir artifacts/hf/Qwen--Qwen3.5-0.8B-tokenizer \
        --prompts tests/data/prompts.jsonl \
        --output tests/data/reference_tokens.jsonl \
        --max-new-tokens 16

    # Reference from repacked FP16 weights:
    python3 tools/reference_decode.py \
        --model-dir artifacts/hf/Qwen--Qwen3.5-0.8B \
        --tokenizer-dir artifacts/hf/Qwen--Qwen3.5-0.8B-tokenizer \
        --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
        --prompts tests/data/prompts.jsonl \
        --output tests/data/reference_tokens.jsonl \
        --max-new-tokens 16
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def repack_name_to_state_dict_key(repack_name):
    """Convert repack manifest tensor name to HF state_dict key."""
    return repack_name.replace("model.language_model.", "model.")


def load_repacked_tensors(repack_dir):
    """Load repacked weights from text_weights.bin."""
    manifest_path = os.path.join(repack_dir, "text_repack_manifest.json")
    weights_path = os.path.join(repack_dir, "text_weights.bin")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    tensors = {}
    with open(weights_path, "rb") as f:
        for entry in manifest["tensors"]:
            name = repack_name_to_state_dict_key(entry["name"])
            f.seek(entry["offset"])
            raw = f.read(entry["nbytes"])

            if entry["dtype"] == "fp16":
                arr = np.frombuffer(raw, dtype=np.float16).reshape(entry["shape"])
                tensors[name] = torch.from_numpy(arr.copy()).to(torch.float32)
            elif entry["dtype"] == "fp32":
                arr = np.frombuffer(raw, dtype=np.float32).reshape(entry["shape"])
                tensors[name] = torch.from_numpy(arr.copy()).to(torch.float32)
    return tensors


def load_model(model_dir, repack_dir=None):
    """Load the HF model in float32 on CPU. Optionally inject repacked FP16 weights."""
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.float32,
        trust_remote_code=True,
    )

    if repack_dir:
        tensors = load_repacked_tensors(repack_dir)
        state_dict = model.state_dict()
        matched = 0
        for key in state_dict:
            if "visual" in key or "mtp" in key:
                continue
            if key in tensors:
                state_dict[key] = tensors[key]
                matched += 1
            elif key == "lm_head.weight" and "model.embed_tokens.weight" in tensors:
                state_dict[key] = tensors["model.embed_tokens.weight"]
                matched += 1
        model.load_state_dict(state_dict)
        print(f"Injected {matched} repacked tensors from {repack_dir}")

    model.eval()
    return model


def load_tokenizer(tokenizer_dir):
    return AutoTokenizer.from_pretrained(tokenizer_dir)


def generate_reference(model, tokenizer, prompt_text, max_new_tokens):
    """Generate deterministic greedy tokens and return token IDs + timing."""
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt")

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - t0

    input_len = input_ids.shape[1]
    all_ids = output_ids[0].tolist()
    prompt_ids = all_ids[:input_len]
    generated_ids = all_ids[input_len:]

    return {
        "prompt_ids": prompt_ids,
        "generated_ids": generated_ids,
        "input_token_count": input_len,
        "output_token_count": len(generated_ids),
        "elapsed_seconds": round(elapsed, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Reference decode for P0 parity")
    parser.add_argument("--model-dir", required=True, help="Path to HF model directory")
    parser.add_argument("--tokenizer-dir", required=True, help="Path to tokenizer directory")
    parser.add_argument("--repack-dir", default=None,
                        help="Optional: inject repacked FP16 weights from this directory")
    parser.add_argument("--prompts", required=True, help="Path to prompts.jsonl")
    parser.add_argument("--output", required=True, help="Output path for reference_tokens.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=16, help="Tokens to generate per prompt")
    parser.add_argument("--max-prompts", type=int, default=0, help="Limit number of prompts (0=all)")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer_dir)
    model = load_model(args.model_dir, args.repack_dir)

    weight_source = "repacked FP16" if args.repack_dir else "original BF16"
    print(f"Model loaded from {args.model_dir} ({weight_source})")
    print(f"EOS token ID: {tokenizer.eos_token_id}")
    print(f"Max new tokens: {args.max_new_tokens}")

    prompts = []
    with open(args.prompts, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))

    if args.max_prompts > 0:
        prompts = prompts[:args.max_prompts]

    results = []
    for i, prompt_entry in enumerate(prompts):
        pid = prompt_entry["id"]
        text = prompt_entry["text"]
        print(f"  [{i+1}/{len(prompts)}] {pid} ({len(text)} chars)...", end=" ", flush=True)

        ref = generate_reference(model, tokenizer, text, args.max_new_tokens)
        ref["id"] = pid
        ref["prompt_text"] = text
        ref["prompt_class"] = prompt_entry.get("class", "unknown")
        ref["eos_token_id"] = tokenizer.eos_token_id
        ref["weight_source"] = weight_source
        results.append(ref)

        decoded = tokenizer.decode(ref["generated_ids"], skip_special_tokens=True)
        print(f"{ref['input_token_count']}+{ref['output_token_count']} tokens, "
              f"{ref['elapsed_seconds']:.2f}s, "
              f"output: {decoded[:60]!r}...")

    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nWrote {len(results)} reference entries to {args.output}")


if __name__ == "__main__":
    main()

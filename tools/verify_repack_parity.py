#!/usr/bin/env python3
"""
Verify that the repacked text_weights.bin produces the same P0 tokens as the
original HuggingFace model, and optionally capture per-layer activations.

This proves the BF16→FP16 repack pipeline is lossless (within FP16 rounding).

Usage:
    # Full P0 parity check with repacked weights:
    python3 tools/verify_repack_parity.py \
        --model-dir artifacts/hf/Qwen--Qwen3.5-0.8B \
        --tokenizer-dir artifacts/hf/Qwen--Qwen3.5-0.8B-tokenizer \
        --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
        --reference tests/data/reference_tokens.jsonl \
        --max-prompts 5

    # With activation capture:
    python3 tools/verify_repack_parity.py \
        ... \
        --capture-activations \
        --activations-output /tmp/spock_activations.json

Exit codes:
    0: full P0 parity (all tokens match)
    1: token mismatch
    2: setup error
"""

import argparse
import json
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Mapping from repack manifest role_path suffix to PyTorch state_dict suffix
# The repack manifest uses names like "model.language_model.layers.X.Y"
# while the HF state_dict uses "model.layers.X.Y"
def repack_name_to_state_dict_key(repack_name: str) -> str:
    """Convert a repack manifest tensor name to a HF state_dict key."""
    # model.language_model.layers.X.Y -> model.layers.X.Y
    return repack_name.replace("model.language_model.", "model.")


def load_repacked_weights(repack_dir: str):
    """Load the repack manifest and weights file, return a dict of tensors."""
    import os

    manifest_path = os.path.join(repack_dir, "text_repack_manifest.json")
    weights_path = os.path.join(repack_dir, "text_weights.bin")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    tensors = {}
    with open(weights_path, "rb") as f:
        for entry in manifest["tensors"]:
            offset = entry["offset"]
            nbytes = entry["nbytes"]
            dtype_str = entry["dtype"]
            shape = entry["shape"]
            name = repack_name_to_state_dict_key(entry["name"])

            f.seek(offset)
            raw = f.read(nbytes)

            if dtype_str == "fp16":
                arr = np.frombuffer(raw, dtype=np.float16).reshape(shape)
                tensor = torch.from_numpy(arr.copy()).to(torch.float32)
            elif dtype_str == "fp32":
                arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
                tensor = torch.from_numpy(arr.copy()).to(torch.float32)
            else:
                raise ValueError(f"Unknown dtype: {dtype_str}")

            tensors[name] = tensor

    return tensors, manifest


def inject_weights(model, tensors: dict):
    """Load repacked weights into the HF model. Returns count of matched params."""
    state_dict = model.state_dict()
    matched = 0
    missing = []

    for key in state_dict:
        if "visual" in key or "mtp" in key:
            continue
        if key in tensors:
            state_dict[key] = tensors[key]
            matched += 1
        elif key == "lm_head.weight":
            # Tied weights: lm_head shares embed_tokens
            if "model.embed_tokens.weight" in tensors:
                state_dict[key] = tensors["model.embed_tokens.weight"]
                matched += 1
            else:
                missing.append(key)
        else:
            missing.append(key)

    if missing:
        print(f"WARNING: {len(missing)} state_dict keys not found in repacked weights:")
        for m in missing[:10]:
            print(f"  {m}")
        if len(missing) > 10:
            print(f"  ... and {len(missing)-10} more")

    model.load_state_dict(state_dict)
    return matched


def generate_with_model(model, tokenizer, prompt_text: str, max_new_tokens: int):
    """Generate deterministic greedy tokens."""
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    input_len = input_ids.shape[1]
    all_ids = output_ids[0].tolist()
    return all_ids[:input_len], all_ids[input_len:]


class ActivationCapture:
    """Hook into model layers to capture intermediate activations."""

    def __init__(self):
        self.activations = {}
        self._hooks = []

    def _hook_fn(self, name):
        def fn(module, input, output):
            if isinstance(output, tuple):
                # Take the first element (hidden states), skip cache etc.
                val = output[0]
            else:
                val = output
            self.activations[name] = val.detach().cpu().float().numpy()
        return fn

    def register(self, model):
        for name, module in model.named_modules():
            # Capture at key layer boundaries
            if any(name.endswith(suffix) for suffix in [
                "input_layernorm",
                "post_attention_layernorm",
                "linear_attn",
                "self_attn",
                "mlp",
                "norm",
                "embed_tokens",
            ]):
                hook = module.register_forward_hook(self._hook_fn(name))
                self._hooks.append(hook)

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def to_json_safe(self):
        """Convert activations to a JSON-safe dict with summary stats."""
        result = {}
        for name, arr in self.activations.items():
            result[name] = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "abs_max": float(np.abs(arr).max()),
                # Store first 16 values for spot checks
                "sample": arr.flatten()[:16].tolist(),
            }
        return result


def main():
    parser = argparse.ArgumentParser(description="Verify repack parity against P0 reference")
    parser.add_argument("--model-dir", required=True, help="Path to HF model directory")
    parser.add_argument("--tokenizer-dir", required=True, help="Path to tokenizer directory")
    parser.add_argument("--repack-dir", required=True, help="Path to repacked artifact directory")
    parser.add_argument("--reference", required=True, help="Path to reference_tokens.jsonl")
    parser.add_argument("--max-prompts", type=int, default=0, help="Limit prompts (0=all)")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--capture-activations", action="store_true",
                        help="Capture per-layer activations for first prompt")
    parser.add_argument("--activations-output", default=None,
                        help="Output path for captured activations JSON")
    args = parser.parse_args()

    # Load reference
    reference = []
    with open(args.reference, "r") as f:
        for line in f:
            if line.strip():
                reference.append(json.loads(line))
    if args.max_prompts > 0:
        reference = reference[:args.max_prompts]

    print(f"Reference entries: {len(reference)}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    # Load repacked weights
    print("Loading repacked weights...")
    t0 = time.perf_counter()
    tensors, manifest = load_repacked_weights(args.repack_dir)
    print(f"  Loaded {len(tensors)} tensors in {time.perf_counter()-t0:.2f}s")

    # Load model and inject repacked weights
    print("Loading model skeleton...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=torch.float32,
        trust_remote_code=True,
    )

    print("Injecting repacked weights...")
    matched = inject_weights(model, tensors)
    print(f"  Matched {matched} parameters")
    model.eval()

    # Optionally capture activations
    capture = None
    if args.capture_activations:
        capture = ActivationCapture()
        capture.register(model)

    # Verify P0 parity
    mismatches = 0
    for i, ref_entry in enumerate(reference):
        pid = ref_entry["id"]
        text = ref_entry["prompt_text"]
        expected = ref_entry["generated_ids"]

        print(f"  [{i+1}/{len(reference)}] {pid}...", end=" ", flush=True)

        _, generated = generate_with_model(model, tokenizer, text, args.max_new_tokens)

        if generated == expected:
            print(f"OK ({len(generated)} tokens)")
        else:
            mismatches += 1
            # Find first divergence
            for j in range(min(len(generated), len(expected))):
                if generated[j] != expected[j]:
                    break
            print(f"MISMATCH at token {j}: got {generated[j]}, expected {expected[j]}")
            print(f"    generated: {generated}")
            print(f"    expected:  {expected}")

        # Only capture activations for the first prompt
        if capture and i == 0:
            activations = capture.to_json_safe()
            capture.remove()
            capture = None

            if args.activations_output:
                with open(args.activations_output, "w") as f:
                    json.dump(activations, f, indent=2)
                print(f"  Wrote activations to {args.activations_output}")
            else:
                print(f"  Captured {len(activations)} activation layers")

    if capture:
        capture.remove()

    # Summary
    if mismatches == 0:
        print(f"\nPASS: all {len(reference)} prompts match P0 reference")
        return 0
    else:
        print(f"\nFAIL: {mismatches}/{len(reference)} prompts mismatch")
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Create a text-only runtime load plan from a Spock model artifact manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


TEXT_PREFIX = "model.language_model."
LAYER_PREFIX = TEXT_PREFIX + "layers."

DELTANET_REQUIRED = {
    "input_layernorm.weight": "input_norm",
    "post_attention_layernorm.weight": "post_norm",
    "linear_attn.A_log": "delta_a_log",
    "linear_attn.dt_bias": "delta_dt_bias",
    "linear_attn.conv1d.weight": "delta_conv",
    "linear_attn.in_proj_a.weight": "delta_in_proj_a",
    "linear_attn.in_proj_b.weight": "delta_in_proj_b",
    "linear_attn.in_proj_qkv.weight": "delta_in_proj_qkv",
    "linear_attn.in_proj_z.weight": "delta_in_proj_z",
    "linear_attn.norm.weight": "delta_norm",
    "linear_attn.out_proj.weight": "delta_out_proj",
    "mlp.gate_proj.weight": "mlp_gate",
    "mlp.up_proj.weight": "mlp_up",
    "mlp.down_proj.weight": "mlp_down",
}

ATTENTION_REQUIRED = {
    "input_layernorm.weight": "input_norm",
    "post_attention_layernorm.weight": "post_norm",
    "self_attn.q_proj.weight": "attn_q",
    "self_attn.k_proj.weight": "attn_k",
    "self_attn.v_proj.weight": "attn_v",
    "self_attn.o_proj.weight": "attn_o",
    "self_attn.q_norm.weight": "attn_q_norm",
    "self_attn.k_norm.weight": "attn_k_norm",
    "mlp.gate_proj.weight": "mlp_gate",
    "mlp.up_proj.weight": "mlp_up",
    "mlp.down_proj.weight": "mlp_down",
}


class PlanError(RuntimeError):
    """Raised for invalid or unsupported artifact plans."""


def _manifest_path(path: Path) -> Path:
    return path / "manifest.json" if path.is_dir() else path


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest_path = _manifest_path(path)
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PlanError(f"unable to read manifest: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise PlanError(f"invalid manifest JSON: {manifest_path}: {exc}") from exc


def _tensor_index(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tensors = manifest.get("tensors")
    if not isinstance(tensors, list):
        raise PlanError("manifest.tensors must be an array")
    index: dict[str, dict[str, Any]] = {}
    for tensor in tensors:
        if not isinstance(tensor, dict) or not isinstance(tensor.get("name"), str):
            raise PlanError("every tensor must be an object with a string name")
        if tensor["name"] in index:
            raise PlanError(f"duplicate tensor name: {tensor['name']}")
        index[tensor["name"]] = tensor
    return index


def _tensor_ref(tensor: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": tensor["name"],
        "file": tensor["file"],
        "offset": tensor["offset"],
        "nbytes": tensor["nbytes"],
        "dtype": tensor["dtype"],
        "shape": tensor["shape"],
    }


def _layer_tensor_name(index: int, suffix: str) -> str:
    return f"{LAYER_PREFIX}{index}.{suffix}"


def _build_layer_plan(layer: dict[str, Any], tensors: dict[str, dict[str, Any]]) -> dict[str, Any]:
    index = layer.get("index")
    kind = layer.get("kind")
    if not isinstance(index, int) or kind not in {"deltanet", "attention"}:
        raise PlanError(f"invalid layer record: {layer}")

    required = DELTANET_REQUIRED if kind == "deltanet" else ATTENTION_REQUIRED
    roles: dict[str, Any] = {}
    missing: list[str] = []
    for suffix, role in required.items():
        name = _layer_tensor_name(index, suffix)
        tensor = tensors.get(name)
        if tensor is None:
            missing.append(name)
        else:
            roles[role] = _tensor_ref(tensor)

    if missing:
        raise PlanError(f"layer {index} missing required tensors: {', '.join(missing)}")

    return {
        "index": index,
        "kind": kind,
        "source_kind": layer.get("source_kind", kind),
        "roles": roles,
    }


def build_plan(manifest: dict[str, Any]) -> dict[str, Any]:
    tensors = _tensor_index(manifest)
    layers = manifest.get("layers")
    if not isinstance(layers, list):
        raise PlanError("manifest.layers must be an array")

    embed = tensors.get("model.language_model.embed_tokens.weight")
    final_norm = tensors.get("model.language_model.norm.weight")
    if embed is None:
        raise PlanError("missing text embedding tensor")
    if final_norm is None:
        raise PlanError("missing final language-model norm tensor")

    text_tensors = [name for name in tensors if name.startswith(TEXT_PREFIX)]
    visual_tensors = [name for name in tensors if name.startswith("model.visual.")]
    mtp_tensors = [name for name in tensors if name.startswith("mtp.")]
    other_tensors = [
        name
        for name in tensors
        if not name.startswith(TEXT_PREFIX)
        and not name.startswith("model.visual.")
        and not name.startswith("mtp.")
    ]

    layer_plans = [_build_layer_plan(layer, tensors) for layer in layers]
    kind_counts = {
        "deltanet": sum(1 for layer in layer_plans if layer["kind"] == "deltanet"),
        "attention": sum(1 for layer in layer_plans if layer["kind"] == "attention"),
    }

    return {
        "schema_version": 1,
        "plan_type": "spock-text-decode-load-plan",
        "source": {
            "model_id": manifest.get("source", {}).get("model_id"),
            "revision": manifest.get("source", {}).get("revision"),
            "artifact_name": manifest.get("artifact", {}).get("name"),
        },
        "model": manifest.get("model", {}),
        "precision": {
            "source_storage_dtype": manifest.get("packing", {}).get("storage_dtype"),
            "runtime_weight_dtype": "fp16",
            "runtime_activation_dtype": "fp16",
            "runtime_recurrent_state_dtype": "fp32",
            "requires_bf16_to_fp16_repack": True,
        },
        "global_tensors": {
            "token_embedding": _tensor_ref(embed),
            "lm_head": {
                "tied_to": "token_embedding",
                "source": _tensor_ref(embed),
            },
            "final_norm": _tensor_ref(final_norm),
        },
        "layers": layer_plans,
        "summary": {
            "text_tensors": len(text_tensors),
            "visual_tensors_excluded": len(visual_tensors),
            "mtp_tensors_excluded": len(mtp_tensors),
            "other_tensors_excluded": len(other_tensors),
            "layer_count": len(layer_plans),
            "layer_kind_counts": kind_counts,
        },
        "excluded_prefixes": ["model.visual.", "mtp."],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan the text-only Spock runtime tensor mapping from a real artifact manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("artifact", type=Path, help="artifact directory or manifest.json path")
    parser.add_argument("-o", "--output", type=Path, help="write plan JSON to this path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        plan = build_plan(_load_manifest(args.artifact))
    except PlanError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    rendered = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

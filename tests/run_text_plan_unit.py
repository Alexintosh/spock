#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


DELTANET_SUFFIXES = [
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "linear_attn.A_log",
    "linear_attn.dt_bias",
    "linear_attn.conv1d.weight",
    "linear_attn.in_proj_a.weight",
    "linear_attn.in_proj_b.weight",
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight",
    "linear_attn.norm.weight",
    "linear_attn.out_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
]

ATTENTION_SUFFIXES = [
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
]


def tensor(name: str, offset: int) -> dict[str, object]:
    return {
        "name": name,
        "file": "weights.bin",
        "offset": offset,
        "nbytes": 16,
        "dtype": "bf16",
        "shape": [1, 8],
    }


def make_manifest() -> dict[str, object]:
    tensors = [
        tensor("model.language_model.embed_tokens.weight", 0),
        tensor("model.language_model.norm.weight", 16),
        tensor("model.visual.stub.weight", 32),
        tensor("mtp.stub.weight", 48),
    ]
    layers = []
    offset = 64
    for index in range(24):
        kind = "attention" if index % 4 == 3 else "deltanet"
        suffixes = ATTENTION_SUFFIXES if kind == "attention" else DELTANET_SUFFIXES
        layers.append(
            {
                "index": index,
                "kind": kind,
                "source_kind": "full_attention" if kind == "attention" else "linear_attention",
                "weight_groups": [],
            }
        )
        for suffix in suffixes:
            tensors.append(tensor(f"model.language_model.layers.{index}.{suffix}", offset))
            offset += 16

    return {
        "schema_version": 1,
        "artifact": {
            "name": "synthetic",
            "format": "spock-packed-model",
            "created_at_utc": "1970-01-01T00:00:00Z",
            "generator": "run_text_plan_unit.py",
            "dry_run": False,
            "manifest_only": False,
        },
        "source": {"model_id": "Qwen/Qwen3.5-0.8B", "revision": "synthetic"},
        "model": {
            "architecture": "qwen35_0p8b_hybrid_deltanet_attention",
            "layers": 24,
            "layer_pattern": [0, 0, 0, 1],
            "hidden_size": 1024,
            "intermediate_size": 3584,
            "attention": {"q_heads": 8, "kv_heads": 2, "head_dim": 256},
            "deltanet": {"heads": 16, "key_dim": 128, "value_dim": 128, "conv_kernel": 4},
            "max_sequence_length": 2048,
        },
        "packing": {
            "alignment": 1,
            "byte_order": "little",
            "storage_dtype": "source",
            "offset_units": "bytes",
        },
        "files": [{"path": "weights.bin", "size_bytes": offset, "sha256": ""}],
        "tensors": tensors,
        "layers": layers,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner", required=True, type=Path)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="spock-text-plan-unit-") as tmp:
        root = Path(tmp)
        (root / "weights.bin").write_bytes(b"\0" * 8192)
        (root / "manifest.json").write_text(json.dumps(make_manifest()), encoding="utf-8")
        result = subprocess.run(
            ["python3", str(args.planner), str(root)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode:
            print(result.stdout)
            print(result.stderr)
            return result.returncode
        plan = json.loads(result.stdout)
        if plan["summary"]["layer_kind_counts"] != {"attention": 6, "deltanet": 18}:
            raise SystemExit("wrong layer counts")
        if plan["summary"]["visual_tensors_excluded"] != 1:
            raise SystemExit("wrong visual exclusion count")
        print("synthetic text plan valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

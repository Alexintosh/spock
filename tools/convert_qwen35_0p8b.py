#!/usr/bin/env python3
"""Convert Qwen 3.5 0.8B weights into a Spock packed artifact.

The dry-run/offline path intentionally depends only on the Python standard
library so artifact manifests can be generated on machines without ML stacks.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-0.8B"
DEFAULT_ARTIFACT_NAME = "qwen35_0p8b.spock"
ATTENTION_PATTERN = (0, 0, 0, 1)


@dataclass(frozen=True)
class ModelSpec:
    model_id: str = DEFAULT_MODEL_ID
    architecture: str = "qwen35_0p8b_hybrid_deltanet_attention"
    layers: int = 24
    hidden_size: int = 1024
    intermediate_size: int = 3584
    attention_q_heads: int = 8
    attention_kv_heads: int = 2
    attention_head_dim: int = 256
    deltanet_heads: int = 16
    deltanet_key_dim: int = 128
    deltanet_value_dim: int = 128
    deltanet_conv_kernel: int = 4
    max_sequence_length: int = 2048


class ConversionError(RuntimeError):
    """Raised for user-actionable conversion failures."""


def _optional_import(module_name: str) -> Any | None:
    if importlib.util.find_spec(module_name) is None:
        return None
    return __import__(module_name)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _align_offset(offset: int, alignment: int) -> int:
    return (offset + alignment - 1) // alignment * alignment


def _layer_kind(index: int) -> str:
    return "attention" if ATTENTION_PATTERN[index % len(ATTENTION_PATTERN)] else "deltanet"


def _base_manifest(args: argparse.Namespace, *, dry_run: bool) -> dict[str, Any]:
    spec = ModelSpec(model_id=args.model_id)
    layers = [
        {
            "index": index,
            "kind": _layer_kind(index),
            "weight_groups": _expected_weight_groups(_layer_kind(index)),
        }
        for index in range(spec.layers)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": {
            "name": args.artifact_name,
            "format": "spock-packed-model",
            "created_at_utc": _utc_now(),
            "generator": Path(__file__).name,
            "dry_run": dry_run,
            "manifest_only": dry_run,
        },
        "source": {
            "model_id": args.model_id,
            "revision": args.revision,
            "local_path": str(args.input) if args.input else None,
        },
        "model": {
            "architecture": spec.architecture,
            "layers": spec.layers,
            "layer_pattern": list(ATTENTION_PATTERN),
            "hidden_size": spec.hidden_size,
            "intermediate_size": spec.intermediate_size,
            "attention": {
                "q_heads": spec.attention_q_heads,
                "kv_heads": spec.attention_kv_heads,
                "head_dim": spec.attention_head_dim,
            },
            "deltanet": {
                "heads": spec.deltanet_heads,
                "key_dim": spec.deltanet_key_dim,
                "value_dim": spec.deltanet_value_dim,
                "conv_kernel": spec.deltanet_conv_kernel,
            },
            "max_sequence_length": spec.max_sequence_length,
        },
        "packing": {
            "alignment": args.alignment,
            "byte_order": "little",
            "storage_dtype": args.dtype,
            "offset_units": "bytes",
        },
        "files": [],
        "tensors": [],
        "layers": layers,
        "notes": [],
    }


def _expected_weight_groups(layer_kind: str) -> list[str]:
    common = ["input_norm", "mlp_gate_up", "mlp_down", "post_norm"]
    if layer_kind == "attention":
        return ["attn_qkv", "attn_o", *common]
    return ["deltanet_in_proj", "deltanet_conv", "deltanet_out_proj", *common]


def _write_manifest(output: Path, manifest: dict[str, Any], *, force: bool) -> Path:
    manifest_path = output if output.suffix == ".json" else output / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists() and not force:
        raise ConversionError(f"{manifest_path} already exists; pass --force to overwrite")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def _artifact_root(output: Path) -> Path:
    return output.parent if output.suffix == ".json" else output


def _validate_dry_run_args(args: argparse.Namespace) -> None:
    if args.alignment <= 0 or args.alignment & (args.alignment - 1):
        raise ConversionError("--alignment must be a positive power of two")
    if args.dtype not in {"fp16", "fp32", "source"}:
        raise ConversionError("--dtype must be fp16, fp32, or source")


def _require_real_conversion_deps() -> tuple[Any | None, Any, Any]:
    transformers = _optional_import("transformers")
    safetensors = _optional_import("safetensors")
    torch = _optional_import("torch")
    missing = [
        name
        for name, module in (
            ("safetensors", safetensors),
            ("torch", torch),
        )
        if module is None
    ]
    if missing:
        raise ConversionError(
            "real conversion requires optional dependencies: "
            + ", ".join(missing)
            + ". Use --dry-run or --offline for a manifest-only artifact."
        )
    return transformers, safetensors, torch


def _find_safetensors_inputs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix != ".safetensors":
            raise ConversionError(f"input file must be .safetensors: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise ConversionError(f"input path does not exist: {input_path}")
    files = sorted(input_path.glob("*.safetensors"))
    if not files:
        raise ConversionError(f"no .safetensors files found in {input_path}")
    return files


def _read_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as handle:
        raw_len = handle.read(8)
        if len(raw_len) != 8:
            raise ConversionError(f"not a safetensors file or truncated header: {path}")
        header_len = int.from_bytes(raw_len, "little")
        if header_len <= 0 or header_len > 256 * 1024 * 1024:
            raise ConversionError(f"invalid safetensors header length in {path}: {header_len}")
        try:
            header = json.loads(handle.read(header_len).decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ConversionError(f"invalid safetensors JSON header in {path}: {exc}") from exc
    if not isinstance(header, dict):
        raise ConversionError(f"safetensors header must be an object: {path}")
    return 8 + header_len, header


def _copy_or_link(source: Path, dest: Path, *, force: bool) -> None:
    if dest.exists():
        if not force:
            raise ConversionError(f"{dest} already exists; pass --force to overwrite")
        dest.unlink()
    try:
        dest.hardlink_to(source)
    except OSError:
        import shutil

        shutil.copy2(source, dest)


def _convert_safetensors_scan(args: argparse.Namespace) -> dict[str, Any]:
    if args.input is None:
        raise ConversionError("--safetensors-scan requires --input")

    input_files = _find_safetensors_inputs(args.input)
    artifact_root = _artifact_root(args.output)
    artifact_root.mkdir(parents=True, exist_ok=True)

    manifest = _base_manifest(args, dry_run=False)
    manifest["artifact"]["manifest_only"] = False
    manifest["packing"]["storage_dtype"] = "source"
    manifest["packing"]["alignment"] = 1
    manifest["notes"].append(
        "Scanned real safetensors headers and linked/copied original tensor bytes without dtype conversion."
    )
    manifest["notes"].append("Production fp16 repacking still requires torch or an equivalent dtype conversion path.")

    config_path = (args.input / "config.json") if args.input and args.input.is_dir() else None
    if config_path is not None and config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        manifest["source"]["config"] = config
        text_config = config.get("text_config", {})
        if isinstance(text_config, dict):
            layer_types = text_config.get("layer_types")
            if isinstance(layer_types, list):
                manifest["layers"] = [
                    {
                        "index": index,
                        "kind": "attention" if kind == "full_attention" else "deltanet",
                        "source_kind": kind,
                        "weight_groups": _expected_weight_groups(
                            "attention" if kind == "full_attention" else "deltanet"
                        ),
                    }
                    for index, kind in enumerate(layer_types)
                ]
                manifest["model"]["layers"] = len(layer_types)
            for source_key, dest_key in (
                ("hidden_size", "hidden_size"),
                ("intermediate_size", "intermediate_size"),
                ("max_position_embeddings", "source_max_position_embeddings"),
                ("vocab_size", "vocab_size"),
                ("rms_norm_eps", "rms_norm_eps"),
            ):
                if source_key in text_config:
                    manifest["model"][dest_key] = text_config[source_key]
            manifest["model"]["attention"] = {
                **manifest["model"]["attention"],
                "q_heads": text_config.get("num_attention_heads", manifest["model"]["attention"]["q_heads"]),
                "kv_heads": text_config.get("num_key_value_heads", manifest["model"]["attention"]["kv_heads"]),
                "head_dim": text_config.get("head_dim", manifest["model"]["attention"]["head_dim"]),
                "full_attention_interval": text_config.get("full_attention_interval"),
                "rope_parameters": text_config.get("rope_parameters"),
            }
            manifest["model"]["deltanet"] = {
                **manifest["model"]["deltanet"],
                "heads": text_config.get("linear_num_key_heads", manifest["model"]["deltanet"]["heads"]),
                "key_dim": text_config.get("linear_key_head_dim", manifest["model"]["deltanet"]["key_dim"]),
                "value_heads": text_config.get("linear_num_value_heads"),
                "value_dim": text_config.get("linear_value_head_dim", manifest["model"]["deltanet"]["value_dim"]),
                "conv_kernel": text_config.get("linear_conv_kernel_dim", manifest["model"]["deltanet"]["conv_kernel"]),
                "state_dtype": text_config.get("mamba_ssm_dtype"),
            }

    files: list[dict[str, Any]] = []
    tensors: list[dict[str, Any]] = []
    for source_file in input_files:
        dest = artifact_root / source_file.name
        _copy_or_link(source_file, dest, force=args.force)
        data_start, header = _read_safetensors_header(dest)
        files.append(
            {
                "path": dest.name,
                "size_bytes": dest.stat().st_size,
                "sha256": _sha256_file(dest),
            }
        )
        for name, meta in sorted(header.items()):
            if name == "__metadata__":
                continue
            if not isinstance(meta, dict):
                raise ConversionError(f"invalid tensor metadata for {name} in {source_file}")
            offsets = meta.get("data_offsets")
            dtype = meta.get("dtype")
            shape = meta.get("shape")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(isinstance(value, int) for value in offsets)
            ):
                raise ConversionError(f"invalid data_offsets for {name} in {source_file}")
            if not isinstance(dtype, str) or not isinstance(shape, list):
                raise ConversionError(f"invalid dtype/shape for {name} in {source_file}")
            start, end = offsets
            tensors.append(
                {
                    "name": name,
                    "file": dest.name,
                    "offset": data_start + start,
                    "nbytes": end - start,
                    "dtype": dtype.lower(),
                    "shape": shape,
                    "source_file": str(source_file),
                    "source_dtype": dtype,
                }
            )

    manifest["files"] = files
    manifest["tensors"] = tensors
    dtype_counts: dict[str, int] = {}
    language_tensors = 0
    visual_tensors = 0
    mtp_tensors = 0
    for tensor in tensors:
        dtype_counts[tensor["dtype"]] = dtype_counts.get(tensor["dtype"], 0) + 1
        name = tensor["name"]
        if name.startswith("model.language_model."):
            language_tensors += 1
        elif name.startswith("model.visual."):
            visual_tensors += 1
        elif name.startswith("mtp."):
            mtp_tensors += 1
    manifest["tensor_summary"] = {
        "total": len(tensors),
        "dtype_counts": dtype_counts,
        "language_model_tensors": language_tensors,
        "visual_tensors": visual_tensors,
        "mtp_tensors": mtp_tensors,
    }
    return manifest


def _load_config_if_available(args: argparse.Namespace, transformers: Any) -> dict[str, Any] | None:
    try:
        config = transformers.AutoConfig.from_pretrained(
            str(args.input) if args.input else args.model_id,
            revision=args.revision,
            local_files_only=bool(args.input),
            trust_remote_code=True,
        )
    except Exception:
        return None
    if hasattr(config, "to_dict"):
        return config.to_dict()
    return None


def _pack_tensor(torch: Any, tensor: Any, dtype: str) -> tuple[bytes, str]:
    target_dtype = torch.float16 if dtype == "fp16" else torch.float32
    packed = tensor.detach().cpu().contiguous().to(target_dtype)
    return packed.numpy().tobytes(order="C"), dtype


def _convert_real(args: argparse.Namespace) -> dict[str, Any]:
    transformers, safetensors, torch = _require_real_conversion_deps()
    if args.input is None:
        raise ConversionError("real conversion requires --input pointing to local .safetensors weights")

    manifest = _base_manifest(args, dry_run=False)
    if transformers is not None:
        config = _load_config_if_available(args, transformers)
        if config is not None:
            manifest["source"]["config"] = config

    input_files = _find_safetensors_inputs(args.input)
    artifact_root = _artifact_root(args.output)
    artifact_root.mkdir(parents=True, exist_ok=True)
    weights_path = artifact_root / "weights.bin"
    if weights_path.exists() and not args.force:
        raise ConversionError(f"{weights_path} already exists; pass --force to overwrite")

    safe_open = safetensors.safe_open
    offset = 0
    tensors: list[dict[str, Any]] = []
    with weights_path.open("wb") as output:
        for source_file in input_files:
            with safe_open(str(source_file), framework="pt", device="cpu") as handle:
                for name in sorted(handle.keys()):
                    tensor = handle.get_tensor(name)
                    aligned = _align_offset(offset, args.alignment)
                    if aligned > offset:
                        output.write(b"\0" * (aligned - offset))
                        offset = aligned
                    data, packed_dtype = _pack_tensor(torch, tensor, args.dtype)
                    output.write(data)
                    tensors.append(
                        {
                            "name": name,
                            "file": "weights.bin",
                            "offset": offset,
                            "nbytes": len(data),
                            "dtype": packed_dtype,
                            "shape": list(tensor.shape),
                            "source_file": str(source_file),
                            "source_dtype": str(tensor.dtype).replace("torch.", ""),
                        }
                    )
                    offset += len(data)

    manifest["artifact"]["manifest_only"] = False
    manifest["files"] = [
        {
            "path": "weights.bin",
            "size_bytes": weights_path.stat().st_size,
            "sha256": _sha256_file(weights_path),
        }
    ]
    manifest["tensors"] = tensors
    manifest["notes"].append("Packed local safetensors tensors in lexicographic name order.")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a Spock packed artifact manifest for Qwen 3.5 0.8B.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(DEFAULT_ARTIFACT_NAME),
        help="artifact directory, or a .json path for manifest-only output",
    )
    parser.add_argument("--input", type=Path, help="local Hugging Face model directory or safetensors path")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="source model identifier")
    parser.add_argument("--revision", default=None, help="source model revision to record in the manifest")
    parser.add_argument("--artifact-name", default=DEFAULT_ARTIFACT_NAME, help="artifact name recorded in manifest")
    parser.add_argument("--dtype", default="fp16", choices=("fp16", "fp32", "source"), help="packed tensor storage dtype")
    parser.add_argument("--alignment", type=int, default=256, help="byte alignment for packed tensors")
    parser.add_argument("--dry-run", action="store_true", help="write a manifest-only artifact without ML imports")
    parser.add_argument("--offline", action="store_true", help="alias for --dry-run; never imports torch")
    parser.add_argument(
        "--safetensors-scan",
        action="store_true",
        help="create a real artifact manifest from safetensors headers without torch or dtype conversion",
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing manifest")
    parser.add_argument(
        "--print-manifest",
        action="store_true",
        help="also print the generated manifest JSON to stdout",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    dry_run = bool(args.dry_run or args.offline)

    try:
        _validate_dry_run_args(args)
        manifest = _base_manifest(args, dry_run=dry_run)
        if args.safetensors_scan:
            manifest = _convert_safetensors_scan(args)
        elif dry_run:
            manifest["notes"].append("Manifest-only artifact emitted without importing torch.")
        else:
            manifest = _convert_real(args)

        manifest_path = _write_manifest(args.output, manifest, force=args.force)
        if args.print_manifest:
            print(json.dumps(manifest, indent=2, sort_keys=True))
        print(f"wrote manifest: {manifest_path}", file=sys.stderr)
        return 0
    except ConversionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

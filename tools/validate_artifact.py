#!/usr/bin/env python3
"""Validate a Spock packed artifact manifest.

This validator uses only the Python standard library. It accepts either an
artifact directory containing manifest.json or a direct path to a manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
REQUIRED_TOP_LEVEL = {
    "schema_version",
    "artifact",
    "source",
    "model",
    "packing",
    "files",
    "tensors",
    "layers",
}
REQUIRED_MODEL_FIELDS = {
    "architecture",
    "layers",
    "layer_pattern",
    "hidden_size",
    "intermediate_size",
    "attention",
    "deltanet",
    "max_sequence_length",
}
REQUIRED_PACKING_FIELDS = {"alignment", "byte_order", "storage_dtype", "offset_units"}


class ValidationError(RuntimeError):
    """Raised when the artifact is invalid."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_path(path: Path) -> Path:
    if path.is_dir():
        return path / "manifest.json"
    return path


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest_file = _manifest_path(path)
    if not manifest_file.exists():
        raise ValidationError(f"manifest not found: {manifest_file}")
    try:
        data = json.loads(manifest_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid JSON in {manifest_file}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValidationError("manifest root must be a JSON object")
    return data


def _require_keys(obj: dict[str, Any], keys: set[str], context: str) -> list[str]:
    errors = []
    missing = sorted(keys - set(obj))
    if missing:
        errors.append(f"{context} missing required keys: {', '.join(missing)}")
    return errors


def _validate_schema(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    errors.extend(_require_keys(manifest, REQUIRED_TOP_LEVEL, "manifest"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")

    artifact = manifest.get("artifact")
    if not isinstance(artifact, dict):
        errors.append("artifact must be an object")
    else:
        for key in ("name", "format", "created_at_utc", "generator", "dry_run", "manifest_only"):
            if key not in artifact:
                errors.append(f"artifact missing required key: {key}")
        if artifact.get("format") != "spock-packed-model":
            errors.append("artifact.format must be spock-packed-model")

    source = manifest.get("source")
    if not isinstance(source, dict):
        errors.append("source must be an object")
    elif "model_id" not in source:
        errors.append("source missing required key: model_id")

    model = manifest.get("model")
    if not isinstance(model, dict):
        errors.append("model must be an object")
    else:
        errors.extend(_require_keys(model, REQUIRED_MODEL_FIELDS, "model"))
        if isinstance(model.get("layers"), int) and isinstance(manifest.get("layers"), list):
            if model["layers"] != len(manifest["layers"]):
                errors.append("model.layers must match length of layers array")

    packing = manifest.get("packing")
    if not isinstance(packing, dict):
        errors.append("packing must be an object")
    else:
        errors.extend(_require_keys(packing, REQUIRED_PACKING_FIELDS, "packing"))
        alignment = packing.get("alignment")
        if not isinstance(alignment, int) or alignment <= 0 or alignment & (alignment - 1):
            errors.append("packing.alignment must be a positive power of two")
        if packing.get("byte_order") != "little":
            errors.append("packing.byte_order must be little")
        if packing.get("offset_units") != "bytes":
            errors.append("packing.offset_units must be bytes")
        if packing.get("storage_dtype") not in {"fp16", "fp32", "source"}:
            errors.append("packing.storage_dtype must be fp16, fp32, or source")

    for key in ("files", "tensors", "layers"):
        if key in manifest and not isinstance(manifest[key], list):
            errors.append(f"{key} must be an array")

    summary = manifest.get("tensor_summary")
    if summary is not None:
        if not isinstance(summary, dict):
            errors.append("tensor_summary must be an object")
        elif summary.get("total") != len(manifest.get("tensors", [])):
            errors.append("tensor_summary.total must match tensors length")
    return errors


def _validate_layers(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    layers = manifest.get("layers", [])
    if not isinstance(layers, list):
        return errors
    seen = set()
    for expected_index, layer in enumerate(layers):
        if not isinstance(layer, dict):
            errors.append(f"layers[{expected_index}] must be an object")
            continue
        index = layer.get("index")
        if index != expected_index:
            errors.append(f"layers[{expected_index}].index must be {expected_index}")
        if index in seen:
            errors.append(f"duplicate layer index: {index}")
        seen.add(index)
        if layer.get("kind") not in {"deltanet", "attention"}:
            errors.append(f"layers[{expected_index}].kind must be deltanet or attention")
        if not isinstance(layer.get("weight_groups"), list):
            errors.append(f"layers[{expected_index}].weight_groups must be an array")
    return errors


def _validate_files(manifest: dict[str, Any], artifact_root: Path, *, check_hashes: bool) -> list[str]:
    errors: list[str] = []
    files = manifest.get("files", [])
    if not isinstance(files, list):
        return errors
    for index, entry in enumerate(files):
        if not isinstance(entry, dict):
            errors.append(f"files[{index}] must be an object")
            continue
        rel_path = entry.get("path")
        if not isinstance(rel_path, str) or not rel_path:
            errors.append(f"files[{index}].path must be a non-empty string")
            continue
        if Path(rel_path).is_absolute() or ".." in Path(rel_path).parts:
            errors.append(f"files[{index}].path must be relative and stay within the artifact")
            continue
        file_path = artifact_root / rel_path
        if not file_path.exists():
            errors.append(f"files[{index}] missing on disk: {rel_path}")
            continue
        size = entry.get("size_bytes")
        if size is not None and size != file_path.stat().st_size:
            errors.append(f"files[{index}].size_bytes does not match {rel_path}")
        sha256 = entry.get("sha256")
        if check_hashes and sha256 and sha256 != _sha256_file(file_path):
            errors.append(f"files[{index}].sha256 does not match {rel_path}")
    return errors


def _validate_tensors(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    tensors = manifest.get("tensors", [])
    files = {entry.get("path") for entry in manifest.get("files", []) if isinstance(entry, dict)}
    file_sizes = {
        entry.get("path"): entry.get("size_bytes")
        for entry in manifest.get("files", [])
        if isinstance(entry, dict) and isinstance(entry.get("size_bytes"), int)
    }
    alignment = manifest.get("packing", {}).get("alignment", 1)
    ranges_by_file: dict[str, list[tuple[int, int, str]]] = {}

    if not isinstance(tensors, list):
        return errors
    for index, tensor in enumerate(tensors):
        if not isinstance(tensor, dict):
            errors.append(f"tensors[{index}] must be an object")
            continue
        name = tensor.get("name")
        if not isinstance(name, str) or not name:
            errors.append(f"tensors[{index}].name must be a non-empty string")
            name = f"<tensor {index}>"
        for key in ("file", "offset", "nbytes", "dtype", "shape"):
            if key not in tensor:
                errors.append(f"tensors[{index}] missing required key: {key}")
        file_name = tensor.get("file")
        if file_name is not None and file_name not in files:
            errors.append(f"tensors[{index}].file is not listed in files: {file_name}")
        offset = tensor.get("offset")
        nbytes = tensor.get("nbytes")
        if not isinstance(offset, int) or offset < 0:
            errors.append(f"tensors[{index}].offset must be a non-negative integer")
        if not isinstance(nbytes, int) or nbytes <= 0:
            errors.append(f"tensors[{index}].nbytes must be a positive integer")
        if isinstance(offset, int) and isinstance(alignment, int) and offset % alignment:
            errors.append(f"tensors[{index}].offset is not aligned to {alignment}")
        if isinstance(file_name, str) and isinstance(offset, int) and isinstance(nbytes, int) and nbytes > 0:
            ranges_by_file.setdefault(file_name, []).append((offset, offset + nbytes, name))
            size = file_sizes.get(file_name)
            if isinstance(size, int) and offset + nbytes > size:
                errors.append(f"tensors[{index}] range exceeds file size for {file_name}")
        if tensor.get("dtype") not in {"bf16", "f16", "f32", "fp16", "fp32", "i32", "u32"}:
            errors.append(f"tensors[{index}].dtype must be one of bf16, f16, f32, fp16, fp32, i32, u32")
        shape = tensor.get("shape")
        if not isinstance(shape, list) or not all(isinstance(dim, int) and dim > 0 for dim in shape):
            errors.append(f"tensors[{index}].shape must be an array of positive integers")

    for file_name, ranges in ranges_by_file.items():
        ranges.sort()
        previous_end = 0
        previous_name = None
        for start, end, name in ranges:
            if previous_name is not None and start < previous_end:
                errors.append(f"tensor range overlap in {file_name}: {previous_name} overlaps {name}")
            previous_end = max(previous_end, end)
            previous_name = name
    return errors


def validate(path: Path, *, check_hashes: bool) -> list[str]:
    manifest_file = _manifest_path(path)
    artifact_root = manifest_file.parent
    manifest = _load_manifest(path)
    errors = []
    errors.extend(_validate_schema(manifest))
    errors.extend(_validate_layers(manifest))
    errors.extend(_validate_files(manifest, artifact_root, check_hashes=check_hashes))
    errors.extend(_validate_tensors(manifest))
    return errors


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a Spock packed artifact manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("artifact", type=Path, help="artifact directory or manifest.json path")
    parser.add_argument("--check-hashes", action="store_true", help="verify file SHA-256 entries when present")
    parser.add_argument("--json", action="store_true", help="emit validation result as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        errors = validate(args.artifact, check_hashes=args.check_hashes)
    except ValidationError as exc:
        errors = [str(exc)]

    if args.json:
        print(json.dumps({"valid": not errors, "errors": errors}, indent=2, sort_keys=True))
    elif errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
    else:
        print("artifact manifest is valid")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

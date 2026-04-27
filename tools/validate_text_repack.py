#!/usr/bin/env python3
"""Validate a text repack manifest produced by repack_text_weights.py."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


class ValidationError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_path(path: Path) -> Path:
    if path.is_dir():
        return path / "text_repack_manifest.json"
    return path


def validate(path: Path, *, check_hashes: bool) -> list[str]:
    manifest_path = _manifest_path(path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValidationError(f"unable to read manifest: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid JSON: {manifest_path}: {exc}") from exc

    errors: list[str] = []
    root = manifest_path.parent
    if manifest.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if manifest.get("artifact", {}).get("format") != "spock-text-repacked-weights":
        errors.append("artifact.format must be spock-text-repacked-weights")
    alignment = manifest.get("packing", {}).get("alignment")
    if not isinstance(alignment, int) or alignment <= 0 or alignment & (alignment - 1):
        errors.append("packing.alignment must be a positive power of two")

    files = manifest.get("files", [])
    if not isinstance(files, list) or len(files) != 1:
        errors.append("manifest must contain exactly one weights file")
        files = []
    file_sizes = {}
    for file_entry in files:
        rel = file_entry.get("path")
        if not isinstance(rel, str) or Path(rel).is_absolute() or ".." in Path(rel).parts:
            errors.append("file path must be relative")
            continue
        file_path = root / rel
        if not file_path.exists():
            errors.append(f"weights file missing: {rel}")
            continue
        size = file_path.stat().st_size
        file_sizes[rel] = size
        if file_entry.get("size_bytes") != size:
            errors.append("weights file size mismatch")
        if check_hashes and file_entry.get("sha256") != _sha256_file(file_path):
            errors.append("weights file sha256 mismatch")

    tensors = manifest.get("tensors", [])
    if not isinstance(tensors, list):
        errors.append("tensors must be an array")
        tensors = []
    seen_roles = set()
    ranges_by_file: dict[str, list[tuple[int, int, str]]] = {}
    for index, tensor in enumerate(tensors):
        role = tensor.get("role_path")
        if not isinstance(role, str) or not role:
            errors.append(f"tensors[{index}].role_path must be a non-empty string")
            role = f"<tensor {index}>"
        if role in seen_roles:
            errors.append(f"duplicate role_path: {role}")
        seen_roles.add(role)
        dtype = tensor.get("dtype")
        if dtype not in {"fp16", "fp32"}:
            errors.append(f"tensors[{index}].dtype must be fp16 or fp32")
        source_dtype = tensor.get("source_dtype")
        if dtype == "fp16" and source_dtype != "bf16":
            errors.append(f"tensors[{index}] fp16 output must come from bf16 source")
        if dtype == "fp32" and source_dtype != "f32":
            errors.append(f"tensors[{index}] fp32 output must come from f32 source")
        offset = tensor.get("offset")
        nbytes = tensor.get("nbytes")
        file_name = tensor.get("file")
        if not isinstance(offset, int) or offset < 0:
            errors.append(f"tensors[{index}].offset must be non-negative integer")
        elif isinstance(alignment, int) and offset % alignment:
            errors.append(f"tensors[{index}].offset not aligned to {alignment}")
        if not isinstance(nbytes, int) or nbytes <= 0:
            errors.append(f"tensors[{index}].nbytes must be positive integer")
        if isinstance(file_name, str) and isinstance(offset, int) and isinstance(nbytes, int):
            ranges_by_file.setdefault(file_name, []).append((offset, offset + nbytes, role))
            size = file_sizes.get(file_name)
            if isinstance(size, int) and offset + nbytes > size:
                errors.append(f"tensors[{index}] range exceeds file size")

    for file_name, ranges in ranges_by_file.items():
        ranges.sort()
        previous_end = 0
        previous_role = None
        for start, end, role in ranges:
            if previous_role is not None and start < previous_end:
                errors.append(f"tensor range overlap in {file_name}: {previous_role} overlaps {role}")
            previous_end = max(previous_end, end)
            previous_role = role

    summary = manifest.get("summary", {})
    if isinstance(summary, dict):
        if summary.get("tensor_count") != len(tensors):
            errors.append("summary.tensor_count must match tensors length")
    else:
        errors.append("summary must be an object")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path, help="repack directory or text_repack_manifest.json")
    parser.add_argument("--check-hashes", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
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
        print("text repack manifest is valid")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

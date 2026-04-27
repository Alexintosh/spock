#!/usr/bin/env python3
"""Repack a text-only load plan into aligned runtime weight buffers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable


F32_PRESERVE_ROLES = {"delta_a_log", "delta_norm"}


class RepackError(RuntimeError):
    """Raised for invalid inputs or unsupported repack operations."""


@dataclass(frozen=True)
class TensorTask:
    role_path: str
    name: str
    source_file: Path
    source_offset: int
    source_nbytes: int
    source_dtype: str
    output_dtype: str
    shape: list[int]


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RepackError(f"unable to read {path}") from exc
    except json.JSONDecodeError as exc:
        raise RepackError(f"invalid JSON in {path}: {exc}") from exc


def _artifact_root(path: Path) -> Path:
    return path if path.is_dir() else path.parent


def _iter_tasks(plan: dict, source_root: Path) -> Iterable[TensorTask]:
    globals_ = plan.get("global_tensors", {})
    token_embedding = globals_.get("token_embedding")
    final_norm = globals_.get("final_norm")
    if not isinstance(token_embedding, dict) or not isinstance(final_norm, dict):
        raise RepackError("plan missing global token_embedding or final_norm")

    yield _task_from_ref("global.token_embedding", token_embedding, source_root, force_output_dtype="fp16")
    yield _task_from_ref("global.final_norm", final_norm, source_root, force_output_dtype="fp16")

    layers = plan.get("layers", [])
    if not isinstance(layers, list):
        raise RepackError("plan.layers must be an array")
    for layer in layers:
        index = layer.get("index")
        roles = layer.get("roles")
        if not isinstance(index, int) or not isinstance(roles, dict):
            raise RepackError("every layer must have integer index and role map")
        for role in sorted(roles):
            output_dtype = "fp32" if role in F32_PRESERVE_ROLES else "fp16"
            yield _task_from_ref(f"layer.{index}.{role}", roles[role], source_root, force_output_dtype=output_dtype)


def _task_from_ref(role_path: str, ref: dict, source_root: Path, *, force_output_dtype: str) -> TensorTask:
    for key in ("name", "file", "offset", "nbytes", "dtype", "shape"):
        if key not in ref:
            raise RepackError(f"{role_path} missing {key}")
    return TensorTask(
        role_path=role_path,
        name=ref["name"],
        source_file=source_root / ref["file"],
        source_offset=int(ref["offset"]),
        source_nbytes=int(ref["nbytes"]),
        source_dtype=str(ref["dtype"]).lower(),
        output_dtype=force_output_dtype,
        shape=list(ref["shape"]),
    )


def _bf16_bits_to_float(value: int) -> float:
    return struct.unpack("<f", struct.pack("<I", value << 16))[0]


def _float_to_fp16_bits(value: float) -> int:
    if math.isnan(value):
        return 0x7E00
    if math.isinf(value):
        return 0xFC00 if value < 0 else 0x7C00

    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    sign = (bits >> 16) & 0x8000
    exponent = ((bits >> 23) & 0xFF) - 127 + 15
    mantissa = bits & 0x7FFFFF

    if exponent <= 0:
        if exponent < -10:
            return sign
        mantissa |= 0x800000
        shift = 14 - exponent
        half_mantissa = mantissa >> shift
        if (mantissa >> (shift - 1)) & 1:
            half_mantissa += 1
        return sign | half_mantissa

    if exponent >= 31:
        return sign | 0x7C00

    half = sign | (exponent << 10) | (mantissa >> 13)
    if mantissa & 0x1000:
        half += 1
    return half & 0xFFFF


def _convert_bf16_to_fp16_chunk(chunk: bytes) -> bytes:
    if len(chunk) % 2:
        raise RepackError("BF16 chunk length must be even")
    out = bytearray(len(chunk))
    for index in range(0, len(chunk), 2):
        bf16 = int.from_bytes(chunk[index : index + 2], "little")
        fp16 = _float_to_fp16_bits(_bf16_bits_to_float(bf16))
        out[index : index + 2] = fp16.to_bytes(2, "little")
    return bytes(out)


def _copy_f32_chunk(chunk: bytes) -> bytes:
    if len(chunk) % 4:
        raise RepackError("F32 chunk length must be a multiple of four")
    return chunk


def _write_converted(task: TensorTask, output: BinaryIO, *, chunk_bytes: int) -> None:
    if not task.source_file.exists():
        raise RepackError(f"source file missing for {task.role_path}: {task.source_file}")
    if task.source_dtype == "bf16" and task.output_dtype != "fp16":
        raise RepackError(f"unsupported BF16 output dtype for {task.role_path}: {task.output_dtype}")
    if task.source_dtype == "f32" and task.output_dtype != "fp32":
        raise RepackError(f"unsupported F32 output dtype for {task.role_path}: {task.output_dtype}")

    with task.source_file.open("rb") as source:
        source.seek(task.source_offset)
        remaining = task.source_nbytes
        while remaining:
            read_size = min(remaining, chunk_bytes)
            if task.source_dtype in {"bf16", "f32"} and read_size % (2 if task.source_dtype == "bf16" else 4):
                read_size -= read_size % (2 if task.source_dtype == "bf16" else 4)
            chunk = source.read(read_size)
            if len(chunk) != read_size:
                raise RepackError(f"short read for {task.role_path}")
            if task.source_dtype == "bf16":
                output.write(_convert_bf16_to_fp16_chunk(chunk))
            elif task.source_dtype == "f32":
                output.write(_copy_f32_chunk(chunk))
            else:
                raise RepackError(f"unsupported source dtype for {task.role_path}: {task.source_dtype}")
            remaining -= read_size


def repack(plan_path: Path, output_dir: Path, *, alignment: int, chunk_bytes: int, force: bool) -> Path:
    if alignment <= 0 or alignment & (alignment - 1):
        raise RepackError("--alignment must be a positive power of two")
    if chunk_bytes < 4096:
        raise RepackError("--chunk-bytes must be at least 4096")

    plan = _load_json(plan_path)
    source_root = _artifact_root(plan_path)
    tasks = list(_iter_tasks(plan, source_root))

    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "text_weights.bin"
    manifest_path = output_dir / "text_repack_manifest.json"
    if not force and (weights_path.exists() or manifest_path.exists()):
        raise RepackError(f"{output_dir} already contains repack output; pass --force")

    offset = 0
    tensors: list[dict] = []
    with weights_path.open("wb") as output:
        for task in tasks:
            aligned = _align(offset, alignment)
            if aligned > offset:
                output.write(b"\0" * (aligned - offset))
                offset = aligned
            nbytes = task.source_nbytes
            _write_converted(task, output, chunk_bytes=chunk_bytes)
            tensors.append(
                {
                    "role_path": task.role_path,
                    "name": task.name,
                    "file": weights_path.name,
                    "offset": offset,
                    "nbytes": nbytes,
                    "dtype": task.output_dtype,
                    "shape": task.shape,
                    "source_dtype": task.source_dtype,
                    "source_file": str(task.source_file),
                    "source_offset": task.source_offset,
                    "source_nbytes": task.source_nbytes,
                }
            )
            offset += nbytes

    manifest = {
        "schema_version": 1,
        "artifact": {
            "format": "spock-text-repacked-weights",
            "source_plan": str(plan_path),
        },
        "packing": {
            "alignment": alignment,
            "byte_order": "little",
            "weights_file": weights_path.name,
        },
        "files": [
            {
                "path": weights_path.name,
                "size_bytes": weights_path.stat().st_size,
                "sha256": _sha256_file(weights_path),
            }
        ],
        "tensors": tensors,
        "summary": {
            "tensor_count": len(tensors),
            "fp16_tensors": sum(1 for tensor in tensors if tensor["dtype"] == "fp16"),
            "fp32_tensors": sum(1 for tensor in tensors if tensor["dtype"] == "fp32"),
            "size_bytes": weights_path.stat().st_size,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Repack a text-only Spock load plan into runtime weights.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("plan", type=Path, help="text_plan.json produced by plan_text_artifact.py")
    parser.add_argument("-o", "--output-dir", type=Path, required=True, help="output artifact directory")
    parser.add_argument("--alignment", type=int, default=256, help="byte alignment for packed tensors")
    parser.add_argument("--chunk-bytes", type=int, default=1024 * 1024, help="streaming conversion chunk size")
    parser.add_argument("--force", action="store_true", help="overwrite existing output files")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        manifest = repack(
            args.plan,
            args.output_dir,
            alignment=args.alignment,
            chunk_bytes=args.chunk_bytes,
            force=args.force,
        )
    except RepackError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote repack manifest: {manifest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

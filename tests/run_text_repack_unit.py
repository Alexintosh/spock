#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import tempfile
from pathlib import Path


def bf16_bytes(value: float) -> bytes:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return (bits >> 16).to_bytes(2, "little")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repacker", required=True, type=Path)
    parser.add_argument("--native-repacker", type=Path)
    parser.add_argument("--validator", required=True, type=Path)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="spock-repack-unit-") as tmp:
        root = Path(tmp)
        source = root / "source.bin"
        source.write_bytes(
            bf16_bytes(1.0)
            + bf16_bytes(-2.0)
            + struct.pack("<f", 3.5)
            + bf16_bytes(0.5)
        )
        plan = {
            "global_tensors": {
                "token_embedding": {
                    "name": "embed",
                    "file": "source.bin",
                    "offset": 0,
                    "nbytes": 4,
                    "dtype": "bf16",
                    "shape": [2],
                },
                "final_norm": {
                    "name": "final_norm",
                    "file": "source.bin",
                    "offset": 8,
                    "nbytes": 2,
                    "dtype": "bf16",
                    "shape": [1],
                },
            },
            "layers": [
                {
                    "index": 0,
                    "roles": {
                        "delta_a_log": {
                            "name": "a_log",
                            "file": "source.bin",
                            "offset": 4,
                            "nbytes": 4,
                            "dtype": "f32",
                            "shape": [1],
                        }
                    },
                }
            ],
        }
        plan_path = root / "text_plan.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        out = root / "out"
        repack = subprocess.run(
            [
                "python3",
                str(args.repacker),
                str(plan_path),
                "--output-dir",
                str(out),
                "--alignment",
                "4",
                "--force",
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if repack.returncode:
            print(repack.stdout)
            print(repack.stderr)
            return repack.returncode
        validate = subprocess.run(
            ["python3", str(args.validator), str(out), "--check-hashes", "--json"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if validate.returncode:
            print(validate.stdout)
            print(validate.stderr)
            return validate.returncode

        manifest = json.loads((out / "text_repack_manifest.json").read_text(encoding="utf-8"))
        if manifest["summary"]["tensor_count"] != 3:
            raise SystemExit("wrong tensor count")
        packed = (out / "text_weights.bin").read_bytes()
        if packed[0:2] != b"\x00\x3c":
            raise SystemExit("expected 1.0 to convert to fp16 0x3c00")
        if packed[2:4] != b"\x00\xc0":
            raise SystemExit("expected -2.0 to convert to fp16 0xc000")
        print("synthetic text repack valid")
        if args.native_repacker is not None:
            tasks = root / "tasks.tsv"
            tasks.write_text(
                "\n".join(
                    [
                        "role_path\tname\tfile\toffset\tnbytes\tsource_dtype\toutput_dtype\tshape",
                        "global.token_embedding\tembed\tsource.bin\t0\t4\tbf16\tfp16\t[2]",
                        "layer.0.delta_a_log\ta_log\tsource.bin\t4\t4\tf32\tfp32\t[1]",
                        "global.final_norm\tfinal_norm\tsource.bin\t8\t2\tbf16\tfp16\t[1]",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            native_out = root / "native-out"
            native = subprocess.run(
                [
                    str(args.native_repacker),
                    "--tasks",
                    str(tasks),
                    "--source-root",
                    str(root),
                    "--output-dir",
                    str(native_out),
                    "--alignment",
                    "4",
                ],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if native.returncode:
                print(native.stdout)
                print(native.stderr)
                return native.returncode
            native_validate = subprocess.run(
                ["python3", str(args.validator), str(native_out), "--json"],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if native_validate.returncode:
                print(native_validate.stdout)
                print(native_validate.stderr)
                return native_validate.returncode
            native_packed = (native_out / "text_weights.bin").read_bytes()
            if native_packed[0:4] != packed[0:4]:
                raise SystemExit("native BF16 to FP16 output differs from Python reference")
            print("native synthetic text repack valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

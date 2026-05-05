#!/usr/bin/env python3
"""Standalone unit checks for run_chunked_decode_sweep GPU timestamp helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def import_sweep():
    repo_root = Path(__file__).resolve().parents[1]
    tool_path = repo_root / "tools" / "run_chunked_decode_sweep.py"
    spec = importlib.util.spec_from_file_location(
        "run_chunked_decode_sweep", tool_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_close(actual, expected, epsilon=1e-9):
    assert abs(actual - expected) <= epsilon, (actual, expected)


def main():
    sweep = import_sweep()

    env = sweep.build_env(4)
    assert env["SPOCK_GPU_DECODE_CHUNK_SIZE"] == "4"
    assert env["SPOCK_GPU_CHUNKED_DECODE"] == "1"
    assert env.get("SPOCK_GPU_TIMESTAMPS") != "1"

    env = sweep.build_env(8, gpu_timestamps=True)
    assert env["SPOCK_GPU_DECODE_CHUNK_SIZE"] == "8"
    assert env["SPOCK_GPU_TIMESTAMPS"] == "1"

    decoded = {
        "decode_submit_count": 2,
        "chunked_decode_submit_count": 1,
        "generated_count": 4,
        "elapsed_ms": 100.0,
        "prefill_ms": 10.0,
        "decode_ms": 90.0,
        "gpu_decode_us": 80000,
        "per_token_gpu_us": [20000, 20001, 20002, 20003],
    }
    rec = sweep._make_run_record(
        "test", 4, 0, decoded, True, gpu_timestamps=True,
    )
    assert rec["gpu_decode_us"] == 80000
    assert rec["per_token_gpu_us_count"] == 4
    assert_close(rec["per_token_gpu_us_mean"], 20001.5)
    assert rec["per_token_gpu_us_min"] == 20000
    assert rec["per_token_gpu_us_max"] == 20003

    no_gpu_rec = sweep._make_run_record(
        "test", 4, 0, decoded, True, gpu_timestamps=False,
    )
    assert "gpu_decode_us" not in no_gpu_rec
    assert "per_token_gpu_us_count" not in no_gpu_rec

    agg = sweep._make_aggregate_record(
        "test", 4, True, 2,
        [
            {
                "elapsed_ms": 100.0,
                "prefill_ms": 10.0,
                "decode_ms": 90.0,
                "gpu_decode_us": 80000,
                "per_token_gpu_us_mean": 5000.0,
            },
            {
                "elapsed_ms": 102.0,
                "prefill_ms": 11.0,
                "decode_ms": 91.0,
                "gpu_decode_us": 82000,
                "per_token_gpu_us_mean": 5100.0,
            },
        ],
        gpu_timestamps=True,
    )
    assert_close(agg["gpu_decode_us_mean"], 81000.0)
    assert agg["gpu_decode_us_min"] == 80000
    assert agg["gpu_decode_us_max"] == 82000
    assert_close(agg["per_token_gpu_us_mean_mean"], 5050.0)
    assert_close(agg["per_token_gpu_us_mean_min"], 5000.0)
    assert_close(agg["per_token_gpu_us_mean_max"], 5100.0)

    assert sweep._gpu_timestamp_validation_error(decoded, 4) is None
    missing_gpu = dict(decoded)
    missing_gpu["gpu_decode_us"] = 0
    assert "gpu_decode_us" in sweep._gpu_timestamp_validation_error(
        missing_gpu, 4,
    )
    bad_len = dict(decoded)
    bad_len["generated_count"] = None
    bad_len["per_token_gpu_us"] = [1, 2, 3]
    assert "expected token count 4" in sweep._gpu_timestamp_validation_error(
        bad_len, 4,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

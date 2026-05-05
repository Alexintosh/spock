#!/usr/bin/env python3
"""Sweep chunked decode sizes against reference token IDs.

Runs build/spock-decode (or --decode override) against
tests/data/reference_tokens.jsonl for one or more --ids, sweeping
--chunk-sizes.  For each chunk size the full fast-path env stack is
applied plus SPOCK_GPU_CHUNKED_DECODE=1 and
SPOCK_GPU_DECODE_CHUNK_SIZE=N.  Supports --warmup-runs and
--timed-runs for controlled repeated host timing.  With
--gpu-timestamps, also sets SPOCK_GPU_TIMESTAMPS=1 and includes
GPU timing fields from spock-decode JSON in per-run and aggregate
records.  Emits a JSON summary and exits nonzero on any decode
failure or token mismatch.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Fast-path environment gates (from diary 0057)
# ---------------------------------------------------------------------------
FAST_ENV_GATES = {
    "SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS": "1",
    "SPOCK_GPU_MERGED_DELTANET": "1",
    "SPOCK_GPU_FUSED_DN_CONV_L2": "1",
    "SPOCK_GPU_FUSED_DN_GBETA_RECURRENT": "1",
    "SPOCK_GPU_FUSED_DN_REC_NORM_GATE": "1",
    "SPOCK_GPU_SINGLE_SUBMIT": "1",
    "SPOCK_GPU_DEVICE_RESIDENT_TOKEN": "1",
    "SPOCK_GPU_DEFER_TOKEN_DOWNLOAD": "1",
    "SPOCK_GPU_MATVEC_TILED": "1",
    "SPOCK_GPU_LM_HEAD_TILED": "1",
}

CHUNKED_ENV_GATES = {
    "SPOCK_GPU_CHUNKED_DECODE": "1",
}

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DECODE = REPO_ROOT / "build" / "spock-decode"
DEFAULT_REFERENCE = REPO_ROOT / "tests" / "data" / "reference_tokens.jsonl"


# ---------------------------------------------------------------------------
# Helpers (mirroring tests/run_vk_decode_parity.py patterns)
# ---------------------------------------------------------------------------

def load_entries(path, ids):
    """Load JSONL entries, optionally filtering by id set."""
    wanted = {item for item in ids if item} if ids else None
    entries = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if wanted is not None and entry.get("id") not in wanted:
                continue
            entries.append(entry)
    return entries


def parse_decode_json(stdout):
    """Extract the first JSON object from spock-decode stdout."""
    start = stdout.find("{")
    if start < 0:
        raise ValueError("decode output did not contain JSON")
    return json.loads(stdout[start:])


def git_short_rev():
    """Return git short rev or None."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(REPO_ROOT),
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except FileNotFoundError:
        pass
    return None


def build_env(chunk_size, gpu_timestamps=False):
    """Return a full env dict with fast-path + chunked decode gates."""
    env = os.environ.copy()
    env.update(FAST_ENV_GATES)
    env.update(CHUNKED_ENV_GATES)
    env["SPOCK_GPU_DECODE_CHUNK_SIZE"] = str(chunk_size)
    if gpu_timestamps:
        env["SPOCK_GPU_TIMESTAMPS"] = "1"
    return env


def _make_error_record(entry_id, chunk_size, error_fields):
    """Build an error result dict with standard fields."""
    rec = {
        "id": entry_id,
        "chunk_size": chunk_size,
        "match": False,
    }
    rec.update(error_fields)
    return rec


def _make_run_record(entry_id, chunk_size, run_index, decoded, match,
                     gpu_timestamps=False):
    """Build a per-run result dict."""
    rec = {
        "id": entry_id,
        "chunk_size": chunk_size,
        "run_index": run_index,
        "match": match,
        "decode_submit_count": decoded.get("decode_submit_count"),
        "chunked_decode_submit_count": decoded.get(
            "chunked_decode_submit_count"
        ),
        "generated_count": decoded.get("generated_count"),
        "elapsed_ms": decoded.get("elapsed_ms"),
        "prefill_ms": decoded.get("prefill_ms"),
        "decode_ms": decoded.get("decode_ms"),
    }
    if gpu_timestamps:
        rec["gpu_decode_us"] = decoded.get("gpu_decode_us")
        ptgu = decoded.get("per_token_gpu_us")
        if ptgu is not None and isinstance(ptgu, list) and len(ptgu) > 0:
            rec["per_token_gpu_us_count"] = len(ptgu)
            rec["per_token_gpu_us_mean"] = statistics.mean(ptgu)
            rec["per_token_gpu_us_min"] = min(ptgu)
            rec["per_token_gpu_us_max"] = max(ptgu)
    return rec


def _make_aggregate_record(entry_id, chunk_size, all_match, timed_runs,
                           run_records, gpu_timestamps=False):
    """Build an aggregate record over timed run records.

    Computes mean/min/max for elapsed_ms, prefill_ms, decode_ms.
    When gpu_timestamps is True, also aggregates gpu_decode_us and
    per_token_gpu_us_mean.
    """
    agg = {
        "id": entry_id,
        "chunk_size": chunk_size,
        "aggregate": True,
        "match": all_match,
        "timed_runs": timed_runs,
    }

    for field in ("decode_submit_count", "chunked_decode_submit_count",
                  "generated_count"):
        vals = [r.get(field) for r in run_records]
        if vals and all(v is not None and v == vals[0] for v in vals):
            agg[field] = vals[0]
        else:
            agg[field] = None

    for field in ("elapsed_ms", "prefill_ms", "decode_ms"):
        vals = [r[field] for r in run_records if r.get(field) is not None]
        if vals:
            agg[f"{field}_mean"] = statistics.mean(vals)
            agg[f"{field}_min"] = min(vals)
            agg[f"{field}_max"] = max(vals)
        else:
            agg[f"{field}_mean"] = None
            agg[f"{field}_min"] = None
            agg[f"{field}_max"] = None

    if gpu_timestamps:
        for field in ("gpu_decode_us", "per_token_gpu_us_mean"):
            vals = [r[field] for r in run_records if r.get(field) is not None]
            if vals:
                agg[f"{field}_mean"] = statistics.mean(vals)
                agg[f"{field}_min"] = min(vals)
                agg[f"{field}_max"] = max(vals)
            else:
                agg[f"{field}_mean"] = None
                agg[f"{field}_min"] = None
                agg[f"{field}_max"] = None

    return agg


def _run_decode(decode_exe, repack_dir, token_path, max_new_tokens, env):
    """Run spock-decode and return (returncode, stdout, stderr)."""
    proc = subprocess.run(
        [
            str(decode_exe),
            "--repack-dir",
            repack_dir,
            "--tokens",
            str(token_path),
            "--max-new-tokens",
            str(max_new_tokens),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _parse_and_validate(stdout):
    """Parse decode JSON stdout, return decoded dict or raise."""
    return parse_decode_json(stdout)


def _gpu_timestamp_validation_error(decoded, max_new_tokens):
    """Return an error string when GPU timestamp fields are invalid."""
    gpu_decode_us = decoded.get("gpu_decode_us")
    if gpu_decode_us is None or gpu_decode_us <= 0:
        return f"gpu_decode_us missing or non-positive: {gpu_decode_us!r}"

    per_token_gpu_us = decoded.get("per_token_gpu_us")
    if not isinstance(per_token_gpu_us, list):
        return "per_token_gpu_us missing or not a list"

    generated_count = decoded.get("generated_count")
    expected_count = (
        generated_count if generated_count is not None else max_new_tokens
    )
    if len(per_token_gpu_us) != expected_count:
        return (
            f"per_token_gpu_us length {len(per_token_gpu_us)} != "
            f"expected token count {expected_count}"
        )

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sweep chunked decode sizes and compare against reference tokens",
    )
    parser.add_argument(
        "--decode",
        default=str(DEFAULT_DECODE),
        help="Path to spock-decode executable (default: build/spock-decode)",
    )
    parser.add_argument(
        "--repack-dir",
        required=True,
        help="Path to repacked text weights",
    )
    parser.add_argument(
        "--reference",
        default=str(DEFAULT_REFERENCE),
        help="Path to reference_tokens.jsonl",
    )
    parser.add_argument(
        "--ids",
        required=True,
        help="Comma-separated prompt IDs to sweep from the reference file",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Generated tokens per prompt (default: 16)",
    )
    parser.add_argument(
        "--chunk-sizes",
        default="1,4,8,16",
        help="Comma-separated chunk sizes to sweep (default: 1,4,8,16)",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="Unmeasured warmup runs per id/chunk_size (must match; default: 0)",
    )
    parser.add_argument(
        "--timed-runs",
        type=int,
        default=1,
        help="Measured runs per id/chunk_size (must match; default: 1)",
    )
    parser.add_argument(
        "--gpu-timestamps",
        action="store_true",
        default=False,
        help="Enable SPOCK_GPU_TIMESTAMPS=1 and record GPU timing fields",
    )
    args = parser.parse_args()

    # --- Validate arguments ---
    if args.warmup_runs < 0:
        print("FAIL: --warmup-runs must be >= 0", file=sys.stderr)
        return 1
    if args.timed_runs < 1:
        print("FAIL: --timed-runs must be >= 1", file=sys.stderr)
        return 1
    if args.max_new_tokens < 1:
        print("FAIL: --max-new-tokens must be >= 1", file=sys.stderr)
        return 1

    ids = [item.strip() for item in args.ids.split(",") if item.strip()]
    if not ids:
        print("FAIL: --ids did not contain any prompt IDs", file=sys.stderr)
        return 1
    try:
        chunk_sizes = [int(s.strip()) for s in args.chunk_sizes.split(",") if s.strip()]
    except ValueError as exc:
        print(f"FAIL: invalid --chunk-sizes value: {exc}", file=sys.stderr)
        return 1
    if not chunk_sizes or any(size <= 0 for size in chunk_sizes):
        print("FAIL: --chunk-sizes must contain positive integers", file=sys.stderr)
        return 1
    decode_exe = Path(args.decode)

    if not decode_exe.is_file():
        print(f"FAIL: decode executable not found: {decode_exe}", file=sys.stderr)
        return 1

    entries = load_entries(args.reference, ids)
    if not entries:
        print("FAIL: no reference entries loaded", file=sys.stderr)
        return 1

    all_env_gate_names = sorted(
        list(FAST_ENV_GATES.keys())
        + list(CHUNKED_ENV_GATES.keys())
        + ["SPOCK_GPU_DECODE_CHUNK_SIZE"]
        + (["SPOCK_GPU_TIMESTAMPS"] if args.gpu_timestamps else [])
    )

    results = []
    any_failure = False

    with tempfile.TemporaryDirectory(prefix="spock-chunked-sweep-") as tmpdir:
        token_path = Path(tmpdir) / "prompt.tokens"

        for chunk_size in chunk_sizes:
            env = build_env(chunk_size, gpu_timestamps=args.gpu_timestamps)

            for entry in entries:
                entry_id = entry.get("id", "<unknown>")
                expected = entry["generated_ids"][: args.max_new_tokens]

                token_path.write_text(
                    " ".join(str(t) for t in entry["prompt_ids"]) + "\n",
                    encoding="utf-8",
                )

                # --- Warmup runs (unmeasured, must match) ---
                warmup_ok = True
                for wi in range(args.warmup_runs):
                    rc, stdout, stderr = _run_decode(
                        decode_exe, args.repack_dir, token_path,
                        args.max_new_tokens, env,
                    )
                    if rc != 0:
                        results.append(_make_error_record(
                            entry_id, chunk_size,
                            {
                                "error": "warmup decode failed",
                                "warmup_run": wi,
                                "returncode": rc,
                                "stderr": stderr.strip()[:512],
                            },
                        ))
                        any_failure = True
                        warmup_ok = False
                        break

                    try:
                        decoded = _parse_and_validate(stdout)
                    except (ValueError, json.JSONDecodeError) as exc:
                        results.append(_make_error_record(
                            entry_id, chunk_size,
                            {
                                "error": f"warmup JSON parse failed: {exc}",
                                "warmup_run": wi,
                                "stdout": stdout.strip()[:512],
                            },
                        ))
                        any_failure = True
                        warmup_ok = False
                        break

                    if "error" in decoded:
                        results.append(_make_error_record(
                            entry_id, chunk_size,
                            {
                                "error": decoded["error"],
                                "warmup_run": wi,
                            },
                        ))
                        any_failure = True
                        warmup_ok = False
                        break

                    actual = decoded["generated_tokens"]
                    if actual != expected:
                        results.append(_make_error_record(
                            entry_id, chunk_size,
                            {
                                "error": "warmup mismatch",
                                "warmup_run": wi,
                            },
                        ))
                        any_failure = True
                        warmup_ok = False
                        break

                if not warmup_ok:
                    continue

                # --- Timed runs (measured, must match) ---
                timed_records = []
                timed_ok = True

                for ti in range(args.timed_runs):
                    rc, stdout, stderr = _run_decode(
                        decode_exe, args.repack_dir, token_path,
                        args.max_new_tokens, env,
                    )
                    if rc != 0:
                        results.append(_make_error_record(
                            entry_id, chunk_size,
                            {
                                "run_index": ti,
                                "error": "decode failed",
                                "returncode": rc,
                                "stderr": stderr.strip()[:512],
                            },
                        ))
                        any_failure = True
                        timed_ok = False
                        break

                    try:
                        decoded = _parse_and_validate(stdout)
                    except (ValueError, json.JSONDecodeError) as exc:
                        results.append(_make_error_record(
                            entry_id, chunk_size,
                            {
                                "run_index": ti,
                                "error": f"JSON parse failed: {exc}",
                                "stdout": stdout.strip()[:512],
                            },
                        ))
                        any_failure = True
                        timed_ok = False
                        break

                    if "error" in decoded:
                        results.append(_make_error_record(
                            entry_id, chunk_size,
                            {
                                "run_index": ti,
                                "error": decoded["error"],
                            },
                        ))
                        any_failure = True
                        timed_ok = False
                        break

                    actual = decoded["generated_tokens"]
                    match = actual == expected
                    if not match:
                        any_failure = True

                    rec = _make_run_record(
                        entry_id, chunk_size, ti, decoded, match,
                        gpu_timestamps=args.gpu_timestamps,
                    )
                    timed_records.append(rec)
                    results.append(rec)

                    # --- GPU timestamp validation ---
                    if args.gpu_timestamps:
                        ts_error = _gpu_timestamp_validation_error(
                            decoded, args.max_new_tokens,
                        )
                        if ts_error:
                            rec["match"] = False
                            rec["gpu_timestamp_error"] = ts_error
                            results.append(_make_error_record(
                                entry_id, chunk_size,
                                {
                                    "run_index": ti,
                                    "error": ts_error,
                                },
                            ))
                            any_failure = True
                            timed_ok = False
                            break

                if not timed_ok:
                    continue

                # --- Aggregate record ---
                all_match = all(r["match"] for r in timed_records)
                agg = _make_aggregate_record(
                    entry_id, chunk_size, all_match,
                    args.timed_runs, timed_records,
                    gpu_timestamps=args.gpu_timestamps,
                )
                results.append(agg)

    rev = git_short_rev()
    summary = {
        "git_rev": rev,
        "env_gates": all_env_gate_names,
        "decode": str(decode_exe),
        "repack_dir": args.repack_dir,
        "reference": args.reference,
        "max_new_tokens": args.max_new_tokens,
        "chunk_sizes": chunk_sizes,
        "ids": ids,
        "warmup_runs": args.warmup_runs,
        "timed_runs": args.timed_runs,
        "gpu_timestamps": args.gpu_timestamps,
        "results": results,
    }

    print(json.dumps(summary, indent=2))
    return 1 if any_failure else 0


if __name__ == "__main__":
    sys.exit(main())

# 0064: Chunked Decode Sweep Tool

## Goal

Provide a reusable offlined tool that sweeps chunked decode sizes for one or more
reference prompt IDs and reports correctness, submit counts, and host-side timing
in structured JSON. This is a measurement and correctness-sweep convenience, not a
runtime or shader change.

The sweep tool automates what was previously a manual per-chunk-size invocation of
`spock-decode`: set the full fast-path environment gate stack plus
`SPOCK_GPU_CHUNKED_DECODE=1` and `SPOCK_GPU_DECODE_CHUNK_SIZE=N` for each chunk
size in the sweep, run the decode, compare generated tokens against the stored
reference, and collect structured results.

## Background

Diaries 0058 through 0063 built the chunked decode gate incrementally: scaffold,
size-1 equivalence, active multi-token path, submit-count instrumentation, and a
size-8 multiprompt CTest. Each step verified correctness at one or two chunk sizes
through dedicated CTest registrations.

What remained manual was sweeping multiple chunk sizes across multiple prompts and
collecting the results in a comparable form. The existing CTest infrastructure
tests fixed configurations; it does not parameterize over chunk sizes or aggregate
results across them. A sweep tool fills this gap without modifying the runtime.

The submit-count geometry established in diaries 0062 and 0063 gives a clear
structural prediction for how host submissions should decrease as chunk size grows:
for `max_new_tokens=16`, a chunk size of 1 should produce 16 decode submits
(one initial skip-layers submit plus 15 one-step chunked submits), while chunk
size 16 should produce just 2 decode submits (one skip-layers submit plus one
15-step chunked submit). The sweep tool
collects these counts so the predicted amortization curve is observable from real
runs without manual arithmetic.

## Implementation Work Completed

### New tool: `tools/run_chunked_decode_sweep.py`

The tool accepts:

- `--decode`: path to the `spock-decode` executable (defaults to
  `build/spock-decode`).
- `--repack-dir`: path to the repacked text-weight directory (required).
- `--reference`: path to `reference_tokens.jsonl` (defaults to
  `tests/data/reference_tokens.jsonl`).
- `--ids`: comma-separated prompt IDs to sweep from the reference file.
- `--max-new-tokens`: generated tokens per prompt (defaults to 16).
- `--chunk-sizes`: comma-separated chunk sizes to sweep (defaults to
  `1,4,8,16`).

For each combination of chunk size and prompt ID, the tool:

1. Writes the prompt tokens to a temporary file.
2. Invokes `spock-decode` with the full fast-path environment gate stack
   (`SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1`, `SPOCK_GPU_MERGED_DELTANET=1`,
   `SPOCK_GPU_FUSED_DN_CONV_L2=1`, `SPOCK_GPU_FUSED_DN_GBETA_RECURRENT=1`,
   `SPOCK_GPU_FUSED_DN_REC_NORM_GATE=1`, `SPOCK_GPU_SINGLE_SUBMIT=1`,
   `SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1`, `SPOCK_GPU_DEFER_TOKEN_DOWNLOAD=1`,
   `SPOCK_GPU_MATVEC_TILED=1`, `SPOCK_GPU_LM_HEAD_TILED=1`) plus
   `SPOCK_GPU_CHUNKED_DECODE=1` and `SPOCK_GPU_DECODE_CHUNK_SIZE=N`.
3. Parses the JSON object from `spock-decode` stdout.
4. Compares `generated_tokens` against the stored reference truncated to
   `max_new_tokens`.
5. Records `match`, `decode_submit_count`, `chunked_decode_submit_count`,
   `elapsed_ms`, `prefill_ms`, `decode_ms`, and `generated_count`.

After all combinations complete, the tool emits a single JSON summary containing
`git_rev`, `env_gates`, `decode`, `repack_dir`, `reference`, `max_new_tokens`,
`chunk_sizes`, `ids`, and a `results` array with per-run records.

The tool exits nonzero on any decode failure or token mismatch. This makes it
suitable for scripted correctness sweeps and future CI gating: a zero exit code
means all chunk sizes produced reference-matching output for all requested
prompts.

The environment gate list mirrors the full fast-path stack from diary 0057 and
the chunked decode gates from diary 0058. The tool hard-codes these so the
caller does not need to manage the full gate set manually. The only parameterized
gate is `SPOCK_GPU_DECODE_CHUNK_SIZE`, which varies per sweep iteration.

### Design decisions

The tool does not write the JSON summary to `artifacts/` or any other persistent
location. The `artifacts/` directory is gitignored, and the diary records the
evidence rather than committing generated JSON. The caller can redirect stdout to
a file if they want to persist the output.

The tool uses a temporary directory for per-run token files rather than writing to
a fixed location, avoiding cross-run file races if multiple sweeps run
concurrently.

The tool does not set `SPOCK_GPU_TIMESTAMPS=1` or any diagnostic/dump flags. It
targets the production fast-path configuration. Timestamp-based GPU measurement of
chunked decode remains separate future work.

## Verification

### Syntax check

```
python3 -m py_compile tools/run_chunked_decode_sweep.py
```

Passed. No import or syntax errors.

### Local sweep run

```
python3 tools/run_chunked_decode_sweep.py \
  --decode build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --ids short_correctness_001,mixed_correctness_023 \
  --max-new-tokens 16 \
  --chunk-sizes 1,4,8,16
```

Passed. Both prompts matched their references at all four chunk sizes. The tool
exited zero.

### Submit-count results

The observed submit counts confirm the expected amortization geometry:

| Chunk size | decode_submit_count | chunked_decode_submit_count |
| ---------- | ------------------- | --------------------------- |
| 1          | 16                  | 15                          |
| 4          | 5                   | 4                           |
| 8          | 3                   | 2                           |
| 16         | 2                   | 1                           |

These counts are consistent with the structural model: one initial skip-layers
submit, then chunked submits covering the remaining eligible steps. At chunk size
1, each eligible step is its own chunk (15 chunked submits plus the skip-layers
step for 16 total decode submits). At chunk size 16, all 15 eligible steps fit in
a single chunk (1 chunked submit plus the skip-layers step for 2 total decode
submits). The submit counts are identical for both prompts at each chunk size,
confirming the geometry depends only on `max_new_tokens` and `chunk_size`, not on
the specific token sequence.

### Timing data

Host-side `elapsed_ms`, `prefill_ms`, and `decode_ms` were captured in the JSON
output. These values are single-run measurements on a specific machine under
arbitrary host load. They are recorded as evidence that the timing fields
populate correctly and the tool captures them. They are not benchmark results and
should not be cited as performance claims. Wall-clock comparison across chunk
sizes requires controlled repeated runs with warmup, which this tool does not
provide.

### Existing CTest suite

No CTest registrations were added or modified. The existing chunked decode CTest
suite (full fast gate, size-1 equivalence, size-4 partial, size-8 multiprompt)
was not re-run because this diary entry does not modify runtime code, shader
code, or test infrastructure.

## Interpretation

This entry adds a convenience tool for parameterized chunked-decode sweeps. It
does not modify the runtime, shaders, or CTest infrastructure. The tool's value
is operational: it reduces the manual effort required to sweep chunk sizes and
collect structured results, which supports future tuning decisions about chunk
size selection.

The submit-count data confirms that the runtime's chunked decode submission
amortization behaves as predicted across all tested sizes. The structural
prediction from diaries 0062 and 0063 — that host submissions decrease
proportionally to chunk size — holds at sizes 1, 4, 8, and 16 for both tested
prompts. This is a structural observation, not a wall-clock performance claim.

The tool is ready for use in future chunk-size tuning sweeps, broader
correctness sweeps with more prompt IDs, and scripted CI gating if the project
decides to sweep chunk sizes as part of regression testing.

## Known Limitations

- **Single-run timing only.** The tool runs each configuration once. Host-side
  timing is included in the JSON for completeness, but it is not a benchmark.
  Controlled repeated measurement with warmup remains future work.

- **Not a replacement for CTest.** The tool is a manual sweep convenience.
  CTest registrations for specific chunk sizes remain the regression gate.

- **Hard-coded fast-path gate stack.** The environment gates are hard-coded in
  the tool. If the fast-path gate stack changes, the tool must be updated to
  match. This is intentional: the tool captures a specific configuration, not a
  generic parameterization over all possible gate combinations.

- **No GPU timestamps.** The tool does not enable `SPOCK_GPU_TIMESTAMPS=1`. GPU-
  side execution time for chunked decode command buffers is not captured.

- **Two prompts, four chunk sizes.** The local sweep covered two prompts at four
  chunk sizes. Broader sweeps across more prompts and chunk sizes are possible
  but were not run for this entry.

- **Still not persistent dispatch.** The sweep runs the chunked decode path,
  which is bounded command-buffer batching on the host. No persistent workgroups,
  no software global barrier, no in-kernel layer loop.

- **Still not the megakernel.** This is a measurement convenience. It does not
  advance the persistent-dispatch or megakernel milestones.

## Next Work

1. Use the sweep tool to run controlled repeated timing across chunk sizes with
   warmup to characterize wall-clock submission-overhead amortization on this
   hardware.
2. Extend GPU timestamp bookkeeping to multi-step command buffers so per-chunk
   execution time is observable, then re-run the sweep with timestamps enabled.
3. Evaluate whether the RADV bounded-dispatch limits from diary 0053 constrain
   viable chunk sizes for the real decode workload at higher token counts.
4. Consider adding the sweep tool to CI as an optional broad-correctness gate
   once the project has a preferred default chunk size.

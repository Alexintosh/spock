# Spock: Vulkan Megakernel Plan

See also `docs/megakernel_development_philosophy.md` and
`docs/megakernel_phase_rationale.md` for the current execution philosophy: why
the project builds artifact, runtime, barrier, skeleton, component gates,
layer-shaped probes, and inference gates before claiming the RX 6750 XT
Vulkan-native persistent megakernel target.
The current concrete execution map is
`docs/rx6750xt_megakernel_execution_map.md`.

Current checkpoint: diaries 0101-0122 have closed the layer-0 DeltaNet
component chain through output projection, norm-gate, z projection, raw qkv
projection, A/B projections, g/beta scalar computation, conv1d mutation, q/k L2
normalization, recurrent core, and a full single-submit composed DeltaNet
mixer. Diary 0114 adds the first 128-lane persistent layer-0 tail scaffold for
`mixer_residual -> post_norm -> MLP -> post_mlp`; diaries 0116-0120 move the
DeltaNet projection-prefix, conv/L2, g/beta, recurrent-core, and mixer-tail
boundaries into `persistent_layer0_probe.comp`; diary 0121 composes all five
persistent sub-blocks into a single 6-barrier full-mixer dispatch (mode=6);
diary 0122 composes that persistent mixer with the post-mixer RMSNorm+MLP tail
into a single captured layer-0 pass with 10 barriers (mode=7 / `layer0`).
The target is still the RX
6750 XT Vulkan-native persistent megakernel, not a generic Vulkan backend. The
immediate path is to localize the mode=7 post-MLP precision bound and widen
from one explainable layer to
representative layers, bounded multi-layer decode, all 24 layers, LM head,
token selection, and archived basic inference.
## Mission

Build a Vulkan-native, model-specific inference engine for `Qwen/Qwen3.5-0.8B` on this machine's `AMD Radeon RX 6750 XT (RADV NAVI22)` that reaches **Luce-style parity** in the dimensions that matter:

1. **Functional parity**
   Exact next-token behavior against a reference decode path for fixed prompts and deterministic sampling.
2. **Architectural parity**
   A single-model, batch-1 engine specialized for Qwen 3.5-0.8B's hybrid DeltaNet + Attention layout.
3. **Operational parity**
   No CPU round-trips between layers in the hot path, or the closest provable Vulkan equivalent if full persistent cross-workgroup synchronization is not robust.
4. **Performance parity**
   Match Lucebox's **relative win over a generic engine on the same hardware class**, not its absolute 3090 token rates.

This plan is written to make parity visible at every stage. Every milestone has a **Parity Gate** and a **Go/No-Go** criterion.

---

## What "Parity" Means Here

We are **not** targeting absolute token-rate parity with Lucebox's RTX 3090 numbers. That is the wrong target for this machine.

We **are** targeting:

- **Reference parity:** exact or controlled numerical agreement with a trusted reference implementation
- **Behavior parity:** same model architecture, same layer schedule, same state transitions, same decode semantics
- **Relative performance parity:** a Luce-style speedup over the best generic Vulkan baseline on this RX 6750 XT

### Parity tiers

| Tier | Meaning | Must-Have |
| --- | --- | --- |
| `P0` | Correctness parity | Exact greedy-token parity on fixed prompts |
| `P1` | Baseline parity | At least match local generic Vulkan baseline |
| `P2` | Relative parity | Meaningful win vs local generic baseline |
| `P3` | Luce-style parity | Decode speedup factor close to Lucebox's relative improvement |

### Concrete success criteria

- `P0`: exact token parity on a fixed test corpus
- `P1`: `tg128` decode throughput `>= 1.0x` local generic Vulkan baseline
- `P2`: `tg128` decode throughput `>= 1.25x` local generic Vulkan baseline
- `P3`: `tg128` decode throughput `>= 1.45x` local generic Vulkan baseline

For prefill:

- `P1`: `pp520 >= 1.0x` local generic Vulkan baseline
- `P2`: `pp520 >= 1.75x` local generic Vulkan baseline
- `P3`: `pp520 >= 2.5x` local generic Vulkan baseline

Those prefill targets are lower than Lucebox's 3090 ratio on purpose. This GPU lacks NVIDIA-specific advantages and does not expose native BF16.

---

## Hard Facts We Must Design Around

### Reference implementation facts from Lucebox

- Model: `Qwen 3.5-0.8B`
- Layers: `24`
- Layer pattern: `18 DeltaNet + 6 Full Attention`, repeating `0,0,0,1`
- Hidden size: `1024`
- Intermediate size: `3584`
- Full Attention:
  - `8` Q heads
  - `2` KV heads
  - head dim `256`
- DeltaNet:
  - `16` heads
  - key dim `128`
  - value dim `128`
  - conv kernel `4`
- Luce decode path:
  - persistent CUDA kernel
  - `82` blocks
  - `512` threads/block
  - BF16 activations/weights
  - FP32 DeltaNet state
- Luce benchmark contract:
  - correctness check in `bench_pp_tg.py`
  - performance numbers from warmed `pp520` / `tg128`

### Local target facts from this machine

- GPU: `AMD Radeon RX 6750 XT (RADV NAVI22)`
- Vulkan API: `1.4.318`
- Driver: `Mesa RADV 25.2.8`
- VRAM exposed: about `12 GiB`
- Subgroup width observed by local Vulkan-backed inference tools: `32`
- Shared memory / LDS visible to local tools: `64 KiB`
- Native `fp16`: yes
- Native `bf16`: no
- Matrix cores / cooperative matrix acceleration: none reported

### Immediate implications

1. **We cannot copy Luce's BF16 path directly.**
   The production Vulkan path must target `fp16 + fp32` mixed precision.
2. **We cannot assume CUDA-like cooperative grid guarantees.**
   A software global barrier across workgroups is the biggest technical risk in the whole project.
3. **Decode comes first.**
   Lucebox's core claim is decode efficiency; prefill is phase two.
4. **Batch size stays at 1.**
   This project is for local agentic inference, not multi-tenant serving.

---

## Scope

### In scope

- Single model: `Qwen 3.5-0.8B`
- Single GPU: `RX 6750 XT`
- Vulkan compute only
- Batch size `1`
- Decode-first bring-up
- Prefill after decode parity
- Exact greedy correctness first, richer sampling second

### Out of scope for v1

- Multi-model support
- Quantization beyond what is needed for later experiments
- Batching / multi-user serving
- Cross-vendor portability
- Training or fine-tuning

---

## Baseline Strategy

We need two baselines:

1. **Correctness baseline**
   A trusted reference implementation of Qwen 3.5-0.8B decode with deterministic settings.
2. **Performance baseline**
   The best generic Vulkan engine we can run on this machine with the same model and comparable precision.

The project does not advance to "megakernel parity" until both baselines are frozen and reproducible.

---

## Implementation Roadmap

## Milestone 0: Freeze The Contract

### Goal

Define the exact parity target before writing kernels.

### Work

- Freeze benchmark prompts and prompt lengths:
  - `pp520`
  - `tg128`
  - short correctness prompt
  - corpus of `32-64` deterministic prompts
- Freeze decode settings:
  - greedy or deterministic sampler
  - fixed BOS/EOS handling
  - fixed max sequence length for v1, starting at `2048`
- Freeze measurement protocol:
  - warmup count
  - timed run count
  - GPU sync points
  - timestamp query usage
- Define the parity scoreboard:
  - `P0`, `P1`, `P2`, `P3`

### Deliverables

- `docs/parity_contract.md`
- prompt corpus
- benchmark CLI spec

### Parity Gate

No implementation begins until we can answer:
"What exactly counts as parity, and how will we prove it?"

### Go / No-Go

- Go if correctness and performance baselines are measurable
- No-Go if parity remains vague or benchmark methodology is not frozen

---

## Milestone 1: Project Skeleton And Instrumentation

### Goal

Create the repo skeleton and the measurement infrastructure first.

### Work

- Create project layout:
  - `src/runtime`
  - `src/model`
  - `src/kernels`
  - `src/reference`
  - `shaders`
  - `tools`
  - `bench`
  - `tests`
- Add:
  - CMake build
  - shader compilation step
  - Vulkan validation toggle
  - timestamp query wrapper
  - GPU event markers
- Build a benchmark CLI:
  - `spock-bench --mode pp520`
  - `spock-bench --mode tg128`
  - CSV / JSON output

### Deliverables

- builds cleanly
- runs a trivial Vulkan compute kernel
- emits timing output

### Parity Gate

Instrumentation parity: the project must be able to measure the same classes of numbers Lucebox measures before any optimization claims are made.

### Go / No-Go

- Go if timestamp queries and synchronized timing are reliable
- No-Go if we cannot trust our measurements

---

## Milestone 2: Freeze Model Artifacts And Packing Format

### Goal

Replace PyTorch pointer-packing with an offline Vulkan-friendly model artifact format.

### Work

- Pin exact Hugging Face model revision
- Extract config, tensor names, and layer schedule
- Write a converter that emits:
  - embeddings
  - final norm
  - LM head
  - per-layer headers
  - fused attention QKV blobs
  - fused MLP gate/up blobs
  - DeltaNet projection blobs
  - metadata manifest
- Choose memory layout:
  - aligned packed buffers
  - explicit offsets instead of runtime pointers
  - fp16 storage for production path
  - optional fp32 shadow export for reference

### Deliverables

- `tools/convert_qwen35_0p8b.py`
- packed model artifact format
- loader spec

### Parity Gate

Artifact parity: the offline packed format must reconstruct the same effective weights as the upstream model.

### Go / No-Go

- Go if sample tensors match expected values after conversion
- No-Go if packing introduces silent transposition, alignment, or precision bugs

---

## Milestone 3: CPU Reference Decode

### Goal

Build a slow but trustworthy reference path we control.

### Work

- Implement step-wise CPU reference for:
  - RMSNorm
  - DeltaNet block
  - Full Attention block
  - MLP
  - LM head
- Mirror Qwen 3.5-0.8B layer schedule exactly
- Add deterministic prompt tests
- Compare against a trusted upstream framework on greedy decode

### Deliverables

- `src/reference/qwen35_cpu_reference.*`
- exact-token test corpus

### Parity Gate

`P0` starts here: CPU reference must match the trusted upstream decode behavior on the fixed corpus.

### Go / No-Go

- Go if reference outputs are stable and exact
- No-Go if we are debugging Vulkan against an untrusted reference

---

## Milestone 4: Vulkan Runtime Bring-Up

### Goal

Bring up the Vulkan runtime and memory model independent of megakernel work.

### Work

- Device selection and capability detection
- Memory allocator:
  - device-local buffers
  - host-visible staging buffers
  - persistent mapped upload ring
- Descriptor set layout for:
  - weights
  - activations
  - DeltaNet state
  - KV cache
  - scratch buffers
- Pipeline cache and specialization constants
- Runtime queries:
  - subgroup size
  - max shared memory
  - max workgroup size
  - preferred residency / occupancy heuristics

### Deliverables

- `src/runtime/vk_context.*`
- `src/runtime/vk_allocator.*`
- capability dump CLI

### Parity Gate

Operational parity begins here: the runtime must expose enough information to tune a persistent GPU path specifically for this RX 6750 XT, not generically.

### Go / No-Go

- Go if device limits and subgroup behavior are clearly discoverable
- No-Go if runtime uncertainty prevents informed kernel design

---

## Milestone 5: Layer-By-Layer Vulkan Decode Baseline

### Goal

Get a correct Vulkan decode path before attempting megakernel fusion.

### Work

- Implement separate Vulkan kernels for:
  - embedding lookup
  - RMSNorm
  - DeltaNet projections and recurrence
  - Full Attention
  - MLP
  - LM head / argmax
- Keep the control flow simple:
  - one or more dispatches per op
  - explicit barriers between dispatches
  - no persistent workgroups yet
- Use production precision plan:
  - fp16 activations and weights
  - fp32 accumulations where needed
  - fp32 DeltaNet state

### Deliverables

- `spock-decode-ref-vk`
- intermediate tensor dump tooling

### Parity Gate

`P0` Vulkan parity: exact token parity with the CPU reference on the fixed corpus.

### Go / No-Go

- Go if outputs are exact or discrepancies are explained by a known precision policy
- No-Go if the first Vulkan path is not fully correct

---

## Milestone 6: Local Performance Baseline Freeze

### Goal

Quantify what "generic" performance means on this exact machine.

### Work

- Benchmark the layer-by-layer Vulkan path
- Benchmark local generic inference engine(s) on the same model
- Freeze:
  - `pp520`
  - `tg128`
  - prompt lengths
  - warmup counts
  - repetition settings
- Store results as the baseline scoreboard

### Deliverables

- `bench/baseline_rx6750xt.json`
- reproducible benchmark script

### Parity Gate

Performance parity cannot be claimed until the local generic baseline is frozen and reproducible.

### Go / No-Go

- Go if we have a stable baseline to beat
- No-Go if the baseline keeps moving

---

## Milestone 7: Weight Layout Optimization

### Goal

Eliminate avoidable memory inefficiencies before fusing control flow.

### Work

- Fuse static weight layouts offline:
  - Full Attention QKV
  - MLP gate/up
- Reorder tensors for contiguous subgroup-friendly fetches
- Align blocks for vectorized fp16 loads
- Reduce descriptor churn by grouping per-layer metadata
- Decide hot-path scratch ownership:
  - registers
  - LDS/shared memory
  - global scratch

### Deliverables

- optimized artifact layout v2
- loader updated for new layout

### Parity Gate

Layout parity: optimized packing must not change model behavior, only access cost.

### Go / No-Go

- Go if token outputs remain unchanged
- No-Go if layout optimization changes semantics

---

## Milestone 8: Fused DeltaNet Layer

### Goal

Make the DeltaNet block fast in isolation before integrating it into a megakernel.

### Work

- Fuse per-token DeltaNet block stages:
  - input RMSNorm
  - qkv/z/beta/alpha projections
  - conv update
  - recurrent state update
  - output projection
  - post-attn residual
  - MLP gate/up/down
  - final residual
- Keep recurrence state in fp32
- Use subgroup reductions for head-local work
- Evaluate register pressure vs LDS spill

### Deliverables

- fused DeltaNet block kernel
- microbench for single layer latency

### Parity Gate

Block parity: the fused DeltaNet block must produce the same outputs as the unfused Vulkan baseline for all tested tokens.

### Go / No-Go

- Go if isolated DeltaNet fusion is exact and faster
- No-Go if fusion breaks correctness or creates unmanageable spills

---

## Milestone 9: Fused Full-Attention Layer

### Goal

Make the attention block fast in isolation before integrating it into a megakernel.

### Work

- Fuse:
  - RMSNorm
  - Q/K/V projection
  - q_norm / k_norm
  - RoPE
  - KV cache write
  - online causal softmax
  - O projection
  - residual
  - MLP gate/up/down
  - residual
- Optimize for single-token decode
- Decide whether LM head remains separate in v1

### Deliverables

- fused attention block kernel
- single-layer attention benchmark

### Parity Gate

Attention parity: exact token agreement and intermediate-state agreement vs the unfused Vulkan path.

### Go / No-Go

- Go if fused attention is correct and faster
- No-Go if attention fusion destabilizes cache behavior or precision

---

## Milestone 10: Single-Submit Decode Pipeline

### Goal

Get rid of CPU round-trips between layers even before full persistent dispatch.

### Work

- Record the full 24-layer decode pipeline into one command buffer
- Submit once per token
- Keep buffers resident across steps
- Reuse descriptor sets and pipelines
- Avoid host waits between layers entirely

This is the first pragmatic Vulkan equivalent to Lucebox's "no CPU round-trips" claim.

### Deliverables

- `decode_single_submit()` path
- command-buffer replay benchmark

### Parity Gate

Operational parity checkpoint: one GPU submission per token, no host mediation between layers.

### Go / No-Go

- Go if this already delivers a meaningful win
- No-Go only if command-buffer overhead remains too high to matter

---

## Milestone 11: Persistent Workgroup + Software Global Barrier Spike

The rationale for building probes and handoff gates before attempting the full
megakernel is documented in
[`docs/megakernel_development_philosophy.md`](docs/megakernel_development_philosophy.md).

### Goal

Prove whether a true Vulkan megakernel is viable on RADV for this GPU.

### Work

- Implement a toy persistent kernel with:
  - fixed resident workgroup count
  - storage-buffer atomics
  - software global barrier
  - spin-based generation counter
- Sweep workgroup counts up to the safe resident ceiling
- Validate:
  - forward progress
  - no deadlock
  - no device loss
  - acceptable barrier overhead
- Repeat with a 2-layer mini-pipeline

### Current Status

- Bare barrier probe implemented as `vk_barrier_probe` and verified locally at
  8, 16, 32, 64, 82, and 128 workgroups x 10000 iterations (diary 0047).
- Two-stage coherent scratch mini-pipeline implemented and verified across the
  same sweep (diary 0048). The non-coherent scratch version failed data
  validation at 82 workgroups despite correct generation/failure counters;
  `coherent` scratch storage fixed the visibility issue.
- First opt-in timestamp hook added to the probe (diary 0049). A local
  82-workgroup x 10000-iteration sample measured about 113576 us GPU dispatch
  time, or about 5.67878 us per software barrier. This is a first sample, not a
  final benchmark.
- Local 82-workgroup x 1000000-iteration soak passed both without timestamps and
  with timestamps (diary 0050), reaching 2000000 barrier generations with zero
  trace mismatches. Timestamped run measured about 1.03471e+07 us GPU dispatch
  time, or about 5.17354 us per software barrier.
- Optional per-lane ALU payload mode added to the probe (diary 0051). The
  default no-payload path remains unchanged, and a local
  82-workgroup x 10000-iteration run with `--payload-iters 64 --timestamps`
  passed with zero trace mismatches and about 7.48249 us per barrier.
- Optional lane-strided memory payload added to the probe (diary 0052). A local
  82-workgroup x 10000-iteration run with `--payload-cols 256 --timestamps`
  passed with zero trace mismatches and about 6.95452 us per barrier; combined
  `--payload-iters 64 --payload-cols 256` also passed.
- Longer `--payload-cols 256` soaks show a driver/runtime boundary (diary
  0053): 750k iterations passed at about 9.62s GPU dispatch time, but 900k and
  1M failed with all-zero GPU output; the 1M non-timestamped rerun printed a
  RADV context-loss/hard-recovery message.
- Five repeated bounded 100k memory-payload runs passed with stable timing
  around 6.45 us per barrier (diary 0054), supporting bounded dispatch chunks as
  a safer direction than one unbounded memory-heavy dispatch on this stack.
- `vk_barrier_probe --repeats N` now supports in-process bounded repeat testing
  without recreating the Vulkan device/pipeline each run (diary 0055). A local
  3-repeat run at 82 workgroups x 100000 iterations with `--payload-cols 256`
  passed with zero aggregate trace mismatches and stable timing.
- A 10-repeat in-process run at the same geometry also passed (diary 0056),
  totaling 2000000 software barriers across bounded dispatches with zero
  aggregate trace mismatches and per-barrier timing around 6.45-6.46 us.
- A CTest gate now protects the current full fast Vulkan decode env stack
  (diary 0057), giving bounded chunked decode work a regression baseline.
- `SPOCK_GPU_CHUNKED_DECODE` and `SPOCK_GPU_DECODE_CHUNK_SIZE` are now parsed as
  force-disabled scaffold gates in `DecodeSession::decode()` (diary 0058), with
  CTest verification that setting them does not change current fast-path output.
- A dedicated CTest now protects that inert scaffold contract with the full fast
  gate stack plus `SPOCK_GPU_CHUNKED_DECODE=1` and chunk size 4 (diary 0059).
- The chunked gate now has a live size-1 equivalence mode (diary 0060):
  `chunked_decode_enabled` is true only for chunk size 1 under the full fast
  prerequisites, and CTest verifies parity against the current fast path.
- Bounded chunked decode is now active for chunk sizes greater than 1 (diary
  0061): one command buffer records up to `SPOCK_GPU_DECODE_CHUNK_SIZE`
  eligible decode steps, submits on full chunk or final partial chunk, with an
  explicit argmax-result barrier for correct intra-chunk token propagation.
  Verified manually at chunk size 4 with max_new_tokens 4 and 5. A size-4
  full-plus-partial CTest with max_new_tokens 6 validates the boundary.
- Submit-count instrumentation now exposes `decode_submit_count` and
  `chunked_decode_submit_count` in `DecodeResult` and `spock-decode` JSON
  output (diary 0062). The initial size-4 partial CTest asserted
  decode_submit_count=3 and chunked_decode_submit_count=2; diary 0067 updates
  the current expectation to 2/2 after absorbing the skip-layers step into the
  chunk. Wall-clock benchmarking of the chunked path remains next work.
- A size-8 multiprompt CTest (diary 0063) extends chunked decode coverage to
  chunk size 8 across two prompts (`short_correctness_001`,
  `mixed_correctness_023`) with full 16-token reference output. The test
  initially asserted decode_submit_count=3 and chunked_decode_submit_count=2;
  diary 0067 updates the current expectation to 2/2 after skip-step absorption.
  Full fast, size-1 equivalence, size-4 partial, and size-8 multiprompt CTests
  all pass. This is correctness broadening, not performance proof.
- A chunked decode sweep tool `tools/run_chunked_decode_sweep.py` (diary 0064) now
  automates multi-chunk-size sweeps across one or more reference prompt IDs.
  It invokes `spock-decode` with the full fast env stack plus chunked decode gates
  for each requested chunk size, compares generated tokens against references, and
  emits structured JSON with submit counts and host-side timing. A local sweep at
  chunk sizes 1, 4, 8, 16 across two prompts confirmed reference parity at all
  sizes and submit-count geometry consistent with the structural model:
  size 1 => 16 decode submits, 15 chunked; size 4 => 5/4; size 8 => 3/2;
  size 16 => 2/1. Single-run host timing was captured but is not benchmark proof.
  The tool does not modify runtime, shader, or test code.
- The sweep tool now supports `--warmup-runs` (default 0) and `--timed-runs` (default 1)
  for controlled host-side repeat measurement (diary 0065). Warmup runs execute first per
  id/chunk_size, must match references, and are excluded from aggregate timing. Timed runs
  must each match references; per-run records include `run_index`. Aggregate records per
  id/chunk_size include mean/min/max for elapsed, prefill, and decode ms. Top-level JSON
  includes `warmup_runs` and `timed_runs`. A local sweep with `--warmup-runs 1 --timed-runs 2`
  at chunk sizes 8 and 16 confirmed correct submit counts and all matches. Default
  `--warmup-runs 0 --timed-runs 1` preserves the one-run usage with an aggregate record.
  This is controlled host-side timing structure, not final performance proof.
- A controlled sweep with `--warmup-runs 1 --timed-runs 3` across chunk sizes 1, 2, 4, 8, 16
  on `short_correctness_001` at `--max-new-tokens 16` confirmed reference parity at all sizes
  (diary 0066). Submit counts match the structural model exactly. Host-side decode_ms shows a
  modest monotonic decrease from 353.0 ms (chunk size 1) to 349.9 ms (chunk size 16), a ~0.9%
  reduction consistent with submit-overhead amortization. Per-chunk-size spread across 3 timed
  runs is under 0.7 ms. This is single-prompt, host-side timing evidence only; not GPU timestamps,
  not multi-prompt, not high-token-count, not final benchmark proof.
- The first post-prefill `skip_layers` decode step (final-norm + LM-head + argmax) is now
  absorbed into the chunked command buffer under `SPOCK_GPU_CHUNKED_DECODE=1` (diary 0067).
  Previously this step was always a separate single submit outside the chunked path. Now it
  opens `chunk_cmd`, increments `chunk_recorded_steps`, inserts the same argmax-result barrier,
  and defers submit like any other eligible step. The submit-count formula changes from
  `1 + ceil((N-1)/C)` to `ceil(N/C)`, with `decode_submit_count == chunked_decode_submit_count`.
  Size-4 partial CTest updated: decode=2, chunked=2 (was 3/2). Size-8 multiprompt updated:
  decode=2, chunked=2 (was 3/2). New size-16 single-chunk CTest asserts decode=1, chunked=1
  for `max_new_tokens=16`. Refreshed sweep confirms all sizes produce reference parity.
  Submit counts: size1 16/16, size2 8/8, size4 4/4, size8 2/2, size16 1/1. Host-side decode_ms
  means: size1 353.09, size2 351.98, size4 351.24, size8 350.14, size16 350.49. Size8 was
  best in this short sample; timing is not monotonic after size8. This is important structural
  progress (every step now chunked) but still not persistent dispatch or megakernel. Do not
  overclaim performance from 3-run host-side timing at 16 tokens.
- GPU timestamp recording now extends into chunked decode command buffers when both
  `SPOCK_GPU_TIMESTAMPS=1` and `SPOCK_GPU_CHUNKED_DECODE=1` are active (diary 0068).
  The previous blanket exclusion of timestamps from the chunked path (`!gpu_timestamps`)
  is removed. Per-step start/end timestamp writes occur inside chunked command buffers,
  including the skip-layers first decode step, so `ts_decode_steps` has one entry per
  generated token. Block-level timestamps (`SPOCK_GPU_BLOCK_TIMESTAMPS=1`) remain excluded
  from the chunked path via `!gpu_block_timestamps`. A new CTest
  `spock_vk_decode_chunked_gate_size16_timestamps_short` verifies chunk size 16,
  16 tokens, `SPOCK_GPU_TIMESTAMPS=1`, asserting decode_submit_count=1,
  chunked_decode_submit_count=1, and positive gpu_decode_us. A direct run showed
  gpu_decode_us=347611 with 16 per_token_gpu_us values. Host per_token_ms remains
  chunk-flush-shaped; GPU per_token_gpu_us is the useful device timing. This is basic
  per-token timestamp instrumentation inside chunked command buffers, not block-level
  timestamps, not final benchmark proof, not persistent dispatch, and not the megakernel.
- The chunked decode sweep tool gains `--gpu-timestamps`, an opt-in flag
  that sets `SPOCK_GPU_TIMESTAMPS=1` and records GPU timing fields from
  `spock-decode` JSON in per-run and aggregate records (diary 0069).
  Per-run records add `gpu_decode_us` and a summary of `per_token_gpu_us`
  (count, mean, min, max). Aggregate records add mean/min/max for both
  `gpu_decode_us` and `per_token_gpu_us_mean`. Validation checks
  `gpu_decode_us` presence/positivity and `per_token_gpu_us` length vs
  `generated_count` (falling back to `max_new_tokens` if absent); failures
  annotate the per-run record with `gpu_timestamp_error` and mark
  `match=False`. Default behavior without the flag is unchanged. A lightweight
  CTest (`spock_chunked_sweep_gpu_timestamp_unit`) covers helper behavior, and a
  short real sweep at chunk size 16 passed with match=true, submit counts 1/1,
  gpu_decode_us=347708, and per_token_gpu_us_count=16.
- A controlled GPU-timestamped sweep with `--warmup-runs 1 --timed-runs 3 --gpu-timestamps` across
  chunk sizes 1, 2, 4, 8, 16 on `short_correctness_001` at `--max-new-tokens 16` confirmed reference
  parity at all sizes (diary 0070). GPU decode time is nearly flat across chunk sizes:
  gpu_decode_us_mean ranges from 348296 (size 1) to 347333 (size 16), a ~0.28% reduction.
  Per-token GPU time is nearly constant at about 21.7 ms regardless of chunk size. Host-side
  decode_ms still improves modestly with submit-count reduction (353.1 ms to 349.5 ms, ~1.0%),
  but the GPU data shows this short run is dominated by actual GPU work, not submit overhead.
  This is measurement evidence only: single prompt, 16 tokens, not persistent dispatch,
  not the megakernel, and not a throughput benchmark.
- `vk_barrier_probe` now supports a decode-shaped iteration mode via `--tokens N`
  and `--layers N` (diary 0071). When both are supplied, `iterations = tokens * layers`.
  This is a semantic wrapper around the existing persistent barrier/payload probe,
  not real decode, not model weights, and not the megakernel. It sets the iteration
  count to match the total layer-forward count for a given token x layer geometry.
  JSON output includes `tokens`, `layers`, and `decode_shape_iterations` when active.
  A CTest gate (`spock_barrier_probe_decode_shape`) exercises 16 tokens x 24 layers
  with 8 workgroups and 128-column memory payload. This is a regression gate for
  persistent barrier correctness at decode-relevant iteration scales.
- A decode-shaped timing run at 82 workgroups, 128 tokens x 24 layers (3072 iterations)
  with `--payload-cols 128 --timestamps` passed correctness checks (diary 0072).
  Single-run per-barrier: ~7.92 us. Three-repeat run: repeat 1 ~8.00 us (warmup),
  repeats 2–3 ~6.31 us (stable). Repeats 2–3 are slightly faster than the ~6.45 us
  baseline from diary 0054/0056, consistent with the lower payload-column count (128 vs 256).
  This is the first decode-shaped run at the full Luce reference workgroup count.
  Still a synthetic barrier/payload probe, not real decode or megakernel.
- A model-width decode-shaped run at 82 workgroups, 128 tokens x 24 layers (3072 iterations)
  with `--payload-cols 1024 --timestamps` passed correctness checks (diary 0073).
  Payload columns match Qwen3.5 hidden_size (1024). Three-repeat run: repeat 1 ~8.62 us,
  repeats 2-3 ~6.96 us. Compared to diary 0072 payload-cols=128 at the same geometry
  (~6.31 us/barrier stable), model-width traffic adds ~0.65 us/barrier (~10%). Still synthetic
  uint32 memory traffic, not real fp16/fp32 matvec or decode, but the column count now matches
  hidden size. Supports bounded persistent chunk feasibility.
- `vk_barrier_probe --qwen35-decode-shape-preset` is a convenience flag that sets
  tokens=128, layers=24, workgroups=82, payload_cols=1024 in one invocation (diary 0074).
  User-supplied --tokens/--layers/--workgroups/--payload-cols override the preset values.
  JSON output includes `qwen35_decode_shape_preset: "active"` when the preset is used.
  A CTest gate (`spock_barrier_probe_qwen35_preset`) exercises the full preset workload.
  This is a reproducibility preset for the synthetic model-width probe, not real decode and not the megakernel.
- A persistent decode skeleton probe (`vk_persistent_decode_skeleton`) now exercises the same software-global-barrier pattern with actual fp16 input/weight buffers and fp32 accumulation (diary 0075). This is the first probe combining persistent dispatch with decode-shaped fp16/fp32 compute rather than uint32 synthetic payloads. It is still synthetic: not model weights, not attention/DeltaNet/KV/LM head, not production decode, and not the megakernel. CTest gates: `spock_persistent_decode_skeleton_help`, `spock_persistent_decode_skeleton_smoke`. A Qwen3.5 preset repeat run (tokens=128, layers=24, hidden=1024, workgroups=82, repeats=3) passed with zero failures/trace mismatches and stable repeats 2-3 around 5.94 us/barrier.
- Diary 0076 extends the persistent decode skeleton with real repacked fp16 model-weight support. The app now accepts `--repack-dir DIR` and `--weight-role ROLE` together, loading a WeightArtifact from the repacked manifest, validating dtype fp16 and rank-2 shape, and using `workgroups` rows and `hidden` columns from the real weight matrix. Hidden is inferred from weight cols when not explicitly supplied. Two fixes were needed for checksum agreement with real weights: CPU reference now mirrors the shader's 64-lane-strided fp32 partial-sum + tree-reduction order, and host fp16 decoding now preserves subnormals. A CTest gate `spock_persistent_decode_skeleton_real_weight_smoke` exercises `layer.0.mlp_gate` from `artifacts/spock-text-repack-qwen35-0p8b`. Verified at both hidden=128 and inferred hidden=1024 with exact checksum match. This is the first real model-weight use inside the persistent skeleton; still synthetic input, only prefix rows/cols, no layer semantics, not inference, not the megakernel.
- Diary 0077 extends the persistent decode skeleton with multi-role real-weight probe mode. The `--weight-role ROLE` option is now repeatable: when two or more roles are specified, the probe loads all roles from the WeightArtifact, validates each independently (dtype fp16, rank 2, workgroups <= rows, hidden <= cols), and dispatches the persistent skeleton separately per role in a loop, reusing the same Vulkan pipeline, descriptor set, and buffers (overwriting weight data per role). Per-role results include checksum, expected_checksum, trace_mismatches, failures, and status. Single-role and synthetic modes produce identical output to diary 0076. A CTest gate `spock_persistent_decode_skeleton_multi_role_smoke` exercises `layer.0.mlp_gate` and `layer.0.mlp_up` from `artifacts/spock-text-repack-qwen35-0p8b`. This is real multi-weight skeleton validation, not inference, not layer semantics, and not the megakernel.
- Diary 0078 extends the persistent decode skeleton with `--row-count N`, enabling row-strided weight coverage where a bounded set of `workgroups` resident workgroups cover `row_count` matrix rows via `row = group; row < row_count; row += workgroups`. Default `row_count == workgroups` preserves all prior checksums and behavior. A real-weight row-strided direct run (layer.0.mlp_gate, workgroups=4, row-count=16) produces checksum 3002794576 with exact expected-checksum agreement and zero trace mismatches. A CTest gate `spock_persistent_decode_skeleton_row_count_real_weight_smoke` exercises this path. This is row-strided projection coverage, not inference, not layer semantics, and not the megakernel.
- Diary 0079 raises row-strided coverage to model-width hidden=1024 with decode-relevant workgroups=82 and row-count=128. The single-role timestamped `layer.0.mlp_gate` run produced checksum 2755310530 with exact agreement and zero trace mismatches; the multi-role `layer.0.mlp_gate`/`layer.0.mlp_up` run also passed, with `mlp_up` checksum 3034398318. A CTest gate `spock_persistent_decode_skeleton_row_count_model_width_smoke` protects this stronger real-weight row-strided path. Still not inference: no real activations, MLP activation, down projection, residual path, or token generation.
- Diary 0080 introduces `vk_persistent_mlp_probe` with shader `persistent_mlp_probe.comp`, the first multi-stage persistent micro-probe. The single-dispatch probe chains gate projection, up projection, SiLU(gate)*up activation, and down projection, using two software global barriers (generation=2) to enforce inter-stage dependencies. CLI defaults: hidden=128, intermediate=16, output-rows=8, workgroups=8. Optional `--repack-dir` loads layer.0.mlp_gate, layer.0.mlp_up, layer.0.mlp_down from the repacked model manifest. Synthetic direct run: status ok, checksum 371183224, expected_checksum 371183224, failures 0, generation 2. Real direct run: status ok, checksum 1616650692, expected_checksum 1616650692, failures 0, generation 2. Focused CTest 9/9 passed. This is the first persistent probe with multi-stage barrier-synchronized dependencies; still not inference, not full MLP coverage, not real activations, not residual/layer semantics, not the megakernel.
- Diary 0081 scales `vk_persistent_mlp_probe` to full real layer.0 MLP dimensions: hidden=1024, intermediate=3584, output_rows=1024, workgroups=82. The full real-weight run produced checksum 2160240877 with exact expected-checksum agreement, failures 0, arrived 0, generation 2. Prefix scale-up checks at intermediate/output 128/64, 256/128, 512/256, and 1024/512 also passed before the full run. A CTest gate `spock_persistent_mlp_probe_full_real_weight_smoke` protects this path. This is full real-weight MLP compute inside a single persistent dispatch; still not inference, not a complete layer, not real hidden activations, and not the megakernel.
- Diary 0082 adds optional residual update mode to `vk_persistent_mlp_probe` via `--residual`. The shader now validates `input + down(SiLU(gate(input))*up(input))` for covered rows, with `output_rows <= hidden` enforced. Default non-residual checksum remains 371183224. Synthetic residual run: checksum 374853240 with exact agreement. Full real-weight residual run: checksum 3327711045 with exact agreement, failures 0, arrived 0, generation 2. A CTest gate `spock_persistent_mlp_probe_full_real_weight_residual_smoke` protects this path. Still not inference: input is synthetic, and there is no RMSNorm, attention/DeltaNet, or real hidden-state handoff.
- Diary 0083 extends `vk_persistent_mlp_probe` with `--input-token ID`, loading a real token embedding row from `global.token_embedding` in the repacked artifact as the input vector. When set, the probe reads exactly one row of length `hidden` from the fp16 embedding tensor, validating dtype fp16, rank 2, shape[1] >= hidden, and ID < shape[0]. Requires `--repack-dir`. All existing default checksums are unchanged. A CTest gate `spock_persistent_mlp_probe_full_real_weight_embedding_input_smoke` exercises full model-width real weights plus real embedding row 0 with residual. This is real layer.0 MLP weights plus real model embedding input, but still not inference: no RMSNorm, no token mixer, no real post-attention residual stream, no attention/DeltaNet, no LM head, not megakernel.
- Diary 0084 adds `--input-fp16-file PATH` to `vk_persistent_mlp_probe`, enabling the input vector to come from an external raw little-endian fp16 file. The file must contain at least `hidden * 2` bytes; exactly the first `hidden` values are loaded. The option is mutually exclusive with `--input-token` (reject both with JSON error and exit 2). It works with both synthetic and real weights (no `--repack-dir` dependency). When the loaded fp16 values match the synthetic 1..8 pattern, the checksum is 371183224, identical to the default. JSON output includes `input_fp16_file: PATH` only when provided. A CTest gate `spock_persistent_mlp_probe_fp16_input_smoke` exercises the path with a checked-in fixture `tests/data/mlp_input_pattern_128.fp16` (128 raw fp16 values matching the synthetic pattern). This creates a clean handoff point for real hidden-state captures from the existing Vulkan decode diagnostics, without taking on RMSNorm fusion yet. Still not inference, not RMSNorm, not attention/DeltaNet, not LM head, not megakernel.
- Diary 0085 adds `tools/extract_component_fp16.py`, a Python utility that extracts a named fp16 array field from a `--dump-step-components` JSON layer entry and writes it as a raw little-endian uint16 binary file compatible with `vk_persistent_mlp_probe --input-fp16-file`. The tool validates all values are integers in [0, 65535] and provides clear errors for missing file, invalid JSON, missing `layers`, layer out of range, missing field, non-list field, and out-of-range values. Ten unit tests cover two success cases (exact byte verification) and eight failure cases. A CTest gate `spock_extract_component_fp16` protects the tool. This is a format bridge from decode diagnostics to the MLP probe input mode, not inference, not a runtime component, and not the megakernel.
- Diary 0086 introduces per-row fp16 output equality as the authoritative pass/fail gate for `vk_persistent_mlp_probe`, validated against a real captured hidden-state vector (layer 0, step 1, `mixer_residual`, 1024 fp16 values from `tests/data/layer0_step1_mixer_residual_1024.fp16`). The captured test runs with real weights and residual update; all 1024 output rows match exactly in fp16 (`output_mismatches == 0`). The fp32 aggregate checksum diverges (67820897 vs 67824746) due to a 1-fp16-ULP SiLU activation difference at intermediate row 3180 (GLSL `exp` vs `std::exp` rounding), absorbed by down-projection fp16 output rounding. A CTest gate `spock_persistent_mlp_probe_captured_fp16_handoff` protects this path. This is the strongest persistent MLP validation to date -- real weights, real decode-pipeline input, real residual, exact fp16 output -- but still not inference, not RMSNorm, not attention/DeltaNet, not LM head, and not the megakernel.
- Diary 0087 makes `vk_persistent_mlp_probe` layer-selectable via `--layer N` (default 0), validating N >= 0 and rejecting non-integer input with JSON error/exit 2. Weight roles now use `layer.N.mlp_gate/up/down` instead of hardcoded `layer.0`. A new captured fixture `tests/data/layer1_step1_mixer_residual_1024.fp16` (1024 fp16 values, layer 1 step 1 mixer_residual from the same decode capture as diary 0086) exercises the probe with `--layer 1`, full real dimensions, and residual: output_mismatches == 0, checksum 2888553996 (exact match, no SiLU rounding divergence). CTest gates: `spock_persistent_mlp_probe_layer1_captured_fp16_handoff`, `spock_persistent_mlp_probe_layer_invalid_negative`, `spock_persistent_mlp_probe_layer_invalid_string`, and `spock_persistent_mlp_probe_layer_invalid_partial` (invalid-input gates use WILL_FAIL TRUE). This confirms the fp16 output equality gate holds for two distinct layers; still not inference, not all layers, not RMSNorm, not attention/DeltaNet, not LM head, and not the megakernel.
- Diary 0088 records a run-only all-layer captured-handoff sweep using step-1 `mixer_residual` vectors extracted to `/tmp`. Layers 0-19 and 21-23 pass exact fp16 output equality; layers 0, 2, 12, 18, and 23 have checksum-only diagnostic differences with output_mismatches == 0. Layer 20 is the only hard failure: output_mismatches == 2, first_mismatch_row 657, with rows 657 (`0x92AC` GPU vs `0x92AD` CPU) and 954 (`0x1F68` GPU vs `0x1F69` CPU) differing by 1 fp16 ULP. Temporary instrumentation found exact up scratch agreement and a single activation mismatch at intermediate row 1874 (`0x1D39` GPU vs `0x1D38` CPU), which propagates through down projection and crosses final fp16 rounding boundaries for those two rows. No tolerance was adopted; exact fp16 output equality remains the gate until an explicit precision policy is chosen.
- Diary 0089 introduces `--output-fp16-ulp-tolerance N` (default 0, exact) for `vk_persistent_mlp_probe`. The probe now reports `output_exact_mismatches` (any nonzero ULP diff), `output_within_tolerance` (0 < ULP <= N), `output_mismatches` (ULP > N, gate-breaking), `max_fp16_ulp_diff`, and `output_fp16_ulp_tolerance`. Parsing rejects negative, non-numeric, and partial strings with JSON error/exit 2. A three-tier model applies: exact equality (default), opt-in bounded ULP tolerance (for CPU-vs-GPU captured-handoff probes), and opposite-sign always-fail. A layer 20 fixture `tests/data/layer20_step1_mixer_residual_1024.fp16` enables paired CTest gates: `spock_persistent_mlp_probe_layer20_captured_fp16_handoff_exact_fails` (WILL_FAIL, default tolerance rejects layer 20) and `spock_persistent_mlp_probe_layer20_captured_fp16_handoff_ulp1` (tolerance 1 passes). Three invalid-tolerance WILL_FAIL gates cover negative, string, and partial parsing. GPU-vs-GPU comparisons remain exact. Checksums remain diagnostic, not gates. Still not inference, not RMSNorm, not attention/DeltaNet, not LM head, not megakernel.

- Still pending before Milestone 11 is complete: repeated long soaks under
  system load, repeated barrier-overhead measurement, residency/occupancy
  characterization, and a watchdog-aware decision on whether the next step is
  bounded persistent chunks or full persistent decode.

### Deliverables

- `tools/vk_barrier_probe`
- documented residency limits
- viability report

### Parity Gate

Megakernel parity hinge: if a software global barrier is not reliable on this driver/GPU stack, we do **not** claim full Luce-style megakernel parity and we pivot to the strongest single-submit path instead of forcing a fragile design.

### Go / No-Go

- Go if `10k+` barrier iterations are stable under load
- No-Go if forward progress is not robust enough for production decode

---

## Milestone 12: Full Persistent Decode Megakernel

### Goal

Fuse the full 24-layer decode pass into one persistent Vulkan dispatch.

### Work

- Integrate:
  - layer loop
  - software global barrier between layers
  - in-kernel KV updates
  - in-kernel DeltaNet state updates
  - residual and scratch reuse
  - in-kernel LM head reduction and token selection
- Tune work distribution:
  - one workgroup per CU or resident slot
  - subgroup-to-head mapping
  - LDS allocation
  - occupancy clamps

### Deliverables

- `decode_persistent_mega()` path
- full decode parity tests

### Parity Gate

`P2/P3` decode parity: this milestone must beat the generic baseline convincingly or it has not earned the "megakernel" label.

### Go / No-Go

- Go if decode reaches `P2` or better with stable correctness
- No-Go if persistent dispatch adds complexity without beating single-submit decode

---

## Milestone 13: Prefill Path

### Goal

Bring prefill closer to Lucebox's architecture after decode parity exists.

### Work

- Start with hybrid prefill:
  - batched projections
  - specialized recurrence kernel
  - handoff directly into decode state
- Reuse offline fused weight layouts
- Optimize prompt lengths around `pp520`
- Only consider persistent prefill after decode path is stable

### Deliverables

- `prefill_hybrid_vk()`
- decode handoff correctness tests

### Parity Gate

Prefill parity: prefill must hand off to decode without token mismatch and must improve over the local generic baseline in the Lucebox benchmark style.

### Go / No-Go

- Go if prefill handoff is exact and throughput reaches at least `P1`
- No-Go if prefill optimization threatens decode correctness

---

## Milestone 14: RX 6750 XT Tuning Sweep

### Goal

Tune for this exact GPU rather than hoping portability will be enough.

### Work

- Sweep:
  - workgroup count
  - subgroup size control if available
  - LDS tile sizes
  - register-heavy vs LDS-heavy kernels
  - LM head split strategy
  - fp16/fp32 accumulation cut points
- Record:
  - `pp520`
  - `tg128`
  - GPU clocks if available
  - thermals / power if measurable

### Deliverables

- tuning report
- chosen default launch parameters

### Parity Gate

Hardware-specific parity: Lucebox hard-tunes for a 3090; we must hard-tune for NAVI22 to make the comparison fair.

### Go / No-Go

- Go if tuning materially changes throughput
- No-Go if tuning surface is too noisy to support stable defaults

---

## Milestone 15: Productization And Reproducibility

### Goal

Make the result reproducible and benchmarkable, not just impressive once.

### Work

- CLI for:
  - correctness run
  - `pp520`
  - `tg128`
  - verbose capability dump
- deterministic benchmark mode
- artifact versioning
- shader cache versioning
- CI on non-performance tests

### Deliverables

- `spock-bench`
- `spock-check`
- release checklist

### Parity Gate

Reproducibility parity: Lucebox ships a concrete benchmark contract; we need the same level of reproducibility to claim parity credibly.

### Go / No-Go

- Go if a clean checkout reproduces correctness and benchmark output
- No-Go if the result depends on ad hoc local state

---

## Critical Design Decisions

## 1. Precision strategy

Lucebox uses BF16 heavily. Our target GPU does not expose native BF16 in the local Vulkan stack.

### Decision

- Production path: `fp16 weights + fp16 activations + fp32 critical accumulations/state`
- Reference path: `fp32` or higher-precision CPU path

### Parity consequence

We must prove **behavioral parity**, not literal BF16 instruction parity.

---

## 2. Persistent cross-workgroup synchronization

This is the biggest unknown.

### Decision

Treat software global barrier viability as a dedicated milestone, not an assumption.

### Parity consequence

- If viable: full megakernel parity remains on the table
- If not viable: the strongest honest parity claim is a **single-submit fused Vulkan engine**, not a strict single-dispatch megakernel

---

## 3. Runtime weight loading

Lucebox uses PyTorch and runtime pointer packing. That is not the right production shape for Vulkan.

### Decision

Use an offline converter and a fixed packed artifact.

### Parity consequence

Artifact specialization is part of architectural parity. This is acceptable because Lucebox is also model-specific.

---

## 4. Decode-first delivery

### Decision

Decode before prefill.

### Parity consequence

This matches the highest-value part of Lucebox's claim: eliminating per-token overhead on local inference.

---

## Risks And Kill Criteria

## Risk 1: Software global barrier is unstable on RADV

### Impact

Blocks full persistent megakernel delivery.

### Response

Pivot to single-submit multi-dispatch as the production path.

### Kill criterion

If barrier probe cannot run stably for long stress tests, we stop trying to force a full persistent design.

## Risk 2: LM head dominates decode cost on this small model

### Impact

Layer fusion wins may be hidden by vocab reduction cost.

### Response

Optimize LM head separately and benchmark it as its own stage.

## Risk 3: fp16 parity is brittle in DeltaNet recurrence

### Impact

Token mismatch despite correct structure.

### Response

Keep recurrence state and selected reductions in fp32 even if the rest stays fp16.

## Risk 4: Register pressure collapses occupancy

### Impact

Fused kernels become slower than the layer-by-layer baseline.

### Response

Use milestone-level fusion first, then persistent full-pass only after occupancy data justifies it.

---

## Recommended Execution Order

1. Freeze parity contract and benchmark harness
2. Build offline converter
3. Build CPU reference
4. Build correct Vulkan layer-by-layer decode
5. Freeze local generic baseline
6. Optimize layouts
7. Fuse DeltaNet block
8. Fuse attention block
9. Move to single-submit decode
10. Run software-barrier viability spike
11. Only then attempt full persistent decode megakernel
12. Add prefill
13. Tune for RX 6750 XT
14. Productize

---

## Final Recommendation

The roadmap to parity should be judged by this question at every milestone:

**Are we getting closer to Lucebox-style parity on this machine, or are we just writing more code?**

If a step does not improve one of:

- correctness parity,
- operational parity,
- architectural parity,
- or relative performance parity,

it should be cut or postponed.

That discipline is the difference between a compelling Vulkan megakernel project and a vague "CUDA port" that never closes the gap.

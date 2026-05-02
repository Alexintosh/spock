# 0023: Tiled Single-Dispatch Chunk-Prefill Probe — Removing the Per-Head Submit Blocker

## Goal

Prove a single-dispatch multi-head DeltaNet chunk-prefill shader that replaces
the per-head submit workaround (384 submit-wait cycles per chunk) with a
tiled `vkCmdDispatch(num_heads, ceil(v_dim / TILE_V), 1)`, on the same
synthetic case previously verified for the per-submit probe.

This is an experimental shader/probe outside runtime behavior. It removes the
major proof blocker for a correct single-dispatch multi-head chunk prefill.
Runtime integration is not part of this entry — the probe proves correctness
in isolation, and runtime wiring is the next step.

## Inference Concepts

### Why per-head submits were still the verified runtime path

The original `deltanet_chunk_prefill.comp` shader had only been proven correct
on the realistic padded case when each head was dispatched and submitted
separately. Earlier single-command-buffer/head-barrier variants failed the
same case, while `runtime-l2-padded-submit` passed. That made per-head submit
the only verified-safe runtime workaround, even though it was slow.

The output layout itself is head-disjoint:
`out_data[(h * total_seq + t) * v_dim + d]` gives every head its own slice.
So the blocker was not a simple global output address collision. The stronger
working hypothesis is that the original all-head shader shape placed too much
private scratch/register pressure on the driver/compiler path when many heads
were live together. Serial per-head submits avoided that simultaneous pressure
and preserved correctness, at the cost of large driver overhead.

### Tiled approach

The new shader (`deltanet_chunk_prefill_tiled.comp`) takes a different
approach: each workgroup handles one head **and one v-dimension tile**
(TILE_V = 16). Dispatch is `(num_heads, ceil(v_dim / TILE_V), 1)` — every
head and every v-tile runs concurrently in a single `vkCmdDispatch`.

Key design choices:

- **Per-workgroup state register.** Each workgroup owns a `state_reg[k_dim][TILE_V]` slice of the full `k_dim * v_dim` state matrix. No workgroup touches another workgroup's v-tile range. This eliminates cross-workgroup synchronization entirely.
- **Disjoint output slices.** Each workgroup writes to `out_data[h][t][v_tile_start..v_tile_end]` — a unique contiguous range per (head, tile). No two workgroups write to the same float.
- **TILE_V = 16.** Matches the v_dim = 128 case with 8 tiles. The state slice
  alone is `k_dim * TILE_V = 128 * 16 = 2048` floats (8 KiB) per workgroup.
  The shader still has large private scratch for Q/K and attention matrices,
  so this is a correctness proof, not a register-pressure or occupancy claim.
  The approach generalizes to larger v_dim (more tiles), but TILE_V must stay
  small enough for the target driver/compiler.
- **Correctness-over-speed priority.** The shader recomputes per-tile work (e.g., each workgroup recomputes the full k_cumdecay matrix for its v-tile). This is intentional — the probe prioritizes demonstrable correctness over performance. Performance optimization is deferred until runtime integration.

### Why this removes a major blocker

The per-head submit workaround dominated wall time on the gated GPU path. The
gated CTest took 115.14 sec for a single token at seq_len=9 vs the baseline
5.11 sec, because 24 layers × 16 heads = 384 submit-wait cycles per chunk
each pay full driver overhead. A single dispatch with all heads concurrent
removes that cost. Whether the tiled shader is fast enough at runtime depends
on workgroup occupancy and memory access patterns — those are measured after
integration.

## Implementation Work Completed

### New files (not wired into runtime)

Two new files outside any runtime path:

1. **`shaders/deltanet_chunk_prefill_tiled.comp`** — Tiled single-dispatch
   DeltaNet chunk-prefill shader. Matches `run_deltanet_chunk_rule` semantics
   per tile. GLSL `#version 450`, compute `local_size_x = 1, local_size_y =
   1, local_size_z = 1` (one workgroup per head+tile). Uses the same binding
   layout as `deltanet_chunk_prefill.comp` so existing descriptor set layouts
   are reusable.

2. **`apps/spock-deltanet-chunk-prefill-tiled-probe.cpp`** — Standalone
   probe that exercises the tiled shader against the CPU chunk rule on a
   synthetic case matching the `runtime-l2-padded-submit` case from the
   existing per-submit probe. Build target `spock-deltanet-chunk-prefill-tiled-probe`
   registered in `CMakeLists.txt` via the existing `foreach(app ...)` loop.

### Shader design details

The shader implements the full chunk-rule computation per (head, v-tile):

1. Load chunk Q (full k_dim), K (full k_dim), V (tile-sliced), g, beta.
2. Compute decay matrix `exp(gcum_i - gcum_j)` (lower triangular).
3. Compute attention raw `-(beta_i * k_i · k_j) * decay` (strict lower tri).
4. Neumann series `(I - L)^-1` via iterative expansion.
5. `solved_value = attn @ (v * beta)` (tile-sliced).
6. `k_cumdecay = attn @ (k * beta * exp(gcum))` (full k_dim — recomputed per tile).
7. `v_prime = k_cumdecay @ state_reg` (tile-sliced, from per-workgroup register).
8. `v_new = solved_value - v_prime`.
9. `attn_inter = (q * q_scale * exp_g) @ state_reg` (tile-sliced).
10. `core_attn_out = attn_inter + local_attn @ v_new` (written to output at
    `out_data[(head * total_seq + token) * v_dim + v_tile_start..v_tile_end]`).
11. State update: `state *= exp(gcum_last)` then accumulate outer products
    `k_chunk[row] * weight * v_new[row]` (tile-sliced).

Step 6 (`k_cumdecay`) is redundant across tiles (same k_dim for all v-tiles
of a head). This is the correctness-over-speed tradeoff acknowledged above.

### Probe case and verification strategy

Same synthetic dimensions as the `runtime-l2-padded-submit` case:

| Parameter | Value |
|-----------|-------|
| `num_heads` | 16 |
| `seq_len` | 104 |
| `total_seq` | 128 |
| `chunk_size` | 64 |
| `chunk_count` | 2 |
| `k_dim` | 128 |
| `v_dim` | 128 |
| `TILE_V` | 16 |
| `tile_count` | 8 |
| Q/K range | [-1, 1], L2-normalized per head/token |
| V range | [-20, 20] |
| g range | [-9, -1e-6] |
| beta range | [0, 1] |
| init_state | zero |

Verification: GPU output compared against `run_deltanet_chunk_rule()` from
`deltanet_chunk.hpp` (CPU reference). Pass threshold: `max_rel_core <= 1e-4`
and `max_rel_state <= 1e-4` and `nan_count == 0`.

## Verification

All commands run on the target RADV RX 6750 XT (NAVI22) hardware.

### Build

```sh
source ~/.zshrc && cmake -S . -B build && cmake --build build -j
```

Passed cleanly. The new `spock-deltanet-chunk-prefill-tiled-probe` target was
registered via the existing `foreach(app ...)` loop — no new CMake logic.

### Tiled probe (compare-ok)

```sh
timeout 600s ./build/spock-deltanet-chunk-prefill-tiled-probe
```

Output summary:

```
{"shader":"deltanet_chunk_prefill_tiled.comp","case":"runtime-l2-padded-tiled",
 "status":"compare-ok","dispatch_mode":"tiled-single","tile_v":16,"tile_count":8,
 "output_l2":262.745,"nan_count":0,"max_abs":2.6933,
 "max_abs_core":1.19209e-07,"max_abs_state":2.38419e-07,
 "max_rel_core":1.19209e-07,"max_rel_state":1.19208e-07,
 "worst_core_head":4,"worst_state_head":7}
```

All metrics within machine epsilon of exact CPU reference:
- `max_rel_core = 1.19e-7` — core attention relative error rounds to zero.
- `max_rel_state = 1.19e-7` — state relative error rounds to zero.
- `nan_count = 0` — no numerical instability.

The worst-case heads (4 for core, 7 for state) are within the noise — no head
shows systematic bias.

### Existing per-submit probe still passes

```sh
./build/spock-deltanet-chunk-prefill-probe --case runtime-l2-padded-submit
```

Returns `compare-ok` with matching metrics (not re-recorded here — regression
checked only). The existing per-submit shader and probe are unmodified and
verified.

### Combined GPU prefill pipeline probe still passes

```sh
./build/spock-deltanet-prefill-pipeline-probe
```

Returns `compare-ok` with:
- `max_rel_core = 8.9407e-08`
- `max_rel_state = 1.19175e-07`
- `max_abs_core = 8.9407e-08`
- `max_abs_state = 2.38419e-07`
- `nan_count = 0`

The collection → chunk-prefill pipeline (diary 0018) is unmodified and
verified against the same synthetic case.

### CTest regression gates (6/6 passed)

```sh
ctest --test-dir build --output-on-failure -R spock_vk_decode_gpu_collect_chunk_prefill_short
```

```
1/2 Test #14: spock_vk_decode_gpu_collect_chunk_prefill_short ............   Passed   99.99 sec
2/2 Test #15: spock_vk_decode_gpu_collect_chunk_prefill_short_baseline ...   Passed    7.66 sec
```

The gated test runtime decreased from 115.14 sec (diary 0022) to 99.99 sec
— likely ambient system variation rather than a meaningful improvement. The
tiled shader does not affect the runtime path, so the gated test exercises
the same per-head-submit code as before. The baseline is 7.66 sec (vs 5.11
sec in diary 0022), also ambient variation.

```sh
ctest --test-dir build --output-on-failure -R "spock_capabilities|spock_deltanet_chunk_unit|spock_vk_decode_prefill_handoff_mismatch|spock_diagnose_handoff_mc023"
```

All 4/4 passed. No regression from the new probe target.

### Focused verification set

Total: 6 CTest cases passed (2 GPU-collect gate + 4 targeted) plus 2
standalone probes (tiled + pipeline) plus the existing per-submit probe. No
runtime source was modified, and the existing `deltanet_chunk_prefill.comp`
shader remains in the build alongside the new tiled shader.

## Known Limitations

1. **Not wired into runtime.** The tiled shader is proven correct on a synthetic
   probe case. It is not called from `vk_session.cpp`, not gated by any env var,
   and not part of any decode path. Runtime behavior is completely unchanged.

2. **Zero-initial-state only.** The probe uses `init_state = 0.0`. The shader
   has binding 6 (`init_state`) for nonzero init_state compatibility, but
   currently does **not consume it** — state always starts from zero. The
   binding compatibility ensures the same descriptor layout works when
   nonzero state support is added, but no work was done to wire it.

3. **Redundant computation.** Each v-tile workgroup recomputes
   `k_cumdecay[chunk_size × k_dim]` independently. For v_dim = 128,
   TILE_V = 16, that is 8× redundancy on the k_cumdecay computation. This
   was a deliberate correctness-first choice. A future optimization could
   share this work across tiles of the same head via workgroup-local
   collaboration or a separate precomputation phase.

4. **Single workgroup per (head, tile).** `local_size_x/y/z = 1` means each
   workgroup does all chunk iterations sequentially. For larger chunk_count
   or larger chunk_size, a single invocation may hit GPU timeout thresholds.
   The probe case (2 chunks × 64 tokens = 128 tokens) completes without
   timeout. A production shader may need workgroup-level parallelism within
   a (head, tile).

5. **Correctness threshold is loose for production.** The current pass
   threshold (`max_rel <= 1e-4`) was chosen for the probe. The observed
   errors are at machine epsilon (`~1e-7`), so the threshold is not
   exercised. A production integration should use a tighter threshold.

## Next Work

### Immediate: Runtime integration behind a gate

Wire the tiled shader into `vk_session.cpp` as a new env-gated path
(e.g., `SPOCK_GPU_CHUNK_PREFILL_TILED=1`). This is the natural next step:

- Add a `vkCmdDispatch(num_heads, tile_count, 1)` path alongside the
  existing per-head loop.
- Gate behind a new env var (separate from `SPOCK_GPU_CHUNK_PREFILL=1` to
  avoid breaking the existing gated path).
- Verify on the same CTest gates used in this diary.

### Near-term: Nonzero init_state

The probe proved zero-initial-state correctness. The shader already has
binding 6 wired. The next step is to consume `init_state` instead of
zero-initializing `state_reg`. This requires:

- Loading `init_state[head * k_dim * v_dim + kd * v_dim + gv]` for the
  workgroup's v-tile slice.
- Applying the chunk-loop state update starting from that loaded state
  instead of zero.

### Medium-term: Performance assessment

Once the tiled path is integrated and producing correct output on real
prompts, measure wall time against the per-head submit path and against
the default CPU path. The tiled shader eliminates submit overhead but
introduces redundant computation. Which dominates depends on chunk size,
number of heads, v_dim, and driver overhead characteristics on the target
RADV stack.

### Long-term: Generalization and megakernel roadmap

The tiled approach is a step toward a fused compute megakernel. Key follow-ons:

- Share k_cumdecay across tiles of the same head (workgroup collaboration or
  precomputation phase).
- Fuse the chunk-prefill state output with the Phase C/MLP compute in a
  single dispatch.
- Eventually fold into the persistent dispatch model per the
  IMPLEMENTATION_PLAN.md megakernel roadmap.

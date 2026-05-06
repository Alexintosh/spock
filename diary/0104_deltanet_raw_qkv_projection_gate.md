# 0104: DeltaNet Raw QKV Projection Gate

## Goal

Make the qkv projection gate valid by adding the missing raw runtime checkpoint,
then prove:

```
dn_input_norm_fp16 + layer.0.delta_in_proj_qkv -> dn_qkv_raw_fp16
```

Diary 0103 deliberately rejected a qkv gate against the existing `dn_q_fp16`,
`dn_k_fp16`, and `dn_v_fp16` fields because q and k are dumped after L2
normalization. This entry adds the correct fixture boundary instead of weakening
the test contract.

## Runtime dump extension

`spock-decode --dump-step-components` now emits:

```
dn_qkv_raw_fp16
```

The field is captured immediately after the `delta_in_proj_qkv` matvec and its
buffer barrier, before `conv1d_step` mutates the qkv buffer and before q/k L2
normalization. The implementation stages a copy inside the same command buffer
so later in-place mutations cannot change the diagnostic value.

The raw vector has 6144 fp16 values:

- q raw: 2048 values
- k raw: 2048 values
- v raw: 2048 values

This is the actual output contract of `layer.0.delta_in_proj_qkv`.

## Why this capture point matters

The DeltaNet qkv path is not a simple projection followed immediately by
recurrent compute. The raw qkv projection writes a packed 6144-value buffer,
then the runtime applies a depthwise conv1d step over that buffer, then q and k
are L2-normalized per head before the recurrent update consumes them. The
existing dump fields `dn_q_fp16` and `dn_k_fp16` are therefore not projection
outputs. They are later semantic values.

That distinction is important for correctness. If the project compares
`delta_in_proj_qkv` directly against post-L2 q/k values, the result is a broad
mismatch that says nothing useful about whether the projection is wrong, the
conv step is wrong, or the L2 normalization is wrong. A megakernel-oriented
project cannot afford tests that collapse multiple contracts into one unclear
failure.

The new `dn_qkv_raw_fp16` field gives the qkv side the same quality of boundary
that the MLP side now has: a captured value immediately after the operation
being tested, before later in-place mutation. It also makes the next gates
well-defined. From here, a conv/L2 probe can consume `dn_qkv_raw_fp16` and
target the existing `dn_q_fp16`, `dn_k_fp16`, and `dn_v_fp16` captures. That
will isolate qkv buffer mutation and per-head normalization before recurrent
state is introduced.

This capture adds diagnostic surface area, not production runtime behavior. It
is active only under `--dump-step-components`, and it exists to keep the final
persistent layer probe debuggable.

## Fixture

Added:

```
tests/data/layer0_step1_dn_qkv_raw_6144.fp16
```

The fixture was extracted from a fresh deterministic dump generated after the
runtime extension:

```
build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --tokens /tmp/spock_step1_tokens.txt \
  --max-new-tokens 2 \
  --dump-step-components 1 \
  > /tmp/spock_components1_qkvraw_stdout.txt \
  2> /tmp/spock_components1_qkvraw_stderr.txt
```

The decode result remained unchanged: `generated_tokens` was `[410, 149852]`.

## CTest gate

Added:

```
spock_matvec_probe_layer0_delta_qkv_proj_exact
```

It runs:

```
vk_matvec_probe \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --weight-role layer.0.delta_in_proj_qkv \
  --in-dim 1024 \
  --out-dim 6144 \
  --input-fp16-file tests/data/layer0_step1_dn_input_norm_1024.fp16 \
  --expected-output-fp16-file tests/data/layer0_step1_dn_qkv_raw_6144.fp16
```

## Result

The direct command passes exact fp16 equality:

```json
{
  "in_dim": 1024,
  "out_dim": 6144,
  "weight_role": "layer.0.delta_in_proj_qkv",
  "status": "ok",
  "output_exact_mismatches": 0,
  "output_mismatches": 0,
  "max_fp16_ulp_diff": 0,
  "output_fp16_ulp_tolerance": 0
}
```

All 6144 raw projection outputs match bit-for-bit.

## Interpretation

The stateless DeltaNet input projection side is now better separated:

```
dn_input_norm -> dn_qkv_raw
dn_input_norm -> dn_z
dn_core + dn_z + delta_norm -> dn_gated
dn_gated -> dn_out
dn_out -> mixer_residual
```

This still does not prove the recurrent core, because `dn_core` depends on
q/k/v after convolution, q/k L2 normalization, g/beta, and recurrent state.
But it removes one major unknown: the merged qkv matvec itself is exact against
the runtime capture when tested at the correct boundary.

## Verification

- `cmake --build build -j`
- fresh `spock-decode --dump-step-components 1` after adding
  `dn_qkv_raw_fp16`
- direct qkv `vk_matvec_probe` command passes exact
- focused CTest coverage includes qkv projection, z projection, norm-gate,
  output projection, residual-add, component extraction, and diary checks
- `python3 tests/run_diary_check.py`
- `git diff --check`

## Remaining scope

- Not conv1d qkv mutation parity.
- Not q/k L2-normalization parity.
- Not g/beta computation parity.
- Not recurrent core parity.
- Not layer-shaped persistent decode.
- Not inference or megakernel completion.

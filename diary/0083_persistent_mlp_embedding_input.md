# 0083: Persistent MLP Micro-Probe -- Real Embedding Input

## Goal

Extend `vk_persistent_mlp_probe` so the input vector can optionally come from a real token embedding row in the repacked artifact instead of the synthetic 1..8 pattern. This is the next persistent-path correctness step: real layer.0 MLP weights + real model embedding input + optional residual.

## Background

The persistent MLP probe (diaries 0080-0082) validated gate/up/SiLU-activation/down projection and residual update with synthetic input and real MLP weights. But the synthetic 1..8 cycling pattern does not exercise real activation magnitudes or the actual embedding distribution of the model. Loading a real embedding row moves the probe closer to the input distribution the megakernel will actually see, while remaining a standalone correctness probe.

The repacked artifact contains `global.token_embedding` — the model's embedding table — as a rank-2 fp16 tensor of shape `[248320, 1024]`. Row `i` is the embedding vector for token ID `i`. For Qwen 3.5 0.8B with hidden=1024, each row is exactly `hidden` fp16 values.

## Implementation

The app now accepts:

```
--input-token ID
```

When `--input-token ID` is provided:

1. It requires `--repack-dir`. If absent, the app returns a JSON error.
2. It loads the `global.token_embedding` tensor from the WeightArtifact.
3. Validates: dtype fp16, rank 2, shape[1] >= hidden, ID < shape[0].
4. Extracts exactly one row of length `hidden` from row `ID` into `input_data`.
5. Sets `use_embedding_input = true` for JSON output.

When `--input-token` is not provided, the existing synthetic 1..8 pattern is used, preserving all prior checksums.

The JSON output includes `"input_token": ID` only when the option is provided.

## Verification

### Non-regression: synthetic default

```
build/vk_persistent_mlp_probe --hidden 128 --intermediate 16 --output-rows 8 --workgroups 8
```

Result: status ok, checksum 371183224, expected_checksum 371183224, failures 0, arrived 0, generation 2.

Confirms the new option did not perturb the default probe.

### Non-regression: synthetic residual

```
build/vk_persistent_mlp_probe --hidden 128 --intermediate 16 --output-rows 8 --workgroups 8 --residual
```

Result: status ok, residual true, checksum 374853240, expected_checksum 374853240, failures 0, arrived 0, generation 2.

### Non-regression: full real-weight residual (diary 0082)

```
build/vk_persistent_mlp_probe \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --hidden 1024 --intermediate 3584 --output-rows 1024 --workgroups 82 --residual
```

Result: status ok, residual true, checksum 3327711045, expected_checksum 3327711045, failures 0, arrived 0, generation 2.

### Full real-weight embedding input with residual

```
build/vk_persistent_mlp_probe \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --hidden 1024 --intermediate 3584 --output-rows 1024 --workgroups 82 --residual --input-token 0
```

Result: status ok, input_token 0, residual true, checksum 2614009546, expected_checksum 2614009546, failures 0, arrived 0, generation 2.

The checksum differs from the synthetic-input residual run (3327711045) because the input vector is now a real embedding row rather than the 1..8 pattern. The CPU reference mirrors the GPU computation exactly.

### Validation error: --input-token without --repack-dir

```
build/vk_persistent_mlp_probe --hidden 128 --intermediate 16 --output-rows 8 --workgroups 8 --input-token 0
```

Result: exit 2 with message `--input-token requires --repack-dir`.

### Validation error: out-of-range token ID

```
build/vk_persistent_mlp_probe \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --hidden 1024 --intermediate 3584 --output-rows 1024 --workgroups 82 --input-token 999999
```

Result: exit 2 with message `--input-token 999999 >= vocab_size 248320`.

### CTest

A new gate `spock_persistent_mlp_probe_full_real_weight_embedding_input_smoke` covers the full real-weight embedding input path:

```
vk_persistent_mlp_probe \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --hidden 1024 --intermediate 3584 --output-rows 1024 --workgroups 82 --residual --input-token 0
```

## Interpretation

This entry moves the MLP probe one step closer to layer semantics by replacing synthetic input with a real model embedding vector. The probe now exercises real activation magnitudes drawn from the actual token embedding table, which are qualitatively different from the small synthetic 1..8 values.

The embedding loading is intentionally simple: it reads exactly one row from `global.token_embedding` and copies `hidden` fp16 values into the input buffer. No RMSNorm, no token mixer, no attention, no DeltaNet, no LM head. The point is to validate the data path — that a real embedding row flows correctly through the persistent MLP probe's gate/up/SiLU/down/residual chain — before composing it with more model semantics.

## What This Is

- **Real embedding input for the persistent MLP probe.** The input vector can now come from a real token embedding row instead of synthetic values.
- **Full validation of embedding loading.** dtype, rank, dimension, and token ID bounds are all checked.
- **A layer-semantics stepping stone.** This validates the data path from embedding table through MLP with residual.

## What This Is Not

- **Not inference.** No RMSNorm, no token mixer, no real post-attention residual stream, no attention/DeltaNet, no LM head.
- **Not a full transformer layer.** Only the MLP side of one layer, with an embedding-row input.
- **Not the megakernel.** This remains a standalone probe.
- **Not a performance claim.** No throughput or timing is reported.

## Next Work

1. Add RMSNorm-before-MLP with real norm weights and real embedding input.
2. Compose the MLP residual probe with the recurrent/attention side of a layer-shaped persistent probe.
3. Feed the MLP probe with a real hidden-state vector from a prior layer's output.

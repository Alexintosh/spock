# 0099: Mixer Output Residual-Add Gate

## Goal

Expose the missing token-mixer output checkpoint and verify the first layer
residual equation with Vulkan:

```
input_hidden + mixer_output -> mixer_residual
```

The MLP-side probes already start from `mixer_residual_fp16`, which is the hidden
state after token mixer output has been added back to the layer input. That was
the correct boundary for isolating RMSNorm+MLP, but it left the token-mixer
handoff partially opaque. We had `input_hidden_fp16` and `mixer_residual_fp16`;
we did not have the intermediate `mixer_output_fp16` term needed to verify the
residual add itself.

This entry adds that missing term and a small Vulkan residual-add probe. It is
not a DeltaNet or attention persistent kernel yet. It is the algebraic gate that
must be correct before a larger token-mixer probe is worth debugging.

## Runtime dump extension

`spock-decode --dump-step-components` now emits:

```
mixer_output_fp16
mixer_output_norm
```

The runtime captures `B.act_b` after the token mixer has produced its output
and before the first residual add writes `act_c`. Later MLP down projection
also uses `B.act_b`, so the diagnostic path cannot safely download `B.act_b`
after the layer completes. The implementation therefore stages a copy of
`B.act_b` inside the layer command buffer immediately before `residual_add`.

This mirrors the staging approach used for values that would otherwise be
overwritten, such as `mlp_normed_fp16`.

## Vulkan residual-add probe

Added:

```
vk_residual_add_probe
```

The probe uses the existing `residual_add.comp` shader. Its inputs are raw fp16
vectors:

```
--input-a-fp16-file PATH
--input-b-fp16-file PATH
--expected-output-fp16-file PATH
--length N
```

It dispatches the Vulkan residual-add shader once, downloads the output, and
compares fp16 bit patterns with the same ULP accounting style used by the MLP
probes:

- `output_exact_mismatches`
- `output_within_tolerance`
- `output_mismatches`
- `max_fp16_ulp_diff`
- optional `first_mismatch_row`

Default tolerance is exact equality.

## Fixtures

Added:

```
tests/data/layer0_step1_input_hidden_1024.fp16
tests/data/layer0_step1_mixer_output_1024.fp16
```

The expected output reuses the existing fixture:

```
tests/data/layer0_step1_mixer_residual_1024.fp16
```

The new input fixtures came from a fresh deterministic step-1 component dump:

```
build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --tokens /tmp/spock_step1_tokens.txt \
  --max-new-tokens 2 \
  --dump-step-components 1 \
  > /tmp/spock_components1_mixer_stdout.txt \
  2> /tmp/spock_components1_mixer_stderr.txt
```

The generated token sequence remained `[410, 149852]`. A fresh extraction of
layer-0 `mixer_residual_fp16` from the same dump was byte-compared against the
existing checked-in `layer0_step1_mixer_residual_1024.fp16` fixture and matched
exactly.

## Direct command

```
build/vk_residual_add_probe \
  --length 1024 \
  --input-a-fp16-file tests/data/layer0_step1_input_hidden_1024.fp16 \
  --input-b-fp16-file tests/data/layer0_step1_mixer_output_1024.fp16 \
  --expected-output-fp16-file tests/data/layer0_step1_mixer_residual_1024.fp16
```

Result:

```json
{
  "status": "ok",
  "output_exact_mismatches": 0,
  "output_mismatches": 0,
  "max_fp16_ulp_diff": 0,
  "output_fp16_ulp_tolerance": 0
}
```

## CTest gates

Two tests encode the new gate:

- `spock_residual_add_probe_help`
- `spock_residual_add_probe_layer0_mixer_residual_exact`

The second test is exact because this is a GPU-vs-runtime GPU residual-add
comparison using captured fp16 inputs and output. No CPU math-library tolerance
is justified here.

## Interpretation

The first layer residual handoff is now algebraically closed:

- captured layer input is known
- captured token-mixer output is known
- captured mixer residual is known
- Vulkan residual add reproduces the captured mixer residual exactly

This gives the next token-mixer probe a solid output contract. If a future
DeltaNet or attention persistent mixer emits `mixer_output_fp16`, the project
can immediately test whether the mixer itself is wrong or whether the residual
handoff is wrong. Without this gate, those two failure modes would be mixed
together.

This is deliberately small. It does not implement DeltaNet inside a persistent
megakernel. It creates the checkpoint and gate that make that work debuggable.

## Verification

- `cmake --build build -j`
- `spock-decode --dump-step-components 1` with the deterministic prompt tokens
- fresh layer-0 `mixer_residual_fp16` byte-compare against the existing fixture
- direct JSON parse of `vk_residual_add_probe`
- focused CTest coverage passed after adding the CTest entries, including
  residual-add, persistent MLP, component extraction, and diary checks

## Remaining scope

- Not token-mixer computation parity.
- Not DeltaNet persistent mixer.
- Not attention persistent mixer.
- Not full layer-shaped persistent decode.
- Not inference or megakernel completion.

# 0098: Layer 20 RMSNorm+MLP Population Gate

## Goal

Add a representative mid-network captured RMSNorm+MLP residual gate.

Layer 0 is the cleanest place to debug because it is close to the prompt
embedding and has smaller accumulated residual history. It is not enough by
itself. A future persistent layer path will execute the same fused shape after
many previous residual updates, where the hidden-state magnitude and rounding
context can differ. Layer 20 already had a captured `mixer_residual_fp16`
fixture from the earlier handoff sweep, so it is the cheapest useful
mid-network point to test next.

## Fixture

Added:

```
tests/data/layer20_step1_post_mlp_1024.fp16
```

The fixture came from the same fresh deterministic step-1 component dump used
for diaries 0095 and 0096:

```
build/spock-decode \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --tokens /tmp/spock_step1_tokens.txt \
  --max-new-tokens 2 \
  --dump-step-components 1 \
  > /tmp/spock_components1_gate_stdout.txt \
  2> /tmp/spock_components1_gate_stderr.txt

python3 tools/extract_component_fp16.py \
  --input /tmp/spock_components1_gate_stderr.txt \
  --layer 20 \
  --field post_mlp_fp16 \
  --output /tmp/layer20_step1_post_mlp_1024.fp16
```

The fresh dump's layer-20 `mixer_residual_fp16` extraction was byte-compared
against `tests/data/layer20_step1_mixer_residual_1024.fp16` and matched exactly.
This matters because the probe input and expected output must come from the same
decode step and component dump. Mixing fixtures from different dumps would make
the ULP gate meaningless.

## Direct command

Exact comparison:

```
build/vk_persistent_mlp_probe \
  --layer 20 \
  --repack-dir artifacts/spock-text-repack-qwen35-0p8b \
  --hidden 1024 \
  --intermediate 3584 \
  --output-rows 1024 \
  --workgroups 82 \
  --pre-mlp-rmsnorm \
  --residual \
  --input-fp16-file tests/data/layer20_step1_mixer_residual_1024.fp16 \
  --expected-output-fp16-file tests/data/layer20_step1_post_mlp_1024.fp16 \
  --output-fp16-population-ulp-threshold 16
```

Result:

```json
{
  "status": "fail",
  "output_exact_mismatches": 185,
  "output_mismatches": 185,
  "max_fp16_ulp_diff": 209,
  "output_ulp_le_1": 995,
  "output_ulp_le_2": 1010,
  "output_ulp_le_4": 1020,
  "output_ulp_le_8": 1022,
  "output_ulp_le_16": 1023,
  "output_ulp_gt_64": 1,
  "output_rows_above_population_ulp_threshold": 1
}
```

With explicit bounded tolerance and population gate:

```
--output-fp16-ulp-tolerance 209 \
--output-fp16-population-ulp-threshold 16 \
--output-fp16-max-rows-above-population-threshold 1
```

Result:

```json
{
  "status": "ok",
  "output_exact_mismatches": 185,
  "output_within_tolerance": 185,
  "output_mismatches": 0,
  "max_fp16_ulp_diff": 209,
  "output_rows_above_population_ulp_threshold": 1,
  "output_population_ok": true
}
```

## CTest gates

Three tests encode the layer-20 captured RMSNorm+MLP residual boundary:

- `spock_persistent_mlp_probe_layer20_captured_rmsnorm_mlp_exact_fails` is
  marked `WILL_FAIL`.
- `spock_persistent_mlp_probe_layer20_captured_rmsnorm_mlp_ulp209_population_ulp16`
  must pass.
- `spock_persistent_mlp_probe_layer20_captured_rmsnorm_mlp_population_ulp16_max0_fails`
  is marked `WILL_FAIL` to prove the population gate catches the single row
  above 16 ULP.

## Interpretation

Layer 20 is not worse in the broad population sense. It has fewer exact
mismatches than layer 0 and only one row above 16 ULP, but the tail row reaches
209 ULP. That is a useful warning for composition: the project should track
both max and population because either one can move independently.

This result does not prove all layers are safe. It does prove that the captured
RMSNorm+MLP residual slice remains tightly concentrated at a mid-network layer,
with a sparse tail that needs to be carried as an explicit contract. The next
layer-shaped persistent probe should preserve the local checkpoint discipline
instead of relying only on final generated-token parity.

## Verification

- Extracted layer-20 `post_mlp_fp16` from the deterministic step-1 component
  dump.
- Byte-compared fresh layer-20 `mixer_residual_fp16` against the existing
  checked-in layer-20 input fixture.
- Parsed direct exact JSON output.
- Parsed direct ULP-209/population JSON output.

The focused CTest suite still needs to be rerun after adding the CTest entries.

## Remaining scope

- Not all-layer RMSNorm+MLP sweep.
- Not exact layer-20 post-residual parity.
- Not token mixer integration.
- Not layer-shaped persistent decode.
- Not inference or megakernel completion.

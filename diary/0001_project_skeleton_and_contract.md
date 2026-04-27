# 0001: Project Skeleton, Parity Contract, and First Build

## Goal

The first phase turned an empty repository containing only `IMPLEMENTATION_PLAN.md` into a buildable project with a reproducible contract. This matters because LLM inference work can easily become a pile of kernels and benchmarks that are impossible to compare. Before writing performance-sensitive code, Spock needs to know exactly what model it targets, what outputs count as correct, what commands people run, and what hardware facts the runtime observes.

For this project, the target is `Qwen/Qwen3.5-0.8B` on an `AMD Radeon RX 6750 XT (RADV NAVI22)` using Vulkan compute. The long-term goal is a specialized batch-1 inference engine, but the first usable result is a scaffold that can build, run, test, and measure consistently.

## Why A Contract Comes Before Kernels

An LLM generates text one token at a time. A token is an integer ID from a vocabulary, not necessarily a whole word. For example, a word can be one token, multiple tokens, or share tokens with punctuation depending on the tokenizer.

At each decode step, the model receives the existing token sequence and produces scores, called logits, for the next token. Greedy decoding chooses the highest-scoring token. If every setting is deterministic, a correct implementation should produce the same token IDs as a trusted reference for the same prompt and model weights.

That is the core of `P0` correctness parity: exact token agreement on fixed prompts. If token parity is not established first, later Vulkan debugging becomes ambiguous. A wrong token might be caused by a kernel bug, a tokenizer mismatch, a different model revision, floating-point differences, an incorrect cache update, or a benchmark accidentally using a different prompt.

This is why the project now has:

- A fixed prompt corpus in `tests/data/prompts.jsonl`.
- A parity contract in `docs/parity_contract.md`.
- A baseline scoreboard shape in `bench/baseline_rx6750xt.json`.
- CLIs that provide stable entry points for checks, benchmarks, and capability dumps.

## Project Layout

The skeleton follows the plan's intended structure:

- `apps/` contains command-line programs.
- `src/model/` stores model constants and layer schedule.
- `src/reference/` contains the CPU reference interface.
- `src/runtime/` contains timing, Vulkan capability, and memory-planning scaffolding.
- `src/kernels/` defines shader/kernel binding contracts.
- `shaders/` contains Vulkan compute shader sources.
- `tools/` contains model artifact conversion and validation scripts.
- `tests/` contains dependency-free smoke checks and prompt fixtures.
- `docs/` contains contracts and design notes.
- `diary/` contains explanatory implementation notes.

This layout deliberately separates model facts, runtime facts, artifact facts, and benchmark facts. In inference projects these concerns often get tangled. Keeping them separate makes it easier to replace placeholder pieces with real implementations while preserving the public contract.

## Model Constants

`src/model/qwen35_config.hpp` encodes the text model shape used by the plan:

- `24` layers.
- Hidden size `1024`.
- MLP intermediate size `3584`.
- Full attention every fourth layer.
- `18` DeltaNet-style linear-attention layers.
- `6` full-attention layers.

The schedule is `DeltaNet, DeltaNet, DeltaNet, FullAttention`, repeated six times.

This matters because inference runtime code needs to know which operation to execute for each layer. A transformer layer with full attention has different state and memory traffic than a DeltaNet/linear-attention layer. The exact layer order is part of model semantics, not an optimization detail.

## Attention And DeltaNet In Plain Terms

Full attention lets a token compare itself with previous tokens. In decode, the model stores keys and values from earlier tokens in a KV cache. For each new token, attention uses the current query and the cached keys/values to decide which earlier information matters.

The downside is that attention cost grows with context length. If the prompt is long, a decode step must look over many cached positions.

DeltaNet-style linear attention uses recurrent state instead of scanning all previous tokens in the same way. For batch-1 local inference this can be attractive because the state update can be more constant-time per token. The target Qwen 3.5 model is hybrid: most layers are linear-attention layers, but every fourth layer is full attention.

The runtime must therefore support both:

- Recurrent DeltaNet state.
- KV cache state for full-attention layers.

## CMake And CLIs

The project now builds with CMake:

```sh
cmake -S . -B build
cmake --build build -j
```

The main CLIs are:

- `spock-check`: validates model-contract facts and the prompt corpus.
- `spock-bench`: emits placeholder benchmark JSON or CSV for `pp520`, `tg128`, and correctness modes.
- `vk-capabilities`: queries the local Vulkan device.
- `vk_barrier_probe`: placeholder CLI for the future persistent workgroup/global-barrier experiment.

The benchmark implementation is intentionally labeled `placeholder-cpu-timer`. It proves the CLI and output contract, not model performance. This distinction is important. False performance numbers are worse than no performance numbers because they can guide optimization work in the wrong direction.

## Vulkan Capability Query

`vk-capabilities` creates a Vulkan instance, selects the first physical device, and prints key compute properties:

- Device name.
- Vulkan API version.
- Subgroup size.
- Shared memory limit.
- Maximum compute workgroup invocations.

On the local machine it reports:

- `AMD Radeon RX 6750 XT (RADV NAVI22)`.
- Subgroup size `64`.
- Shared memory `65536` bytes.
- Maximum workgroup invocations `1024`.

A subgroup is a set of GPU lanes that execute together and can cooperate through subgroup operations. On AMD hardware this often corresponds to wavefront behavior. Subgroup size matters because kernels frequently use subgroup reductions for vector sums, max reductions, and argmax operations.

The original plan mentioned a locally observed subgroup width of `32`, but the actual Vulkan query reports `64` here. The baseline notes were updated to make the runtime query the source of truth. This is an example of why capability detection should be executable, not only written in a plan.

## Shaders

Two shader sources exist:

- `shaders/trivial_compute.comp`
- `shaders/persistent_barrier_probe.comp`

The trivial shader writes deterministic values to an output buffer. It is a bring-up shader: before running complex inference kernels, the runtime must prove it can compile shaders, create pipelines, bind buffers, dispatch work, and read results.

The barrier-probe shader establishes the descriptor and push-constant shape for a future persistent-kernel experiment. A true Luce-style megakernel needs workgroups to coordinate across layers. Vulkan does not provide a normal global barrier inside one dispatch, so the project must explicitly test whether a software global barrier is reliable on this GPU and driver.

CMake compiles shaders to SPIR-V when `glslangValidator` is available.

## Artifact Tools

The first converter mode was dry-run/offline:

```sh
python3 tools/convert_qwen35_0p8b.py --offline --output /tmp/spock-artifact --force
python3 tools/validate_artifact.py /tmp/spock-artifact --json
```

An artifact is the model data in a format the runtime can load without Python objects or framework-specific pointer structures. Real inference engines need explicit offsets, file names, dtypes, shapes, and alignment rules. This is especially important for Vulkan because shaders operate on buffers, not Python tensors.

The validator checks that a manifest is structurally sound. At this phase, the dry-run artifact did not contain real weights. It existed to freeze the schema and make the rest of the project testable.

## Tests

CTest now runs dependency-free checks:

```sh
ctest --test-dir build --output-on-failure
```

The checks cover:

- `spock-check` against the prompt corpus.
- `spock-bench` benchmark modes.
- benchmark output file writing.
- Vulkan capability CLI smoke.
- artifact dry-run conversion and validation.
- JSON fixture validation.

Python `pytest` tests also exist, but the local Python environment does not include `pytest`, so CTest is the reliable baseline.

## What This Phase Does Not Do

This phase does not run real inference. The CPU reference decode still uses a deterministic placeholder transition. The Vulkan code does not yet load real weights into GPU buffers or execute layer kernels.

That limitation is intentional. The value of this phase is that future work has a stable harness:

- Add real CPU reference.
- Add real artifact packing.
- Add layer-by-layer Vulkan kernels.
- Compare against the same prompts and same model revision.
- Only then claim correctness or performance.

## Verification

The phase was verified with:

```sh
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/vk-capabilities
./build/spock-check --prompts tests/data/prompts.jsonl
```

The CTest suite passed, and the Vulkan runtime saw the expected RX 6750 XT device.

## Next Phase

The next phase was to try the actual Hugging Face model artifact instead of only a dry-run manifest. That tests whether the plan's architecture assumptions match the real downloaded `Qwen/Qwen3.5-0.8B` files.

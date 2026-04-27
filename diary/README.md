# Spock Engineering Diary

This diary records the implementation phases of Spock as the project moves from a plan to a working Vulkan-native inference engine.

The intended reader is an experienced programmer who may know little about LLMs, token generation, model artifacts, GPU inference, or Vulkan compute. Entries therefore explain both what changed and why the change matters for inference correctness, reproducibility, and performance.

## Entries

- [0001: Project Skeleton, Parity Contract, and First Build](0001_project_skeleton_and_contract.md)
- [0002: Actual Qwen 3.5 0.8B Artifact Ingestion](0002_actual_model_ingestion.md)

## Conventions

Each entry should cover:

- The phase goal.
- The implementation work completed.
- The inference concepts needed to understand the work.
- Verification performed.
- Known limitations and next work.

The diary is not a changelog. It should explain the engineering reasoning behind the code, including mistakes, constraints, and tradeoffs.

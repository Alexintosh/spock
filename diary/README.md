# Spock Engineering Diary

This diary records the implementation phases of Spock as the project moves from a plan to a working Vulkan-native inference engine.

The intended reader is an experienced programmer who may know little about LLMs, token generation, model artifacts, GPU inference, or Vulkan compute. Entries therefore explain both what changed and why the change matters for inference correctness, reproducibility, and performance.

## Entries

- [0001: Project Skeleton, Parity Contract, and First Build](0001_project_skeleton_and_contract.md)
- [0002: Actual Qwen 3.5 0.8B Artifact Ingestion](0002_actual_model_ingestion.md)
- [0003: Text-Only Artifact Plan](0003_text_only_artifact_plan.md)
- [0004: Text Weight Repack](0004_text_weight_repack.md)
- [0005: Reference Decode P0 Freeze](0005_reference_decode_p0_freeze.md)
- [0006: Weight Pipeline Verification](0006_weight_pipeline_verification.md)
- [0007: Vulkan Decode Pipeline](0007_vulkan_decode_pipeline.md)
- [0008: MLP Decode Path — First Correct GPU Forward Pass](0008_mlp_decode_path.md)
- [0009: Full Attention Layers — KV Cache, QK-Norm, mRoPE, GQA Decode](0009_attention_decode_path.md)
- [0010: DeltaNet Recurrent Decode — All 24 Layers Active](0010_deltanet_decode_path.md)
- [0011: Attention V-Accumulation Bug — First Correct Multi-Layer Output](0011_attention_v_accumulation_bug.md)
- [0012: Qwen3.5 RMSNorm Parity Fix](0012_qwen35_rmsnorm_parity_fix.md)
- [0013: Native DeltaNet Chunk Rule Primitive](0013_native_deltanet_chunk_rule.md)
- [0014: RADV Device Selection and Prefill-Sensitive Hardware Gate](0014_radv_device_selection_and_prefill_gate.md)
- [0015: Decode Drift Diagnostics and Handoff Checkpoint](0015_decode_drift_diagnostics_and_handoff_checkpoint.md)
- [0016: Precision Drift Isolation and Refined Next Work](0016_precision_drift_isolation_and_refined_next_work.md)
- [0017: GPU DeltaNet Chunk-Prefill Shader — First Experimental Probe](0017_gpu_deltanet_chunk_prefill_probe.md)
- [0018: Combined GPU Prefill Pipeline Probe](0018_current_verified_state_checkpoint.md)
- [0019: Runtime Diagnostic GPU Prefill Collection Comparison](0019_gpu_collect_prefill_diagnostic.md)
- [0020: GPU Chunk-Prefill From GPU Collection — Avoiding CPU Intermediate Packing](0020_gpu_chunk_prefill_from_gpu_collect.md)

## Conventions

Each entry should cover:

- The phase goal.
- The implementation work completed.
- The inference concepts needed to understand the work.
- Verification performed.
- Known limitations and next work.

The diary is not a changelog. It should explain the engineering reasoning behind the code, including mistakes, constraints, and tradeoffs.
